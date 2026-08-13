import torch
import math
from torch import Tensor
from dataclasses import dataclass
from torch.quasirandom import SobolEngine



@dataclass
class Unified_TS_State:
    """
    Unified state class that handles both constrained (SCBO) and unconstrained (TuRBO) cases.
    When constraints are not present (C is None), it behaves like TuRBO.
    When constraints are present, it behaves like SCBO.
    Source: https://botorch.org/docs/tutorials/turbo_1/
    """
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3, botorch uses 10
    best_value: float = -float("inf")
    best_constraint_values: Tensor = None  # Will be initialized if constraints exist
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )





def update_ts_state(state: Unified_TS_State, Y_next: Tensor, C_next: Tensor = None):
    """
    Unified update method that handles both constrained and unconstrained cases.
    Args:
        state: UnifiedTurboState instance
        Y_next: Tensor of objective values
        C_next: Optional tensor of constraint values. If None, unconstrained case is assumed.
    Source: https://botorch.org/docs/tutorials/turbo_1/
    """
    if C_next is None:
        # Unconstrained case (TuRBO)
        if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = max(Y_next).item()
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        # Constrained case (SCBO)
        # Initialize best_constraint_values if not already done
        if state.best_constraint_values is None:
            state.best_constraint_values = torch.ones_like(C_next[0]) * float("inf")
        
        # Pick the best point from the batch
        best_ind = get_sorted_indices(Y=Y_next, C=C_next)
        best_ind = best_ind[0]
        y_next, c_next = Y_next[best_ind], C_next[best_ind]

        if (c_next <= 0).all():
            # At least one new candidate is feasible
            improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
            if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
                state.success_counter += 1
                state.failure_counter = 0
                state.best_value = y_next.item()
                state.best_constraint_values = c_next
            else:
                state.success_counter = 0
                state.failure_counter += 1
        else:
            # No new candidate is feasible
            total_violation_next = c_next.clamp(min=0).sum(dim=-1)
            total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
            if total_violation_next < total_violation_center:
                state.success_counter += 1
                state.failure_counter = 0
                state.best_value = y_next.item()
                state.best_constraint_values = c_next
            else:
                state.success_counter = 0
                state.failure_counter += 1

    # Update trust region length
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:
        state.restart_triggered = True
        state.length = min(8.0 * state.length, state.length_max)

    return state




def generate_batch_PFN(
                        state,
                        WEIGHTS, # REQUIRE WEIGHT DEFINITION
                        X,  # Evaluated points on the domain [0, 1]^d
                        Y,  # Function values
                        n_candidates=None,  # Number of candidates for Thompson sampling
                        C=None,  # Constraint values (optional)
                        tkwargs={"device": torch.device('cpu'), "dtype": torch.float32}
                    ):
    """
    Generalized generate_batch function that handles both constrained and unconstrained cases.
    Only generates trust region bounds and candidate points.
    
    Args:
        state: TurboState or ScboState object
        WEIGHTS: Weights derived from PFN
        X: Evaluated points on the domain [0, 1]^d
        Y: Function values
        n_candidates: Number of candidates for Thompson sampling
        C: Constraint values (optional, if None then unconstrained problem)
    
    Returns:
        tr_lb: Trust region lower bounds
        tr_ub: Trust region upper bounds
        X_cand: Candidate points generated
    """
    X = X.to(**tkwargs)
    Y = Y.to(**tkwargs)
    WEIGHTS = WEIGHTS.to(**tkwargs)
    if C is not None:
        C = C.to(**tkwargs)
    
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    with torch.no_grad():
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))
    
        # Select trust region center
    
        # Get the best as center
        score_idx = get_sorted_indices(Y, C) # The largest feasible value should be at index 0
        x_center = X[score_idx[0], :].clone()
        

        # Ensure state.length is a tensor on the correct device
        length = torch.tensor(state.length, **tkwargs)
        
        # Calculate trust region bounds using TuRBO's scaling 
        # No weights scaling for PFN
        # weights = WEIGHTS / weights.mean()
        # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        weights = WEIGHTS.clone()
        tr_lb = torch.clamp(x_center - weights * length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * length / 2.0, 0.0, 1.0)
    
        # Generate candidates
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(**tkwargs)
        pert = tr_lb + (tr_ub - tr_lb) * pert
    
        # Create perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=tkwargs['device'])] = 1
    
        # Create candidate points
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

    return tr_lb, tr_ub, X_cand





def get_sorted_indices(Y, C):
    """Return sorted indices based on three scenarios.
    
    Args:
        Y: Tensor of objective function values
        C: Tensor of constraint values (optional)
    
    Returns:
        sorted_indices: Tensor of indices sorted from best to worst according to:
            - If no constraints: sorted by Y values (descending)
            - If constraints but no feasible points: sorted by total constraint violation (ascending)
            - If constraints with feasible points: sorted by Y values with infeasible points set to -inf (descending)
    """
    # print(f"C: {C == None}")
    if C is None:
        # Case 1: No constraints - sort by objective value (descending)
        # print(f'Case1 Y: {Y}')
        
        return torch.argsort(Y.squeeze(-1), descending=True)
    
    # Check feasibility for all points
    is_feas = (C <= 0).all(dim=1)
    # print(f'is_feas: {is_feas}')
    
    if is_feas.any():
        # Case 3: Has feasible points - set infeasible points to -inf and sort
        score = Y.clone()
        score[~is_feas] = float('-inf')

        # print(f'Case3 score: {torch.argsort(score.squeeze(-1), descending=True)}')
        return torch.argsort(score.squeeze(-1), descending=True)
    else:
        # Case 2: No feasible points - sort by total constraint violation (ascending)
        violation = C.clamp(min=0).sum(dim=-1)

        
        # print(f'Case2 violation')
        return torch.argsort(violation)
    
