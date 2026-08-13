import torch
import numpy as np
import random
import time
import os
import uuid
from botorch.utils.sampling import draw_sobol_samples
from datetime import datetime
from torch.quasirandom import SobolEngine
import sys
from .tabpfn_wrapper import VanillaDirectTabPFNRegressor
from ._utils import *






def GITBO(
    Function,
    SEED: int,
    Trail_N: int = 0,
    N_iterations: int = 100,
    Acquisition: str = 'Quantile_UCB_95',
    INITIAL_DIR: str = None,
    SAVE_DIR: str = None,
    N_PENDING: int = 5000,
    N_CANDIDATES: int = 1,
    DEVICE: str = 'cpu',
    GPU_DEVICE: str = 'cuda:0',
    GI_SUBSPACE = False,
    rank_r = 10,
    scale = 1.0,
):
    """
    Bayesian Optimization using TabPFN v2 as surrogate model.
    
    Args:
        Function: Objective function to optimize
        SEED: Random seed for reproducibility
        N_iterations: Number of BO iterations
        Acquisition: Type of acquisition function
        N_PENDING: Number of candidate points to evaluate acquisition on
        N_CANDIDATES: Number of points to evaluate per iteration
        PREPROCESS: Whether to use preprocessing pipeline
        
    Returns:
        tuple: (evaluated_points, optimization_history)
    """
    print(f'Compute Setting: {DEVICE}')
    print(f'GI_SUBSPACE: {GI_SUBSPACE}, Acquisition: {Acquisition}')
    tkwargs = {"device": torch.device(DEVICE), "dtype": torch.float32}
    
    if GI_SUBSPACE:
        rank_r = rank_r
        scale = scale
        INIT_SCALE = scale
    else:
        rank_r = None
        scale = None
        INIT_SCALE = None
    # with torch.no_grad():
    # Set random seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    DIM = Function.dim
    
    # Initialize training data
    trained_X = (torch.load(f'{INITIAL_DIR}/_trial_{Trail_N}.pt') 
                if INITIAL_DIR else torch.rand(50, DIM, **tkwargs))
    N_INIT = trained_X.shape[0]
    
    # Initial evaluation
    GX, trained_Y = Function.evaluate(trained_X)
    
    # Move tensors to the specified device
    trained_X = trained_X.to(**tkwargs)
    trained_Y = trained_Y.to(**tkwargs)
    if GX is not None:
        GX = GX.to(**tkwargs)

    # Initialize PREV_Xpen and PREV_Eval
    PREV_Xpen = trained_X
    PREV_Eval = trained_Y

    # Pre-allocate memory for all iterations - create tensors directly on the correct device
    INIT_LOC = trained_X.shape[0]
    trained_X = torch.cat([trained_X, torch.zeros(N_iterations * N_CANDIDATES, DIM, **tkwargs)])
    trained_Y = torch.cat([trained_Y, torch.zeros(N_iterations * N_CANDIDATES, 1, **tkwargs)])
    if GX is not None:
        GX = torch.cat([GX, torch.zeros(N_iterations * N_CANDIDATES, GX.shape[1], **tkwargs)])
    ITER_IND_LOC = INIT_LOC
    
    # Initialize tracking arrays
    TIME_ARR = torch.zeros(N_iterations, **tkwargs)
    MAX_ARR = torch.zeros(N_iterations, **tkwargs)
    TOTAL_TIME = 0
    
    # Setup for trust region methods
    sobol_engine = SobolEngine(Function.dim, scramble=True)
    if 'TR' in Acquisition:
        TR_LB_List = torch.zeros(N_iterations, DIM, **tkwargs)
        TR_UB_List = torch.zeros(N_iterations, DIM, **tkwargs)
        batch_size = N_CANDIDATES
        state = Unified_TS_State(Function.dim, batch_size=batch_size)
        weights = torch.ones(1, Function.dim, **tkwargs)
    else:
        TR_LB_List = TR_UB_List = state = weights = batch_size = None

    grad_est = None

    # Main optimization loop
    for iter_ in range(N_iterations):
        
        # Start the timer
        start_time = time.monotonic()
        
        # Get the Xpen and compute the acquisition values
        tr_lb, tr_ub, X_pen = get_Xpen(
            Function, Acquisition, sobol_engine, N_PENDING, N_CANDIDATES, DIM, state, weights,
            trained_X[:ITER_IND_LOC,:], trained_Y[:ITER_IND_LOC,:], GX[:ITER_IND_LOC] if GX is not None else None, 
            rank_r, scale, 
            trained_X[:ITER_IND_LOC,:],
            trained_Y[:ITER_IND_LOC,:],
            GI_SUBSPACE, grad_est, tkwargs,
        )
        ACQ, CONS, grad_est = compute_acquisition_values(
            Acquisition, DIM, sobol_engine, N_PENDING, N_CANDIDATES, None, trained_X[:ITER_IND_LOC,:],
            trained_Y[:ITER_IND_LOC,:], GX[:ITER_IND_LOC] if GX is not None else None, 
            X_pen.to(**tkwargs), Function, GPU_DEVICE,
            tr_lb, tr_ub, state, weights, batch_size, GPU_DEVICE, tkwargs,
        )

        if ACQ is None and CONS is None:
            print('PFN output NANs')
            return trained_X, MAX_ARR


        X_pen = X_pen.permute(1,0,2)
        ACQ = ACQ.permute(1,0)


        best_candidate_indices = torch.argmax(ACQ, dim=1)


        num_batches = X_pen.shape[0]
        row_indices = torch.arange(num_batches)

        best_candidate = X_pen[row_indices.to('cpu'), best_candidate_indices.to('cpu'), :].detach()

        # Update timings
        TOTAL_TIME += time.monotonic() - start_time
        TIME_ARR[iter_] = TOTAL_TIME
        

        # Evaluate new points
        ITER_IND_LOC = INIT_LOC + N_CANDIDATES
        trained_X[INIT_LOC:ITER_IND_LOC,:] = best_candidate
        g_, y_ = Function.evaluate(best_candidate)
        trained_Y[INIT_LOC:ITER_IND_LOC,:] = y_
        if GX is not None:
            GX[INIT_LOC:ITER_IND_LOC,:] = g_
        INIT_LOC = ITER_IND_LOC
        
        # Update trust region if needed
        if 'TR' in Acquisition:
            state = update_ts_state(state, trained_Y, GX)
            TR_LB_List[iter_,:] = tr_lb
            TR_UB_List[iter_,:] = tr_ub
            
        # Track best value
        sorted_ind = get_sorted_indices(
            trained_Y[:ITER_IND_LOC,:],
            GX[:ITER_IND_LOC,:] if GX is not None else None
        )
        MAX_ARR[iter_] = trained_Y[:ITER_IND_LOC,:][sorted_ind[0],:].cpu().detach()


        # Print progress
        if SAVE_DIR is None:
            print(f'GITBO Iter {iter_}) Opt: {MAX_ARR[iter_]} Time: {TOTAL_TIME}')
            if GX is not None:
                print(f'Feasible solutions: {(GX <= 0).all(dim=1).any()}')
            

    if SAVE_DIR == None:
        print("Done GITBO")
    else:
        
        save_folder = f"{SAVE_DIR}/GITBO_{Acquisition}/{Function.__class__.__name__}_DIM_{DIM}/"
        
        # Setup output file Directory
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        now = datetime.now() # Get current date and time
        timestamp = now.strftime("%Y%m%d_%H%M%S") # Format it as YYYYMMDD_HHMMSS
        save_file_name = f"{save_folder}/_trial_{Trail_N}_rankR_{rank_r}_iters_{N_iterations}_pend_{N_PENDING}.pt"
        
        torch.save({
            'TIME_ARR': TIME_ARR.cpu().detach(),
            'MAX_ARR': MAX_ARR.cpu().detach(),
            'trained_X': trained_X.cpu().detach(),
        }, save_file_name)

        print(f'Save GITBO file at {save_file_name}')

    return trained_X, MAX_ARR




def _assert_query_aligned(tensor, name: str, n_pending: int, n_candidates: int) -> None:
    """Guard for issue I-001: one acquisition/mean/variance row per pending candidate."""
    actual = tuple(tensor.shape[:2])
    expected = (n_pending, n_candidates)
    if actual != expected:
        raise AssertionError(
            f"{name} is not aligned with candidates: leading shape {actual}, expected {expected}"
        )


def compute_acquisition_values(
    Acquisition: str,
    DIM: int,
    sobol_engine,
    N_PENDING: int,
    N_CANDIDATES: int,
    PFN_MODEL,  # Not used in v2 but kept for interface compatibility
    trained_X: torch.Tensor,
    trained_Y: torch.Tensor,
    GX: torch.Tensor,
    X_pen: torch.Tensor,
    Function,
    DEVICE: str,
    tr_lb=None,
    tr_ub=None,
    state=None,
    weights=None,
    batch_size=None,
    GPU_DEVICE="cuda:0",
    tkwargs=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute acquisition values using TabPFN v2 model, with memory‑efficient gradient estimation.
    Returns: (acquisition_values [N_PENDING×N_CANDIDATES], constraint_values=None, grad_est [N_PENDING×N_CANDIDATES×DIM])
    """
    
    # reset stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    regressor = VanillaDirectTabPFNRegressor(device=GPU_DEVICE)
    single_eval_pos = trained_X.shape[0]
    
    # report
    peak_bytes = torch.cuda.max_memory_allocated()
    # print(f"Peak GPU memory used: {peak_bytes/2**20:.2f} MiB")

    # --- build the full X and Y concatenation ---
    X_train = trained_X.unsqueeze(1).expand(-1, N_CANDIDATES, -1)           # [single_eval_pos, N_CANDIDATES, DIM]
    X_full  = torch.cat([X_train, X_pen], dim=0)                            # [(single_eval_pos+N_PENDING), N_CANDIDATES, DIM]

    Y_pad   = torch.zeros(N_PENDING, 1, **tkwargs)                          # [N_PENDING, 1]
    Y_full  = torch.cat([trained_Y, Y_pad], dim=0).unsqueeze(1)             # [(single_eval_pos+N_PENDING), 1, 1]
    Y_full  = Y_full.expand(-1, N_CANDIDATES, -1)                           # [(single_eval_pos+N_PENDING), N_CANDIDATES, 1]

    # --- split off the candidate portion for gradient tracking ---
    X_train_det = X_full[:single_eval_pos].detach()
    X_cand      = X_full[single_eval_pos:].clone().requires_grad_()         # [N_PENDING, N_CANDIDATES, DIM]
    X_concat    = torch.cat([X_train_det, X_cand], dim=0).to(GPU_DEVICE)

    # --- get devices ---
    amp_dtype = tkwargs["dtype"]
    # ``torch.amp.autocast`` expects a device *type* (for example ``cuda``),
    # whereas the public GITBO API accepts indexed devices such as ``cuda:0``.
    amp_device = torch.device(GPU_DEVICE).type
    amp_enabled = amp_device == "cuda" and amp_dtype in {
        torch.float16,
        torch.bfloat16,
    }
    peak_bytes = torch.cuda.max_memory_allocated()


    if Acquisition in ['EI', 'TR_EI']:
        # --- forward + EI under autocast for mixed precision ---
        with torch.amp.autocast(
            device_type=amp_device, dtype=amp_dtype, enabled=amp_enabled
        ):
            out    = regressor.forward(X_concat, Y_full, single_eval_pos)
            logits = out["standard"]
            # TabPFN-v2 returns query rows only. Keep them aligned one-to-one
            # with X_cand; slicing by single_eval_pos again repeats I-001.
            EI = regressor.predict_ei(logits, trained_Y.max())
        _assert_query_aligned(EI, "EI acquisition", N_PENDING, N_CANDIDATES)

        # --- gradient only wrt X_cand ---
        grad_cand, = torch.autograd.grad(EI.sum(), X_cand, retain_graph=False, create_graph=False)
        grad_est   = -grad_cand.view(N_PENDING, N_CANDIDATES, DIM).detach()

        return EI.to(**tkwargs), None, grad_est.to(**tkwargs)
    
    elif Acquisition in ['Quantile_UCB_95', 'Quantile_UCB_975', 'Quantile_UCB_995']:
        
        if Acquisition == 'Quantile_UCB_95':
            percentile = 0.05
        elif  Acquisition == 'Quantile_UCB_975':
            percentile = 0.025
        elif  Acquisition == 'Quantile_UCB_995':
            percentile = 0.005
        # Reuse the common [time, batch, feature] query construction above.
        # The released branch attempted to concatenate 2-D trained_X with 3-D
        # X_pen, so every Quantile-UCB option failed before model inference.
        with torch.amp.autocast(
            device_type=amp_device, dtype=amp_dtype, enabled=amp_enabled
        ):
            out_dict = regressor.forward(X_concat, Y_full, single_eval_pos)
            logits = out_dict["standard"]
            UCB = regressor.predict_ucb(logits, trained_Y.max(), percentile)
            output_mean = regressor.predict_mean(logits)

        _assert_query_aligned(UCB, "quantile UCB acquisition", N_PENDING, N_CANDIDATES)
        _assert_query_aligned(output_mean, "predictive mean", N_PENDING, N_CANDIDATES)
        grad_cand, = torch.autograd.grad(
            output_mean.sum(), X_cand, retain_graph=False, create_graph=False
        )
        grad_est = -grad_cand.view(N_PENDING, N_CANDIDATES, DIM).detach()

        return UCB.to(**tkwargs), None, grad_est.to(**tkwargs)

        

    elif Acquisition in ['SamplingUCB', 'TR_SamplingUCB']:
        # --- forward + mean/variance under autocast ---
        with torch.amp.autocast(
            device_type=amp_device, dtype=amp_dtype, enabled=amp_enabled
        ):
            out             = regressor.forward(X_concat, Y_full, single_eval_pos)
            logits          = out["standard"]
            output_mean     = regressor.predict_mean(logits)                   # [N_PENDING, N_CANDIDATES]
            output_variance = regressor.predict_variance(logits)               # same shape

        # print('forward pass done')

        # FIX I-001: TabPFN-v2 decodes only the QUERY rows, so mean/variance already have
        # exactly N_PENDING rows. Production sliced them a second time with [single_eval_pos:],
        # dropping the last n_train candidates and shifting acquisition row i onto X_pen[i]
        # when it actually belonged to X_pen[n_train + i].
        mu_cand  = output_mean                                                # [N_PENDING, N_CANDIDATES]
        var_cand = output_variance
        _assert_query_aligned(mu_cand, "predictive mean", N_PENDING, N_CANDIDATES)
        _assert_query_aligned(var_cand, "predictive variance", N_PENDING, N_CANDIDATES)
        std_cand = torch.clamp(var_cand, min=1e-8).sqrt()

        # --- sample if requested ---
        sample_count = 512
        mu_expanded  = mu_cand.unsqueeze(-1).expand(-1, -1, sample_count)
        std_expanded = std_cand.unsqueeze(-1).expand(-1, -1, sample_count)
        sampled_y    = torch.normal(mu_expanded, std_expanded)
        sampled_y = torch.max(sampled_y, dim=-1).values
        _assert_query_aligned(sampled_y, "acquisition", N_PENDING, N_CANDIDATES)

        # --- gradient only wrt X_cand, using the mean as loss ---
        loss       = mu_cand.sum()
        grad_cand, = torch.autograd.grad(loss, X_cand, retain_graph=False, create_graph=False)
        grad_est   = -grad_cand.view(N_PENDING, N_CANDIDATES, DIM).detach()
        if tuple(grad_est.shape) != (N_PENDING, N_CANDIDATES, DIM):
            raise AssertionError(
                f"gradient shape {tuple(grad_est.shape)} != {(N_PENDING, N_CANDIDATES, DIM)}"
            )
        
        # print('grad done')

        return sampled_y.to(**tkwargs), None, grad_est.to(**tkwargs)

    else:
        raise ValueError(f"Unknown acquisition type: {Acquisition}")





def get_Xpen(Function,
             Acquisition, 
             sobol_engine,
             N_PENDING,
             N_CANDIDATES,
             DIM,
             state,
             weights, 
             trained_X,  
             trained_Y,  
             GX = None,  
             rank_r = None,
             scale = None,
             PREV_Xpen = None,
             PREV_Eval = None,
             GI_SUBSPACE = False,
             grad_est = None,
             tkwargs = None,
             
            ):
    
    if 'TR' in Acquisition:
        if rank_r is not None:
            tr_lb, tr_ub, X_pen = generate_batch_PFN(
                                        state,
                                        weights, # REQUIRE WEIGHT DEFINITION
                                        trained_X,  # Evaluated points on the domain [0, 1]^d
                                        trained_Y,  # Function values
                                        n_candidates=None,  # Number of candidates for sampling
                                        C=GX,  # Constraint values (optional)
                                        tkwargs=tkwargs
                                )
            if grad_est is None:
                total_points = N_CANDIDATES * N_PENDING
                X_pen = sobol_engine.draw(total_points)
                X_pen = X_pen.view(N_PENDING, N_CANDIDATES, DIM)
                return tr_lb, tr_ub, X_pen
            else:
                X_pen, _, _ = sample_dominant_subspace(trained_X, trained_Y, DIM, grad_est, sobol_engine, 
                                rank_r=rank_r, n_samples=N_PENDING, N_CANDIDATES=N_CANDIDATES,
                                scale=scale, GI_SUBSPACE=GI_SUBSPACE, tkwargs=tkwargs)
                return tr_lb, tr_ub, X_pen
        else:
            tr_lb, tr_ub, X_pen = generate_batch_PFN(
                                        state,
                                        weights, # REQUIRE WEIGHT DEFINITION
                                        trained_X,  # Evaluated points on the domain [0, 1]^d
                                        trained_Y,  # Function values
                                        n_candidates=None,  # Number of candidates for sampling
                                        C=GX,  # Constraint values (optional)
                                        tkwargs=tkwargs
                                )
            
            X_pen = X_pen.view(N_PENDING, N_CANDIDATES, DIM)
            # print("X_pen.shape", X_pen.shape)
            return tr_lb, tr_ub, X_pen
    
    if grad_est is None:
      #   standard_bounds = torch.zeros(2, DIM, **tkwargs)
      #   standard_bounds[1] = 1

        # change it to batch (April 17)
        total_points = N_CANDIDATES * N_PENDING
        X_pen = sobol_engine.draw(total_points)
        X_pen = X_pen.view(N_PENDING, N_CANDIDATES, DIM)
        return None, None, X_pen
    if rank_r is not None:
        X_pen, _, _ = sample_dominant_subspace(trained_X, trained_Y, DIM, grad_est, sobol_engine, 
                                rank_r=rank_r, n_samples=N_PENDING, N_CANDIDATES=N_CANDIDATES,
                                scale=scale, GI_SUBSPACE=GI_SUBSPACE, tkwargs=tkwargs)
        # print(X_pen.shape)
        return None, None, X_pen
    else:
        # Sample pending points
        total_points = N_CANDIDATES * N_PENDING
        X_pen = sobol_engine.draw(total_points)
        X_pen = X_pen.view(N_PENDING, N_CANDIDATES, DIM)
        return None, None, X_pen
















def sample_dominant_subspace(x, y, DIM, grad_vals, SEED, 
                             rank_r=1, n_samples=100, N_CANDIDATES=1, scale=1.0, 
                             GI_SUBSPACE=False, new_origin = None, tkwargs=None):
    """
    Sample points in the dominant subspace given input data x and Function.
    
    Args:
        x: Input points tensor of shape (n, d) where n is number of points and d is dimension
        Function: Function object with evaluate method that can handle batched inputs
        rank_r: Number of dominant directions to keep (default=1)
        n_samples: Number of samples to generate in subspace (default=100)
        scale: Scale factor for sampling range (default=1.0)
    
    Returns:
        samples: New points sampled in dominant subspace
        U_r: Top r eigenvectors
        eigenvals: Eigenvalues sorted in descending order
    """
    
    # rng = np.random.RandomState(SEED)
    
    # Keep as torch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, **tkwargs)
    
    # Extract dimensions
    n_points, batch_size, feat_dim = grad_vals.shape
    
    # Convert to numpy for eigendecomposition
    grad_vals_np = grad_vals.cpu().detach().numpy()
    x_np = x.cpu().detach().numpy()
    
    
    # Initialize containers for H_est matrices
    X_pen = torch.zeros((n_samples, batch_size, feat_dim))
    
    # Calculate H_est for each batch in a vectorized way
    for b in range(batch_size):
        # Get dominant subspace for this batch
        H_est = (grad_vals_np[:, b, :].T @ grad_vals_np[:, b, :]) / n_points
        
        eigvals, eigvecs = np.linalg.eigh(H_est)
        idx_sorted = np.argsort(eigvals)[::-1]
        eigenvals = eigvals[idx_sorted]
        eigenvecs = eigvecs[:, idx_sorted]
        
        
        if rank_r < 1: 
            # (Mode 1) Percentage coverage of variance where 0<rank_r<1
            # --- Auto-selection logic starts here ---
            total_variance = np.sum(eigenvals)
            cumulative_variance_ratio = np.cumsum(eigenvals) / total_variance

            # 2. Find the number of components to explain 95% of the variance
            # We find the index of the first element that is >= rank_r (e.g. 0.95) and add 1 (for 0-based index)
            n_components_percent = np.argmax(cumulative_variance_ratio >= rank_r) + 1

            print(f"Number of components to capture 95% variance: {n_components_percent}\n")
            rank_r = n_components_percent
            # --- Auto-selection logic ends here ---
        else:
            # (Mode 2) Fixed r (default GIT-BO) where rank_r<=1
            rank_r = rank_r # just showing that it doesn't change :))
            
        
        # Get top r eigenvectors
        U_r = eigenvecs[:, :rank_r]
        
        # Get mean of input points as origin
        if new_origin is None:
            origin = x_np.mean(axis=0)
        else:
            origin = new_origin.mean(axis=0)
            
        alpha = np.random.uniform(-scale, scale, size=(n_samples, rank_r))
        
        # Map back to original space: x = origin + U_r * alpha
        samples = origin + (alpha @ U_r.T)
        
        # Convert samples back to torch tensor
        samples = torch.tensor(samples, **tkwargs)
        
        # Clamp to [0, 1] (design space min max)
        samples = torch.clamp(samples, 0.0, 1.0).to(**tkwargs)
        
        X_pen[:,b,:] = samples.detach().to(**tkwargs)
    
    return X_pen, U_r, eigenvals
