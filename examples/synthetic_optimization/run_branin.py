#!/usr/bin/env python3
"""End-to-end GIT-BO smoke test on an embedded Branin objective.

This exercise uses the real local TabPFN-v2 model and the released GIT-BO loop.
It is intentionally separate from ``synthetic_fixes``, which isolates I-001 and
I-004 with an analytic posterior and does not run Bayesian optimization.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from algorithms import GITBO


BRANIN_MINIMUM = 0.39788735772973816
ACQUISITIONS = (
    "SamplingUCB",
    "EI",
    "Quantile_UCB_95",
    "Quantile_UCB_975",
    "Quantile_UCB_995",
)


class EmbeddedBranin:
    """Negative Branin on the first two coordinates of a unit hypercube."""

    def __init__(self, dim: int = 6):
        if dim < 2:
            raise ValueError("EmbeddedBranin requires dim >= 2")
        self.dim = dim

    def evaluate(self, unit_points: torch.Tensor, to_verify: bool = True):
        del to_verify
        if unit_points.ndim != 2 or unit_points.shape[1] != self.dim:
            raise ValueError(
                f"expected [n, {self.dim}] unit points, got {tuple(unit_points.shape)}"
            )
        if not torch.isfinite(unit_points).all():
            raise ValueError("non-finite input")
        if (unit_points < 0).any() or (unit_points > 1).any():
            raise ValueError("input outside [0, 1]^D")

        x1 = 15.0 * unit_points[:, 0] - 5.0
        x2 = 15.0 * unit_points[:, 1]
        a = 1.0
        b = 5.1 / (4.0 * math.pi**2)
        c = 5.0 / math.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * math.pi)
        branin = a * (x2 - b * x1.square() + c * x1 - r).square()
        branin = branin + s * (1.0 - t) * torch.cos(x1) + s
        return None, -branin.unsqueeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=6)
    parser.add_argument("--n-init", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--pending", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--acquisition", choices=ACQUISITIONS, default="SamplingUCB"
    )
    parser.add_argument("--gpu-device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_dir = args.output_dir / "initial"
    initial_dir.mkdir(exist_ok=True)

    objective = EmbeddedBranin(args.dim)
    init_engine = torch.quasirandom.SobolEngine(
        args.dim, scramble=True, seed=args.seed
    )
    initial_x = init_engine.draw(args.n_init).to(torch.float32)
    torch.save(initial_x, initial_dir / "_trial_0.pt")
    _, initial_y = objective.evaluate(initial_x)
    initial_best = float(initial_y.max())

    trained_x, best_history = GITBO(
        Function=objective,
        SEED=args.seed,
        Trail_N=0,
        N_iterations=args.iterations,
        Acquisition=args.acquisition,
        INITIAL_DIR=str(initial_dir),
        SAVE_DIR=str(args.output_dir / "gitbo_results"),
        N_PENDING=args.pending,
        N_CANDIDATES=1,
        DEVICE="cpu",
        GPU_DEVICE=args.gpu_device,
        GI_SUBSPACE=True,
        rank_r=args.rank,
        scale=args.scale,
    )

    evaluated_x = trained_x[: args.n_init + args.iterations].detach().cpu()
    _, evaluated_y = objective.evaluate(evaluated_x)
    final_best = float(evaluated_y.max())
    history = best_history.detach().cpu()
    in_bounds = bool(((evaluated_x >= 0) & (evaluated_x <= 1)).all())
    monotone = bool((history[1:] >= history[:-1] - 1e-7).all())
    history_matches = abs(float(history[-1]) - final_best) <= 1e-5

    baseline_engine = torch.quasirandom.SobolEngine(
        args.dim, scramble=True, seed=args.seed + 100_000
    )
    baseline_x = baseline_engine.draw(args.n_init + args.iterations)
    _, baseline_y = objective.evaluate(baseline_x)
    baseline_best = float(baseline_y.max())

    best_index = int(torch.argmax(evaluated_y))
    best_x = evaluated_x[best_index].tolist()
    payload = {
        "objective": "negative embedded Branin",
        "known_global_maximum": -BRANIN_MINIMUM,
        "seed": args.seed,
        "dimension": args.dim,
        "active_dimensions": [0, 1],
        "n_initial": args.n_init,
        "bo_iterations": args.iterations,
        "total_evaluations": args.n_init + args.iterations,
        "candidate_pool": args.pending,
        "subspace_rank": args.rank,
        "sampling_scale": args.scale,
        "acquisition": args.acquisition,
        "initial_best": initial_best,
        "final_best": final_best,
        "improvement_over_initial": final_best - initial_best,
        "same_budget_sobol_best": baseline_best,
        "improvement_over_sobol": final_best - baseline_best,
        "gap_to_known_optimum": (-BRANIN_MINIMUM) - final_best,
        "best_unit_point": best_x,
        "best_branin_coordinates": [15.0 * best_x[0] - 5.0, 15.0 * best_x[1]],
        "all_points_in_unit_cube": in_bounds,
        "best_history_monotone": monotone,
        "history_matches_recomputed_objective": history_matches,
        "best_history": history.tolist(),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))

    if not (in_bounds and monotone and history_matches):
        raise RuntimeError("optimizer invariant check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
