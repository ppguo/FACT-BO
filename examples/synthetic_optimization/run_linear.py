#!/usr/bin/env python3
"""Minimal directional test for the full TabPFN/GIT-BO loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from algorithms import GITBO


class EmbeddedLinear:
    """Maximize the sum of the first two unit-cube coordinates."""

    def __init__(self, dim: int = 6):
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
        return None, unit_points[:, :2].sum(dim=1, keepdim=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=6)
    parser.add_argument("--n-init", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--pending", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--no-subspace", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_dir = args.output_dir / "initial"
    initial_dir.mkdir(exist_ok=True)
    objective = EmbeddedLinear(args.dim)
    engine = torch.quasirandom.SobolEngine(args.dim, scramble=True, seed=args.seed)
    initial_x = engine.draw(args.n_init).to(torch.float32)
    torch.save(initial_x, initial_dir / "_trial_0.pt")
    initial_best = float(objective.evaluate(initial_x)[1].max())

    trained_x, history = GITBO(
        Function=objective,
        SEED=args.seed,
        Trail_N=0,
        N_iterations=args.iterations,
        Acquisition="SamplingUCB",
        INITIAL_DIR=str(initial_dir),
        SAVE_DIR=str(args.output_dir / "gitbo_results"),
        N_PENDING=args.pending,
        N_CANDIDATES=1,
        DEVICE="cpu",
        GPU_DEVICE=args.gpu_device,
        GI_SUBSPACE=not args.no_subspace,
        rank_r=args.rank,
        scale=args.scale,
    )
    total = args.n_init + args.iterations
    evaluated_x = trained_x[:total].detach().cpu()
    evaluated_y = objective.evaluate(evaluated_x)[1][:, 0]
    new_y = evaluated_y[args.n_init :]
    best_index = int(torch.argmax(evaluated_y))
    final_best = float(evaluated_y[best_index])
    history = history.detach().cpu()
    payload = {
        "objective": "x0 + x1",
        "known_global_maximum": 2.0,
        "seed": args.seed,
        "subspace_enabled": not args.no_subspace,
        "n_initial": args.n_init,
        "bo_iterations": args.iterations,
        "candidate_pool": args.pending,
        "initial_best": initial_best,
        "new_points_best": float(new_y.max()),
        "new_points_mean": float(new_y.mean()),
        "final_best": final_best,
        "improvement_over_initial": final_best - initial_best,
        "gap_to_known_optimum": 2.0 - final_best,
        "best_active_point": evaluated_x[best_index, :2].tolist(),
        "all_points_in_unit_cube": bool(
            ((evaluated_x >= 0) & (evaluated_x <= 1)).all()
        ),
        "best_history_monotone": bool(
            (history[1:] >= history[:-1] - 1e-7).all()
        ),
        "history_matches_recomputed_objective": abs(
            float(history[-1]) - final_best
        )
        <= 1e-6,
        "new_active_points": evaluated_x[args.n_init :, :2].tolist(),
        "best_history": history.tolist(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))
    if not all(
        payload[key]
        for key in (
            "all_points_in_unit_cube",
            "best_history_monotone",
            "history_matches_recomputed_objective",
        )
    ):
        raise RuntimeError("optimizer invariant check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
