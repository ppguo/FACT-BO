#!/usr/bin/env python3
"""Run the paper's FAS-WCA configuration on OpenYield's 4x2 SRAM example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.quasirandom import SobolEngine

# Make the documented ``python examples/.../run.py`` invocation work from a
# source checkout without requiring the project itself to be installed first.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from algorithms import GITBO
from fas_wca import LineProcessEvaluator, SramReadDelayProblem


def prepare_initial_points(directory: Path, seed: int, count: int, dimension: int) -> Path:
    """Create a deterministic Sobol initial design when no archived design is supplied."""

    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"_trial_{seed}.pt"
    if not destination.exists():
        points = SobolEngine(dimension, scramble=True, seed=seed).draw(count)
        torch.save(points, destination)
    return directory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openyield-root", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--worker-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--initial-dir", type=Path)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-count", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=270)
    parser.add_argument("--n-pending", type=int, default=5000)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=16.0)
    parser.add_argument("--fill", choices=("solid", "surface"), default="solid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--acquisition", choices=("EFMI", "ST-EFMI", "SamplingUCB", "EI"), default="EFMI")
    parser.add_argument("--threshold-ps", type=float, default=290.6)
    args = parser.parse_args()

    run_dir = args.workdir.resolve()
    initial_dir = args.initial_dir
    if initial_dir is None:
        initial_dir = prepare_initial_points(run_dir / "initial", args.seed, args.initial_count, 144)
    save_dir = (args.save_dir or (run_dir / "results")).resolve()
    worker = Path(__file__).with_name("openyield_worker.py").resolve()
    command = [
        str(args.worker_python.resolve()),
        str(worker),
        "--openyield-root",
        str(args.openyield_root.resolve()),
        "--meta",
        str(args.meta.resolve()),
        "--workdir",
        str(run_dir / "simulator"),
    ]

    evaluator = LineProcessEvaluator(command, run_dir / "worker")
    problem = SramReadDelayProblem(evaluator, radius=args.radius, fill=args.fill)
    try:
        _points, history = GITBO(
            problem,
            args.seed,
            Trail_N=args.seed,
            N_iterations=args.iterations,
            Acquisition=args.acquisition,
            threshold_y=args.threshold_ps,
            INITIAL_DIR=str(initial_dir.resolve()),
            SAVE_DIR=str(save_dir),
            N_PENDING=args.n_pending,
            N_CANDIDATES=1,
            DEVICE=args.device,
            GPU_DEVICE=args.device,
            GI_SUBSPACE=True,
            rank_r=args.rank,
            scale=args.scale,
        )
    finally:
        evaluator.close()

    worst = float(history.max().item())
    print(f"highest read delay: {worst:.6f} ps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
