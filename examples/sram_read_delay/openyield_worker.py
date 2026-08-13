#!/usr/bin/env python3
"""Persistent OpenYield/Xyce worker for the 4x2 SRAM read-delay example."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np


def load_sampling_module(openyield_root: Path):
    """Load the compatible OpenYield ``sample_yield_4x2.py`` by explicit path."""

    source = openyield_root / "size_optimization" / "sample_yield_4x2.py"
    if not source.is_file():
        raise FileNotFoundError(f"missing OpenYield sampling driver: {source}")
    spec = importlib.util.spec_from_file_location("fas_wca_sample_yield_4x2", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source.parent))
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openyield-root", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--keep-batches", action="store_true")
    args = parser.parse_args()

    sampling = load_sampling_module(args.openyield_root.resolve())
    metadata = json.loads(args.meta.read_text(encoding="utf-8"))
    if metadata.get("vary", "all") != "all":
        raise SystemExit("the SRAM worker requires a vary=all (144-D) metadata file")

    args.workdir.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(sys.stderr):
        config = sampling.load_sram_config()

    design = metadata["design_point"]
    cell = config.sram_6t_cell
    cell.nmos_width.value = [design["pd_width_m"], design["pg_width_m"]]
    cell.pmos_width.value = design["pu_width_m"]
    cell.length.value = design["length_m"]
    cell.nmos_model.value = [design["pd_model"], design["pg_model"]]
    cell.pmos_model.value = design["pu_model"]

    if "nominal" not in metadata or "sigma" not in metadata:
        raise SystemExit("metadata must contain explicit nominal and sigma arrays")
    nominal = np.asarray(metadata["nominal"], dtype=float)
    sigma = np.asarray(metadata["sigma"], dtype=float)
    if nominal.shape != (144,) or sigma.shape != (144,):
        raise SystemExit("metadata nominal and sigma arrays must each contain 144 values")

    namespace = argparse.Namespace(
        std_ratio=float(metadata["std_ratio"]),
        corner=metadata.get("corner", "TT"),
        target_row=int(metadata["target_row"]),
        target_col=int(metadata["target_col"]),
        w_rc=False,
    )

    print("READY", flush=True)
    previous: Path | None = None
    request_index = 0
    for line in sys.stdin:
        request = line.strip()
        if not request:
            continue
        if request == "EXIT":
            break
        try:
            vector = np.asarray([float(value) for value in request.split(",")], dtype=float)
            if vector.shape != (144,):
                raise ValueError(f"expected 144 values, received {vector.size}")
            absolute = nominal[None, :] + vector[None, :] * sigma[None, :]
            batch_dir = args.workdir / f"batch_{request_index:06d}"
            request_index += 1
            with contextlib.redirect_stdout(sys.stderr):
                result = sampling.simulate_batch(config, absolute, namespace, batch_dir)
            delay = float(result.get(0, {}).get("read_delay_s", math.nan))
            if previous is not None and not args.keep_batches:
                shutil.rmtree(previous, ignore_errors=True)
            previous = batch_dir
        except Exception as error:  # protocol must return one response per request
            print(f"[worker] evaluation failed: {error!r}", file=sys.stderr, flush=True)
            delay = math.nan
        print(repr(delay), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
