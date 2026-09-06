# 4x2 SRAM read-delay example

This example exposes the paper configuration without embedding any server,
account, conda, PDK or simulator paths in source code.

Prerequisites:

1. A compatible audited OpenYield checkout containing
   `size_optimization/sample_yield_4x2.py`.
2. Xyce available to the OpenYield Python environment.
3. A redistributable or private metadata JSON containing the exact 144-element
   `nominal` and `sigma` arrays plus `design_point`, `std_ratio`, `corner`,
   `target_row`, `target_col`, and `vary: "all"`.
4. The GIT-BO TabPFN-v2 regression checkpoint described in the repository
   README.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/sram_read_delay/run.py \
  --openyield-root /path/to/OpenYield \
  --meta /path/to/meta.json \
  --worker-python /path/to/openyield/bin/python \
  --workdir /path/to/private/run_seed0 \
  --seed 0
```

Defaults reproduce the paper-level settings: 30 initial evaluations, 270 BO
iterations, 5000 candidates, rank 5, scale 1.0, the solid `K=16` sigma-ball, and corrected ST-EFMI at `--threshold-ps 290.6`.
Use `--acquisition SamplingUCB` to select the earlier acquisition. Invalid
simulator outputs stop the default SRAM adapter; they are not physical failures.
For bit-level paper reproduction, supply the archived initial designs with
`--initial-dir`; the generated Sobol designs are deterministic replacements,
not a claim of identity with the private archived files.

## Compare a committed revision with a corrected reference

`verify_revision.py` runs the public `GITBO` entrypoint at the reference seed,
reuses only its 30 initial observations, and performs 270 fresh adaptive
simulations plus at most two endpoint rechecks. It checks every candidate score
against independent density quadrature and compares all selected coordinates and
responses with the reference. A mismatch is reported without replacing points.
Run from a clean committed checkout with the matching external checkpoint:
The verifier records both GPU models and requires the reference PyTorch version.
A different GPU is allowed, but can change floating-point gradients and the
subsequent active-subspace trajectory; report numerical agreement separately
from exact coordinate reproduction.

```bash
python examples/sram_read_delay/verify_revision.py \
  --reference-run /path/to/corrected/main_sram_read_delay_seed0 \
  --initial-dir /path/to/archived/initial \
  --openyield-root /path/to/OpenYield \
  --meta /path/to/meta.json \
  --worker-python /path/to/openyield/bin/python \
  --output /path/to/new/private/verification
```

The reference directory must contain `RUN.done`, `provenance.json`,
`result.json`, and `trace.jsonl`. The output records the Git commit, source and
input hashes, simulator-call ledger, score checks, complete trajectory and
`result.json`; inspect `matches_reference` and `full_budget` separately from
`RUN.done`. `--iterations 2` is a short integration smoke, not a full rerun.
An interrupted run with a closed simulator-call ledger can be supplied using
`--resume-prefix /path/to/interrupted/output`. Source/input hashes must match and
each adaptive point is regenerated before its observation is reused. The first
coordinate mismatch permanently ends reuse; subsequent points are simulated.
Recovery uses the interrupted run's GPU model. Reused observations and new calls
are counted separately in the final result.
