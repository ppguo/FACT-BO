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
iterations, 5000 candidates, rank 5, scale 1.0, and the solid `K=16` sigma-ball.
For bit-level paper reproduction, supply the archived initial designs with
`--initial-dir`; the generated Sobol designs are deterministic replacements,
not a claim of identity with the private archived files.
