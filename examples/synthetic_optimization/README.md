# End-to-end synthetic optimization

This example runs the released optimizer and the real local TabPFN-v2 checkpoint
on a six-dimensional embedded Branin objective. Unlike `synthetic_fixes`, this
is a full Bayesian-optimization smoke test rather than an isolated integration
check.

The objective is negative Branin, so it is maximized. Its known global maximum
is `-0.3978873577`; only the first two of six coordinates are active. The
default run uses a rank-2 gradient-informed subspace, 12 deterministic Sobol
initial points, and 20 BO evaluations.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/synthetic_optimization/run_branin.py \
  --output-dir results/branin_seed0 \
  --seed 0
```

The generated `summary.json` reports the initial and final best values, the gap
to the analytic optimum, a same-budget Sobol reference, and basic optimizer
invariants. The comparison with one Sobol sequence is a smoke-test diagnostic,
not a statistical benchmark.

For a simpler directional check, run the smooth embedded quadratic, whose
unique active optimum is `(0.75, 0.25)` with maximum value `0`:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/synthetic_optimization/run_quadratic.py \
  --output-dir results/quadratic_seed0 \
  --seed 0
```
