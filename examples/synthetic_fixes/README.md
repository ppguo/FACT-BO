# Synthetic validation of implementation corrections

This checkpoint-free example provides a small before/after validation of the
model-integration corrections included in this release. It deliberately uses a
controlled one-dimensional analytic objective rather than fitting TabPFN, so
the implementation behavior can be checked without surrogate accuracy or GPU
non-determinism becoming confounding factors.

```bash
python examples/synthetic_fixes/validate_fixes.py
```

The script writes a JSON summary and a two-panel plot under `results/`. The two
panels exercise candidate handling and predictive uncertainty scaling. This is
an implementation-level validation only; it does not imply that every real
objective will obtain the same improvement.
