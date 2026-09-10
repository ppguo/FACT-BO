# Source provenance

The optimizer is derived from the official ICLR 2026 GIT-BO implementation in
the sibling research snapshot `GITBO-main`. The SRAM paper results were produced
from the frozen server tree `GITBO_fix14_20260731`.

Paper-snapshot SHA-256 fingerprints:

```text
6427d214760f5b9b05b3706362408519af413c9df7413fa82108ea44d569c25a  algorithms/_GITBO.py
c047537654147308c084ee6bf44f0bf9c28d6db77756d1fbcb083dae7f9ac1aa  algorithms/tabpfn_wrapper.py
0edde3e6e0d4713a853b12164da788b842538f2ab6c14a78704d6c6936333002  tabpfn/model/bar_distribution.py
```

The first two files in this repository begin from those exact adapted versions.
The vendored `tabpfn/model/bar_distribution.py` is byte-identical to the paper
snapshot. Machine-specific runners and SRAM adapters were replaced with the
parameterized implementation under `fas_wca/` and `examples/sram_read_delay/`.

## Implementation notes

This release includes a small number of implementation corrections for
compatibility with the packaged dependencies and for numerical consistency.
These changes do not alter the high-level GIT-BO algorithm. Focused regression
checks are provided in `tests/test_semantic_fixes.py`.

## Corrected acquisition integration (2026-09-06)

Integrated the independently audited ST-EFMI half-normal EI formulas from the
2026-09-05 correction. The direct implementation replaces the process-local
patch; positive response scales are preserved. Added an explicit specification
threshold and posterior-mean-gradient ST-EFMI path to the public optimizer,
independent quadrature and decision-chain tests, and strict SRAM error handling.
Legacy EI and UCB policy names remain available. The default SRAM example now
uses EFMI at 290.8 ps. Models, PDKs and private experiment data remain external.
