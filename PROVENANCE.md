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
