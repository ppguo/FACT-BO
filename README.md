# FACT-BO

Repository: https://github.com/ppguo/FACT-BO

Built with TabPFN.

FACT-BO is a research implementation of an AMS worst-case-analysis workflow
built on the GIT-BO optimizer: a frozen TabPFN-v2 surrogate supplies gradients,
a gradient-informed active subspace concentrates candidate generation, and a
SPICE/Xyce evaluator closes the optimization loop over a physical process
sigma-ball.

The optimizer and TabPFN integration originate from the official
[GIT-BO GitHub repository](https://github.com/rosenyu304/GITBO), the reference
implementation of *GIT-BO: High-Dimensional Bayesian Optimization with Tabular
Foundation Models* (ICLR 2026).

The SRAM simulation path integrates with
[OpenYield](https://github.com/ShenShan123/OpenYield), an Apache-2.0 SRAM yield
analysis framework. OpenYield supplies the external 4x2 SRAM sampling and
Xyce-facing evaluation flow; FACT-BO supplies the worst-case search, sigma-ball
mapping, and the process boundary between the optimizer and simulator. OpenYield
source, PDK/device models, simulator binaries, and private SRAM metadata are not
redistributed here.

## Installation

Python 3.10 or 3.11 and a CUDA-capable PyTorch installation are recommended.
The 144-D SRAM experiment requires a GPU with enough memory for TabPFN and a
CPU environment capable of running OpenYield/Xyce.

The SRAM paper run was verified in Python 3.10.18 with
PyTorch 2.1.2+cu118 and BoTorch 0.16.1. See `environments/paper-192.txt` for the
recorded package versions; the normal project metadata uses compatible ranges
so users can install an appropriate CUDA build for their machine.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Download the `tabpfn-v2-regressor.ckpt` model from the Prior Labs
`TabPFN-v2-reg` distribution and place it at:

```text
tabpfn/model/tabpfn-v2-regressor.ckpt
```

The checkpoint is intentionally ignored by Git and must not be committed.

## Verification

The default tests do not download a model, invoke a GPU, or run a circuit
simulator:

```bash
pytest
# or, when pytest is unavailable:
python tests/run_smoke.py
```

They verify full-support EI by independent density quadrature,
EFMI target/scale/gradient contracts, surrogate integration, the
cube-to-ball mapping and the SRAM objective/evaluator contract.

For an instance-level before/after demonstration that does not require the
TabPFN checkpoint, run:

```bash
python examples/synthetic_fixes/validate_fixes.py
```

The example uses an analytic double-peak objective to validate the relevant
model-integration behavior before and after the corrections.

For a full optimization smoke test that loads the actual TabPFN-v2 checkpoint,
computes SamplingUCB, differentiates the surrogate, and updates the
gradient-informed subspace, see `examples/synthetic_optimization/`. These runs
require the normal GPU/model environment; they are diagnostics rather than
statistical benchmark claims.

## EFMI

`GITBO(..., Acquisition="EFMI", threshold_y=theta)` maximizes
`E[(Y - max(theta, max(observed_Y)))_+]`. Orient responses and the threshold
so larger means worse. Before failure this rewards expected exceedance severity,
not failure probability alone. The active subspace uses posterior-mean gradients.
The `EI` option uses its incumbent target, acquisition-gradient policy,
and the full-support EI integral.

The integral includes both half-normal tail displacement terms and
preserves positive training scales, including values below float32 epsilon.
EFMI stops on invalid scores or degenerate gradients; it does not silently
switch to SamplingUCB. Small positive scores are not treated as zero.

## SRAM paper configuration

The current paper uses a 4x2 6T-SRAM read-delay maximization problem:


| setting             |                   value |
| ------------------- | ----------------------: |
| process dimension   |                     144 |
| domain              | solid sigma-ball,`K=16` |
| initial evaluations |                      30 |
| BO iterations       |                     270 |
| candidate pool      |                    5000 |
| subspace rank       |                       5 |
| sampling scale      |                     1.0 |
| acquisition         |                 EFMI |
| failure threshold   |                290.8 ps |
| seeds               |                    0--4 |

See `examples/sram_read_delay/README.md` for the parameterized OpenYield/Xyce
command. The real integration is kept separate from the `gitbo` Python process
through a persistent line protocol so that their conda/library environments do
not contaminate one another.

## Layout

```text
algorithms/                  adapted GIT-BO optimizer and TabPFN wrapper
tabpfn/                      GIT-BO's vendored TabPFN fork (no checkpoint)
fas_wca/                     sigma-ball and simulator-facing API
examples/sram_read_delay/    configurable OpenYield/Xyce example
tests/                       CPU-only correctness tests
PROVENANCE.md                source snapshot and modification record
NOTICE                       upstream attribution and dependency boundary
```

## Reproducibility boundary

- The adapted optimizer corresponds to the code used for the paper experiments.
- Exact paper reruns additionally require archived seed-specific initial points,
  the audited OpenYield sampling tree, the matching SRAM metadata and the same
  simulator/model environment.
- The example can generate deterministic Sobol initial points for new runs, but
  these are not claimed to be byte-identical to the archived private inputs.
- Generated `.pt`, simulator work directories and private model data are not
  source artifacts and are excluded from the repository.

## License and citation

Upstream projects used by this implementation:

- GIT-BO optimizer: [rosenyu304/GITBO](https://github.com/rosenyu304/GITBO)
  (MIT).
- OpenYield SRAM yield-analysis framework:
  [ShenShan123/OpenYield](https://github.com/ShenShan123/OpenYield)
  (Apache-2.0).

FACT-BO code and the adapted GIT-BO optimizer use the MIT license in `LICENSE`.
The vendored TabPFN source uses the Prior Labs License v1.0 in `tabpfn/LICENSE`,
including its attribution requirements. Preserve both licenses and the notices
in derivative distributions. Model weights, OpenYield, Xyce and device models
remain subject to their own terms and are not included.

Citation metadata is provided in `CITATION.cff`.
