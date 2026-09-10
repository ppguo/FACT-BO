# Source-release checks

The FACT-BO source release includes the optimizer and parameterized simulator
adapters. OpenYield, model weights, device models and private SRAM metadata
remain external dependencies.

## Publication preparation (2026-09-10)

- Author order and paper title in `CITATION.cff` match the current manuscript.
  No unassigned DOI or arXiv identifier is claimed.
- The MIT license retains GIT-BO attribution. The vendored TabPFN source has
  its own Prior Labs License v1.0, modification notices and required attribution.
- TabPFN's license was copied from the official 2.0.3 source archive after
  verifying its PyPI SHA-256 checksum; see `tabpfn/NOTICE`.
- A fresh mirror of the remote repository was scanned with Gitleaks 8.30.1.
  The seven pre-publication commits had no detected secrets. Their 83 historical
  file versions contained no model checkpoints, PDK files or simulator data.
- The remote had one branch (`main`), no tags, no release assets and no previous
  Actions runs at the time of the audit.
- `.github/workflows/tests.yml` runs CPU correctness and smoke tests without
  model downloads or circuit simulation. The Actions result records each run's
  actual outcome.

## Reproducibility follow-up

- Add the final publication DOI or arXiv identifier when available.
- Exact simulator replay requires the audited OpenYield revision, matching
  private metadata, archived initial designs and the simulator/model environment.
  Generated Sobol points are supported but are not asserted to match the archives.
- Record a fresh-clone full simulator seed and its environment/checksums when
  those external inputs are available. A CPU CI pass does not establish circuit
  reproduction or estimator quality.

For subsequent releases, rescan all history and new assets before publication.
Keep unrelated working repositories and private simulator assets out of this
source repository.
