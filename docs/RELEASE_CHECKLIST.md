# Public-release checklist

The source tree is technically staged and its CPU smoke tests pass in the
paper's 192-server `gitbo` environment. Complete these items before making the
repository public:

- Replace generic author metadata in `pyproject.toml` and
  `CITATION.cff.template`; rename the completed file to `CITATION.cff`.
- Add the final paper title, author order, DOI/arXiv identifier and public
  repository URL.
- Confirm redistribution terms for the vendored TabPFN fork. Keep the model
  checkpoint external regardless of the outcome.
- Decide whether an audited OpenYield commit will be a documented external
  dependency or a submodule. Preserve its Apache-2.0 attribution.
- Audit the SRAM metadata, configuration and device models. Publish only assets
  that are demonstrably redistributable.
- Recover or regenerate the five archived 30-point initial designs if exact
  paper reruns are a release goal.
- Run one full OpenYield/Xyce seed from a fresh clone and record the command,
  GPU, Xyce version, OpenYield revision and output checksum.
- Configure public CI for `pytest` and `python tests/run_smoke.py` without a
  model download or circuit simulator.
- Review `git status`, the full Git history and a fresh-clone secret scan before
  pushing. Do not import history from the credential-bearing OpenYield working
  copy.
