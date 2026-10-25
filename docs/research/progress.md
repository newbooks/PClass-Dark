# Research and development progress

## 2026-10-24 — Initial Python package and CLI

- Added version 0.0.1 packaging with the `pclass-dark` console command and
  importable `pclass_dark` namespace. The CLI currently supports help and version
  reporting; no scientific subcommands or algorithms are implemented yet.
- Followed neighboring projects' Typer command/subcommand convention. Scientific
  logic will live in reusable library modules with thin CLI adapters.
- Declared NumPy, SciPy, and Typer as initial core dependencies, optional PyTorch
  and Transformers for future embeddings, and development tooling separately.
- Updated README installation/interface guidance and AGENTS interface rules.
  Added [SPEC-0001](../specs/0001-package-cli.md), tests, and build/cache ignores.
- Validation: three pytest tests passed; wheel build, editable installation,
  console help/version, module invocation, and installed-wheel import/version
  checks passed. Used a temporary virtual environment with existing dependencies
  because the default user install location is read-only in this environment.
- Next: specify and implement the first scientific operation, including its
  Python API, CLI subcommand, data contracts, and tests. Refine dependencies and
  model compatibility when those requirements are known.
