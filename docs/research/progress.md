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

## 2026-10-24 — Bundled authoritative profunc references

- Copied the authoritative profunc membership TSV and accession FASTA unchanged
  into `src/pclass_dark/resources/reference/`, with checksummed provenance and
  preserved source manifests. Cluster generation:
  `20260911T225655Z-153032271c41`; dataset generation:
  `20260911T184540Z-d7e84090016d`.
- Verified 455,508 unique clustered accessions across 3,524 clusters, all covered
  by 575,748 unique nonempty FASTA sequences (209,017,843 residues). Preserved
  120,240 unclustered records and 1,651 clustered noncanonical sequences.
- Added setuptools resource inclusion, README access guidance,
  [SPEC-0002](../specs/0002-bundled-references.md), and
  [reference update instructions](../reference-data.md).
- Validation: all four tests passed with `PYTHONPATH=src`; source distribution
  and wheel built successfully. Verified archived resource bytes against the
  manifest and verified installed-wheel resource checksums from outside the
  checkout in a separate temporary virtual environment.
- No reference transformations or scientific predictions were performed.
  Future updates must replace the matched pair and provenance together, check
  downstream artifact compatibility, and define noncanonical-residue handling
  before embedding generation.
