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

## 2026-10-24 — Git LFS storage for bundled references

- GitHub rejected the initial reference-data push because the FASTA exceeded
  its ordinary Git file-size limit. Migrated all files recursively under
  `src/pclass_dark/resources/reference/` to Git LFS, retaining full local data
  and the existing package-resource layout.
- Installed official Git LFS 3.8.1 after verifying the release asset SHA-256.
  Configured repository-local filters and the pre-push hook. The only rewritten
  commit was the unpublished `issue#3` reference-data commit; `main` and remote
  refs were unchanged. Saved a pre-migration bundle at
  `/tmp/pclass-dark-before-lfs.bundle` for local recovery.
- Updated README, reference-update instructions, and SPEC-0002 with LFS checkout
  and release-build requirements. Published Python distributions still contain
  full reference data and require no Git LFS for installation.
- Validation: all four tests passed, including reference checksum and coverage
  checks; `git lfs fsck` passed for all seven objects. Verified that the outgoing
  history contains no ordinary Git blob larger than 100 MiB and that `main`
  retains its original commit. Whitespace checks passed.
- No data bytes or scientific behavior changed. No remote upload performed;
  the branch can be pushed normally, with Git LFS uploading its objects.
