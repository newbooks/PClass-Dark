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

## 2026-10-25 — Exact sequence identity audit and offline setup scope

- Confirmed that planned setup-data uses bundled references directly, without
  another raw-data copy or Internet downloads. User includes embedding-window
  planning and training/validation splitting and excludes homology searches;
  model acquisition remains separate. No setup command implemented yet.
- Compared actual full sequence strings after removing FASTA line wrapping;
  verified source checksums, unique IDs, nonempty sequences, and coverage.
- Found 381,183 distinct sequences among 455,508 clustered accessions; 106,709
  accessions (23.43%) have an identical clustered partner. 1,071 sequence groups
  span clusters, involving 5,386 accessions in 739 clusters.
- Recorded complete counts, input identities, method, and split implications in
  [the audit note](notes/2026-10-25-sequence-identity.md) and its result JSON.
  Detailed local group listings and the analysis script are under ignored
  `work/sequence-identity/`. Reference bytes were unchanged.
- Validation: checksum/coverage assertions and small summary sanity checks
  passed. This was a local data audit, not a code implementation or benchmark.

## 2026-10-25 — Cluster size distribution after exact deduplication

- Counted full sequences once within each functional cluster, preserving shared
  sequences in every assigned cluster. Obtained 382,268 cluster/sequence
  memberships, with 81 of 3,524 clusters falling below 10 unique members.
- Added requested size bins to the sequence-identity audit note; per-cluster
  counts and JSON summaries are in ignored `work/sequence-identity/`.
- Reverified source checksums and the total bin count. No reference edits,
  cluster reassignment, or actual deletion performed.

## 2026-10-25 — Draft offline setup-data specification

- Added [SPEC-0003](../specs/0003-setup-data.md) for offline preparation using
  bundled references directly, global sequence storage with preserved multiple
  cluster labels, within-cluster unique membership counts, and exclusion below
  `--min-clusters` (default 10) before splitting/window counting.
- Inspected PClass's actual `_plm_embedding_windows`: adopted terminal N-/C-
  naming and slices, explicitly changing its current 1024 limit to the user's
  requested 1022. Documented overlap and omitted central regions.
- Proposed deterministic parent-group training/validation assignments, shared
  terminal-window diagnostics, schemas, provenance, reuse/publication safeguards,
  Python/CLI interfaces, and required tests. Split fraction/policy and terminal
  window collision handling remain draft design choices.
- Validation: checked arithmetic against the audit (81 excluded, 3,443 retained
  clusters; 381,627 retained cluster/sequence memberships), inspected PClass code
  and corresponding tests, and ran documentation whitespace checks. No code or
  reference-data changes; setup-data is not implemented yet.

## 2026-10-25 — Required residue-mask pooling contract

- Expanded SPEC-0003 with a downstream embedding contract: final-layer ESM-2
  640-dimensional hidden states, combined attention/non-special-token mask,
  verified residue/token alignment, float32 residue-only pooling, and explicit
  failure behavior. Offline setup still performs no model loading or embedding.
- Required independent numerical fixtures, mask truth-table and failure tests,
  special/padding exclusion, tokenizer boundaries, noncanonical policy tests,
  batching invariance, direct-slice comparisons, and real-checkpoint validation
  before embedding implementation is complete. Added provenance requirements.
- Documentation-only change; checked consistency with the setup scope and ran
  whitespace checks. No code tests or model inference were run for this change.

## 2026-10-25 — Move frozen embeddings into resumable setup-data

- Revised SPEC-0003: setup-data now prepares references and computes frozen,
  residue-pooled PLM embeddings; signature discovery, dataset-fitted transforms,
  calibration, and evaluation remain in the next model-building step.
- Supersedes earlier offline-only setup scope: local preparation remains offline;
  full setup may acquire a pinned checkpoint. Added prepare-only/offline modes,
  GPU preference, explicit CPU support, and portable prepared/embedding bundles
  suitable for later inclusion in package distributions.
- Specified automatic resume with immutable row-range shards, checksummed commit
  receipts, identity checks, writer locking, recovery of invalid/incomplete work,
  and final matrix assembly without repeating validated inference. Specified
  fault-injection tests and model-call assertions to prove resumability.
- Documentation-only specification update. Checked internal consistency and
  whitespace; implementation, checkpoint pinning, and execution remain pending.

## 2026-10-25 — Setup-data readiness review paused at split-unit decision

- Reviewed SPEC-0003 for implementation readiness. Identified a scientific
  choice: full-sequence grouping alone permits identical terminal windows from
  different parents to occur in both training and validation.
- Marked the draft as awaiting this decision and documented a recommendation
  to split connected components of parents sharing exact window strings,
  preserving full-sequence deduplication and all cluster labels. Documented the
  alternative and its evaluation limitation. The split algorithm remains
  provisional; no implementation or scientific policy selection was made.
- Stopped specification work for user input as requested. Documentation
  whitespace checks passed; no code or data changed.

## 2026-10-25 — Approve exact-window connected components for splitting

- User approved keeping all windows of one parent together and keeping different
  parents with identical windows together. Updated SPEC-0003 to require
  transitive connected components, stable component IDs, preserved multi-cluster
  labels, explicit component outputs, and fatal cross-split identity checks.
- Continued readiness review and reached the next scientific choice: whether
  setup produces separate training/calibration/test partitions or a two-way split
  whose held-out data are used for calibration. Recommended three partitions;
  marked fractions and assignment policy as pending rather than final.
- Stopped for user input as requested. Documentation whitespace checks passed;
  no implementation or reference-data edits.

## 2026-10-25 — Replace grouped splitting with five random window folds

- User explicitly permits N/C windows of one protein to occupy different folds
  and requests five folds in setup-data. This supersedes earlier approvals of
  parent-group and shared-window connected-component restrictions.
- Revised SPEC-0003 to assign window records independently using a seeded,
  versioned hash shuffle and round-robin fold IDs 0..4. No stratification or
  parent/identity constraints; global fold window counts differ by at most one.
- Replaced component artifacts with window fold assignments and per-cluster,
  per-fold counts. Defined next-step rotation and mean/sample-SD reporting,
  including undefined metrics and unsupported rounds. Shared parents/identical
  windows across folds are reported diagnostics, not errors.
- Retained explicit limitations of window-level internal validation and deferred
  calibration methodology to the next step. Documentation checks passed; no
  implementation, data changes, or tests of new code were performed.

## 2026-10-25 — Clarify per-cluster C folds and ProStruct capped B

- User clarified that five-fold assignment applies separately within target C,
  not globally, and requests ProStruct's fixed per-source-cluster-capped B.
  Inspected ProStruct cluster_sets.py and SPEC-0027: full uncapped C, cap 200 per
  other source cluster, one fixed selection reused across targets/features.
- Revised SPEC-0003 with target-specific `(cluster_id, window_id)` folds and B
  fixed across rounds. Added handling for multi-label target exclusion and union
  deduplication, plus interpretation of shared-background diagnostics.
- Reached a scientific adaptation question: cap 200 full-sequence parents (up
  to 400 N/C vectors), or cap 200 windows. Recorded alternatives and paused for
  user decision; no sampler or setup code implemented.
- Documentation whitespace checks passed. Earlier globally balanced assignment
  and per-window global fold IDs are superseded.

## 2026-10-25 — Approve 200-window cap for each background source cluster

- User selected windows as the B sampling unit. Updated SPEC-0003 with fixed
  cap 200, uniform sampling without replacement, source-specific SHA-256/PCG64
  seeds, persistent selections, and reuse across targets/features/C-fold rounds.
- Specified union by window ID, exclusion of all known target parents across
  source labels, no refill after exclusions, final background identity/counts,
  compact definitions and summary outputs, and explicit empty-B status.
- No sampling or code implementation performed. Documentation checks passed;
  technical readiness items remain listed in the specification.

## 2026-10-25 — Compact C/B definitions bound to exact inventories

- Compared ProStruct's C/B JSON validation with SPEC-0003. The earlier draft
  specified compact B with selected IDs but lacked an exact C/index contract.
  Added ProStruct-style C arrays and B per-source indices or `all`, with exact
  C-file checksum binding, pair identity, and prepared-manifest file hashes.
- Defined strict index/schema/count/order validation, immutable verified reads,
  final expanded-B count/digest checks, and separate verified window-ID joins to
  embedding rows. Documented corruption/reordering/mixed-generation rejection
  and required fault-injection tests; no runtime code implemented.
- Documentation checks passed. Checksums protect accidental integrity, not
  authenticity when all files and manifests can be maliciously replaced.

## 2026-10-25 — SPEC-0003 implementation-readiness proofread

- Completed a consistency/readiness pass and marked SPEC-0003 ready for
  implementation. Preserved approved per-cluster C folds and 200-window B cap.
- Pinned official ESM-2 revision a695f6045e2e32885fa60af20c13cb35398ce30c;
  verified configuration/vocabulary bytes and upstream safetensors size/digest.
  Specified supported residue alphabet and fixed float32 execution/tolerances.
- Resolved custom thresholds below five, exact TSV headers/serialization,
  identity/hash dependencies, model-free reuse, locks, publication/selection
  semantics, and dependency boundaries. Reduced resumable shards to 64 windows
  to bound recomputation after interruption. Model weights were not downloaded.
- Validation: local EsmTokenizer initialized from the pinned vocabulary passed
  residue-ID/order and combined-mask counts for lengths 4, 5, and 1022,
  including X/B/Z/U/O. Markdown link/fence/table checks and whitespace checks
  passed. No setup-data runtime code or full-model embeddings were implemented
  or tested. Calibration methodology remains explicitly outside this spec.

## 2026-10-25 — Implement setup-data and validate resumable embeddings

- Implemented reusable reference loading, preparation, compact C/B sets,
  five per-cluster window folds, and pinned ESM-2 mask-based float32 pooling.
  Added the setup-data CLI, optional inference stack, portable resource patterns,
  LFS rules, identity selection, locks, archives, and checksummed 64-row shards.
- Added automated integrity, mask, tokenizer, numerical, multiprocessing,
  publication/rollback, and inference recovery tests. Real-checkpoint CPU checks
  and a 66-window interrupted/resumed inference run passed; CUDA unavailable.
- Offline full-reference preparation reconciled all audit bins and produced
  3,443 clusters, 380,578 unique parents, and 393,548 windows. Verified complete
  reuse succeeded. Raw reference resources were unchanged.
- 48 tests passed, one GPU check skipped; Ruff and whitespace checks passed.
  Source/wheel packaging and installed read-only, model-free artifact access
  were validated with a small synthetic prepared/embedding bundle.
- Synchronized README, SPEC-0003 and its index. Evidence and remaining limitations
  are in [the execution note](notes/2026-10-25-setup-data.md). Full production
  embeddings, GPU validation, signature discovery, and calibration were not run.

## 2026-10-25 — Bracket CLI log levels

- Changed setup-data logging to `[LEVEL] message`, including background-identity
  progress. Synchronized SPEC-0003 and added a subprocess CLI rendering check.
- Validation: all four CLI tests passed; Ruff and whitespace checks passed.

## 2026-10-25 — Log durable embedding completion only at the end

- Moved the durable-commit message from each shard to successful final bundle
  publication. Batch/range progress remains. Added a finalization-failure and
  resumed-success log test; synchronized SPEC-0003.

## 2026-10-26 — Recover from structurally malformed checkpoint receipts

- Reproduced review failures for a top-level JSON list and null/list execution
  metadata. Require object-shaped receipt, execution, and contract fields;
  route schema TypeError/AttributeError into quarantine as a defensive fallback.
- Added four end-to-end receipt-corruption cases. Each proves that only the
  affected 64-window shard is recomputed, the array and receipt are quarantined,
  other shards survive, and the completed result is reusable on the next retry.
- Synchronized SPEC-0003. Validation: 53 tests passed, two opt-in real-model
  checks skipped; Ruff and whitespace checks passed. No reference data changed.

## 2026-10-26 — Own publication transactions through selection and rollback

- Fixed the concurrent-reuse rollback race with a root-wide setup transaction
  guard held until selection publication succeeds or rollback finishes. Other
  invocations and mutable C/B/vector readers fail busy while a generation can
  still be reverted. Same-thread internal calls are reentrant; independent
  output roots remain concurrent. Installed immutable releases need no locks.
- Added a deterministic threaded regression pausing force publication before
  selection, checking concurrent reuse/read rejection, then verifying restored
  selection digests and successful subsequent reuse. Added same-root nesting,
  cross-process exclusion/release, and independent-root tests.
- Synchronized SPEC-0003. Validation: 55 tests passed, two opt-in checkpoint
  checks skipped; Ruff and whitespace checks passed. No reference data changed.
