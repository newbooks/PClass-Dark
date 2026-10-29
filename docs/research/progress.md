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

## 2026-10-26 — Draft build-library signature-discovery specification

- Added SPEC-0004 for 640-dimensional PLM signatures using four C folds for
  discovery, one held-out C fold for validation, and unchanged compact B capped
  at 200 windows per other source cluster, preserving the approved setup policy.
- Specified fold-fitted means/spreads, five R/D and individual-AUROC values per
  dimension, AUROC ranking, arithmetic Gaussian composites, and ProStruct's
  protected-core/prefix/candidate/backward-removal algorithm generalized to 640.
- Defined training/validation metric and fold-stability reporting, residue-only
  pooling provenance, output proposals, integrity/performance requirements, and
  tests. Proposed training-only balanced-accuracy thresholds; production refit
  versus fold-ensemble and query confidence/assignment remain explicit decisions.
- Documentation-only draft. Checked local ProStruct implementations and setup
  contracts; no code, reference changes, or discovery experiment performed.

## 2026-10-26 — Approve build-library threshold and production refit

- User approved maximizing balanced accuracy for evaluation thresholds and
  refitting the final signature on all C members. Updated SPEC-0004 accordingly.
- Each rotation selects its cutoff using training C versus fixed B only and
  applies it unchanged to validation. The final fit reruns discovery and cutoff
  selection using full C versus the same B; its diagnostics are in-sample and
  are not included as a sixth fold. Added final.json and explicit refit tests.
- Query confidence, cross-cluster competition, and N/C aggregation remain open;
  technical schemas/resource/checkpoint details still require finalization.
  Documentation checks passed; no code or experiment run.

## 2026-10-26 — Finalize build-library technical specification

- Revised SPEC-0004 to implementation-ready without changing approved science:
  exact AUROC pair-count comparisons, training-only balanced-accuracy cutoff
  ties, rank-ordered float64 composite arithmetic, per-fit insufficiency states,
  and full-C refit separated from five-fold aggregates.
- Defined CLI/API, explicit table and signature schemas, representation/library
  identities, deterministic compression, complete release layout, NumPy scoring,
  bounded CPU/mapped-cache execution, and an initial reference backend. Discovery
  needs no Torch or model download; setup remains the PLM/GPU stage.
- Specified profile/prefix blocks and per-trial backward checkpoints, dependency
  validation, malformed-receipt recovery, input/output transaction ownership,
  selection/rollback rules, immutable release reads, and performance gates.
- Added acceptance cases for numerical equivalence, within-fit resume,
  concurrency, serialization, and installed scoring. Query confidence,
  cross-cluster competition, and query-window aggregation remain out of scope;
  any performance shortcut changing selection science requires user input.
- Documentation-only revision. Checked links, headings, code fences, and
  whitespace. No code, computational benchmark, or scientific run performed.

## 2026-10-26 — Proofread SPEC-0004 for implementation readiness

- Confirmed the approved fold-wise discovery, fixed background, training-only
  balanced-accuracy threshold, and all-C final refit remain consistent.
- Closed technical gaps: persisted exact per-dimension AUROC counts, explicit
  predict-none serialization, empty-validation aggregate handling, unsupported
  OOF rows, partial-profile dependencies, and sequential per-target scheduling.
- Made library reader/rollback guard roots explicit: `<library-dir>.parent`,
  rather than setup-data's deeper parent.parent layout. Added associated tests
  to the required acceptance list and clarified safe prefix-cache reuse.
- SPEC-0004 remains ready for implementation; no new scientific decision is
  required. Documentation links, fences, tables, field contracts, and whitespace
  checked. No runtime code or scientific calculations were executed.

## 2026-10-26 — Implement build-library

- Implemented [SPEC-0004](../specs/0004-build-library.md): five C-fold rotations,
  fixed capped B, all 640 dimension profiles, exact AUROC ranking, protected-core
  arithmetic Gaussian composition/backward selection, training balanced-accuracy
  thresholds, and a separate all-C refit.
- Added CLI/Python discovery and immutable NumPy scoring APIs, bounded process
  workers, resumable within-fit caches, strict input/receipt integrity, complete
  library reports, and guarded publication/rollback. Added explicit publication
  domain support to the shared transaction helper.
- Full suite: 79 passed, 2 opt-in real-checkpoint tests skipped; subsequent
  publication regressions: 4 passed. Ruff passed. Built staged wheel/source
  archives with a small complete library; installed scoring outside the checkout
  worked with PLM imports blocked and no package writes.
- Completed the scoped real-data build for cluster_0945 (915 C / 213,061 B),
  including five rotations and final refit; [benchmark and limitations](notes/2026-10-26-build-library-benchmark.md).
  Validation AUROC 0.994898 ± 0.001078 is an internal diagnostic; final in-sample
  precision was 0.097486 at the balanced-accuracy threshold. No full-cluster
  production build or independent predictive evaluation was performed.
- Synchronized README, packaging/LFS rules, specification, and index. Query
  assignment, cross-cluster calibration, and N/C score aggregation remain future
  scientific work. No commit or push performed.
- Final post-change full-suite rerun: **79 passed, 2 skipped** in 147.35 s,
  with one environment warning from Torch about AMD_SERIALIZE_KERNEL. Ruff lint,
  format checks, documentation links/fences, and `git diff --check` passed.
  Rebuilt wheel/source archives after final changes and repeated installed
  scoring; the real-data library also passed loading/integrity/scoring checks.

## 2026-10-26 — Clarify signature progress messages

- Changed fit-start wording to “computing signature”. Removal-trial progress now
  includes the cluster and fit (`fold_0`–`fold_4` or `final`), since iterations are
  local to each signature fit. Checkpoint behavior and selection are unchanged.
- Verified fold/final logging, selection mathematics, backward-trial resume, and
  end-to-end library construction: 12 tests passed. Ruff and whitespace checks
  passed; synchronized SPEC-0004.

## 2026-10-26 — Accelerate backward signature selection

- Implemented the user-approved first-improvement policy: visit remaining
  non-core dimensions in individual-rank order, accept the first strict gain,
  restart, and stop on a complete non-improving round. Resume receipts validate
  the visited prefix and accepted decision rather than assuming all trials ran.
- Added bounded per-subset running-prefix caches, cached total log sums, stable
  `expm1` leave-one-out subtraction, cancellation and near-tie reference
  fallbacks, and reference confirmation of every proposed gain. Reuse output
  buffers in composite, prefix, deletion, and runtime scoring; prefix resume
  copies read-only mapped checkpoints before in-place updates.
- Versioned library/numerical identities and fit/selection dependencies to keep
  old best-improvement results and checkpoints separate. Raised conservative
  workspace admission estimates for the bounded prefix caches. Old complete
  libraries remain readable; setup-data embeddings remain reusable.
- Final suite: **91 passed, 2 opt-in tests skipped**, one existing Torch
  environment warning, 142.86 s. Tests cover first-versus-best paths, cancellation,
  underflow, ties, seeded reference-equivalent selections, buffer/cache reuse,
  version invalidation, interrupted trials, held-out isolation, and parallel
  determinism. Ruff, formatting, whitespace, and documentation checks passed.
- Rebuilt wheel/source archives and verified installed scoring outside the
  checkout with PLM imports blocked and no package writes.
- Focused real-kernel timing: **3.45× faster** for a complete 158-trial round,
  identical AUROC counts. [Reproduction details and scope](notes/2026-10-26-removal-optimization.md).
  A new end-to-end run could not acquire the setup-data transaction lock; the
  other invocation was left undisturbed. No whole-library speedup claim, full
  production build, commit, or push performed.

## 2026-10-26 — Keep library workers continuously occupied

- Replaced target batch barriers with a rolling, bounded future queue. Each
  completion admits the next cluster without waiting for slower active targets;
  descriptors and expanded background indices are created only at submission.
- Report durable completions individually and restore original target order
  before report assembly, preserving scientific identities, checkpoint reuse,
  deterministic artifacts, and the admitted workspace/worker bounds.
- Added event-controlled tests proving refill while an earlier target remains
  blocked, bounded descriptor creation, and failure propagation without further
  submissions. Pending work is cancelled on scheduler exit.
- All 17 library tests passed in 318.43 s, including real spawned-process
  serial/parallel consistency, checkpoint recovery, and publication regressions.
  Ruff lint/format, documentation links/fences, and whitespace checks passed.
  Updated README and SPEC-0004. No commit or push performed.

## 2026-10-27 — Reclaim completed-cluster caches and bound disk growth

- Added cluster-completion cleanup after atomic, fsynced `target.json`
  publication. Verify all six complete fit receipts against target results,
  payload hashes, task/library/cluster/fit bindings, profile coverage, and trace
  identities before unlinking known Gaussian/prefix arrays and their receipts.
- Preserve complete results, profiles, compact selection checkpoints, published
  artifacts, and unfinished cluster caches. Cleanup is idempotent and retryable
  after partial unlink failure. Maintenance API `clean_library_cache(output_dir)`
  uses output-root and target locks; runtime cleanup holds its target lock.
- Executed user-authorized cleanup of 86 completed target checkpoints in
  `work/work/library` and one in `work/library-benchmark`: **564.917 GiB**
  reclaimed across 64,641 disposable files. Preserved 10 unfinished targets.
  Run directory shrank from 585 GiB to 26 GiB; benchmark directory to 22 MiB.
  Available disk rose from about 224 MiB to about 565 GiB. Exact counters:
  [cleanup record](notes/2026-10-27-library-cache-cleanup.json).
- Cluster-level cleanup keeps six-fit arrays only for active/unfinished clusters:
  roughly 50–60 GiB for eight active workers with this dataset, plus compact
  results and reports. Scientific identities and embedding data are unchanged.
- Full suite: **102 passed, 2 opt-in tests skipped** in 160.73 s, with the existing
  Torch environment warning. Strengthened recovery test also passed separately:
  a damaged complete receipt regenerated only its fit's deleted arrays; intact
  fits resumed without arrays. Tests cover automatic cleanup, integrity gates,
  unknown-file preservation, symlink rejection, root/target locks, and retry
  after partial deletion. Ruff lint/format, docs, and whitespace checks passed.
- Synchronized README/SPEC-0004; annotated earlier benchmark notes to record
  removal of their disposable caches. No production build restart, commit, or
  push performed.

## 2026-10-27 — Draft final-signature relative-score distribution extension

- Created [SPEC-0005](../specs/0005-relative-score-distributions.md), a planned
  `build-library` stage computing sorted final-signature composite log scores on
  the exact final C and fixed B pools after signature definition. Existing five
  rotations and all-C refit remain unchanged.
- Defined explicit empirical B-tail/C-percentile formulas, inclusive ties,
  finite-sample resolution, unsupported cases, data provenance, artifact schemas,
  independent companion publication, bounded scoring, integrity/resume rules,
  installed APIs, and tests. Keep scientific library/checkpoint identities stable
  and backfill old results without discovery, threshold fitting, or PLM inference.
- Recorded [reasoning](notes/2026-10-27-relative-score-reasoning.md): B comparison
  uses C only indirectly; direct C comparison adds member compatibility. Reused
  B and in-sample final C are descriptive, not calibrated membership probabilities.
  Fold diagnostics are separate and are not pooled into final-score references.
- Linked the draft from README, SPEC-0004, and the spec index. This is documentation
  only; no implementation, discovery run, data changes, or runtime tests performed.
- Checked local links, code fences, trailing whitespace, and `git diff --check`
  for all six changed documents; all passed.

## 2026-10-27 — Revise production discovery to final-only by default

- Rewrote [SPEC-0004](../specs/0004-build-library.md) as an unimplemented revised
  target: one all-C fit plus final C/B distributions by default; optional
  `--cross-validate` research with five rotations, usually cluster-scoped.
  Preserved the prior implemented contract and its evidence in the explicitly
  [historical six-fit specification](../specs/0004-build-library-legacy-six-fit.md).
- Separated production, distribution and research identities/publication, with
  final-fit dependencies independent of research mode. Defined validated reuse
  of legacy final receipts/bundles, including incomplete surrounding clusters,
  unchanged model payloads/scores, provenance and conflict handling. Research
  cannot replace production parameters or block initial production availability.
- Revised completion/report/cleanup contracts to remove six-fit assumptions;
  no fabricated CV metrics on default runs. Retained first-improvement numerics,
  fixed B, all 640 profiles, rolling workers and mandatory integrity protections.
- Synchronized SPEC-0005 stage order, migration wording and optional research;
  updated README/index and the relative-score reasoning record. These are planned
  changes only: no runtime code, data, tests, discovery or deployment changed.
- Documentation links, code fences and trailing whitespace checked across seven
  changed documents; `git diff --check` passed. Runtime tests were not needed
  for this documentation-only revision.

## 2026-10-27 — Review SPEC-0004/0005 implementation readiness

- Reviewed both specs against current orchestrator/loader, receipts, publication
  locks, report headers, packaging and actual private final checkpoint metadata.
  Found and closed technical gaps; both specs are now marked ready to implement,
  with no new scientific decision needed and no runtime implementation claim.
- Fixed concrete v2 production/signature, research, companion and selector
  contracts; stable scientific-payload hashing; research-independent production
  metadata; exact force/scope/reuse/error semantics; and typed cleanup gates.
- Documented the private-legacy proof limitation: receipts omit the full library
  identity preimage. Require a verified published recipe or reconstructed exact
  legacy hash/dependency; preserve/log unprovable sources instead of guessing.
  Recorded the misleading legacy largest-strict label under first-improvement.
- Resolved distribution-stage ordering, stable private-array versus destination
  wrapper bindings, finite/tied/empty-query behavior, unsupported counts, bounded
  runtime LRU cache and mutable lazy-read integrity checks. Defined one-way
  production-to-companion-to-selector dependency to avoid identity cycles.
- Required release of the production root guard during optional research so
  publication genuinely makes production queryable before CV finishes; added
  concurrent-reader and generation-change acceptance tests. Clarified which
  provenance differences are excluded from deterministic-byte comparisons.
- Updated index/README status links. Documentation-only review: no source code,
  input data, model computation or runtime tests changed/executed.
- Local links, code fences, trailing whitespace and `git diff --check` passed
  for the five updated documents.

## 2026-10-27 — Implement final-only production, relative scores and optional research

- Implemented SPEC-0004/0005: one all-C fit by default, exact C/B empirical score
  distributions after final definition, and separate `--cross-validate` research.
  Maintained the scientific first-improvement algorithm, fixed pools, thresholds,
  residue-only representation and bounded rolling multiprocessing.
- Added stable per-fit/payload identities, source provenance, proved legacy
  recipe reconstruction, published/private final import and v2 scoring-only
  backfill. Complete legacy final fits are reusable without six-fit target
  markers; unknown sources remain preserved. No PLM or threshold refit on import.
- Added independent production/relative/research schemas, paired and research
  selectors, explicit transaction domains, report/parameter/rank validations,
  staged file-tree fsync, mode-aware cleanup, and immutable NumPy evidence APIs.
  Production remains readable by other threads while CV runs; CV failure retains
  the published production pair. Distribution caches are bounded/thread-safe.
- Full suite: **127 passed, 2 opt-in tests skipped** in 239.88 s; one existing Torch
  environment warning. Subsequent hardened checks: **34 focused passed**, **3
  publication/rollback passed**, and **2 production/research metadata passed**.
  Ruff lint/format, whitespace and docs checks passed.
- Isolated real-data import/scoring/CV check: cluster_0001, 20 C/213,241 B windows,
  all six fits imported, zero discovery fits computed; 20.97 s. Bitwise original/
  migrated query parity for 32 vectors; no Gaussian caches after completion.
  [Record](notes/2026-10-27-final-only-backfill.md). No full production sweep or
  user-run restart performed.
- Built source/wheel with a small complete production/relative/research release.
  Installed loading/scoring/evidence worked outside the checkout with PLM imports
  blocked, setup-data matrices absent, and package writes denied. Updated README,
  package-data patterns, specs, index and reasoning record. No commit or push.

## 2026-10-27 — Clean the unfinished legacy run for an all-C restart

- Located the user run in `work/work/library` with setup inputs in
  `work/work/setup-data`. Acquired both transaction guards in canonical order;
  all examined per-cluster writer locks were free. Did not stop/restart any job.
- Revalidated the actual prepared/embedding data and reconstructed supported
  legacy identity recipes. Verified **194 current first-improvement final fits**
  through the same import/content/dependency checks used by the new builder.
- Removed **4,110 disposable files**: 2,752 Gaussian/prefix arrays and 1,358
  associated cache receipts, totaling **37.763 GiB**. These were legacy fit
  caches, including incomplete folds that default production will not execute.
  Retained profiles and compact trial/decision state so optional research can
  reuse completed results or reconstruct missing intermediates later.
- Preserved **199 final receipts** (194 supported plus five older-version copies),
  **1,021 completed fold receipts**, target records, all other compact state,
  setup-data matrices, sequences and references. Fingerprinted 1,419 complete/
  target records before/after deletion; all were unchanged. No Gaussian/prefix
  array files remain in this legacy tree.
- The run directory shrank from about 44 GiB to **5.8 GiB**; available filesystem
  space increased to about **586 GiB**. No full new production build was launched.
  [Summary/provenance](notes/2026-10-27-all-c-restart-cleanup.json); detailed deleted
  file inventory is `work/all-c-restart-cleanup-2026-10-27.json`.
- Maintenance/data cleanup only, no runtime code changes. The import validators,
  retained-record hash checks, cache absence checks and documentation whitespace
  checks passed; no additional code tests were required. Restart normally using
  the same input/output roots, without `--force` or `--cross-validate`.

## 2026-10-27 — Share artifacts independently of cwd and export them in packages

- Implemented [SPEC-0006](../specs/0006-shared-artifacts.md): source/editable CLI
  and Python defaults resolve through imported package location to repository
  `data/setup-data` and `data/library`, regardless of cwd. Explicit overrides
  retain their existing semantics. Created the shared folder instructions.
- Installed packages prefer available user-cache selections then bundled data,
  and write new output only below the user cache. Selected-library loading now
  accepts no path. Fixed immutable detection for the setup resource root itself
  so builds from bundled embeddings do not attempt package-directory locks.
- Added setuptools hooks exporting only selected completed prepared/embedding,
  scientific/relative/research bundles through manifest-bound copies. Source
  archives carry the selected shared data/build helper; their wheel rebuilds
  export the same resource payload. Writer locks, checksum/binding checks,
  generation fingerprints and stale-output cleanup protect release assembly.
- Added Git LFS coverage for completed data and ignore rules for private hidden
  checkpoint/staging/archive/lock directories. README remains ordinary text.
  No full data preparation/inference/discovery or migration of custom work paths
  performed; existing work folders were left untouched.
- Full suite **135 passed, 2 opt-in tests skipped** in 202.13 s, with the existing
  Torch environment warning. After the installed-root fix, **13 focused path,
  release-validation and CLI tests passed**. Ruff lint/format and docs/whitespace
  checks passed. The base editable install resolves the new absolute repo paths.
- Built source/wheel from a small complete shared fixture: 49 resource files,
  setup embeddings and library companions included, private dummy cache excluded.
  Installed verification outside checkout passed with PLM imports blocked and
  package writes denied: bundled reads/query evidence, cache-only build output,
  and cache-only setup prepare-only output. No commit or push performed.
- Verified an isolated editable install outside its checkout; it resolved shared
  source data correctly. Editable builds skip generated-data copies. Rebuilt the
  source/wheel after this hook refinement and ignored hidden temporary files as
  well as private directories.

## 2026-10-27 — Document future A/B/C query ranking presentation

- Added planned [SPEC-0007](../specs/0007-query-ranking-presentation.md) and linked
  it from the specification index and README. Adopted A — Acceptance (Yes/No),
  B — Background score ↓, and C — Cluster score ↑ for top query matches.
- Recorded existing cutoff, plus-one background fraction and inclusive member
  percentile semantics, with percentage display and lay explanations. Described
  an initial accepted-candidate/B-ascending ranking heuristic, supporting C
  evidence, ties, unavailable data and no-match handling.
- Marked ranking as future work requiring held-out validation; no calibrated
  probability or protein-level window aggregation behavior is claimed.
- Documentation-only change; checked local links, formulas and whitespace. No
  runtime code, data or tests changed; no code tests required.

### 2026-10-27 — Keep research selection compatible with production

Fixed the release-blocking sequence of building research for one cluster subset
and then selecting a different production subset. Production selection now
archives an incompatible research selector under `.archives/`, preserving the
research bundles and checkpoints; compatible selectors survive reuse. Research
publication also checks the currently selected production under the root guard,
preventing a research run from restoring a stale selector after a concurrent
production switch. See [SPEC 0006](../specs/0006-shared-artifacts.md).

Focused production/distribution regression tests: **18 passed** in 125.93 s.
Coverage includes selected-artifact packaging inventory after the switch,
archived selector and preserved research manifest, compatible reuse, and a
production switch between research computation and publication. Ruff lint,
formatting checks, and `git diff --check` passed. Full-suite result follows.

Full regression suite: **138 passed, 2 opt-in tests skipped** in 257.37 s;
the existing Torch `AMD_SERIALIZE_KERNEL` environment warning remains.

### 2026-10-27 — Bind research reports to their production provenance

Research report identity now includes the source production ID, relative bundle
path, and manifest digest. Previously, K0 research built against K0-only
production could be incorrectly reused after selecting full production, leaving
a selector that disagreed with the report manifest and blocking packaging.
Changing production now creates a distinct report bundle. Fold checkpoints
are report-scoped, so a production switch currently repeats fold computations;
compatible final fits remain reusable. Report loading checks identity-bound
provenance against the manifest; historical reports remain readable without
being reused under the new identities. Specifications [0004](../specs/0004-build-library.md)
and [0006](../specs/0006-shared-artifacts.md) record the contract.

Added a regression covering K0-only research, full production, K0 research again,
release inventory validation, unchanged original report, no repeated final fits,
same-production report reuse, and rejection of conflicting provenance.

The first focused run had 18 passing tests and one failing regression assertion:
it incorrectly expected report-scoped fold checkpoints to survive a report-ID
change without recomputation. Updated that expectation to the actual namespace
contract (five fold fits repeated, final fit reused), and documented the resource
cost rather than broadening this fix to change checkpoint storage.

Final validation: **19 production-library/distribution tests passed** in 111.84 s,
including the provenance-switch regression and tampered-provenance rejection.
Ruff lint and formatting checks and `git diff --check` passed. The full suite was
not rerun for this focused provenance fix.

### 2026-10-27 — Import legacy research with empty validation folds

Fixed published legacy research migration when a fold has zero validation
members. Such folds have no OOF table rows and therefore no spooled JSON file.
Migration now supplies an empty OOF list only after checking the signature's
validation count is zero. A missing file for a nonempty fold remains an explicit
integrity error, and imported results still validate exact validation IDs.
See [SPEC 0004](../specs/0004-build-library.md).

The four-protein/two-cluster regression (`min_clusters=1`) removes private legacy
checkpoints and forbids signature recomputation. It verifies ten published fold
imports (six empty, four populated), four OOF rows, preserved source manifest,
valid research output, and usable selected production. Initial regression run:
**1 passed** in 18.79 s. Broader library checks follow.

Broader validation: **33 production and legacy library tests passed** in 256.29 s,
including serial/parallel agreement, migration, resume, and integrity checks.
Ruff lint/formatting and whitespace checks passed. Full suite not rerun for this
focused migration fix.

### 2026-10-28 — Short help options for data and library commands

Added `-h` as an alias of `--help` for `setup-data` and `build-library`.
README and specifications 0003/0004 document both forms. Parameterized CLI tests
verify identical output and successful exit for both subcommands: **6 CLI tests
passed**. Ruff lint/formatting and whitespace checks passed. Existing untracked
setup-data artifacts were left untouched.

### 2026-10-28 — Label reference distribution progress as ABC match evidence

Renamed the per-cluster build progress message to `ABC match evidence: computing
or reusing reference distributions`. This is a display-only change; distribution
construction, artifact names and APIs remain unchanged. SPEC 0005 records the
label and distinguishes this step from probability calibration. The existing
production-distribution test now checks the message for both fixture clusters.

Validation: production-distribution regression **1 passed**; Ruff lint/formatting
and whitespace checks passed.

### 2026-10-28 — Draft explain-signature specification

Drafted [SPEC 0008](../specs/0008-explain-signature.md) using the user-reviewed
cluster_0001 narrative. Defines the CLI and importable report/rendering API,
window membership explanation, Gaussian similarity terminology, ranked-prefix
candidate selection, removal history, composite score/threshold, in-sample
metrics and reference distributions for ABC evidence. Scope excludes query
examples, prediction, biological mechanism claims and probability calibration.
Documents provenance/integrity checks, immutable installed use, unavailable
artifacts, resource limits, deterministic formats and implementation tests.
Implementation remains planned. No scientific code or local artifacts changed.

### 2026-10-28 — Review explain-signature implementation readiness

Revised [SPEC 0008](../specs/0008-explain-signature.md) against actual loaders,
selection/production schemas, fit traces and relative arrays. Clarified the
versioned report shape and immutable API result, exact artifact loading and
snapshot guards, v1-selector/v2-production no-companion support versus legacy
production rejection, malformed-field errors, supported selection version,
integer-based trace checks and core-only termination. Defined a target summary
API with bounded verified array reads and explicit log-space quantile/underflow
behavior. Corrected the overly strict transitive-import requirement: reports
need no setup-data artifacts or heavy PLM runtime, while lightweight modules may
be imported by existing loaders. Added implementation sequence and refined tests.
No scientific decisions remain open; marked ready for implementation, with the
command still unimplemented. Existing data and code changes were left untouched.

Readiness validation passed: local documentation links, fenced-block balance,
trailing-whitespace and scoped diff checks. Documentation-only revision; no code
tests or artifact regeneration were required.

### 2026-10-28 — Implement explain-signature

Implemented [SPEC 0008](../specs/0008-explain-signature.md): required single
`--cluster`, default shared/bundled library resolution, text/Markdown/JSON,
`--detailed`, and both help aliases. The immutable Python report and pure renderer
explain window counts, dimension evidence, ranked candidates, exact recorded
first-improvement decisions, Gaussian/composite scoring, threshold, in-sample
metrics, and C/B distributions for ABC match evidence. They do not evaluate a
query or infer biochemical functions. No signature or threshold is refitted.

Added integer-based trace consistency checks and target-only relative summaries
with checksummed read-only memory maps and bounded sorted/finite validation.
Intentionally absent companions, unsupported signatures and undefined metrics
are distinguished from malformed or incompatible artifacts. Reports detach their
evidence before rendering and coordinate with publication guards.

Development checks exposed two test assumptions: publication readers correctly
report the existing `Setup busy` error, and all-equal scores select predict-none
under the highest-cutoff tie rule (precision is undefined, not 50%). Corrected
these expectations. A network blocker in an initial installed-input test broke
SSL class initialization; changed it to block connection attempts instead.

Manually verified real cluster_0001 from the full 3443-target selected local
library: 20 C windows, 213241 B windows, 15 candidates, removals 567 then 279,
13 final dimensions, AUROC 0.9966748889753847, cutoff 0.549914086149925,
TP=20/FP=5656/TN=207585/FN=0. Output: `/tmp/pclass-dark-cluster0001.md`.
This read-only CPU check used saved artifacts and no embedding or fitting run.

Built/installed an isolated wheel with a tiny synthetic library only; default
Python and CLI use passed from `/tmp` with no setup-data artifacts, no heavy PLM
imports/network access and no package writes. Packaging/install logs and smoke
script are under `/tmp/pclass-dark-explain-*`. README and SPEC 0005 document the
new interfaces. Full regression and final formatting checks follow.

Validation completed: **167 full-suite tests passed, 2 opt-in tests skipped**
in 322.73 s, with the pre-existing Torch `AMD_SERIALIZE_KERNEL` environment
warning. A final focused run on the latest display/error changes and added
argument/symlink coverage passed **49 tests** in 51.55 s. Ruff lint and format
checks passed for 40 Python files; scoped whitespace and documentation
link/fence checks passed. Baseline revision: `7750d424`, branch `issue#11`;
changes remain uncommitted. Existing user data changes were preserved. This
read/format feature uses CPU only; no GPU behavior was added or required.

### 2026-10-28 — Target explain-signature validation to the requested cluster

Replaced the explanation's full-library audit with explicit target-scoped loading.
`load_library(..., cluster_id=...)` and `load_selected_library(..., cluster_id=...)`
verify target signatures/traces/profile rows and C/B arrays using existing
scientific validators. Selected manifest identity/digests, production provenance,
full identity target inventory and target setup bindings remain checked. The
shared compressed profile table is checksummed and scanned only as far as needed;
redundant shared metrics/relative summaries and unrelated cluster files are not
opened. Existing loader defaults still perform full validation. Reports record
`source.validation_scope="cluster"`. No artifacts, selection algorithms or scores
are changed, and no validation cache or alternate data format is introduced.

Initial focused regression: **49 passed** in 41.69 s. Added tests for target/full
evidence parity, instrumented file reads, unrelated missing artifacts, full-audit
failure on that same corruption, and requested dependency corruption. README and
SPEC 0005/0008 now describe the explicit scope and remaining shared-table cost.

Read-only local benchmark: cluster_0001 assembled in **0.250 s**, and the complete
CLI including Python startup took **0.443 s**, with existing filesystem cache.
The selected 3443-target library and the signature's 13 dimensions, saved AUROC
0.9966748889753847 and cutoff 0.549914086149925 were unchanged. Artifacts:
`/tmp/pclass-dark-cluster0001-targeted.md` and
`/tmp/pclass-dark-targeted-benchmark.txt`. No full-loader timing comparison or
cold-cache speedup is claimed. Final regression results follow.

The first last-cluster check (`cluster_3524`) took 14.324 s because report assembly
repeated the shared profile lookup after validation. Scoped library loads now
retain their immutable 640-profile `target_result` for immediate reuse, avoiding
that second pass. This is request-local parsed evidence, not a persistent
validation cache; full loads keep the field null. Added a file-open count check
and snapshot immutability/evidence-parity assertions.

Final targeted validation: **180 full-suite tests passed, 2 opt-in tests skipped**
in 348.13 s (the existing Torch environment warning). After request-local parsed
row reuse, **57 focused tests passed** in 73.63 s. Rebuilt/installed the final
wheel and verified default bundled Python/CLI reports from `/tmp`, with heavy PLM
imports/network/package writes prohibited and no setup-data artifacts. Ruff lint,
formatting and scoped whitespace/documentation checks passed.

Final warm-cache timings: cluster_0001 CLI including startup **0.415 s**;
last target cluster_3524 API assembly **7.244 s**, down from the measured 14.324 s
with duplicate sequential lookups. Remaining late-target cost is shared compressed
profile lookup; a future index/per-cluster storage change would be needed to
remove that cost. No such format change was made here. User data and scientific
results were preserved; no commit or push was made.

### 2026-10-28 — Draft signature summary listing specification

Drafted [SPEC 0009](../specs/0009-list-signature-summaries.md) for
`explain-signature --list`, preserving the single-cluster mode and requiring
exactly one mode. Default columns are Cluster, Status, C/B windows, Dimensions,
AUROC, Recall %, Specificity % and Precision %; FP is excluded. Specifies
alphabetic ordering, aligned full-width monospace output, fixed three decimals,
integer counts and unavailable em dashes. JSON retains full precision. Defines
the immutable Python API, summary-only artifact validation, resource limits,
error handling and tests. Listing deliberately avoids profiles, traces and C/B
score arrays to retain practical startup time. Implementation remains planned;
no code or scientific artifacts changed.

Draft validation passed: local links, Markdown fence balance, trailing whitespace
and scoped diff checks. No runtime tests were required for this documentation-only
change.

### 2026-10-28 — Review signature listing implementation readiness

Reviewed [SPEC 0009](../specs/0009-list-signature-summaries.md) against current
CLI callbacks, signature/manifest schemas and exact metric calculations. Resolved
the optional-cluster mode dispatch, explicit Rich cell-width dependency, signed
zero formatting, raw status versus displayed reason, metric/null reason rules,
AUROC/confusion-count consistency and unsupported statuses. Clarified that each
signature is parsed once while existing checksum helpers may read it separately.
The listing still avoids profiles, traces and all relative-score data. No
scientific decisions remain open; marked ready for implementation and updated
the index. No runtime code changed.

Readiness checks passed: local links, Markdown fence balance, trailing whitespace
and scoped diff checks. Documentation-only revision; no code tests were required.

### 2026-10-28 — Implement aligned signature summary listings

Implemented [SPEC 0009](../specs/0009-list-signature-summaries.md) through
`explain-signature --list` and importable listing/rendering APIs. CLI requires one
of `--cluster` or `--list`; listing rejects `--detailed`, and single-cluster output
is preserved. Alphabetic tables use the approved nine columns without FP, three
fixed decimals for AUROC/percentages, integer counts, em dashes for unavailable
metrics, display-cell alignment and escaped controls. JSON retains full precision.
Rich is declared explicitly. Immutable listing and explanation types share their
implementation base while remaining distinct public types.

The summary loader verifies selected production manifests and each final JSON,
including identity/population/representation bindings and exact displayed metric
consistency. It does not read profiles, traces, relative arrays or research files.
Tests remove those artifacts and instrument file reads. Unsupported statuses,
predict-none/undefined precision, corrupt inputs, empty inventories, Unicode
alignment and invalid CLI modes have dedicated coverage.

Initial listing tests: **21 passed**. Expanded summary/explanation/CLI checks:
**63 passed** in 135.40 s, plus a new immutable-validator-input test passed
separately. Full regression results follow. A read-only local listing covered all
3443 clusters in **2.369 s** with cached files; source identity/timing and output
are saved under `/tmp/pclass-dark-signatures-list*`. No scientific fitting or
artifact regeneration occurred.

Built and installed an isolated wheel with tiny bundled library data; default
API/CLI listing succeeded from `/tmp`, with heavy imports, network and package
writes blocked in the API smoke check. Automated read-only bundled tests omit
setup-data, profiles/traces and relative outputs. README, CLI help, SPEC 0008/0009
and the index are synchronized. User changes were preserved; no commit or push.

Final validation: **212 full-suite tests passed, 2 opt-in tests skipped** in
406.73 s, with the pre-existing Torch environment warning. Ruff lint/formatting
passed for 43 Python files, as did documentation link/fence and scoped diff
checks. Baseline revision: `1d9b3963`, branch `issue#14`; changes are uncommitted.
Executed `pytest -q`; earlier focused run covered `tests/test_signature_summaries.py`,
`tests/test_explanations.py` and `tests/test_cli.py`. CPU-only listing behavior
requires no GPU validation. No scientific data or model parameters were changed.

### 2026-10-28 — Use whitespace-free signature listing headers

Changed listing headers to `C_windows`, `B_windows`, `Recall%`, `Specificity%`
and `Precision%`; the other headers already contain one token. Alignment and
three-decimal formatting remain unchanged. Updated README and SPEC 0009, and
added an independent whitespace-splitting regression for the exact nine headers
and a normal data row. Human-readable title/footer and raw JSON fields are
unchanged. No signature data or scientific calculations changed.

Validation: **39 listing/CLI tests passed** in 36.78 s. Ruff lint/formatting,
scoped whitespace and documentation checks passed. No full-suite rerun was needed
for this display-label-only change.

### 2026-10-28 — Hide redundant all-ok Status column

Text and Markdown listings now omit Status when the listing is nonempty and all
signatures have status `ok`. Mixed/unsupported and empty listings retain the
column. Numeric columns remain right-aligned after its removal; JSON/API rows
always retain their status/reason. Metrics, precision formatting and column order
are unchanged. Updated README/SPEC 0009, the whitespace-splitting test and explicit
all-ok/mixed/empty/Markdown-alignment coverage.

Validation: **40 listing/CLI tests passed** in 36.91 s; lint, formatting,
documentation and scoped whitespace checks passed. Display-only change; no full
regression rerun or scientific artifact changes were needed.

### 2026-10-29 — Add profunc-derived Description column

Inspected the implementation behind `profunc browse cluster`: it calls
`profunc.go.description.describe_go_function` on `composed_go`, using the CLI's
modifier setting. Added an offline maintenance exporter that imports that module
from the neighboring checkout, checks the snapshot's membership/metadata/graph
hashes and setup artifacts, and removes the exact annotation prefix/article.
The catalog stores 3524 source entries and 3443 retained-membership bindings.
It is outside the identity-bearing reference directory, so no embeddings or
signatures need rebuilding. No runtime profunc dependency was introduced.

Runtime labels require matching C-window and row-map digests; incompatible/custom
references receive no guessed label. Added Description immediately after Cluster,
left-aligned and capped at 24 characters/display cells including literal `...`.
JSON retains full text and an unavailable reason. Six current clusters have empty
composed GO sets and therefore no resolvable description: cluster_0064, _0129,
_0492, _0678, _2185 and _2884. They display `—`.

Verified all 3524 catalog entries against the imported profunc describer and the
first entry against the command's formatter. cluster_0001 is `NADP+ nucleosidase
with NAD+ nucleosidase activity`, displayed as `NADP+ nucleosidase wi...`.
Saved parity evidence in `/tmp/pclass-dark-profunc-description-parity.txt` and
real output in `/tmp/pclass-dark-list-descriptions.txt`. Initial description,
listing and CLI tests: **52 passed** in 37.49 s. Full regression results follow.

The initial all-described completeness assertion found six empty composed-GO
sets, rather than mismatched reference bindings; these are preserved as unknown
function descriptions instead of assigning fallback biological labels. Every
source description (including unresolved entries) matched the imported module.

Final verification: **227 full-suite tests passed, 2 opt-in tests skipped** in
380.51 s, with the existing Torch environment warning. Ruff lint/format checks
passed for 46 Python files; documentation links/fences and scoped whitespace
checks passed. Built both wheel and sdist with the catalog included, and verified
installed CLI/API listing from `/tmp` without profunc/heavy PLM imports, network
or package writes. The source archive also includes the exporter. Baseline code
revision: `1d9b3963`, branch `issue#14`; changes remain uncommitted. No embeddings,
signatures, authoritative reference files or model parameters were changed.

### 2026-10-29 — Append 20-bin signature metric distributions

Added a distribution section after `explain-signature --list` with 20 fixed bins
for AUROC, recall, specificity and precision. It counts clusters equally using
unrounded saved fractions, excludes null metrics with explicit availability
counts, uses left-inclusive/right-exclusive internal bins and includes 1 in the
last bin. Text/Markdown show an aligned combined table with fraction/percentage
intervals; JSON and immutable Python listings preserve edges and exact counts.
The listing constructor maintains the derived distributions consistently without
additional artifact reads or fitting. Existing cluster-table formatting remains.

Focused histogram/listing/description/CLI tests: **65 passed** in 41.91 s,
including analytical edges/nextafter cases, zero/one, unavailable/empty data,
rounding-sensitive values, unweighted counts, 20-row output and JSON agreement.
Real-library output is saved at `/tmp/pclass-dark-list-metric-distributions.txt`;
all per-metric counts/availability totals were checked against 3443 cluster rows.

Real snapshot counts: each of AUROC, recall, specificity and precision has 3443
valid values and zero unavailable entries; every 20-bin count sum matches.
Installed-wheel text/Markdown/JSON reports were verified from `/tmp`. Lint and
formatting passed for 48 Python files; documentation and scoped whitespace checks
passed. This isolated descriptive aggregation/rendering change did not require
rerunning the embedding/training suites. No scientific artifacts were modified.

## 2026-10-29 — Readable published cluster paths

Implemented [SPEC-0011](../specs/0011-readable-cluster-paths.md): production,
relative-score and research bundles now write `clusters/<cluster_id>/` directories.
Validated portable IDs reject unsafe paths/device names and case-fold collisions.
An explicit manifest layout marker preserves legacy hexadecimal reads, including
scoped explanation/listing and legacy research import. Private checkpoint/cache
names remain stable for resume. Scientific fitting and pooling are unchanged.

Added `pclass_dark.library_paths.migrate_library_paths()` and
`scripts/migrate_library_paths.py`. Migration holds the transaction guard, fully
validates old and staged bundles, hard-links immutable payloads, rewrites metadata
atomically, retains the original generation, commits selection last and recovers
from a durable journal. Production scientific identity stays unchanged; relative
identity is refreshed to bind the updated production manifest. An incompatible
research selector is archived, with research artifacts preserved unselected.

Migrated the current selected library using
`python scripts/migrate_library_paths.py data/library`, without fitting/scoring.
Production remains `5f9117a77f64b30426661dff87533a462a8c0771c7150ae76eb903574a5e8100`;
new relative ID is `cab57dc6d25c492f233f59197d2cc62c4bf2c05cdc89d19fbe3580863ba3e5b0`.
Both have 3443 readable cluster directories. The complete original pair and exact
selector are under
`data/library/.archives/cluster-paths-693395768393481da3f98b38a140a6fa/`.
Migration stdout: `/tmp/pclass-dark-current-library-migration.json`.

Initial path refactor tests: 63 passed. Migration regression tests: 13 passed,
including legacy full/scoped reads, byte-identical signatures/arrays, identical
query scores/ABC evidence, listing, release inventory, repeat no-op, exception
rollback and recovery after subprocess termination. Full-suite and real-library
postpublication verification results follow below.

Postpublication verification: all 6886 production payloads and 10329 relative
payloads match the original manifests byte-for-byte by SHA-256. Production and
relative selectors/manifests match, with production scientific ID preserved.
`pclass-dark explain-signature --cluster cluster_0001` and `--list` succeeded;
reports are `/tmp/pclass-dark-readable-cluster-0001.txt` and
`/tmp/pclass-dark-readable-cluster-list.txt`. Release `shared_files()` inventory
validated 17238 selected files and excluded archives. Extended migration tests
also verify normal build reuse without fitting: 13 passed in 24.15 s. Ruff and
format checks passed for 51 Python files; scoped whitespace checks passed.

Full suite: **252 passed, 2 skipped** in 451.58 s. The skipped tests are
opt-in real-checkpoint inference tests; this path/storage change does not alter
inference. One existing PyTorch environment warning concerned the value of
`AMD_SERIALIZE_KERNEL`. The latest extended migration tests were rerun separately
(13 passed), including release inventory and postmigration build reuse. No
scientific values changed, no refitting occurred, and nothing was committed/pushed.
