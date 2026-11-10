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

### 2026-10-29 — Compact signature listing presentation

Updated [SPEC-0009](../specs/0009-list-signature-summaries.md) and the text/Markdown
`explain-signature --list` renderer: `Dimensions` is now `Dims`, retaining numeric
right alignment. Each metric's total binned count appears in the distribution
heading; the trailing valid/unavailable table and binning explanation are removed.
Missing counts remain in JSON/API; scientific calculations are unchanged.

Validation: `python -m pytest -q tests/test_signature_summaries.py
 tests/test_summary_distributions.py tests/test_cli.py` — **56 passed** in 43.12 s.
Regression coverage checks header alignment, heading totals with missing/empty
metrics, 20 bins, unchanged JSON/API and absence of trailing explanatory content.
Scoped Ruff lint/format and `git diff --check` passed. No fitting or experiments
were required.

### 2026-10-29 — Replace copied profunc inputs with a runtime dependency

Implemented [SPEC-0012](../specs/0012-profunc-dependency.md), superseding the
runtime behavior of SPEC-0002 and the static catalog in SPEC-0010. Explicit user
request authorizes a declared profunc dependency; neighboring checkouts are no
longer silently consulted through hard-coded paths. Preparation calls installed
profunc's runtime resolver, reads its current memberships, and verifies source
FASTA hashes against that dataset's manifest. The runtime excludes FASTA, so
preparation uses the source generation through profunc's workspace module or
an explicit `--reference-fasta` / Python `reference_fasta` override. No sequences
are downloaded. Custom manifest-bound `reference_dir` integrations remain usable.

New fit inputs carry membership/FASTA hashes; descriptions call profunc's actual
GO describer on the active validated runtime. Stale references and older fits
without the binding show an unavailable reason. Retired the static catalog and
export script; historical reference files remain in Git for provenance and are
excluded from distributions. Existing scientific algorithms, worker/device
selection, and training/evaluation procedures are unchanged.

Validation so far: 23 focused reference/description/CLI tests passed; the new
signature-binding and CLI-forwarding checks passed (2 tests); the retired-resource
build cleanup regression passed. Isolated wheel and sdist builds under
`/tmp/pclass-dark-profunc-package-check` passed, declared `profunc>=0.1.0,<0.2`,
and excluded historical references even with simulated stale build files.
Build log: `/tmp/pclass-dark-profunc-build.log`; artifacts:
`/tmp/pclass-dark-profunc-dist`. Active profunc packaged publication
`28f87149af01ed624c2814ca93ce1111ed6c915ad8aac4545ed1af48c00f0ee0`
resolved successfully: 3524 clusters, 455508 members, complete sequence coverage;
membership SHA-256 `b616109726acb0e0403060725c097c19bb245062aba62beabf4fcc6d3db8f1de`.
This is input validation, not a held-out performance experiment. No PLM inference
or full signature rebuild was run. Full pytest validation remains in progress;
final outcomes will be appended below. No commit or push was performed.

Full-suite validation caught a missing `id` on the new operation-local description
mapping (3 listing/migration failures after 117 passing tests). Added a stable
content-derived ID to preserve the listing's provenance field. The production
suite separately passed all 32 tests, including scientific legacy imports and
serial/parallel checks. An expanded focused suite passed 54 tests. Full-suite
and real-library listing checks were restarted after the catalog-ID fix.

Final validation: **261 passed, 2 skipped, 1 warning in 524.15 seconds** with
`python -m pytest -q --disable-warnings --maxfail=3`; output saved to
`/tmp/pclass-dark-profunc-final-pytest.log`. The two pinned real-checkpoint tests
were not enabled (`PCLASS_DARK_TEST_CACHED_ESM` unset). All 34 listing tests passed
separately. `python -m ruff check src tests build_support.py` and
`git diff --check` passed. Final isolated wheel and sdist checks passed.
`python -m pclass_dark explain-signature --list --format json` succeeded against
the existing 3443-signature library; all older signatures correctly reported
`missing_profunc_reference_binding`, retaining their metrics. JSON artifact:
`/tmp/pclass-dark-profunc-list.json`. Users should rerun setup and build after
updating profunc to obtain signatures with current reference bindings. Historical
failed/superseded validation runs above are retained. SPEC-0012 and its index
are synchronized with the implemented behavior and final test results.

### 2026-10-29 — Default setup/build CLI workers to eight

Changed `setup-data` and `build-library` CLI `--workers` defaults from 1 to 8.
Explicit `--workers 1` preserves serial mode; Python API defaults and resource
admission limits are unchanged. Updated README and SPEC-0003/SPEC-0004 interface
notes. CLI regression tests check both default forwarding and explicit serial
overrides for both commands. Validation: `python -m pytest tests/test_cli.py -q`
passed (11 tests); targeted Ruff checks passed. Full-tree Git inspection/diff
checks hit a read-only `.git/lfs/tmp` error on `data/setup-data/selection.json`;
checks scoped to source/docs/tests avoid the LFS filter and pass. Existing
unrelated data and prior changes were preserved; no commit or push performed.


### 2026-10-29 — Remove retired copied profunc artifacts

At the user's explicit cleanup request, removed all seven files in
`src/pclass_dark/resources/reference/`: membership TSV, source FASTA, local
manifest, and four source manifests/statistics files. Removed the obsolete
reference-tree Git LFS rule. The static catalog/exporter were already retired
in the dependency change. Removed the obsolete snapshot-copying/release procedure
from current reference documentation, updated README and SPEC-0012, and clearly
marked SPEC-0002/SPEC-0010 as retired historical records. Earlier findings remain
in this log and Git history. Selected and archived prepared data, embeddings,
and libraries were preserved; these remain valid reproducible outputs.

Replaced the removed snapshot's integrity test with an independent three-record
custom-reference fixture verifying membership, sequence coverage, residue counts
and noncanonical residues. Kept packaging exclusions/stale-build cleanup guards
to prevent old build directories from republishing retired resources. Validation:
`python -m pytest tests/test_reference_data.py tests/test_setup_data.py
 tests/test_cluster_descriptions.py tests/test_cli.py tests/test_shared_distribution.py
 -q --disable-warnings` passed **58 tests in 4.51 seconds**. Ruff and scoped
`git diff --check` passed. Live profunc resolution and a cluster_0001 description
lookup succeeded with the retired tree absent. No scientific algorithm changed;
no PLM inference, signature rebuild, commit or push was performed.


### 2026-10-30 — Narrow the planned infer report to accepted matches

Updated [SPEC-0007](../specs/0007-query-ranking-presentation.md) following the
user's reporting decision: include only A=Yes matches, union cluster matches
across one query's windows, and sort the combined result by unrounded B.
Documented a representative-window rule: lowest B among accepted windows, with C
from that same window and deterministic coordinate/ID ties. Preserve supporting
window evidence in structured results. No likely/unlikely labels, hard B cutoffs,
gap detection, combined probabilities or interpretation categories are planned.
The compact output is Cluster/A/B/C. This is a reporting specification, not a
new calibrated protein-level classifier. Query interfaces/resource contracts and
implementation remain planned. No runtime code changed. Validation: checked
local specification links, synchronized index and whitespace; no code tests
required for this documentation-only refinement.


### 2026-10-30 — Specify infer input, device defaults and stdout reports

Expanded [SPEC-0007](../specs/0007-query-ranking-presentation.md), the planned
implementation specification for `pclass-dark infer`. Defined mutually exclusive
`--seq` (one sequence string) and `--fasta` (one or more FASTA records) inputs.
Batch queries retain independent window-union match results and input order.
Specified GPU-first embedding with CPU fallback, an explicit CPU option, and a
default CPU budget of eight through `--workers`, with bounded CPU use and no
per-worker GPU model replication. Both input modes print reports to stdout;
users redirect stdout to files, while diagnostics/progress use stderr. No
automatic report file or output-path option is planned. Added validation
expectations for input modes, batch separation, devices, worker defaults and
redirection. Updated the spec index. This is documentation-only planning;
no infer command or runtime behavior was implemented. Validation: local links
and scoped whitespace/diff checks passed; no code tests required.


### 2026-10-30 — Make infer specification implementation-ready

Revised [SPEC-0007](../specs/0007-query-ranking-presentation.md) against the actual
windowing, ESM backend, representation/scoring, empirical-evidence/cache, artifact
paths and root-guard implementations on `issue#23`. Fixed the earlier draft's
unspecified CLI/Python signatures, record/JSON contracts, batch/resource controls,
CPU thread semantics, initialization-only automatic GPU fallback, failure/output
atomicity, per-query IDs/window keys and context/iterator lifetime. Retained all
previously approved scientific rules: literal reference residue policy and
terminal windows, existing A/B/C math, accepted-window union, minimum-B accepted
representative with paired C and coordinate ties, and no interpretive categories.
No new scientific decision was required or silently made. The spec explicitly
requires stopping for user decisions before changes to those scientific rules.

An IO audit replaced model-batch-first classification, which would repeatedly
reload the large C/B libraries, with independent model/score batch sizes,
cluster-first scoring across private temporary vectors, one pinned lazy reference
view per cluster, and bounded SQLite accepted-window working storage. This changes
execution order only, not scoring math/ranking. Specified temporary quotas,
primary-key scan requirements, reusable pure reduction, model release before
completed-result iteration and report spooling before stdout publication.

The current branch does not contain the prior cleanup specification/helper/error;
removed a broken SPEC-0013 dependency and specified minimal read-only pending-state
and cache-hit lifetime checks against current artifacts code, with reuse of cleanup
helpers if merged later. Added planned RelativeScores metadata and scoped evaluator
APIs rather than depending on private cache fields or loading quantiles. Corrected
iterator exhaustion to StopIteration and defined failed-entry resource cleanup.
Updated specification status and index to ready-to-implement, not implemented.
Validation: local links, code fences, formulas/contract consistency and scoped
whitespace checks; no runtime code, inference, scientific experiment or code-test
results are claimed for this documentation-only revision.


### 2026-10-30 — Implement infer query classification and reporting

Implemented [SPEC-0007](../specs/0007-query-ranking-presentation.md) on `issue#23`:
`pclass-dark infer --seq/--fasta`, stdout text/JSON reports, GPU-first embedding
with initialization-only CPU fallback, explicit CPU/device options, default eight
CPU threads and independently bounded model/scoring batches. Added reusable
`pclass_dark.inference` APIs, frozen records and one shared accepted-window reducer.
Terminal window coverage, fitted A acceptance, inclusive empirical B/C, minimum-B
accepted representative with paired C and coordinate ties remain unchanged. No new
scientific decision, threshold, probability calibration or functional assignment
was introduced.

Added bounded input preflight and stable FASTA checks; private JSONL, memmap and
SQLite staging with workspace/disk admission; cluster-first scoring with one lazy
pinned reference view per target; source-generation/maintenance checks including
cache hits; scoped thread/precision restoration; failed-entry cleanup; and report
spooling before stdout publication. Model/guards are released before completed
results are iterated. No project reference/prepared data, embeddings or library
artifacts were rebuilt or removed. README, CLI help, optional-dependency notes,
spec status and index are synchronized.

Validation:

- Full regression: `pytest -q` — **290 passed, 2 skipped**, 461.20 seconds. The
  two default skips are opt-in real-checkpoint tests. This run collected before
  the final additional edge-case tests and input/error hardening.
- Final focused run: `pytest -q tests/test_inference.py
  tests/test_relative_scores.py tests/test_reference_data.py` — **56 passed**,
  including the final changes and additional cases. Covers independent coordinate,
  hash and empirical-score examples; representative pairing/native sorting;
  batch/worker consistency; cache lifetime; malformed inputs/model matrices;
  quotas/indexed retrieval; cleanup; device failures and thread restoration;
  CLI atomic output and help without model initialization.
- `PCLASS_DARK_TEST_CACHED_ESM=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 pytest -q
  tests/test_pinned_esm.py` — **1 passed, 1 skipped**, 9.90 seconds, offline using
  the existing pinned checkpoint. CPU pooling/batching passed; no compatible GPU
  was available for CUDA comparison. GPU routing/fallback is covered by mocks.
- `ruff check src tests`, changed-file formatting checks, local documentation
  links and scoped `git diff --check` passed. Torch emitted an environment warning
  about `AMD_SERIALIZE_KERNEL=3`; it did not fail tests.

These are implementation checks with synthetic fixtures and cached-model pooling,
not held-out biological performance estimates. GPU numerical agreement remains
unverified on this host. No commits or pushes were made.

### 2026-10-30 — Draft dependency-aware build-library cleanup

Drafted [SPEC-0013](../specs/0013-build-library-clean.md) at the user's request.
`build-library --clean` would retire proven obsolete prepared datasets and their
exclusive completed libraries, C/B distributions, embeddings and checkpoint/import
caches after validated full publication/reuse. Shared embeddings/fits, selected
pipelines, current-preparation experiments, partial resumable work and explicit
archives are protected. The draft distinguishes historical provenance IDs from
runtime-required dependencies so a shared embedding's original prepared ID does
not unnecessarily retain the old dataset. Includes read-only `--clean --dry-run`,
full-scope gates, both-root locking, per-schema dependency adapters, unknown-file
protection, durable recoverable quarantine, failure behavior, storage accounting
and tests. Worktree cleanup does not prune Git LFS storage or change Git history.
Inspected current layout: two preparations, two production generations and two
relative-score generations share one selected embedding bundle; no IDs are
hard-coded in the design. No cleanup code was implemented and no data was deleted.
Updated spec index; local links and scoped diff/whitespace checks passed.


### 2026-10-30 — Resolve cleanup specification implementation details

Revised [SPEC-0013](../specs/0013-build-library-clean.md) through code/schema and
internal-consistency audits and marked it ready to implement (not implemented).
Confirmed v2 production/fit identities lack prepared IDs; the two current C/B
companion manifests provide verified target-level ownership for their respective
preparations and share the same embedding ID. Specified complete ownership proof,
retention fixed points, exact full-selection gates, old/new fit-format handling,
provenance-only edges and conservative protection of legacy/migration/unknown
artifacts. No experimental performance or deletion results are claimed.

Resolved CLI/API and frozen plan/result/error contracts; clean runs after a
successful full build with an expected snapshot, preview does not build, and
failed requested research prevents cleanup. Corrected the earlier draft's lock
assumption: transaction guards are same-thread reentrant. Identified necessary
runtime and packaging pending-journal checks, lazy evidence cache-hit checks, and
pure checkpoint validators (existing read helpers may create/quarantine files).
Specified coordinator/peer identities, ordered root-local quarantine, durable
commit, rollback before commit, resumable reclamation after commit, marker-only
crash cases, hard-link recovery and resource caps. Documented implementation
sequence and fault-injection/regression acceptance matrix. Kept explicit archives,
shared inputs, current experiments and resumable work protected.

Validation: inspected actual writer/reader/lock code and current ownership
manifests without modifying artifacts; verified local spec links, index status,
required contracts and scoped Git whitespace checks. This revision changed only
documentation; no runtime tests were required or claimed, no cleanup code was
implemented and no data was deleted. Earlier draft decisions remain recorded in
this log and are superseded by the ready specification.


### 2026-10-30 — Implement opt-in build-library cleanup

Implemented [SPEC-0013](../specs/0013-build-library-clean.md): public immutable
plan/result records, pure dependency-aware inventory, protected/shared/resumable
lineages, full-build/preview gates, nonfollowing POSIX deletion, durable two-root
quarantine/journal recovery, build snapshot handoff, `--clean`/`--dry-run`, and
importable planning/execution/recovery APIs. Added pending-journal checks to
runtime guards, lazy relative cache hits and packaging. Added README/data docs;
normal builds still retain generations. No actual project dataset was cleaned.

Focused fixtures currently pass 40 cleanup tests. Failure-driven fixes included
recognizing persisted selection decision/trial receipts and their versioned hashes,
retaining missing private completion receipts as resumable work, and excluding
compressed selection traces from archived selector parsing. Tests exercise old
preparation/exclusive-output removal with shared embeddings retained, forced
archives, partial work, forged/stale plans, extra files/symlinks, guard contention,
clean CLI preview, selected scoring/explanation survival and isolated process death
before/after durable journal stages, moves and unlinks. One preliminary full-suite
run was stopped while the archive-parser regression was corrected. Final full
regression, additional edge cases and final status remain in progress. This is
artifact management, not a scientific performance experiment; no PLM inference
was run or performance claims made.


### 2026-10-30 — Complete cleanup acceptance and regression validation

Completed [SPEC-0013](../specs/0013-build-library-clean.md) implementation and
synchronized its status, index, README and data instructions. Full regression
passed **322 tests, 2 skipped, 1 warning in 556.81 seconds** with
`python -m pytest -q --disable-warnings --maxfail=3`; log:
`/tmp/pclass-dark-cleanup-regression.log`. Added four final cleanup edge cases
after full-suite collection, then ran all cleanup plus analytical relative-score
tests: **74 passed in 117.29 seconds** (61 cleanup and 13 relative tests), log:
`/tmp/pclass-dark-cleanup-acceptance.log`. This final run covers unsupported
complete targets, research failure preventing cleanup, LFS-store preservation,
and recovery after unlinking one of two candidate hard-link aliases. Prior
cleanup acceptance also passed 57 cases including full research and ten isolated
process-death/fsync boundaries. Earlier full-suite failure was a toy relative
fixture lacking generation metadata required by cache-hit validation; fixed the
fixture without changing its independent numerical score expectations.

Ruff and scoped whitespace checks passed. Isolated wheel/sdist builds succeeded;
archive checks found all cleanup modules and no private journal/quarantine state.
Installed-wheel build-library help exposes `--clean` and `--dry-run`. Packaging
artifacts: `/tmp/pclass-dark-cleanup-dist`; log:
`/tmp/pclass-dark-cleanup-package.log`. Two real pinned-checkpoint tests remained
opt-in and were skipped. No model inference, scientific performance experiment,
project cleanup, commit or push was performed. All deletion/recovery tests used
isolated temporary roots. Conservative legacy/ambiguous retention, POSIX platform
scope, archive protection and Git LFS limitations are documented.


### 2026-10-30 — Correct build-library cleanup semantics

User clarification supersedes the earlier build-then-clean contract:
`build-library --clean` validates the existing full selected production/C/B pair
and cleans obsolete generations without entering build orchestration. Missing or
incomplete selections fail without creating or repairing a library. Reject force
and research requests in clean mode; preserve cleanup recovery and stale-selection
checks. Log validation explicitly instead of build target/worker counts. Updated
[SPEC-0013](../specs/0013-build-library-clean.md), its index and README.

Validation: `python -m pytest tests/test_cli.py tests/test_production_library.py
 tests/test_data_cleanup.py -q` passed **92 tests** in 280.74 s; output at
`/tmp/pclass-dark-clean-validation.log`. Focused missing/incomplete/no-build
regressions passed (3 tests). Ruff and Git whitespace checks passed. Tests use
synthetic temporary datasets; no project-data cleanup was executed. Existing
uncommitted work on `issue#22` was preserved; no commit or push.


### 2026-10-30 — Disk-backed cleanup planning for large retained caches

The reported inventory-budget failure came from aggregate accounting of all
retained and removable trees, not evidence of an OS out-of-memory failure.
Read-only traversal of this checkout found 5,218,158 files / 13,773 directories
under `data/library/.fits`, and 13,772 files / 6,889 directories under
`.production-checkpoints`. No data was deleted. Existing one-million-entry
aggregate accounting cannot handle that inventory.

Updated [SPEC-0013](../specs/0013-build-library-clean.md): planning spools exact
node inventories to an automatically closed private temporary file, retaining
only offsets in the graph. Aggregate limits now cover removal inventories;
per-tree and journal caps remain. Plan hashes, ownership gates and recovery
checks retain their formats. CLI/README explain temporary disk use and limits.
Candidate-limit errors now identify entry counts and serialized bytes.

Validation: cleanup suite **66 passed** in 109.15 s
(`/tmp/pclass-dark-clean-spool-tests.log`); CLI suite **11 passed**
(`/tmp/pclass-dark-clean-spool-cli.log`). Added regressions for retained inventories
outside the aggregate removal budget, candidate-budget refusal, deterministic
plans and spool closure on success/failure. Ruff and whitespace checks passed.
Full production-data planning/deletion was not run; hashing millions of files
still costs IO/time and temporary storage. A candidate set exceeding removal or
64 MiB recovery-journal limits still fails before mutation. No commit or push.


### 2026-10-30 — Simplify build-library CLI options

Removed user-exposed data-dir, output-dir, memory-mib, epsilon (user spelling
“epsion”), sigma-floor and dry-run options from build-library. Internal values
remain None/None for default root resolution, 1024 MiB workspace and 1e-8 for
both numerical safeguards. Cleanup executes directly; preview and configurable
paths/settings remain available through Python APIs. Other subcommands retain
their options. Updated README, [SPEC-0004](../specs/0004-build-library.md),
[SPEC-0013](../specs/0013-build-library-clean.md) and specification index.

Validation: `python -m pytest tests/test_cli.py tests/test_data_cleanup.py -q`
passed **85 tests** in 108.85 s; log `/tmp/pclass-dark-build-cli-defaults.log`.
Tests verify removed options are absent from help and rejected before building,
internal defaults are forwarded, and clean uses default root resolution. Ruff
and Git whitespace checks passed. No project data was cleaned; no commit/push.


### 2026-10-30 — Retire rollback archives after replacement validation

User clarification supersedes earlier permanent archive protection: preserve a
valid fallback during replacement construction, but once a full selected
setup/production/C/B replacement validates, old generations and recognized
rollback/migration backups are obsolete. Archive references no longer seed the
retention graph. Shared embeddings and partial work retain existing protections;
unknown/unsafe history fails closed. `.archives` is an exact-inventory cleanup
unit under both managed roots and uses existing journal, quarantine and recovery.
Updated [SPEC-0013](../specs/0013-build-library-clean.md), index, README and CLI
help/result text. Earlier archive-retention conclusions above are superseded.

Validation: `python -m pytest tests/test_data_cleanup.py tests/test_cli.py -q`
passed **87 tests** in 123.06 s (`/tmp/pclass-dark-obsolete-archives.log`). Tests
cover old/current archive retirement, invalid replacement preservation, forced
build backups, shared embeddings, and archive-bearing rollback/recovery under
injected IO failures. Ruff and whitespace checks passed. No project-data deletion
or full real-data cleanup plan executed. Existing removal-inventory/journal caps
remain; large obsolete cache sets may still exceed them. No commit or push.


### 2026-10-30 — Scale execution beyond aggregate inventory limits

The user's retry reached 1,000,237 removal entries / 228,543,433 serialized bytes
and failed before deletion. Earlier disk-backed planning alone was insufficient:
it still accumulated removal inventories and embedded them in a single 64 MiB
journal. Supersedes earlier statements that these aggregate limits remain.

Updated [SPEC-0013](../specs/0013-build-library-clean.md): temporary SQLite stores
candidate inventories and inode counters; plans bind sorted exact tree hashes.
New v2 coordinator journals reference per-candidate bounded inventory sidecars,
all durable before detachment. Retain one all-candidate commit boundary, exact
inventory validation, inode/link checks, and v1 journal recovery. Prepared
transactions without peers cancel partial sidecar publication safely; done-state
recovery tolerates already removed sidecars. Quarantine validation uses disk-backed
hard-link counters instead of accumulating every observed file in RAM. Clean now
plans once under guards; public preview execution still independently replans.
Added execution-stage progress logs. README and specification index synchronized.

Validation: cleanup, CLI, shared-distribution and library-cache suites **106
passed** in 144.94 s (`/tmp/pclass-clean-scalable-final.log`). After final storage
closure/log/readability changes, focused scale/publication/single-scan/closure
checks **4 passed** (`/tmp/pclass-clean-scalable-final-focused.log`). Six new
sidecar/scale/recovery tests passed separately (`/tmp/pclass-clean-sidecar-new.log`).
An earlier run had two failures: the synthetic limit exceeded its total dataset
size, and done-state error reporting tried reading a reclaimed sidecar. Corrected
both; preserved failed-run logs at `/tmp/pclass-clean-scalable-tests.log` and
`/tmp/pclass-clean-sidecar-focused.log`. Ruff and Git whitespace checks passed.

Scaled regression lowers per-tree and per-document limits so aggregate removal
exceeds them while individual inventories fit, then asserts successful cleanup.
Coverage includes sidecar publication failure, corruption refusal, v1 recovery,
interrupted finalization, hard links and process-death recovery. No project-data
cleanup or full real-data planning executed. Temporary and recovery disk storage
scale with inventory size; individual-tree and coordinator/sidecar document caps
remain, but aggregate candidate file count/size no longer blocks this workflow.
No commit or push; unrelated uncommitted work preserved.


### 2026-10-30 — Direct removal of committed obsolete trees

User reported prolonged reclamation of 6,891 candidates and requested direct
removal without relinking. Inspection found a repeated quarantine hash pass,
per-file descriptor traversal/identity checks and fsync per unlink. No relinking
was occurring: detachment renames directories. The real-data coordinator was v2,
committed, with 6,891 entries; user-process data deletions in the working tree
were preserved. No cleanup was executed or process stopped by the agent.

Updated [SPEC-0013](../specs/0013-build-library-clean.md): committed trash is now
removed with descriptor-relative, symlink-resistant shutil.rmtree, checking its
root inode/device/mode and refusing recreated original paths. No postcommit
content rehash or hard-link recount; fsync once per candidate parent rather than
once per file. Precommit selected-data validation, inventory checks and rollback
remain. Sidecars/journal support interruption and old v1/v2 committed runs.
Progress every 100 candidates. Recovery completing an existing transaction returns
its result directly rather than scanning retained checkpoints for another plan.
Removed obsolete per-file unlink/rmdir helpers. README and index synchronized.

Validation: broader cleanup/CLI/distribution/cache run had **107 passed, 2 failed**
in 144.26 s (`/tmp/pclass-clean-direct-final.log`). Failures were obsolete crash
injections waiting for fsync calls 100/150, beyond the reduced operation count.
Changed those injection points to reachable calls 35/40; all **10 fsync crash
cases passed** (24.00 s; `/tmp/pclass-clean-direct-fsync.log`). Seven initial
focused recovery checks passed (`/tmp/pclass-clean-direct-focused.log`); three
direct-delete checks passed after removing unused helpers
(`/tmp/pclass-clean-direct-last.log`). Final recovery/single-scan/direct-delete
checks **5 passed** (18.45 s; `/tmp/pclass-clean-direct-resume.log`). Ruff and scoped
Git whitespace checks passed. The full broad suite was not rerun after test-only
injection updates and the specifically tested recovery short circuit.

The 200-file deletion test observes exactly one fsync and no hashing/relinking,
preserves an outside hard-link alias and symlink target, and root-replacement
checks refuse removal. Existing process-death and hard-link tests cover interrupted
direct removal. The existing running process keeps its old imported code: if it
is still running, user can Ctrl+C and rerun build-library --clean to resume its
committed journal through the updated path. Initial validation and metadata-heavy
filesystem deletion still take time; no production throughput benchmark claimed.
No commit or push.


### 2026-10-30 — Preserve unassembled partial embedding checkpoints

Resolved review P2: cleanup finish unconditionally indexed assembled embeddings
for embedding-checkpoint nodes even though shards() supports resumable jobs with
no assembled generation. Inherit owners only when an assembled embedding exists;
retain partial checkpoint owners, preparation dependencies and protection.
Updated [SPEC-0013](../specs/0013-build-library-clean.md).

Added a valid full-range first shard for a new embedding identity without assembly,
verified via the embedding receipt validator. Regression covers preview, execution
and build-library clean, unchanged shard hashes and retained owning preparations.
Cleanup suite **77 passed** (104.02 s; `/tmp/pclass-clean-partial-embedding.log`);
the final full-range fixture also passed independently (1 test, 13.29 s;
`/tmp/pclass-clean-partial-embedding-focused.log`). Ruff and scoped Git whitespace
checks passed. No project-data cleanup, commit or push.


### 2026-10-30 — Make rollback finalization restartable

Resolved review P2: after restoring precommit candidates, recovery deleted sidecars
and peer while the coordinator still said prepared/detaching. An interruption
then made subsequent validation require metadata already removed. Recovery now
persists terminal rolled_back before any metadata removal, including cancellation
of prepared transactions without peers. Validation accepts terminal rollback with
missing sidecars/peer; recovery resumes only marker finalization and reports zero
removed bytes/artifacts. Root/coordinator identity and allowed-path checks remain.
Updated [SPEC-0013](../specs/0013-build-library-clean.md).

Validation: eight fault cases passed (16.21 s;
`/tmp/pclass-clean-rollback-focused.log`) across prepared/no-peer, prepared/peer
and detaching paths, interrupted after terminal publication, sidecar removal and
peer removal. Tests verify exact restored inventories, unchanged selections,
zero removals, cleared markers, idempotent recovery and unblocked readers/writers.
Full cleanup/CLI suites **104 passed** (107.61 s;
`/tmp/pclass-clean-rollback-final.log`). Ruff and scoped Git whitespace checks
passed. No project-data cleanup, commit or push.


### 2026-10-30 — Validate archive descendants and reject unpaired legacy clean

Resolved review P1: archive retirement previously checked only immediate unknown
files. Recursive coverage now requires every archived file to be a recognized
metadata record, a verified manifest/receipt member, or a validated completed-fit
cache record. Directory paths must be ancestors of covered files; unknown empty
directories are rejected. Fail before clearing archive protection or deletion.
An approved preview is revalidated and refuses descendants added afterward.
Resolved P2: clean checks pair.relative before constructing LibraryResult and
returns an actionable full-build-without-clean error for valid v1 selections.
Updated [SPEC-0013](../specs/0013-build-library-clean.md).

Validation: initial focused new cases **5 passed** (40.47 s;
`/tmp/pclass-clean-review-focused.log`). The first broader run had **108 passed,
1 failed** (`/tmp/pclass-clean-review-final.log`): legitimate forced-build profile
and decision receipts needed explicit coverage. Added the existing fit validator,
then archive/forced-build cases **5 passed** (19.72 s;
`/tmp/pclass-clean-archive-coverage.log`). Final cleanup/CLI suites **109 passed**
(144.96 s; `/tmp/pclass-clean-review-verified.log`). Tests preserve exact archive
inventories on rejection, selected valid libraries after retirement, and real
legacy selections on API/CLI refusal without building. Ruff and scoped Git
whitespace checks passed. No project-data cleanup, commit or push.


### 2026-10-30 — Resolve main integration on issue#23

Integrated main `bf4f6704` into issue#23 `c2483cdc`, resolving conflicts in
artifact guards, relative evidence and this chronological log. Preserved both
branches' records, main's recoverable cleanup guard and private transaction bypass,
and inference's scoped evaluator/manifest identity checks. Updated
[SPEC-0007](../specs/0007-query-ranking-presentation.md) to record integration with
[SPEC-0013](../specs/0013-build-library-clean.md). Updated the inference regression
to require the shared recovery exception, covering both marker types and immutable
mode. No scientific calculation changed.

Validation: `python -m pytest -q tests/test_inference.py
tests/test_relative_scores.py tests/test_data_cleanup.py tests/test_cli.py`: **162
passed** in 142.56 seconds. After expanding the marker regression,
`python -m pytest -q tests/test_inference.py -k
relative_scope_closed_and_pending_markers`: **4 passed**, 39 deselected, in 5.69
seconds. `python -m ruff check src tests`, formatting checks for the resolved
Python files and regression test, and staged whitespace checks passed. These are
engineering checks; no biological experiment or GPU evaluation was performed.
All conflicts resolved; merge staged for review, without commit or push.


### 2026-10-31 — Draft external sequence collection specification

Drafted [SPEC-0014](../specs/0014-get-external-sequences.md) for the requested
`scripts/get-external-seq.py`. Inspected surviving PClass external-audit filtering
and retrieval, current PClass-Dark reference resolution and profunc graph/Swiss-Prot
parsers. The draft retains the exact experimental evidence allowlist and all
eligible accessions, excludes the complete local Swiss-Prot accession snapshot,
and replaces PClass's duplicate term union with latest-qualifying-GAF-date
selection (lexicographic tie break). Proposed FASTA headers preserve winner-only
MF terms and date; `external_filtered.txt` distinguishes annotation, accession
and retrieved-sequence counts. Includes disk-backed state, bounded retrieval,
provenance, resumability, publication recovery and test expectations. Updated the
spec index. Documentation only: no downloader implemented or acquisition run.
Validation: local Markdown links and scoped whitespace checks passed.


### 2026-10-31 — Make external sequence specification implementation-ready

Revised [SPEC-0014](../specs/0014-get-external-sequences.md) against actual
profunc graph schema/loading, complete-reference FASTA resolution and existing
CLI/package conventions. Made optional Python/CLI inputs consistent, added the
manifest-bound source-FASTA override, fixed the exact accession/date/qualifier
contracts, and specified batched UniProt stream retrieval, retry/timeouts and
release consistency. Defined SQLite aggregation, count reconciliation, cache
identity/resume/refresh, rollback publication and output ownership. Removed the
circular report-hash requirement and distinguished deterministic FASTA/stage
counts from per-run timestamp metadata. Added an independent 11-row acceptance
fixture, explicit winner-only MF retention, exceptions, script/API/CLI behavior
and completion tests. Index marks the specification implementation-ready, not
implemented. No script or network acquisition was run.

Validation: independently checked fixture counts `11,10,9,8,7,6` and five
remaining distinct parents at each annotation filter, reviewed final retrieval/
rejection/deduplication arithmetic, checked local Markdown links/fence balance
and scoped whitespace. These are documentation checks, not passing runtime
implementation tests.


### 2026-10-31 — Implement external sequence collection (SPEC-0014)

Implemented [SPEC-0014](../specs/0014-get-external-sequences.md) on `issue#26`
from revision `122c7886` (uncommitted implementation). Added
`scripts/get-external-seq.py`, the `pclass-dark get-external-seq` command and
importable `collect_external_sequences` API. Scientific filtering is streamed
into SQLite, using the explicit experimental evidence allowlist and local MF
nodes, parent normalization and complete local Swiss-Prot accession exclusion.
Exact sequence duplicates retain the newest qualifying annotation's accession,
lexical ties and only that winner's terms. No sampling, sequence-to-Swiss-Prot
exclusion, model computation or new scientific threshold was introduced.

Added bounded request threads, retry/rate-limit handling, validated release
headers, source/cache identities, batch resume, cache registration recovery,
writer locking and durable rollback/commit of the two cwd outputs. Final file
ownership intentionally protects manual changes even with refresh; the spec and
[operation/API documentation](../external-sequences.md) now state that
behavior explicitly. Added requests as a direct dependency, source-distribution
script/docs inclusion, README examples and the synchronized spec index/status.

Early fixture validation exposed the actual profunc graph schema `"2.0"` (the
initial test fixture used schema 1) and an independently written report golden
fixture used a 31-character first column instead of the required 32. Corrected
those fixtures, then expanded failure, resume and publication tests, including
rollback after the refreshed FASTA actually changes winner. Tests independently
check exact stage arithmetic, rejected-newer annotation behavior, winner-only GO
terms, ties/order/thread consistency, deliberate hash collisions, malformed
inputs, non-ASCII residue handling, missing/invalid sequences, unknown schemas,
queue bounds, cache corruption, missing backups, release changes and CLI exits.

Validation (CPU; synthetic fixtures/mocked HTTP unless identified below):

- Broad integration baseline: `python -m pytest -q` — **432 passed, 2 skipped**,
  612.37 s; `/tmp/pclass-dark-spec0014-full-tests.log`. Two opt-in tests skipped;
  one existing Torch warning about `AMD_SERIALIZE_KERNEL=3` did not fail tests.
- Final focused run after new-feature hardening: `python -m pytest -q
  tests/test_external_sequences.py tests/test_cli.py` — **68 passed**, 10.99 s;
  `/tmp/pclass-dark-spec0014-focused-tests.log`. This includes 49 external-sequence
  cases and existing CLI integration coverage.
- `python -m ruff check src tests scripts/get-external-seq.py`, changed-file
  formatting, local Markdown links/fence balance and `git diff --check` passed.
- `python /tmp/pclass-dark-spec0014-package-smoke.py` built clean code-only wheel
  and sdist in an isolated temporary project (no source data/model artifacts,
  `python -m build --no-isolation`). Inspected requests metadata, module and
  sdist script inclusion. Log: `/tmp/pclass-dark-spec0014-package-smoke.log`.
- Real default reference adapter: **10041 MF nodes; 575748 Swiss-Prot accessions**.
  Resolved graph at `../profunc/src/profunc/library/default/bundle/dataset/go_graph.pkl`
  and FASTA at `../profunc/datasets/20260911T184540Z-d7e84090016d/accession.fasta`;
  both verified against their source contracts.
- Real historical-source parser smoke: first **10000 annotation rows** plus
  headers from `../PClass/work_plm1/goa_uniprot_all.gaf.gz` (11664243116 compressed
  bytes), streamed into a temporary gzip/SQLite with the above references.
  Uncompressed prefix SHA-256:
  `366d8d4857387f52a44460f64f732717e4349b95ccbc3886a4a98aa51621bc96`.
  Annotation-stage counts: **10000, 10000, 9999, 4640, 10, 10**; distinct parents
  after DB/no-NOT/MF/evidence/graph: **1683, 1683, 1383, 4, 4**; **2** eligible
  parents remained after local Swiss-Prot exclusion. Temporary artifacts removed.
- Live transport smoke: one GET to UniProt `/uniprotkb/stream` with
  `query=(accession:P69905)`, `format=fasta`, `includeIsoform=false`, retries 0.
  Returned one canonical record, **142 residues**, release **2026_03**, release
  date **02-September-2026**; required headers and FASTA parser passed. Only
  temporary files were used; no full-source acquisition was started.

These are engineering/source-contract checks, not held-out performance or a
complete external dataset build. Large-source runtime/peak disk use remains an
estimate; historical run directories remain retained until manually managed.
Next operational step is an explicitly chosen full acquisition using the new
command in the desired output directory. No commit or push was made.


### 2026-10-31 — Remove external-sequence package subcommand

At the user's request, removed `pclass-dark get-external-seq` from the root CLI.
Moved the standalone Typer adapter to `external_sequence_cli.py`; the requested
`scripts/get-external-seq.py` and Python API remain available independently of
that registry. Updated README, operation documentation and SPEC-0014. Earlier
command examples in this chronological log describe the superseded interface.
Scientific filtering and artifacts are unchanged.

Validation: `python -m pytest -q tests/test_external_sequences.py tests/test_cli.py`
— **69 passed**, 10.73 s, including explicit root-command rejection, standalone
script help, cwd output placement and interruption handling. Ruff lint/format,
local Markdown link checks and whitespace checks passed. No commit or push.


### 2026-10-31 — Readable GOA download progress

Updated SPEC-0014 and GOA progress logging to show comma-separated decimal KB
(1000 bytes), total KB and one-decimal completion percentage from Content-Length.
Missing/invalid totals or HTTP content decoding produce an explicit percentage-
unavailable message instead. Retained 128 MiB updates and added final EOF
progress, including short downloads. Already running processes retain their
loaded logging implementation; no acquisition was restarted.

Validation: `python -m pytest -q tests/test_external_sequences.py` — **57 passed**,
10.20 s, including independently calculated 65.7% progress for the observed
18790481920/28598433883 byte counts, unknown-total fallbacks and EOF 100% logging.
Ruff lint/format and whitespace checks passed. No commit or push.


### 2026-11-01 — Handle non-protein GOA IDs and readable parse counts

Investigated the actual failed parse in `work-unknown/`. Source GOA gzip is
28598433883 bytes, cached SHA-256
`4da0481d560cd09a870ab89283014e712f6bf10b8bafcdce916134eb1f13be9a`;
embedded headers report GAF 2.2, generation 2026-10-11 19:22 and GO release
2026-09-27. A read-only sequential gzip scan captured the first three CPX rows.
At physical line **648144824**, the source has **17 columns**, DB `UniProtKB`,
object `CPX-10041`, symbol `il18-il18r1_human`, type `protein_complex`, aspect
`P`, evidence `IEA`, date `20261009`. Adjacent IDs are `CPX-1014` and `CPX-10161`.
This is a well-formed complex annotation rather than a broken column layout;
protein-isoform normalization reduced its ID to `CPX` before the old strict
validator aborted ahead of MF/evidence filtering. Added the captured rows and
source attribution in `tests/fixtures/goa_nonprotein_ids.gaf`.

Updated [SPEC-0014](../specs/0014-get-external-sequences.md) before the fix, then
made invalid/non-protein accession rows a counted rejection instead of a fatal
GAF error. Metadata retains a total and up to ten raw-ID/line examples, warnings
are bounded, and final report/manifest include `invalid_accession_rows`. Input
counts include rejected rows; downstream normalized-protein counts exclude them.
Header/structural/date/gzip validation and local-reference/sequence-ID validation
remain strict. Valid protein annotation policy is unchanged. Parse progress and
GAF diagnostic line numbers now use commas, e.g.
`Parsed 648,000,000 GAF rows; 582,876 graph-qualified rows`.

Validation: `python -m pytest -q tests/test_external_sequences.py tests/test_cli.py`
— **85 passed**, 13.33 s (`/tmp/pclass-dark-gaf-invalid-accessions-tests.log`).
Covers malformed IDs/isoform suffixes, bounded warnings, actual complex records,
exact stage counts, comma formatting, strict reference validation and restarting
a legacy failed parse with a registered cached download and committed partial
SQLite state. After adding fixture source attribution, its focused regression
passed again (**1 passed**, 65 deselected, 0.43 s). Ruff lint/format, local links
and whitespace checks passed.

The actual user acquisition artifacts and failed run state were not modified or
restarted. Repeating the same command without refresh reuses the downloaded GAF;
incomplete GAF parsing restarts from the beginning by design. Full-source
reprocessing was not performed. No commit or push.


### 2026-11-01 — Draft durable external inference evaluation

Drafted [SPEC-0015](../specs/0015-evaluate-external.md) for the requested
`scripts/evaluate_external.py`. Inspected current PClass infer composition,
JSON models, additional Needle/ProSig steps, client serialization and service
limits (1024 sequences and 10 MiB default JSON body), Dark JSON/B-score/window
reduction and profunc graph/IC similarity code. The draft preserves the exact
seven requested columns and all original inference JSON for downstream analyses.
Defines annotated FASTA truth, staged batching, independent expensive inference
and cheap scoring identities, resume, full-precision floats, abstentions and
verified cluster-to-composed-GO mapping. Top-ranked Dark prediction and Lin/AMB
are explicitly proposed scientific defaults, not user-approved metric decisions.
Updated the index. Documentation only: no evaluator implemented, model loaded or
inference/evaluation run. Local Markdown links/fences and whitespace checks passed.


### 2026-11-01 — Make external evaluation specification implementation-ready

Revised [SPEC-0015](../specs/0015-evaluate-external.md) against the actual PClass
CLI/native pipeline, process dependencies, bundle manifests, Dark query ID and
quota contracts, GO bindings and profunc graph/IC methods. Replaced unresolved
scientific placeholders with explicit, versioned `top1-v1` / `lin-amb-v1` design
defaults and a seven-column analytical fixture; this records the chosen defaults,
not a claim of user-approved performance endpoints. Preserved all returned
observations and documented which future changes still require inference.

Resolved the draft's cache/annotation identity collision: canonical sequences,
truth and original-source/header snapshots now have separate identities/stores.
Defined complete Python/config/result records, exact same-interpreter command
arguments and native equivalence, PClass body/count admission and HTTP413 split
state, Dark conservative quota admission, process-group watchdog/cleanup,
streaming JSON validation, unambiguous offline generation/resource selection,
portable saved workspaces, per-engine collection fingerprints independent of
scoring code, ownership and cross-filesystem publication recovery. Specified
remote server bundle attestation because the current API does not expose its
model/index identity; version/help checks do not perform biological scans.

Additional reference inspection identified PClass's Python >=3.12 requirement,
ProSig's local subprocess/core-library behavior and the inspected profunc graph
implementation's datetime.UTC dependency (cached scoring requires Python >=3.11,
not a presumed 3.10 from metadata). Removed an unnecessary requirement for an
English Dark description: verified composed terms are loaded directly from
profunc cluster records and bound to saved fits/reference identities. Documented
that an idle PClass service can still occupy GPU memory and how a completed engine
is reused while the other runs, without automatically stopping user services.
Updated the spec index to implementation-ready, not implemented.

Documentation validation: checked local links/fence balance and whitespace;
checked exact seven-column fixture headers/row widths. Executed the synthetic
schema-2.0 MF graph against profunc's existing fast helper: {X,Y} versus {X} and
{Y} both **0.875**, X versus unrelated Z **0.0**, empty prediction **undefined**,
root-only self comparison **undefined**. Independently checked directional means
0.75 and 1 yield 7/8, distinct from pooled 5/6. No evaluator implementation,
evaluator test suite, service/model job or real external evaluation was run.

## 2026-11-01 — Implement durable external inference comparison (SPEC-0015)

Implemented [SPEC-0015](../specs/0015-evaluate-external.md) on `issue#28`
(base revision `4b15854e`). Added directly executable
`scripts/evaluate_external.py` and the importable `external_evaluation` API,
without a new `pclass-dark` subcommand. Added typed configs/records, exact
seven-column TSV serialization, analytical Lin/AMB scoring and deterministic
minimum-B Dark reduction. Preserved undefined scores as NA and retained complete
inference JSON for subsequent scientific choices.

Collection now validates/canonicalizes FASTA and GO truth independently, strips
truth from model inputs, verifies installed PClass/Dark assets and GO bindings,
admits batches by actual serialized PClass body/count and Dark quotas, and runs
bounded same-interpreter CLI or equivalent native children. Native PClass retains
Needle and ProSig. Added process-group watchdog/cleanup, explicit size-rejection
splitting, durable plans/raw receipts, independent engine/score identities,
immutable resource snapshots, SQLite joins and bounded spawned scoring workers.
Resume reuses verified completed chunks; score-only avoids inference imports,
services and model checks. Output ownership, locks, same-filesystem staging,
backup/journal recovery and refusal to overwrite corrupt/manual results protect
expensive completed work.

Implementation corrected the draft's claimed common alphabet: actual Dark
rejects J, so evaluator preflight rejects it too. Actual report precision uses
`inference_dtype`. A test initially exposed that ijson's C builder cannot safely
propagate a ValueError from custom dict insertion (SystemError); the parser now
marks duplicate keys and rejects them after object construction. Other fixture
failures revealed an injected-capability alias and incorrect test expectations;
these were corrected before passing checks. Parallel-scoring validation was
strengthened to delete the score cache first so it really exercises spawn.
Final hardening verifies raw contract/chunk hashes, detects graph mutation during
snapshot loading, and checks ijson availability before expensive collection.

Updated optional dependency groups (`evaluation`, `evaluation-score`, development
ijson), source packaging, README, [operational/API guidance](../external-evaluation.md),
specification and index. Documented O(query-count) chunk-plan/coverage metadata
memory, graph copies per scoring worker, raw storage retention and exact prior
selection-pointer restoration on rollback.

Validation environment: Python 3.14.6, installed PClass 0.3.0, ProSig 0.5.1 and
profunc 0.1.0. ijson 3.6.0 was installed into the isolated temporary target
`/tmp/pclass-evaluation-deps`; no model/service job or production output was
launched. Commands used `PYTHONPATH=/tmp/pclass-evaluation-deps:src:.`:

- Earlier broad baseline: `python -m pytest -q` — **498 passed, 2 skipped** in
  600.59 seconds. The opt-in real-model tests were skipped; an existing Torch
  AMD_SERIALIZE_KERNEL warning appeared. This preceded the final additional
  evaluator hardening/tests.
- Final relevant suite: `python -m pytest -q tests/test_external_evaluation.py
  tests/test_cli.py` — **84 passed** in 25.14 seconds. Covers independent analytical
  examples, invalid inputs, batch boundaries, malformed reports, native/CLI
  pipeline equivalence with simulated engines, CPU/parallel consistency,
  annotation/graph rescoring, cache corruption, publication failures/rollback,
  locking, child timeout/output caps and grandchild cleanup. All prediction
  fixtures are synthetic; they are not biological performance measurements.
- `python -m ruff check src tests scripts/evaluate_external.py`, executable
  `./scripts/evaluate_external.py --help`, and `git diff --check` passed.
- Built a clean temporary code-only project with `python -m build --no-isolation`;
  wheel/sdist creation and evaluator modules, extras metadata, operational guide
  inclusion and executable source-script mode checks passed. This did not build
  a full data-bearing release or exercise fresh isolated dependency installation.
- Real local artifact-only probes verified the PClass serving bundle
  `bbaca1b4ca82876f4173a38da3b975d0c9757a723b33e1d70eb3027442679283`
  (22 recorded files), Needle/ProSig version/help availability, 3,443 usable Dark
  clusters with verified reference-bound composed GO mappings, and 10,041 MF
  graph nodes. No serving request, alignment/ProSig biological scan or embedding
  was performed.

Remaining experimental work: run an explicitly selected real cohort with recorded
input/reference identities and hardware; validate actual CPU/GPU inference,
device numerical differences, runtime and total storage. Exact Swiss-Prot
exclusion/deduplication alone do not demonstrate homology-independent performance.

## 2026-11-01 — Simplify external evaluation to consume working classifier commands

Supersedes the evaluator dependency/attestation/native-adapter requirements in the
previous entry, following the user's instruction to minimize dependencies. The
current [SPEC-0015](../specs/0015-evaluate-external.md) and
[usage guide](../external-evaluation.md) describe the replacement contracts.

The evaluator now invokes PATH-resolved `pclass infer --fasta ... --format json`
and `pclass-dark infer --fasta ... --format json`. It does not import PClass,
ProSig, Torch or transformers; check models/bundles; probe alignment executables;
or require a bundle fingerprint. Removed evaluator optional extras and ijson,
and removed the duplicated native classifier pipelines. The classifiers own
all their internal operations/dependencies. Python >=3.11/POSIX and the existing
base profunc dependency/data suffice for orchestration/scoring; the commands can
use different interpreter environments. No package installation is needed for
this revision when the existing base environment and commands work.

PClass_go now follows the user's explicit top-match wording: nearest embedding
neighbor's GO MF terms, with its distance; additional voted GO terms and
alignment/identity confirmation are ignored by the comparison. Dark uses its
lowest-B accepted cluster's composed GO set. The installed profunc command is
`analyze go-set-sim`, whose `explain_set_amb` calls the same detailed Lin/AMB
helper used directly here. This avoids per-query subprocesses and rounded text
parsing. Dark reports lack GO terms, so their composed cluster labels are loaded
automatically from profunc and frozen with the graph. Reference equivalence to
the classifier is an operator assumption rather than asserted fit attestation.

Retained bounded chunks, strict standard-library JSON parsing, exact result
coverage, full raw outputs, independent truth identity, SQLite scoring, bounded
spawned scoring workers, raw checksum verification and journaled publication.
Default chunk counts are 128; PClass is capped at 1,024 with conservative byte
admission and durable explicit size-rejection splitting. Reuse identity records
the executable/command, working directory, relevant environment and run tag.
Remote model/index changes cannot be detected automatically; `--run-tag` forces
new observations. Old attested-format raw files are preserved but are not adopted
as new command-output generations. Documented per-chunk JSON memory and retained
storage costs, and updated README/index without changing base dependencies.

Validation uses lightweight fake PATH executables with real subprocesses, plus
analytical and failure fixtures. An intermediate run found missing test imports;
these were restored. Actual installed-command/profunc preflight passed using a
single synthetic sequence without running either classifier: MF graph hash
`819a5675ebd00edd6622d1bd45c8feb05928a8fc2f12d7dc67e555fc843b4f06`,
map hash `9368b97b0d8e66779717d4254fff07db0c30dfd3e61cb3de52d792ba62a0d4f5`.
Clean temporary code-only wheel/sdist builds passed and confirmed executable
script/documentation inclusion and absence of ijson/evaluation-extra metadata.
Lint and executable help passed. Final test counts are recorded below.

No real cohort inference, GPU comparison or fresh data-bearing release was run.
The prior full-suite baseline applies to the earlier implementation; this
revision is validated with the relevant evaluator, CLI and external-sequence
suites, including retained publication/recovery behavior.

Final validation: `PYTHONPATH=src:. python -m pytest -q
 tests/test_external_evaluation.py tests/test_cli.py tests/test_external_sequences.py`
— **139 passed in 30.74 seconds**, without the temporary ijson dependency path.
Coverage includes top-neighbor/lowest-B analytical scoring, ignored alignment and
voted fields, dependency-import blocking, real PATH command collection, batching
and size splitting, one-engine failure/resume, independent Dark overrides,
annotation/graph rescoring, run tags, parallel determinism, strict JSON/report
validation, subprocess limits/grandchild cleanup, locks, corruption/manual edits,
publication interruption/rollback and backup corruption. `python -m ruff check
src tests scripts/evaluate_external.py`, formatting, documentation links/fences
and `git diff --check` passed.

## 2026-11-01 — Increase external evaluation processing defaults

Changed both CLI and Python config defaults to PClass batches of 512 sequences,
Dark chunks of 512 sequences, and 8 scoring workers, as requested. Explicit
smaller values and existing limits remain supported. Updated SPEC-0015 and the
operational guide. Added regression coverage for default and overridden CLI/API
values, and kept the parallel-consistency test's baseline explicitly serial.

Validation: `PYTHONPATH=src:. python -m pytest -q
 tests/test_external_evaluation.py tests/test_cli.py` — **75 passed in 25.25s**.
The explicitly serial-versus-parallel scoring regression also passed separately
(**1 passed**). Ruff lint/format and whitespace checks passed.

## 2026-11-02 — External collection progress percentage

Added cumulative per-engine cohort completion percentages to the existing
chunk-completion log used by `scripts/evaluate_external.py`. Counts include
reused receipts and split leaves. Updated SPEC-0015 and added regression checks
for ordinary batches, a partial final batch, and recursively split requests.

Validation: `python -m pytest -q tests/test_external_evaluation.py tests/test_cli.py`
— **77 passed in 26.87s**. Ruff lint/format and `git diff --check` passed.


### 2026-11-02 — External audit plots and summary

Implemented [SPEC-0016](../specs/0016-external-audit.md):
`scripts/external-audit.py INPUT` and `pclass-dark external-audit INPUT`, backed
by importable parsing, binning, coverage, summary and rendering functions. Accept
arbitrary TSV filenames with SPEC-0015 headers. Four external_audit_ PNGs provide
paired match scatter, stacked hit/non-hit/undefined confidence counts with hit
percentages, side-by-side coverage/hit-rate curves, and a summary table; also emit
a summary TSV. Keep tied evidence together, preserve distance overflow, and handle
missing predictions/scores/confidence explicitly. Default cutoff 0.7, ten equal
[0,1] bins, default output beside input. Optional audit Matplotlib dependency;
headless rendering only on explicit API calls. README/API/CLI/spec index and sdist
script manifest synchronized. CPU O(N log N) ranking needs no workers/GPU.

Validation: audit/evaluation/CLI suites **87 passed** (30.31 s;
`/tmp/external-audit-final-tests.log`). Ruff and scoped whitespace checks passed.
Tests cover independent analytical counts, ties, endpoints/overflow, missingness,
invalid TSV/numbers, empty inputs, PNG output, TSV summary and both CLI adapters.

Real-data command: `MPLCONFIGDIR=/tmp/pclass-external-audit-mpl python
scripts/external-audit.py work-unknown/external_evaluation.tsv`; default bins/cutoff,
no sampling, fitting, preprocessing or random seeds. Input SHA256
09943c5a25388cf4f38e1b712c7470f0e5c33824f4841bbe50df92c8e5b97821;
baseline 609e2a6e5099603e3f3afa2795f7a68d6f37ee55 plus uncommitted audit changes.
CPU, Python 3.14.6, NumPy 2.4.4, Matplotlib 3.11.0. Four PNGs and summary TSV at
`work-unknown/external_audit_*`; final run log `/tmp/external-audit-real-final.log`.
Visually inspected scatter, confidence, coverage and summary designs. Restricted
home config caused Matplotlib cache warnings initially; final run used writable
/tmp cache, without changing application environment defaults.

Results: 27,910 queries, distance >1 count **0**, maximum 0.3711543083190918.
Both methods predict GO sets for every query. At cutoff 0.7, PClass has 7,017 hits
among 27,845 scorable queries (25.20%); Dark has 3,229/27,847 (11.60%). Both hit
2,190; PClass-only 4,827 and Dark-only 1,039 (other method may be unscorable).
These are descriptive cohort results, not a leakage-controlled held-out estimate.
B scores occupy the first equal-width bin, illustrating that identical bins do
not equate confidence scales. Whole-set output replacement is not atomic, though
all artifacts render in staging before publication. No commit or push.


### 2026-11-02 — Clarify coverage axis direction

Added “Strong to weak” below the coverage x-axis label in both panels
([SPEC-0016](../specs/0016-external-audit.md)). Added a rendering regression
assertion and regenerated the work-unknown audit PNGs from the same input.

Validation: all 10 audit tests passed (4.29 s); Ruff and scoped whitespace
checks passed. PNGs regenerated successfully; no analysis values changed.


### 2026-11-02 — BLASTP lookup for A0A078IB52

Extracted the 150-aa query from work-unknown/external_sequences.fasta and searched
local /home/jmao/projects/blastdb/swissprot (487,728 sequences; database date
2026-09-15). CPU, four threads, BLASTP defaults except reporting E-value 1000,
max_target_seqs 20 and max_hsps 1. Command: blastp -query
work-unknown/A0A078IB52.fasta -db /home/jmao/projects/blastdb/swissprot -evalue 1000
-max_target_seqs 20 -max_hsps 1 -num_threads 4 -outfmt '6 saccver stitle pident
length qlen qcovs evalue bitscore'. Kept first five unique nonself subjects in
BLAST output order; output work-unknown/A0A078IB52_blast_top5.tsv. No significant
sequence matches: best E-value 3.5, 29% query coverage. Expanded the reporting
threshold to return five subjects, without interpreting these as homologs.


### 2026-11-02 — Unmatched-query GO frequencies

Ranked work-unknown/unmatched_accessions.tsv GO annotations by distinct-accession
frequency (descending, GO ID tie-break), with descriptions and frozen graph IC
from evaluation graph 819a5675ebd00edd6622d1bd45c8feb05928a8fc2f12d7dc67e555fc843b4f06.
Verified graph SHA256; IC uses stored -ln(reference frequency), not unmatched
cohort frequencies. 2,755 unique accessions, 760 distinct terms, 3,559 term
assignments. Wrote work-unknown/unmatched_go_terms.tsv with rank, term, count,
percentage of accessions, description and IC. Checked unique IDs, term membership,
sorted order, bounded counts and total assignment conservation. Most frequent:
GO:0005515 protein binding, 1,331 accessions (48.31%), IC 2.3043773196925526.


### 2026-11-02 — V2 audit excluding protein-binding annotations

Updated [SPEC-0016](../specs/0016-external-audit.md): script/CLI defaults exclude
any query whose truth MF set contains GO:0005515, including mixed sets. Use saved
query truth validated against input/publication/score/status file hashes, not
predicted GO sets. Standalone inputs accept --truth-tsv (Query_id/go_mf_terms),
requiring exact unique query coverage. Library API adds exclude_go/truth_tsv while
preserving its unfiltered default. Recompute all denominators on retained queries
and report original/excluded counts. Filtered PNG/summary TSV filenames add _v2,
preserving unfiltered outputs. README and API/CLI documentation synchronized.

Validation: audit/evaluation/CLI suites **92 passed** (32.79 s;
`/tmp/external-audit-v2-final-tests.log`); Ruff and whitespace checks passed.
Tests verify truth-based filtering even when predicted GO differs, multi-term
sets, empty cohorts, ties/coverage denominators, original-file preservation,
verified metadata and malformed annotation refusal.

Ran `MPLCONFIGDIR=/tmp/pclass-external-audit-mpl python scripts/external-audit.py
work-unknown/external_evaluation.tsv` using default cutoff 0.7, ten bins, same CPU
software/input identity as initial audit. Source SHA256 remains
09943c5a25388cf4f38e1b712c7470f0e5c33824f4841bbe50df92c8e5b97821. Verified truth
from score 5f3b999fddc68ad6e1e08e046a235b07c0899b1d14d1fdbcc06ca8f64a74c2c2.
Excluded 13,530 queries, retained 14,380. All four _v2 PNGs and summary TSV created
at work-unknown/external_audit_*_v2; log /tmp/external-audit-v2-real.log. Visually
checked summary output. Each method has 14,317 scorable queries: PClass 5,439 hits
(37.99%); Dark 2,548 (17.80%); both-hit 1,780, method-only 3,659/768. Zero retained
distances >1; max 0.3559103012084961. Annotation-selected descriptive cohort, not
an independently held-out accuracy estimate. No model rerun, commit or push.

## 2026-11-02 — Draft Swiss-Prot evaluation specification

Drafted [SPEC-0017](../specs/0017-evaluate-swiss.md) for
`scripts/evaluate_swiss.py`, the installed command and Python API. Planned output
is `swiss_evaluation.tsv`, reusing the external evaluator with accession truth
from the same profunc mapping used by `analyze function`. Truth includes the full
direct MF set before description pruning. Recorded strict missing-truth behavior,
separate cache/provenance, audit integration and potentially in-sample scope.

Inspected profunc's function CLI, GO-set loader/resolver, description logic and
Swiss-Prot annotation writer. Read-only command `profunc analyze function
A0A011QK89 -v` succeeded with `GO:0047545`. This is an annotation-source check,
not classifier validation or a full-cohort coverage check. Documentation only;
implementation and automated validation remain planned. Next: implement and test
the shared orchestration and frozen truth lookup described in the draft.

## 2026-11-02 — Plan faster Swiss-Prot evaluation

Revised draft [SPEC-0017](../specs/0017-evaluate-swiss.md) after inspecting the
actual PClass infer CLI, artifacts loader, neighbor client and serving engine,
and Dark sessions, precomputed-window scorer and existing embedding files.
PClass currently runs Needle and ProSig unconditionally even for JSON; planned
neighbor-only collection bypasses both plus descriptions and alignment sequence
loading. Dark recreates its model per inference session; planned persistent
runtime shares model/library resources across bounded checkpoint chunks. Existing
Dark vectors are a promising validated cache source, with coverage/representation
checks required. Added bounded cache/worker behavior, evidence contracts, parity
checks and staged throughput benchmarking. Kept self matches and prediction
semantics explicit; no unsupported speedup estimate or full-cohort run claimed.

Documentation only. No classifier code changed or performance tests run. Checked
specification links, code fences and whitespace. Next: implement and benchmark
neighbor-only PClass and persistent/cache-backed Dark adapters against saved
baseline evidence before the full cohort evaluation.

## 2026-11-02 — Resolve Swiss evaluator implementation contracts

Revised [SPEC-0017](../specs/0017-evaluate-swiss.md) into an implementation-ready
design. Chose a single native PClass-service/Dark-runtime path; deferred alternate
backends and speculative optimizations. Defined configs/defaults, module/API
boundaries, worker protocol/deadlines, evidence/identity contracts, cache join
keys and integrity checks, offline scoring and phased acceptance tests/pilot.

Inspection found the audit hardcodes `.external-evaluation`, embedding row files
lack coordinates, existing preparation verification temporarily loads reference
sequences, and relative-score arrays already have a bounded LRU. The revised
spec explicitly addresses each instead of assuming existing APIs support the
new orchestration. Added publication marker v2/state path migration with v1
compatibility and audit tests. Updated spec index. Documentation only; no code,
benchmark or pytest execution. Link/fence/content/whitespace checks passed.
Remaining work is implementation and the specified fixture/live validation;
full-cohort coverage, performance and GPU behavior are still unmeasured.


## 2026-11-02 — Implement native Swiss-Prot evaluation

Implemented [SPEC-0017](../specs/0017-evaluate-swiss.md): executable
`scripts/evaluate_swiss.py`, installed `pclass-dark evaluate-swiss`, and importable
`evaluate_swiss` with validated typed configs. Defaults publish
`swiss_evaluation.tsv` and separate `.swiss-evaluation` state. Frozen profunc
accession truth preserves full direct MF annotations, original query IDs and
normalized accession joins. Shared external snapshot/scoring/publication adapters
retain external defaults. Audit marker v2 resolves/validates the Swiss workspace,
including outputs in another directory; v1 remains supported.

PClass uses its installed native neighbor client, bypassing infer CLI, Needle,
ProSig, descriptions and alignment-sequence loading. Dark's new resident runtime
shares guarded library/model resources across disk-backed checkpoint sessions,
initializes no model for fully cached windows, and verifies a coordinate/sequence
embedding index. Bounded workers, deadlines, split/restart receipts, lossless
numeric evidence, per-worker GO caches and numerical thread budgets are tested.
Source fingerprints/package versions and per-chunk execution metadata preserve
provenance. Dark collection runs before PClass to validate/pin its resources once.

Real-data validation initially rejected the existing embedding cache because its
originating preparation ID differs from the current preparation. This supersedes
the draft's requirement for equal preparation IDs: full verified inventory,
sequence/window coordinates and representation determine vector compatibility;
both preparation IDs are retained as provenance. The final read-only preflight
passed for the first three records of `work-swiss/sequences.fasta`: **3 windows,
3 cache hits, 0 misses**. No service calls, models, predictions or TSV publication.
Input `/tmp/pclass-swiss-preflight/queries.fasta` (982 bytes), SHA-256
`7702cd41d68236028f83f7d5efb9ad9f371171ae3a80cfaa81683c820b66d5c7`;
log `/tmp/swiss-preflight-final.log`, state
`/tmp/pclass-swiss-preflight/.swiss-evaluation/`. Earlier failed preflight logs
`/tmp/swiss-preflight.log` and `/tmp/swiss-preflight-rerun.log` are retained.

Reproduce the preflight from `/tmp/pclass-swiss-preflight` with
`python /home/jmao/projects/PClass-Dark/scripts/evaluate_swiss.py queries.fasta
--dark-library-dir /home/jmao/projects/PClass-Dark/data/library
--dark-embedding-cache /home/jmao/projects/PClass-Dark/data/setup-data/embeddings/9b71f736c6eb714c1f963578f50faa77a2feee219bedfc82d39f33aefc086980
--dark-prepared-dir /home/jmao/projects/PClass-Dark/data/setup-data/prepared/8e5994726cb958fce5d2396835f9a3767da897007110a9fda4f2e3b06cccbe4e
--preflight-only`. Input selection is deterministic first-three order, no seed.
Graph/mapping identities are in that log and frozen state; selection identities
and hashes are recorded in `/tmp/pclass-swiss-preflight/source-identities.json`.
Environment: Python 3.14.6, NumPy 2.4.4, profunc 0.1.0, PClass 0.3.0; baseline
revision `3af0a2a6bae7906ec0b8efba56f43553a9c81b95` plus uncommitted implementation.
This validates source/cache integration, not full-cohort coverage or held-out
performance. No live model or GPU throughput measurements were taken.

Validation:

- `python -m pytest -q`: **592 passed, 2 skipped in 731.56 s**,
  `/tmp/swiss-full-suite.log`. The opt-in real pinned-checkpoint CPU/GPU tests
  require `PCLASS_DARK_TEST_CACHED_ESM=1` and were not enabled.
- Final Swiss suite: **52 passed in 75.89 s**, `/tmp/swiss-final-suite.log`.
  Localhost protocol fixtures required socket permission in this environment.
  Final cached numerical-thread limit/restoration change: **3 affected tests
  passed in 15.23 s**, `/tmp/swiss-thread-checks.log`; current file has 53 tests.
- Ruff lint/format, whitespace and documentation link/fence checks passed.
  Temporary code-only wheel/sdist builds passed; checked native modules and
  script/guide inclusion. Build logs `/tmp/swiss-package-build-final.log`.

Synchronized README, [operational/API guide](../swiss-evaluation.md), specs
0015/0016/0017 and index. Full cohort inference, a real classifier throughput
pilot and real-device parity remain operational validation work; no speedup
factor is claimed. Existing self/training matches remain intentionally possible.


### 2026-11-03 — Warn on unknown GO terms during scoring

Downgraded the shared scoring error for graph-missing GO terms (reported for
`GO:0018491` after Swiss PClass collection) to a `UserWarning` describing a
possible graph/annotation version mismatch. Profunc computes AMB Lin using
available terms; original annotations/source counts remain intact and affected
metric statuses record `missing_terms`. Scores with no usable terms are `NA`.
Preflight truth/map compatibility and unknown cluster checks remain strict.
Updated specs [0015](../specs/0015-evaluate-external.md) and
[0017](../specs/0017-evaluate-swiss.md) and the operational guide.

Validation: external/Swiss suites: 113 passed, one localhost fixture setup error
under sandbox socket restrictions; rerunning that test with socket permission
passed (1 passed). Four analytical regressions cover partially/fully missing
truth and prediction sets. Ruff lint/format and `git diff --check` passed.
No live cohort or GPU execution was needed for this scoring-only change.

### 2026-11-03 — Recover completed Swiss inference before scoring

The unknown-GO-warning edit changed a scoring hash included in native inference
contracts. A normal rerun therefore started new Dark work, even though both old
generations were complete; score-only could not recover without a published
selection. Added conservative completed-generation discovery to the normal Swiss
pipeline, independent of selection. The reviewed before/after scoring hashes are
compatible; all other contract fields and chunk configuration must match. Kept
the inference fingerprint intact because the scoring module also owns evidence
projections; arbitrary future scoring edits are deliberately not exempted.
Original generation IDs and receipts remain immutable. The existing verifier
checks receipt/response hashes and exact sequence/query coverage before reuse;
corruption fails rather than silently reclassifying. Logs explicitly say
`reused completed generation`. Resource preflight still runs.

Read-only recovery validation on `work-swiss/.swiss-evaluation` selected Dark
`3e4311b6a98769b952ec49b5c73644863821b6ade81d537369db0ce750e599ac`
and PClass `09aeaea23c0c4c88237b769ff3dbba31e39fd7753f97a8132bea5cc578697d21`,
each with 744 chunks covering 380,578 queries. No inference or publication was
performed. Sources, saved truth and graph were verified during assessment.

Validation: `python -m pytest -q tests/test_swiss_evaluation.py
 tests/test_external_evaluation.py`: **118 passed in 81.24 s**, log
`/tmp/swiss-resume-tests.log` (localhost fixture socket permission enabled).
New regressions cover scoring failure before selection, reviewed scoring change
reuse without worker tasks, configuration/unreviewed-source invalidation, and
corrupt-response rejection. Ruff lint/format and whitespace checks passed.
Updated [SPEC-0017](../specs/0017-evaluate-swiss.md) and operational guide.
Full-cohort scoring remains to be run by the user; no GPU/model run was needed.

### 2026-11-03 — Draft Swiss audit implementation specification

Drafted [SPEC-0018](../specs/0018-swiss-audit.md) for the requested
`scripts/swiss-audit.py`, based on the actual external audit library, CLI and
SPEC-0016. Plans shared calculations/rendering, Swiss output names, the existing
GO:0005515 truth exclusion, verified marker-v2 metadata, standalone truth,
resource bounds and analytical/integration/real-artifact validation. Explicitly
retains external compatibility and distinguishes planned work from implemented
behavior. Added the spec index entry. Documentation-only change; no code or
scientific results changed and no audit/model run performed. Validation: local
Markdown links/fences and whitespace checks (see this change's handoff).


### 2026-11-03 — Swiss audit specification ready for implementation

Re-reviewed [SPEC-0018](../specs/0018-swiss-audit.md) against the actual external
audit parser, CLI, options, truth resolution, calculations and renderer. Corrected
the draft's prediction-GO validation and extra-column claims: preserve presence
masks and acceptance of additional unique headers. Defined exact summary keys,
filter variants, explicit-truth precedence, empty-cohort behavior, output protection
and publication limitations. Specified a private shared renderer with unchanged
external defaults and Swiss title/prefix, plus implementation sequence and
regression acceptance checks. Marked the spec/index ready for implementation;
no code implementation or scientific execution is claimed. Documentation links,
Markdown fences, headings and whitespace checks passed.


### 2026-11-03 — Swiss audit implemented and validated

Implemented [SPEC-0018](../specs/0018-swiss-audit.md): shared private renderer
with external compatibility wrapper; Swiss API, Typer command and executable
`scripts/swiss-audit.py`; installed `pclass-dark swiss-audit`; script distribution
entry. Shared calculations/parser remain unchanged. New defaults filter own-truth
GO:0005515; explicit truth precedes verified saved metadata; unfiltered Python
mode remains available. Added hard-link/source and destination symlink protection.
Updated README, Swiss guide/API, specs 0016/0018 and index.

Validation: external refactor regression 15 passed in 7.47 s; final specified
audit/evaluation/CLI suites **165 passed in 97.06 s**, log
`/tmp/swiss-audit-regression.log`; additional import-side-effect regression
**1 passed in 0.24 s**. The existing localhost fixture ran with socket permission.
Ruff lint/format, whitespace, documentation links/fences and installed help checks
passed. Temporary code-only wheel/sdist smoke build verified both new modules
and executable script inclusion (`/tmp/swiss-audit-packaging.log`).

Ran `/usr/bin/time -v python scripts/swiss-audit.py
work-swiss/swiss_evaluation.tsv` (bins 10, cutoff 0.7). Table SHA-256
`bfd8749640c56d022a0cf9a4bcec750056947be74b5c8a67aaf82a7abaef6476`;
verified saved truth from score
`0709f16e28b50387710b7f1d8e63b27802939a7147a008f3b89af589d900c9e2`.
Revision `3af0a2a6bae7906ec0b8efba56f43553a9c81b95` plus dirty implementation.
380,578 queries, 0 binding exclusions. PClass 374,085/380,578 hits (98.2939%);
Dark 234,274/380,568 scorable predictions (61.5590%); 10 Dark abstentions.
Both-hit 231,239, PClass-only 142,846, Dark-only 3,035. No distance >1.
No preprocessing fit or sampling/seeds. These are descriptive in-sample diagnostics
with possible self/training/homology matches, not held-out model estimates.

Wrote `work-swiss/swiss_audit_{match_scores,confidence,coverage,summary}_v2.png`
and `swiss_audit_summary_v2.tsv`; inspected all four images visually. Dense scatter
remains readable enough for comparison; no aggregation introduced. Full regular
bins remain displayed and tied evidence is preserved in coverage. Output sizes
and full counts are in SPEC-0018. CPU run: 9.63 s wall, peak RSS 271,640 KiB; AMD
Ryzen 9 6900HX/Linux x86_64, Python 3.14.6, NumPy 2.4.4, Matplotlib 3.11.0.
Logs `/tmp/swiss-audit-real.log`, `/tmp/swiss-audit-real-resources.log`. Home
Matplotlib config was unwritable; its temporary cache fallback rendered normally.
No inference, live annotation/graph lookup or GPU work. No remaining implementation
blocker; future quality evaluation still requires controlled held-out cohorts.


## 2026-11-06 — Script-only evaluation and audit interfaces

Implemented [SPEC-0019](../specs/0019-script-only-evaluation.md) on `issue#34`:
removed `external-audit`, `swiss-audit`, and `evaluate-swiss` registrations and
imports from the root CLI. Existing scripts, adapters and scientific Python APIs
remain available. Updated README/operational guides and marked the installed
interfaces in SPEC-0016–0018 as superseded. No scientific behavior changed.

Validation: `python -m pytest -q tests/test_cli.py tests/test_external_audit.py
tests/test_swiss_audit.py tests/test_swiss_evaluation.py` yielded 107 passed and
one sandbox-blocked loopback fixture setup. Reran
`tests/test_swiss_evaluation.py::test_spawned_pclass_worker_splits_and_reuses`
with loopback permission: 1 passed. After adding script help subprocess checks,
`python -m pytest -q tests/test_cli.py`: 25 passed. Ruff lint/format and
`git diff --check` passed. Script-only use requires access to repository/source
scripts and the installed package; wheel installation does not install those
scripts as commands. No remaining validation blocker.

## 2026-11-06 — ProStruct reranking feature feasibility

Inspected ProStruct's actual sequence/residue inference, trained dual-head
artifact, 19-feature aggregation, profiling and signature code. Recorded the
assessment and proposed workflow in
[ProStruct reranking feasibility](prostruct-reranking-feasibility.md).
Implementation is feasible, but pooled Dark embedding caches cannot supply the
required residue embeddings, and Dark's terminal 1022-residue windows differ
from ProStruct's full-coverage 1021-residue windows. Preserve all Dark memberships
when joining reusable ProStruct predictions/features by exact sequence.

Validated the existing dual-head library's files/tensors and finite synthetic
CPU outputs. Reference feature/extraction/batched-ESM suites: 49 passed; one
existing AMD environment warning. GPU unavailable. No production inference,
reranking implementation or performance experiment. Next step: specify the
whole-protein versus candidate-window feature unit and leakage-controlled
reranking protocol before implementation. Current B/C inference is unchanged.


## 2026-11-06 — Portable frozen ProStruct structural heads

Implemented [SPEC-0020](../specs/0020-structural-prediction-models.md), extending
`build-library` with an independent checksum-verified RSA/ALL-CEF + Q3 model
companion, including on reused/scoped builds. `--clean` remains deletion-only.
Imported the unmodified dual-head weights, task-specific normalization and
training provenance into package resources; wheel/sdist include the model bundle.
No neighboring checkout is a runtime dependency. Core model preparation does not
load torch/ESM or download models. Added full-sequence `StructuralPredictor`,
`StructuralHeads`, exact ProStruct windows/ownership, masking and bounded CPU/GPU
execution. Shared pooled embeddings retain their existing mean semantics; a new
residue-state operation serves the structural heads. README, data/API guide,
SPEC-0004/0006, specification index and model provenance are synchronized.

Code revision: `f85c74842878af7f5d0ef2af3cb9ec09a2d77b3e` plus these uncommitted
changes, branch `issue#34`. Source ProStruct revision
`fa8fa0da7a9f2ecb1ab096f979b3936dd2409d54`; source library ID and imported bundle
ID/SHA-256 are recorded in SPEC-0020. Existing feasibility note and prior progress
entries preserved. The selected preparation contains 380,578 distinct FASTA
representative accession headers (independently counted), 381,627 memberships
and 3,443 retained clusters. Original accessions and all retained cluster
relations remain in aliases.tsv/memberships.tsv; no data export was rewritten.

Validation and resolved failures:

- First focused run: 76 passed, 7 setup errors from a prototype referring to
  `FrozenEmbedder` instead of this repository's actual `EsmBackend`. Corrected
  before completion. A subsequent broad run had 210 passes and two test-fixture
  failures (missing test Path import; old SimpleNamespace stub where the public
  build now returns an extended dataclass). Both fixed; no production test
  failures remain unresolved.
- `python -m pytest -q tests/test_structure.py tests/test_embeddings.py
  tests/test_cli.py tests/test_production_library.py tests/test_data_cleanup.py
  tests/test_library_cleanup.py tests/test_shared_distribution.py
  tests/test_shared_paths.py`: 210 passed, two fixture failures as above.
- `python -m pytest -q tests/test_structure.py tests/test_library.py
  tests/test_library_paths.py`: 68 passed (before additional device/failure tests).
- Final `python -m pytest -q tests/test_shared_paths.py tests/test_structure.py
  tests/test_cli.py`: 78 passed, 1 GPU test skipped. This includes all corrected
  fixture checks, CPU numerical tests, model-copy failure atomicity, preservation
  of valid production on a corrupt companion, clean non-creation and standalone
  wheel-from-sdist artifact loading away from either source checkout.
- Explicit copy-failure/production-preservation checks also passed separately
  (2 passed). Ruff lint/format and `git diff --check` passed.
- Real pinned ESM CPU/offline queries `ACDEKRHX` and
  `MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEX`; original ProStruct
  enrichment.write_accession versus the new API on the latter: every Q3 label
  matched, maximum exposure difference 4.930724994967051e-10 (TSV rounding).
  Four numerical threads; no network, fitting, sampling or splits. This is an
  implementation agreement check, not held-out biological performance.

Validation environment: Linux x86_64, Python 3.14.6, NumPy 2.4.4,
PyTorch 2.9.1+rocm6.3, Transformers 5.13.1, Safetensors 0.8.0.
PyTorch exposes no GPU here; actual GPU comparison was skipped, while mocked
initialization-fallback tests passed. An existing AMD_SERIALIZE_KERNEL warning
and the existing ESM loader's unused LM-head/pooler report appeared. Residue
states do not use the pooler. No full-reference prediction or 19-feature table,
rerank command, fitted reranking score, throughput benchmark or manuscript change.
Next: aggregate the descriptors with matched reference/query policies, then fit
and evaluate reranking using training-only profiles and homology-controlled
held-out data. No additional pretrained model is needed for those descriptors.


## 2026-11-06 — Draft feature-table extraction for the first ML experiment

Drafted [SPEC-0021](../specs/0021-extract-structural-features.md) for requested
`scripts/extract_features.py` and importable aggregation/orchestration modules.
Rechecked current preparation joins, SPEC-0020 and ProStruct's actual 19-feature
formulas. The script will resolve filtered unique representative accessions and
full sequences from one verified preparation, run full-coverage structural
windowing/residue embeddings/heads, and aggregate once per sequence hash.

The planned main TSV has accession_id first, 19 ordered numeric features, and
cluster_id last. Preserve every retained cluster membership: the current 380,578
unique sequences imply 381,627 labeled rows, with repeated accession IDs for
multi-cluster sequences. Retain an identity sidecar for leakage-safe grouping.
Formulas follow ProStruct; deliberate export deviation is .17g feature precision
instead of its three-decimal quantization for ML. Spec includes missingness/QC,
model/checkpoint provenance, bounded workers/queues, checksum-bound resume,
rollback/publication contracts and analytical/integration acceptance tests.

Future training/evaluation must keep identical/homologous sequences together,
fit preprocessing on training data, and evaluate ordering of held-out accepted
infer candidates rather than claiming reranking from classifier accuracy alone.
No script/code implementation, feature extraction, ML training or production
experiment was performed. Validation: checked 19-feature order against ProStruct's
source constant, table column count, local Markdown links and whitespace. Index
updated; implementation tests remain planned.


## 2026-11-06 — Clarify automatic experimental extraction inputs and cwd outputs

Updated draft [SPEC-0021](../specs/0021-extract-structural-features.md) following
user clarification: a zero-argument script invocation automatically resolves the
selected preparation, filtered/deduplicated representative accessions and their
full sequences. The experimental script writes features.tsv and all report/work
companions in the launch working directory, with no CLI output-path option.
Keep the explicit output parameter for Python integrations. Extraction remains
manually invoked experimental work, outside root subcommands and ordinary
setup-data/build-library/infer execution. Added input-discovery/cwd acceptance
checks. Documentation/whitespace checked; no implementation or extraction run.


## 2026-11-06 — SPEC-0021 implementation-readiness revision

Revised [SPEC-0021](../specs/0021-extract-structural-features.md) to implementation-
ready, preserving its experimental scope, automatic selected-cohort discovery
and launch-cwd features.tsv. No implementation or extraction run was requested
or performed in this revision. This supersedes the earlier draft's unspecified
resume/cache/publication details; earlier decisions remain in this log.

Read current artifact guards, path fallback, prepared validation/FASTA, aliases,
structural bundle and predictor code. Spec now defines precise input resolution
without requiring pooled embeddings, a streaming checksum/identity/join validator
instead of the eager verify_prepared path, excluded-alias/representative rules,
concrete API result/QC types, all companion schemas, SQLite snapshot/receipt
transactions and logical snapshot hashes, worker/thread/byte admission, runtime
fingerprints, existing-state decisions and journaled output recovery. Added a
public StructuralPredictor.prepare metadata extension as an explicit implementation
task. Clarified float64 comparison of raw exposure to 0.05 (float32 rounding matters),
full-precision export, exact SCD algorithm and an independently derived DEKRH
feature oracle. Full reference/ML/reranking runs remain separate experimental work.

Validation: independently checked 19-feature order and KD constants against the
ProStruct source, analytical fixture counts/charge/hydropathy, float32 threshold
behavior, 21-column contract, local Markdown links, fenced-code consistency and
whitespace. Implementation acceptance gates and real-model/GPU pilot checks are
specified but not executed; no passing implementation tests are claimed.

## 2026-11-06 — Implement experimental selected-cohort feature extraction

Implemented [SPEC-0021](../specs/0021-extract-structural-features.md): executable
scripts/extract_features.py, pure structural_features aggregation and importable
feature_extraction API, with supporting feature_store and feature_reports modules.
The zero-argument script resolves the selected preparation automatically and
writes features.tsv plus QC/identity/provenance and hidden work state in the launch
cwd. It is experimental, outside the root CLI and normal setup/build/infer flow.
No preparation, signature or production selector is changed.

The 21-column table preserves every retained sequence/cluster membership while
computing each full-sequence feature vector once. All 19 definitions follow
ProStruct, with explicit float64 threshold comparison, FFT/direct SCD and .17g
serialization. Added StructuralPredictor.prepare for public runtime metadata;
kept model weights, prediction behavior and mean-pooled inference unchanged.
The workflow uses streaming verified input joins, an immutable logical SQLite
snapshot, checksum-bound feature receipts, bounded spawned CPU aggregation,
explicit device/thread compatibility and journaled multi-file publication/recovery.
Updated source packaging, README, operational guide, SPEC-0020/0021 and index.

Code revision: 5f4facf41a9ce541e3ad55fc6bc5cd860aa6ed25 plus this implementation,
branch issue#34. Exact scientific source-code hashes and runtime/model IDs are
preserved in the pilot manifests. Existing draft/readiness history above remains
intact; implementation details supersede those earlier statuses.

Validation:

- First focused analytical/integration run: 22 passed. Initial expanded
  structure/embedding/CLI run: 119 passed, one GPU skip.
- Development checks caught a SQL-whitespace regression introduced while wrapping
  long literals, a replacement-indexing resume edge, and test-fixture issues
  (missing helper import and fault injection triggering the staged marker write
  instead of publication). Corrected and reran the affected checks; no unresolved
  test failure remains. Broader interim runs: 141, then 183, then 187 passes.
- Final command: python -m pytest -q tests/test_structural_features.py
  tests/test_feature_extraction.py tests/test_structure.py tests/test_embeddings.py
  tests/test_cli.py tests/test_setup_data.py tests/test_reference_data.py
  tests/test_shared_paths.py tests/test_shared_distribution.py.
  Result: **193 passed, 1 skipped in 38.66 s**. Log:
  /tmp/pclass-dark-extract-features-tests-final.log. The all-unknown subprocess
  test was then strengthened to block torch/transformers/safetensors imports during
  both extraction and complete-report reuse; its targeted rerun passed (1 passed).
- Numerical oracles cover all features, threshold equality/float32 promotion,
  unknowns, no-surface cases, invalid inputs and direct O(L^2) SCD comparison.
  Independent full-precision ProStruct summarize parity: 15 fixtures, seed 21;
  all defined values agree at atol/rtol 1e-12, with matching NA values.
- Integration/fault checks cover multi-cluster/excluded representative joins,
  automatic cwd discovery, snapshot/receipt/source/code corruption, indexing and
  execution fingerprint resume, CPU serial/spawn equality, workers without torch,
  payload bounds/oversized serial tasks, each publication swap, recovery after
  interrupted rollback, third-party modification, old-report preservation, lazy
  imports, CLI exits and source/wheel packaging.
- Ruff lint/format, local documentation/content checks and git diff --check pass.
  GPU tests were skipped because PyTorch exposes no GPU; one existing
  AMD_SERIALIZE_KERNEL environment warning remains.

Real-model CPU pilot and durable artifacts:
[record](notes/2026-11-06-extract-features-pilot.md),
[table/metadata](notes/2026-11-06-extract-features-pilot/).
Synthetic six-accession source, min_clusters=2, seed=42, prepare_only=True:
preparation 5e16555f574ce68c71a573e6499d3717e1da738c64068ee1379120aa07a9f175.
Four unique sequences, five labeled rows, two retained synthetic clusters,
including a representative originating in an excluded cluster and one all-unknown
sequence. Ran the script with --device cpu --workers 1 --window-batch-size 1
--offline from /tmp/pclass-dark-feature-pilot-20261106-595td09r/results;
complete-report --resume also passed without model initialization.
Extraction d9ae16e8ba255581d7ffa3c3c4ed44b6652775bae36d1f5de0ecff92ba61bbfa;
feature TSV SHA-256
7669a2d8e9329270e0c8001d2f69b0e4d8ce7c511458a7fb1b02c3beab23ef75.
Serial external timing: 5.86 s, peak RSS 1,465,432 KiB, CPU numerical threads 8.
A second real run with --workers 2 (two spawn workers, one inference thread)
produced a byte-identical main feature table; extraction
52f75bfce8d7c6b094245c5c37380fff8e86ca91c4c64a584ceb923b348707bd.
Earlier intermediate-code pilots produced the same table; transient artifacts
were retained, and the committed pilot records identify the final code hash set.
These are correctness checks with synthetic labels, not held-out performance.

No network downloads, full-reference extraction, classifier training, rerank
command or ranking-performance experiment. No manuscript changes. Next: explicitly
run reference extraction in an experiment directory, then specify homology-aware
train/validation/test construction and candidate reranking evaluation.


## 2026-11-07 — Three-decimal feature table export

- Updated [SPEC-0021](../specs/0021-extract-structural-features.md), extractor/API
  report serialization, README and feature extraction guide. Export uses exactly
  three decimal places, `0.000` for rounded negative zero, and `NA` for undefined
  values. Full-precision calculations and cached receipts remain unchanged.
  This supersedes the earlier full-precision TSV export decision.
- Export-only conversion of `work/features.tsv` used
  `python /tmp/round_pclass_features.py` (stream the validated original table through
  `feature_reports.write_table`, update serialization provenance/file checksum,
  then `publish_family` under `FileLock`). No inference or random computation;
  CPU serialization only. Starting revision: `5f4facf4` with working-tree changes.
  All 381,627 rows, 380,578 sequences and 3,443 clusters retained. Original report
  family: `work/.features.features-archives/20261107T125159Z-fba0c9a18dd84be6900bd9637c16cb73/`.
  Source table SHA-256:
  `49c2d0054db7ef7000d8cc0a485562000e21e5ea7b56377cca8455f21e5510f4`.
  Converted table: 51,898,567 bytes, SHA-256
  `96351fca885b3831b86aec3ba020c215a3ef8be66dd41fc8e44c42cf1101bf9c`.
  Manifest checksums and serialization provenance updated and complete-family
  verification passed. QC/index files were copied unchanged; full-precision
  receipt cache and original computation identity preserved.
- Validation: `python -m pytest tests/test_feature_extraction.py
  tests/test_structural_features.py tests/test_structure.py -q`: **115 passed,
  1 skipped** (GPU unavailable). Tests cover rounded integration output,
  full-precision receipt checks, NA, signed zero, positive/negative rounding,
  identifier/integer preservation, legacy numeric parsing, serial/parallel and
  publication/resume behavior. Ruff checks/formatting and `git diff --check` passed.
- No classifier training or predictive accuracy claim. Next: implement the planned
  additive classifier with similarity-informed soft targets; changing extraction
  code/precision does not make historical work compatible with a new resume contract.


## 2026-11-07 — Resume complete feature reports after cache removal

- Fixed complete-report `--resume` after deletion of the hidden work directory.
  Previously it created an empty receipt database and then failed receipt-dependent
  verification, leaving unusable work state. It now returns the independently
  verified report after contract/device checks, before creating new work state.
  Existing work directories retain snapshot and receipt integrity checks.
- Added a regression that removes work state and resumes twice, requiring identical
  report bytes, no recreated work directory/snapshot and no predictor initialization.
  Confirmed the test failed before the fix.
- Updated [SPEC-0021](../specs/0021-extract-structural-features.md) and the
  [extraction guide](../feature-extraction.md). Validation:
  `python -m pytest tests/test_feature_extraction.py tests/test_structural_features.py
  tests/test_structure.py -q`: **116 passed, 1 skipped** (GPU unavailable).
  Ruff checks/formatting and `git diff --check` passed. No production data changed.


## 2026-11-07 — Draft additive-classifier exploration specification

- Added [SPEC-0022](../specs/0022-explore-ml.md) for planned
  `scripts/explore_ml.py`, equivalent library/CLI operations and the README-selected
  additive classifier. Inputs exclude sequence_charge_decoration (18 predictors).
  Synchronized the README next-step note and specification index.
- Defined exact hit rate (accuracy/micro recall), mean similarity-weighted hit
  rate, macro/size-stratified diagnostics, matched alpha=0 and constant baselines,
  training-only fitting, five-fold homology-grouped evaluation and optional repeated
  CV for per-protein prediction stability. Fold score SD is not a confidence interval.
- Documented previously observed local input facts: 381,627 membership rows,
  380,578 accessions, 3,443 feature labels; similarity input covers 3,524 clusters.
  The 1,039 accessions with conflicting labels require explicit resolution before
  a single-label run; default validation fails, explicit exclusion is audited.
  No input table changed and no new predictive experiment ran.
- Consulted actual ProFunc medoid AMB Lin implementation and official scikit-learn /
  PyTorch documentation for grouped CV, splines and soft targets. Proposed defaults
  are unvalidated; implementation, automated tests and full training remain pending.
- Documentation validation: local Markdown links, 18-feature input list, metric and
  target analytical examples, and `git diff --check` checked. No code tests required
  for this specification-only change.


## 2026-11-07 — SIAC exploration specification made implementation-ready

- Revised [SPEC-0022](../specs/0022-explore-ml.md) through a contract/readiness
  review. Kept implementation and empirical validation explicitly pending; updated
  the index and README link wording to reflect specification readiness.
- Fixed cohort identity/conflict ordering, exact alias exclusions, complete
  retained-label similarity coverage and duplicate detection across extra labels.
  Defined all outer/inner split eligibility checks and the deterministic seed
  derivation, training-only spline preprocessing, sparse soft targets, optimizer,
  epoch selection/refit behavior and canonical class-specific softmax curve gauge.
- Specified metric denominators, baseline/top-k conventions, repeated prediction
  stability formulas, exact TSV/JSON/model-array schemas, API/CLI options, optional
  dependencies, bounded CPU/GPU resource admission and task-level atomic resume.
  Removed the unsupported promise to quantify attribution stability without a
  fixed comparator protocol; prediction stability and curve export remain in scope.
- Documentation validation: local links, source feature ordering (18 inputs), all
  tabular schema headers, complete example options, fenced blocks and readiness
  markers checked. `git diff --check` passed. Consulted official sklearn spline /
  grouped-split and PyTorch soft-target documentation; verified installed spline
  construction/constant extrapolation and grouped split APIs on tiny arrays.
  Independent analytical checks passed for target weights, metric credit and
  preservation of pairwise margins by the class-mean softmax gauge.
- No source implementation, production input edits, model training, CV performance
  measurements or automated implementation acceptance tests were performed.
  Next: implement against the fixed specification, then execute its acceptance
  suite and a bounded synthetic pilot before full-data experimental work.


## 2026-11-07 — Implement SIAC five-fold exploration

- Implemented [SPEC-0022](../specs/0022-explore-ml.md) through
  `scripts/explore_ml.py`, `pclass-dark explore-ml` and reusable Python modules.
  SIAC uses 18 features (SCD excluded), cluster-specific centered cubic splines,
  similarity-informed soft targets and deterministic mini-batch AdamW. Added
  ordinary-target additive and two constant controls on matched folds.
- Added strict cohort/alias/conflict handling, complete retained-label similarity
  alignment, homology-grouped or explicitly exploratory accession splits, inner
  epoch selection and outer refit, unsupported-label accounting, exact/semantic
  hit rates, macro/size diagnostics, optional repeated prediction stability,
  margin explanations, plain-array model reload and streamed reports. Source
  inputs and existing production inference remain unchanged.
- Added validation-only audits, bounded CPU spawn workers with shared mmap numerical
  arrays, pre-fit GPU probing/CPU fallback, memory admission, immutable task-level
  checkpoints, checksum verification and recovery after interrupted report
  publication. Rejected resumes preserve existing input arrays/audits; foreign
  stage files are preserved and refused. Added optional `ml` dependencies,
  packaging entries, [usage/API guide](../explore-ml.md), README and index updates.
- Implemented refinements are recorded in the specification: conservative
  coordinator high-water RSS replaces a separate worker-memory probe; expanded
  bases are cached only for the active fit; `fit_siac` returns model plus epoch
  records, metric helpers return dictionaries; aggregate similarity sums permit
  1e-8 absolute summation-order tolerance. Float32 is explicit even if callers
  change the Torch default dtype. GPU numerical validation remains unavailable.
- Automated validation:
  - Final `python -m pytest tests/test_explore_ml.py -q`: **24 passed, 1 skipped**
    (GPU unavailable), including analytical gradients/metrics, held-out additive
    recovery, train-only splines, serial/parallel consistency, repeated stability,
    unsupported labels, identity exclusions, model reload/schema validation,
    interrupted/incompatible/corrupt resume and publication, input-inode preservation,
    foreign-stage protection, CLI/API/script help, lazy imports and dtype isolation.
  - Broader feature/structure/CLI selection: **164 passed, 2 skipped** before the
    last dtype regression; final focused SIAC run above rechecked modified ML code.
  - `python -m pytest -q`: **760 passed, 4 skipped, 1 setup error**, 765.87 s.
    Existing `test_spawned_pclass_worker_splits_and_reuses` could not create its
    127.0.0.1 HTTP fixture inside the restricted sandbox. Reran only that test with
    loopback permission: **1 passed in 2.54 s**. Do not report the original run as
    an uninterrupted all-passing suite. One preexisting AMD_SERIALIZE_KERNEL
    environment warning appeared in an embedding test.
  - Ruff checks/formatting, `git diff --check`, local documentation links, feature
    order and pilot input hashes checked. Built source-only wheel and sdist in
    `/tmp/pclass-dark-siac-package-otizj7mw/`, outputs
    `/tmp/pclass-dark-siac-dist/`, log `/tmp/pclass-dark-siac-build-final.log`.
    Verified script/guide/modules/`ml` extra inclusion and installed-wheel command
    help from `/tmp/pclass-dark-siac-installed/`. Initial checkout sdist build was
    stopped when it began copying selected multi-GB reference data; removed only
    its newly generated staging tree. No package was published.
- Ran the bounded [reproducible CPU synthetic pilot](notes/2026-11-07-siac-pilot.md):
  90 artificial examples, three labels, seed 42, five folds, 20-epoch cap, patience 5,
  learning rate 0.05, batch 64, one worker/thread, no homology claim. Retained inputs,
  exact versions/code hashes and output metrics under
  `docs/research/notes/2026-11-07-siac-pilot/`; full ephemeral results under
  `/tmp/pclass-dark-siac-pilot-20261107/results-final/`. Experiment
  `5e1a1cdc0914d3e856b038fcf3d0c983b5f07e1766912991da1dd54537879aed`.
  SIAC and hard-target control both recover the trivial rule (exact/weighted=1);
  constants exact=1/3, weighted=0.7333. This is software validation, not evidence
  of protein performance or a smoothing benefit. Starting Git revision 7ca58d8f
  plus this working-tree implementation; full scientific source hashes retained.
- No full-reference experiment or deployment fit ran. Next: resolve ambiguous
  reference memberships or explicitly choose audited exclusion, supply documented
  homology groups for biological generalization, then run the five-fold experiment.


## 2026-11-07 — Rename SIAC exploration script

- Renamed `scripts/explore_ml.py` to `scripts/explore-ml.py`; updated README,
  usage guide, SPEC-0022, pilot reproduction command, sdist manifest and script
  integration test. Historical entries above retain the original filename.
  The importable `pclass_dark.explore_ml` module and `pclass-dark explore-ml`
  command keep their existing names. Scientific implementation and saved pilot
  code identities are unchanged.
- Validation: renamed-script help/lazy-import regression and CLI tests passed;
  manifest target and documentation references checked; Ruff and whitespace
  checks passed. No model training or input-table changes.


## 2026-11-07 — Default SIAC to random stratified five-fold splits

- At the user's explicit request, removed the default requirement for homology
  groups/provenance. API and CLI now default to accession mode with seeded shuffled
  StratifiedKFold for outer and inner splits. Exact sequence aliases still collapse
  before splitting; optional explicitly requested homology mode retains its grouped
  splitter. Reports continue to mark random evaluation as homology uncontrolled.
- Updated SPEC-0022, README, usage guide, index and historical pilot interpretation.
  The prior homology-default decision is superseded for this exploration. Conflicting
  label handling is unchanged; existing input tables are not modified. Old cached
  experiments have different source/split contracts and need a fresh output directory.
- Validation: SIAC and CLI selection **50 passed, 1 skipped** (GPU unavailable).
  Regression covers default API/command execution without group files, independently
  checks seeded StratifiedKFold outer/inner assignments, class proportions and seed
  sensitivity. Existing explicit homology, serial/parallel, reports, stability and
  resume tests remain passing. Final changed regression rerun, Ruff checks and
  whitespace/documentation checks passed. No full-reference training performed.


## 2026-11-07 — Request eight SIAC workers by default

- API and CLI now default to eight requested workers, with allowed range 1..8.
  CPU admission caps actual workers by usable CPUs divided by per-worker threads,
  available fold tasks (5*repeats), and memory budget. Low-CPU machines clamp the
  requested default instead of failing validation. Single-repeat CV can run at
  most five fold jobs concurrently; repeated CV can use eight. GPU remains serial.
- Updated SPEC-0022 and usage guide; preserved the historical pilot's explicit
  single-worker reproduction setting. Serial test fixtures explicitly request one
  worker so failure injection and serial/parallel comparisons remain meaningful.
- Validation: SIAC/CLI tests **51 passed, 1 skipped** (GPU unavailable). New regression
  checks API/CLI default eight, two-CPU admission, five-task cap, eight workers for
  repeated CV, serial override and rejection above eight. Ruff formatting/checks,
  documentation references and `git diff --check` passed. No full-data training.


## 2026-11-07 — SIAC progress logging and membership explanation

- Added scoped timestamped stderr progress for source verification/snapshotting,
  feature/index and similarity scans, split/device/worker admission, fold/model
  selection/refit epochs/losses, early stopping, reused/committed tasks, aggregation
  and final publication. Scan/batch counters throttle to five seconds; spawned
  workers get matching handlers. CLI restores existing handlers/levels afterward;
  library imports do not configure logging. Persistent numerical epoch history
  remains training.tsv. Added ml_progress.py to scientific code provenance.
- Improved conflicting-target errors with the CLI exclusion option, conflict audit
  path and --resume retry guidance. No default exclusions, automatic target choices,
  numerical algorithm changes or input modifications. Updated SPEC-0022 and guide.
- Verified original Q2FH94 (cluster_0868) and Q2FYU1 (cluster_1372) sequences are
  exactly equal, length 207, SHA-256
  `000e1ae445249eb65e86ecc55aa05066c28e936c33901f41d9d88c833448a7de`.
  Source: `/home/jmao/projects/profunc/datasets/20260911T184540Z-d7e84090016d/accession.fasta`.
  Prepared aliases collapse both to representative Q2FH94; memberships preserve
  both clusters. The feature writer joins both memberships to one feature vector.
  Sequence deduplication succeeded; the target ambiguity is unrelated to windows
  or export rounding. This does not establish either annotation is biologically wrong.
- Validation: `python -m pytest tests/test_explore_ml.py tests/test_cli.py -q -x`:
  **54 passed, 1 skipped** (GPU unavailable), including real spawned-process stderr,
  epoch/phase messages, repeated CLI handler cleanup, actionable conflict diagnostics,
  throttled counts and existing numerical/parallel/resume regressions. Ruff checks,
  formatting, documentation links and whitespace checks passed. No full-reference
  training performed.


## 2026-11-07 — Automatic identical-sequence prefilter with removal accounting

- At the user's request, exploration API/CLI now default to an audited identity
  prefilter: collapse identical sequences with one consistent target, and remove
  the entire ambiguous identity when different targets remain. Never choose an
  arbitrary cluster or equate rounded feature vectors with sequence identity.
  The sequence-index SHA-256 is used when available; accession-only fallback logs
  the lack of cross-accession verification. Explicit --conflict-policy error and
  the strict low-level load_cohort default remain available.
- Added population.prefilter_counts to preflight/summary reports, distinguishing
  original feature rows/accessions, unique identities, repeated membership rows,
  same-target rows collapsed, ambiguous identities/rows excluded and final count.
  Removed rows reconcile with retained examples and audit TSVs. Updated the guide,
  SPEC-0022 and README CAUTIONS; previous default-error policy is superseded.
- Validated against the actual feature table with load_cohort(features,
  sequence_index=features.sequences.tsv, conflict_policy='exclude') without fitting
  a model or modifying inputs. Preparation ID:
  `8e5994726cb958fce5d2396835f9a3767da897007110a9fda4f2e3b06cccbe4e`.
  Input: 381,627 membership rows, 380,578 accessions/sequence identities; 1,049
  rows beyond one per identity. Prefilter removes 1,039 ambiguous identities
  (2,088 source rows), collapses zero additional same-target rows, and retains
  **379,539 unique sequences across 3,443 labels**. The original selected reference
  manifest records 455,508 clustered accessions and 575,748 total FASTA records;
  prior exact-sequence deduplication plus cluster filtering already produced
  the 380,578 retained sequences. These populations are not interchangeable counts.
- Validation: `python -m pytest tests/test_explore_ml.py tests/test_cli.py -q -x`:
  **56 passed, 1 skipped** (GPU unavailable). Added same-target hash-alias collapse,
  whole ambiguous-hash exclusion, no feature-equality dedup without hash evidence,
  exact removal accounting, default CLI filtering/progress and summary denominators.
  Existing explicit-error and conflict-audit tests retained. Ruff formatting/checks,
  documentation links and whitespace checks passed. No full-data training ran.


## 2026-11-07 — Clarify SIAC training stages and shorten epoch progress

- Training messages now identify Repeat r/R, Fold f/5 and one of four numbered
  stages: SIAC epoch-selection (1/4), SIAC outer-refit (2/4), HARD epoch-selection
  (3/4), HARD outer-refit (4/4). START/DONE transitions and all periodic batch/epoch
  lines share the contextual label so concurrent workers remain distinguishable.
- Removed validation_loss and best_epoch fields from terminal epoch lines; they
  remain in training.tsv and still govern inner epoch selection. Numerical training,
  thresholds, data and saved report schemas are unchanged. Updated SPEC-0022 and
  the user guide with the exact log example and stage definitions.
- Validation: `python -m pytest tests/test_explore_ml.py tests/test_cli.py -q -x`:
  **56 passed, 1 skipped** (GPU unavailable). Logging regressions check all four
  stages, repeat/fold totals, spawned worker context, terminal validation-loss
  absence and saved validation history. Ruff checks/formatting and documentation
  links/whitespace checks passed. No full-reference training performed.

## 2026-11-07 — Summarize completed reference SIAC exploration

- Assessed the user's completed `work/explore-ml/` experiment
  `4683ddea21e305608fc8e81f38c2f71276d83d5e194fe2547890acec98db8163`.
  Verified contract identity and checksums of summary, fold metrics, per-cluster
  and stability reports. Added the [results note](notes/2026-11-07-siac-reference-results.md)
  with population, settings, provenance, limitations and reproducible report queries.
- Five random-stratified folds, one repeat, seed 42; 379,539 sequences, 3,443 labels,
  18 inputs, 1,039 ambiguous identities excluded. GPU cuda:0, one effective worker.
  SIAC alpha=.1, k=10, power=2; eight knots, max 100 epochs. Source hashes, versions
  and original dirty revision are recorded by the completed run manifest.
- SIAC exact/weighted/top-5: 59.48% / 70.76% / 77.16%; ordinary-target control:
  60.04% / 71.15% / 77.70%. Control makes 2,139 more exact predictions and wins both
  primary metrics in every fold. Preserve this negative result: the tested target
  smoothing did not improve held-out scores. Macro recalls 38.08% / 40.33%; small
  clusters perform substantially worse than overall accuracy suggests.
- Fold SDs are small (SIAC exact 0.193 percentage points, weighted 0.175), but are
  not confidence intervals. One repeat cannot measure per-protein prediction
  agreement; stability fields are NA. All ten learned models selected the cap of
  100 epochs, so convergence is not established. Homology is uncontrolled and
  supplied similarity generation provenance unverified: no remote-homology claim.
- No fitting, parameter selection, input changes or work-report changes in this
  assessment. Documentation content/link/formatting checks only; code tests are
  not needed. Follow-up options: repeated held-out predictions for stability and
  inner-validation-only assessment of longer fitting or revised smoothing settings.

## 2026-11-07 — Train all-reference model and expose query prediction

- Added [SPEC-0023](../specs/0023-final-ml-model.md), ml_model.py / CLI adapters,
  scripts/train-ml.py and scripts/predict-ml.py, root train-ml/predict-ml commands,
  package inclusion and [usage/API guide](../ml-model.md). Refits use every eligible
  CV example once, not concatenated repeated-CV folds or averaged spline weights.
  HARD is the default based on source CV; similarity-informed SIAC remains optional.
- Verified complete input snapshots, fitted full-cohort preprocessing and selected
  epoch budget from inner-selected fold counts. Portable plain-array bundle has
  strict checksums/schema, source/current code provenance, recipe identity,
  owned failure/restart behavior and no dependence on the CV folder for prediction.
  Query paths support exact 18-feature TSV, inline sequence and streamed FASTA,
  matching frozen full-coverage feature extraction and .3f export before saved
  preprocessing. Added top-five logits/scores, JSON explanations/flags and atomic
  output staging. Imports do not initialize models or parse arguments.
- Actually trained `work/ml-model`: all 379,539 sequences / 3,443 clusters, HARD,
  100 epochs, final seed 3525665611, GPU cuda:0 (AMD Radeon Graphics), 8 CPU threads.
  Fit/publication took 123.97 s after source validation; terminal log
  `work/ml-model-fit.log`, final mean training loss 1.777485 (not accuracy).
  Source experiment 4683ddea21e305608fc8e81f38c2f71276d83d5e194fe2547890acec98db8163.
  Model ID e24b23cbe6240e5d3519ff1a3efdff174b6f33e74fdb7037b88e7b5ebbdc16f9;
  weights SHA-256 d267fd34a7ee6394488573f3c76945ccbf23649a2fc892b15af0d05f8bf27789.
  Full reproducibility/settings and selected artifact copies:
  [final-model record](notes/2026-11-07-final-ml-model.md).
- Offline GPU query smoke on excluded ambiguous Q2FH94 (207 aa) reproduced all 18
  rounded reference values exactly. Predicted cluster_0868, score 0.367643; margin
  and additive explanation reconcile. This is pipeline validation, not resolution
  of the biological conflict or an independent accuracy estimate. Source CV HARD
  exact 60.04% / weighted 71.15% remain historical evidence only.
- Validation: final ML/model/CLI selection with GPU access **70 passed** (60.86 s);
  restricted run 68 passed / 2 skipped, followed by actual GPU checks. Tests cover
  full-cohort/repeat accounting, median/override epochs, inherited configs, strict
  source/bundle data, model reload/portable prediction, query schemas/rounding,
  raw-feature parity, late-error output atomicity, interrupted publication/restart,
  lazy imports, script/root CLI and existing exploration regressions.
- Existing balanced GPU toy test first failed the strict 1e-4 logit check. Diagnosed
  near-zero bias-gradient sensitivity to float32 reductions with AdamW eps=1e-8:
  max probability drift 0.0003477, no changed labels; nondegenerate case matched
  to 3.73e-9 logits. No optimizer/model changes made. Explicit balanced-case 5e-4
  score bound and ordinary-case 1e-4 checks now reflect measured behavior; initial
  gradient checked against analytical zero. SPEC-0022/0023 document the exception.
  No full CPU/GPU reference retraining-equivalence claim.
- Source-only wheel/sdist built and contents verified; installed-wheel help, real
  bundle loading and CPU prediction passed. Logs `/tmp/pclass-dark-final-ml-build.log`;
  no release published. Ruff/format/whitespace, doc links and final bundle/source
  hash checks passed. Existing source data/CV reports remain unchanged. Next:
  apply the model to actual queries and evaluate a separate held-out deployment set.

## 2026-11-07 — Interpret seq2 query in biophysical language

- Ran the existing predict-ml.py raw FASTA path on work/seq2.fasta (129 residues),
  offline cached GPU extraction, saved explanation to work/seq2-interpretation.jsonl
  and a readable report to work/seq2-interpretation.md. No code/model/input changes.
  Query SHA-256 eaf1f8d7f5bf78bbefe1100d912a09893361c57dda057f634adb5c5af6dfe79c;
  no exact match in final training aliases (homology not assessed).
- Model e24b23cbe6240e5d3519ff1a3efdff174b6f33e74fdb7037b88e7b5ebbdc16f9:
  cluster_1096 score 0.853623; runner-up cluster_2264 score 0.037153. Verified live
  descriptions against original preparation reference files and clusters_meta hash.
  Top annotation: peptidoglycan muralytic/hydrolase; runner-up NADH dehydrogenase.
- Comparative explanation: hydropathy -0.472 (+2.622 logits), length 129 (+1.066),
  surface acidic fraction .097 (+.787), buried basic fraction zero (+.595) support
  the top. Surface basic fraction .194 (-1.190), surface charged fraction .280
  (-.831), among other descriptors, favor the runner-up. Baseline +2.038 plus net
  feature evidence +1.097 yields +3.134 margin. All 18 features available/in support.
  Contributions explain top versus runner-up, not independent biological causes;
  scores are uncalibrated and this query is not a measured accuracy result.
- Verified explanation reconstruction within float32 tolerance, source-reference
  identity/metadata binding and report content/formatting. No training or automated
  implementation tests needed for this read-only query analysis/reporting task.

## 2026-11-07 — Revise seq2 interpretation to chosen-cluster support

- At the user's request, revised work/seq2-interpretation.md into an editable
  human-language template using each feature curve's contribution to cluster_1096,
  centered by its average across all 379,539 training examples. Replaced the
  terminology for the alternative with "second choice". No model or CLI/schema edits.
- Recomputed own-class values from the saved centered spline basis/weights:
  length +2.459, strand fraction +2.341, hydropathy +2.153, exposed fraction +1.359.
  Basic fraction among exposed residues .194 contributes +.144 (slight support),
  rather than the earlier -1.190 difference against the second choice. The only
  negative own-class contribution is surface charged fraction .280, -.190.
  Baseline 2.451 + net feature contributions 15.402 reconstructs logit 17.853.
- Retained chosen-only numerical evidence in
  work/seq2-chosen-cluster-support.json; preserved the original comparative JSON.
  The revised report explicitly distinguishes these two references and notes that
  a future predict-ml.py summary needs own-class fields instead of relabeling margins.
  Prior comparative findings remain valid for the original comparison; this report
  supersedes that presentation, not its underlying numerical evidence.
- Validation: reconstructed the top logit within float32 tolerance, checked all
  18 features and sign/direction in the template. Independently evaluated centered
  curves across the full saved training X in bounded batches to verify their mean
  contributions are within 1e-6 of zero. Markdown/whitespace checks passed.
  No training or implementation tests were needed for this analysis/report edit.


### 2026-11-07 — Shortened prediction interpretation reports

Implemented [SPEC-0024](../specs/0024-ml-interpretation.md) using the user's shortened
work/seq2-interpretation.md template. predict-ml.py / pclass-dark predict-ml now
support --format markdown: predicted function, brief explanation and all 18 signed
chosen-cluster feature supports relative to their training means. JSON adds own-class
terms without changing comparative fields. Optional labels are bound to the saved
preparation and reference hashes; unavailable metadata falls back to cluster IDs.

Validation: 77 passed, 2 GPU tests skipped across model/exploration/CLI/interpretation
suites; final wording check 6 passed; Ruff and whitespace checks passed. CPU offline
raw FASTA command: python scripts/predict-ml.py --model-dir work/ml-model --fasta
work/seq2.fasta --format markdown --output work/seq2-generated-interpretation.md
--device cpu. Its 18 contributions matched independently saved seq2 own-class terms
at 1e-7 tolerance. Chosen cluster_1096 score 0.854; basic surface-residue support
+0.144 and charged surface-residue support -0.190. Preserved the user's editable
report. Existing full-model bundle and weights unchanged; this is interpretation
validation, not additional held-out performance evidence. Usage in
[model guide](../ml-model.md). No retraining required.


### 2026-11-07 — Default prediction format

Changed predict-ml.py / pclass-dark predict-ml to emit the Markdown interpretation
by default, as requested. --output report.md saves it; stdout remains available.
Explicit --format tsv and --format jsonl preserve machine-readable alternatives.
Updated README, model guide and SPEC-0024. Regression tests verify default
Markdown on stdout and in a saved file, explicit TSV, and overwrite protection.
Validation: 21 passed in 5.44s. Ruff and git diff --check passed.


### 2026-11-07 — Automatic interpretation file publication

Clarified the requested default: Markdown must be saved automatically, not only
selected as stdout format. Input seq2.fasta now writes seq2-interpretation.md
beside the input; feature tables follow the same rule. Inline sequences use a
sanitized query ID in the current directory. --output overrides the path and
--stdout explicitly prints instead; TSV/JSONL retain stdout defaults. Existing
files are protected, publication is atomic, and the saved path is reported on
stderr. Updated SPEC-0024, README and model guide. Validation: 22 model and
interpretation tests passed, including automatic FASTA/TSV/inline naming,
collisions, stdout override and late failure protection; Ruff and diff checks
passed. This supersedes the preceding stdout-default behavior.


### 2026-11-07 — Top-five interpretation summary

Added a Top 5 predictions table to the default Markdown interpretation, showing
cluster IDs, descriptions capped at 24 displayed characters (including ellipsis)
and original softmax scores to three decimals. Preserves prediction order and
full function descriptions in the prose; missing labels show Unavailable.
Updated SPEC-0024 and model guide. Validation: 23 model/interpretation tests
passed; Ruff and git diff --check passed. The previous saved seq2 contribution
fixture is no longer present in work/, so no existing sample report was refreshed.


### 2026-11-07 — Terminal top-five predictions

Markdown file predictions now also print a per-query top-five table on stdout,
sharing the report formatter for cluster IDs, 24-character descriptions and
softmax scores. Summaries are staged on disk and displayed after successful
publication, preserving bounded memory and avoiding partial success on errors.
Explicit --stdout prints the full report once; TSV/JSONL behavior is unchanged.
Updated SPEC-0024, README and model guide. Validation: 23 model/interpretation
tests passed, including multi-query summary parity, default file output,
collision silence and late errors; Ruff and git diff --check passed.


### 2026-11-07 — Completed exploration CPU-fallback resume

Fixed completed-run resume to apply the same GPU memory admission and automatic
CPU fallback as initial preflight before comparing the recorded actual device.
Uses checksum-verified saved arrays through read-only memory maps, with no refit
or artifact mutation. A regression mocks GPU availability and insufficient memory,
trains on CPU, then verifies complete reuse and unchanged experiment checksums;
it also verifies that a newly admitted GPU still rejects a real device change.
Updated [SPEC-0022](../specs/0022-explore-ml.md). Validation: exploration suite
32 passed, 2 GPU tests skipped; model/interpretation suites 23 passed. Ruff and
git diff --check passed. GPU admission behavior exercised through mocks; actual
GPU execution was not tested in this environment.


### 2026-11-07 — Sparse-target memory admission

Corrected exploration _resources and final fitting _admit to include the full
sparse neighbor tables, simultaneous host copies and sparse batch-loss workspace.
A shared pure estimator accounts for int64 indices and float32 weights, active
labels/alpha, neighbor-width capping and per-worker copies, including hard-target
allocations. GPU estimates now include transferred tables and batch workspace,
so oversized auto requests fall back to CPU before fitting. Resource audits expose
the component byte estimates. Updated SPEC-0022 and SPEC-0023.

Validation: 58 passed, 2 GPU tests skipped across memory, exploration, final-model
and interpretation suites. New independent array-size and mocked 10,000-label
tests cover host refusal, worker reduction, GPU fallback and explicit refusal
without allocating large tables. Ruff and git diff --check passed. Actual GPU
execution was not exercised; no training algorithm or target semantics changed.


### 2026-11-07 — Bounded-memory semantic-constant control

Changed semantic-constant report scores to (train_counts @ S)[active], eliminating
the advanced-indexing copy of all active similarity columns in every fold worker.
The shared read-only similarity matrix remains available through its memory map;
only label-sized score vectors are created. Active-label restriction and stable
tie breaking are preserved. Updated [SPEC-0022](../specs/0022-explore-ml.md).

Validation: 60 passed, 2 GPU tests skipped across exploration, final-model, memory
and interpretation suites. Added read-only mmap regression rejecting dense column
selection and checking weighted-score winners, ties, and an inactive label with
the highest unrestricted score. Ruff and git diff --check passed. No new
performance experiment or actual GPU execution was performed.


### 2026-11-08 — Production integration and combined report contract

Recorded the user-approved scope in [SPEC-0025](../specs/0025-integrated-additive-evidence.md):
keep standalone ML scripts; exclude exploration from production; train within
build-library using shared frozen cluster preparation. Reports provide comprehensive
JSON and human-readable analysis. The opening paragraph combines PClass-Dark
classification, additive reranking and actual feature support, followed by detailed
PClass-Dark analysis and additive reranking analysis. Preserve acceptance/window
formulas and distinguish actual HARD/SIAC configurations. Integration is planned,
not implemented; explicit remaining interface/schema decisions are recorded.
Validation: documentation content/link and git diff --check checks only; no code
changed or new experimental performance measured.


### 2026-11-08 — Draft integrated reranking implementation specification

Drafted [SPEC-0026](../specs/0026-integrated-additive-implementation.md) from
SPEC-0025. Defines frozen reference/recipe ownership, direct full-data fitting
without exploration artifacts, structural-head → full-sequence feature → additive
fit dependency order, immutable library publication, accepted-candidate ranking,
availability policies and selected-class explanations. Defines comprehensive JSON,
opening human classification paragraph followed by detailed PClass-Dark/additive
analysis, terminal summaries and recoverable report-file publication. Preserves
standalone scripts. Proposed default HARD recipe reflects existing evidence;
configuration identity is explicit. Updated parent and index links.

Documentation-only draft; no integration implemented or performance measured.
Default frozen matrix source still requires actual provenance/import before
shipping. Validation: local documentation links and git diff --check.


### 2026-11-08 — Integrated reranking specification implementation readiness

Revised [SPEC-0026](../specs/0026-integrated-additive-implementation.md) against the
current setup/library/inference/fitting/structural code. Replaced unresolved source
and interface choices with concrete contracts: frozen six-column matrix import,
recipe validation, setup companion selection, library v2/selection v3 migration,
full-data fitting without exploration, structural-head/extraction dependencies,
shared backend ownership, candidate score/rank API, partial coverage/fallback enums,
selected-class explanations, report schemas and recoverable multi-file publication.
Defined modules, API arguments, CLI defaults/errors, cache/cleanup ownership,
dependency boundaries, schema fixtures and implementation acceptance gates.
Preserved standalone scripts and legacy scientific API. Updated parent and index.

Read-only data audit: work/cluster_dmatrix.tsv SHA-256
`d934cc904212c9e1d901c937ca818a7a6246386b5e51f97cb6962dc3faa9bd05`,
354315214 bytes, 6207526 rows, 3524 cluster IDs; every listed medoid matches the
prepared reference's bound clusters_meta.tsv. Recorded provided-export origin and
unknown historical generator honestly; full pair/score validation remains an
implementation gate. Verified actual parser schema and structural window/thread
caps, and NumPy/SciPy numerical inference dependency boundary. No matrix copied,
model trained, production code edited or performance improvement measured.

Documentation checks passed: local links, sequential sections, Markdown fences
and git diff --check. Specification is implementation-ready; implementation and
all integration acceptance tests remain pending.


### 2026-11-08 — Integrated additive workflow implemented and validated

Implemented [SPEC-0026](../specs/0026-integrated-additive-implementation.md): frozen
reference import in setup-data; explicit-reference final fitting with no exploration
prerequisite; library-owned portable model/head envelope; accepted-candidate
reranking; global top-five agreement; selected-class centered feature evidence;
JSON, human Markdown and terminal summaries; no-overwrite/recoverable publication.
Standalone scripts and legacy scientific APIs remain available. The CLI enables
the component by default; legacy Python setup/build APIs use explicit additive=True.

The initial recipe is HARD, reflecting the earlier comparison; SIAC remains an
explicit recipe. Preserve signature.v2 identities and use a separate integrated
manifest with library-selection.v3. Feature/model cache identities include the
scientific/execution contracts needed for valid reuse. Guard publication against
changed source inventories and selection snapshots; preserve both setup selectors
on import failure. Keep alias/conflict exclusion audits. Explicit operational limits:
no integrated force-in-place rebuild, and conservative retention of additive caches.

Frozen raw matrix SHA d934cc904212c9e1d901c937ca818a7a6246386b5e51f97cb6962dc3faa9bd05;
6207526 source pairs over 3524 clusters. The selected prepared cohort retains
3443 clusters and 5925403 pairs. Source historical generator is recorded unknown;
source medoid/member identity and all pair values/coverage were validated.

Code baseline 1ece3d5c, branch issue#42, uncommitted implementation. Full-suite
snapshot: 811 passed, 5 skipped, one sandbox localhost-socket error. Exact blocked
worker test passed outside sandbox (1 passed); local GPU agreement checks 3 passed.
Final core regression selection: 108 passed; final integration/ML/setup selection:
94 passed. Ruff, diff checks, local links and JSON syntax passed. Source-only
wheel/sdist and installed numerical/CLI checks passed; no model downloads for
validation. One packaging invocation was stopped after starting in repository
context; its generated sdist staging tree was removed and packaging repeated in
a clean temporary source-only project, avoiding multi-GB development data exports.

The [real GPU parity smoke](notes/2026-11-08-integrated-reranking-smoke.md) uses
existing work/ml-model weights, the existing read-only C/B signature library and
129-aa seq2 demonstration input. It accepts 37 clusters and reranks cluster_1096
from position 4 to 1, with feature evidence and global score 0.85362256. This is
runtime/interpretation validation, not independent accuracy evidence. Original
production selection and model weights were unchanged; no full-cohort refit was
started. Commands and component identities are in the linked note/JSON.

Usage: [integrated classification](../integrated-classification.md). Run the new
build-library workflow to fit and select a production integrated generation;
production fitting and independent homology-separated reranking evaluation remain
operator/research runs rather than automatic side effects of this code change.


### 2026-11-08 — ML root registrations retired; scripts/APIs preserved

Per explicit user request, removed explore-ml, train-ml and predict-ml imports and
registrations from the production root CLI. The three standalone scripts, their
Typer applications and Python scientific interfaces are unchanged. Updated README,
operational guides and SPEC-0022/0023/0024/0025/0026 plus index references. Earlier
progress entries retain their historical root-command descriptions; this entry
supersedes those interface statements.

Moved former root adapter checks to the actual standalone script applications.
Added regression checks for missing/rejected root commands and all three script
entry-point help paths. Validation: 94 passed, 2 GPU tests skipped across CLI,
exploration, final-model, interpretation and integrated inference/build suites;
Ruff, local documentation links and git diff --check passed. This interface-only
change does not alter model/data identities, numerical behavior or production
acceptance/reranking, and needs no retraining or migration.


### 2026-11-09 — Default additive build feature-cache admission repaired

Diagnosed the missing data/library/selection.json after the first production
rebuild. The integrated builder deliberately defers selection, but its call into
experimental extraction attempted data/library/additive-features/<id>/features.tsv,
which the extractor's default-library protection rejected. Reproduced the exact
protected-directory error without running extraction. Read-only validation verified
all 3443 saved production signatures, the production-bound relative companion and
structural-head bundle; no additive model or published library selector exists.
The earlier mocked extraction fixture did not cover default-root admission.

Implemented [SPEC-0026 section 22](../specs/0026-integrated-additive-implementation.md#22-managed-feature-cache-path-repair--2026-11-09):
private builder context permits only its exact managed feature family. Standalone
restrictions, setup/prepared/model/resource protections and link checks remain.
No scientific formulas, multiprocessing/device policy or artifact schemas changed.
Updated SPEC-0021, specification index and integrated operational guide, including
ordinary rerun recovery without --force. Earlier implementation validation entries
remain historical; this repair closes the untested real-extraction path.

Branch issue#42, baseline HEAD 1ece3d5c plus existing uncommitted implementation.
Regression data: synthetic 20-sequence/two-cluster library_data fixture; frozen
synthetic medoid similarity 0.2; HARD default recipe with epochs=2, batch_size=4,
seed=42; explicit CPU, workers=1. New tests run actual extraction/cache publication,
cohort loading, fitting, envelope publication, additive-required CLI inference and
JSON/Markdown completion receipts. Replace PLM inference with deterministic fake
predictions/states; query inference still executes the real frozen structural
heads. Test a fresh build and injected extractor failure followed by resume;
completed feature/model reuse leaves selection bytes unchanged. Initial query
fixture lacked residue-state/device support; extended only the fixture backend
and reran successfully. Existing extraction suite also exercises multiprocessing.

Validation: pytest -q tests/test_additive_build.py tests/test_feature_extraction.py
— 61 passed. Real existing production cache admission and standalone rejection
both passed without writing cache files. Broader command: pytest -q
 tests/test_integrated_inference.py tests/test_inference.py tests/test_cli.py
 tests/test_structural_features.py tests/test_structure.py
 tests/test_production_library.py — 164 passed (225 passed across both runs).
Ruff check and format checks, git diff --check, local documentation links and
Markdown fences passed.
CUDA unavailable (torch.cuda.is_available() false; nvidia-smi absent), so GPU
execution was not tested and no full-cohort feature extraction/refit was launched.
Production selector remains absent until an ordinary build-library rerun completes;
existing production artifacts, prior work/features and saved model weights were
not replaced. No commits or pushes. These checks establish runtime correctness,
not held-out classification performance.


### 2026-11-09 — Feature extraction progress counts clarified

Updated [SPEC-0021](../specs/0021-extract-structural-features.md) and extractor logs
per user feedback: periodic messages now read `Feature extraction complete:
1,000/380,578`. The numerator counts computed plus reused unique sequences; the
denominator comes from the sequence inventory, not cluster memberships. Both use
thousands separators. The final message retains computed/reused counts alongside
completed/total. Logging interval and scientific behavior are unchanged.

Validation: pytest -q tests/test_feature_extraction.py — 58 passed; newly added
1,001-sequence periodic-format regression — 1 passed. Tests cover fresh/resumed
counts and thousands separators using actual SQLite receipts and deterministic
CPU predictor fixtures. Ruff check/format and git diff --check passed. No production
job was restarted or artifacts changed; an already running process retains its
loaded logging code. No commit or push.


### 2026-11-10 — Recover completed features after a progress-only code change

Diagnosed an offline library build that cached all 380578 unique sequence features
but stopped before publication with Extraction code/dependencies changed during
the run. The saved contract differs only in feature_extraction.py; dependencies
and ten other module hashes match. Reversing only the newer progress formatting
and its inventory-count SELECT reproduces the saved source SHA exactly:
622223dd6288afcd583ece890ff82866e6f5fd790abb00393252dd2344fc1ea7.
Post-logging source SHA is 18cb0f5cddb55f0af6059abae634369d629de2e150161d8f4b8c5130c4a4b1d6.
The SQLite cache has 380578 sequences and 380578 receipts. No receipts/contracts
were edited, deleted or relabeled.

Added an explicit audited extractor whole-file compatibility allowlist. Its
canonical identity preserves the pre-logging numeric contract for the two reviewed
progress variants and the lifecycle-only compatibility shim; all unknown changes
remain strict. Publication records actual raw code hashes separately and drift
errors identify changed modules/packages. Updated SPEC-0021. Regression validation:
65 extractor/build tests passed, including known/unknown hash handling, compatible
in-run logging changes, real source drift rejection and all-receipt recovery with
prediction forbidden; 102 inference/CLI/model regressions passed. Ruff and diff
checks passed. The resulting contract ID and managed feature directory key match
the production d601dfa7a3e2a8164f214caee77d2316a10e464f02aae2e42d50d4eb36d76fd6
cache exactly.

Resumed python -m pclass_dark build-library --offline with local GPU access and
all existing artifacts retained. Log: work/build-library-recovery-2026-11-10.log.
Missing data/library/selection.json was a downstream effect of intentionally
unpublished incomplete additive stages, not an independent inference failure.
Production recovery/training and inference results will be recorded on completion.


Production recovery completed successfully (exit 0): all 380578 cached sequence
features reused, zero recomputed; 381627 membership rows yielded 379539 training
identities after excluding 1039 ambiguous identities (2088 rows). Full-data HARD
training completed 100 epochs on cuda:0 (18 features, 3443 clusters); this is a
production refit, not a new held-out performance measurement. Published
library-selection.v3 at data/library/selection.json and integrated generation
aad027d2d79db80e504fd3077912d18fe39e9271e4845ad56f3aa9c690e93c57,
with additive model f69a843cdbb3eafc2718c072dee4a692e7b37879ef884aa150257ece96479b83.
Command: python -m pclass_dark build-library --offline. Full log:
work/build-library-recovery-2026-11-10.log. Existing cached feature receipts and
previous outputs were preserved.


End-to-end verification succeeded (exit 0): python -m pclass_dark infer --fasta
work/seq2.fasta --offline --report-prefix work/seq2-recovered-2026-11-10.
All 3443 clusters evaluated, none unavailable; 37 PLM-accepted candidates.
Additive reranking selected accepted cluster_1096 (original rank 4; global
softmax 0.854). Reports: work/seq2-recovered-2026-11-10-classification.json and
work/seq2-recovered-2026-11-10-interpretation.md. Log:
work/infer-recovery-2026-11-10.log. This confirms selector publication and the
production inference path; the single query is not an accuracy evaluation.
Final Ruff and git diff --check passed; no source changed during recovery.
