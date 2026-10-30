# SPEC-0013: Explicit cleanup of obsolete dataset generations

- Date: 2026-10-30; technical revision: 2026-10-30.
- Status: **Implemented and validated (2026-10-30).** No project data was cleaned during implementation.
- CLI: `pclass-dark build-library --clean`.
- Related: [setup](0003-setup-data.md), [production](0004-build-library.md),
  [C/B distributions](0005-relative-score-distributions.md),
  [shared artifacts](0006-shared-artifacts.md),
  [migration](0011-readable-cluster-paths.md),
  [profunc inputs](0012-profunc-dependency.md).

## Purpose, authorization and limits

An ordinary build preserves data. `--clean` explicitly authorizes permanent
retirement of complete, unselected dataset generations in the resolved setup and
library roots, together with their exclusive derived results. Execute only after
validation of an existing full production/C/B pair; never build missing results. This is
not just disposable-array cleanup and can remove old published results. CLI help
must state that old preparations and exclusive results will be deleted; offer the
Python planning preview below. Do not add an interactive confirmation to the authorized command.

Keep the selected pipeline, current-preparation experiments, shared data,
resumable work. Retire recognized rollback archives after replacement validation. Do not infer obsolescence from
mtime, directory ordering, cluster-count differences, or differences from profunc.
Cleanup does not resolve profunc, download anything, load a model or refit scores.

A nonselected complete preparation is an obsolete-generation seed, unless
protected below. Different seeds/parameters also create generations; within the
two explicitly managed roots `--clean` retires those unselected old experiments
too if their lineage is proven. This is not a global claim of non-use. Users
needing an old experiment must keep its pipeline outside these roots, retain an
archive outside these roots, or omit `--clean`. No new pinning mechanism is introduced.

Initial implementation supports whole-generation deletion for the current
prepared/embedding/v2 production/research/relative schemas. Legacy six-fit and
unrecognized artifacts are retained, with known dependencies protected. Thus
cleanup can be conservative without making speculative legacy deletions.

## Observed implementation and why dependency adapters are necessary

Inspect `setup_data`, `embeddings`, `library_build`, `library_reports`,
`library_migration`, `library_cleanup`, `relative_scores`, `library_paths`,
`artifacts`, and `build_support` when implementing.

The inspected dataset has two prepared generations, two v2 production generations,
two paired C/B generations, and one shared embedding generation. C/B manifest
`identity_inputs.targets[*].prepared_id` distinguishes the two preparations; their
`embedding_id` is identical. V2 production identities contain final fit IDs but
no prepared ID. New fit identities may contain reference hashes, but older v2
fits do not; neither format includes a prepared ID. Never invent that field or
classify a production generation from an arbitrary signature alone.

Existing `clean_library_cache` and `_clean_fit` reclaim disposable numeric arrays
behind durable-result gates. Preserve their ordinary behavior. Do not call these
mutating helpers during preview or candidate validation.

## CLI and API contract

```bash
pclass-dark build-library --clean
```

The CLI exposes clean without a dry-run option. Paths and numerical settings use
internal defaults as documented in SPEC-0004. Python callers can preview with
`plan_cleanup(data_dir, library_dir)` and configure `build_library` explicitly.
Clean validates existing full selections and never builds. Reject `--cluster`,
`--force` and `--cross-validate` with clean. Missing, incomplete or mismatched
selections fail without building. Log validation/cleanup explicitly. Recover
pending cleanup before validation when a selection exists; snapshot checks protect
the handoff to execution.

Add `clean: bool = False` to `build_library`; validate it strictly, like
`cross_validate`. Append `cleanup: CleanupResult | None = None` to `LibraryResult`
after existing fields so positional callers remain compatible. Do not add a
union-return dry-run branch to this API. Python callers use the planning API for previews.

Implement `pclass_dark.data_cleanup` with these signatures:

```python
plan_cleanup(data_dir=None, library_dir=None) -> CleanupPlan
execute_cleanup(plan: CleanupPlan) -> CleanupResult
recover_cleanup(data_dir=None, library_dir=None) -> CleanupResult | None
```

Resolve defaults through `default_setup_input_dir` and
`default_library_output_dir`, exactly as building does. `None` is allowed only
at resolution; never implicitly clean a package resource tree or infer a second
writable root from an archive. Both roots must already exist, be disjoint regular
directories, and have no symlink components. Reject installed resource inputs and
`immutable=True` workarounds. Packaged data remain read-only; users can build from
it without clean or supply writable prepared data explicitly.

Planning acquires guards but makes no generation, quarantine, journal, selector
or report-file writes. Operational `.locks` files may be created. Execution
accepts only a `CleanupPlan` and re-plans under guards; it never trusts supplied
candidate paths as a deletion program. A caller may serialize plans for reporting,
but there is no public execute-from-JSON API.

CLI writes cleanup results to stdout; progress/errors to stderr. Include
artifact kind, root-relative path, reason, logical bytes, protected/skipped reasons,
and total byte estimates. No cleanup report file is created implicitly. Exit 0
for a successful empty/partially protected plan, 1 for validation/busy/recovery/IO
errors, and 130 for interruption. Protected entries are not errors unless they
make dependency analysis unsafe. Always show that LFS objects may retain storage.

## Immutable plan/result records

Use frozen dataclasses, tuple collections, primitive fields, and `to_dict()`
serialization. Export them from `pclass_dark.data_cleanup`; imports do no work.
Paths in records are strings; convert and validate them at operation boundaries.

- `CleanupEntry`: `root` (`setup|library`), `path` (POSIX relative directory),
  `kind`, `artifact_id`, `disposition` (`remove|protect`), `reason`,
  `dependencies` (sorted tuple of artifact keys), `tree_sha256` (str or None), `files`,
  `logical_bytes`, `allocated_bytes_estimate` (int or None).
- `CleanupPlan`: `schema=pclass-dark.cleanup-plan.v1`, canonical absolute
  `data_dir`, `library_dir`, `snapshot` (tuple of root/path/SHA-256 selector and
  validated manifest records), `entries`, `inventory_sha256`, `plan_id`.
- `CleanupResult`: `schema=pclass-dark.cleanup-result.v1`, `plan_id`,
  `status` (`complete|rolled_back|pending_recovery`), `removed` (entry keys),
  `protected` (entry keys/reasons), `failed` (entry key/message pairs),
  `logical_bytes_removed`, `allocated_bytes_estimate`, `recovery_required`.

An artifact key is `root + ':' + relative_path`. Sort entries by root/path and
all dependency lists lexicographically. Plan ID is `artifacts.identity` of all
serialized plan fields except `plan_id`. Do not include timestamps, PID or
elapsed time. Logical totals count each candidate file path once; allocated
estimates count candidate device/inode groups once, and count zero if observed
link count exceeds links being removed. They are estimates, not guaranteed freed
space. Result IDs/counters never imply that LFS objects were pruned.

For a candidate, `tree_sha256` binds the sorted exact directory/file inventory:
relative names, file SHA-256/size and lstat device/inode/mode/link-count/mtime_ns;
directories include names/device/inode/mode. No links or special files are allowed.
A file inventory record has exactly `path`, `sha256`, `size_bytes`, `device`,
`inode`, `mode`, `nlink`, `mtime_ns`, `blocks` (optional filesystem allocation); a directory record has `path`, `device`,
`inode`, `mode`. Paths are relative to the candidate, with `.` for its root
directory only; inventory collections are sorted by path and separate files from
directories. Spool detailed inventories to temporary disk during planning, keep
removal inventories in memory for execution, and store them durably
in the journal before moving data. Inventory digest binds these details and all retention/owner
edges, so changed content, added files or changed ownership invalidates a plan.
Missing/unclassifiable protected objects use `tree_sha256=None`; unknown objects
whose dependency scope is unbounded block planning as specified below.

Errors: `CleanupStalePlanError(IntegrityError)` for changed plan/snapshot;
`CleanupRecoveryRequiredError(RuntimeError)` for an existing cleanup journal;
`CleanupExecutionError(RuntimeError)` carries `.result` (CleanupResult or None
when planning failed before an executable plan existed) and `.production_result`
(the latter None outside build integration). Reuse `SetupBusyError` for contention
and `IntegrityError` for bad inputs/artifacts. On interruption retain durable
recovery state and re-raise `KeyboardInterrupt`; CLI prints recovery instructions.

## Validation and lineage rules

Validate current setup with `_read_inputs` semantics (selected hashes,
`verify_prepared`, `validate_sets`, `verify_embeddings`, shared inventory).
Validate the complete current production/C/B pair with `load_selected_library`
without target restriction. Its selector's preparation/embedding IDs must equal
setup's selection. Its sorted production cluster IDs must equal
`sets.cluster_ids`, not just have the same count. Validate optional research and
its exact production source hash. A malformed/mismatched selected artifact aborts
cleanup before any moves. Complete unsupported scientific targets are valid
outputs and do not by themselves fail the gate.

Adapters must distinguish required file access from historical provenance:

| Artifact/schema | Ownership and edges |
|---|---|
| `pclass-dark.prepared.v1` | Directory ID, manifest/config/reference identity and all files verified. Owns members, folds, both FASTAs, aliases, windows, C/B definitions. No external runtime dependency. |
| `pclass-dark.embeddings.v1` | Verify ID/contract/inventory and matrix/rows. `originating_preparation_id` is historical; do not pin its originating preparation merely for provenance. Selected and retained consumers protect the embedding ID. |
| `pclass-dark.library.v2` | Verify full bundle/fit payloads. Required edge to a paired C/B bundle for a retained pipeline. Determine prepared ownership from validated companion bindings or complete target receipts as below. Published files are self-contained; no required edge to fit checkpoint/source migration files. |
| `pclass-dark.relative-score-library.v1` / `relative-score-cluster.v1` | Verify source production path/ID/hash, target fit/payload/signature hashes, counts and array metadata/files. Targets' prepared/embedding IDs are ownership evidence; for retained pipelines protect their preparation and embedding as reproducible build inputs. Do not rewrite historical IDs. |
| `pclass-dark.library-research.v1` | Verify files, `identity_inputs.source_production` and top-level source binding using current loader. Required edges to source production and its retained pipeline. No edge to checkpoint copies after self-contained publication. |
| `pclass-dark.target-result.v2` | Verify result/relative receipt hashes, final-fit identity and payload identity. `prepared_id`/`embedding_id` bind checkpoint ownership, cross-checked against corresponding prepared C/B IDs and embedding matrix/row-map/representation hashes. Result and relative paths are required only while this receipt is retained for resume. |
| `pclass-dark.fit-inputs.v1` / `library-checkpoint.v1` | Verify fit ID, task/dependency/payload identity and complete scientific result. Associate by exact fit ID from verified production/target manifests, including all owners. Completed checkpoints without provable owners are protected. |
| `pclass-dark.embedding-shard.v1` | Verify row ranges, contract-derived embedding ID, files, shapes and finite arrays using a read-only equivalent of `_valid_shard`. Complete shards inherit the verified assembled embedding ID. No assembled generation means a partial resumable embedding job. |
| `pclass-dark.library.v1` / legacy checkpoints | Protected in v1 cleanup. Parse verified `identity_inputs.prepared_id` / `embedding_id` when available and retain those dependencies. No whole-legacy-generation deletion. |

**Production ownership:** a companion must cover exactly the production target
inventory, with each binding matching its persisted fit, payload and signature.
Every binding must agree on one preparation and embedding, both of which must
exist and validate with the corresponding C/B window hashes/counts and embedding
hashes. Alternatively, a full set of validated production target receipts proves
the same association. Every available source must agree. Mixed/missing/conflicting
ownership protects the production generation and known dependencies; if the
potential preparation/embedding set cannot be bounded, abort planning. Matching
C-window counts, reference-file hashes, filenames or numeric scores alone is
insufficient to establish ownership. Do not run signature discovery to prove it.

Current and retired normalized fit formats are both supported: new fit inputs
can include `reference`; older v2 fits can omit it. Verify their own identity
preimages as stored and join ownership through verified companion/target records.
Treat `source_provenance`, source manifest hashes and completed embedding origin
IDs as historical unless an actual supported reader opens the source. This
allows the selected embedding to survive while its old originating preparation
is retired. Never alter old scientific identities to simplify cleanup.

Read-only validators must not call `Checkpoints.__init__/read`, `_valid_shard`
or recovery-aware loaders that quarantine/repair bad files. Factor out pure
checks or implement pure adapters and exercise parity against existing validators.
Do not mark mutable worktree files immutable to bypass guards/recovery checks.

## Retention and retirement algorithm

1. Inventory known top-level managed namespaces and classify objects with the
   adapters above. Include complete, partial, archived and ambiguous objects.
2. Seed retention with current setup and library selectors and dependencies,
   current compatible research, all complete generations associated with the
   current preparation and all incomplete/resumable jobs.
   Selection itself is always retained. Preserve selected/full unsupported fits.
3. Validate `.archives/` read-only using recognized metadata and manifests.
   Resolve only recognized in-root layouts, with no traversal. With the full
   selected replacement validated, mark the archive namespace as removable
   rollback history and do not seed retention from its dependency references.
   Unknown/unreadable dependencies or unsafe trees block planning. Archives use
   exact tree inventories and the same quarantine/rollback machinery as datasets.
4. For partial work, retain its preparation/embedding IDs when explicitly bound.
   If only inventory/matrix/reference hashes exist, protect *all* compatible
   prepared/embedding owners; a partial unidentified job with unbounded owners
   blocks planning. A missing per-target receipt in an otherwise recognized run
   is partial work, not evidence of obsolescence. An old completed run remains
   complete if its disposable numeric arrays were intentionally cleaned.
5. Compute the transitive required-dependency retention closure. Candidate seeds
   are validated nonselected preparations outside that closure. Associate complete
   old production/research/relative/checkpoint generations with these seeds.
   Shared fits/embeddings are deleted only if every verified consumer is retired.
   A retained consumer wins, regardless of the artifact's original source ID.
6. Preserve unrecognized paths. Known unrelated files outside managed namespaces
   are reported and untouched. An unknown schema/corrupt artifact *inside* a
   namespace that can reference setup/library artifacts aborts the entire plan:
   there is no safe general dependency inference. Valid known artifacts with
   missing ownership but a bounded owner set protect that entire set instead.
7. Candidate tree contents must equal the schema-derived expected files plus
   known optional caches, receipts and empty structural directories. Extra user
   files, `.quarantine/` contents, symlinks and special files protect that owner
   and its required dependencies. Do not delete them as arbitrary descendants.
8. Iterate retention and shared-consumer checks to a fixed point. Select maximal
   nonoverlapping candidate directories; never include both ancestor and child.
   Preview each candidate and each protected reason. Empty eligible set succeeds.

Supported removable units: whole `prepared/<id>`, `embeddings/<id>`, completed
embedding `.checkpoints/<id>`, v2 production `<id>`, `relative-scores/<id>`,
`research/<id>`, `.production-checkpoints/<id>`, `.research-checkpoints/<id>`,
`.fits/<fit-id>`, and `.relative-checkpoints/<fit-id>`. Delete complete checkpoint
run directories only when every target is complete and retired; otherwise protect
the entire run and dependencies. Do not partially prune its completed targets.
`.migration/` caches and legacy `.checkpoints/` are protected in this first version;
their recognized source/recipe dependencies are retained. This conservative
boundary replaces the draft's speculative promise to delete every legacy cache.

Expected published bundle files are `manifest.json` plus its exact verified
`files` inventory. Fit root files are `identity.json` and the `final/` directory;
research/production run trees use their validated target receipts and cluster
layout. Within fit directories admit only `complete.json`, optional `source.json`,
`kernels.json` and receipt-bound `kernels.{C,B,V}.npy`, `prefix-NNNN.json` and
receipt-bound prefix arrays, and `profiles-NNNN.json` (NNNN in 0001–0640;
writer-generated chunk boundaries only). Target directories may additionally
have `background.json`, whose dependency/payload must validate against its
background ID. Relative checkpoint units contain `complete.json` and exactly its
recorded `complete.{C,B}.npy` arrays when supported. Missing already-disposed
kernel/prefix records are allowed; present records must validate. Filename
matching alone is never deletion authorization: validate
receipt task/dependency/file bindings and ownership. If a variant cannot be
validated, retain the unit. Embedding shards require the exact 12-digit row-index
receipt/array pairs and assembled inventory coverage. Nonempty quarantines and
unbound extra files remain protected.

## Guard protocol and build integration

Use the existing `.locks/setup-transaction.lock` for both roots in sorted absolute
path order. `transaction_guard` is already reentrant in the same thread/process;
it is fail-fast across other callers. Never acquire two roots in reverse order.
Plan/execute hold both guards throughout inventory/moves/reclamation. Normal
per-fit locks are unnecessary while exclusive root guards exclude their owning
builds; do not acquire them first and introduce inversion.

`build_library(clean=True)` completes the current `library_build.build` call and
all its research phases, allowing its guards to exit. Then invoke cleanup with
an expected gate snapshot captured by the build: setup selection hash,
prepared/embedding manifest hashes, library selector hash, production/relative
manifest hashes, optional research selector/manifest hashes. Private integration
helper checks this snapshot after reacquiring both guards and raises stale-plan
error on any change. Do not discover an unrelated new current selection and
clean against it. Public `plan_cleanup` constructs its own gate snapshot.
`execute_cleanup` re-plans, compares full plan ID and inventories, and only then
starts mutation. Failed cleanup is outside publication rollback scopes.

Before implementing cleanup, extend guard consumers with a shared *read-only*
pending-cleanup check: runtime transaction guards, `bundle_read_guard`, mutable
library/relative/summary/explanation loaders, setup/build/migration/cache cleanup,
and `build_support.shared_files`. Packaging uses FileLock directly and must
perform the same check on both roots before inventory and again under its locks.
Check pending state after lock acquisition and before even a same-thread
reentrant guard yields. Introduce a private cleanup-only guard-pair context that
acquires these raw locks and sets a thread/process-local bypass scoped to the two
verified roots; no public flag or immutable hint grants this bypass. It permits
cleanup's own validation/recovery while markers exist. No ordinary reader/writer
silently performs cleanup recovery; public
`immutable=True` must not bypass a pending marker in a managed mutable root.
Read-only installed resources have no cleanup markers and retain existing behavior.

Already loaded library scoring from in-memory signatures remains valid after an
unselected source is deleted. Lazy relative-array reads must check pending state
under the library guard *even on cache hits*; after deletion, generation checks
must raise an actionable integrity/reload error, not expose partial arrays or
silently continue from cached reference data. Direct immutable/external filesystem
access is outside the lock contract; document this supported-concurrency boundary.

## Durable journal and deterministic recovery

Use a two-root journal, not two independent commits:

- Coordinator: `<library-root>/.cleanup-journal.json`.
- Peer marker: `<setup-root>/.cleanup-peer.json`.
- Quarantine: `<each-root>/.cleanup/<token>/<candidate-index>/`.
- `token`: 32 lowercase hex characters from `uuid4().hex`.

Legacy journal schema `pclass-dark.cleanup-journal.v1` contains `token`, `plan_id`,
canonical roots and their lstat device/inode identities, gate snapshot, inventory
hash, ordered candidates (root, original path, quarantine index, full verified
file/dir inventories), and `state` (`prepared|detaching|committed|done|rolled_back`). Its
`identity` hashes all other fields. Each state update is atomic write/replace and
fsync. Peer schema `pclass-dark.cleanup-peer.v1` contains token, canonical library
root and plan ID; it does not bind the changing journal state/hash. Write
coordinator first, peer second, fsync both root directories, and detach nothing
until both are durable. Paths are relative known candidate units, never arbitrary
journal-supplied absolute deletion targets. Canonical root paths must match the
recovery caller's pair and stored root device/inode identities.

Under both guards:

1. Create the `prepared` coordinator, publish/fsync all v2 inventory sidecars,
   publish/fsync the peer, then write durable `detaching` state.
2. Move candidates to same-filesystem root-local quarantine in plan order,
   fsyncing source/quarantine parents after each rename. Each original/quarantine
   pair must have exactly one existing copy with its original verified identity.
   Do not follow directory links or cross mount points; candidates with a device
   different from their root/quarantine device are ineligible.
3. Revalidate current gate and all detached inventories. Atomically persist
   `committed` in the coordinator and fsync its parent. This is the sole commit
   point. No unlink of candidate contents occurs before it.
4. Directly remove each committed obsolete tree with descriptor-relative,
   symlink-resistant recursive removal. Check its directory inode/device/mode and
   refuse recreated original paths. Do not hash or validate individual trash
   files or re-count hard links during reclamation. Unlink directory entries,
   preserving aliases outside the tree; do not relink or rewrite references.
   Sync the quarantine parent once per candidate, not after each file. Recovery
   tolerates already missing files/subdirectories and resumes the remaining tree.
   POSIX descriptor-safe recursive removal is required. Report progress every
   100 candidates and at completion.
5. Once all candidates are absent, set `done`, then remove empty quarantine
   containers, peer marker, and finally coordinator (fsync root parents). Leave
   `.cleanup/` empty or remove it if empty. No artifacts are moved to `.archives/`.

`recover_cleanup` uses caller-supplied/resolved roots; it does not blindly follow
an arbitrary peer path into another workspace. Validate schema/hash/root binding
and allowed inventories before touching data. Before `committed`, restore all
moved directories in reverse order and validate, then durably persist terminal
`rolled_back` before removing any sidecar, quarantine container or peer. Recovery
from terminal rolled_back finishes only metadata removal, tolerating sidecars and
peer already removed, and returns `rolled_back`. The same terminal transition is
required when cancelling a prepared transaction without a peer. With `committed`, resume direct removal of remaining
trash directories and return `complete`. With `done`, finish marker/container
removal. After commit the exact individual-file inventory is historical accounting;
all contents of the committed trash tree are removable. Recreated original paths,
changed quarantine directory identity, ambiguous precommit copies, or corrupt
journals/sidecars require inspection and preserve remaining content. Never select a newer generation as a
recovery shortcut.

Crash after coordinator but before peer: no moves are allowed yet. With matching
roots, validate absence of any detached data and cancel the uncommitted journal;
a changed gate also permits this no-move cancellation. A lone peer after loss of
its coordinator is an integrity error, not permission to delete or remove it.
For committed/done recovery every original candidate path must remain absent;
a recreated original is an integrity error even if quarantine is empty. Recovery
must validate that recorded inventory totals, maximal candidate paths and token
subdirectories agree before acting. State updates and journal moves must be
fault-injected individually in tests.
For precommit IO/interrupt failure attempt rollback under the held guards; if it
cannot finish, leave markers and return/raise recovery-required status. After
commit never restore partially deleted candidates. Normal clean performs
`recover_cleanup` under both raw guards before beginning a new build, then the
normal build and new cleanup. Dry-run never recovers: it reports recovery required.

After a successful build, wrap cleanup planning/execution failures in
`CleanupExecutionError` with that `LibraryResult` as `.production_result`,
preserving the original exception as the cause. Direct planning/execution APIs
retain their specific error types where no execution has begun. Do not fabricate
a nonempty removal result for a stale or blocked plan. A build's pre-build
recovery failure has no new production result and prevents the build.

No automatic recovery in ordinary non-clean commands: pending markers block them
with instructions to use `build-library --clean` with the same roots or the
Python recovery API. A non-clean command needing only one root checks that root's
marker. This works across separate filesystems because the single coordinator
records commit and every candidate rename stays within one root filesystem.

## Resource, accounting and repository behavior

Traversal/cleanup are serial. Do not copy matrices or load models. Full validation
streams hashes and uses mmap/blocks for numerical checks. Reuse validations within
a guarded inventory to avoid hashing each matrix per target. Build indexes by
prepared/embedding/production/fit IDs and matrix/inventory hashes once; complexity
is linear in total inventory/edges plus validation bytes. No all-pairs fit-to-prep
comparison. Bound one JSON document at 64 MiB, and each tree at 1,000,000
entries and 512 MiB serialized size. Spool inventories and hard-link accounting to
private temporary disk storage; no aggregate removal-inventory limit applies.
Canonical serialized coordinator journal and each inventory sidecar must
also fit the 64 MiB document limit; calculate its size before creating markers.
Exceed a cap before mutation with
an actionable error. These are initial constants documented in API/CLI help.
The disk-backed planner preserves exact inventories and deterministic plan IDs.

For Python planning previews, compute counts via the same validators/plan as real execution; it is
not an existence-only approximation. Allocated-space estimates use `st_blocks`
where available, accounting for hard links; compression, reflinks and snapshots
can make actual freed space smaller. Do not promise exact filesystem-free deltas.
Git LFS worktree deletions leave `.git/lfs`, Git history and remote objects intact.
Run no Git staging/commit/push/LFS-prune commands. A later checkout may restore
tracked files. Do not touch profunc data, model caches, selectors, locks,
nonempty staging/quarantine from other subsystems or anything outside these roots.

## Required tests and completion gate

Use independent tiny fixtures and injected IO operations; no real data cleanup.
Implementation is complete only after these pass and actual results are recorded:

1. CLI flags/defaults, strict API booleans, all invalid combinations, missing
   current selection, full cluster-set equality, unsupported-but-complete targets,
   read-only package rejection and compatible installed-path defaults.
2. Two preparations with a shared completed embedding: delete the old preparation
   and exclusive old production/C/B/fit trees, keep the selected embedding despite
   its historical prepared ID. Verify old/new fit formats without recomputation.
3. Unique old embeddings/complete shards removed; partial shard/target/research
   protection; ambiguous owners retain the union or block; current-scope and shared
   fit IDs retained; legacy and migration caches protected with dependencies.
4. Archives with main-root versus backup-root relative selectors, migration hard
   links, known provenance-only IDs, unknown/corrupt archives and detached roots.
5. Pure preview/validators: byte-for-byte tree comparison excluding lock files,
   including corrupt-checkpoint fixtures that existing readers would quarantine.
6. Unknown files/schemas, symlinks, special files, mount boundaries, stale inventory,
   duplicate/nested candidates, nonfollowing IO and arbitrary forged plan/journals.
7. Build/research failure never cleans; build snapshot race, setup/build/packager/
   reader contention, same-thread guard reentrance and canonical two-root ordering.
   Lazy evidence cache-hit access honors pending state and deleted generations.
8. Crash before/after every coordinator/peer/state write, rename, fsync and unlink;
   no-peer/no-move recovery, rollback before commit, roll-forward after commit,
   done-marker cleanup, stale selector on recovery and corrupt-journal refusal.
9. Returned counters, hard links, caps, empty plans, second cleanup no-op, stale
   saved plans, partial deletion failures, no selected-byte changes and no Git or
   LFS-store mutations. Load/score/explain selected pipeline after every success
   and injected failure. Serial/parallel builds produce the same retirement plan
   for identical snapshots; cleanup itself has no GPU execution path.
10. Packaging still includes only selected complete artifacts and never journals,
    quarantine or deleted generations. Full pytest plus lint, CLI help, API and
    README synchronization; record any unavailable platform/hardware tests.

## Implementation sequence and readiness review

1. Add frozen types, errors and pure validation/lineage adapters with fixtures.
2. Add marker checks to runtime and packaging guards, then concurrency tests.
3. Implement inventory/retention planning and preview with deterministic plans.
4. Implement journal/quarantine execution and recovery with fault injection.
5. Integrate clean and preview CLI/API paths, including pre-build recovery and
   post-build snapshot handoff. Preserve existing cache-cleanup/publication APIs.
6. Update documentation, run full regression suite and isolated package checks.

Technical review resolved the draft's missing ownership rules, stale-plan
semantics, actual reentrant locking, mutating-checkpoint validator hazards,
packaging/lazy-reader coordination, two-filesystem commit ordering and public
return/error contracts. Conservative legacy/unknown handling is an explicit v1
limit, not an implementation TODO. No algorithm or schema decisions are deferred
before coding. Validation of this documentation revision consists of code/schema
inspection, current read-only ownership observations, local links and scoped
whitespace checks. Runtime implementation/tests remain future work.


## Implementation record (2026-10-30)

Implemented the public records/errors in `cleanup_types`, read-only planning in
`cleanup_inventory`, bounded inventories/nonfollowing IO in `cleanup_io`, and
public APIs/journal recovery in `data_cleanup`. Build integration captures its
handoff snapshot under the build's guards, then validates it after reacquisition.
Scalar options and root safety are validated before pre-build recovery. Runtime
and packaging guards reject pending cleanup; lazy relative cache hits also verify
generation state. The CLI exposes clean/preview and reports valid production paths
if post-build cleanup fails. No cleanup was run against project data.

Pure checkpoint validation also recognizes persisted `initial`, `decision` and
`trial` selection receipts, matching them to the complete scientific trace and
versioned dependency hashes. These writer-produced files were not enumerated in
the original draft's short optional-cache list. Numeric caches with unrecognized
extra files remain protected. Archived checkpoint origins are resolved through
validated fit IDs, not assumed prepared IDs. Whole legacy/migration directories
remain protected. Unsupported or unbounded retained-history layouts block cleanup.

Focused cleanup validation currently passes 40 tests, including isolated process
death at journal/move/unlink boundaries, forced-build archives, hard links,
preview purity, partial-work retention, stale plans and post-build handoff races.
Earlier tests caught and fixed selection-dependency hashing and gzip trace files
being misclassified as archived selectors. Full regression and final validation
results will be recorded before completion.


### Final validation and practical limits

- Full regression: `python -m pytest -q --disable-warnings --maxfail=3` passed
  **322 tests, 2 skipped, 1 warning in 556.81 seconds**. Log:
  `/tmp/pclass-dark-cleanup-regression.log`.
- Final targeted acceptance, including four additional edge cases added after full
  suite collection: `python -m pytest tests/test_data_cleanup.py
  tests/test_relative_scores.py -q --maxfail=1` passed **74 tests in 117.29 seconds**.
  This covers 61 cleanup cases plus 13 analytical relative-score cases. Log:
  `/tmp/pclass-dark-cleanup-acceptance.log`. The in-memory analytical fixture now
  supplies a source-generation identity for the required cache-hit checks; its
  independent numerical expectations are unchanged.
- Process-death tests cover both sides of coordinator/peer/state writes, early
  and later renames/unlinks, and ten fsync boundaries. Recovery tests cover
  internal candidate hard-link aliases as well as retained external aliases.
- Selected scoring and explanations survive cleanup; full optional research can
  be built/retained; failed research never enters cleanup; complete unsupported
  scientific targets do not fail the gate; unrelated LFS-store sentinels survive.
- Ruff and scoped Git whitespace checks pass. Isolated wheel/sdist builds under
  `/tmp/pclass-dark-cleanup-package` succeeded, contain all cleanup modules and
  exclude private cleanup state. Installed-wheel help exposes clean/preview.
  Build log: `/tmp/pclass-dark-cleanup-package.log`; artifacts:
  `/tmp/pclass-dark-cleanup-dist`.

The two real pinned-ESM acceptance tests were not enabled; cleanup itself needs
no GPU/model. Tests use tiny synthetic embeddings and temporary artifact roots,
never real project deletion. Legacy/migration generations and archive/partial
variants whose ownership cannot be bounded are intentionally retained or block
planning. Cleanup is POSIX-only; no Windows destructive cleanup is claimed.
Atomic-write temporary metadata can remain after abrupt process death before a
coordinator is published; these unmanaged files are never treated as deletion
instructions. Cooperating readers/writers honor root guards; direct external
filesystem mutations remain outside the supported concurrency contract.

## 2026-10-30 contract correction

The earlier build-then-clean behavior above is superseded by validation-only
cleanup. Existing build settings never trigger computation in clean mode.
Regression coverage asserts the build entry point is never called for clean,
missing libraries remain absent, and rebuilding/research options are rejected.

Validation of the corrected contract: CLI, production-library and cleanup suites
passed **92 tests** (280.74 s); log `/tmp/pclass-dark-clean-validation.log`.
Ruff and Git whitespace checks passed. No project data was cleaned.

## Inventory scaling correction (2026-10-30)

Spool per-node exact inventories to a private temporary file during planning;
retain only offsets with graph nodes. Read one inventory at a time when validating
and forming plan entries. Apply aggregate entry/serialized-size limits to removal
candidates only, since only these inventories enter execution and its journal.
Keep per-tree caps, exact tree hashes, hard-link accounting, plan identity and
journal validation unchanged. Temporary storage is proportional to total scanned
inventory, automatically closed on success/failure, and never written under data
roots. Validation must cover retained trees exceeding the aggregate cap, candidate
caps, deterministic plans, temporary-file closure, and recovery regression tests.

Implemented and validated: cleanup suite 66 passed (109.15 s), CLI suite 11
passed; Ruff and whitespace checks passed. Production-data read-only traversal
confirmed 5,218,158 fit-cache files, exceeding the former global entry limit.
No production-data cleanup/complete planning was run. Temporary storage scales
with total inventory; candidate and journal caps remain explicit limitations.

## CLI surface correction (2026-10-30)

The earlier CLI preview/dry-run descriptions and acceptance records are historical:
`--dry-run` and explicit root/numerical CLI options are now removed. The Python
planning API remains available. `--clean` uses default roots and executes validated
cleanup directly. See SPEC-0004 for the reduced CLI contract.

Reduced CLI validation: CLI and cleanup suites passed **85 tests** (108.85 s),
including removed-option rejection, help, default forwarding and default-root
cleanup integration. Ruff and Git whitespace checks passed.

## Obsolescence contract correction (2026-10-30)

User clarification supersedes unconditional rollback-archive protection. A prior
valid generation is a fallback only until a complete replacement production/C/B
pair and its selected setup are validated. Under that gate, recognized `.archives`
trees in managed roots become removable rollback history, including migration
backups and copies of current results. Their dependency references do not pin old
preparations or publications. Keep shared inputs, current selection, partial work
and unknown/unsafe history protected or fail closed. Legacy active migration and
checkpoint namespaces retain their existing conservative behavior. Delete archive
trees through the same exact-inventory journal/quarantine/recovery machinery.
No deletion during implementation. Test archive retirement, invalid replacement
refusal, shared embeddings, and rollback/recovery with archive candidates.

Archive-obsolescence correction implemented and validated: cleanup/CLI suites
**87 passed** (123.06 s), including archive-bearing recovery and invalid-replacement
refusal. Ruff and whitespace checks passed. No project data was deleted. Existing
removal-inventory and journal limits remain operational constraints.

## Scalable execution correction (2026-10-30)

Remove the aggregate removal-entry/memory cap by storing candidate inventories
on temporary disk. Use disk-backed inode accounting, preserving exact hard-link
space estimates. Bind plans to sorted per-tree hashes. Keep per-tree caps.
New v2 coordinator journals reference bounded per-candidate inventory sidecars;
all sidecars are durable before any detachment. Keep the single all-candidate
commit boundary and v1 recovery compatibility. Prepared transactions without a
peer roll back safely even if sidecar publication was interrupted. Done-state
recovery tolerates sidecars already reclaimed. Validate sidecars against entry
hashes before any rollback or reclamation. Stream quarantine checks and hard-link
accounting, avoiding aggregate in-memory file lists. Clean plans once under the
root guards; public execute_cleanup still independently revalidates previews.
Test scaled execution with artificially small per-document limits, sidecar
publication/corruption failures, recovery and empty-plan scan counts. No project
data deletion during implementation.

New execution writes `pclass-dark.cleanup-journal.v2`. Its `inventories` list
contains `{sidecar: "inventory-N.json", sha256: entry.tree_sha256}` references.
Files live at `library/.cleanup/<token>/inventory-N.json`; path names are derived
from validated token and index, not trusted paths. The coordinator is durable
first in prepared state; sidecars are durable before publishing the setup peer
and starting detachment. Without a peer, prepared recovery cancels only metadata;
with a peer, all exact sidecars must validate before touching detached data.
Done-state recovery can finish removing remaining sidecars after interruption.
Legacy v1 inline-inventory journals remain recoverable. Candidate inventories and
hard-link counters use bounded SQLite caches; plans hash sorted tree digests and
owners rather than serializing aggregate trees into RAM. Individual-tree and
coordinator-document caps remain. Temporary and durable disk use scale with the
inventory; disk exhaustion before commit cannot authorize deletion.

Scalable execution implemented and validated: cleanup/CLI/distribution/cache
suites **106 passed** (144.94 s), with four final focused checks passing after
last cleanup/log changes. Synthetic scale tests exceed aggregate budgets while
keeping each tree/sidecar bounded; v1/v2 recovery, corrupt sidecars, incomplete
publication, interrupted finalization and single-scan empty cleanup are covered.
Ruff and whitespace checks passed. Real-data deletion was not executed.

## Direct reclamation correction (2026-10-30)

After the selected replacement, candidate ownership and detached inventories have
validated and the transaction is committed, treat quarantined contents as trash.
Use descriptor-relative symlink-resistant recursive deletion instead of repeated
content hashing, per-file identity checks and per-unlink fsync. Keep root inode,
device/mode identity, canonical root/selector binding and original-path recreation
checks. Sync the quarantine parent once per candidate. Retain the coordinator and
sidecars until reclamation finishes, supporting partial-tree restart and existing
v1/v2 committed journals without relinking or rebuilding anything. Interrupted
committed cleanup resumes removal; precommit rollback semantics remain unchanged.
Progress logs report candidate/file counts as deletion proceeds. Tests cover
partial-tree interruption, shared hard links, symlink containment, replaced roots,
no reclamation hash pass and no per-file fsync. Do not interfere with the live
user process or run real-data deletion during implementation.

Direct reclamation implemented: 107 broader regression cases passed; two crash
injection tests targeted fsync calls no longer reached and were updated. All ten
fsync crash/recovery cases then passed. Final five direct-delete, no-replan resume
and single-scan checks passed; unit deletion of 200 files requires one fsync and
neither hashing nor relinking. Ruff and scoped whitespace checks passed. Broad
suite was not repeated after these targeted corrections. No agent-initiated
real-data cleanup; a running old process must be interrupted/restarted to load the
new implementation. Completed recovery returns without a second full plan scan.

## Partial embedding ownership regression (2026-10-30)

Implemented correction: an interrupted embedding job can have valid shard receipts
without an assembled embedding generation. Keep its existing preparation owners,
dependencies and protection; inherit assembled-embedding owners only if that
embedding exists. Validate preview and execution against a valid unassembled shard
and verify checkpoint files and owning preparations survive cleanup.

Validation: cleanup suite **77 passed** in 104.02 s
(`/tmp/pclass-clean-partial-embedding.log`). The final full-range-shard regression
also passed separately (`/tmp/pclass-clean-partial-embedding-focused.log`),
covering preview, direct execution and clean orchestration with unchanged shards
and retained owning preparations. Ruff and scoped whitespace checks passed.

## Rollback-finalization correction (2026-10-30)

Persist a terminal `rolled_back` coordinator state after all precommit originals
are restored, and before deleting any sidecar/peer. Apply the same terminal state
to prepared-without-peer cancellation. Recovery from rolled_back only finalizes
metadata; it does not require already removed sidecars or peer, inspect selected
pipeline changes, or repeat candidate restoration. Keep root/coordinator identity
and candidate-path checks. Test interruption after sidecar removal, peer removal,
and terminal state publication for prepared and detaching recovery paths.

Implemented and validated: eight rollback-finalization fault cases passed,
covering prepared without peer, prepared with peer, and detaching transactions.
Each interrupts after terminal-state publication, first sidecar removal, or peer
removal as applicable; subsequent recovery preserves exact original inventories,
clears all markers and permits readers/writers. Cleanup/CLI suites **104 passed**
in 107.61 s (`/tmp/pclass-clean-rollback-final.log`). Ruff and scoped whitespace
checks passed. No project-data cleanup was executed.
