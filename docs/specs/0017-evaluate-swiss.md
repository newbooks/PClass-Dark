# SPEC-0017: Swiss-Prot classifier comparison

- Date/revision: 2026-11-02.
- Status: Implemented; final validation results are recorded below.
- Scope: This revision replaces earlier exploratory execution choices in this
  draft. Version 1 has one native execution path, specified below.

## Purpose and scientific contract

Provide `scripts/evaluate_swiss.py FASTA`, `pclass-dark evaluate-swiss FASTA`, and
`pclass_dark.swiss_evaluation.evaluate_swiss` to compare PClass and PClass-Dark
against direct Swiss-Prot molecular-function (MF) annotations from profunc.
Default output is `WORK_DIR/swiss_evaluation.tsv`. Reuse the scientific scoring
and durable publication machinery from [SPEC-0015](0015-evaluate-external.md),
without changing that evaluator's command-output defaults.

The immediate input, `work-swiss/sequences.fasta`, contains 380,578 filtered,
deduplicated prepared sequences, not all 575,748 original Swiss-Prot records.
Evaluate exactly the input records, once each, in input order; do not expand
aliases, deduplicate further, retrieve sequences, or substitute accession-based
self predictions for inference. Exact self matches and training overlap remain
possible. Label outputs/provenance as a potentially in-sample reference-cohort
diagnostic, not held-out performance. No homology split, fitting, threshold tuning,
self-neighbor removal, or classifier scientific changes belong to this operation.

## Inspected sources and required behavior

Neighboring sources are provenance references, not checkout-path dependencies:

- profunc `cli/inspect.py:function_description`, `go/sets.py` loaders/resolver,
  `go/description.py:describe_go_function`, `swissprot/annotations.py`.
- PClass `infer/cli.py:infer_cmd`, `infer/artifacts.py`,
  `query/client.py:request_neighbors`, `serving/engine.py:ServingEngine`.
- Here: `external_evaluation*.py`, `external_audit.py`, `inference.py`,
  `inference_backend.py`, `inference_records.py`, `embeddings.py`,
  `relative_scores.py`, and the selected prepared/embedding manifests.

`profunc analyze function A0A011QK89 -v` first resolves the accession's direct
MF set, then composes prose. Truth is the full resolved set before description
pruning, not just head/modifier/supporting terms. Retain ancestors, root and
`GO:0005515` if present; do not add ancestor closure or cluster labels. profunc's
current builder excludes ND/NAS evidence, rather than requiring experimental-only
annotations. Record its dataset/evidence provenance without adding a new filter.

PClass infer currently always executes Needle and ProSig, including JSON mode,
and its GO-only artifact helper still loads alignment sequences. Native neighbor
retrieval bypasses those costs. Dark's current `infer_queries` constructs a fresh
model/library per session; importing it per chunk does not solve repeated loads.
Its 100,000-query and input/temp caps prohibit one unbounded cohort session.
Existing `RelativeScores` already has a 16-entry/64-MiB LRU; reuse it initially.

## Version 1 decisions and exclusions

Use installed PClass's neighbor HTTP client against a resident PClass service;
use a new resident Dark runtime here. No CLI compatibility backend, local PClass
engine backend, HTTP-client rewrite, mixed precision, approximate scoring,
length bucketing, shared PClass/Dark embeddings, compression, or cross-workspace
cache adoption in version 1. These are deferred rather than optional requirements.
The external evaluator remains available separately for baseline collection.
Do not require ProSig, Needle, local PClass inference bundles or rendering modules.

Native PClass collection requires the installed `pclass` distribution/API. Initial
compatibility target: 0.3.x, inspected at 0.3.0, with callable `request_neighbors`,
`SequenceInput` and `NeighborResponse.model_validate/model_dump`. Test that range
and reject unsupported major/minor APIs clearly. Do not add a PyPI dependency
whose identity has not been verified; document installation of the actual PClass
project in the evaluator environment. Lazy imports preserve base-package,
score-only and unrelated CLI operation. Dark model misses need existing
`pclass-dark[embeddings]`; all-cache-hit runs must not import Torch or load a model.
Keep profunc as the existing dependency; no new scientific package is needed.

## Public interfaces and defaults

Library signature:

```python
evaluate_swiss(fasta, *, work_dir, output=None, pclass=None, dark=None,
               scoring=None, execution=None, score_only=False,
               preflight_only=False)
```

Return existing `EvaluationResult`/`EvaluationPreflight` types; use existing
`ScoringConfig` and exception classes. Add frozen validated `SwissPClassConfig`,
`SwissDarkConfig`, `SwissExecutionConfig` in `swiss_evaluation_types.py` rather
than overloading external-evaluation flags that no longer apply. No import-time
argument parsing, worker creation, resource resolution or model activity.

| CLI option / config field | Default and contract |
| --- | --- |
| `--output` | `work_dir/swiss_evaluation.tsv`; parent must exist |
| `--pclass-batch-size` / `pclass.batch_size` | 512; integer 1..1024 |
| `--pclass-server-url` / `pclass.server_url` | `PCLASS_SERVER_URL`, else `http://127.0.0.1:8000`; explicit option wins |
| `--pclass-timeout` / `pclass.timeout_seconds` | 60 seconds; finite positive HTTP timeout |
| `--dark-chunk-size` / `dark.chunk_size` | 512; integer 1..1024 |
| `--dark-library-dir` / `dark.library_dir` | existing Dark default resolution |
| `--device` / `dark.device` | `auto`; allow `cpu`, `cuda`, `cuda:N` |
| `--dark-batch-size` / `dark.batch_size` | 8 embedding windows; integer 1..64 (existing backend cap) |
| `--dark-score-batch-size` / `dark.score_batch_size` | 4096 windows; integer 1..65536 |
| `--dark-memory-mib` / `dark.memory_mib` | 1024; integer >=128; existing scoring admission formula applies |
| `--dark-workers` / `dark.workers` | 8; integer 1..8, clamped to CPU availability |
| `--dark-embedding-cache` / `dark.embedding_cache` | None; explicit completed embedding directory |
| `--dark-prepared-dir` / `dark.prepared_dir` | None; required iff embedding cache supplied |
| `--offline` / `dark.offline` | false; existing model asset policy |
| `--score-workers` / `scoring.workers` | 8; existing range 1..8 |
| `--go-graph`, `--cluster-go-map` | existing scoring overrides only |
| `--process-timeout` / `execution.process_timeout` | 3600 seconds per initialization/task, finite positive |
| `--max-raw-chunk-mib` / `execution.max_raw_chunk_mib` | 128; integer >=1; each evidence file |
| `--run-tag` / `execution.run_tag` | `default`; manual invalidation of externally changed models/indexes |
| `--score-only`, `--preflight-only` | false; mutually exclusive |

Reject booleans as numeric config values. Do not accept `--truth-tsv`,
`--pclass-artifact-dir`, neighbor-count/cutoff changes or skipped truth rows.
`work_dir` defaults to cwd only in CLI; Python requires it explicitly. Relative
paths are resolved once against invocation cwd; output defaults against work_dir.
Supported platform is POSIX/Python >=3.11 as for the external evaluator.

## Input, truth and preflight

Accept regular uncompressed UTF-8 FASTA, nonempty input, unique first-token IDs
(length <=256), sequences validated/uppercased with SPEC-0015 alphabet and limits.
Accept bare primary accessions or `sp|ACCESSION|ENTRY`; preserve original first
token as `Query_id`, store normalized accession separately. Validate accession
syntax with the existing UniProt primary-accession pattern; reject `tr|`, malformed
pipe headers, isoforms and duplicate normalized accessions. Dataset membership,
not syntax alone, establishes a supported primary accession. Strip metadata and
truth before submission. Optional `go_mf_terms` header truth must agree exactly
as a set with profunc; unknown header metadata is ignored.

Resolve profunc once through `resolve_runtime`, validate manifest-bound bytes,
and call `load_go_set_resources(runtime.dataset)` once for graph and accession
mapping. Copy the annotation source/manifest into immutable source state; validate
before/after copies to detect concurrent changes. Freeze exact per-query sorted
unique truth in the existing SQLite snapshot format, plus source identity.
Fail before inference for missing/empty annotations, incompatible MF graph terms
or unknown mapped clusters. Error lists show total counts and at most 20 IDs.
Graph overrides affect scoring, not truth acquisition. Full normal invocation
refreshes truth from current profunc; changed annotations can reuse predictions.

Preflight performs input/truth/scoring/library/cache checks and API compatibility
inspection, not service requests, model creation/downloads or biological inference.
Report query/window counts, source IDs, pending/cached query counts, cache hit/miss
windows, needed optional dependencies and estimated evidence/temp admission limits.
Report remote availability/device as unverified. No raw generation or selection
is published. Frozen validation snapshots and an index cache may be retained.

## Module boundaries and shared-code changes

| Module | Responsibility |
| --- | --- |
| `swiss_evaluation.py` | modes, paths, locks, truth resolution, task ordering, scoring/publication orchestration |
| `swiss_evaluation_types.py` | validated configs, native evidence version constants |
| `swiss_evaluation_cli.py`, `scripts/evaluate_swiss.py` | thin Typer and script adapters; register installed command |
| `swiss_evaluation_collect.py` | worker supervisor, chunk receipts/generations and evidence validation |
| `swiss_evaluation_worker.py` | explicit worker entry point; lazy PClass/Dark imports and task handling |
| `inference_runtime.py` | public context-managed resident Dark runtime and embedding-cache adapter |

Extract or parameterize reusable external evaluator helpers for state/output
names, explicit-truth snapshot ingestion and reader dispatch by evidence format.
Reuse `project_pclass`, `project_dark`, `score_query`, `combine_matches`, metric
helpers and publication transaction logic; do not duplicate scientific formulas.
Do not invoke `evaluate_external` against a temporary FASTA/truth subprocess.
Keep existing defaults, manifest readers and external cache identities unchanged.

New public runtime API:
`DarkInferenceRuntime(library_dir=None, *, device='auto', workers=8, batch_size=8,
score_batch_size=4096, memory_mib=1024, offline=False, embedding_cache=None,
prepared_dir=None)`, context-managed, with `infer_chunk(queries)` returning a context-managed
`InferenceSession` yielding existing `inference_records.QueryResult` records. Expose read-only `source` and `execution`
metadata properties, available after runtime entry and after first chunk,
respectively; execution records distinguish cached vectors from model forwards.
It owns one pinned library and lazy embedding backend across chunks. Reuse the
current disk-backed session's scoring loop/reducer and caps with an injectable
runtime owner, not an all-cohort in-memory `score_query_windows` result. Existing
`infer_queries` keeps its signature, behavior and self-owned lifecycle.

## Resident collection and failure behavior

Collect pending Dark chunks first, then pending PClass chunks, sequentially.
This validates the complete Dark source/cache before requesting remote predictions
while loading the Dark library only once per worker lifetime. Run
one supervised worker at a time using multiprocessing `spawn`, a POSIX process
group, and a single request/response Pipe. The worker may spawn no child workers.
Tasks carry paths/digests, not matrices or full responses through IPC. Parent
allows one outstanding task; child writes bounded attempt files, closes them and
returns their paths/digests. Parent validates before commit. Stop/terminate/kill
and reap the group on cancellation, timeout, crash or byte-cap overrun, as in
SPEC-0015. No automatic retry of arbitrary failures within the invocation;
rerunning starts a new worker and reuses already committed chunks.

Protocol messages are versioned JSON-compatible dicts with `protocol=1`, `kind`,
and monotonic integer `task_id`. Initialization config/source paths are passed as spawn arguments; parent sends
`chunk` (input snapshot, ordered ordinals/member digest, attempt directory), or
`close`. Child replies `ready` (adapter contract), `done` (evidence path, file
record, execution metadata), or `error` (exception category/message, log path).
Accept paths only inside the assigned attempt directory and matching task IDs;
EOF, malformed messages or unexpected state transitions fail the task. Never
send exception objects or serialized model objects. Parent is the sole committer.
No-worker cache reuse still verifies receipts and generation coverage.

Separate worker initialization and each chunk have process-timeout deadlines.
Parent polls at 0.1 seconds, checking evidence size; stderr goes to attempt logs.
A successful handshake identifies adapter/package code versions and resource
contracts. Model initialization, if needed, happens once during Dark task handling
and must fit that task's deadline. Heartbeats do not extend deadlines. On normal
exit explicitly close Pipes, sessions, memory maps, backend context and library
guards. Scoring process pools start only after inference workers have exited.

PClass calls `request_neighbors` with `SequenceInput`, resolved URL and HTTP
timeout; omit `max_neighbors`/`max_distance` to preserve service defaults. Use the
existing count/8-MiB planner. Bisect only typed size errors and HTTP 413 as encoded
by the existing client; singleton rejection fails. Persist split plans/leaves.
Validate exact returned IDs, unique neighbors, finite distance/confidence and GO
syntax. Preserve report-order ties. Store the validated service response. Do not
import PClass infer CLI, Needle, ProSig, descriptions or alignment sequence loaders.

Dark loads/verifies production/C/B artifacts once. Pin hashes/generation IDs and
hold existing transaction guards for mutable roots through the worker lifetime.
Do not assert `immutable=True` for mutable inputs just to avoid checks. Check
selection/generation identity at chunk boundaries and fail if changed. Reuse the
existing reference-array LRU and score batch admission; no unbounded all-cluster
cache. Keep deterministic query windows, scoring and accepted-window reduction.
Compute float32 embeddings with existing device/fallback policy, no autocast/TF32
change. Record actual backend device; for all-cache-hit chunks record
`embedding_device=not_used`, not an assumed GPU. No model per scoring process.

## Verified embedding reuse

Cache reuse is explicit; no automatic selection discovery. Require both completed
embedding and originating prepared directories. Validate prepared manifest/files
with existing preparation readers, and `verify_embeddings(cache, windows)` with
the prepared `windows.tsv` inventory. Require matching verified inventory IDs (record originating/supplied preparation
IDs separately),
array row count, C-contiguous little-endian float32 width 640, finite values and
manifest file checksums. Compare `representation_contract` against the selected
Dark library, including pinned assets, pooling, layer and residue policy. Verify
completed cache once per invocation, never once per query/chunk.

Build a disk-backed SQLite index in the workspace keyed by
`(sequence_id, parent_length, kind, start, end, window_sha256)` to embedding row.
Join cache `rows.tsv.window_id` to verified prepared `windows.tsv`; the former
alone lacks coordinates. Verify prepared sequence hashes and regenerate window
slices to check the latter's coordinates/content. The query side uses normalized
sequence SHA-256 and regenerated `make_query_windows` metadata, independent of
accession/header names. Duplicate keys must have identical vector bytes or fail;
choose lowest row deterministically. Index identity includes both source manifests
and index algorithm version. Stage index before atomic publication. Version 1
rebuilds the index after source verification each invocation; it does not trust
a previously generated index without rebuilding. Cache source stat identities
(device/inode/size/mtime/ctime) are checked at chunk boundaries after the initial
full content checks.

Memory-map the verified matrix, gather only a bounded chunk's hit vectors, and
embed only misses in configured microbatches. Reassemble vectors in window order
before the unchanged scorer. No missing vector is replaced with zero; bad cache
content is an integrity error, not a cache miss. No cache means all misses.
All-hit runs initialize no model. Mixed runs retain one backend after first miss.
Do not modify the supplied cache or publish new model vectors there in version 1.
Record hit/miss counts and per-chunk cache provenance. Cache/device/precision
execution identity belongs to the inference contract because numerical differences
may change acceptance. No PClass vector reuse is permitted.

## Evidence, identities and publication

Native envelopes have exactly the required top-level fields `schema`, `source`,
`execution`, `members`, `payload`; `members` is the ordered list of
`{ordinal, id, length, sequence_sha256}`. PClass `payload` is its response
object; Dark `payload` is the query-record list. Source/execution records use the
contract fields defined above; validators require them and reject unrecognized
schema versions. Dark record `query_index` is chunk-local; reconcile to global
ordinal through members before scoring. Keep window IDs query-local and validate
against regenerated query windows. Do not rewrite cached evidence ordinals.

Use `.swiss-evaluation/` under work_dir with existing workspace UUID, lock,
input/truth/resource/attempt/raw/generation/score/selection layout. Version native
formats independently from SPEC-0015; reject unknown versions rather than guessing.

- Input snapshot: existing normalized query records plus `(ordinal, Query_id,
  primary_accession)` mapping; input identity binds IDs/order/sequence hashes.
- Truth snapshot: existing exact-coverage SQLite records; truth identity also
  binds annotation source/dataset identity and truth resolver version.
- PClass evidence: `pclass-dark.swiss-pclass-neighbors.v1` envelope with source
  contract and complete `NeighborResponse.model_dump(mode='json')` payload.
- Dark evidence: `pclass-dark.swiss-dark-evidence.v1` envelope with pinned source,
  actual execution/cache provenance and an ordered list of complete existing
  `pclass-dark.infer-query.v1` numeric records from `to_dict()`. Retain all accepted
  clusters and supporting windows, not only the winner; omit prose/rendering.

Envelope validation rejects duplicate JSON keys, nonfinite numbers, unknown schema,
missing/extra/duplicate queries and sequence/window mismatches. Include normalized
query member hashes in each receipt. Each raw chunk gets a checksum-bound receipt
and atomic staged-directory commit. Oversized evidence fails with a resource error
and suggests smaller chunk size; no partial raw publication or silent truncation.
Raw output cap is per chunk, not a total workspace quota. Keep input/temp limits
from existing Dark sessions and surface disk-write failures without partial publication.

Chunk ID = hash(adapter format/code/package fingerprint, explicit inference
configuration/source contract, ordered query members). Include model/cache
execution settings; exclude truth and scoring graph/map. Generation ID binds
input ID, contract and chunk plan, with exact coverage verified. Score ID binds
both raw generations, truth ID, scoring graph/map and scoring implementation.
Package versions alone are insufficient for editable installs: fingerprint loaded
adapter/client/schema and Dark numerical/reducer source bytes or fail if unavailable. PClass remote model
identity may be unavailable: record that limitation and require a new run tag
when its index/model/defaults change. No guessed remote identity.

TSV columns/order and selection/scoring equal SPEC-0015:
`Query_id`, `PClass_go`, `Embedding_d`, `Pclass_match`, `Dark_go`, `Dark_Bscore`,
`Dark_match`. Select minimum-distance neighbor with report-order ties, using its
reported direct GO set, and minimum-B accepted Dark cluster with cluster-ID ties.
Reuse profunc AMB Lin with frozen graph IC. Empty/undefined values are `NA`,
defined zero stays zero. Unknown GO terms in collected predictions warn and are excluded by profunc from
semantic scoring; original GO sets remain in outputs and missing terms are
recorded in metric status. No usable terms yields an unavailable (`NA`) score.
Preflight truth/map compatibility checks and unknown cluster IDs still fail.
Validate truth compatibility before inference and all predictions before scoring.
Log cumulative per-engine query percentage (one decimal), including reused leaves.

Refactor publication to accept internal score-table basename; Swiss score
folders contain `swiss_evaluation.tsv`. Preserve ownership checks, journaled
rollback/recovery, immutable score copies and refusal of altered/unowned outputs.
Never overwrite input/source files. Score-only verifies the supplied FASTA against
the saved input and uses saved truth/resources/generations without importing
PClass, Torch or resolving live profunc. It rejects all inference configs/flags;
explicit graph/map overrides can rescore, workers can change. Missing selection
fails. Source/cache files need not remain accessible for offline scoring. The supplied
FASTA is still required, while live annotation refresh is never performed in
this mode. Defaults create no inference config objects in score-only CLI mode.

## Audit metadata migration

The current audit hardcodes `INPUT.parent/.external-evaluation/scores/SCORE_ID`;
a separate Swiss workspace is not compatible without a change. Add publication
marker v2 fields `state_relative` (relative POSIX directory from output parent to
workspace) and `evaluation_kind` (`external` or `swiss`). Preserve score ID,
workspace UUID, TSV and score-manifest checksums. Publication/recovery must hash
and compare the entire v2 marker; never change old markers in place.

Audit resolves v2 paths from the output parent, rejects absolute paths, symlink
components and mismatched workspace UUID, then verifies manifest/status checksums,
score identity, exact query order and truth coverage as today. Relative `..` is
allowed for explicit output outside work_dir; no file is imported/executed from
metadata. Marker v1 retains its current external-only lookup. Test external v1,
external v2 and Swiss v2, including explicit output in another directory and
corrupt/path-substituted metadata. No automatic copying of state next to output.
The audit's [SPEC-0016](0016-external-audit.md) binding-term exclusion remains an
audit-only operation. Update that spec with the implemented marker contract.

## Resource bounds, performance acceptance and tests

Truth mapping/graph load once; sequence and query snapshots are disk backed.
`verify_prepared` currently materializes reference sequence/window dictionaries;
version 1 may use that existing verifier during preflight, reporting its peak
memory and releasing dictionaries before worker startup. This bounded-to-source
validation allocation is separate from streaming cohort inference; do not retain
it alongside model state. Do not promise constant-memory preflight.
Per-task vectors and accepted evidence obey existing Dark temp/input and score
memory checks; the memory setting is workspace admission, not a total RSS cap.
Mapping/graph/model overhead must be measured. Keep the existing 64-MiB reference
LRU, and bound repeated canonical GO-pair result caches to 4096 entries per score
worker, keyed by graph identity and both full sets. Numerical thread limit is one
per score process; Dark threads are clamped to its budget. No inference/scoring
pool overlap. Deterministic order survives worker-count changes.

Required automated coverage:

1. Truth/header parsing, annotation provenance changes, missing/empty truth,
   malformed mappings, duplicate normalized IDs, isoforms, invalid sequences and
   overridden graph incompatibility. All fail before inference when detectable.
2. Independent analytical AMB Lin example and equality with external scoring on
   the same saved neighbor/Dark evidence; tie/abstention/undefined handling.
3. PClass fixture client validation, size splitting, partial chunks; assert no
   Needle, ProSig, infer/description or alignment-sequence calls/imports.
4. Dark session/runtime parity for full and multi-window queries, one initialization
   per worker lifetime, all-hit no-model case, mixed misses, memory admission,
   cache hash/coordinate/contract mismatch, conflicting duplicate vector keys.
5. Native receipt schemas, deterministic identities, restart/resume and changed
   truth reuse; injected crashes/timeouts/oversized files, group cleanup,
   transaction-guard release, corrupt state, publication recovery and lock refusal.
6. Offline score-only with inference/live-resource imports blocked and cache/source
   paths removed; all script/library/installed CLI modes and defaults/overrides.
7. Audit v1/v2 truth recovery/exclusion; unchanged external evaluator tests.
   Verify CPU and serial/parallel consistency; exercise GPU when available and
   record unavailable hardware honestly. Tolerances: float scores `atol=1e-6`,
   `rtol=1e-5` across compatible device/cache executions; discrete winners and
   acceptances must agree on parity fixtures. Never excuse changed winners with
   tolerance; investigate and keep execution contracts separate.

Before full inference, select up to 96 accession-sorted queries: 32 each of
length <=256, 257..1022 and >1022 (take available counts, no replacement). Save
IDs, source hashes, code revision, commands and hardware. Use checkpoint size 16
for at least three warm tasks when enough queries exist. Compare cold/steady
PClass full CLI versus native neighbors, and Dark per-chunk CLI versus resident
uncached versus resident cached execution. Bound costly full PClass CLI baseline
to the first two queries per stratum; use saved baseline fixtures for broad parity.
No full-cohort execution is required to implement the feature.

For a throughput pilot, report per-stage wall time, queries/windows per second, initialization count,
peak RSS/VRAM, evidence/temp bytes, cache coverage and estimated full-cohort time
with assumptions. Benchmark GPU microbatches 8/16/32 on the same pilot only when
admission allows; failures are recorded, not hidden by silent parameter changes.
No promised speedup factor: acceptance requires one resident initialization,
zero auxiliary-evidence calls, verified cache hits avoiding forwards, and measured
end-to-end evidence. A slower resident path must be investigated before rollout.

## Implementation sequence and completion record

1. Add typed configs, truth resolver and snapshot/state parameterization with
   external-regression tests; implement native evidence readers and projections.
2. Add worker protocol and PClass neighbor adapter with failure/resume tests.
3. Add resident Dark runtime and verified embedding adapter; preserve ordinary
   inference API behavior and validate native/cache parity and lifecycle.
4. Wire scoring/publication, marker v2/audit support and all three entry points.
5. Run focused then full relevant pytest suites, Ruff/format/whitespace and
   packaging checks. Include the new script in distributions. Synchronize README,
   API/operational docs, specs 0015/0016 where shared contracts change, and log.
6. Run bounded pilot when installed service/model/hardware allow; record blockers
   and unverified live/GPU behavior separately from passing fixture tests.

Initial source check: live `profunc analyze function A0A011QK89 -v` returned
`GO:0047545`. The implementation is present in the modules and entry points specified above.
It shares external snapshot/scoring/publication helpers and the existing Dark
session/reducer. Test and real-data validation results are recorded below.
Full-cohort inference, model throughput and GPU parity remain unverified.

### Implementation decisions from real-data validation

The selected embedding manifest originates from preparation
`94d930743f63abfa074f0b111ce7fe4a8c5558b79f35090dcb5ea6aec06ff11f`, while the supplied
selected preparation is `8e5994726cb958fce5d2396835f9a3767da897007110a9fda4f2e3b06cccbe4e`.
An initial preflight rejected that pair. Both have the same verified window
inventory `cb1f4d8a94f0067d729fff579e89b854f3ab5110ba800676598471266b873ff1`.
Preparation identity also covers clustering/background settings irrelevant to
vector representation. Therefore cache compatibility uses the full verified
sequence/window inventory and representation, preserving originating preparation
as provenance rather than requiring identical preparation IDs. This is a
meaningful relaxation of the original draft, not an accession-only cache join.


### Validation results (2026-11-02)

- Full regression suite: **592 passed, 2 skipped in 731.56 s**;
  `/tmp/swiss-full-suite.log`. The two opt-in pinned-checkpoint tests were skipped
  because `PCLASS_DARK_TEST_CACHED_ESM=1` was not set; they are not GPU passes.
- Final complete Swiss suite: **52 passed in 75.89 s**;
  `/tmp/swiss-final-suite.log`. After adding numerical thread limits for model-free
  cached execution, the affected resident/cache/parity tests and new thread
  restoration regression passed (**3 passed in 15.23 s**;
  `/tmp/swiss-thread-checks.log`). The current Swiss file contains 53 tests.
  CPU fixtures cover model/cache parity, lifecycle, mixed misses, mutation,
  admission, independent Lin scores, parallel scoring, offline rescoring, native
  evidence, worker deadlines, response validation and publication/audit recovery.
- Real-data preflight: first three records of `work-swiss/sequences.fasta`, saved
  in `/tmp/pclass-swiss-preflight/queries.fasta`; three query windows, three cache
  hits, zero misses. No model or service inference. Output log:
  `/tmp/swiss-preflight-final.log`; frozen validation artifacts in
  `/tmp/pclass-swiss-preflight/.swiss-evaluation/`. Graph ID:
  `819a5675ebd00edd6622d1bd45c8feb05928a8fc2f12d7dc67e555fc843b4f06`.
  This is a source/cache integration check, not full-cohort coverage or throughput.
- Code-only temporary wheel/sdist smoke build succeeded. Verified all seven new
  modules in the wheel and executable script/operational guide in the sdist.
  Ruff lint/format, local documentation links/fences and whitespace checks passed.
- Full model inference, native-versus-full-CLI throughput pilot and GPU parity
  were not run; no service/model performance claim is made.

### Scoring warning validation (2026-11-03)

Unknown collected GO terms now warn, with excluded terms recorded in metric
status; preflight compatibility remains strict. External/Swiss suites: 113
passed, with one sandbox-blocked localhost fixture subsequently passing with
socket permission. Four new analytical cases cover partially/fully missing
truth/prediction terms. Ruff lint/format and whitespace checks passed.

### Completed-generation recovery (2026-11-03)

Before collection, search completed native generations for the same engine,
input and chunk configuration. Reuse only identical contracts or the explicitly
reviewed pre/post unknown-GO-warning scoring fingerprints (their projections are
unchanged); all other source, adapter, model, package, configuration, environment
and run-tag fields must match exactly. Do not ignore arbitrary scoring source
changes: this module also contains evidence projections. Keep existing immutable
IDs/receipts, verify checksums and exact coverage, and choose deterministically
by generation ID if multiple compatible generations exist. Corruption fails;
absent/incompatible generations use ordinary collection. This works before any
selection/publication exists. Worker resource validation still runs; no inference
tasks execute for reused generations. No artifact rewriting or migration.

Recovery implementation complete: `swiss_evaluation_resume.py` is called by the
shared script/library/CLI pipeline before collection. External/Swiss regression
suites: **118 passed in 81.24 s** (localhost permission enabled). Ruff and
whitespace checks passed. Read-only recovery selected the original 744-chunk
Dark and PClass generations for all 380,578 real Swiss queries without inference
or artifact mutation. Full-cohort scoring was not executed in this change.
