# SPEC-0007: Infer command and A/B/C match reporting

- Original date: 2026-10-27; technical revision: 2026-10-30.
- Status: **Implemented** on `issue#23`; validation results are recorded below.
- Provenance: user-approved infer inputs, devices, stdout reporting and window
  union rules; existing [signature scoring](0004-build-library.md),
  [C/B evidence](0005-relative-score-distributions.md),
  [artifact paths](0006-shared-artifacts.md) and current
  [artifact guards](../../src/pclass_dark/artifacts.py).

## Purpose and scientific boundary

`pclass-dark infer` reports accepted cluster matches for protein queries. Keep the
report focused on Cluster, A=Yes, B and C. Interpretation belongs to users. Do not
add likelihood labels, hard B cutoffs, gap detection, confidence categories,
combined probabilities, top-k filtering or automatic functional annotation.

Scientific behavior already approved by the user is fixed here: reference-style
terminal windows; existing fitted acceptance; accepted-window union; one cluster
row represented by its accepted window with lowest B; C from that same window.
The details below settle engineering interfaces and failure/resource behavior.
No new scientific decision is needed to implement this contract. **Stop and ask
the user before changing** residue mappings, window coverage, embedding/pooling,
acceptance or evidence formulas, representative selection, or introducing
calibration/decision thresholds. Do not silently resolve such a change as a
formatting or performance optimization.

## CLI contract

```bash
pclass-dark infer --seq ACDEFGHIKLMNPQRSTVWY
pclass-dark infer --fasta queries.fasta
pclass-dark infer --fasta queries.fasta --device cpu --workers 8
pclass-dark infer --fasta queries.fasta --format json > matches.json
pclass-dark infer --fasta queries.fasta > matches.txt
```

| Option | Default | Contract |
|---|---|---|
| `--seq TEXT` | absent | One protein sequence; query ID `query` |
| `--fasta PATH` | absent | UTF-8, uncompressed regular FASTA file, one or more records |
| `--library-dir PATH` | shared/cache/bundled selection | `paths.default_library_input_dir()`; resolve explicit relative paths from cwd |
| `--device TEXT` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:<index>` |
| `--workers INTEGER` | `8` | Positive CPU thread budget, not number of model processes |
| `--batch-size INTEGER` | `8` | Maximum windows per model forward; integer 1–64 |
| `--score-batch-size INTEGER` | `4096` | Windows per CPU scoring block; integer 1–65536 |
| `--memory-mib INTEGER` | `1024` | Reference/scoring array workspace budget; minimum 128 MiB; excludes model/runtime and Python metadata |
| `--offline` | false | Forbid model/tokenizer network requests; require cached pinned assets |
| `--format TEXT` | `text` | `text` or `json` |
| `-h`, `--help` | n/a | Same help; no model initialization or computation |

Require exactly one of `--seq` and `--fasta`, including presence checks before
normalizing empty strings. No positional sequence, stdin `-`, compressed-input,
model/revision override, output path, rejected-match view or cluster-subset option
in v1. Both input modes write the report to stdout (on screen by default). Users
redirect it to a file. Logs/progress/device/fallback/errors go to stderr. Disable
ANSI markup/color in redirected and on-screen report text; query IDs must never
be interpreted as Rich markup or terminal escape sequences.

Usage errors (missing/both inputs, invalid options, nonexistent/nonregular input
file) exit 2. Data, integrity, model, resource and busy-root failures exit 1;
interruption exits 130. A closed stdout pipe exits without a traceback using the
normal CLI broken-pipe convention; already emitted bytes cannot be recalled.
No accepted matches is successful inference, exit 0. No persistent report or query
embedding files are created automatically.

## Input validation and identity

Use exact reference residue semantics: uppercase ASCII
`ACDEFGHIKLMNPQRSTVWYXBZUO`, retaining X/B/Z/U/O as their existing model tokens.
Do not substitute unknown residues, strip terminal `*`, expand ambiguity codes,
remove internal gaps or uppercase input silently. For `--seq` and Python sequence
strings, strip outer ASCII whitespace only; reject internal whitespace/newlines.
For FASTA, join wrapped sequence lines after the reference parser's ASCII edge
whitespace stripping. Reject lowercase, non-ASCII residues, internal spaces,
gaps, stop symbols, empty sequences and sequence text before a header.

FASTA IDs are the first whitespace-delimited header token. Reject empty or
repeated IDs (case-sensitive) anywhere in the batch. IDs must be printable,
nonempty, at most 256 Unicode characters, without whitespace/control characters.
Descriptions after the primary ID do not affect scoring and need not be retained
in v1 reports. Preserve record order. Repeated biological sequences under
*different* IDs remain distinct query results. Do not deduplicate their output or
persist query embeddings. Scope window keys by `(query_index, window_index)`;
FASTA IDs such as `X` and `N-X` must not collide after terminal window naming.

Preflight the entire input before loading the model or publishing any report.
Streaming iteration must not hold all input sequences in memory. Reuse/factor the
reference FASTA parser with optional bounded-read limits; keep reference parsing
behavior unchanged when limits are omitted. Read lines with an explicit maximum
length, enforcing per-record limits while accumulating, not after unbounded join.
A symlink to a readable regular input file is allowed (no input is modified).
For file inputs, compare open-file stat size/mtime before and after preflight and
fail on detected changes. Freeze the validated records in an internal normalized
spool, so later original-file changes cannot alter the run.

Initial fixed engineering caps (document in help/API docs):

- Raw input and normalized input spool: each at most 1 GiB.
- Queries: at most 100,000; sequence length: at most 1,000,000 residues per record.
- FASTA header: at most 4,096 UTF-8 bytes; sequence line bounded by record limit.
- Combined temporary input/vector/database/report storage: at most 2 GiB.

Exceeding a cap raises an actionable resource error; never truncate or drop data.
Use an OS temporary directory with private files and remove it in `finally` on
success, errors, interruption or context exit. A system-level crash can leave
normal OS temporary files, not selectable project artifacts. Do not write into
`data/setup-data`, library roots or package resources. Internal temporary spools
are working storage, not user-facing output files.

## Exact reference windowing

Call `setup_data.window_records(query_id, sequence)` rather than duplicating its
algorithm. Use its current `WINDOW_LENGTH=1022` contract:

- Length 1–1022: one `full` window, ID equal to query ID, coordinates `[1, L]`.
- Length >1022: N window `N-<ID>` with `[1, 1022]`, then C window `C-<ID>` with
  `[L-1021, L]`. Coordinates are 1-based, inclusive.
- N/C windows can overlap or have identical sequences; retain both identities
  and evidence. Do not merge them, add sliding/middle windows or average vectors.
- For length >2044, the intervening middle is not examined. Record the windows
  and full query length in the result; do not imply full-residue coverage.

Hash the full normalized sequence and each exact window with
`reference_data.sequence_id` (SHA-256 of ASCII sequence). Full-query hash includes
any unexamined middle. Construct windows in the reference order, never by sorting
window IDs globally. Reject overlimit/empty or wrong-length backend outputs.

## Selected library and availability

Load `relative_scores.load_selected_library(library_dir)` once per run. Require a
validated v2 production selection with its paired final C/B distributions. Legacy
six-fit selections or missing C/B companions fail with an instruction to run
`build-library`; never silently rebuild or migrate during infer. No setup-data,
reference FASTA, reference embedding matrix, profunc publication, GO graph or
cluster-description catalog is read. All model parameters come from the selected
production's validated representation contract and signatures.

"All clusters" means every cluster in the selected production inventory, which
may itself be a user-built subset. Do not discover expected clusters from profunc
or claim full functional-space coverage. Older v2 fit identities lacking the
optional `reference` provenance field remain scoreable through existing loaders.
Do not require reference-description bindings for query scoring.

For each target, use the validated signature status and companion status. A known
unsupported status is unavailable, with its recorded reason; do not manufacture
A=No or B/C=0. Score all available targets, including valid predict-none models.
Corrupt/missing files, mismatched companions/contracts and unknown schemas/statuses
are fatal, not unavailable candidates. If no target is usable, fail before model
loading with `No usable cluster signatures/distributions`; do not present this as
an ordinary no-match query.

Add a read-only `RelativeScores.describe()` returning a frozen mapping keyed by
cluster ID with `status`, `reason`, `C_n`, `B_n`, `final_fit_id`,
`scientific_payload_id` and `representation_contract_id`. It checks pending state
and generation hashes under the existing guard and reads no arrays/quantiles.
Also expose read-only `manifest_sha256` from its existing captured digest.

Add `RelativeScores.for_cluster(cluster_id)`, a context manager yielding a scoped
`RelativeEvaluator` with `evaluate(log_scores) -> RelativeEvidence`. It validates
status/generation under the root guard and its existing reentrant lock, lazily
loads/validates C/B arrays on the first evaluation, then pins that one pair until
exit even if it exceeds the 64 MiB cache limit. No evaluation means no array load.
Closing releases its references; calling the evaluator after closing raises
RuntimeError. The context does not transfer persistent array ownership to callers.
`RelativeScores.evaluate` delegates through this context and a shared empirical
calculation helper; it must not duplicate/change the formulas. Preserve existing
cache limits and finite-input/error behavior. Pinning one target prevents repeated
large-array reloads during a many-block run. Infer must not reach into `_metadata`,
`_digest`, `_cache` or call the quantile-summary method to obtain these contracts.

Run coverage records contain total/evaluated cluster counts and the list of
unavailable cluster IDs/statuses/reasons. Text includes one coverage line; JSON
and Python preserve the full list. Availability is operational/scientific-status
metadata, not an interpretation of query membership.

## Embedding execution and device policy

Use the existing `embeddings.EsmBackend` with model/revision from the selected
validated contract, pinned assets, residue-only final-layer mean pooling,
float32 inference/output, `eval()` and inference mode. Validate through
`library_runtime.representation_contract` before scoring. Do not normalize,
standardize or fit anything on queries. No new runtime dependency is needed;
missing optional model packages must explain installing `[embeddings]`.

For `auto`, prefer a compatible CUDA GPU; unavailable CUDA or CUDA initialization/
model-transfer failure falls back to CPU. Record requested and actual device and
fallback reason (`cuda_unavailable` or `cuda_initialization_failed`, otherwise
null). Put detailed exception text on stderr, not in stable semantic metadata.
Reuse the existing backend's initialization fallback rather than
catching every RuntimeError as a GPU failure. Corrupt assets, contract errors,
missing offline assets and network failures remain errors. Explicit `cuda` or
`cuda:N` fails if unavailable; it does not silently switch to CPU. Explicit CPU
never initializes a GPU model. Use the existing default seed 42 internally;
there is no new inference seed option or randomized sampling.

Fix the execution device once initialization succeeds. Forward-time CUDA OOM or
other embedding failures abort the run and explain reducing `--batch-size` or
using `--device cpu`. Do not silently mix device results, retry individual windows
with different residue policies, or drop failed queries. Initialization fallback
is the approved automatic CPU fallback; runtime CPU migration is outside v1.

One process and one model instance serve the batch. `--workers 8` is the requested
CPU thread budget for embedding/scoring, not eight model replicas. Effective
threads are `max(1, min(requested, affinity CPU count))`, falling back to
`os.cpu_count() or 1` when affinity is unavailable. Scope Torch intra-op threads
and `threadpoolctl` limits to that budget, restoring prior settings in `finally`.
Do not change Torch's global inter-op pool or launch nested process pools.
Serialize this API's model-execution contexts with a process-local reentrant lock
so concurrent sessions cannot race temporary Torch/global numerical settings;
external code changing these settings is outside that coordination contract.
Restore backend-modified precision flags on exit. Record the fixed seed and
actual numerical execution settings; do not promise that all external RNG state
is unaffected by the existing backend's explicit seed initialization.

CPU/GPU and batch-size float32 differences are possible. Preserve exact saved A
threshold comparisons and unrounded B/C calculations on actual computed vectors.
Do not introduce score rounding, threshold tolerances or device-specific
acceptance corrections. Existing real-checkpoint tests use `rtol=1e-4, atol=1e-5`
for vector comparisons; that tolerance is a validation bound, not an A decision
rule and not a guarantee of identical near-threshold classifications.

## Batching, temporary evidence and numerical admission

Model and scoring batches are independent. Extract windows in stable query order
and pass at most `batch_size` window sequences to each model forward. A query's
two windows may span two forwards (`batch_size=1` must work). Write resulting
float32 rows into a private `.npy` memmap of shape `(total_windows, 640)` using
stable global row ordinals. Record `(query_index, window_index)` and metadata for
each row. Keep only one bounded model batch plus one input record in memory.
Do not deduplicate query/window vectors or compute all input embeddings in RAM.
Remove the normalized full-sequence spool once vector generation completes.

After embedding, use **cluster-first scoring across the entire query batch**:

1. Iterate available cluster IDs in sorted order; enter `for_cluster(cluster_id)`.
2. Read the temporary vector matrix in blocks of at most `score_batch_size` rows.
   Call `Library.score` once per block. If no rows are accepted, do not evaluate
   B/C or load that target's reference arrays.
3. Evaluate B/C for accepted rows only through the pinned evaluator and append
   their window keys and paired evidence to the temporary database below.
4. Complete all blocks before moving to the next cluster. Query-stage C/B
   materialization happens at most once per target, rather than per model batch/
   query. The initial loader's existing full mmap/checksum audit is separate and
   must not be bypassed; the no-load rule for all-rejected targets refers to
   additional query-stage materialization, not that initial integrity audit.
5. After all targets finish, verify the library snapshot, release artifact/model
   resources and remove the temporary vector file. Results now depend only on
   the private evidence database and frozen run metadata.

Use stdlib SQLite working storage, never a database under project data roots:

- `queries(query_index INTEGER PRIMARY KEY, query_id TEXT UNIQUE, length INTEGER,
  sequence_sha256 TEXT)`.
- `windows(row_index INTEGER PRIMARY KEY, query_index INTEGER, window_index INTEGER,
  window_id TEXT, kind TEXT, start INTEGER, end INTEGER, window_sha256 TEXT,
  UNIQUE(query_index, window_index))`.
- `accepted(query_index INTEGER, cluster_id TEXT, window_index INTEGER,
  background_tail_fraction REAL, cluster_percentile_le REAL,
  PRIMARY KEY(query_index, cluster_id, window_index))`.

Rows in `accepted` always have A=Yes; preserve B/C as finite IEEE-754 float64 values
without rounding. Repeated inserts are an error, not `INSERT OR IGNORE`. Query and
window metadata are recorded during preflight. Population sizes and availability
remain in frozen cluster metadata. Use parameterized SQL, a 32 MiB page cache, `temp_store=MEMORY`,
`journal_mode=OFF`, and `synchronous=OFF`. This database is disposable working
storage: any SQL/IO failure aborts the run, and a crashed database is never resumed
or published. Transactions commit accepted rows per score block only to bound
work; they are not a durability guarantee. Create schema/indices before inserting
records and use primary-key/prefix scans only; do not introduce unbounded SQL
sorts/temp B-trees. Per-query B ordering is performed by the shared Python reducer.

Set `PRAGMA max_page_count` before inserts from remaining temporary quota divided
by `page_size`; account for input/vector/report files first. Count actual working
file sizes before/after each block and before each report write. Map SQLITE_FULL
to InferenceResourceError. Do not rely on an after-the-fact size check to prevent
an oversized transaction. There are no SQLite journals/WAL files or durable resume
state; no user database or files outside the private working directory are opened
by SQL. Verify the intended indexed scan plans in tests.

Yield results in query-index order from this database. Reduce only one query's
accepted rows at a time through the same pure window-union/representative helper
used by `score_query_windows`; do not implement a second ranking rule in SQL.
The model does not remain resident while the caller reads completed results.
Library arrays are never loaded for all clusters at once. This execution order
preserves scalar scoring and avoids repeated reference IO for large FASTA batches.

Before model initialization, get counts from `describe()`. For available targets
let `L = max(8*(C_n+B_n))` bytes, and `S=score_batch_size`. Required workspace is
`96 MiB + L + 16*640*S + 32*S` bytes: reference cache (64 MiB), SQLite cache
(32 MiB), one additional largest C/B pair, and conservative vector/score workspace.
Reject when this exceeds `memory_mib * 2**20`, reporting required MiB rounded up
and suggesting smaller score blocks or a larger budget. This is a NumPy/reference/
SQLite workspace budget, not a total-RSS/VRAM guarantee: model weights, Torch
activations/model-forward output buffers, validated library/Python records and
renderer metadata are additional.
Input limits, on-disk evidence and block processing bound growth. Actual hardware
allocation failures remain errors; never prune targets/features to fit memory.

The 2 GiB temporary cap includes normalized input, `.npy`, SQLite sidecars and
report spool by logical file size (including allocated memmap extent), checked
incrementally. Remove obsolete working spools between phases to free quota. Too
many accepted matches can exceed the cap: fail with a request to split the input,
not silently discard matches. No new numerical/biological cutoff is introduced.

## Scoring, union and deterministic ordering

Use the verified final signature and paired in-sample reference distributions.
For query window log score `l`, retain existing formulas and inclusive ties:

```text
A = (l >= fitted log cutoff), or No for a predict-none threshold
B = (1 + count(B_log_scores >= l)) / (n_B + 1)
C = count(C_log_scores <= l) / n_C
```

Call the existing scoring/evidence APIs; do not duplicate their scientific
calculation. A uses the fitted training balanced-accuracy cutoff, independently
of B/C. Input vectors/scores must be finite; numerical errors fail rather than
being converted into missing or rejected matches.

For each query independently:

1. Include a cluster iff at least one of its windows has A=Yes.
2. Include exactly one cluster row. Choose the accepted window with lowest
   unrounded B; ties use `(start, end, window_id)` ascending. C and population
   sizes come from that *same* representative window/cluster evidence.
3. Preserve every accepted supporting window with its own paired B/C and
   coordinates, ordered by the original window order. Rejected windows do not
   enter the supporting list or representative choice.
4. Sort rows by `(unrounded B, cluster_id)`. Equal B has equal evidence; cluster ID
   is only deterministic display ordering. No numeric rank implying a tie-break
   by C, no top-k truncation, no extra B cutoff.

No matches returns `matches=()` and `status=no_accepted_matches`, with the text
`No accepted matches`. This does not conclude that no known function exists.
Supported queries with matches use `status=matches`. There are no per-query
likelihood classes or partial-success rows. B/C are descriptive comparisons,
not membership probabilities or calibrated false-discovery rates; high C is a
higher score rank among members, not necessarily greater typicality.

## Python interfaces and immutable records

Implement reusable logic in `pclass_dark.inference`; keep Typer a thin adapter:

```python
make_query_windows(query: QuerySequence, *, query_index: int = 0) -> QueryWindows
score_query_windows(pair: SelectedLibrary, queries: Sequence[QueryWindows],
                    vectors: ndarray, *, embedding_contract: Mapping
                    ) -> tuple[QueryResult, ...]
infer_queries(queries: Iterable[QuerySequence], *, library_dir=None,
              device="auto", workers=8, batch_size=8, score_batch_size=4096,
              memory_mib=1024, offline=False) -> InferenceSession
write_inference_report(session: InferenceSession, stream: TextIO,
                       *, format="text") -> None
```

Use frozen dataclasses with tuple collections and detached, JSON-serializable
`to_dict()` output (native floats/bools/ints, no NumPy scalars). Public types:

| Record | Fields |
|---|---|
| `QuerySequence` | `query_id: str`, `sequence: str` |
| `WindowInfo` | `window_id`, `kind: full|N|C`, `start`, `end`, `window_sha256` |
| `QueryWindow` | `info: WindowInfo`, `sequence: str` (input only) |
| `QueryWindows` | `query_index`, `query_id`, `length`, `sequence_sha256`, `windows: tuple[QueryWindow,...]` |
| `WindowMatch` | `window: WindowInfo`, `accepted=True`, `background_tail_fraction`, `cluster_percentile_le` |
| `ClusterMatch` | `cluster_id`, `accepted=True`, `background_tail_fraction`, `cluster_percentile_le`, `C_n`, `B_n`, `representative: WindowInfo`, `supporting_windows: tuple[WindowMatch,...]` |
| `QueryResult` | `schema=pclass-dark.infer-query.v1`, `query_index`, `query_id`, `length`, `sequence_sha256`, `windows: tuple[WindowInfo,...]`, `status`, `matches: tuple[ClusterMatch,...]` |

Window output contains hashes/coordinates, not repeated sequence strings.
`make_query_windows` performs input normalization/validation and hashing.
`score_query_windows` requires nonempty strictly increasing, unique query indices,
unique IDs within the chunk, exact flattened vector row count/shape, valid window
lengths/coordinates/hashes and a matching representation contract. Reuse existing
floating/finite vector validation without adding a hidden cast/normalization;
actual model output must be float32. It performs no
embedding, fitting, mutation or implicit printing. It need not initialize Torch.
Full-parent provenance in externally constructed `QueryWindows` is caller-supplied;
use `make_query_windows` to derive it from actual complete sequences.

`InferenceSession` is a context-managed, one-shot iterator:

```python
from pclass_dark.inference import QuerySequence, infer_queries

with infer_queries([QuerySequence("protein_1", "ACDEFGHIKLMNPQRSTVWY")]) as run:
    for result in run:
        print(result.to_dict())
```

Construction stores arguments without consuming input/loading models. Entry
validates options, preflights/spools all input, verifies library/coverage, checks
admission, initializes one model context, embeds windows, and computes accepted
window evidence into temporary storage. Model loading happens before long-lived
artifact guards as specified below. If entry fails, it must itself release guards,
thread settings and temporary files; Python will not call `__exit__` on failed
`__enter__`.

After successful entry, `source`, `execution`, `coverage`, `counts` and `query_count`
are available as frozen mappings/scalars. All scoring is complete; iteration
streams completed per-query results from bounded working storage, not early
partial inference. Query indices are zero-based; text headings are one-based.
`__iter__` returns the active one-shot iterator without rewinding; exhaustion keeps
raising StopIteration. `next()` outside the entered context or re-entering a used
session raises RuntimeError. Yield one QueryResult per input record in order.
Only the current query's result rows need to be materialized in Python memory.

Context exit closes/removes remaining spools even after early iteration exit.
Owned model/tensor references and numerical settings are released/restored when
scoring completes, before returning from successful entry; allocator caches may
remain and are not globally flushed. Imports/construction never parse CLI
arguments, download models, launch workers or start inference.

Validate Python booleans strictly (`offline`), integers as real positive ints
excluding bools, strings/formats and paths before any model work. Reuse
`IntegrityError` and `SetupBusyError` for artifact failures; preserve any
specialized maintenance/recovery error if a subsequently merged guard raises it.
Introduce `InferenceResourceError(RuntimeError)` for caps/budget failures
and `InferenceError(RuntimeError)` for backend execution failures with original
cause. Invalid input/options raise ValueError; filesystem IO errors may propagate
as OSError with path context. Errors do not fabricate A/B/C.

## Snapshot lifetime and cleanup coordination

Initial library loading uses existing guarded loaders. Record selector, production
and C/B manifest hashes plus representation identity. Do model resolution/loading
outside the library root guard, so downloading weights does not lock artifacts.
Then acquire guards for both the selector root and actual production source root
(in sorted canonical order; normally the same root). Verify the selector and
source hashes still match the captured snapshot before computing query embeddings.
A changed snapshot fails and requests a retry; never silently attach the already
loaded model to a new selection.

Hold these guards through embedding and cluster-first scoring, then final
source-snapshot verification. Release them before completed-result iteration and
report serialization. Immutable installed resources use existing read-only rules without
creating locks; pending cleanup must not be bypassed by an immutable hint. Root
contention fails promptly. Source cleanup/replacement cannot detach live arrays
mid-run. Honor lazy array generation checks and pending-maintenance markers, including
cache hits, as detailed below. Do not hold the setup-data root: infer does not
consume it.

### Required guard additions on the current branch

The audited `issue#23` base has `transaction_guard`, but does not yet contain the
separate cleanup implementation or its recovery exception/helper. Infer must be
implementable on this base; do not import an absent cleanup module or require a
missing spec as a runtime dependency.

Add a small read-only `artifacts.check_cleanup_pending(root)` helper when absent.
It refuses an existing or dangling-symlink `.cleanup-journal.json` or
`.cleanup-peer.json` under that root with `SetupBusyError` and an instruction to
recover pending maintenance. It never parses a journal, follows a peer to another
root, removes files or performs recovery. If the cleanup feature is merged before
implementation, reuse its helper and propagate its specialized recovery error
instead; do not replace that feature's recovery bypass or introduce a second
journal format.

Inference checks its selector/source roots before initial loading, after acquiring
the run guards, and before completing the report. `RelativeScores.describe()`
uses the same check under its existing source-root guard. Also factor generation
checks in `RelativeScores.evaluate` so its public method holds the source-root
guard and verifies its captured production/C/B manifests before cache-hit as well
as cache-miss evaluation. Avoid duplicate full-manifest hashes in a nested
cache-miss branch. Missing/deleted/changed generations raise IntegrityError with
an instruction to reload. Detached signature-only `Library.score` remains usable
without reading its old source path; only lazy empirical evidence needs this
extra file-lifetime check. Update analytical in-memory fixtures to provide valid
source-generation identity, preserving their independent numerical expectations.
These additions preserve scoring formulas and close a lifetime gap in the current
cache implementation; they do not implement artifact cleanup.

Root guards have already been released before the report is serialized/copied,
so a slow output stream cannot block a new library build. The completed database
and frozen metadata are self-contained.
A Python caller using a preloaded pair with `score_query_windows` is responsible
for guard lifetime across its own batch loop; individual evidence calls retain
existing safety checks and may fail if the source was retired.

## Report schemas, formatting and failure atomicity

The text report contains one coverage line and one labeled section per query in
input order, including no-match queries:

```text
Evaluated 3520/3524 clusters; 4 unavailable
Query 1: protein_1 (length 500; windows 1)
Cluster         A    B ↓       C ↑
cluster_0001    Yes  0.00125%   70%
```

Use six significant digits for percentages (`100*fraction` with `.6g`, then `%`),
always without rounding before sorting. Thus positive B never renders as zero;
scientific notation for tiny values is allowed. Show C=0/100% exactly where
appropriate. Use display-width-aware alignment and literal/escaped identifiers,
not Rich markup. No raw cross-cluster compatibility scores or interpretations in
the compact report. Coordinates/supporting evidence remain in JSON/Python.

JSON is one object with these top-level fields:

- `schema: pclass-dark.infer-report.v1`.
- `source`: `library_dir`, `production_id`, `production_manifest_sha256`,
  `relative_id`, `relative_manifest_sha256`, `representation_contract_id`,
  `probability_calibrated=false`. Paths are resolved; hashes bind the captured run.
- `execution`: `requested_device`, `device`, `fallback_reason`,
  `workers_requested`, `workers_used`, `batch_size`, `score_batch_size`, `model`,
  `revision`, `seed=42`, `inference_dtype=float32`, `torch_version`,
  `transformers_version`. `workers_used` records the effective thread budget,
  not measured simultaneous CPU utilization.
- `coverage`: `clusters_total`, `clusters_evaluated`, `unavailable` list with
  `cluster_id`, `status`, `reason`.
- `queries`: ordered `QueryResult.to_dict()` records, B/C fractions unrounded.
- `counts`: queries_total, queries_with_matches, matched_cluster_rows, windows_total.

The writer computes counts while consuming the session, confirming the emitted
query count equals preflight `query_count`. Reject a partially consumed session:
a complete report must not silently contain only its remaining queries.

Do not include elapsed time or timestamps in semantic records. JSON native floats
must be finite and `allow_nan=False`; preserve zero/one endpoints and full query
IDs. Encode Unicode deterministically; terminal widths never change JSON data.
No rejection rows, implicit descriptions, GO lookups or probability conversions.

CLI serializes into a private temporary report spool while consuming the completed
session. Check combined spool caps incrementally. On validation/scoring/resource/
rendering failure, discard the spool and write only the error to stderr; publish
no normal stdout report. After the report is fully serialized and the session is
closed, rewind and copy the finished report spool to stdout. The CLI owns that
spool outside the session's temporary directory so it survives context exit.
This protects against late input/model errors but cannot make a physical stdout
write atomic if a consumer/disk fails mid-copy.

`write_inference_report` requires a freshly entered, unconsumed session and writes
to its explicitly supplied stream; it does not enter/restart/close the caller's
context, implicitly print, or guarantee rollback of a caller's stream. It counts
serialized UTF-8 report bytes and enforces the combined current-working-file/
report quota even when the supplied stream is not a disk file. The CLI quota counter
includes its separate report spool until session working files have been removed.
The CLI supplies the spool; Python users choose their output stream and handle
stream errors themselves.
Previously yielded Python results remain valid snapshots if a later operation
fails; the session must not silently skip failures and continue.

## Validation and implementation completion

Tests use independent analytical/small-reference examples and fake model contexts,
not expected values copied from implementation. Factor backend execution/context
creation so tests can substitute a deterministic context without installing Torch
or downloading assets; this is an internal seam, not a CLI bypass of contracts.

Required automated cases:

1. Exactly-one-input/format/options, empty/malformed/duplicate FASTA, multiline
   records, uppercase ambiguity tokens, lowercase/gaps/stops rejected, caps applied
   during bounded reads, stable IDs and input order, no stdout on late bad input.
2. Lengths 1/1022/1023/2044/2045 and long-query omitted middle; overlapping and
   identical terminal windows; cross-query `N-X` name collision; repeated sequences
   under distinct IDs; full/window hashes and inclusive coordinates.
3. Independent Gaussian-score and empirical-count oracles through the existing
   APIs: cutoff equality, predict-none, plus-one B, inclusive ties and C endpoints.
   Every usable selected target is scored; unsupported targets are excluded with
   reasons, all-unavailable fails, integrity failures are never soft skips.
4. One accepted terminal window versus rejected other; repeated cluster matches
   produce one row; C comes from the representative window; equal-B coordinate
   tie choice; cluster-ID display ordering; sorting before percentage rounding;
   every accepted supporting window retained and queries never combined.
5. Model/score block independence, `batch_size=1` long query, backend shape/nonfinite
   failures, one reference load per target across all score blocks, no array load
   for all-rejected targets, identical shared-helper reduction from SQLite and
   in-memory scoring, duplicate accepted-row refusal, bounded buffers, workspace
   admission formula, SQLite page quotas/indexed scans and temporary-disk limits.
6. Fake auto GPU success, unavailability/init fallback, explicit CPU and explicit
   CUDA failure; runtime OOM abort without changing device or dropping results;
   worker default=8/affinity clamp/serial mode, no multiprocess model replication,
   context/global-thread/precision restoration on success and failure.
7. Source selection changes during model initialization, cleanup markers, root
   contention, nested source-root lock ordering, packaged read-only operation,
   no setup/profunc/reference dependency, early session exit releasing resources.
8. Exact text/JSON schema and finite native scalars, tiny nonzero B percentages,
   Unicode/markup-safe IDs, no-match sections, output redirection, stdout/stderr
   separation, failed-entry/mid-run report discarded, early iterator exit and broken-pipe behavior.
9. Real cached pinned-model CPU batching/pooling and CUDA/CPU comparison when
   available, using existing opt-in tests without downloads. Mocked backend tests
   require exact result equality across worker/chunk choices; real vectors use
   existing tolerance. Record observed near-cutoff CPU/GPU differences in validation
   findings rather than hiding them with a new threshold tolerance.

Before handoff synchronize CLI help, API docs, optional dependency error text,
README examples, spec status and dated progress results. Run relevant tests and
full regression; report skips/hardware limitations explicitly. No held-out
classification performance is established by engineering tests. Any future
calibrated thresholds or ranking-utility study needs its own training/development/
held-out protocol with homology controls, not an implementation-time guess.

## Readiness audit (2026-10-30)

Checked actual `window_records`, EsmBackend/tokenizer/pooling behavior, normalized
representation contracts, Library.score, RelativeScores/evidence/cache semantics,
selected artifact paths and cleanup guards. Resolved previous unspecified API,
formatting, batch sizing, thread budgeting, input limits, failure/publication and
snapshot lifetime details. Public metadata and scoped-reference-view access are
explicit required API additions; quantile summaries/private cache access are not
substitutes. An IO audit replaced tiny model-batch-first reference reloads with
separate cluster-first score blocks, temporary vectors and bounded SQLite evidence;
this is an execution-order change, not a scientific scoring/ranking change. Scientific rules
remain the user's approved rules. No scientific decision is silently introduced. The branch audit also caught and
removed a dependency on the unmerged cleanup specification/exception; required
minimal guard and cache-hit additions are now specified against current code.

Documentation-only validation: local links, code fences, contract consistency and
scoped Git whitespace checks. No command implementation, model inference or runtime
tests are claimed by this revision.


## Implementation and validation (2026-10-30)

Implemented `infer` in `cli.py`, with reusable `inference` entry points, frozen
records/shared reduction in `inference_records`, scoped execution in
`inference_backend`, and explicit report serialization in `inference_render`.
Added bounded FASTA reading, read-only pending-maintenance checks, public relative
metadata/scoped evaluators and generation checks on cache hits. The existing
Gaussian scoring, fitted acceptance and inclusive empirical B/C formulas are
unchanged. No scientific deviation or new threshold was introduced.

The context owns private JSONL/SQLite/memmap storage, validates all queries before
model initialization, guards/rechecks its selected generation, scores cluster
first, and releases the model/guards before result iteration. The CLI publishes
only a completely serialized report. Python streams remain caller-owned. README
and CLI help document commands, resource limits, optional embedding installation,
stdout reporting, device behavior and Python integration.

Validation: 56 focused inference/reference/evidence tests pass, including independent
window coordinates/hashes and empirical evidence, paired representative reduction,
native-fraction sorting before rounding, CLI failures without stdout, batch/worker
consistency, lazy reference reads, stale generations, file replacement, failed-entry
cleanup, invalid vectors/backend output, quotas and indexed result retrieval.
Device/thread restoration and initialization fallback use fake backends. The real
cached pinned ESM CPU pooling/batching test passes offline; the CUDA comparison
is skipped because no compatible GPU is available. Ruff lint/format, local links
and scoped whitespace checks pass. Full regression passed 290 tests with two opt-in skips; the final focused
run passed 56 tests after additional edge cases and validation hardening. See the
[progress log](../research/progress.md) for commands, timing and limitations. These are engineering checks, not held-out classification performance.
