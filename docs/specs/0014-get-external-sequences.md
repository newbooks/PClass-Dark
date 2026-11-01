# SPEC-0014: Collect experimentally annotated external UniProt sequences

Status: implemented; regression and smoke validation completed 2026-10-31.
Full-source acquisition and biological evaluation have not been run.

## Purpose and provenance

Add `scripts/get-external-seq.py` to download GOA UniProt annotations, select
experimentally annotated molecular-function proteins outside the local Swiss-Prot
reference, retrieve their full sequences, and retain one accession per identical
sequence using the most recent qualifying annotation date. Process all eligible
accessions; there is no sampling cap or minimum dataset size.

Reference implementations inspected: neighboring PClass
`src/pclass/audit/external.py` (`_select_accessions`, `_fetch_sequences`,
`_build_parent_records`) and `docs/specs/audit.md`; PClass-Dark
`reference_data.py` and `cluster_descriptions.py`; profunc `runtime.py`,
`go/inspect.py`, and `swissprot/annotations.py`. Keep the new implementation and
its policy/constants/tests here. PClass is not a runtime dependency; installed
profunc remains the existing supported reference provider (SPEC-0012).

Differences from PClass's audit: no sampling, embedding, prediction, windowing,
exact Swiss-Prot sequence exclusion, or union of annotations across duplicate
accessions. Deduplication selects a single accession by annotation recency. The
local profunc Swiss-Prot annotation parser's broader evidence policy must not
replace this script's explicit experimental-only allowlist.

## Interfaces and inputs

The script is a thin adapter to importable `pclass_dark.external_sequences`
logic. This operation is available through the standalone script and Python API;
it is not registered in the `pclass-dark` command hierarchy. Python interface:

```python
collect_external_sequences(
    *, work_dir: str | Path,
    go_graph: str | Path | None = None,
    swissprot_accessions: str | Path | None = None,
    reference_fasta: str | Path | None = None,
    gaf: str | Path | None = None,
    refresh: bool = False, workers: int = 4,
    batch_size: int = 100, retries: int = 3,
) -> ExternalSequenceResult
```

Paths resolve against the caller's cwd before any work; `work_dir` must already
exist. `ExternalSequenceResult` is frozen, containing absolute `fasta_path`,
`report_path`, `manifest_path`, `run_id`, and a tuple of frozen `StageCount`
records (`stage`, `annotation_rows`, `accessions`, `sequences`; unavailable counts
are `None`). Separate private transport injection points permit fixture tests;
public callers need no HTTP-specific objects. Imports do not resolve references
or do work.

All interfaces accept an explicit work directory in Python; the standalone script
always use the invocation's current working directory, never the script/repo
location. No CLI output-directory option. Options:

- `--go-graph PATH`: local validated MF graph; default to active profunc dataset
  graph via `profunc.runtime.resolve_runtime()`, verifying its manifest before
  loading. Use its existing graph loader and schema, including namespace checks.
  An explicit override is a trusted local profunc-format pickle, hashed before
  loading; no manifest is required for that override, and its trust/source is
  reported. Never unpickle downloaded/untrusted data automatically.
- `--swissprot-accessions PATH`: optional UTF-8 file containing one accession
  per line (blank lines and `#` comments allowed). Without it, obtain the complete
  source dataset `accession.fasta` through the SPEC-0012 profunc resolver and
  checksum contract, and collect every FASTA accession, including unclustered
  records. Cluster memberships alone are not a complete Swiss-Prot exclusion set.
  Record whether the input is a user-supplied list or the validated reference.
- `--reference-fasta PATH`: explicit source FASTA for default profunc exclusion
  resolution; use `profunc_fasta_path(runtime, reference_fasta)` and verify its
  size/hash against `profunc_reference_contract(runtime)`. Mutually exclusive
  with `--swissprot-accessions`. Its default is `None`.
- `--gaf PATH`: optional local gzip source for reproducible/offline annotation
  parsing; otherwise download/cache `goa_uniprot_all.gaf.gz` in cwd from
  `https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz`.
- `--refresh`: replace the cached annotation download and start a new retrieval
  generation, preserving prior successful outputs until the new generation is
  ready. With `--gaf`, reparse the supplied file and refetch sequences, without
  downloading or altering that file. Without refresh reuse only validated caches.
- `--workers` (default 4; range 1–32), `--batch-size` (default 100; range 1–100),
  `--retries` (default 3; range 0–10). Serial retrieval uses one worker.

Explicit inputs must exist and validate; do not silently substitute references.
Report resolved sources and snapshot hashes. Swiss-Prot exclusion means absent
from that local snapshot, not verified unreviewed status in today's UniProt.

Examples (full acquisition was not executed during implementation validation):

```bash
# Installed package; profunc reference defaults; outputs in the shell cwd.
python /path/to/PClass-Dark/scripts/get-external-seq.py
python /path/to/PClass-Dark/scripts/get-external-seq.py --workers 1
# Explicit local annotation/graph/exclusion sources; sequence retrieval is online.
python /path/to/PClass-Dark/scripts/get-external-seq.py --gaf /data/goa_uniprot_all.gaf.gz \
  --go-graph /data/go_graph.pkl --swissprot-accessions /data/swissprot.txt
```

`--gaf` avoids only the annotation download; it does not promise offline sequence
retrieval. No offline-output mode, configurable source URL or reset command is
part of version 1. `--refresh` is the supported explicit recomputation action.

## Filtering and data contracts

Stream-decompress the entire GAF with strict UTF-8 decoding; accept supported
GAF 2.1/2.2 headers and 17-field rows. Unsupported headers, malformed noncomment
rows or invalid dates in otherwise qualifying rows fail with line-number
diagnostics. Invalid accession IDs on UniProtKB rows are skipped and counted
before downstream filters, with bounded examples as specified below. Blank lines and `!` comments do not count as records.
Use documented GAF columns (one-based): DB 1, object ID 2, qualifiers/relation 4,
GO ID 5, evidence 7, aspect 9, annotation date 14. Source:
[GO Consortium GAF specification](https://www.geneontology.org/docs/go-annotation-file-gaf-format-2.2/).

Apply filters in exactly this order:

1. Require DB exactly `UniProtKB`; reject/count rows with invalid protein
   accession IDs before accumulating downstream row/distinct-parent counts.
2. Exclude rows with a `NOT` token in the pipe-delimited qualifier field.
3. Require aspect exactly `F`.
4. Keep evidence exactly in `EXP, IDA, IPI, IMP, IGI, IEP, HTP, HDA, HMP, HGI, HEP`.
5. Require the GO ID to be an active molecular-function node in the resolved
   local graph. Reject absent, obsolete and other-namespace nodes. Do not remap
   obsolete/alternate IDs or infer ancestor annotations implicitly.
6. Normalize the object ID by removing a terminal numeric isoform suffix
   (`P12345-2` -> `P12345`); retain a mapping to observed source IDs. GAF column
   17 is not an alternative accession source. Group qualifying rows by parent
   accession, keeping all distinct qualifying GO IDs and evidence/date provenance.
7. Remove parent accessions in the normalized local Swiss-Prot accession set.
8. Retrieve canonical full sequences for remaining parent accessions.
9. Deduplicate obtained sequences using the rule below.

Positive and negative rows are filtered independently; a NOT row neither becomes
positive evidence nor cancels a distinct retained positive row. Isoform-to-parent
normalization is an explicit approximation: it transfers isoform annotations to
the parent sequence, and must be identified in metadata, not claimed as exact
isoform sequence evidence.

### Annotation recency and sequence deduplication

For each accession, `latest_annotation_date` is the maximum valid calendar date
(`YYYYMMDD`) in GAF column 14 **among rows surviving filters 1–5**. Unsupported,
negative, non-MF and graph-invalid annotations cannot make an accession newer.
This is an annotation date, not a sequence update date or evidence-strength score.
Keep the accession's entire qualifying MF set, not only terms on its newest date.

Group by exact full sequence after removing FASTA line breaks and uppercasing.
Use SHA-256 for indexing and verify sequence equality before merging a group.
Choose the accession with the greatest `latest_annotation_date`; break date ties
by lexicographically smallest normalized accession. Preserve only the winner's
GO terms and annotation provenance: never union terms from discarded duplicates.
Retain losing accessions' original annotations and duplicate-to-winner mappings
in private state for auditability; only winner annotations appear in public output. All
results and ordering must be independent of GAF order and worker completion order.

Sequences must be nonempty uppercase ASCII amino-acid strings. Allow the 20
standard residues plus B, Z, X, U, O, J; reject gaps, stops and other characters,
counting those accessions separately. Do not trim, truncate or replace residues.
No sequence-length restriction and no exclusion by identity to Swiss-Prot.

## Retrieval, resource use and failure behavior

Fetch selected accessions, lexicographically sorted and partitioned into batches
of at most `batch_size`, using HTTPS GET
`https://rest.uniprot.org/uniprotkb/stream` with URL-encoded parameters
`query=(accession:P12345 OR accession:Q12345)`, `format=fasta`,
`includeIsoform=false`. Use one exact `accession:` clause per requested ID.
A batch is a complete stream response, not a page; do not use the search
endpoint or silently stop after its first page. Do not add a `reviewed:false`
query filter: the specified exclusion is the local reference. Never download
the complete TrEMBL FASTA. This replaces the old PClass accessions endpoint.
Reference: [UniProt query API](https://www.uniprot.org/help/api_queries) and
[UniProt API paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12230682/).
Validate returned IDs against requested parents; do not silently remap obsolete
accessions, union aliases, or attach terms to unexpected IDs. Report missing IDs
as unresolved; distinguish a successful response omitting an ID from a failed
HTTP request. Conflicting repeated sequences for an accession are fatal.

Use bounded I/O threads, capped pending batches (at most twice workers), request
connect/read timeouts (10/120 seconds), and retries for connection failures,
timeouts and HTTP 408/429/500/502/503/504. `retries` is additional attempts after
the first; backoff is `min(60, 2**retry_index)` seconds, starting at index 0.
For Retry-After seconds or HTTP dates, wait at least that duration; if over 300
seconds abort with resumable state instead of retrying early. Other non-2xx
responses fail immediately. Use `requests>=2.32,<3` as a declared direct runtime
dependency, per-thread sessions, streamed responses, and close responses on all
paths. The GAF downloader uses the same retry rules; a failed stream retries
from the start of its temporary download, not by appending bytes. Exhausted
network failures abort publication; successful omissions and invalid sequences
are counted and excluded. Validate gzip integrity to EOF;
interrupted downloads remain `.part` files and never count as complete caches.

Use disk-backed SQLite aggregation/state in `.external-seq/`, streaming GAF,
batch FASTA parsing and final output, and indexed sequence hashes. Do not retain
all GAF rows or sequences in RAM. Bound transaction sizes and queues; one state
writer, no nested numerical threads or GPU requirement. CPU mode works without
embedding/model dependencies. Record download sizes, disk usage/progress and
estimated required free space when available; never assume this complete GOA
file is small. Multiprocessing is unnecessary for this streaming/I/O workload.

Resume only when input SHA-256 identities, filtering policy version, reference
contract, retrieval configuration and batch identities agree. Store source URL,
retrieval timestamp, GAF headers, graph/Swiss-Prot hashes, required sequence response release headers, optional download headers,
sequence response hashes, and rejection counts. A cached retrieval
is a dated observation, not an assurance of current UniProt contents. Detect
incompatible release headers across batches and refuse mixed-release publication.
Acquire a cwd-specific writer lock; fail clearly on concurrent runs. Check local
input identities again before publication. Surface worker failures and remove
only this run's temporary files; preserve prior successful outputs on failure.

## Artifacts and reporting

The two primary outputs are fixed names in cwd:

### `external_sequences.fasta`

One full sequence per winning accession, sorted by accession, wrapped at 80
characters, UTF-8 with LF. Header contract:

```text
>P12345 go_mf_terms=GO:0003674|GO:0003824 latest_annotation_date=20260712
```

GO IDs are sorted and unique and belong only to the retained accession. The
example is illustrative, not a claim that these terms qualify in real data.
The first whitespace-delimited header token remains the plain accession.

### `external_filtered.txt`

Human-readable, left-aligned stage labels and right-aligned integer columns,
with a header, separators, no thousands separators, and explicit units. Report
both remaining annotation rows and distinct normalized accessions at each GAF
filter, because row counts are not sequence counts. Actual sequences are known
only after retrieval; use `-` for inapplicable values. Required stage order and labels:

```text
Stage                              Annotation rows  Accessions  Sequences
---------------------------------  ---------------  ----------  ---------
Input GAF records                                0           -          -
UniProtKB                                        0           0          -
Without NOT                                      0           0          -
MF aspect F                                      0           0          -
Experimental evidence                            0           0          -
Present in local MF graph                        0           0          -
Grouped parent accessions                        -           0          -
After local Swiss-Prot exclusion                 -           0          -
Retrieved full sequences                         -           0          0
Valid full sequences                             -           0          0
Deduplicated (latest annotation)                  -           0          0
```

Zeros illustrate formatting only. Distinct-accession counts for filtering rows
use the same parent normalization, without changing filter order. At retrieval,
count sequence records before cross-accession deduplication. Add a separate
rejection summary for Swiss-Prot exclusions, unresolved IDs, invalid sequences,
and discarded exact duplicates, and provenance/policy summary including the
experimental allowlist, date/tie rules, timestamps, input hashes and final FASTA
SHA-256. The report does not contain its own hash or a manifest hash: the final
manifest contains both primary output hashes, avoiding circular digests.
Do not imply input GAF accessions cover all UniProt proteins: only annotated
proteins are represented. Distinguish zero results from a failed run: successful
zero results publish an empty FASTA and complete zero-count report.

Cache/state and a small `.external-seq/manifest.json` are auxiliary artifacts,
not additional primary datasets. Stage both primary files on the same filesystem;
use backups and a recovery journal to publish or roll back the pair together.
Plain two-file replacement is not atomic. Mark completion only after both file
hashes match the manifest; recover an interrupted publication before resuming.
Logs/progress go to stderr; stdout gives final paths and summary only.

## Normative implementation details

These contracts refine the sections above; none are deferred to implementer
policy choices.

### Local reference adapter and parser

Resolve profunc only if at least one default input is needed. With explicit graph
and Swiss-Prot list, profunc resolution is unnecessary. When defaults are used,
resolve once, capture manifest and source paths, verify graph member size/SHA-256
before loading, and verify the whole FASTA via `profunc_reference_contract`.
At completion re-resolve and require the same runtime manifest and input hashes.
Do not call `_read_reference` for exclusion: it retains only clustered sequences.
Iterate the entire source FASTA instead. Accept plain accession and standard
`sp|ACCESSION|NAME` headers; the first AC token is primary. Duplicate source
accessions are an input error. Do not infer secondary ACs absent from that local
FASTA; report this snapshot limitation. User lists are likewise primary/specified
IDs, normalized and deduplicated, not expanded through network mappings.

The actual graph is `{'meta': ..., 'terms': {GO_ID: term_record}}`, validated by
`profunc.go.inspect._load_go_graph`, with MF namespace in `meta`. Profunc's
builder removes obsolete nodes; term records need not repeat namespace or have
an obsolete flag. Require `GO:[0-9]{7}` keys and dict records; reject a term when
an explicit `obsolete`/`is_obsolete` flag is true or an explicit `namespace` is
not `molecular_function`. Otherwise graph membership under its validated MF
metadata suffices. Reject malformed graph records as input errors rather than
silently dropping them. No IC threshold or requirement for a nonzero count.

Validate accession parents using fullmatch of
`(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})`;
optionally strip one suffix `-[1-9][0-9]*` before validation. Do not uppercase or
strip arbitrary suffixes; trim only surrounding whitespace. On UniProtKB rows,
validate/normalize once before accumulating any distinct-accession counts.
The parent pattern follows [UniProt accession syntax](https://www.uniprot.org/help/accession_numbers).
Other DB IDs need not follow UniProt syntax. Strip CR/LF, split on tabs preserving
empty trailing fields, require exactly 17 columns. Require exactly one supported
`!gaf-version: 2.1` or `2.2` declaration before data; repeated identical version
headers are allowed, conflicting ones fail. An empty GAF without a header fails;
a valid header-only GAF succeeds. Parse dates only after graph acceptance;
require eight digits and a valid Gregorian date, without a moving 'today' cutoff.
Qualifiers split on `|` with token whitespace trimmed, case sensitive.

### Disk-backed aggregation and count reconciliation

SQLite schema version 1 and policy version 1 are fixed for this implementation.
JSON state and manifests also use `schema_version: 1`; unknown versions fail
without modification. SQLite is private to this tool and has these logical tables:

- `stage_accessions(accession PRIMARY KEY, stage_mask INTEGER)`: one row per
  UniProtKB parent, bit i set when any of its rows survives GAF filter stage i,
  i=0..4. Insert/update using bitwise OR. Together with five integer row counters
  this computes exact distinct counts without storing every GAF row.
- `annotations(accession, source_id, go_id, evidence, annotation_date,
  PRIMARY KEY(all five fields))`: unique qualifying annotation tuples. Identical
  input rows still increment stage row counters but not this provenance table.
- `candidates(accession PRIMARY KEY, latest_date, excluded_swissprot)` and
  `terms(accession, go_id, PRIMARY KEY(accession, go_id))` derived after parsing.
- `batches(batch_id PRIMARY KEY, request_ids_json, response_path, response_sha256,
  release, release_date, status)`; ID is SHA-256 of sorted IDs joined with LF and
  a final LF. `request_ids_json` is a canonical JSON array.
- `sequences(accession PRIMARY KEY, status, sequence, sequence_sha256)` with status
  `valid`, `invalid` or `unresolved`; only valid rows retain sequence text.
- `representatives(accession PRIMARY KEY, winner)` including self-mappings.

Store counters and source metadata in run JSON. GAF header metadata contains
the validated version, total comment-line count, first 1000 comment lines and an
explicit truncation flag, bounding report/state memory for unusually long headers. Commit at most 10000 GAF rows per
transaction. No partial parse resume: interruption rolls back/discards that
run's parse DB and reparses the source; do not attempt gzip byte-offset seeking.
Completed parse state is reusable only after full gzip verification and source
hash validation. Query by indexed keys and stream ordered cursors. Deduplication
sorts valid rows by hash, date descending and accession ascending and verifies
full sequence equality; distinct strings sharing a hash get distinct groups.

Let A be parents after Swiss-Prot exclusion, R returned requested IDs, U missing,
I invalid returned sequences, V valid, D duplicate losers, W final winners.
Require `A = R + U`, `R = I + V`, `V = D + W`; FASTA records equal W. GAF row
counts and distinct counts separately decrease monotonically across filters;
'Grouped parent accessions' equals the final graph-filter distinct count.
Retrieved/valid/final accessions equal their sequence counts. Exact repeated
response records for one ID count once; conflicting repeats fail. A nonempty
candidate set with every ID unresolved remains a successful zero-output run,
with explicit unresolved counts and warning, not a fabricated network success.

For deterministic alignment, render column widths as max(header length, maximum
cell length), labels left-aligned, counts right-aligned, separated by two ASCII
spaces. Separators use `-` repeated per column width. Emit one LF per line, no
trailing spaces. Provenance follows a blank line with stable key ordering.
Stage values and FASTA bytes must match across worker counts; run timestamps,
network metadata and the resulting report bytes may differ across fresh runs.
A completed cached run returns its existing outputs unchanged.

### HTTP response validation and snapshot identity

Decode sequence response bodies as UTF-8 strictly. Accept only FASTA headers
`>sp|ACC|...`, `>tr|ACC|...` or `>ACC ...`; normalize the accession using the same
parent rule. Explicit isoform response IDs are unexpected when canonical was
requested and fail rather than silently replacing parents. Ignore blank sequence
lines; remove line endings only, uppercase ASCII letters, and apply the residue
allowlist. Sequence before the first header and invalid headers fail the batch;
a well-formed requested header with an empty/invalid sequence counts as invalid.
An empty HTTP 200 body is a successful response omitting every requested ID.
Only 200 is accepted as a successful sequence response. HTTP errors and decode or
truncated-stream errors are not converted into unresolved accessions.

Require both `X-UniProt-Release` and `X-UniProt-Release-Date` on successful sequence
responses, with identical values across all batches in a generation. Missing or
changed headers fail the run; retain state for diagnosis, requiring `--refresh`
to obtain a fresh consistent retrieval generation. GOA and graph releases may
be older and need not equal the retrieval release; explicitly report each rather
than suggesting they form one synchronized release. Even if there are zero
candidates, record null retrieval release and make no sequence HTTP calls.

### Cache identity, overwrite policy and pair publication

Use filelock on `.external-seq/lock` with fail-fast contention. Reject symlinks
at tool-owned state/output/download paths and reject input paths overlapping
outputs/state; never follow them for deletion or replacement. Inputs must be
regular files. Existing unrelated output files (no matching completed tool
manifest) cause a refusal before network work; move them aside manually.

State layout: `.external-seq/runs/<run_id>/` contains `state.sqlite`,
`run.json`, `batches/<batch_id>.fasta`, staging files and rollback backups;
`run_id` is a UUID hex. `.external-seq/active.json` selects the resumable run.
The cwd GAF cache is registered by source URL, bytes, SHA-256, download time and
headers in `.external-seq/gaf.json`; an existing unregistered GAF must be supplied
explicitly with `--gaf` or replaced by `--refresh`. A successful fresh download
replaces that cache only after gzip verification. Compute the compressed-file
SHA-256 while downloading, and register it only after complete validation. Failed refresh preserves the
previous cached file and prior outputs. Retain old run directories; no automatic
cache pruning in version 1.

`run.json` records schema/policy version, canonical absolute input paths and
hashes, source contracts, configuration and phase (`parsing`, `fetching`,
`deduplicating`, `ready`, `complete`, `failed`). Identity includes source content,
reference contract, batch size, retry policy and residue/date/filter rules;
worker count is execution metadata and does not invalidate valid completed
batches. Resume a compatible incomplete run without `--refresh`; rehash batch
files before reuse, redownload corrupted batches. Resume refuses a differing
identity with instructions to use `--refresh`; never combine old/new policy or
reference inputs. Refresh creates a new run and re-fetches every sequence batch.
Compatible complete outputs are verified by manifest hashes and returned
unchanged. Modified final outputs are refused even with refresh: restore the
recorded contents or move both primary files and `.external-seq/` aside before
starting again. Refresh recomputes stale sources/caches; it does not authorize
overwriting manual edits to final outputs.

Write each batch to a per-run temporary file, validate to EOF, then atomically
rename and commit its DB record. Incomplete/unregistered batch files are ignored
and removed only inside the selected run. A missing batch record never counts as
completed. Persist HTTP metadata alongside every batch, including release/hash.

For publication, under the writer lock:

1. Finish and fsync both staging files, and their directory. Write a prospective
   manifest with `schema_version`, `policy_version`, `run_id`, immutable input
   contract, counts, retrieval metadata, and `{path,size_bytes,sha256}` for both
   outputs. Timestamps are UTC ISO-8601. No self-hash.
2. Copy any previous tool-owned output pair and completed manifest into the run's
   backups, fsync them. Record absence for a first run. Reject changed previous
   output hashes before proceeding. Atomically write and fsync
   `.external-seq/publication.json` describing old hashes/backups and new hashes/
   staged files. Before this journal is durable, never modify public outputs.
3. Replace FASTA, then report, fsync cwd, replace `.external-seq/manifest.json`
   with the prospective manifest and fsync its parent. Set run phase `complete`.
4. Remove the publication journal only after verifying both outputs against the
   new manifest, then fsync the state directory. Backup pruning is not required.

On startup recover publication before validating/resuming run state: if both
public files and completed manifest match the journal's new hashes/run ID, finish
commit bookkeeping. Otherwise restore old backups and old manifest (or remove
newly created outputs on a first run), fsync, then remove journal. Rollback is
idempotent; retain backups until journal removal. Mark the interrupted run
`ready` after rollback so a compatible resume can restage and republish it. Missing/corrupt required backup
is a recovery error: preserve journal and stop. Uncoordinated readers can observe
a transient mixed pair; consumers requiring consistency must take this same lock
or verify both against the completed manifest. No claim of atomic two-file reads.

Exceptions: `ValueError` for configuration/format/schema errors; `OSError` for
filesystem/network I/O; subclasses of `RuntimeError` named `ExternalDownloadError`,
`ExternalSnapshotError` and `ExternalRecoveryError` for exhausted transport,
release/input changes and recovery failures. CLI catches these, prints actionable
stderr, exits 1; usage errors exit 2; Ctrl-C exits 130 after closing workers and
leaving resumable state. Help performs no reference resolution or network work.
Use a script-specific Typer adapter with `-h`/`--help`; the script runs
with an installed/editable package from any cwd, without modifying `sys.path`.

## Validation and completion expectations

Automated coverage uses independent small fixtures and mocked HTTP:

- Every filtering stage, qualifier token boundaries, exact allowlist, MF graph
  membership/namespace/obsolete rejection, malformed rows and invalid dates.
- Parent/isoform grouping, repeated terms, independent positive/NOT rows,
  complete Swiss-Prot exclusion including unclustered source FASTA accessions.
- A newer rejected annotation must not change a winner; same-sequence candidates
  with different dates and disjoint terms select only the newest candidate's
  complete term set. Equal dates choose lexicographically, independently of order.
- Full sequence parsing, residue rules, missing/unexpected/conflicting returned
  IDs, timeouts, retries, rate limiting, release mismatch and truncated gzip.
- Serial/threaded identical results, bounded queues, interrupted download/resume,
  stale cache rejection, worker failure, publication interruption and rollback.
- Independently calculated stage counts and aligned report golden fixture;
  FASTA accession/term/date contract, zero results and cwd output placement.
- Python API and standalone Typer script integration/help without import-time work.

Implementation must synchronize README/API/CLI help and dependency declarations,
run relevant tests/lint/format checks, record results in the progress log and
update this status. A real full-source acquisition is a separately recorded run,
with source identities, command, code revision, resource use and artifact paths;
synthetic tests are not biological held-out performance. Local Swiss-Prot
accession exclusion alone does not establish homology-independent evaluation.


### Required independent acceptance fixture

Use a tiny graph with exactly two keys: `GO:0003824` and `GO:0005488`.
Use a proper MF metadata header and literal expected counts, not production
calculations to derive them.
Create these 11 well-formed GAF rows in arbitrary order:

| Row | DB / qualifier / aspect / evidence / GO | Parent | Date |
| --- | --- | --- | --- |
| 1 | OtherDB / enables / F / EXP / GO:0003824 | other-id | 20200101 |
| 2 | UniProtKB / NOT / F / EXP / GO:0003824 | P00001 | 20200101 |
| 3 | UniProtKB / enables / P / EXP / GO:0003824 | P00001 | 20200101 |
| 4 | UniProtKB / enables / F / IEA / GO:0003824 | P00001 | 20200101 |
| 5 | UniProtKB / enables / F / EXP / GO:9999999 | P00001 | 20260101 |
| 6 | UniProtKB / enables / F / EXP / GO:0003824 | P00001 | 20200101 |
| 7 | UniProtKB / enables / F / IDA / GO:0005488 | P00001-2 | 20210101 |
| 8 | UniProtKB / enables / F / EXP / GO:0003824 | P00002 | 20220101 |
| 9 | UniProtKB / enables / F / EXP / GO:0003824 | P00003 | 20230101 |
| 10 | UniProtKB / enables / F / EXP / GO:0005488 | P00004 | 20240101 |
| 11 | UniProtKB / enables / F / EXP / GO:0003824 | P00005 | 20250101 |

Expected annotation rows at input/DB/no-NOT/MF/evidence/graph stages:
`11,10,9,8,7,6`; distinct parents after each UniProtKB stage: `5,5,5,5,5`.
Exclusion list contains P00003: five grouped parents become four candidates.
Retrieval returns P00001=`AAAA`, P00002=`AAAA`, P00004=`AA*`; P00005 omitted.
Thus R=3, U=1, I=1, V=2, D=1, W=1. Only P00002 is published, with
GO:0003824 and date 20220101. P00001's rejected 2026 annotation must not change
this choice; its qualifying binding term must not leak into the winner.
Add a separate tie case and row-order/thread-count permutations.


## Implementation outcome (2026-10-31)

Implemented the library coordinator in `external_sequences.py`, reference/filter/
SQLite calculations in `external_sequence_data.py`, HTTP/cache/publication logic
in `external_sequence_io.py`, and frozen records/errors in
`external_sequence_types.py`. Added the standalone Typer adapter and
`scripts/get-external-seq.py`, the direct requests dependency, sdist script/docs
inclusion, README examples and [API/operation documentation](../external-sequences.md).

The scientific policy is unchanged from this specification: all eligible parents,
experimental-only positive MF rows in the local graph, complete local Swiss-Prot
accession exclusion, exact-sequence groups and winner-only MF terms selected by
latest qualifying GAF date with lexical ties. Input/resource controls use CPU
streaming and bounded I/O threads; no embedding model or GPU is involved.

Implementation refinements recorded during validation:

- Final output ownership is stricter than the earlier 'mismatch requires refresh'
  wording: manual edits are protected even during refresh. Source/configuration
  changes and failed release snapshots still use refresh normally.
- GAF cache replacement retains the prior compressed-file inode through a hard
  link and backed-up metadata, avoiding a third full source copy. Failed
  registration restores the old cache; the next invocation repairs interrupted
  registration. Primary outputs continue to use the documented journal protocol.
- Source-format gzip truncation is an actionable ValueError; database failures
  are surfaced as ExternalRecoveryError. Invalid Retry-After advice falls back
  to exponential backoff; valid HTTP date/seconds advice is respected.
- Parsing logs counts every million rows, and GOA downloads log every 128 MiB.
  Report provenance bounds retained GAF comments as specified above.

Validation results and exact commands are recorded in the
[progress log](../research/progress.md). The full regression run passed 432 tests
with two opt-in skips and one existing Torch environment warning. Final focused
external-sequence/CLI validation after hardening is recorded there separately.
Clean code-only wheel/sdist builds include the module, standalone script (sdist)
and requests dependency. The real local reference adapter validated 10041 MF terms
and 575748 Swiss-Prot accessions. A 10000-row historical GOA prefix passed parsing
with independent stage counts; a live one-accession UniProt stream request returned
142 residues and both release headers. No complete GOA download, full external
dataset publication, embedding experiment or held-out biological assessment was
performed. Large-source runtime and peak disk estimates remain unmeasured.


### Standalone interface revision

At the user's request, removed the package subcommand. The standalone script
uses `external_sequence_cli.py` independently of the root command registry;
`collect_external_sequences` remains the importable API. Tests require the
removed package command to fail with usage exit code 2. Scientific behavior,
options, output placement and recovery contracts are unchanged.


### Download progress display revision

GOA progress logs use decimal KB (1000 bytes) with comma-separated integer
counts, and display downloaded/total KB and one-decimal completion percentage
when a positive Content-Length describes the unencoded response. If the total
is absent, malformed or not comparable because HTTP content decoding is active,
report downloaded KB and explicitly say the percentage is unavailable. Retain
the 128 MiB logging interval and emit a final EOF progress message.


### 2026-11-01 — Reject malformed annotation accession IDs without aborting

UniProtKB-labelled GAF rows whose object IDs cannot normalize to valid UniProt
parent accessions are now rejected and counted instead of aborting a complete
parse. This applies generically, including `CPX`/`CPX-...`, empty IDs and malformed
isoform suffixes; never reinterpret such records as proteins or repair identifiers
by guessing. This supersedes the earlier fatal-invalid-GAF-accession rule. Local
reference lists, returned sequence IDs, dates and structural/header/gzip/encoding
validation retain their strict contracts.

Input GAF record counts include rejected accession rows, while all downstream
UniProtKB/filter/distinct-parent counts exclude them. Persist
`invalid_accession_rows` and up to ten `{line, source_id}` examples in parse
metadata; add the rejection count to the final report and manifest. Warn for the
first ten examples and summarize the total at EOF, without unbounded logging or
retaining every malformed row. All GAF line/parse-progress numbers in logs use
comma-separated decimal integers, including graph-qualified row counts.

This is rejection/error-handling hardening: valid-row scientific policy and
artifact schemas remain unchanged. Previously successful strict parses contain
no invalid accessions and remain valid caches. Failed incomplete parses restart
using their existing identity/download; no refresh or source redownload is needed.


The actual failure was verified in the downloaded 2026-10-11 GOA snapshot:
physical line 648144824 has 17 columns, DB `UniProtKB`, object `CPX-10041`,
type `protein_complex`, aspect `P`, evidence `IEA`, annotation date 20261009.
The next two rows use `CPX-1014` and `CPX-10161`. These are well-formed complex
annotations, outside the protein-accession contract; normalizing the first
identifier as if it were a protein isoform produced the misleading bare `CPX`
error. Captured the three rows in `tests/fixtures/goa_nonprotein_ids.gaf` as an
independent regression fixture. Invalid-accession counters refer to eligibility
for this protein pipeline, not necessarily invalid source annotations.
