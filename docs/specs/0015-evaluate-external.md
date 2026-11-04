# SPEC-0015: External classifier comparison

Status: implemented command-output revision, 2026-11-01. This supersedes the original
artifact-attested/native-adapter design following the user's request to minimize
dependencies and consume the already working classifier commands.

## Purpose and interfaces

Provide executable `scripts/evaluate_external.py FASTA` and an importable
`pclass_dark.external_evaluation` API. Collect costly classifier output once,
retain it for reuse, and write `external_evaluation.tsv` in the working directory.
Do not calculate aggregate accuracy or use alignment/sequence identity as scores.
No new `pclass-dark` subcommand is introduced.

Required environment: existing `pclass infer --fasta` and `pclass-dark infer
--fasta` commands on PATH and the already required profunc Python package with
its local MF graph and composed cluster GO sets. No new Python dependencies,
`ijson`, bundle fingerprints, external executable probes, local model inspection
or separate evaluator-managed model installation are required. Classifier
commands own server/library/model resolution, embeddings, alignment and other
internal operations. The evaluator does not replicate them or parse alignment.
Commands may use different Python environments. File orchestration uses POSIX;
profunc similarity requires Python >=3.11 in the evaluator environment.

Input is uncompressed UTF-8 FASTA, unique first-token identifiers, uppercase
common engine alphabet `ACDEFGHIKLMNPQRSTVWYBOUXZ`. Each query needs truth in
`go_mf_terms=GO:...|GO:...` or an exact-coverage `--truth-tsv` with columns
`Query_id` and `go_mf_terms`. Both sources must agree. Truth is stripped from
FASTA sent to classifiers. Empty/invalid/duplicate input is an error.

CLI defaults work with only FASTA when classifier commands and profunc data work:

```bash
../scripts/evaluate_external.py external_sequences.fasta
../scripts/evaluate_external.py external_sequences.fasta --score-only
```

Optional options: output/truth paths; PClass batch size (512, range 1..1024), Dark
chunk size (512); PClass server URL and artifact directory, Dark library directory
and device (omitted by default, delegated to classifiers); timeout (3600 seconds
per child), raw stdout cap (128 MiB per chunk); score workers (1..8, default 8);
explicit graph/map overrides; preflight-only; run tag for deliberate new
inference when externally managed resources change. Expose typed configs and
pure `score_external_predictions` for Python integrations. Native classifier
adapters and the old server-bundle attestation flag are removed.

## Collection and retention

Invoke PATH-resolved commands with argument lists, no shell:
`pclass infer --fasta CHUNK --format json` and
`pclass-dark infer --fasta CHUNK --format json`, adding only requested optional
flags. Record resolved executable identity, command options, working directory, selected environment
variables and user run tag. Do not import PClass, Torch, ProSig or transformers.

PClass chunks respect count <=1024 and a conservative 8-MiB estimated JSON body
bound (default service cap 10 MiB). User batch size may reduce this. The classifier
itself handles live service limits. Bisect only documented REQUEST_TOO_LARGE or
HTTP413 rejections; retry leaves independently and reuse successful leaves after
restart. An oversized singleton fails. Dark chunks have a separate count bound;
its command enforces its own model/memory quotas. Other failures preserve logs
and fail rather than silently dropping sequences. One inference child at a time.

Redirect stdout/stderr to attempt files; poll every 0.1s for timeout/stdout cap,
terminate the POSIX process group and reap it on failure/cancellation. Validate
JSON (including duplicate/nonfinite values), root schema, exact query coverage,
IDs, Dark sequence hashes/lengths and accepted match/B ranges. Ignore unneeded
alignment, sequence-confirmation and signature fields. Keep complete JSON, not
only projections. Standard-library JSON loads one bounded chunk at a time;
parsing may consume several times its byte size. The cap bounds each file, not
total retained storage. Never publish a receipt for a failed/malformed chunk.

Raw observations are immutable, hashed and stored under `.external-evaluation/`.
Canonical sequence and truth identities remain independent so annotation-only
changes reuse inference. Separate generation manifests describe complete query
coverage per engine. Cache identity cannot attest remote/live model contents:
reuse assumes unchanged command-managed resources. Set a new `--run-tag` after
changing server index/models or classifier libraries. Full raw source metadata
is preserved. Do not claim server bundle verification or fit/reference attestation.

## Labels and scientific scores

Exact TSV header, in input order, one row per query:

`Query_id PClass_go Embedding_d Pclass_match Dark_go Dark_Bscore Dark_match`

The delimiters are tabs. GO sets are sorted unique IDs joined by `|`. Empty GO
sets and unavailable/undefined numbers are NA; a defined zero is 0.0.

- PClass top match: choose lowest-distance neighbor, preserving report order on
  ties. PClass_go is that neighbor's `go_mf_terms`, Embedding_d its distance.
  Do not rank by alignment/sequence identity or use additional voted predictions.
  Preserve the full reported predicted GO set and other neighbors in raw JSON.
- Dark top match: accepted cluster minimizing `(background_tail_fraction,
  cluster_id)`. Dark_Bscore is its fraction (not percentage), Dark_go the
  corresponding profunc composed GO set. Never select by agreement with truth.
- Both *_match columns: profunc AMB Lin similarity of truth versus chosen GO set.
  Use the same underlying implementation as `profunc analyze go-set-sim` (the
  installed command name, rather than `sim-go-set`). Call its Python helper
  directly, avoiding per-query subprocess startup and rounded text parsing.
  Freeze the graph/IC; do not estimate IC from this cohort. Reject unknown terms.
  Empty predictions and undefined comparisons stay NA. Independently test the
  two directional means and empty/root-only behavior.

Dark JSON currently has cluster IDs but no GO terms. Resolve composed terms via
profunc's runtime and cluster-record APIs, without generating English descriptions
or inspecting Dark model fits. Snapshot the graph/map for reproducible rescoring;
require top-match cluster IDs to exist. The operator must use profunc data
corresponding to the Dark classifier's cluster IDs. The evaluator cannot prove
that reference equivalence from the current inference JSON alone. Explicit map
schema: version 1, clusters mapping IDs to `go_mf_terms` lists, optional reference
metadata, and content-derived mapping_id. Defaults are resolved automatically.

## Scoring, storage, publication

Reuse existing durable cohort/SQLite storage and journaled publication. Preserve
sequence/truth/header snapshots, raw reports/receipts, failed-attempt logs, frozen
graph/map resources, score tables/statuses, manifests and publication backups.
Do not migrate or erase existing artifact-attested generation formats; recognize
only command-output v2 generations in new collection/selection. Legacy raw files
remain available for manual analysis. Old unsupported cache selection errors
must be actionable rather than triggering unexpected inference in score-only.

Scoring joins typed records through SQLite and yields deterministic input order.
Configurable spawned score workers share work in bounded blocks; each worker
holds its own graph/map, with no model. Plan and coverage ID metadata is O(query
count); sequences and prediction joins stay on disk. Revalidate raw hashes before
use. Score IDs include raw generation IDs, truth, graph/map, scoring code and
metric implementation. Graph/map/truth changes do not require inference.

Score-only uses saved verified generations/resources, never resolves classifier
commands or checks services/models. Without a saved selection, fail clearly.
Preflight checks input, PATH commands and scoring data only, without inference or
biological scans. Preserve output ownership/locks, manual edit detection,
immutable score copies and journaled recovery before work. Interruptions do not
publish partial tables. Output parents must already exist. Do not overwrite
unowned or altered outputs or repair corrupt expensive raw data silently.

## Validation and provenance

References inspected: PClass inference CLI/JSON/neighbor request schema and exit
codes; Dark inference rendering/records; profunc cluster listing, runtime,
`analyze go-set-sim` and similarity functions. These are installed dependencies,
not runtime imports from neighboring checkout paths.

Test analytical scores, top-match selection ignoring alignment and extra voted
GOs, abstentions, malformed/duplicate/extra/missing JSON, input validation,
PClass batch/split boundaries, real lightweight subprocess timeout/output limits,
resume after one-engine failure, offline rescoring without inference imports,
parallel ordering, graph/truth changes, corruption refusal and publication
recovery/ownership. Smoke test executable help and actual local profunc resource
loading. Update operational docs, README, dependency metadata, index and progress
with results. Full dataset inference and GPU performance are separate experiments.

Implemented validation (2026-11-01): evaluator/CLI/external-sequence suites passed
139 tests in 30.74 seconds with the normal source path and no ijson dependency
path. Lint, formatting, executable help, local documentation links/fences and
whitespace checks passed. A clean temporary code-only wheel/sdist verified no
new dependency metadata and executable script/guide inclusion. Actual local
PATH-command/profunc preflight passed without inference. Full cohort inference,
GPU execution and large-cohort memory/runtime remain experimental work; exact
commands and resource identities are in the progress log.

Collection progress logs include the percentage of cohort queries completed for
that engine, accumulated across chunks (including reused receipts and split
leaves). Each completed generation reaches 100.0%; percentages use one decimal.

Shared Swiss evaluation integration (SPEC-0017): snapshot ingestion supports an
explicit truth resolver/source contract and optional normalized-accession table.
The default external CLI behavior, state directory, table name and inference
format remain unchanged. Scoring/publication accept explicit native readers and
table/state names; worker-scoped GO-pair caches are bounded to 4096 entries and
numerical-library threads to one per score process. Scientific scoring is unchanged.

### Unknown collected GO terms (2026-11-03)

Shared semantic scoring warns for GO terms absent from the frozen graph and
uses profunc's available-term AMB Lin calculation. Preserve original GO sets and
source counts; record `missing_terms` for affected metric statuses. No usable
terms produces an unavailable score (`NA`). Preflight truth/map validation and
unknown cluster rejection remain unchanged.
