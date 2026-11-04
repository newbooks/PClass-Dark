# SPEC-0018: Swiss-Prot evaluation audit

- Date: 2026-11-03.
- Status: Implemented; validation results recorded below.

## Purpose and provenance

Provide `scripts/swiss-audit.py` to summarize the comparison table produced by
[SPEC-0017](0017-evaluate-swiss.md), with the plots and scientific definitions of
[SPEC-0016](0016-external-audit.md). The reference script is `scripts/external-audit.py`.

The reference implementation is local: `external_audit.py`,
`external_audit_cli.py`, their tests, and SPEC-0016. Its parser, truth verification,
GO filtering, binning, coverage and summary functions already support Swiss
publication marker v2. Reuse these methods rather than duplicate scientific
calculations. The meaningful differences are the Swiss entry points, titles and
output prefix. No neighboring repository becomes a runtime dependency.

These are descriptive semantic hit rates using saved AMB Lin scores, not exact
GO-label accuracy or a fresh evaluation. Do not rerun inference, recompute GO
similarities, fit thresholds, access live annotations or download models.

## Interfaces and configuration

Interfaces:

```bash
python scripts/swiss-audit.py work-swiss/swiss_evaluation.tsv
pclass-dark swiss-audit work-swiss/swiss_evaluation.tsv
python scripts/swiss-audit.py results.tsv --output-dir plots --bins 10 --hit-cutoff 0.7
python scripts/swiss-audit.py results.tsv --truth-tsv query_truth.tsv
```

Accept any input basename; require an explicit input path. Default output directory
is the input's parent. `--bins` is an integer in [1,100], default 10;
`--hit-cutoff` is finite and in [0,1], default 0.7. `--truth-tsv` is optional.
Use Typer with thin script and installed-command adapters. Print original,
excluded and retained query counts, distance-over-one diagnostics, and output
paths. Invalid input, missing plotting dependencies, unverified truth or I/O
failures exit nonzero with a useful error, matching the external adapter.
Typer usage errors exit 2; operational errors exit 1; successful execution exits 0.

Expose importable `pclass_dark.swiss_audit.audit_swiss(input_path,
output_dir=None, *, bins=10, cutoff=0.7, exclude_go="GO:0005515", truth_tsv=None)`.
Return `(paths, rows, diagnostics)` with the same types and meanings as
`audit_external`: a tuple of five Paths in the output order below, a list of two
method dictionaries ordered PClass then PClass-Dark, and a diagnostics dictionary. The Python default deliberately matches the Swiss script's
filtered behavior; existing `audit_external` remains unfiltered by default for
compatibility. Python callers may explicitly set `exclude_go=None` for an
unfiltered audit. For this wrapper, reject any other `exclude_go` value with
ValueError so `_v2` always denotes this exact exclusion; the existing external
API retains its broader behavior. No new CLI filtering option is required.
Document ValueError/OSError input/output failures and RuntimeError for missing
optional plotting support. Imports perform no rendering, argument parsing,
resource resolution or scientific computation.

## Input and truth contracts

Reuse the exact seven-column evaluation contract:

`Query_id`, `PClass_go`, `Embedding_d`, `Pclass_match`, `Dark_go`, `Dark_Bscore`,
`Dark_match`.

Reuse `read_audit_table` unchanged. Require each of the seven named headers
exactly once; accept reordered columns and additional uniquely named columns,
which are ignored for analysis. Reject duplicate headers, missing required
headers, ragged rows and missing/duplicate query IDs. IDs are stripped and must
not be empty or `NA`. The existing reader accepts a UTF-8 BOM.

Prediction GO columns are presence indicators: stripped empty/`NA` means absent;
any other text means present. The audit does not validate prediction GO syntax,
resolve obsolete terms or query a graph. Truth GO sets, by contrast, are validated
by the shared truth reader. This distinction deliberately preserves external
parser compatibility; stricter prediction validation is outside this change.

Numeric empty/`NA` values are missing; other values must be finite, with distances
in [0,2] and match/B scores in [0,1]. A defined match requires a present GO
prediction. Finite evidence without a prediction remains permitted: it contributes
to all-row distance diagnostics but not prediction bins or ranking. Zero is defined.
Undefined matches, including unknown-GO scoring outcomes, remain undefined;
do not replace them with zero or infer a score from GO presence.

Like the current external CLI, exclude whole queries whose own truth set contains
`GO:0005515`, including mixed truth sets. Never filter using either prediction
column. Resolve truth from the publication marker beside the supplied table,
verified score manifest and `query_status.jsonl`, using the existing marker-v2
`state_relative`, workspace UUID and checksum checks. Honor relocated output
paths and reject invalid versions, unsafe paths, symlinks and identity mismatches
according to the shared reader. Preserve its legacy marker-v1 compatibility;
Swiss-specific rendering does not justify weakening or duplicating verification.

Alternatively, `--truth-tsv` must have exactly `Query_id` and `go_mf_terms`, with
pipe-separated valid nonempty GO sets, unique IDs and exact input query coverage.
This is the explicit source for standalone/renamed tables; record its use in the
run log. Missing or unverifiable truth fails before output publication. No live
accession lookup, graph loading, or implicit fallback to predicted annotations.
With `exclude_go=None`, ignore `truth_tsv` without opening it, matching external
behavior; it remains a protected source path if supplied. With filtering enabled,
explicit truth takes precedence over publication metadata; do not also require a
marker. Both input and truth are read-only. Even headers-only filtered inputs
require a valid truth source with exact empty coverage.

Apply filtering before every summary, bin and curve; all-query denominators refer
to the retained cohort. Report original/excluded/retained counts clearly.

## Analysis and plots

Preserve SPEC-0016 definitions and visual conventions. A hit is saved semantic
match >= cutoff; scorable means match is defined. Missing matches are excluded
from hit-rate denominators but retained in prediction coverage where applicable.

1. Match-score scatter: Pclass_match on x, Dark_match on y; equal [0,1] axes,
   identity and cutoff lines. Include jointly scorable queries only; annotate
   omitted counts. Method-only hits are reported in the summary, as in the
   existing renderer. Do not randomly subsample in version 1.
2. Confidence: distance and B panels with equal-width [0,1] bins and shared count
   scale. Stack hit/non-hit/undefined counts among predictions; percentages use
   scorable predictions per bin. Preserve distance >1 overflow and missing-evidence
   categories. Smaller values mean stronger evidence; do not imply cross-method
   calibration. Bins are left-closed/right-open except the last, which includes 1;
   internal-edge values enter the higher bin. Append >1 and NA categories only
   when present. Keep all regular bins, including empty ones.
3. Coverage: strongest-to-weakest rankable predictions, ascending evidence, ties
   retained together. x is retained predictions / all retained queries (%);
   y is hits / scorable retained predictions (%). Undefined matches affect
   coverage but not hit rate. Missing evidence is unrankable; undefined rates
   are gaps. No fabricated origin or extension past available coverage.
4. Summary table: per-method total, predicted, scorable and hit counts, prediction
   coverage, hit rate, hits/all queries, undefined scores and missing confidence;
   both-hit and method-only-hit counts; distance >1 count/fraction and maximum.
   Include filtering counts. Preserve the existing summary TSV schema: per-method
   keys `method`, `queries`, `predictions`, `scorable`, `hits`, `coverage_pct`,
   `hit_rate_pct`, `hits_per_query_pct`, `undefined_matches`, `missing_confidence`,
   `both_hit`, `method_only_hit`; diagnostics `distance_defined`,
   `distance_over_one`, `distance_over_one_pct`, `distance_max`,
   `jointly_scorable`; filtered runs additionally have `original_queries`,
   `excluded_queries`, `retained_queries`, `excluded_go`. Merge diagnostics into
   both method rows and write missing values as `NA`. Preserve the external
   writer's column order, including its early original/excluded count fields. No schema
   extension or new machine-readable provenance artifact is required.

Use Swiss audit titles and PClass/PClass-Dark labels. Bin counts are descriptive;
cutoff is a user-selected descriptive operating point, not a learned threshold.
Support headers-only inputs, zero retained queries, no jointly scorable rows,
all-missing evidence/scores and empty panels without division-by-zero artifacts.

## Outputs, publication and implementation boundaries

Filtered outputs are:

- `swiss_audit_match_scores_v2.png`
- `swiss_audit_confidence_v2.png`
- `swiss_audit_coverage_v2.png`
- `swiss_audit_summary_v2.png`
- `swiss_audit_summary_v2.tsv`

Unfiltered Python outputs omit `_v2`. Default filtered filenames intentionally
match the external audit's version convention, with a distinct Swiss prefix.
Only the five listed destination basenames may be replaced, under the explicit
or default output directory. Reject resolved destination collisions with input
or truth paths before staging; never mutate evaluation manifests/status/raw
artifacts or `external_audit_*` files. Check an existing destination for hard-link
identity with either source as well as resolved-path identity, and reject
collisions before rendering. Do not introduce recursive cleanup or artifact
migration. An existing audit destination symlink must be rejected rather than
followed. Directory creation alone may persist after failure. Stage all five
outputs before replacement and clean staging on failure. Match existing per-file publication semantics: replacing the
whole set is not an atomic multi-file transaction. Do not claim stronger recovery.

Add a shared internal renderer/output wrapper parameterized by audit title and
basename prefix; keep existing external public signatures, defaults and filenames
unchanged. Concretely, move the orchestration/rendering body behind a private
function in `external_audit.py` with internal `prefix` and `summary_title`
parameters. Keep `audit_external` as its compatibility wrapper; `audit_swiss`
uses `swiss_audit` and `Swiss-Prot audit`. Generic plot titles/labels and layout
remain unchanged; the summary title identifies the cohort. Do not add new public
prefix/path-template arguments or duplicate the renderer. Retain existing
external temporary-directory naming if that avoids needless behavior changes.
`swiss_audit.py` supplies Swiss defaults; `swiss_audit_cli.py` adapts arguments. Keep new script in the sdist through MANIFEST.in and register the
installed subcommand. Update README, Swiss operational/API guide and CLI help.
Use optional `[audit]` Matplotlib dependency already declared; no new dependency.

## Resource use and interpretation

CPU NumPy analysis with O(N) arrays/GO masks, O(N log N) stable ranking and
headless Matplotlib rendering. No independent CPU-heavy tasks warrant a process
pool; no GPU benefit is expected. Parse once and reuse data across plots. At the
current 380,578-query scale, measure wall time, peak RSS and output sizes; inspect
scatter readability before proposing any future aggregation. Deterministic
ordering, bins and tie groups; no sampling seed is needed. Do not read the 12-GB
raw inference directory: only the table and required verified scoring metadata.

Swiss self/training and homologous matches may be present. GO-based exclusion is
posthoc cohort selection. Plots do not establish held-out generalization or
control leakage; preserve this limitation in documentation and recorded results.

## Validation and completion criteria

Automated validation requirements:

- Small independent analytical fixture with both-hit/method-only cases, zero,
  undefined matches, missing confidence, overflow, exact boundaries and ties;
  verify denominators after truth-based exclusion, including mixed truth sets.
- Swiss/external numerical parity on identical retained rows and settings,
  while verifying distinct prefixes/titles and unchanged external outputs/API.
- Verified Swiss marker-v2 truth resolution, relocated output, standalone truth,
  exact coverage, duplicate IDs, missing truth and corrupted metadata rejection.
- Empty input/retained cohorts, no predictions, all undefined scores, invalid
  schemas/numbers/options and source/output path collision protection (resolved
  paths, hard links and destination symlinks). Test reordered/additional columns
  and non-GO prediction text as accepted inputs, while malformed truth GO fails.
  Cover explicit truth precedence and unsupported `exclude_go` values.
- Script, installed CLI and library parity; usable help; optional plotting import
  behavior; valid PNG signatures and summary TSV content; staging/render failure
  leaves existing outputs intact before publication begins.
- Run Swiss/external audit, affected evaluation and CLI suites, Ruff lint/format,
  whitespace and documentation checks. Record commands/results; do not mark
  unexecuted tests as passing. GPU tests are inapplicable to this CPU-only work.

When a valid Swiss publication is available, run the new command on it. Record
input SHA-256, selected score ID/truth source, code revision plus dirty status,
command, cutoff/bins, filtering and score counts, software/hardware, elapsed
time/peak RSS, artifact paths and sizes. Visually inspect all four PNGs. Preserve
original findings and distinguish these descriptive results from held-out quality.
Record blockers if publication/truth is unavailable; do not fabricate results.

Before completion, synchronize implementation status and deviations here, update
the spec index and dated progress log, and document the exact API/CLI behavior.

## Implementation sequence and readiness review

1. Extract the private rendering/orchestration function without changing external
   defaults, filenames, numerical results or public signatures; run external audit
   regression tests before adding Swiss behavior.
2. Add the Swiss API and thin script/Typer adapter; register `swiss-audit` in
   `cli.py` and the script in MANIFEST.in. Add `tests/test_swiss_audit.py` for the
   contracts above and extend external tests for the shared refactor as needed.
3. Synchronize user documentation and run the affected suites. A suggested
   baseline is `python -m pytest -q tests/test_external_audit.py
   tests/test_swiss_audit.py tests/test_external_evaluation.py
   tests/test_swiss_evaluation.py tests/test_cli.py`. Include script help/execution
   and installed command registration.
4. Run and inspect real Swiss artifacts when available; record results/limitations
   and then mark this specification implemented. Do not mark an unavailable
   real-data check as passed.

Readiness review (2026-11-03): checked parser, options, truth lookup, bin and
coverage helpers, summary schema, renderer and publication behavior against the
current code. Resolved the earlier extra-column/GO-validation mismatch, specified
filter variants and truth precedence, exact output schema, shared-refactor
boundary and failure semantics. No scientific choice or interface decision is
left pending. New source-collision protection is an intentional implementation
requirement; numerical results and ordinary external public behavior remain compatible;
unsafe output collisions are deliberately rejected by both wrappers. This
readiness review preceded implementation; execution results follow below.


## Implementation and actual-data validation (2026-11-03)

Implemented the shared private `_audit` renderer, compatibility wrapper, Swiss
API/CLI/script, subcommand registration and sdist inclusion. Existing scientific
helpers remain unchanged. Swiss title and prefix are explicit internal parameters;
external public defaults and output names remain unchanged. Added source hard-link
and destination symlink protection to both wrappers. No numerical deviations.
The existing TSV writer places original/excluded counts before shared diagnostics;
that existing schema order is preserved rather than reordered by this change.

Real command: `/usr/bin/time -v python scripts/swiss-audit.py
work-swiss/swiss_evaluation.tsv`, defaults bins=10 and cutoff=0.7. Table SHA-256:
`bfd8749640c56d022a0cf9a4bcec750056947be74b5c8a67aaf82a7abaef6476`.
Truth came from verified Swiss publication marker v2 and saved query statuses;
score ID `0709f16e28b50387710b7f1d8e63b27802939a7147a008f3b89af589d900c9e2`.
Revision `3af0a2a6bae7906ec0b8efba56f43553a9c81b95` plus dirty implementation.
No seed/sampling, preprocessing fit, model/GPU work, graph or live lookup.

- Original/retained queries: 380,578; GO:0005515 exclusions: 0.
- PClass: predicted/scorable 380,578; hits 374,085 (98.2939%).
- Dark: predicted/scorable 380,568; hits 234,274 (61.5590%).
- Both-hit 231,239; PClass-only 142,846; Dark-only 3,035.
- Distance >1: 0/380,578; maximum 0.24682319164276123.
- Ten queries lack a Dark prediction and are omitted from paired scatter.
- 9.63 seconds wall time; peak RSS 271,640 KiB (about 265 MiB).
- AMD Ryzen 9 6900HX, Linux x86_64, Python 3.14.6, NumPy 2.4.4,
  Matplotlib 3.11.0; CPU rendering with no process pool.

All five outputs are under `work-swiss/`: `swiss_audit_match_scores_v2.png`
(735,231 bytes), `swiss_audit_confidence_v2.png` (133,243),
`swiss_audit_coverage_v2.png` (106,444), `swiss_audit_summary_v2.png` (125,358),
and `swiss_audit_summary_v2.tsv` (640). Visually inspected all four PNGs: axes,
legends, annotations and summary are legible. Scatter is dense near high PClass
scores; full-data rendering was practical, so no aggregation was introduced.
The PClass curve starts near full coverage because tied evidence is retained as
a group, not because an origin was manufactured.

Logs: `/tmp/swiss-audit-real.log` and `/tmp/swiss-audit-real-resources.log`.
Matplotlib used a temporary cache because the home config directory was not
writable; rendering succeeded. These results are descriptive in-sample diagnostics
with possible self/training/homologous matches, not held-out generalization.

Automated validation: external audit regression immediately after extraction
**15 passed in 7.47 s** (`/tmp/swiss-audit-external-refactor.log`). Final specified
audit/evaluation/CLI suites **165 passed in 97.06 s**
(`/tmp/swiss-audit-regression.log`); the subsequently added isolated import
regression **1 passed in 0.24 s**. Tests cover analytical parity, title/output
names, script/CLI/API, Swiss metadata/explicit truth precedence, corrupt status,
empty/all-undefined inputs, parser compatibility, source collisions, rendering
failure preservation, optional dependencies and import side effects. Existing
audit/evaluation tests cover bin/tie calculations and legacy metadata behavior.
Installed `pclass-dark swiss-audit --help`, Ruff lint/format, local documentation
checks and whitespace checks passed. Code-only temporary wheel/sdist build
verified both Swiss modules in the wheel and executable script in the sdist
(`/tmp/swiss-audit-packaging.log`). No new dependency or GPU validation required.
