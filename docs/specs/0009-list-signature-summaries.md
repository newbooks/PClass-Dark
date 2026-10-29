# SPEC-0009: List cluster signature summaries

- Date: 2026-10-28.
- Status: Implemented (2026-10-28); verification recorded below.
- Readiness review: 2026-10-28, checked against CLI callbacks, production manifest
  bindings, signature validation and exact training metric conventions.
- Purpose: Add an aligned, alphabetically ordered summary table through
  `pclass-dark explain-signature --list`.
- Provenance: User-approved column selection and formatting: no FP column,
  left-aligned identifiers/status, right-aligned numbers, three decimal places
  for AUROC and percentages, integer counts and `—` for unavailable values.
- Related: [SPEC-0008](0008-explain-signature.md) provides individual explanations;
  [SPEC-0004](0004-build-library.md) defines saved production signatures and metrics.

## Scope

List every cluster in the selected production library, including unsupported
signatures. Show saved final all-C, in-sample metrics. This is a summary operation;
it does not explain feature selection, evaluate queries, rank clusters by match,
run research folds, or fit/calibrate a model.

Use selected production manifests, per-cluster final signature JSON files, and
an optional packaged description catalog as specified in [SPEC-0010](0010-cluster-descriptions.md).
Do not load profile tables, selection traces, embeddings, setup-data artifacts,
relative-score arrays, or research reports. Do not call the full-library loader
or invoke the individual explanation once per cluster. Those paths do unnecessary
scientific validation and IO for this table.

## CLI modes and compatibility

```bash
pclass-dark explain-signature --list
pclass-dark explain-signature --list --library-dir /shared/library
pclass-dark explain-signature --list --format markdown
pclass-dark explain-signature --list --format json
pclass-dark explain-signature --cluster cluster_0001
pclass-dark explain-signature -h
```

- Make `--cluster` optional at argument parsing, retaining its existing rejection
  of repeated occurrences. Add boolean `--list` (internally `list_mode`).
- Require exactly one mode: one `--cluster` ID or `--list`. Neither or both exits
  2 with usage guidance, before reading library artifacts.
- Preserve `--library-dir`, default path resolution and `-h`/`--help` from SPEC-0008.
- `--format` remains `text` (default), `markdown` or `json`, in both modes.
- `--detailed` continues to apply to individual explanations only. Reject
  `--list --detailed` with exit 2. Extra listing columns are outside this version;
  the earlier suggested extended table is not part of the initial implementation.
- No positional cluster ID, multiple-cluster filter, metric sort, top-k, pager,
  query input or partial-results mode is added.
- Success exits 0; invalid syntax/mode/format exits 2; artifact/IO failures exit 1;
  interruption exits 130. Errors go to stderr, reports to stdout.
- Assemble and render the complete report before writing stdout. Corrupt summaries
  cause failure, not skipped rows or a partially printed table.

The current required-list callback for `--cluster` must be adjusted to tolerate
its absence in list mode. Do not run missing-cluster validation before handling
`--list`, and do not turn repeated `--cluster` values into a last-value-wins option.
Help must work without opening a library. Concretely, use an optional repeatable
cluster option with default None. Its callback accepts absence, returns a supplied
single-element list unchanged, and rejects two or more occurrences. The command
checks `(cluster is present) XOR list_mode` before dispatch. Define the boolean
flag with the explicit public name `--list`, avoiding the Python built-in `list`.
The library APIs below do not require callers to reproduce this CLI dispatch.

## Default table

Title: **Final signatures — in-sample reference metrics**.

Column order and sources (Status is conditional):

| Column | Saved source | Display |
|---|---|---|
| Cluster | `cluster_id` | Exact ID, alphabetically ordered |
| Description | Bound packaged function description | At most 24 characters/cells including `...`; `—` if unavailable |
| Status | Signature status/reason | `ok`; otherwise the nonempty saved reason, falling back to status |
| C_windows | `counts.training_positive_n` | Integer |
| B_windows | `counts.background_n` | Integer |
| Dimensions | Number of `selected_dimensions` | Integer; zero for an unsupported fit with an empty set |
| AUROC | `training_metrics.auroc` | Fraction, fixed 3 decimals |
| Recall% | `training_metrics.recall * 100` | Fixed 3 decimals |
| Specificity% | `training_metrics.specificity * 100` | Fixed 3 decimals |
| Precision% | `training_metrics.precision * 100` | Fixed 3 decimals |

For nonempty text/Markdown listings, omit Status when every row is `ok`.
Keep it if any row is unsupported, and retain it for an empty listing. All
remaining columns keep their order and numeric alignment. JSON/API rows always
retain `status` and `reason`; this is a presentation-only choice.

Description text can contain spaces even though headers are single tokens; JSON
preserves full descriptions for structured parsing. See SPEC-0010 for provenance,
reference bindings and maintenance. There is no FP column. No counts of parent chains, accuracy, balanced accuracy,
F1, average precision, cutoff or distribution quantiles are added to this table.
Those remain available in individual explanations. No cluster-level ABC values
are fabricated: ABC evidence requires an evaluated window.

An illustrative row using the documented cluster_0001 snapshot:

```text
Final signatures — in-sample reference metrics

Cluster       Description               C_windows  B_windows  Dimensions  AUROC  Recall%  Specificity%  Precision%
cluster_0001  NADP+ nucleosidase wi...         20     213241          13  0.997  100.000        97.348       0.352
```

This example is not a fixed-width schema or a golden expectation for future
libraries. Generate every row from the currently selected artifacts.

## Ordering, alignment and numeric formatting

- Sort exact stored cluster IDs with case-sensitive lexicographic ordering
  (`sorted(ids)`); do not apply natural-number, locale or metric ordering.
- Format cell strings first, then compute each column's width as the maximum
  displayed width of its header and every cell. Use two spaces between columns.
- Column headers are single whitespace-free tokens: `Cluster`, `Description`, `Status`,
  `C_windows`, `B_windows`, `Dimensions`, `AUROC`, `Recall%`, `Specificity%`,
  `Precision%`. This permits whitespace splitting of the table header.
- Left-align Cluster, Description and Status. Right-align all other headers and cells,
  including unavailable numeric cells. Counts have no decimal places or grouping
  separators. Do not turn counts into rounded scientific notation.
- Use fixed-point `.3f` for every available AUROC/percentage, including trailing
  zeros. Percent units appear in headers, not as repeated suffixes in cells.
  Examples: AUROC 0.9966748889753847 becomes `0.997`; precision
  0.0035236081747709656 becomes `0.352` in Precision%; recall 1 becomes `100.000`.
- Round only for display, using Python's ordinary float fixed-point formatting.
  Do not round stored or API values before validation, and do not replace a tiny
  positive value with an adaptive notation that violates the three-decimal rule.
- Render unavailable metrics as `—`, never zero, `nan`, `None` or an empty cell.
  Preserve the reason in structured output. Format signed zero as `0.000`, not
  `-0.000`, without changing the raw JSON value. Undefined precision for a supported
  predict-none signature is also `—`; the signature remains `ok`.
- Use monospace plain text, without ANSI codes, borders, terminal-dependent
  cropping, row wrapping, pagination or elision. Emit all rows without a pager.
  A narrow physical terminal may still wrap long lines; do not silently discard
  columns or truncate identifiers to fit its width. File redirection retains
  the aligned complete table.
- Measure display cells using `rich.cells.cell_len` after escaping controls.
  Declare `rich>=13.8.0` explicitly in project dependencies; this is already the
  minimum required by the installed Typer dependency and avoids relying on a
  transitive import. Do not invoke Rich's terminal-layout renderer: construct
  deterministic plain strings. Pad by `width - cell_len(cell)` spaces so the em
  dash and wide/combining characters align. Terminal width does not affect output.
- Escape control characters in identifiers/status to visible single-line forms
  before table layout (for example, `\n`, `\t`, `\x1b`). JSON keeps original IDs
  with normal JSON escaping. Do not let artifact strings inject terminal styling.
- Markdown uses the same formatted values and column order, escapes pipes/control
  characters, and marks numeric columns right-aligned. Measure/pad the visible
  cell strings before adding Markdown pipe escaping; Markdown source padding is
  cosmetic, while alignment markers define rendered table alignment. JSON retains full-precision
  fractions and raw integers rather than copying formatted percentage strings.

Include a short note below human-readable tables:

> Metrics are in-sample; reference populations were used during signature
> discovery and threshold selection. Values are rounded to three decimal places;
> small positive values may display as 0.000. Use --cluster for an explanation.

Do not label these metrics confidence, membership probabilities, or held-out
performance. Precision depends on the sampled reference population balance.

## Python API and structured report

Expose reusable operations in `pclass_dark.explanations`, with a thin CLI adapter:

```python
from pclass_dark.explanations import (
    list_signature_summaries,
    render_signature_summaries,
)

listing = list_signature_summaries(library_dir=None)
print(render_signature_summaries(listing, format="text"), end="")
rows = listing.to_dict()["rows"]
```

Return a frozen `SignatureSummaryListing` with recursively frozen data and a
`to_dict()` method returning a detached mutable copy, following SPEC-0008.
Rendering performs no IO and returns exactly one trailing newline. Invalid
arguments/formats raise `ValueError`, malformed artifacts `IntegrityError`, and
inaccessible files `OSError`. Imports must not load data, parse CLI arguments,
start workers or import heavy PLM runtimes.

JSON schema: `pclass-dark.signature-summary-list.v1`. Top-level fields in order:

- `schema`.
- `source`: `library_id`, `manifest_sha256`, `selection_schema`,
  `evaluation_mode="all-C-in-sample"`, `validation_scope="signature_summaries"`,
  and nullable `description_catalog_id`.
- `counts`: `clusters`, `available`, `unsupported`.
- `rows`: alphabetically ordered records, each containing `cluster_id`, `status`,
  `reason`, `description`, `description_unavailable_reason`, `C_windows`, `B_windows`, `dimensions`, `auroc`, `recall`,
  `specificity`, `precision`, and `unavailable_reasons`.
- `limitations`: stable code/message records for in-sample evaluation,
  reference-population precision and display rounding.

Metric fields store fractions in [0,1], or null with a reason. Counts are integers.
Unsupported clusters have zero dimensions and null metrics, retaining the saved
reason. For supported but undefined metrics, preserve the metric-specific reason
from `training_metrics.unavailable_reasons`. Row `status` retains the saved
status code, independently of the human table's Status label. For unsupported
rows, `unavailable_reasons` maps each of the four metric keys to the saved reason
(or status if the reason is absent). For supported rows it contains only null
metric keys and their saved reasons. No row contains an FP field.

JSON rendering uses insertion-order keys, `indent=2`, `allow_nan=False`, with no
volatile timestamps or installation-specific paths. Preserve unrounded numbers;
three-decimal formatting applies to text/Markdown only.

## Summary loading and integrity contract

Reuse the existing path resolution, safe paths and root transaction guards.
Check root/selector existence before acquiring a guard so a missing input does
not create directories. Honor immutable installed-resource behavior, without
writing locks into the package. Hold the mutable root guard through snapshot
assembly and release it before formatting/output.

1. Read `selection.json` and verify the selected production manifest SHA-256,
   ID, complete status, schema, representation identity and identity-input hash.
   Accept selection v1/v2 pointing to production v2. Reject legacy production v1
   with the existing `build-library` migration guidance, and unknown schemas.
2. Validate unique sorted nonempty-string production cluster IDs and a one-to-one
   final-fit inventory with exactly those IDs, without duplicate entries. Validate
   nonnegative integer manifest counts: `targets=len(cluster_ids)`,
   `dimension_final_rows=640*targets` and `available_final+unsupported_final=targets`.
   The selected production inventory determines which clusters are listed; do not scan directories for
   additional bundles, archived signatures or checkpoint results.
3. Require a manifest file-record mapping and a well-formed SHA-256/size record
   for each listed signature. For every listed cluster, verify its
   `clusters/<encoded>/final.json` against that record using existing safe path,
   checksum/size and JSON decoding helpers (including duplicate-key/nonfinite-JSON
   rejection). Unused manifest entries need not be opened. Validate with existing
   `validate_signature`, then check v2 schema, library/cluster/final bindings, all-C evaluation mode, fit
   identity preimage and representation binding. Also bind the signature's
   `fit_identity` to the manifest's final-fit entry and its input `C_n`/`B_n` to
   the signature population counts. Validate stored payload-ID shape;
   do not claim to reconstruct its full scientific payload without profiles/trace.
4. Validate the counts and metrics used by the table. The existing signature
   validator does not fully validate training metrics, so implement a reusable
   summary validator rather than assuming that call proves metric consistency.
   Reject booleans as numeric counts, negative counts, invalid dimensions,
   nonfinite/out-of-range fractions and contradictory availability/status fields.
5. For supported signatures, verify TP+FN=C windows, TN+FP=B windows and saved
   positive/negative counts agree. Check defined recall, specificity and precision
   against those counts using existing numerical conventions. Verify saved AUROC
   numerator/denominator consistency, with denominator `2*C_n*B_n`; do not derive
   AUROC from confusion counts. Accept undefined precision only with the saved
   appropriate reason; unsupported signatures have no scoring parameters/metrics.
   The exact rules below resolve undefined values and rounding tolerance.
6. Assemble rows and compare available/unsupported totals with the manifest.
   A missing/corrupt selected signature is an error; never mark corruption as an
   ordinary unsupported fit or omit it from the listing.

### Exact summary validation rules

Implement a reusable `validate_signature_summary(signature, *, manifest,
cluster_id)` in a library module, separate from rendering. It may reuse
`validate_signature` and existing identity helpers; it must not call scientific
profiling or reconstruct scores. Catch malformed structured-field exceptions as
`IntegrityError`, consistent with `schema_checked`.

- Window counts and confusion counts must be real Python integers, excluding
  bool, and nonnegative. Validate `C_n`/`B_n` bindings before producing rows.
- For `status="ok"`, require C>=2, B>0, `reason=None`, at least two valid selected
  dimensions, and a training metric record with `status="ok"`. Its population
  counts must equal C/B, and TP+FN=C and TN+FP=B must hold.
- The displayed recall is exactly TP/C; specificity is exactly TN/B. Precision is
  exactly TP/(TP+FP) when that denominator is positive. Validate these saved
  fractions before formatting, using the existing exact Python-division
  conventions, without tolerance derived from three-decimal display rounding.
- Precision is null iff TP+FP=0. Its saved reason must then be
  `undefined_denominator_or_population`, as produced by `signature_metrics.evaluate`.
  Defined displayed metrics must not carry an unavailable reason. Recall,
  specificity and AUROC must be defined for a supported final signature.
- AUROC has integer denominator `2*C*B`, integer numerator in [0, denominator],
  and saved value exactly numerator/denominator, matching the existing count
  validator. Reject bools masquerading as numbers, NaN, Infinity, and values
  outside [0,1]. No AUROC is inferred from a threshold's confusion matrix.
- When the threshold kind is `above_maximum` (predict-none), require TP=FP=0,
  FN=C, TN=B and null precision with its appropriate reason. Other threshold
  kinds/parameter consistency are checked by `validate_signature`.
- For unsupported signatures, require null training metrics, null threshold,
  empty selected dimensions/IDs and zero displayed dimension count. Preserve
  the supported status enumeration from the existing validator. An
  `empty_background` status requires B=0; `insufficient_training` requires B>0
  and C<2; `insufficient_core` requires B>0 and C>=2. Do not invent numerical
  metrics for any of them. Validate reason as absent or a nonempty saved string.
- Validate only summary inputs and the signature fields described here; do not
  independently recompute average precision, fitting thresholds, selection
  history or the scientific payload hash. Those require evidence outside this
  operation's stated scope. Internal FP/TN/TP/FN checks do not expose an FP
  column or JSON field.

Production listing deliberately ignores relative companions and research selectors,
even when declared by the library selection: their data is not used by this
report. Missing/corrupt relative arrays must not block production summary listing.
Do not claim their validity, independently held-out metrics, or full-library audit
status. A checksum-bound summary verifies saved values and internal consistency;
it does not reproduce discovery AUROC or prove selection-history correctness.

Do not read `dimension_final.tsv.gz`, `cluster_metrics.tsv`, selection traces,
relative metadata/arrays/summary, setup-data or model assets. Do not call
`load_selected_library()` or `load_library()` in their full or target modes for
this operation, since those paths check dependencies outside this summary scope.
Factor shared manifest/signature checks where useful, preserving existing loader
semantics and strict validation when those APIs are called explicitly.

## Resource use and edge cases

This is bounded CPU/file IO, with no GPU or worker pool. Read each final signature
once for parsing and retain only compact rows/source metadata. Reusing checksum
helpers may require a separate bounded checksum read; a new shared JSON parser is
not required merely to combine hashing and parsing into one file open. Memory is
O(number of clusters) for formatting widths and output; do not retain full signature/profile inventories.
No artifact format changes, persistent caches or regenerated signatures are needed.

Include all unsupported clusters. A structurally valid empty production inventory
prints the headers, an explicit `No clusters in selected library.` message and
the in-sample note; JSON contains an empty row list and zero totals. Do not invent
an empty state for a missing or corrupt selector/manifest. Neither cluster ID
ordering nor alignment may depend on worker count, locale or terminal dimensions.

## Implementation and validation plan

1. Refactor CLI mode validation without changing single-cluster behavior.
2. Implement immutable summary records and a narrowly scoped summary loader.
3. Add pure text/Markdown/JSON renderers with complete-table width calculation.
4. Update README, help, SPEC-0008 cross-reference, spec index and progress log;
   record actual tests, timings and any deviations before marking implemented.

Tests must cover:

- Exactly-one-mode validation, repeated `--cluster`, help without data, invalid
  format, `--list --detailed`, and unchanged individual explanations.
- Alphabetic rather than metric/natural ordering, long IDs/statuses, wide and
  combining characters, escaped controls, aligned headers/cells and no ANSI.
- Fixed three decimals/trailing zeros, integer counts, percentages versus raw
  fractions, zero/one/tiny positive values, undefined metrics and `—` alignment.
- Correct default column order, no FP column/field, all rows present, immutable API
  results, deterministic output and JSON/API agreement at full precision.
- Analytical confusion-count examples, AUROC integer counts, all-equal-score
  predict-none with undefined precision, unsupported fits and inconsistent data.
- Corrupt/missing requested signatures/manifests, generation replacement under
  root guards, unknown schemas, legacy migration guidance, and the empty state.
- Instrumented reads proving no profile/trace/relative/embedding/model files are
  opened; missing unrelated relative arrays/research outputs must not block lists.
- Installed bundled operation from outside the checkout, without setup-data,
  network, heavy PLM imports or package writes.

## Readiness and completion criteria

No scientific questions remain: the feature displays already fitted production
metrics. The review resolved the CLI callback migration, exact metric/availability
rules, explicit cell-width dependency and bounded summary-loading strategy.
No production or distribution schema migration is required. Implement the public
APIs and tests above, synchronize docs, and record executed validation before
considering the implementation complete.

Use small synthetic fixtures for mandatory tests. Inspect a real local listing
when artifacts are available, without making its values permanent test constants
or running setup-data/build-library. Readiness-review documentation checks cover
links, fenced blocks and whitespace. Runtime implementation is recorded below.

## Implementation record — 2026-10-28

Implemented the two public functions and immutable `SignatureSummaryListing` in
`pclass_dark.explanations`. `signature_summaries.py` validates manifest/signature
summary bindings and exact metric/count consistency without reading profiles,
traces or relative artifacts. `signature_summary_render.py` produces fixed-three-
decimal, display-cell-aligned text/Markdown with escaped controls; JSON retains
unrounded values. Rich is now an explicit dependency. Existing explanation and
listing snapshots share a private immutable base, keeping their public types
distinct so each renderer rejects the other report type.

CLI dispatch requires exactly one of `--cluster` or `--list`, with `--detailed`
restricted to single-cluster explanations. No columns beyond the approved nine
were added; FP remains absent from the table and row schema.

Initial listing tests: **21 passed**. Expanded listing/explanation/CLI regression:
**63 passed** in 135.40 s; an additional immutable-input validator test passed
independently. Read-only local inspection of all **3443** production clusters
took **2.369 s** for API assembly with existing filesystem cache. The output is
`/tmp/pclass-dark-signatures-list.txt`; timings and source library identity are
in `/tmp/pclass-dark-signatures-list-benchmark.txt`. No fitting or data generation
was run.

Built and installed a wheel with a tiny bundled library, then verified API and
`--list --format json` outside the checkout. The API smoke test blocked heavy PLM
imports/network/package writes. Automated bundled-input tests additionally omit
all profiles/traces/relative artifacts to prove listing independence. This is
a synthetic packaging check rather than a release of the full local dataset.

Final verification: **212 full-suite tests passed, 2 opt-in tests skipped** in
406.73 s. The existing Torch `AMD_SERIALIZE_KERNEL` environment warning remains.
Ruff lint/formatting passed for 43 Python files; documentation links, fenced
blocks and scoped whitespace checks passed. The listing is CPU/file-IO only;
no GPU behavior, scientific fitting, artifact schema or query ranking changed.

## Twenty-bin metric distributions — 2026-10-29

Append a distribution section after the cluster table and explanatory note, in
text/Markdown. Count one observation per cluster for each non-null saved AUROC,
recall, specificity and precision value; do not weight by C/B window counts.
Use unrounded fractions and 20 fixed equal-width bins over [0,1], with edges
`i/20` for i=0..20. Bins are left-inclusive/right-exclusive except the final bin,
which includes 1. Values exactly on an internal edge enter the bin to its right.
Do not bin the three-decimal display strings or mix missing metrics with zero.

Display one aligned 20-row table with headers `AUROC_bin`, `Rate_bin%`, `AUROC_n`,
`Recall_n`, `Specificity_n`, `Precision_n`. The first column labels fraction
intervals with three decimals; the second gives the same intervals on the 0–100%
scale for the three rate metrics. Counts are integers, and empty bins remain
visible. Interval strings contain no embedded spaces. Report valid and unavailable
counts per metric in a second small table. All-null and empty libraries still
show 20 zero-count bins and the correct missing counts.

Include a structured `distributions` field in JSON and the immutable listing API:
`bin_count=20`, `edges` (unrounded fraction values), `closure` and `metrics`.
Each metric has `counts` (20 integers), `valid` and `unavailable`. Sum of counts
must equal valid; valid+unavailable equals the number of cluster rows. Compute
this descriptive summary from detached report rows without artifact reads,
score refitting or additional runtime dependencies. The listing constructor
maintains this derived field consistently when rows are supplied programmatically.

Tests cover all edges, exact 0/1, just-below-edge values, unavailable metrics,
rounding-sensitive values, unequal sample sizes (still one count per cluster),
all-null/empty inputs, 20-row rendering, JSON/API agreement and integer totals.
Existing table columns, order, description truncation and conditional Status are
unchanged. These are distributions of in-sample reference metrics, not validation
performance or query-match probabilities.

Implemented in `summary_distributions.py` and the listing snapshot/renderer.
Validation: **65 focused distribution/listing/description/CLI tests passed** in
41.91 s. Real-library totals reconcile to 3443 observations for each of the four
metrics, with zero unavailable values in this snapshot. Installed wheel text,
Markdown and JSON output passed from outside the checkout. Lint, formatting and
documentation checks passed. No embedding/training-suite rerun was needed for
this isolated descriptive aggregation and presentation change.
