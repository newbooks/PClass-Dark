# SPEC-0008: Explain a cluster signature

- Date: 2026-10-28.
- Status: Implemented (2026-10-28); validation recorded below.
- Readiness review: 2026-10-28, checked against current v2 production and
  relative-score loaders, profile/trace schemas, and unsupported-fit behavior.
- Purpose: Explain a saved production signature using its recorded evidence,
  selection history, metrics, and C/B reference distributions.
- Provenance: The user-reviewed `cluster_0001` narrative; scientific contracts in
  [SPEC-0003](0003-setup-data.md), [SPEC-0004](0004-build-library.md), and
  [SPEC-0005](0005-relative-score-distributions.md).
- Related: [SPEC-0007](0007-query-ranking-presentation.md) defines future query
  ranking. This command explains a cluster, without evaluating a query.

## Scope

Provide a human-readable, reproducible explanation of how a production signature
was derived. Every numerical claim must come from verified artifacts or an
explicitly defined descriptive calculation on those artifacts. Use deterministic
templates, not generated prose requiring an external language model or service.

The command must work with a distributed library without setup-data matrices,
raw sequences, model weights, neighboring repositories, or private checkpoints.
It must not fit a signature, embed sequences, resample background, download data,
or change selections. No new scientific scoring or selection rule is introduced.

Explain the algorithm's recorded decisions, without claiming biological causation,
a globally optimal subset, calibrated probabilities, or demonstrated performance
on unseen proteins. Do not infer a functional description from a cluster ID or
embedding dimension ID. A function label requires separately verified annotation
provenance; annotation lookup is outside this initial implementation.

## Command and Python interfaces

```bash
pclass-dark explain-signature --cluster cluster_0001
pclass-dark explain-signature --cluster cluster_0001 --detailed
pclass-dark explain-signature --cluster cluster_0001 --format markdown
pclass-dark explain-signature --cluster cluster_0001 --format json
pclass-dark explain-signature --cluster cluster_0001 --library-dir /shared/library
pclass-dark explain-signature -h
```

- Require one exact `--cluster` ID; do not accept patterns or multiple clusters.
  Reject repeated `--cluster` options with exit 2 rather than silently using the
  last value. Match the stored ID exactly; do not strip or normalize it.
- `--library-dir` identifies the library output root containing `selection.json`,
  not an individual bundle. Its default follows SPEC-0006 repository/cache/bundled
  input resolution. Explicit relative paths retain cwd-relative interpretation.
- `--format` accepts `text` (default), `markdown`, or `json`.
- `--detailed` adds full dimension and selection-trial evidence; it does not
  change calculations. Default output must remain useful for large signatures.
- Support both `-h` and `--help`. Help requires no artifact reads.
- Write the report to stdout for terminal use or ordinary shell redirection.
  Operational errors go to stderr. Do not mix INFO logs with report content.
  Assemble and render the complete report before writing stdout, so validation
  failures cannot leave a partial report.
- Use exit 0 for a successfully explained supported or unsupported signature,
  exit 1 for artifact/input failures, Typer exit 2 for invalid CLI syntax/options,
  and exit 130 for interruption. An unknown cluster is an input failure.

Reusable operations live in `pclass_dark.explanations`; the CLI is a thin adapter.
Public API to implement:

```python
from pclass_dark.explanations import explain_signature, render_signature_explanation

report = explain_signature("cluster_0001", library_dir=None, detailed=False)
text = render_signature_explanation(report, format="text")
```

`explain_signature` returns a structured `SignatureExplanation` with documented
fields and availability reasons; rendering must not read files or run scientific
work. Implement a frozen dataclass holding recursively frozen JSON-compatible
report data and a `to_dict()` method returning a detached mutable JSON-compatible
copy. The renderer returns a UTF-8-compatible string ending in one newline. The
JSON renderer serializes `report.to_dict()` with `allow_nan=False`, `indent=2`,
and insertion-order keys; it does not serialize the rendered narrative.
Validate nonempty string cluster IDs, real boolean `detailed`, and supported
format names. Use existing integrity exceptions for corrupt artifacts, `ValueError`
for invalid arguments/unknown clusters, and `OSError` for inaccessible files.
Imports must have no computation, argument parsing, or model-loading side effects.

## Inputs, verification, and snapshot consistency

Read the selected production manifest, representation contract, target signature,
selection trace, target rows from dimension profiles, and paired relative-score
metadata/arrays when present. Reuse existing library verification and score
contracts rather than create an alternative scientific implementation.

Resolve paths with `default_library_input_dir()` and inspect root/selector
existence before acquiring a guard, so missing inputs do not create directories.
Use `safe_file` and `encoded_cluster` rather than interpolating unchecked paths.
Reject symlink roots/ancestors and artifact symlinks consistently with the existing
artifact contracts. Reject unknown selection/manifest schemas explicitly.

Hold the existing output-root transaction guard across report assembly for mutable
roots. Use the existing immutable-resource path for installed bundled libraries;
never create locks or write within installed package resources. Render only after
assembling the snapshot, so a slow terminal cannot hold the root guard.

Verify manifests, checksums, selected IDs/digests, relative production binding,
cluster identity, dimension inventory, signature-to-trace identity, and final
selected dimensions. Confirm trace ranking, candidate/core membership, prefix
lengths, recorded AUROC counts, removal transitions, and termination agree with
stored profiles and signature. Reuse existing validators where available. Do not
replay selection over embedding matrices to establish these bindings.

An intentionally absent relative companion (a v1 selector for v2 production)
produces an unavailable distribution section. A declared selected companion with
missing/corrupt files or mismatched provenance is an error, not missing evidence.
Unsupported cluster distributions carry their saved status/reason.

Initial support targets production `pclass-dark.library.v2` and the paired
relative-score schema. For legacy v1 production manifests, fail with an actionable
request to run `build-library` to migrate; never silently reinterpret a six-fit artifact
as a final-only report. Optional separate research reports are outside the initial
scope. Clearly state that held-out results are not included in this explanation;
do not imply that no research exists elsewhere.

### Concrete loading path and compatibility

1. Under the root guard, read `selection.json` and the selected production
   manifest. Accept selection v1 or v2 only. Reject a production v1 manifest with
   migration guidance before running the expensive legacy six-fit validator.
2. For a production v2 manifest, call
   `load_selected_library(root, immutable=..., cluster_id=cluster_id)` under the
   same reentrant guard, validating only this target's dependencies. A selection v2 requires its complete companion;
   selection v1 pointing to production v2 is the supported no-companion case.
   Missing selectors, required fields or declared bundles are errors. Never scan
   for a different bundle when the selected generation is invalid.
3. Reuse the scoped library's immutable `target_result`, containing the parsed
   640 profile rows, final signature and single final trace, to avoid a second
   sequential profile lookup. Scoped production loading uses
   `library_reports.final_results(path, {cluster_id})` to assemble this snapshot.
   Scoped `load_production`/`validate_result` validate the target's scientific
   payload and table bindings. Add only the explanation-specific trace checks below.
4. If relative data is available, use its verified target metadata and arrays.
   Add `RelativeScores.summarize(cluster_id)` as a reusable target-only summary
   operation rather than access private metadata/cache internals from the CLI.
   It returns a detached, recursively frozen mapping with the distribution
   section fields defined below (including availability status/reason); unknown
   clusters raise `ValueError`. It uses the same root guard/immutable handling
   and production/relative generation-digest checks as `evaluate`. It must preserve
   checksum, dtype, shape, finite, sorted-order and min/max validation from
   `_read_arrays`. A memory-map option/shared internal validator may be added
   with default existing-reader behavior preserved and tested. Do not compute
   quantiles from unverified paths.
5. Detach the report and release the guard before rendering. On malformed fields,
   translate schema `KeyError`/`TypeError`/`AttributeError`/`IndexError` to
   `IntegrityError`, following existing `schema_checked` conventions.

The loaders import lightweight setup/embedding modules today. The requirement is
absence of setup-data **artifacts and execution**, plus no imports of heavyweight
PLM runtimes (`torch`, `transformers`, `huggingface_hub`); it does not require
refactoring all lightweight transitive imports for this reporting command.

### Explanation-specific trace checks

Use integer AUROC counts, with denominator `2 * C_n * B_n`, for defined
training comparisons. Validate finite stored values against their numerator/
denominator using the existing calculation tolerance, never display rounding.
Do not recalculate AUROC from embedding rows.

For supported fits, require the ranked profile inventory and trace order to
agree. Prefix lengths are exactly 1 through the rankable count (640 for currently
supported fits); derive the earliest maximum by numerator. Require the saved
peak, protected first two and ordered union candidate set to agree with this
prefix and the strict individual `>0.90` integer rule. Bind initial/final candidate
AUROC to `selection.initial_count`/`final_count`; bind the final to saved training
metrics. Do not infer candidate-union AUROC from a prefix if their sets differ.

Replay recorded decision **metadata**, not scientific scoring: `before` must be
the current ordered set and `before_count` the current AUROC; trial dimensions
must follow the eligible rank order, have no duplicates, and exclude the core.
For each trial, check `gain_numerator` against its count and the iteration's
before count. A successful iteration stops at its first strictly positive trial,
removes that dimension and sets `after_count` to that trial's count. A terminal
unsuccessful iteration visits all eligible dimensions with no positive gain,
keeps the set and count unchanged, and is last. If only the protected core
remains, an additional unsuccessful scan is unnecessary. The resulting set and
count must match the signature and final trace. Unsupported fits may have no
prefix/count/iteration fields; report unavailable evidence instead of applying
supported-fit assumptions. Only the current `first-improvement-logdiff-prefix-v2` selection version is
initially supported. If a different selection algorithm version is found,
fail with an actionable unsupported-algorithm error rather than narrate it as
first-improvement.

## Report data contract

Provide a versioned JSON representation, `pclass-dark.signature-explanation.v1`,
shared by the Python API and JSON renderer. Store numbers at full precision and
use JSON null for unavailable values; reject NaN/Infinity. Include:

- Source identity: library ID/manifest digest, cluster ID, fit identity, scientific
  payload ID, evaluation mode, representation contract, and relative ID if present.
- Population counts: C/B window counts and distinct parent-chain counts when
  recorded; identify counts that are unavailable rather than infer them.
- Dimension evidence: IDs, individual ranks/AUROCs, member/background means and
  SDs, R/D, effective Gaussian widths, protected status, selected status/weights.
- Selection evidence: ranked order, best prefix length/AUROC, candidate IDs,
  protected IDs, accepted removals with before/after AUROC, terminal scan outcome,
  and final IDs/AUROC. Detailed mode also includes all 640 profile rows, all 640
  prefix results and every recorded removal trial. Default mode includes the
  top three profile rows, all selected profile rows, and accepted-removal summary.
- Score definition and threshold: Gaussian/equal-weight composition, width floor,
  stored log cutoff and ordinary score cutoff or explicit predict-none status,
  threshold objective and tie rule.
- Metrics: saved training metrics, confusion counts, population prevalence, and
  explicit held-out availability. Do not relabel training metrics as validation.
- C/B score distribution summaries with counts, quantile convention, source role,
  availability reasons, and the ABC definitions.
- Limitations/availability messages selected by deterministic conditions.

Output ordering follows the narrative below. Dimension tables follow individual
rank; selected IDs preserve scientific scoring order. JSON arrays use the same
stable order, and omit volatile timestamps so repeated renders are reproducible.
No schema or identity changes to existing production/relative artifacts are needed.

### Required JSON shape and availability

Top-level keys, in order, are `schema`, `source`, `status`, `reason`, `detailed`,
`populations`, `representation`, `dimensions`, `selection`, `scoring`, `metrics`,
`distributions`, `abc_definitions`, and `limitations`. No paths tied to the local
installation, volatile timestamps, or lazy array/file handles enter the report.

- `source` contains `library_id`, `manifest_sha256`, `cluster_id`, `fit_identity`,
  `scientific_payload_id`, `evaluation_mode`, `representation_contract_id`,
  `relative_id` (nullable), `selection_schema` and `validation_scope="cluster"`.
- `populations` contains `C_windows`, `C_parents`, `B_windows`, `B_parents`,
  and `unavailable_reasons` mapping any unavailable field name to its reason.
- `representation` copies the verified recorded representation contract.
- `dimensions` contains `top` (up to three rankable profile records), `selected`
  (all selected profile records, with weights from the signature), and `all`
  (all 640 records only in detailed mode, otherwise null). Preserve existing
  profile field names; add `weight`, null for nonselected dimensions.
- `selection` contains `status`, `order`, `core`, `candidates`, `selected`,
  `best_prefix_length`, `comparisons`, `accepted_removals`, `termination`,
  `prefixes` and `iterations`. The last two are the full saved arrays only in
  detailed mode, otherwise null. Each comparison contains `labels` (a list, so
  identical sets can share a row), `dimension_ids` and `auroc` with
  `{value,numerator,denominator}`. Accepted removals contain `dimension`,
  `remaining_count`, `before_auroc`, `after_auroc` and `gain_numerator`.
  `termination` is `no_improving_removal`, `protected_core_only` or `unsupported`,
  derived from the validated final scan/set.
- `scoring` contains `epsilon`, `sigma_floor`, the saved `threshold` (including
  its kind), and `gaussian_formula="member-mean-gaussian-v1"`,
  `composite_formula="equal-weight-arithmetic-mean-v1"` plus formula descriptions.
- `metrics` contains the full saved `training` metrics or null, `training_unavailable_reason`, and `held_out={"status":"not_included","reason":"production_only"}`.
  Default rendering may omit AP/F1, but structured metrics retain all saved values.
- `distributions` has `status`, `reason`, `relative_id`, `quantile_method`, and
  `C`/`B` summaries. Set `quantile_method="linear-in-log-space-v1"`.
  Each available summary contains `n`, saved `role`, and ordered
  quantile records with `q`, `log_score`, `score`, `score_underflow`. Unsupported
  summaries are null with the saved reason; intentionally absent companions use
  `status="unavailable", reason="no_selected_companion"`.
- `abc_definitions` contains stable labels, direction and formulas, no numerical
  query scores. `limitations` is an ordered list of stable code/message records.

Use null plus a reason for missing scientific quantities. Use empty lists for
known empty sets such as an unsupported fit's selected dimensions; do not conflate
an empty set with missing evidence. A supported predict-none threshold is distinct
from an unsupported fit with null threshold. Availability reasons in the stored
metrics/profiles remain authoritative. The JSON `status`/`reason` mirror the
signature, while section-level availability is tracked independently.

## Narrative template

### 1. What the signature represents

Lead with the cluster ID, number of selected dimensions out of 640, status and
all-C evaluation mode. Say it is a pattern learned from reference windows.
For unsupported signatures, lead with the recorded reason and explain available
population/diagnostic evidence without fabricating a threshold or selected set.

### 2. Member and background windows

Explain chains and windows before presenting their counts:

- A chain of at most 1022 amino acids contributes its full sequence as one window.
- A longer chain contributes its first and last 1022 residues, named
  `N-<accession>` and `C-<accession>`. They overlap for lengths 1023–2043,
  meet without overlap at 2044, and omit the central region above 2044.
- Windows, rather than chains, are the C/B observations. Both terminal windows
  inherit the chain's retained cluster memberships. Exact full-sequence
  deduplication precedes windowing; terminal windows are not independently
  deduplicated by their sequence during setup-data.

Display actual C/B window and parent counts. Explain fixed out-of-cluster B
sampling, up to 200 windows per source cluster, with the existing parent exclusion
and deduplication rules. Do not claim that the cap equalizes all source clusters.
Explain 640-dimensional residue-only mean pooling using attention AND NOT
special-token masks. State that build-library consumes these fixed embeddings;
it does not pool tokens again. Gate model-specific wording on the recorded
representation contract rather than assume an unrecorded model.

### 3. Evidence in individual dimensions

Introduce means/SDs, R and D in plain language, with their existing formulas:

```text
R = sigma_C / (sigma_B + epsilon)
D = abs(mu_C - mu_B) / (hypot(sigma_C, sigma_B)/sqrt(2) + epsilon)
```

Use sample SDs. Lower R means tighter member variation relative to background;
higher D means larger standardized mean separation. Neither is a calibrated
confidence measure. They support interpretation; individual Gaussian training
AUROC determines the ranking. Briefly describe Gaussian similarity here as
closeness to the member mean with the member SD setting tolerance, deferring the
full score formula to section 6.

Show the top three dimensions with means/SDs, R/D and individual AUROC. Include
all selected dimensions in a compact parameter table (detailed mode exposes the
remaining profiles). Do not call numeric dimensions biochemical mechanisms.

### 4. Ranked prefixes and candidate cutoff

State descending individual AUROC order, numeric-ID ties, and evaluation of all
ranked prefixes (640 for supported current fits). Each prefix combines Gaussian similarity scores by an
equal-weight mean and is evaluated by composite training AUROC; individual AUROCs
are not averaged to form composite AUROC.

Explain the earliest maximum-AUROC prefix and the union candidate rule: top two,
best prefix, and any dimensions with individual AUROC strictly greater than 0.90.
The prefix is an initial candidate cutoff, not the later acceptance threshold.
Use recorded dimensions/counts throughout. If extra >0.90 dimensions extend the
prefix, state how many were added; never assume candidates equal the prefix.

Show AUROC for the best dimension, top two, chosen prefix, all dimensions, initial
candidate union and final set. Merge duplicate rows when their dimension sets are
identical, while retaining labels. Handle fewer than two rankable dimensions with
explicit availability reasons, not invented values.

### 5. Removal reasoning

Name protected dimensions. Describe rank-ordered, first-improvement removal and
restart, with strict AUROC improvement. Show each accepted removal and its
before/after AUROC and remaining count. Explain the recorded terminal condition.
When a complete terminal scan is recorded, explain that removing any retained
removable dimension did not improve the final composite;
a protected dimension was retained by policy, not tested for final removal.
Do not claim that removed dimensions have no biological relevance or are weak
individually. Do not claim global subset optimality or that every dimension was
tried in every iteration. Detailed mode exposes rejected trials and their gains.

### 6. Composite score and acceptance threshold

Explain the full score immediately before threshold and metrics:

```text
width_j = max(sigma_C,j, sigma_floor)
g_j(x) = exp(-0.5 * ((x_j - mu_C,j) / width_j)**2)
S(x) = sum(g_j(x) for selected dimensions) / number_selected
```

A dimension has Gaussian similarity 1 at its member mean, exp(-0.5) about 0.607
at one effective width away, and exp(-2) about 0.135 at two widths away.
The score is an arithmetic mean, not a product or a Gaussian probability density.
Explain that computation uses stable log scores, mathematically log(S), and an
ordinary displayed score may underflow; the log cutoff remains authoritative.

State the fitted cutoff, inclusive comparison, balanced-accuracy objective and
highest-cutoff tie rule. A predict-none threshold explicitly accepts no windows.
Do not invent a finite threshold in this case. No query example is included.

### 7. Metrics and their consequences

Display saved AUROC, recall, specificity, balanced accuracy, accuracy, precision,
and TP/FP/TN/FN. Explain AUROC as ranking separation (with half credit for ties),
and explain the actual numbers of accepted C/B windows. Include average precision
and F1 in detailed mode. Describe the reference class imbalance alongside
precision; do not equate sampled-reference precision with deployment precision.

Always label all-C metrics in-sample: C and B contributed to discovery and
threshold selection. Balanced accuracy is (recall + specificity)/2, not precision
optimization. Report missing/undefined metrics with their saved reasons. A final
fit has no held-out positives; optional research is separate from this report.

### 8. Reference distributions and ABC match evidence

After the signature is fixed, score C/B distributions supply future match context.
Report counts and min, 25th percentile, median, 75th percentile, 95th percentile,
99th percentile and max for each available distribution. Calculate summaries on
stored sorted log scores with NumPy `quantile(method="linear")`, then exponentiate
for displayed ordinary scores; label that convention in JSON. Preserve log
quantiles too. These are descriptive summaries, not fitted Gaussian distributions.
For sorted logs `a` of length n, use `h=(n-1)*q`, `i=floor(h)`,
`j=ceil(h)`, and `(1-(h-i))*a[i] + (h-i)*a[j]`, with direct endpoints when
`i==j`. This defines interpolation without sorting/copying the population;
validate against NumPy's linear method within float64 tolerance. A single value
has the same result for all quantiles. Available arrays must be nonempty; empty
available arrays are corruption. An exponentiated zero from finite negative
logs sets `score_underflow=true` and renders as "underflow; log score ...", not a
mathematical zero. These log-space interpolated quantiles need not equal linear
quantiles calculated after exponentiating the entire population.

Define A as acceptance against the fixed threshold, B as the plus-one background
upper-tail fraction, and C as the inclusive member percentile (SPEC-0005).
Lower B is stronger background-relative evidence; higher C is higher rank among
members, not necessarily greater typicality. No single ABC value belongs to a
cluster independently of an evaluated window. Do not include a hypothetical or
real query, combined confidence, probability interpretation or ranking output.

## Presentation and resource constraints

Use plain narrative paragraphs with compact tables, consistent dimension labels
and explicit window/chain units. Default text must be readable without ANSI color,
terminal-width assumptions or truncated values. Markdown is an exportable report;
JSON contains structured evidence rather than only rendered prose.

Use approximately six significant digits for ordinary statistics and scores,
percentages with sufficient precision to preserve small nonzero values, and
scientific notation when necessary. Compare full-precision recorded values before
rounding. Selection improvements smaller than display precision must show their
scientific-notation delta. Never display an unavailable value as zero.

This is a single-cluster read/format operation: no GPU, worker pool, or new
numerical dependencies. Stream shared compressed profile tables and retain only
the target's rows. Validate selected manifest identities and only requested
cluster files. The shared profile table still requires a complete compressed-byte
checksum read and sequential lookup to the target's rows; do not read unrelated
signatures, traces or score arrays. Memory-map the target arrays and use bounded
scans for required verification. Do not require loading all clusters'
relative arrays or the embedding matrix into memory.

## Validation and completion

Automated tests must cover:

1. Independently calculable Gaussian values, arithmetic composite examples,
   R/D, percent formatting, sorted-log interpolation and ABC formula descriptions.
2. Known synthetic selection histories: prefix ties, >0.90 union additions,
   protected dimensions, accepted/rejected removals, terminal non-improvement,
   and tiny positive gains that ordinary rounding could hide.
3. Correct population units, terminal-window boundaries, attention/special-token
   explanation, confusion counts, low precision under imbalance and in-sample
   labels. Include zero SD, unsupported fits, predict-none and undefined metrics.
4. CLI `-h`/`--help`, required/unknown cluster, invalid format, text/Markdown/JSON,
   no ANSI reliance, stable repeated output, and API/CLI evidence agreement.
5. Corrupt/mismatched signatures, traces, selections and companions; deliberately
   absent companions; concurrent selection replacement; immutable bundled inputs.
6. Installed-library operation with setup-data artifacts absent, heavyweight PLM
   imports blocked, network denied, and package writes forbidden. Reject legacy
   v1 production with migration guidance; accept a v1 selector for v2 production.
7. A small real `cluster_0001` manual inspection when local artifacts exist,
   without making its data a mandatory test fixture or regenerating signatures.

Update README/help/API docs, SPEC index and chronological progress log when
implemented. Record executed tests and limitations; update this specification with implementation status and executed validation
before considering the command complete.

## Local example evidence (illustrative snapshot)

The selected local library inspected on 2026-10-28 has ID
`5f9117a77f64b30426661dff87533a462a8c0771c7150ae76eb903574a5e8100`.
Its `cluster_0001` final fit has C=20 windows/20 parents and B=213241 windows/206085
parents. The prefix/candidate count is 15; protected dimensions are 12 and 184.
Removing 567 then 279 leaves 13 dimensions and AUROC 0.9966748889753847.
The cutoff is 0.549914086149925; TP=20, FN=0, FP=5656, TN=207585.

These are examples for reviewing narrative fidelity, not constants, golden
scientific expectations for future libraries, or runtime dependencies on local
untracked data. Read current artifacts to produce every report.

## Implementation sequence and readiness

1. Add structured report records and pure renderers with small independent tests.
2. Add guarded artifact assembly, trace checks and target distribution summaries;
   reuse production validation and retain existing reader defaults.
3. Add the thin Typer adapter, help aliases and user-facing examples.
4. Run report/library/relative-score and CLI regression suites, including installed
   read-only behavior, then inspect a local cluster report if data is available.

Readiness review resolved selection/production schema combinations, actual loader
entry points, unsupported trace shapes, core-only termination, exact report keys,
no-companion versus corrupt-companion handling, and bounded quantile calculation.
No new scientific decision is required: this command explains the existing method.
It introduces report formatting and integrity validation only. The readiness review preceded the implementation recorded below.

The earlier documentation-only revision passed local link, fence and whitespace
checks. Runtime validation for the implementation is recorded below.

## Implementation record — 2026-10-28

Implemented the Typer command and importable operations. Modules are
`explanations.py` (immutable report/guarded assembly/public rendering API),
`explanation_trace.py` (integer-bound recorded-decision validation), and
`explanation_render.py` (pure text/Markdown narratives). `relative_scores.py`
provides public target summaries using verified read-only memory maps. No artifact
schema, fitting algorithm, threshold or query-ranking behavior changed.

The actual saved predict-none threshold kind is `above_maximum`; renderers retain
that schema and describe it as predict-none. The all-equal score regression
confirms highest-cutoff ties select this threshold and precision is undefined.

Verified a real `cluster_0001` Markdown report from selected library
`5f9117a77f64b30426661dff87533a462a8c0771c7150ae76eb903574a5e8100`
(3443 targets), including its 15 candidates, removals 567/279, 13 selected
dimensions, 0.549914 acceptance cutoff and in-sample confusion counts. Report
artifact: `/tmp/pclass-dark-cluster0001.md` (local manual-check output, not a
required distributed input).

Built and installed an isolated wheel with a small library and no setup-data
artifacts. From `/tmp`, Python/CLI default bundled resolution passed for JSON,
text and Markdown reports, with heavy PLM imports and network blocked and package
writes forbidden. This wheel test uses synthetic references, not the full local
reference distribution.

Final validation: full regression **167 passed, 2 opt-in tests skipped** in
322.73 s, with the existing Torch `AMD_SERIALIZE_KERNEL` environment warning.
After final display/error normalization and added argument/symlink cases,
**49 focused explanation, relative-score and CLI tests passed** in 51.55 s.
Ruff lint/format checks passed (40 Python files), as did scoped diff whitespace,
local documentation links and Markdown fence checks. GPU execution is not needed
for this read/format operation. No scientific fitting or query-ranking changes
were introduced.

## Targeted validation revision — 2026-10-28

Replace the full-library loading step for explanations with an explicit
`load_selected_library(..., cluster_id=<requested ID>)` scope. Its default without
a cluster remains a full-library audit for existing callers. Scoped production
loading verifies selected manifest identity/digest and target inventory, the
requested signature and trace, and the shared compressed profile table before
parsing just the requested 640 rows. Reuse existing scientific-result validators.
Metrics come from the verified signature; the redundant shared metrics table is
not needed for this report. Do not inspect other clusters' signatures or traces.

Scoped companion loading verifies its selected manifest digest/identity, complete
identity target inventory against the production manifest, source production and
setup bindings, and only the requested distribution metadata and C/B arrays.
Do not open other clusters' metadata/arrays or the redundant global summary table.
Missing/corrupt requested dependencies remain errors; unrelated broken artifacts
must not block an explanation. Full loaders must still detect those artifacts.
The shared profile file must still be read for checksum verification; locating a
late cluster also requires sequential decompression up to its rows. Scoped `Library.target_result` retains only the requested parsed fit in memory
for immediate report assembly, eliminating a duplicate sequential profile lookup.
Full library loads keep this field null. No new file format, duplicate persisted
profiles, cache trust policy or persistent validation cache is introduced. Manifest inventory checks do not imply validation of unrequested data.

Add `source.validation_scope="cluster"` to explanation reports. Tests must prove
scoped/full report evidence agrees, unrelated files are not read, requested
corruption still fails, and ordinary unscoped loaders retain full validation.
Benchmark a saved cluster report without fitting or modifying the data.
This revision supersedes earlier full-loader/cost wording in this specification.

Targeted revision validation: **180 full-suite tests passed, 2 opt-in tests
skipped** in 348.13 s (existing Torch environment warning). After request-local
row reuse, the final **57 focused tests passed** in 73.63 s, including file-read
scope/count assertions and equality with full-load evidence. Installed-wheel
Python/CLI use passed outside the checkout with no setup-data, PLM imports,
network or package writes. Lint, formatting and documentation checks passed.

Final local timings with existing filesystem cache: cluster_0001 CLI including
startup **0.415 s**; last target cluster_3524 **7.244 s** for API assembly,
reflecting its sequential shared-profile lookup. Prior to removing the duplicate
lookup, that last-target measurement was 14.324 s. These are local observations,
not cold-cache or cross-machine performance guarantees. No full-loader baseline
was timed in this revision.
