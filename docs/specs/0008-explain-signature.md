# SPEC-0008: Explain a cluster signature

- Date: 2026-10-28.
- Status: Draft; implementation is not started.
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
- `--library-dir` identifies the library output root containing `selection.json`,
  not an individual bundle. Its default follows SPEC-0006 repository/cache/bundled
  input resolution. Explicit relative paths retain cwd-relative interpretation.
- `--format` accepts `text` (default), `markdown`, or `json`.
- `--detailed` adds full dimension and selection-trial evidence; it does not
  change calculations. Default output must remain useful for large signatures.
- Support both `-h` and `--help`. Help requires no artifact reads.
- Write the report to stdout for terminal use or ordinary shell redirection.
  Operational errors go to stderr. Do not mix INFO logs with report content.
- Use exit 0 for a successfully explained supported or unsupported signature,
  exit 1 for artifact/input failures, Typer exit 2 for invalid CLI syntax/options,
  and exit 130 for interruption. An unknown cluster is an input failure.

Reusable operations live in `pclass_dark.explanations`; the CLI is a thin adapter.
Proposed public API:

```python
from pclass_dark.explanations import explain_signature, render_signature_explanation

report = explain_signature("cluster_0001", library_dir=None, detailed=False)
text = render_signature_explanation(report, format="text")
```

`explain_signature` returns a structured `SignatureExplanation` with documented
fields and availability reasons; rendering must not read files or run scientific
work. Validate nonempty string cluster IDs, real boolean `detailed`, and supported
format names. Use existing integrity exceptions for corrupt artifacts, `ValueError`
for invalid arguments/unknown clusters, and `OSError` for inaccessible files.
Imports must have no computation, argument parsing, or model-loading side effects.

## Inputs, verification, and snapshot consistency

Read the selected production manifest, representation contract, target signature,
selection trace, target rows from dimension profiles, and paired relative-score
metadata/arrays when present. Reuse existing library verification and score
contracts rather than create an alternative scientific implementation.

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

An intentionally absent relative companion (for example, an older library)
produces an unavailable distribution section. A declared selected companion with
missing/corrupt files or mismatched provenance is an error, not missing evidence.
Unsupported cluster distributions carry their saved status/reason.

Initial support targets production `pclass-dark.library.v2` and the paired
relative-score schema. For legacy v1 selections, fail with an actionable request
to run `build-library` to migrate; never silently reinterpret a six-fit artifact
as a final-only report. Optional separate research reports are outside the initial
scope. Clearly state that held-out results are not included in this explanation;
do not imply that no research exists elsewhere.

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
640 ranked prefixes. Each prefix combines Gaussian similarity scores by an
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
A retained removable dimension did not improve the final composite when removed;
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
Avoid a full-array copy where sorted-array indexed interpolation suffices.

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
the target's rows. Existing manifest verification may scan the library; document
that cost and do not bypass integrity checks. Memory-map relative arrays and use
bounded scans for required verification. Do not require loading all clusters'
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
6. Installed-library operation with setup-data/PLM imports unavailable, network
   denied, and package writes forbidden. Reject legacy v1 with migration guidance.
7. A small real `cluster_0001` manual inspection when local artifacts exist,
   without making its data a mandatory test fixture or regenerating signatures.

Update README/help/API docs, SPEC index and chronological progress log when
implemented. Record executed tests and limitations; keep this specification marked
planned until the command and importable operations actually exist.

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

Draft validation: local documentation links, Markdown fence balance and
whitespace checks. No runtime behavior is changed by this specification.
