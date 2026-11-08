# SPEC-0024: Human-readable prediction interpretations

- Date: 2026-11-07.
- Status: implemented and validated.
- Parent: [SPEC-0023](0023-final-ml-model.md).
- User template: the shortened `work/seq2-interpretation.md`, read before changes.
- Purpose: produce its predicted-function / plain-language / full feature-support
  write-up directly from predict-ml.py and pclass-dark predict-ml.

## Interface and compatibility

Default to `--format markdown`; explicit tsv and jsonl remain available. Output/stdout
staging and refusal to overwrite existing files remain unchanged. Multiple FASTA
queries produce independent Markdown sections in input order, separated by `---`.
No hosted language model, network lookup, model retraining or user-template code
execution. Existing model bundles remain loadable without mutation/retraining.

Prediction API adds explicitly named chosen_feature_contributions, chosen_bias,
chosen_logit, training_example_count and sequence_length fields; JSON includes them.
Original feature_contributions, margin and bias_difference keep their comparative
semantics for backward compatibility. Raw sequences carry their exact input length;
precomputed rounded ln_length can only yield an approximate length, labeled as such.

## Chosen-cluster interpretation contract

For predicted local class c and saved centered spline basis B(x), calculate each
feature's own contribution sum_k B_jk(x_j)*W_ck. This is relative to that curve's
mean across all training examples, not a difference against another cluster or
its probability contribution. Sum of 18 terms plus saved bias must reconstruct the
chosen raw logit within atol/rtol=1e-4. Use original feature order for calculation;
presentation table sorts descending support, stable source order for ties.

Class names/scores and all text come from actual prediction fields. Describe
positive terms as more supportive than their average training contribution,
negative terms as less supportive, |term|<=1e-6 as numerically near average.
Summarize up to three strongest positive descriptors, then up to three additional
ones; identify up to three most negative descriptors and name remaining smaller
negative effects in the table. If no supporting/opposing terms exist, state this
rather than fabricating evidence. Relative strength descriptions compare magnitudes
within the same query only. Do not infer monotonic effects, protein mechanism,
calibrated probabilities or correctness from these contributions.

Readable values: fractions as one-decimal percentages; hydropathy and charge proxy
as three decimals; exact raw sequence length in residues, or approximate exp(rounded
ln_length) for feature tables. Missing inputs display NA (imputed) and actual saved
imputed contributions; do not invent observed values. If finite ln_length cannot
be exponentiated safely, show log length and no guessed integer. No imputed/OOS
flags are omitted: add a data-quality note when either applies. Exposure/Q3 are
predicted descriptors, charge is a composition proxy. Correlated descriptors share
information and are not independent causal mechanisms.

## Functional labels

Descriptions are optional and never change prediction. A local resolver locates
the extraction contract's preparation through configured setup paths and verifies
its saved manifest SHA-256/ID. It then binds live ProFunc membership/dataset files,
cluster metadata and GO graph to that original reference before accepting labels.
Reuse the existing validated load_catalog/reference utilities; neighboring checkouts
are not runtime dependencies. Build the catalog once per Markdown command, not
per query. If original preparation/runtime metadata are absent or mismatched,
report cluster IDs and annotation-unavailable text, logging the reason. Do not
silently use same-named clusters from a different reference snapshot. No downloads.
Formatter API can accept an explicitly supplied descriptions mapping for offline
integration; the caller then owns reference provenance.

## Report structure

1. Prediction interpretation: escaped query ID.
2. Predicted function: exact/approximate length, chosen cluster's bound annotation,
   its score, and "second choice" annotation/score if available. Explain scores
   are normalized and uncalibrated. No "runner-up" wording.
3. Plain-language explanation from actual own-class signed contributions and
   observed descriptors, keeping activity language conditional on the learned model.
4. Support from each feature: centering reference/count, sign meanings, 18-row table,
   noncausal/correlation and predicted-structure/charge-proxy notes. No extra
   baseline-reconstruction/reproduction sections removed by the user's template.

Escape query IDs and annotation text for Markdown/HTML. Values/rows are derived,
not interpolated executable templates. Pure formatter render_interpretation(prediction,
*, descriptions=None)->str; numerical helper chosen_support(classifier,X,classes)
returns [N,18] terms and [N] biases, using bounded batch input and saved parameters.
No import-time logging, runtime discovery or device initialization. CLI lazily
resolves annotation metadata only for Markdown; TSV/JSON need no live catalog.

## Acceptance

- Analytical two-class spline fixture: own contribution versus pairwise difference
  are deliberately distinct; reproduce chosen logit and comparison independently.
- Formatter positive/negative/zero/missing/out-of-support cases, exact and approximate
  length, stable feature order, injection-safe text, absent label metadata.
- Bound-description resolution, missing/mismatched metadata and source snapshot
  checks; no live labels accepted from a mismatched reference.
- Existing TSV/JSON values unchanged; added JSON own-class fields correct. CLI
  Markdown/API parity, multi-query separators and atomic late-failure behavior.
- Run final-model/exploration/CLI tests, real seq2 smoke report and compare its chosen
  contributions to independently saved work/seq2-chosen-cluster-support.json.
  Preserve the user's editable report; publish generated output to a new path.
- Update usage/API guide, README, spec index and progress with actual checks/results.

## Validation results

Implemented in ml_interpretation.py, ml_model.py and ml_model_cli.py. Existing
bundles load unchanged; Markdown is the default (revised after initial validation). Relevant model, exploration,
CLI and interpretation suites: **77 passed, 2 skipped** (GPU unavailable in the
sandbox). After the final wording adjustment, six interpretation tests passed
again. Ruff and git diff --check passed.

Raw seq2 FASTA inference completed on CPU with local offline checkpoints, producing
work/seq2-generated-interpretation.md. All 18 own-class contributions also matched
the independently saved work/seq2-chosen-cluster-support.json within 1e-7. Original
user-edited work/seq2-interpretation.md was preserved. Annotation provenance checks
accepted the current matching reference. No model weights were changed and no new
held-out accuracy claim is made.

Default-format revision: Markdown stdout and --output file behavior are covered
by the model/interpretation regression suite; 21 passed in 5.44s.

## Default file publication revision

Markdown now writes a file when --output is omitted: beside FASTA/feature input,
using <input-stem>-interpretation.md. Inline --seq uses <sanitized-query-id>-
interpretation.md in the current directory (default query-interpretation.md).
Multi-query inputs produce one combined report. Explicit --output overrides the
path; --stdout prints instead and cannot be combined with --output. TSV/JSONL
retain stdout unless --output is supplied. Existing files are never overwritten;
all modes retain atomic publication and late-error protection. Report the saved
path on stderr after success. Tests cover automatic names, inline input, collision,
stdout and failure cleanup.

Automatic-file revision validated: 22 model/interpretation tests passed; Ruff
and git diff --check passed. FASTA/TSV and inline naming, collisions, explicit
stdout and late-failure atomicity covered.

## Top-five prediction summary

Add a Top 5 predictions table between predicted-function text and the explanation.
Preserve ranked prediction order, showing up to five available clusters, their IDs,
short descriptions and softmax scores (three decimals). Normalize description
whitespace and truncate to 24 displayed characters including a trailing ellipsis
when needed (23 characters plus …). Escape after truncation for Markdown/HTML;
missing annotations display Unavailable. Keep full annotations in the existing
predicted-function prose. Do not renormalize scores within the top five.

Top-five validation: 23 model/interpretation tests passed, including ranking,
24-character boundary/ellipsis, whitespace normalization, escaping, missing
annotations and fewer than five available classes. Ruff and diff checks passed.

## Terminal summary

When saving Markdown, also print each query ID and its top-five table to stdout,
with the same shortened labels and scores as the report. Stage summaries on disk
and emit only after successful report publication; failed inputs/collisions emit
no partial table. Explicit --stdout already prints the report and needs no second
summary. TSV/JSONL outputs retain their existing behavior.

Terminal-summary validation: 23 model/interpretation tests passed, including
multi-query table parity and collision/late-error output protection; Ruff and
git diff --check passed.
