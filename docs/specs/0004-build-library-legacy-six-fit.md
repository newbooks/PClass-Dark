# Historical six-fit build-library specification

This document preserves the implemented six-fit contract and its experimental
records through 2026-10-27. Its mandatory five-fold execution, six-fit publication,
identity, and cleanup rules are superseded by the revised
[SPEC-0004](0004-build-library.md). It is a historical reference, not the target
implementation contract. Numerical evidence below remains historical evidence;
it does not validate the newly planned mode separation.

---

# SPEC-0004: Fold-wise PLM signature discovery with build-library

- Date: 2026-10-26.
- Status: Implemented; numerical, recovery, publication, packaging, and scoped
  real-data checks completed. Independent predictive validation remains future work.
- Inputs: [SPEC-0003 setup-data](0003-setup-data.md).
- Reference: ProStruct `scripts/profile_clusters.py`,
  `scripts/compose_signature.py`, `src/prostruct/cluster_profiles.py`, and
  `src/prostruct/signatures.py` (SPEC-0028/0029/0030).

## Purpose and boundary

Adapt ProStruct's conserved-feature signatures to the 640 coordinates of frozen
ESM-2 window embeddings. Discover signatures independently in each target
cluster's five fold rotations, evaluate training and held-out positives against
its fixed background, and report dimension-level evidence and signature stability.

The observations are window records; the candidate features are the 640 scalar
embedding coordinates. "AUROC for each embedding" means AUROC for each
**dimension**, not AUROC for an individual protein or one pooled vector.

This command consumes prepared data and embeddings. It does not change folds,
resample B, rerun sequence processing, download models, or pool tokens again.
It requires neither profunc nor ProStruct at runtime. Signature discovery,
evaluation, and serialized model/scoring functions live in reusable modules;
the CLI is a thin adapter.

## Command and Python interface

```bash
pclass-dark build-library --data-dir work/setup-data --output-dir work/library
```

```python
from pclass_dark.library import build_library, load_library

result = build_library(
    data_dir="work/setup-data",
    output_dir="work/library",
    workers=1,
    memory_mib=1024,
    epsilon=1e-8,
    sigma_floor=1e-8,
)
library = load_library(result.library_dir)
```

| Option | Default | Contract |
|---|---|---|
| `--data-dir` | `work/setup-data` | Root containing a verified setup-data `selection.json`; may be an installed immutable release root |
| `--output-dir` | `work/library` | Writable library root, disjoint from input roots and installed resources |
| `--workers` | `1` | Positive requested target-worker count; bounded by available CPUs, selected targets, workspace budget, and 32 |
| `--memory-mib` | `1024` | Total explicit numerical workspace budget across workers; positive integer |
| `--epsilon` | `1e-8` | Positive finite safeguard in R/D |
| `--sigma-floor` | `1e-8` | Positive finite Gaussian width floor |
| `--cluster` | All retained targets | Repeatable exact target ID for a scoped build/benchmark; unknown or duplicate IDs are errors |
| `--force` | false | Archive and recompute only the matching library and its resumable checkpoints |

The API uses `cluster_ids=None` or a sequence of target IDs; other option names
use underscores. Output scope is part of library identity: a one-cluster build
cannot be mistaken for a full library. A scoped build still uses the original
B derived from all retained source clusters; do not restrict B to the requested
target subset. No random seed, split, background-cap, threshold-optimization,
or selected-dimension-count override is exposed here.
These are fixed by verified setup inputs and the approved algorithm.

Return frozen `LibraryResult(library_id, library_dir, counts, reused)`.
Use the existing `IntegrityError` and `SetupBusyError`, `ValueError` for invalid
options, and `OSError` for IO failures. CLI usage errors exit 2, execution errors
exit 1, Ctrl-C exits 130, and a completed build exits 0 with explicit usable/
unsupported-target counts. No imports start work. Logs use `[LEVEL] message`.

Read the exact prepared/embedding pair selected by `selection.json`. Require
completed embeddings, not just prepared data. Validate both manifest-byte
digests, reference/inventory identities, compact C/B bindings, matrix checksum,
window-ID joins, dtype, finite values, width 640, and target-specific fold coverage.
Reject missing, escaping, or symlinked selection paths. Verify every expected
(cluster_id, window_id) fold assignment exactly once, including cross-label
windows; source window/matrix dimensions must stay in canonical order.
Never equate a local C index to a matrix row index.

Separate reusable modules: `library` for orchestration/loaders, `profiles` for
moments and individual AUROC, `signatures` for composites and selection, and
`signature_metrics` for thresholds/metrics. Reuse existing artifact/cluster-set
helpers rather than importing neighboring projects. Build and score depend on
core NumPy/SciPy/filelock only; Torch, Transformers, and the checkpoint are not
required once embeddings exist.

## Populations for each target and rotation

For retained target cluster k, define:

- C_all,k: its entire prepared window set, uncapped.
- V_k,f: its windows assigned to target-specific fold f, f in 0..4.
- C_k,f: C_all,k minus V_k,f, the four training folds.
- B_k: the prepared, fixed background for k, reused unchanged in all five rounds.

Do not interpret another target's fold IDs as a global split. N/C windows of one
parent and identical windows from different parents may cross folds, as approved
in SPEC-0003. Results are window-level internal validation, not independent-parent
or homology-separated evaluation. Include their overlap diagnostics in reports.

B_k follows the existing compact setup definition: sample at most **200 windows
per other retained source cluster**, uniformly without replacement, once per
source cluster; union by window ID; exclude every known target parent across
all source labels. No refill after exclusion or duplicate removal. The user's
"accessions" description does not change the previously approved window unit.
Unclustered sequences and excluded clusters do not enter B. Never redraw B per
fold, dimension, candidate subset, or metric.

B is large but is not an independent negative test set: it influences ranking,
selection, and thresholds, and is reused in reported validation metrics.
Record its counts and identity rather than imply that size eliminates bias.

## Residue-only embedding provenance

Accept only the supported embedding contract from setup-data: pinned
`facebook/esm2_t30_150M_UR50D`, final-layer 640-dimensional hidden states,
1022-residue terminal windows, and float32 stored vectors. Setup-data must have
used the token-position mask

```text
residue_mask = attention_mask.bool() & ~special_tokens_mask.bool()
pooled[d] = mean(hidden[t,d] for t where residue_mask[t])
```

BOS/CLS, EOS, and padding are positions to exclude, not feature columns.
Validate model/tokenizer revisions, pooling algorithm, residue policy, dtypes,
window inventory, and its recorded token-length checks before using vectors.
Reject incompatible or legacy pooling provenance; do not silently mix it with
supported embeddings. A pooled matrix alone cannot prove upstream masking was
correct: retain setup-data's analytical, tokenizer, and actual-checkpoint tests
as dependencies of this contract. No attention/token masks are applied to the
640 columns during signature discovery.

## Fold-specific per-dimension profiles

For every dimension j in 0..639, target k, and rotation f, calculate in float64:

- mu_C,k,f,j and sigma_C,k,f,j from C_k,f only.
- mu_B,k,j and sigma_B,k,j from the fixed B_k.
- Sample standard deviations use ddof=1; cache B moments once per target.

Relative conservation:

```text
R_k,f,j = sigma_C,k,f,j / (sigma_B,k,j + epsilon)
```

Smaller R indicates lower within-cluster dispersion relative to B. R<1 is
supporting evidence, not a required feature-selection cutoff or significance test.

Discrimination by standardized mean difference:

```text
D_k,f,j = abs(mu_C,k,f,j - mu_B,k,j)
          / (sqrt((sigma_C,k,f,j**2 + sigma_B,k,j**2) / 2) + epsilon)
```

D is unsigned, with equal weighting of C/B variances. Larger D indicates greater
mean separation; it is descriptive evidence, not a p-value. R/D require at least
two observations in both populations. Epsilon and sigma_floor default to 1e-8,
must be positive and finite, and are recorded numerical safeguards in the raw
embedding coordinate units. No validation-dependent adjustment is permitted.

Retain five values of each profile statistic per cluster/dimension. Never fit
moments on C_all,k before running folds. V_k,f does not contribute to training
means, spreads, ranking, candidate selection, or the decision threshold.

## Gaussian compatibility and individual-dimension AUROC

Use fold-specific target parameters:

```text
width_k,f,j = max(sigma_C,k,f,j, sigma_floor)
g_k,f,j(x) = exp(-0.5 * ((x[j] - mu_C,k,f,j) / width_k,f,j)**2)
```

g is high near the training-cluster center. It is a compatibility kernel, not a
normalized density, likelihood ratio, or posterior membership probability.
Background moments are not substituted for its target mean or width.

For each j, calculate training AUROC from scores on C_k,f (positive) and B_k
(negative). AUROC is the fraction of positive-negative pairs with higher positive
score, adding half credit for exact ties. Compute it using the ordering of
`abs(x[j] - mu_C,k,f,j)`, reversed so smaller distance is better, to avoid
exponential underflow creating spurious ties. Implement sorted rank/binary-search
methods, not an explicit positive-by-negative comparison matrix.

Also calculate validation AUROC from V_k,f versus the same B_k using unchanged
g_k,f,j. Validation AUROC is a diagnostic only and must never rank features.
Require a defined training sample SD (at least two C training windows); AUROC
itself supports at least one positive and one negative, including a one-positive
validation fold, while reporting its low support. Empty populations yield an
explicit unavailable metric and reason, not a fabricated zero.

Rank all 640 eligible dimensions by descending **training** individual AUROC;
break ties by the original numeric dimension index. R and D are reported but
do not introduce extra filters or weights. With valid finite input, every
dimension including constant dimensions is eligible if its training moments and
AUROC are defined; lack of discrimination is not a schema failure.

### Numerical reference and statuses

Initial implementation uses CPU float64 numerical kernels for this rank- and
branch-heavy discovery algorithm, with single-threaded inner numerical libraries.
No reduced precision, autocast, approximate AUROC, gain tolerance, GPU-dependent
selection, or silent backend switching. Embedding generation retains setup-data's
GPU preference. A future discovery accelerator must be versioned and validated
against the reference; adding an unverified GPU path is not required here.

C/B member order is ascending window ID. Extract contiguous float64 vectors from
the immutable float32 matrix for each moment calculation. Use NumPy mean and
sample SD on that fixed order, never merge worker-dependent partial reductions.
Record numerical-engine, NumPy/SciPy versions and platform architecture in the
library identity. Block sizes affecting reductions belong to the algorithm
version, not an adjustable execution option.

For AUROC, retain the integer numerator `q = 2*wins + ties` and denominator
`2*n_positive*n_negative` as well as the float value. All candidate comparisons
within one fit share populations; compare q exactly for ranking, prefix maxima,
and removal gains. Individual-dimension distance ordering avoids Gaussian underflow ties.
The v2 removal backend uses near-tie reference fallback as described below;
bitwise equivalence of fast intermediate scores is not promised.
Count using overflow-checked integers or Python integers. This implements strict
AUROC improvement without changing the pair-count definition.

A missing/empty B yields `empty_background`; fewer than two training C windows
yields `insufficient_training`; fewer than two eligible features yields
`insufficient_core`. Emit an explicit fit-status record and unavailable profile/
metric fields as appropriate. An empty validation fold leaves training/final
fits intact and marks validation metrics `empty_validation`. A one-member B
supports AUROC but not R/D or sample background SD; do not exclude dimensions
solely because R/D are unavailable. Never silently filter nonfinite matrix values
or dimensions: these indicate corrupt input or a numerical failure.

Use checked float64 arithmetic. Overflow or nonfinite Gaussian log-intermediates
must stop the affected computation with target/rotation/dimension context while
preserving checkpoints; do not clamp scores, alter the floor, or label a failed
computation as a successful low-scoring signature. Supported-status records are
statistical insufficiency, not a catch-all for software/IO failures.

## Composite signatures and dimension selection

For selected dimension set J in one target/rotation:

```text
S_k,f,J(x) = (1 / len(J)) * sum(g_k,f,j(x) for j in J)
```

This is an equal-weight arithmetic mean, not a product or geometric mean.
Compute its logarithm by stable log-sum-exp minus log(len(J)), then evaluate
ranks and thresholds on the log scores. Gaussian widths matter for composite
scores even though a single dimension's AUROC depends only on distance order.
Record all parameters at full computational precision. The numeric reference
accumulates log kernels with left-to-right `numpy.logaddexp` in rank order,
starting with the first kernel, then subtracts `log(len(J))`. Exp-transformed
scores/thresholds are display values only; underflow to zero must never replace
their finite log values in comparisons. Never multiply Gaussian kernels.

Adapt the ProStruct algorithm to 640 dimensions, with the user-approved
first-improvement removal policy:

1. Evaluate all ranked prefixes of lengths 1..640 on C_k,f versus B_k. Record
   composite training AUROC and incremental changes; do not look at V_k,f.
   For the one-dimension prefix, use the individual distance-order AUROC,
   matching ProStruct's underflow-safe special case. Longer prefixes use the
   composite log score.
2. Identify the earliest prefix attaining the maximum training prefix AUROC.
3. Protect the top two individually ranked dimensions as the mandatory core.
4. Candidate set = the top two, all dimensions through the best prefix, and
   every dimension with individual training AUROC strictly greater than 0.90,
   including dimensions beyond that prefix. Test the 0.90 cutoff using exact
   counts (`10*q > 9*denominator`), not a rounded display AUROC.
5. Starting from that union, evaluate removing each non-core candidate. Accept
   the first removal giving a strict increase in composite training AUROC,
   visiting candidates in original individual-rank order. Stop that round
   immediately on improvement and restart from the first remaining candidate.
6. Recompute removal effects after every accepted deletion. Stop when none
   improves AUROC; never remove either protected dimension or reinsert dimensions.

Evaluate every eligible dimension during initial ranking and prefix profiling;
backward removal applies to the candidate union, not automatically all 640.
The candidate rule matches ProStruct; first-improvement removal is an approved
PClass-Dark deviation. Do not replace the candidate rule with a truncation,
correlation filter, R/D cutoff, or a fixed signature size without a recorded
scientific decision. If fewer than two dimensions are eligible, no signature is
published for that round; report `insufficient_core`.

Use float64 score arithmetic and exact integer AUROC-count comparisons with
the specified tie conventions for this initial reference algorithm. Record attempted removals and accepted gains. Selection
is greedy and not a global optimality claim. No Gaussian-distribution assumption
is required to interpret the empirical AUROC pair count.

## Training and validation signature metrics

After selecting J, evaluate its composite scores on:

- In-sample: C_k,f versus B_k.
- Validation: V_k,f versus B_k, with the same selected dimensions and Gaussian
  parameters. No selection step or parameter is refit using validation positives.

Required metrics for both populations: AUROC, precision, recall, specificity,
and accuracy, together with TP/FP/FN/TN, positive/negative counts, and prevalence.
Also retain balanced accuracy, F1, false-positive rate, and average precision
for interpretation and comparison to ProStruct.

### Approved threshold rule: training balanced accuracy

AUROC needs no cutoff; precision/recall/specificity/accuracy do. The user approved
selecting a threshold maximizing balanced accuracy on **C_k,f versus B_k only**,
then applying that exact threshold to V_k,f versus B_k unchanged. Consider all
distinct training score groups and predict-none; break equal objective values by choosing the highest threshold.
Maximize the integer objective `TP*n_negative + TN*n_positive`, equivalent to
balanced accuracy on a fixed population, to avoid floating-point tie drift.
Use score >= threshold, store log threshold and its score-space interpretation.
Represent predict-none explicitly rather than a nonfinite JSON literal.

Do not separately optimize a validation threshold or transfer a threshold fitted
on all positive folds. Precision is unavailable when there are no predicted
positives, recall when there are no positives, and specificity when there are no
negatives. Report other undefined quantities explicitly. Never replace missing
values with zero. Finite thresholds, ties, all-constant scores, and predict-none
must have independently tested behavior. Average precision uses grouped exact
score ties and `sum(precision_at_group * recall_increment)`, not trapezoidal PR
area. Confusion counts are integers; threshold-dependent validation specificity
and false-positive rate must equal their training counterparts because B and
the selected cutoff are unchanged within a fit. Precision and accuracy can
change simply because the positive prevalence differs; retain those counts.

For every rotation, save the threshold-selection objective, training confusion
counts, and unchanged validation cutoff. Background-side performance remains
conditional on B used during discovery, even when C positives are held out.

When V is empty, keep the validation population counts and confusion counts for
B, but mark validation status `empty_validation`. Any mathematically defined
background-only diagnostics may be retained with that status; exclude the entire
round from validation aggregate metrics so repeated B observations do not create
an apparent held-out result. Retain supported training/discovery statistics.
For nonempty V and valid fits, aggregate each metric over its finite defined
values only; an unavailable precision does not invalidate a defined recall.

## Five-rotation aggregation and stability

Store each fold's profiles, ranking, selected dimensions, parameters, search
trace, threshold, training metrics, and validation metrics separately. Different
folds may legitimately select different dimensions and signature sizes.

For each cluster and **each of all 640 dimensions**, report mean, sample SD
(ddof=1), valid-fold count, and all five original values for:

- Training target mean and sample SD.
- Relative conservation R and discrimination D.
- Individual training AUROC and individual validation AUROC.
- Numeric rank where available.

Do not aggregate only selected dimensions: that would hide unstable or rejected
features. Also report selection count/frequency (with valid discovery rounds as
denominator), protected-core count, and per-round selected flag. Do not average
selected parameter vectors as if dimensions were interchangeable.

For each cluster, report mean/sample SD/valid-fold count of every training and
validation signature metric and signature size. A single valid observation has
no sample SD; report unavailable. Zero valid observations yield null mean/SD,
not zero. Selection frequency is null when no discovery fit was valid.
Always include per-fold counts and failure reasons. Fold estimates overlap in training data, and B is identical across
rounds: SD describes variation across these rotations, not a confidence interval
or standard error from five independent experiments.

Save target-positive out-of-fold scores, keyed by `(cluster_id, window_id)`,
with fold and signature identity. Emit exactly one row for every C_all,k window:
a supported held-out fit supplies its score, while an unsupported fit supplies
null scores/acceptance and its explicit reason. Do not silently drop windows.
Each available score comes from a signature that did not train on its fold. Do not treat five copies of
B as five independent negative datasets in pooled performance. Different rounds
have different score functions; raw out-of-fold scores are not automatically
comparable across rounds or clusters. Use foldwise metric summaries as primary
reports until a pooled-score calibration rule is specified.

## Artifacts and portable library

Output bundle:

```text
<output-dir>/<library-id>/
  manifest.json
  cluster_metrics.tsv
  cluster_summary.tsv
  dimension_summary.tsv.gz
  dimension_folds.tsv.gz
  dimension_final.tsv.gz
  out_of_fold_positive_scores.tsv.gz
  clusters/<encoded-cluster-id>/
    fold_0.json ... fold_4.json
    final.json
    selection_trace.json.gz
```

Each fold record contains full selected-dimension identities and Gaussian
parameters, threshold/status, counts, metrics, and input identities. Global
profile tables retain every dimension and all rotations. Compress large audit
TSVs, stream output, and do not duplicate embeddings or expanded B memberships.
Use safe reversible filenames and preserve original cluster IDs inside records.

Bind the library identity to prepared folds/C/B identities, embedding inventory
and matrix digest, algorithms, epsilon/floor, selection rule, threshold rule,
and numerical conventions. Record software versions and execution provenance.
Validate artifacts before reuse, publish under a transaction lock, checksum all
outputs, support archive/rebuild, and keep progress logs. The checkpoint and
publication contract below defines resumable progress inside each fit.

Completed libraries must be importable/scorable without the neighboring repos
or discovery-time worker processes. Release artifacts belong under
`src/pclass_dark/resources/library/<library-id>/` with package-data and LFS rules;
working output defaults remain outside installed resources. A release selection
manifest must choose exact library/reference/embedding identities.

### Approved production signature: refit on all C members

After completing the five discovery/validation rotations, fit one final signature
per target using **C_all,k versus the same fixed B_k**. Recompute target moments,
R/D, individual AUROC ranking, all 640 prefixes, the protected-core candidate
union, backward removal, and the balanced-accuracy-maximizing threshold on these
full populations. Use identical algorithm rules, epsilon/floor, tie breaking,
and dimension IDs as the fold fits. Do not average parameters across folds,
choose the best validation fold, select by validation results, or build a fold
ensemble. The final selected dimensions may differ from every fold-specific set.

Save the complete final scoring parameters, selected dimensions, ranking,
selection trace, threshold, counts, and full-data training metrics in `final.json`.
Retain final per-dimension R/D/moments/AUROC as a separate full-data profile,
not a sixth cross-validation observation. Fold means, sample SDs, selection
frequencies, and valid-fold counts use only the five rotations.

Mark final metrics as **full-data in-sample**. The final refit has no new held-out
C members; do not reuse its training observations as a final validation set or
attach the fold metrics as though the final parameters produced those scores.
Link the final signature to the five-fold summaries as evidence about discovery
stability and internal validation of the procedure. Final-fit failure must have
an explicit status and must not silently substitute a fold model.

A later query implementation uses this final signature and its training-derived
threshold as the baseline scoring artifact. This cutoff is not a calibrated
probability or a guarantee of deployment precision. Cross-cluster score comparison,
multiple/rejected assignments, and N/C aggregation remain separate decisions.

## Serialization, exact schemas, and library identity

Use zero-based dimension IDs 0..639, fold IDs 0..4, and fit labels `fold_0` through
`fold_4` plus `final`. Keep every dimension parameter tied to its numeric ID.
The order used for score accumulation is the stored individual-rank order,
restricted to the selected subset, not an arbitrary set/dictionary order.

Text is UTF-8/LF. JSON uses sorted keys, compact separators, no NaN/Inf, and
one terminal newline. Hash canonical JSON without that newline. TSV floats use
17 significant digits for round-trip float64 precision; unavailable cells are
empty, with status/reason fields, and JSON unavailable values are null.
Gzip streams use `mtime=0` and no original filename, making worker-independent
content reproducible. Metadata timestamps are provenance, not identity inputs.

Use a cluster path component `c_` plus lowercase hexadecimal UTF-8 cluster-ID
bytes. This is reversible, avoids special path components and case-fold clashes,
and keeps the original ID in records. Reject path components exceeding 200 ASCII
bytes before doing work; never truncate IDs or silently skip clusters.

Required per-fit signature JSON schema `pclass-dark.signature.v1`:

- `library_id`, `cluster_id`, `fit_id`, `status`, and explicit failure reason.
- `input_identity`: prepared/embedding manifests, fold/C/B digests, selected
  member-list digests, inventory and model/pooling contract identity.
- `counts`: training/validation positive and fixed-background window counts,
  distinct-parent counts, and positive prevalences.
- `selected_dimensions`: ordered records with `dimension`, `individual_rank`,
  `mu`, `sigma`, `effective_sigma`, `R`, `D`, individual training/validation AUROC
  with exact numerator/denominator fields, `core`, and `weight=1/m`. Core/candidate/selected ID arrays and selection trace
  identity must agree; means and widths are not rounded display values.
- `representation_contract_id`: canonical hash of checkpoint/tokenizer asset
  digests, model/revision, layer/width, window length, pooling, residue policy,
  inference and stored dtype. Execution versions remain library provenance and
  identity inputs; do not confuse a specific training matrix ID with the
  representation contract that a new query must match.
- `threshold`: `kind` (`log_score_ge` or `above_maximum`), `log_value`, and
  `score_value` for display. Both values are null for predict-none. For finite
  thresholds always compare in log space, even if display exp(log_value)
  underflows to zero.
- `training_metrics`, `validation_metrics`, and provenance. Final validation
  metrics are null with `not_applicable_full_refit`, never copied from folds.

Metric objects contain `positive_n`, `negative_n`, `tp`, `fp`, `fn`, `tn`,
`auroc`, its exact numerator/denominator, `precision`, `recall`, `specificity`,
`accuracy`, `balanced_accuracy`, `f1`, `false_positive_rate`, `average_precision`,
`positive_prevalence`, and per-field unavailable reasons. Store diagnostics at
unrounded precision. A predict-none threshold evaluates false for every query;
do not replace it with an arbitrary finite cutoff.

Long-form table schemas (columns in this order):

| File | Columns |
|---|---|
| `dimension_folds.tsv.gz` | cluster_id, fold_id, dimension, positive_n, background_n, mu_C, sigma_C, mu_B, sigma_B, effective_sigma, R, D, training_auroc, training_auroc_numerator, training_auroc_denominator, validation_auroc, validation_auroc_numerator, validation_auroc_denominator, rank, core, candidate, selected, status |
| `dimension_final.tsv.gz` | cluster_id, dimension, positive_n, background_n, mu_C, sigma_C, mu_B, sigma_B, effective_sigma, R, D, training_auroc, training_auroc_numerator, training_auroc_denominator, rank, core, candidate, selected, status |
| `dimension_summary.tsv.gz` | cluster_id, dimension, statistic, mean, sample_sd, valid_folds, selection_count, selection_frequency, core_count |
| `cluster_metrics.tsv` | cluster_id, fit_id, population, metric, value, positive_n, negative_n, status |
| `cluster_summary.tsv` | cluster_id, population, metric, mean, sample_sd, valid_folds, final_status |
| `out_of_fold_positive_scores.tsv.gz` | cluster_id, window_id, fold_id, signature_digest, log_score, score, accepted, status |

Dimension-summary `statistic` enumerates `mu_C`, `sigma_C`, `R`, `D`,
`training_auroc`, `validation_auroc`, and `rank`. All five source values remain
in the fold table; no final-fit value participates in these summaries. Cluster
summary includes required metrics and `signature_size` (population `discovery`).
Table keys must be unique; sort by cluster ID, fit/fold order, dimension, then
statistic/metric name as applicable. Preserve rows with unavailable values.
`population` is `training`, `validation`, or `discovery`; final has training
metrics only. Table metrics include confusion counts and prevalence as well as
rates. Final `dimension_final` contains 640 rows per target, not just selected
dimensions; unsupported fits keep rows with null values/status. Fold dimension
tables contain exactly 5*640 rows per target, and summary rows retain their
fixed statistical keys even when all five values are unavailable.

`selection_trace.json.gz` records each fit's ranked dimension IDs, every prefix
AUROC/count and delta, candidate union/core, each removal trial's subset identity
and exact AUROC count, selected deletion, and stopping reason. Store trials in
original rank order and fits in fold_0..fold_4,final order. Retain completed
iterations even when a later computation fails. Bind a signature to the canonical
per-fit trace payload digest, not the whole trace file's hash; traces do not
contain signature-file digests. OOF rows reference the actual completed fold-JSON
file digest. The global manifest binds all files without a circular dependency.

The manifest schema is `pclass-dark.library.v1`. Its identity inputs include
selected target IDs; exact prepared/embedding manifest, matrix, row-map, folds,
C and B file digests; inventory/model contract; epsilon/floor; all algorithm/
schema/threshold versions; float64/count conventions; and numerical-engine
versions/architecture. Hash this canonical object for `library_id`. Exclude
paths, worker count, workspace budget, timestamps, and hardware scheduling.
Record complete identity inputs, not just the resulting hash.

The manifest records target/status counts, output file-byte hashes/sizes, and
all fit identities. It does not hash itself; external selection records bind
its actual bytes. Reject missing/extra required fields, duplicate JSON keys,
invalid dimension IDs, mismatched parameter lengths, invalid widths/weights,
unknown schema versions, and unverified fit/library bindings before scoring.
Unknown optional metadata may be retained without changing scientific behavior.

## Resumable computation and safe publication

Use `<output-dir>/.checkpoints/<library-id>/<encoded-cluster>/<fit-id>/` for
private work. Work units are six fits per target, scheduled target-by-target;
process the five folds then final refit. A resumed final fit must not invalidate
completed fold results. Statistical insufficiency is a completed status unit;
IO/numerical failures are not.

Each fit has stages with content-bound JSON receipts:

1. **Profiles:** commit blocks of 16 dimension moments/individual scores and
   exact AUROC counts, bound to population digests and dimension ranges. After
   all 640 are verified, commit complete ranking and fitted parameter digests.
   A rank-dependent stage cannot resume against partially profiled dimensions.
   Cache B moments once per target in a separate verified target-level receipt
   bound to the B-ID digest, matrix identity, and numerical algorithm.
2. **Prefixes:** commit progress every 16 dimensions, including finished prefix
   results and durable float64 running log-sum arrays for training C and B.
   Last commit's prefix length and ordered dimension digest bind those arrays.
3. **Backward selection:** commit after each completed removal trial and every
   accepted deletion, because one trial can itself be expensive. Bind partial
   trials to the exact current subset/core/iteration;
   persist the current exact AUROC count and deterministic trial visit order. Reuse
   unchanged Gaussian parameters, not moments refitted after each removal.
4. **Evaluation:** commit selected signature, training-derived threshold,
   training/validation metrics, dimension rows, positive OOF rows, and full trace.
   `final` has no OOF/validation-positive output.

Profile/prefix blocks of 16 dimensions and single-trial backward checkpoints
are fixed constants in the current algorithm, independent of inference/setup batch size and worker
scheduling. A restart may repeat only uncommitted work
within the active stage/block. Gaussian caches may be regenerated from verified
inputs and parameters; their existence does not establish stage completion.

For every receipt, require schema, library/fit/stage identities, population and
parameter digests, block range, shapes/dtypes/counts, and hashes of every durable
payload. A partial profile block binds only the dimensions/parameters completed
so far; the full parameter/rank digest is required once all profiles are done.
Use zero-based half-open ranges. An iteration receipt names its current subset,
next trial index, completed trials, and current exact count so a restart cannot
skip or repeat a committed deletion. Write payloads to unique temporary files, flush/fsync and rename, then
publish the receipt last. Structural JSON errors (including non-object metadata),
invalid ranges, stale identities, nonfinite payloads, or checksum failures go
through quarantine/recompute for private checkpoints. Never swallow an exception
and claim completion. Verify dependent stages; rebuilding an invalid earlier
stage invalidates its downstream receipts unless their complete dependency
identities still match. Do not reuse a prefix state under a different rank order.

Use persistent nonblocking file locks under each root's `.locks/`. The coordinator
holds the library output-root transaction guard until selection publication or
rollback finishes; readers and concurrent invocations cannot select a generation
that is still rollback-eligible. Acquire the mutable setup input-root transaction
guard for the whole operation so its selection/inputs cannot be force-replaced
under mapped readers. Acquire input/output root guards in sorted absolute-path
order, and reject equal/overlapping roots or output paths inside installed
resources.
Take guards before reading selection/manifests. Immutable installed inputs need
no write locks. Workers receive validated immutable descriptors while the
coordinator retains ownership, rather than reacquiring its locks in subprocesses.

Cache/checkpoint writers have one owner per fit. The coordinator alone publishes
final outputs. On interruption or worker failure, leave validated receipts intact;
never silently switch algorithm, precision, dimension count, or background size.
Changing workers/workspace budget must reuse scientifically identical checkpoints.
`--force` archives only the selected library's completed/private state before
intentional recomputation; normal resume needs no force.

After all selected targets' six units have completed or explicit unsupported
statuses, merge report fragments in canonical order, validate fit and summary
joins, checksum the staged bundle, and publish its manifest last. Unsupported
final fits remain status records; do not substitute fold models. A complete
library can have zero usable signatures, which must be reported explicitly.
Temporary arrays, caches, and receipts never enter the release bundle.

Atomically write `<output-dir>/selection.json` with schema, library ID, relative
library path, and manifest-byte digest only after publication. Retain root
ownership through this step or rollback; restore previously selected bundles on
ordinary failure. Different library IDs remain separate and are never pruned by
force. Completed output corruption is an error requiring explicit rebuild;
private checkpoint quarantine must not repair a published library in place.

Log fit/phase transitions, target completion counts, and throttled progress at
least every 30 seconds during long loops. Include cluster ID, fit ID, and
completed versus total dimensions/trials. Distinguish verified checkpoints from
uncommitted work. Log detailed per-trial values at DEBUG, not INFO; emit final
library completion only after selection publication. Record failures and resume
counts without reporting partial fits as durable completed targets.

## Runtime loading and scoring API

`load_library(library_dir, *, immutable=False)` validates completion, output
checksums, schemas, fit references, model/pooling provenance, thresholds, and
parameter-to-dimension joins while holding the mutable output-root read guard.
For this layout the mutable library root is `library_dir.parent`, and that is
the transaction-guard root. Do not reuse setup-data's `parent.parent` inference,
which applies to its `prepared/<id>` and `embeddings/<id>` trees. Checkpoint
loaders receive the root explicitly. Extend common helpers with an explicit
root/domain argument or use library-specific wrappers; do not guess a library's
lock root or rollback ownership from an embedding/preparation path heuristic.
Installed resource directories are automatically immutable/read-only; explicit
`immutable=True` asserts that an external release cannot be replaced. A caller
must not use that flag to bypass an active writer on a mutable working root.
Return an immutable in-memory snapshot of final signatures, detached from paths
that a later force rebuild could replace. Large diagnostic tables may be loaded
separately with verified read guards; they are not prerequisites for each score.

Expose `score_signature(signature, vectors, *, embedding_contract)` for one
verified final signature and a finite NumPy matrix of shape `(n, 640)`. Check
model/pooling/residue-policy compatibility, select dimension columns by stored
IDs, and return float64 log compatibility plus accepted/rejected booleans using
the stored training cutoff. The same function scores fold signatures for tests.
Unsupported-status signatures raise a documented unavailable-signature error.
No PLM inference, arbitrary vector normalization, cluster competition, forced
label, or multiwindow aggregation occurs in this API. Output row order follows
the caller's input row order, and arrays have no shared writable library state.
Ordinary scoring uses NumPy only after loading, not discovery workers or Torch.
For arbitrary caller vectors, validate checked Gaussian arithmetic and report
numerical overflow with the signature ID instead of silently normalizing them.

The v1 release contains the complete declared library bundle, including reports;
do not omit a file still listed by its manifest. Loading keeps only final scoring
parameters in memory after validating output hashes; scoring does not rescan
reports for every query. A slim release schema is outside this version.

Publish `resources/library/selection.json` separately from setup-data's existing
`resources/selection.json`, preserving both selectors and their distinct schemas.
The library selector uses schema `pclass-dark.library-selection.v1`, library ID,
relative path, library manifest-byte digest, and originating prepared/embedding
identities. Use that same selector schema for `<output-dir>/selection.json`.
Package-data rules include only the declared JSON, TSV, and GZIP files at the
library root and per-cluster paths, plus the library selector. LFS tracks
`resources/library/**` as one resource tree. Do not capture `.checkpoints`,
locks, staging, archives, or arbitrary working files with a broad package glob.
Training/discovery and installed scoring require no neighboring repo at runtime.

## Computational and validation requirements

640 dimensions and large B make naive repeated subset recomputation expensive.
Use memory-mapped Gaussian log caches shaped `(640, n_population)` with each
dimension contiguous, and row blocks of at most 4096 for cache generation.
Allow no more than the requested total explicit numerical workspace; divide it
across active workers, reserving at least 64 MiB per worker and reducing worker
count when needed. Reject a budget below 64 MiB. This budget excludes interpreter
metadata, OS page cache, and read-only input mapping; do not describe it as a
hard process-RSS limit. Keep one fit's mappings active per worker and
record their sizes; on-disk caches can accumulate across its six fits until
cluster-completion cleanup. Use memory-mapped/blockwise arrays when caches exceed RAM.
Estimate scratch requirements from population sizes and mapped cache shapes
before starting a fit; fail with required/available bytes if storage is clearly
insufficient. A later disk-full exception must preserve validated checkpoints,
not trigger dimension/background reduction. Never allocate pairwise C-by-B
matrices or keep every target's expanded B in RAM. Read shared embeddings via
memory mapping; schedule independent target jobs with bounded processes, each
processing its five folds and refit sequentially with one inner numerical thread.
This avoids nested target/fold pools and permits per-target B-moment reuse. The v1 discovery backend is CPU float64 as specified above; report
that device explicitly rather than implying GPU execution. Accelerating PLM
embedding remains a setup-data responsibility.

Implementation optimizations are tested against ordered reference scores and
selection traces using the same approved removal policy on independent fixtures.
The v2 fast path is mathematically equivalent but not claimed bitwise identical. A safe prefix optimization reuses the exact
left-to-right log sum before the trial's omitted dimension, then accumulates
remaining selected dimensions in the same rank order. Rebuild these caches after
every accepted deletion and bind them to the subset/parameter identities.
Subtracting exponentiated totals to test removal
can suffer cancellation; verify numerical stability rather than blindly porting
a fast subtraction rule. Do not silently subsample B, reduce dimensions, alter
sampling weights, or relax gain comparisons to make execution cheaper. Profile
one representative large cluster before a full production run.

Required tests include:

- Input checksum/schema mismatches, compact-index corruption, inventory/row
  misalignment, unsupported pooling/model provenance, and nonfinite vectors.
- Hand-calculated fold means/SDs/R/D, constant features and zero spread,
  epsilon/floor behavior, insufficient samples, and target exclusion from B.
- Gaussian/log scores and brute-force AUROC reconciliation, including distance
  ties, concentrated target distributions, underflow, and one-positive validation.
- A held-out-data perturbation test: changing V values cannot change its round's
  fitted means, ranking, selected subset, training threshold, or training scores.
- Exact 640-dimension ranking/prefix coverage; candidate union, strict >0.90,
  protected top two, first-improvement removal, deterministic visit order, rescanning,
  stopping, and insufficient-core status. Compare to an independent reference.
- Arithmetic-mean versus geometric/product counterexamples, unchanged fold
  parameters during validation, and no validation-based model selection.
- Training-only balanced-accuracy threshold selection and confusion-count tests,
  prevalence-sensitive precision, undefined metrics, class imbalance, unchanged
  validation thresholds, and highest-threshold objective ties.
- Full-C refit recomputes every discovery step and threshold using fixed B,
  serializes a usable final signature, preserves fold results, and never treats
  final R/D/metrics as a sixth fold. Test final-fit failure without fold fallback.
- Five original metric values, hand-calculated sample means/SDs/counts,
  absent-fold handling, zero/one-valid-fold aggregates, selection frequencies,
  stable dimension IDs, exact AUROC-count serialization, and explicit unsupported
  OOF rows without accidental omission.
- Fixed B across rounds, target-specific fold semantics, permitted sibling/window
  overlap diagnostics, and no duplication of negatives in pooled summaries.
- Serial/parallel consistency, reproducible gzip/table ordering, numerical
  workspace bounds, and memory-mapped cache fallback with identical results.
- Resume within profiles/prefix/removal/evaluation stages; missing and malformed
  receipts, stale parameter/subset identities, corrupt running arrays, downstream
  invalidation, interruption at trial/deletion commit boundaries, and changed
  workers/budget. Instrument evaluator calls to show committed work is reused.
- Pause force publication before selection, reject concurrent mutable reuse,
  exercise rollback and cross-process locking, and verify the old selection
  still resolves to the restored manifest. Verify readers guard the actual
  `<output-dir>` for this shallower layout, not its parent. Installed immutable
  loading must not create locks. Test source/wheel resource inclusion without private files.
- Loaded final-signature scores match discovery scores, parameter ordering and
  log-space acceptance survive serialization, input contract mismatches fail,
  and no query N/C aggregation or cross-cluster label selection is introduced.

No tests or production signature run are claimed by this specification; these
are implementation acceptance requirements.

## Open question 1: How to assess predictive power?

Recommended evidence separates three aspects:

1. Discrimination: held-out-positive AUROC versus fixed B, with training versus
   validation gaps and cluster-size/support counts. High AUROC does not by itself
   establish useful precision or an acceptance threshold.
2. Stability: R/D/AUROC variation, selected-dimension frequencies, signature sizes,
   and variation in held-out metrics over the five rotations.
3. Operating performance: recall at an acceptable false-positive rate or precision
   target, assessed with explicit class prevalence and a threshold chosen without
   its evaluated positives. The desired precision/coverage tradeoff is a scientific
   decision, not implied by balanced-accuracy optimization.

Fixed capped B is not deployment prevalence and is not independently held out.
Future cross-cluster competition, independent background tests, and comparisons
against full-vector nearest neighbors would strengthen assessment of the actual
project goal. Do not present this initial internal protocol as proof of recovery
on distant proteins or a calibrated probability model.

## Open question 2: How to assign a query to a cluster?

Raw S_k values are not automatically comparable across clusters: selected
features, signature size, spread, and score distribution differ. Do not simply
select the largest raw score and force a label.

A proposed future query workflow is to compute the identical frozen PLM/window
representation, score candidate signatures, apply validated cluster-specific
acceptance criteria, and allow multiple accepted clusters or no assignment.
Possible cross-cluster evidence includes background-relative score percentiles
or empirically calibrated error rates; neither is a posterior probability by
default. Query N/C aggregation (mean, maximum, or separate evidence), ambiguous
cluster handling need explicit choices. The production signature is the approved
full-C refit; no fold ensemble is used.
Max aggregation changes false-positive behavior and requires its own validation.

These open questions belong to the production calibration/query specification;
they must not be silently decided inside the dimension-selection algorithm.

## Implementation readiness and boundaries

The implementation follows the approved four-fold discovery,
training-balanced-accuracy threshold, held-out C evaluation, fixed capped B,
and all-C final refit. Interfaces, numerical semantics, compact input checks,
outputs, within-fit checkpoints, resource bounds, publication/rollback ownership,
and runtime scoring are specified. Implementation verification and the scoped benchmark are recorded below; no full
production build has been performed.

No additional scientific decision is needed for this command. Query-level
predictive confidence, cross-cluster competition, and N/C aggregation remain
explicitly outside build-library; do not resolve them implicitly during coding.
If performance work would require fewer dimensions, a different B, altered
selection rules, approximate ranks/gains, or validation-driven choices, stop and
request a scientific decision rather than treating it as a technical shortcut.

## Implementation and verification — 2026-10-26

Implemented in `library.py`, `profiles.py`, `signature_metrics.py`,
`signatures.py`, `library_checkpoints.py`, and `library_runtime.py`, with a thin
CLI adapter. Numerical workers use `threadpoolctl` to limit inner threads to one.
Memory estimates conservatively reduce concurrent workers; B index arrays are
expanded only for active tasks. Gaussian caches are private resumable files
while a cluster is unfinished; completion cleanup reclaims them as specified below.
Publication explicitly declares the library domain to retain correct rollback
ownership independently of the output directory name.

The full suite passed: **79 passed, 2 opt-in real-checkpoint tests skipped**.
Four additional publication/rollback regression tests and a final full-suite
rerun (79 passed, 2 skipped in 147.35 s) passed after the explicit
publication-domain change. Tests cover independent numerical oracles, serial/
parallel equivalence, held-out isolation, checkpoint recovery, reader locks,
rollback, immutable scoring, and invalid inputs. A staged wheel and source
archive containing a small complete library were built; the installed wheel
loaded and scored outside the checkout with PLM imports blocked and no package
writes.

A scoped real-data build completed all five rotations and the final refit for
`cluster_0945` (915 C windows, 213,061 B windows). Five-fold validation AUROC was
0.994898 ± 0.001078 (sample SD); the final in-sample AUROC was 0.995025, with
21 selected dimensions. These are internal diagnostics, not independent
predictive evidence. See the [benchmark record](../research/notes/2026-10-26-build-library-benchmark.md)
for data identities, timing scope, commands, metrics, and limitations.

Removal-trial progress messages identify both the target cluster and fit
(`fold_0`–`fold_4` or `final`) alongside iteration and completed-trial counts.

## Approved performance revision — 2026-10-26

The user approved first-improvement removal and fast leave-one-out arithmetic.
Version the library algorithm and fit/selection checkpoint dependencies so old
best-improvement outputs and receipts cannot be silently reused. Keep all five
rotations, fixed B, candidate screening, protected core, and training-only fitting.
Visit removable dimensions in original individual-rank order. Accept the first
strict improvement and restart; a terminal round checks all remaining candidates.
This order-dependent path can differ from the previous best-improvement path.

Per selected subset, cache the full log sum and at most eight running-prefix
snapshots per population. Fast deletion uses log subtraction of a single kernel
from the cached total, then subtracts log(K-1). Use `expm1` for stable subtraction.
For cancellation-prone rows, reconstruct the omitted-dimension sum in original
rank order from the nearest preceding cached prefix. Rebuild subset caches after
each accepted removal; reuse scratch output arrays within a round. Prefix and
reference scoring also use in-place output buffers without changing order.
Fast arithmetic is mathematically equivalent but is not claimed bitwise equal.
For each row, a removed-term log share greater than -1e-4 triggers ordered
recomputation. Nearest cross-population scores within
`1e-8 * (1 + max(abs(score_C), abs(score_B)))` trigger reference scoring of
the full trial (a numerical fallback, not a minimum-gain threshold). Proposed gains
are confirmed using reference scores before commitment. Final fitting metrics,
thresholds, and runtime scoring continue to use rank-ordered reference scoring.
Bound cache memory in worker admission estimates. Tests must cover cancellation,
underflow, ties, cache invalidation, first-versus-best paths, interruption/resume,
and serial/parallel determinism. Record a focused timing comparison without
claiming an unmeasured whole-library speedup.

Validation of v2: **91 tests passed, 2 opt-in tests skipped** in 142.86 s;
Ruff lint/format and whitespace checks passed. Installed wheel scoring passed
without PLM imports or package writes. The focused real-kernel comparison
completed 158 trials 3.45× faster with identical AUROC counts; this excludes IO
and is not a whole-library timing. The input-root lock prevented a new scoped
end-to-end run. See the [optimization record](../research/notes/2026-10-26-removal-optimization.md).

## Continuous worker scheduling — 2026-10-26

Use a rolling bounded set of target futures, replenishing available slots as
soon as jobs finish. No worker waits for all targets in a batch. At most the
admitted worker count of expanded task descriptors/futures is outstanding;
create C/B row lists lazily at submission. Report each durable target completion.
Collect results by original target index and assemble reports in sorted target
order so completion order does not alter artifact bytes or library identity.
Propagate worker failures, cancel pending submissions on exit, and retain existing
private checkpoint recovery and output publication behavior. This execution-only
change does not invalidate scientific checkpoints or change the algorithm ID.

Validation: all 17 library tests passed, including bounded rolling-refill and
failure tests plus spawned-process serial/parallel artifact consistency. Ruff
lint/format, documentation link/fence checks, and whitespace checks passed.

## Disposable-cache lifecycle — 2026-10-27

After all six fits complete, durably publish the private `target.json` before
removing Gaussian kernel files, prefix running-array files, and their receipts.
Retain `target.json`, all six `complete.json` receipts, profile blocks, and compact
selection receipts. Verify complete-result content hashes, target/library/fit
bindings and trace identities before deleting anything. Keep partial clusters'
files for within-fit resume; if a complete receipt is damaged later, regenerate
only the missing disposable caches using retained profiles as necessary.
Cleanup is idempotent and uses the existing per-target writer lock; a maintenance
API additionally holds the output-root transaction lock. Refuse symlink members,
unknown directory identities, incomplete targets, and inconsistent receipts.
A resumed worker repeats cleanup even if its complete fit receipts were reused.
This lifecycle-only change preserves the scientific library identity. With eight
workers, retain at most eight active clusters' six-fit arrays (roughly 50–60 GiB
for this reference), plus accumulated compact results and reports, rather than
all clusters' arrays. Never remove setup-data embeddings or model assets.

Cleanup validation: full suite **102 passed, 2 opt-in tests skipped**; strengthened
post-cleanup recovery test passed separately. Automatic cleanup, partial-cluster
preservation, corrupted receipt rejection, symlink/lock protection, and deletion
retry are covered. Reclaimed 564.917 GiB from the user's completed target caches;
see [cleanup counters](../research/notes/2026-10-27-library-cache-cleanup.json).

## Planned extension: final C/B score distributions

[Draft SPEC-0005](0005-relative-score-distributions.md) proposes an additional
post-signature stage within `build-library` that preserves sorted final C/B
composite log-score distributions and exposes empirical query-relative evidence.
It keeps this specification's discovery method and five rotations unchanged,
allows scoring-only backfill, and preserves the disposable-cache cleanup policy.
It is not implemented. It adds neither probability calibration nor automatic
cluster assignment; [reasoning and interpretation](../research/notes/2026-10-27-relative-score-reasoning.md)
are recorded separately.
