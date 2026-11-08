# SPEC-0022: Explore similarity-informed additive classification

- Date: 2026-11-07.
- Status: **Implemented and validated on CPU — experimental; full-data evaluation pending.**
- Entry point: `scripts/explore-ml.py`; equivalent planned command
  `pclass-dark explore-ml` and importable `pclass_dark.explore_ml` operations.
- Inputs: `features.tsv`, `cluster_dmatrix.tsv`, and optional sequence/group identity
  companions. Parent: [SPEC-0021](0021-extract-structural-features.md).
- Algorithm: **Similarity-Informed Additive Classifier (SIAC)**, an additive
  spline classifier trained with similarity-informed soft targets, using 18
  descriptors after excluding `sequence_charge_decoration`.

## Purpose and assessment

Determine whether whole-protein biophysical descriptors predict existing functional
clusters, whether similarity-informed training improves biological closeness of
predictions, and how results vary across held-out splits. This is direct
classification over represented clusters, not candidate reranking, a change to
`infer`, or discovery of new functional classes.

Five-fold CV is a reasonable first assessment. Overall exact hit rate and mean
similarity credit answer complementary questions. Neither establishes usefulness
on remote homologs unless split groups prevent homology leakage. Large clusters
can dominate overall scores; report macro and size-stratified metrics too.
Five different test folds measure score variability, not repeated prediction
agreement on the same protein. Repeated CV provides the latter.

The defaults and contracts below are fixed for version 1, not scientifically
validated optima. Implementers must not silently substitute different choices.
A change selected after inspecting outer-fold scores is exploratory and needs a
new independent assessment. No current biological performance or stability claim is made.
Evaluation is conditional on the supplied fixed cluster system, medoids and
feature predictor. It does not independently validate cluster construction or
pretrained structural heads; deriving those artifacts from the full reference
must be disclosed rather than claimed as an end-to-end held-out pipeline.

## Provenance and inspected data

Local inspection of `work/features.tsv` found 381,627 membership rows, 380,578
accessions and 3,443 target IDs. Cluster sizes range from 10 to 14,323, median 27.
All remaining 18 features are populated in that snapshot; only excluded SCD has
NA. `work/cluster_dmatrix.tsv` contains 6,207,526 unordered pairs across 3,524
clusters: 81 IDs have no feature examples. Counts are observations, never constants.

The actual feature table has 1,039 accessions with multiple target IDs. Examples:
`A0A087WNH4` has `cluster_0118` and `cluster_0635`; `A0A0F6B8D8` has
`cluster_0035` and `cluster_0630`. Each example has identical repeated feature
vectors. This contradicts the intended single-label contract and cannot be
interpreted as different embedding windows. The default exploration prefilter
removes whole ambiguous identities and reports all exclusions. Explicit strict
mode fails with the conflict audit. Do not select the first label or treat these
memberships as independent observations. This population policy is defined below. This specification does not assert that multiple biological
memberships are wrong; a multilabel experiment would need a separate target/metric
contract.

Similarity provenance: inspected `../profunc/src/profunc/cluster_listing.py`,
`iter_medoid_distances` and `write_cluster_dmatrix_tsv`, plus
`../profunc/src/profunc/go/similarity.py`. Scores compare medoid GO-MF profiles
using symmetric average-of-directional-best-match Lin similarity (AMB).
`Distance = 1 - Similarity`. They are functional annotation similarities, not
feature-space distances, sequence identity, or guaranteed metric distances.
No neighboring checkout is a runtime dependency. A TSV alone does not identify
its generating GO release; accept and record a provenance sidecar when available,
and otherwise mark generation provenance unverified. Matching IDs alone does not
prove matching annotation snapshots.

## Input contracts and preflight

1. Main TSV: exact SPEC-0021 header, valid nonempty identifiers, finite numeric
   values or literal `NA`; read source numbers without further rounding.
   Keep exactly the following ordered inputs; accession/target are not predictors:

   ```text
   ln_length acid_fraction base_fraction
   surface_acid_over_all buried_acid_over_all
   surface_base_over_all buried_base_over_all
   q3_h_fraction q3_e_fraction q3_c_fraction
   net_charge_balance charged_fraction histidine_fraction surface_fraction
   surface_acid_over_surface surface_base_over_surface
   surface_charged_over_surface mean_kyte_doolittle_hydropathy
   ```

2. If SPEC-0021 companions are present, honor the extraction lock and verify the
   entire checksum-bound family; never silently bypass an invalid manifest.
   Otherwise hash standalone inputs and mark unverified extraction provenance.
   Recheck hashes before publication; reject concurrent changes.
3. Require one feature vector and one target per accession. Identical repeated
   accession/label rows collapse to one example; conflicting feature vectors fail.
   `--conflict-policy exclude` is the exploration default and removes every row
   for each conflicting identity, recording exclusions/counts before splitting.
   Explicit `error` stops preflight instead. Direct load_cohort defaults to error.
   Report this as a changed evaluation population. No automatic label repair.
4. Use a supplied/discovered `features.sequences.tsv` sequence hash to unite exact
   duplicates across accessions. A hash-shared conflicting target fails or excludes
   the entire hash group under the same policy. Hash-shared consistent labels
   collapse to one sequence example, retaining aliases in a mapping artifact.
   Without sequence hashes, accession is the evaluation unit and exact-sequence
   deduplication is unverified. Never deduplicate by rounded feature equality.
5. Similarity TSV header is exactly `cluster_A, Medoid_A, Cluster_B, Medoid_B,
   Distance, Similarity` (tab separated). Reject malformed IDs, nonfinite or
   out-of-range scores, duplicate unordered pairs, inconsistent medoids and diagonal
   rows. Require `abs(Distance + Similarity - 1) <= 1e-5`. Validate all supplied
   rows; retain only the post-policy feature-label universe. Every distinct pair
   in that universe must occur once. Missing pairs are errors, not zeros.
   Mirror pairs and assign diagonal S[c,c] = 1. Preserve five-decimal input values
   in float64 for evaluation; float32 training copies are permitted.
6. Persist one lexicographically ordered cluster-to-index map. Report all matrix-only
   clusters; do not add output classes with no examples. At least two labels and
   five distinct split groups are required. No silent row/cluster subsampling.
7. An optional homology map has `accession_id, group_id` columns, exactly one group
   per represented accession/alias, with no conflicting/missing assignments.
   If exact-sequence aliases have different groups, fail and request a corrected
   unioned map. Record grouping method, identity/coverage thresholds, tool/version,
   source sequence hashes and command in a JSON provenance sidecar. Existing
   prepared C-folds are not presumed homology-separated. No clustering tool is
   launched or downloaded implicitly.

## Five-fold evaluation and leakage control

- Default `--split-mode accession` uses seeded shuffled
  `StratifiedKFold(n_splits=5)` for random class-stratified outer and inner splits.
  Exact sequence aliases have already collapsed to one example. No homology map
  or provenance file is required. Mark results **homology uncontrolled**.
- Optional explicit `--split-mode homology` retains seeded shuffled
  `StratifiedGroupKFold(n_splits=5)` and requires a group map/provenance. Never
  split a homology group to balance folds. This is not the default exploration.
- Persist fold assignments before model fitting. Each eligible example occurs in
  exactly one outer test fold per repeat, with disjoint train/test groups. Refuse
  empty folds or fewer than two train labels. Report class and group support in
  each fold; five-fold class stratification is not always feasible with few groups.
- Maintain the global evaluation label map, but train/normalize only over labels
  observed in each training partition. Similarity targets cannot introduce unseen
  labels into the active classifier. Test examples with absent train labels remain
  in overall denominators (exact miss, actual S credit); also report their fraction
  and metrics separately. These are unsupported-class cases, not a zero-shot test.
- All imputation, knots, centering, optimization, early stopping and tuning occur
  inside outer training data. Outer test folds never select epochs or parameters.
  Fixed external label similarity is available for training and scoring; do not
  recompute GO information content or similarities from held-out annotations.
- Primary experiment uses fixed configurations, not a hyperparameter search.
  Within each outer train partition, use one deterministic inner validation split
  (target approximately 20%, via five inner folds). It is random stratified by
  default, and group-held-out only in explicitly selected homology mode.
  Select the epoch by soft-target validation cross-entropy using the exact
  improvement/patience rule below; earliest epoch wins ties. Train up to 100 epochs, patience 10, improvement tolerance 1e-5.
  Inner-training-only preprocessing/active labels apply during epoch selection;
  report inner unsupported-label coverage. Compute epoch-selection loss only on
  inner-validation examples whose labels occur in inner training; require at least
  one such example. This restriction never removes outer-test examples. Refit
  preprocessing and a fresh model
  on the entire outer train partition for the selected epoch count, then predict
  outer test examples once. Fail preflight if viable inner groups cannot be formed.
- If later searching alpha/knots/regularization, perform selection inside each
  outer training partition, or reserve a separate locked test cohort. Do not choose
  the best configuration using the five outer test scores and call it unbiased.

## Model and similarity-informed objective

Use cubic B-spline feature expansions followed by a linear multiclass score layer:

```text
z_c(x) = b_c + sum_j f_jc(x_j)
p(c | x) = softmax(z(x))_c
```

Each feature has a separately learned contribution curve for each cluster:
`f_jc` is specific to feature j and cluster c. Spline knots/basis functions are
shared across clusters within a fold; their coefficients and cluster biases are
cluster-specific and optimized jointly under multiclass softmax. Cluster similarity
changes the training targets; it does not force similar clusters to share curves.

There are no cross-feature interactions in version 1. This preserves explicit
per-feature contributions without a generic black-box MLP. Planned implementation
uses scikit-learn spline preprocessing and PyTorch optimization; an off-the-shelf
EBM is not assumed to support this custom target distribution.

For each feature, impute NA by its training median; all-missing training columns
fail. Constant columns contribute zero, recorded explicitly. Start with eight
training-quantile knots, degree 3, constant extrapolation, with
`include_bias=False` and a separate class intercept. Deduplicate coincident knot values and record the effective knot count;
if fewer than two distinct values remain, use the constant-column rule. Center
basis columns using training means so feature contributions have a defined common
reference. Retain exactly 18 source features, no missingness indicator predictors.

For true label y, among active train labels choose up to k=10 other clusters with
strictly positive S[y,c], ordered by decreasing similarity then cluster ID. Define:

```text
w_y(c) = S[y,c]^gamma / sum_{a in neighbors(y)} S[y,a]^gamma
q_y(c) = (1-alpha) * 1[c=y] + alpha * w_y(c)
L_data = mean_i -sum_c q_yi(c) * log_softmax(z(x_i))_c
```

Outside the neighbor set, w is zero, including at y. Defaults: alpha=0.10,
gamma=2.0, k=10. If no positive neighbors remain, q is exactly one-hot. Require
0 <= alpha < 1, finite gamma > 0 and k >= 1. Verify nonnegativity/unit sums.
These targets encourage related alternatives, but do not directly minimize
expected similarity error. Do not claim this surrogate guarantees metric gains.

Use AdamW, learning rate 1e-3, weight decay 1e-4 on feature coefficients only;
no decay on biases. Batch size 1024, shuffled with recorded seeds, float32 default.
These fixed starting values require empirical evaluation. Every example has equal loss weight; no
implicit class balancing/oversampling. Stream mini-batches; do not materialize
N-by-C targets. Sparse neighbor gathers can compute exactly the stated loss.

Primary decision is argmax over active logits, with cluster-ID tie breaking.
Scores trained on smoothed labels are not automatically calibrated probabilities
of exact membership. Keep expected-similarity decoding or rejection thresholds
out of the primary comparison; they would change the prediction decision rule.

Run matched controls on the same folds and seeds:

- Same additive model with alpha=0 (essential test of similarity-informed training).
- Majority-class constant prediction using outer training counts.
- Similarity-central constant prediction maximizing sum_y train_frequency(y)*S[y,c]
  among active train labels. This reveals whether weighted hits can be obtained
  merely by predicting a functionally central cluster. Ties use cluster ID.

## Metrics: exact definitions

One held-out sequence example contributes once; retain alias mappings, do not
multiply weights by number of windows, aliases or memberships. For N examples:

```text
exact_hit_rate = (1/N) * sum_i 1[pred_i == true_i]
weighted_hit_rate = (1/N) * sum_i S[true_i, pred_i]
```

Exact hit rate equals multiclass accuracy and micro-averaged recall under this
single-label, one-prediction-per-example contract. Weighted hit rate is mean
semantic similarity credit, **not** scikit-learn support-weighted recall, a sample
weighting scheme, or a calibrated probability. Exact hits get 1; distant misses
can get 0; partial matches get their similarity. Both metrics use denominator N
and must include every eligible out-of-fold example. Weighted hit rate must be
at least exact hit rate and at most 1, within numerical tolerance.

Analytical metric oracle: truths [A,B,C], predictions [A,C,B], S[B,C]=S[C,B]=0.25
with diagonal 1 gives exact=1/3 and weighted=(1+0.25+0.25)/3=0.5.

Also report macro recall, macro mean similarity, per-cluster support/hits/similarity,
train-supported versus unsupported-label results, and held-out metrics grouped by
outer-training cluster size (0, 1–19, 20–99, 100–499, >=500). Empty subgroups are
NA with zero support. Required top-5 exact hits are secondary diagnostics, never
mixed with primary top-1 rates. No acceptance threshold or hidden denominator.

For every repeat/configuration, report pooled out-of-fold metrics (primary),
each fold's values, unweighted fold mean, sample SD, min/max and support. Pooled
rates weight folds by test counts; an unweighted mean may differ. Fold SD is
descriptive variability, not a confidence interval or five independent replicates.
Report paired per-fold and pooled differences against alpha=0 and baselines.

## Stability and interpretation

Default `--repeats 1` provides five-fold score variability only. Optional
`--repeats 3` uses three recorded split/training seeds and 15 outer fits per learned
configuration. Each example then has three held-out predictions from different
splits, whose training sets overlap; these predictions are not independent.
For each example report modal-label agreement, number of distinct predictions,
and average pairwise S between predicted labels. Aggregate distributions, including
by true cluster and cluster size. High semantic agreement with changing exact labels
means functionally stable but cluster-ambiguous predictions. These measures are
undefined for one repeat; output NA rather than comparing in-sample predictions
from the other four fold models. Repeated fits share data; do not treat them as
independent observations for naive standard errors or significance tests.

Persist f_jc curves, centering references, knot support and model state. For each
outer-test example export top-1 versus runner-up logit margin and all 18 signed
feature contributions to that difference, plus bias difference. They must sum to
the actual margin. Explain logit contributions, not additive probability changes.
Do not infer biological causation or present a smooth curve as a literal decision
rule. Contributions outside training support are explicitly flagged. With strongly
correlated charge/exposure features, disclose attribution ambiguity and avoid
claiming that one interchangeable descriptor is uniquely responsible. Direct
curve-stability quantification is deferred as specified below.

## Interfaces, resources and artifacts

Equivalent script and command usage:

```bash
python scripts/explore-ml.py --features work/features.tsv \
  --similarities work/cluster_dmatrix.tsv \
  --output-dir work/explore-ml --device auto

# Explicit exploratory population/split choices for the currently observed conflicts:
pclass-dark explore-ml --features work/features.tsv \
  --similarities work/cluster_dmatrix.tsv --split-mode accession \
  --conflict-policy exclude --output-dir work/explore-ml-exploratory --device cpu
```

CLI defaults: features=`./features.tsv`, similarities=`./cluster_dmatrix.tsv`,
output=`./explore-ml`, folds fixed at 5, repeats=1, seed=42, workers=8,
device=auto, split-mode=accession, conflict-policy=exclude. Expose the documented model
parameters, `--sequence-index`, `--groups`, `--group-provenance`, `--validate-only`
and `--resume`. Validation-only publishes a preflight audit, not a success marker
for completed training. Exit 2 for invalid options, 1 for data/operational errors,
130 on interruption. No downloads, embedding inference or production model export.

Reusable operations in `src/pclass_dark/explore_ml.py` and focused helper
modules: load/validate cohort, align similarities, create folds, construct targets,
fit additive model, predict/explain, calculate metrics and run experiment. Explicit
inputs/returns and ValueError/IntegrityError/IOError/RuntimeError contracts; imports
perform no IO, model initialization, argument parsing or computation. Frozen
`ExploreMLConfig` and `ExploreMLResult` expose settings, output paths, counts and
completion identity. Both thin Typer adapters call the same library API.

Declare an optional `ml` extra (`scikit-learn>=1.6,<2`, `torch>=2.6,<3`);
exercise supported APIs in CI and record exact installed versions
and include the script in sdist. Synchronize CLI help, README and API guide at
implementation. Read only supplied data and local implementation; no runtime
imports from neighboring checkouts.

Prefer a compatible GPU for batched softmax optimization; explicit CPU works.
Auto falls back before fitting if GPU initialization is unavailable, with actual
device recorded. Mid-fit device failures abort that fold; no silent partial result.
One GPU trains folds sequentially. CPU `--workers` permits independent fold jobs,
capped by available CPUs, 5*repeats fold tasks and configured memory budget; workers=1
is serial. Bound numerical threads, queues and per-worker copies; share/memory-map
immutable arrays where practical. No nested parallel validation. Seeds and artifact
ordering are independent of scheduling; document device tolerance/determinism.

For C=3443, a float64 similarity matrix is about 95 MB; float32 is about 47 MB.
Batch-1024 float32 logits are about 14 MB, excluding gradients/workspaces. A full
N-by-C float32 matrix would be about 5.3 GB and must not be retained. Estimate
basis/model/optimizer and worker memory before fitting; fail with actionable
resource estimates rather than unbounded task creation. Stream OOF/top-k outputs.

Output directory includes `preflight.json`, `exclusions.tsv`, `aliases.tsv`,
`labels.tsv`, `folds.tsv`, `fold_metrics.tsv`, `summary.json`, `per_cluster.tsv`,
`oof_predictions.tsv`, `stability.tsv`, `explanations.tsv`, fold model/preprocessing
artifacts and `manifest.json`. Each prediction identifies repeat/config/fold,
evaluation example, true/predicted labels, hit, similarity credit, train support
and top-5 scores; scores are explicitly model scores, not claimed calibration.

Manifest records source hashes/provenance, inclusion policy and counts, grouping,
all parameters/seeds/split assignments, dependency versions, source hashes/Git
revision+dirty status, device/hardware, threads/workers, timings, peak memory when
available, artifact hashes and execution status. Version schemas. Lock the output;
refuse unrelated/nonempty destinations. Stage fold checkpoints and publish atomically,
write complete marker last. Resume reuses only checksum-verified identical-contract
fold artifacts and never combines stale folds/metrics. Failures remain incomplete
with an explicit audit trail, never a successful summary.

## Frozen engineering contracts

### Cohort construction and identity

Version all JSON with `schema` and `status`; schema names below use the prefix
`pclass-dark.explore-ml` and suffix `.v1`. JSON permits finite numbers, strings,
booleans, arrays, objects and null; never NaN/Infinity. TSV is UTF-8, tab-delimited,
LF-terminated, with one exact header. Missing values are `NA`; booleans are 0/1.
Write floating report values with `.17g`; input three-decimal values are unchanged.
Report precision does not imply physical measurement accuracy. All rows have
schema-defined ordering; JSON keys are canonicalized with existing artifact helpers.

Deterministic preflight sequence:

1. Resolve paths, reject symlink output roots, input/output overlap and output
   directories inside an extraction/prepared/model bundle. Compute source SHA-256
   and sizes. When a sibling extraction manifest or QC companion exists, require
   the whole SPEC-0021 family. A lone explicitly supplied sequence index is allowed
   for standalone data. Acquire the source family lock nonblocking while verifying
   and copying its files into an owned immutable input snapshot; release it before
   training. For standalone inputs, copy with source hashes checked before/after.
2. Parse the main TSV in source order. Identifiers follow SPEC-0021 identifier
   validation. Check width/header/finite numbers even in excluded SCD; `NA` there
   is allowed. Source-domain bounds follow SPEC-0021, allowing 0.001 absolute
   tolerance for rounded identities; do not enforce exact fraction-sum identities
   on rounded values. Retain all source rows in audit counts. Identical repeated
   rows collapse; feature disagreement for one accession is fatal regardless of
   conflict policy. Compare parsed values with exact equality and NA equivalence.
3. Read optional index with exact header `accession_id, sequence_id, length,
   membership_count`. Sequence IDs are lowercase 64-digit SHA-256; length and
   membership count are positive integers. Require exactly one record per distinct
   source accession, matching unique source label count, and no missing/extra IDs.
   Aliases sharing a hash must have equal length and all 19 source values; otherwise
   fail. Sequence hashes cannot be validated against absent FASTA: disclose this.
4. Build identity groups by sequence hash if available, otherwise accession. Detect
   label conflicts on the entire identity group **before exclusions**. This avoids
   leaving a nominally unambiguous alias of an excluded conflicting sequence.
   Under `exclude`, remove every alias and row of that identity group. Error mode
   writes the complete conflict audit and returns exit 1; it never starts training.
5. `example_id` is `sha256:<hash>` with index, else `accession:<id>`. Choose the
   lexicographically smallest accession as representative; sort examples by
   `example_id`. Retain one vector/target and one evaluation weight per example.
   Record every alias, collapsed row, and excluded identity group. Labels are
   sorted after exclusion; fail on fewer than two. Group maps may contain unused
   source accessions; audit extras, reject duplicates even if identical. Require
   complete maps for retained aliases only; their group IDs must agree per example.
6. Stream similarity input twice from its immutable snapshot. Pass 1 validates
   scalar fields and stable medoid mapping and collects all IDs. Pass 2 detects
   duplicate unordered pairs with a packed triangular bitset over **all** matrix
   IDs, including extra labels; admit bitset/matrix memory against the budget before
   allocation. Allocate only the aligned retained-label float64
   matrix. Reject self-pairs in either pass. Assert complete retained-label pair
   coverage, zero NaNs, symmetry and unit diagonal. Extra-extra pairs need not be
   complete. Reject missing retained medoid IDs. No all-row Python dictionary/set
   of millions of pair tuples. Prohibit reparsing with a different label order.
7. Validate all outer and inner split plans and training-column viability before
   the first optimization job. The audit includes error counts and representative
   diagnostics; full conflict/exclusion details are in TSV. No random sample is
   substituted for a failed full-data preflight.

Source metadata sidecars are optional explicit `--similarity-provenance` and
`--group-provenance` JSON files. Preserve their bytes/hash; require a top-level
object. For homology mode, require the group-provenance object to contain nonempty
`method`, `tool_version`, `command`, `source_sequences_sha256` and `parameters`
(object recording identity/coverage definitions, or explicitly describing a
non-threshold grouping method). This records the declared method without claiming
to revalidate homology or the GO graph. Accessions mode rejects `--groups` and
`--group-provenance` rather than ignoring supplied grouping information.

### Exact split, seed and optimization rules

IDs r and f are zero-based repeat/fold indexes. Base seed must be an integer in
[0, 2^31-1]. Derive seeds with SHA-256 of UTF-8
`"siac-v1:<base_seed>:<r>:<f>:<purpose>"`, taking the first four digest bytes as
unsigned big-endian. For an outer split, f=0 and purpose=`outer`; inner splits use
purpose=`inner`. Training uses `select` and `refit`. Record every derived seed;
never use Python's randomized hash. Both learned configurations share seed values.
Seed Python, NumPy and Torch explicitly. Set Torch deterministic algorithms on,
cuDNN benchmark off and TF32 off; set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before
CUDA initialization when using CUDA builds. Probe deterministic forward/backward
and AdamW on the selected backend before admission; auto may fall back to CPU
there. Record backend settings. Unsupported deterministic kernels during actual
fitting fail the fold, never silently switch algorithms/devices. Exact byte parity
across hardware is not promised. Sort data/labels before applying splitters.
Use `shuffle=True` and `random_state=derived_seed` for both outer/inner
splitters: StratifiedKFold in accession mode, StratifiedGroupKFold in homology mode.

Outer splits are persisted first. For each outer train partition, create the five
inner splits using its inner seed; choose the lowest inner fold index satisfying
nonempty train/validation, at least two inner train labels, at least one supported
validation example and non-all-missing retained training columns. Audit rejected
inner candidates and fail if none qualifies. Preflight distinguishes fewer than
five inner groups from unsuccessful stratification; it does not silently change
fold count. Every outer train partition also requires non-all-missing columns.
This choice uses labels/group support, never feature-target fit or test outcomes.

For each feature, fit medians and eight quantiles at `linspace(0,1,n_knots)` using
NumPy `method="linear"` on imputed **training** values; take sorted unique knots.
Fit a separate SplineTransformer using those explicit knots, degree 3,
`extrapolation="constant"`, `include_bias=False`, `order="C"`. Constant columns
have zero basis width and exactly zero contributions. Store fitted BSpline knot
vectors/coefficients, degree, feature offsets, imputation medians, training min/max
and basis means as plain arrays; do not pickle fitted estimators. Evaluate restored
bases using the same mathematical spline convention and test reload parity.
Basis calculations/centering are float64, then cast to contiguous float32.
No standardization, clipping, quantile transformation or feature selection beyond
this contract. Flag a feature outside original finite training min/max; missing
imputed inputs have their own flag, not an extrapolation flag.

Initialize all feature coefficients and biases to zero. Use full batches plus the
last partial batch, without dropping examples. One CPU-seeded `torch.Generator`
produces an epoch-specific permutation via `randperm`; use the same sample order
for paired configurations. AdamW: betas=(0.9,0.999), eps=1e-8, amsgrad=False,
learning rate and decay as specified above, `foreach=False`, `fused=False`;
no scheduler, mixed precision, gradient
clipping or gradient accumulation. Nonfinite loss, parameters or gradients fail
that fold. Validation averages loss across supported examples, weighting batches
by their counts; it excludes AdamW decay. Evaluate after each full epoch.

Track best validation loss and epoch: accept a new best only when
`new_loss < best_loss - min_delta` (default 1e-5). Otherwise increment a patience
counter; stop at 10 consecutive non-improvements or max_epochs. The first epoch
is always a candidate. Refit for the selected best epoch count, not the stopping
epoch. Select separately for SIAC and alpha=0; report both selected counts.
Recreate targets and preprocessing using refit train labels only.

Compute neighbor weights in float64, stably normalizing positive log weights
`gamma*(log(S)-max_neighbor_log_S)` before log-sum-exp. The maximum
log weight is exactly zero; negative overflow may map to zero weight without
losing the maximum-weight neighbor. Cast the normalized sparse weights to float32.
This prevents underflow for allowed gamma values. Neighbor ordering uses original
float64 similarities; padding has zero weight and never contributes loss. Verify
row sums at absolute tolerance 1e-6. The alpha=0 control uses ordinary index-target
cross-entropy; test numerical equivalence with dense one-hot cross-entropy.

After fitting, subtract the across-active-class mean from each coefficient column
and from class biases. This fixes the otherwise arbitrary shared softmax score
component without changing probabilities or score differences. Persist these
canonical parameters; feature centering alone does not resolve this ambiguity.
All exported curves refer to this gauge. Inactive global labels have no parameter
rows and may never be predicted; active-to-global mappings are always explicit.

### Metrics, scores and stability schemas

Exactly four configuration IDs: `siac`, `hard`, `majority`, `semantic_constant`.
`siac` uses configured alpha; `hard` always uses zero. Reject alpha=0 for the SIAC
run command (the target helper still supports it) to avoid duplicate comparisons.
All four configurations use identical eligible examples/folds and top-1 scoring.
The two constant baselines have no selection/refit phase. Required learned-model
top-5 selects min(5, active class count) labels by decreasing logit, then label ID.
Export both logits and softmax scores as separate fields. Baseline probabilities,
logits, top-5 and margins are NA; baseline top-5 metrics are NA, never fabricated.

Schemas below specify exact TSV column order; comma-separated column names here
mean tab-separated header fields in files. All files also have version metadata
in the owning manifest. `feature_values` is not an opaque JSON TSV field.

| File | Exact columns / representation |
| --- | --- |
| `exclusions.tsv` | `example_id,accession_id,cluster_id,reason,source_row_count`; one row per excluded accession/label, sorted example/accession/label; reason=`conflicting_labels` |
| `conflicts.tsv` | Same columns as exclusions, covering all detected conflicts even in error mode |
| `aliases.tsv` | `example_id,accession_id,representative_accession,sequence_id,cluster_id,source_row_count`; retained aliases, sequence_id=NA if unavailable |
| `labels.tsv` | `label_index,cluster_id,example_count,medoid_accession`; global map |
| `folds.tsv` | `repeat_id,outer_fold_id,example_id,group_id`; one held-out assignment per example/repeat |
| `inner_folds.tsv` | `repeat_id,outer_fold_id,example_id,role`; role=`train` or `validation` within selected inner split |
| `fold_metrics.tsv` | `repeat_id,config_id,outer_fold_id,stratum,stratum_value,n,n_exact,similarity_sum,exact_hit_rate,weighted_hit_rate,macro_recall,macro_similarity,top5_hit_rate`; stratum=`all`, `train_support` or `train_size` |
| `per_cluster.tsv` | `repeat_id,config_id,scope,outer_fold_id,cluster_id,n,n_exact,similarity_sum,recall,mean_similarity`; scope=`fold` or `pooled`, fold_id=NA for pooled |
| `oof_predictions.tsv` | `repeat_id,config_id,outer_fold_id,example_id,true_cluster_id,pred_cluster_id,train_class_count,exact_hit,similarity_credit,top5_hit,runner_up_cluster_id,logit_margin`; learned-model top5_hit is 0/1 |
| `oof_topk.tsv` | `repeat_id,config_id,outer_fold_id,example_id,rank,cluster_id,logit,softmax_score`; learned configs only, rank starts at 1 |
| `stability.tsv` | `config_id,example_id,n_repeats,modal_cluster_id,modal_agreement,n_distinct,semantic_agreement`; one row per config/example |
| `explanations.tsv` | `repeat_id,config_id,outer_fold_id,example_id,pred_cluster_id,runner_up_cluster_id,logit_margin,bias_difference`, followed by the 18 feature names in input order, then `imputed_features,outside_support_features`; lists are semicolon-separated names or `none` |

`train_class_count` means count of the true label in that outer training partition.
Sort prediction/explanation/top-k rows by repeat, configuration order listed above,
fold, example, then rank. Sort folds and other keyed tables by their declared key
order. Explain only learned configurations; all examples still receive metrics.
For fold macro metrics average over true labels present in that test stratum only;
pooled macro averages over labels with at least one eligible example. Per-cluster
fold rows have n=0/NA rates when the global label is absent from that test fold;
these NA rates are excluded from that fold macro average. Stratum values are
`all`, `supported`/`unsupported`, and `0`/`1-19`/`20-99`/`100-499`/`500+`.

Top-5 exact credit is 0 for unsupported true labels. Store integer numerators and
float64 similarity sums before computing rates. Metric reconciliation compares rates/individual credits at 1e-12 absolute
tolerance in float64; accumulated similarity sums allow 1e-8 absolute tolerance
for different summation orders over large cohorts. `summary.json` contains all
pooled metrics per repeat/config and fold descriptive summaries (sample SD,
ddof=1). Also report mean and sample SD across repeat pooled metrics; for one
repeat that SD is null. Paired differences are calculated against each other
configuration within repeat and fold; no significance tests or automatic winner.
Baseline top5 fields remain null. A complete run always has all four configurations.

For R>=2 predictions per example, modal agreement=max label count/R; lexicographic
label breaks modal ties. Semantic agreement is
`sum_{r<s} S[pred_r,pred_s] / choose(R,2)`. Identical predictions contribute 1.
For R=1 set modal_cluster_id to the prediction, n_distinct=1, but both agreement
fields to NA. Summarize agreement mean/median/p10/p90 and fraction exactly 1 by
configuration, true label and true-label full-cohort size bins. Stability bins are
explicitly descriptive, not used for training/splitting. No curve-stability claim
in version 1: margin contributions describe different competitor pairs across
folds; direct cross-fold attribution stability requires a future fixed comparator
and common evaluation-grid protocol.

Numerical reconstruction acceptance: float32 prediction/explanation margins
within `atol=1e-4, rtol=1e-4`; CPU reload probabilities within 1e-6. Class ordering
uses raw computed scores with deterministic tie breaking, not a tolerance-based
near-tie rule. Do not require identical winners across devices when margins are
within numerical tolerance; report disagreement counts and margins explicitly.

### API, command options and resource admission

Source files: `explore_ml.py` (orchestration/API), `ml_progress.py` (logging), `ml_data.py`
(cohort/similarity/contracts), `ml_splits.py`, `siac.py` (basis/model/training),
`ml_reports.py` and `explore_ml_cli.py`. Keep Torch/sklearn imports inside operation
paths so base-package imports and `--help` work without the `ml` extra. Missing
optional dependencies produce one actionable installation error before training.

```python
# Python API; PathLike inputs are normalized to absolute Path values.
run_exploration(config: ExploreMLConfig) -> ExploreMLResult
validate_exploration(config: ExploreMLConfig) -> PreflightResult
load_cohort(features, *, sequence_index=None, conflict_policy="error") -> MLCohort
load_similarity(path, *, labels) -> SimilarityData
make_split_plan(cohort, *, groups, split_mode, repeats, seed) -> SplitPlan
make_soft_targets(similarity, *, active_labels, alpha, neighbors, gamma) -> SparseTargets
fit_siac(X, y, *, preprocess, targets, parameters, seed, device,
         validation=None, epochs=None) -> tuple[SIACModel, tuple]
predict_siac(model, X, *, batch_size) -> Iterator[PredictionBatch]
score_predictions(truth, predicted, similarity, *, labels=None, top5=None) -> dict
```

Freeze dataclasses; array-bearing results expose read-only NumPy arrays.
`MLCohort`: ordered example IDs/representatives/aliases, float64 X[N,18] (NaN
only for missing values), integer global y[N], ordered feature names and counts.
`SimilarityData`: ordered labels, float64 S[C,C], medoids and source inventory.
`SplitPlan`: outer assignments, selected inner assignments, seeds/support audit.
`SparseTargets`: active global IDs, neighbor-local indices/weights, effective alpha
per label and fingerprint. `SIACModel`: preprocessing/basis arrays, canonical
float32 W[C_active,B], bias[C_active], active labels and selected epoch count.
`ExploreMLResult`: absolute output/manifest/summary/OOF paths, experiment ID,
example/label/repeat counts, status=`complete`, reused fold count, actual devices.
`PreflightResult`: preflight/conflict paths, status=`ready` or `invalid`, error
counts, example/label counts and estimated resources. Invalid validation returns
the audit result to Python; `run_exploration` raises IntegrityError referring to
it. CLI validation exits 1 on invalid, 0 on ready. Input IO failures remain OSError;
invalid config is ValueError; fitting/worker failure is RuntimeError.

`ExploreMLConfig` contains only the options below plus their documented defaults;
no magic environment overrides. Scientific config excludes `resume`, `validate_only`,
output path, workers and memory/thread admission settings, but includes actual
selected device/runtime versions, all model parameters and exact split fingerprint.

| Option | Default and valid values |
| --- | --- |
| `--features`, `--similarities`, `--output-dir` | Paths given above |
| `--sequence-index` | Auto-discover `<features.stem>.sequences.tsv`; explicit path overrides; no silent fallback from invalid explicit file |
| `--groups`, `--group-provenance`, `--similarity-provenance` | None; homology mode requires the first two |
| `--split-mode`, `--conflict-policy` | `accession` (default) / `homology`; `exclude` (default) / `error` |
| `--repeats`, `--seed` | 1, integer >=1; 42, range defined above |
| `--alpha`, `--neighbors`, `--similarity-power` | 0.10, finite 0<alpha<1; 10, integer >=1; 2.0, finite >0 |
| `--knots`, `--max-epochs`, `--patience`, `--min-delta` | 8 (>=2); 100 (>=1); 10 (>=1); 1e-5 (finite >=0) |
| `--learning-rate`, `--weight-decay`, `--batch-size` | 1e-3 (finite >0); 1e-4 (finite >=0); 1024 (>=1) |
| `--device` | `auto`, `cpu`, `cuda`, `cuda:<nonnegative index>`; GPU API includes compatible ROCm builds |
| `--workers`, `--threads`, `--memory-budget-gib` | 8 (1..8); 1 (>=1); 8 (finite >0) |
| `--validate-only`, `--resume` | False, boolean flags; cannot be combined |

Fold count, spline degree, optimizer algorithm and dtype are fixed in version 1;
unknown options fail. Explicit GPU initialization failure errors; auto probes and
records a CPU fallback before training. Device resolution applies to preflight too.
A changed actual device/runtime is a different experiment and cannot resume old
folds. Limit worker threads to available CPUs; reject threads greater than
available CPUs, and reduce workers to floor(available_cpus/threads) rather than
rejecting the default on small machines. One GPU admits one worker and records an
effective worker count of 1 even if more requested. CPU worker processes use spawn.
Ready queue holds at most effective_workers tasks; no nested fits. Each task owns
one repeat/fold and completes all four configurations; inner selection and refit
are sequential. Worker failure cancels remaining submissions, joins workers and
preserves previously committed fold artifacts.

Memory admission uses C labels, N examples, maximum basis width B, batch size M
and effective worker count J. Budget includes immutable X/S storage, worker baseline
RSS conservatively estimated from coordinator high-water RSS (including parsed
inputs and imported dependencies, at least 256 MiB), each fit's training/basis arrays, W/gradients/
Adam moments, and at least four M*C float32 work buffers. Add a 50% workspace
reserve to the variable-array estimate; record formula/estimates. Bound expanded
bases to the current fit or memory-map them, never copy the full dataset per queued
task. CPU J is reduced until estimated peak <= budget; fail if J=1 does not fit.
Report requested/effective J and threads. Available GPU memory is checked separately
with the same variable-array reserve; fail an explicit GPU request if it cannot
fit. Auto may choose CPU at admission, never mid-fit. Estimates are admission
bounds on planned arrays, not a guarantee against OS/driver OOM. Catch OOM, mark
fold failure and require a new compatible configuration or retry.

### Artifact state machine, checkpoints and resume

Use sibling lock `<output-dir>.lock`, nonblocking. An existing directory is usable
only if empty or owned by `pclass-dark.explore-ml.run.v1`; refuse foreign files and
unsafe nested/symlink paths. No `--force` deletion in version 1. Paths and names in
manifests are relative and validated with artifact helpers. No pickle/joblib/Torch
object deserialization. Model arrays are NPZ loaded with `allow_pickle=False` plus
schema-validated JSON (validate exact keys, dtype, shape, finite values and label
mapping against the model schema). These are fold research artifacts, not
deployment exports.

Owned layout: root audit/aggregate files listed above; `inputs/` (snapshotted TSVs
and sidecars), `arrays/` (read-only mmap data), `folds/rNNN/fNNN/` (committed tasks),
`.stage/`, `run.json` and `audits/` (prior invalid preflight audits only). Exact
allowed names derive from this schema; unknown files cause refusal. Each stage
starts with an atomically written `owner.json` naming the run ID,
repeat/fold and stage purpose before any model/report file is written. A stage
without a valid owner marker, or containing unknown files, is refused rather than
silently removed; this includes abrupt termination during owner-marker creation. Each task stages both learned configurations' `model.npz`,
`model.json`, `training.tsv` (phase, epoch, n, mean_loss, validation_loss, best_epoch)
under `siac/` and `hard/`, with task-level prediction/report files alongside them,
OOF/top-k/explanations for all four configurations as applicable, and metrics.
`task.json` lists every file's hash, size, row count and contract fingerprint.
Flush/fsync files; validate reconciliation; rename the stage directory to its final
unused task path atomically; sync parent. A committed task is the resume unit.
No optimizer/half-epoch resume: interrupted tasks restart from their fixed seeds.

`run.json` states are `validating`, `invalid`, `ready`, `running`, `failed`,
`complete`; record timestamps and last error. Contract fingerprint hashes source
snapshot inventory, eligibility policies, provenance bytes, exact source-code
hashes (ML modules and invoked shared scientific helpers), dependency versions,
model settings, actual device/determinism settings and split plan. Do not include
elapsed times or mutable status. SHA-256 of this contract is `experiment_id`.
Before split construction finishes the experiment ID is null, not a fake complete
identity. `preflight.json` has status `ready` or `invalid`, counts, diagnostics,
source inventory, declared provenance verification and support/resource reports.

A validation-only ready run may later start training **only with `--resume`** and
matching input/config. An invalid run may be revalidated with `--resume` after a
policy/input change only before any committed task exists: archive its prior audit
under an owned `audits/` directory together with its old input inventory and split
plan, replace owned input snapshots/arrays atomically, and construct a fresh
contract. A running/failed
run with committed tasks requires exact contract match; stale/missing/corrupt
committed tasks raise IntegrityError, never silently retrain under the same marker.
Orphaned `.stage/` content is disposable only when ownership/manifest proves it is
an uncommitted task for this run. Never delete arbitrary stage contents.

After all expected 5*R task manifests verify, rebuild aggregates from them in
`.stage/aggregate/`, reconcile exact OOF coverage per configuration/repeat, recheck
source inputs and write staged final manifest. Publish aggregate files by atomic
per-file replace, manifest last, then set run state complete. During partial
publication the root has no valid complete marker; `--resume` regenerates aggregates
from immutable tasks. A preexisting invalid final manifest is corruption, not an
interrupted first publication. A complete manifest verifies all aggregate/task
files and records input hashes; `--resume` returns verified results without fitting.
No public reader may treat `run.json` alone as completion. Moving a complete folder
is permitted because all artifact references are relative; input paths remain
provenance, not required to read already completed metrics. Reuse via the run API
still requires supplied original inputs/config to match the completed contract.


Model persistence is schema `pclass-dark.explore-ml.siac-model.v1`. `model.json`
contains `schema`, `feature_names`, `active_global_indices`, `global_labels`,
`basis_offsets` (19 integers), `basis_width`, `class_count`, `degree`, `include_bias`,
`extrapolation`, `constant_features`, `selected_epochs`, `fit_seed`, `parameters`,
`gauge` (`class_mean_zero`), `dtype` (`float32`), `training_example_count`, and
`arrays` (key/dtype/shape inventory). NPZ keys are `weight`, `bias`, `medians`,
`training_min`, `training_max`, `basis_mean`, plus `fJJ_t` and `fJJ_c` for each
nonconstant feature's fitted SciPy BSpline representation (JJ is 00..17). Weight
shape is [C_active,B], bias [C_active], medians/min/max [18], mean [B]. Store
preprocessing arrays as float64. For each nonconstant feature, persist its
`bsplines_[0].t` and `.c` exactly; reconstruct `BSpline(t,c,degree)` and omit the
final spline column to reproduce include_bias=False. Clip evaluation x to original
training min/max before BSpline evaluation for constant extrapolation, then subtract
stored means. Constant features require no fJJ keys. Require exact array key inventory,
monotonic offsets/knots, valid class maps and absolute class-mean gauge tolerance 1e-6 before accepting
artifacts. Reconciliation covers logits, probabilities and feature margins from
this plain-array reconstruction, including both out-of-range endpoints.

`summary.json` schema is `pclass-dark.explore-ml.summary.v1`, with required fields
`schema`, `status` (`complete` only), `experiment_id`, `population`, `split_mode`,
`homology_controlled`, `provenance_status`, `configurations`, `repeat_metrics`,
`fold_descriptive`, `repeat_descriptive`, `paired_differences`, `stability_summary`,
`limitations` and `artifacts`. Lists of metric objects use explicit keys
`repeat_id`, `config_id`, `scope`, `stratum`, `stratum_value`, `n`, `n_exact`,
`similarity_sum` and the metric field names above; absent repeat/fold keys are null,
never overloaded sentinel integers. `manifest.json` schema
`pclass-dark.explore-ml.complete.v1` has `schema`, `status`, `experiment_id`,
`contract`, `source_inventory`, `task_inventory`, `files`, `execution`,
`created_at`, `completed_at`; each inventory entry records relative path, sha256,
size_bytes and TSV rows/columns when applicable. Manifest creation time is not
part of the scientific identity.

## Automated validation and implementation acceptance

- Analytical target oracle: A has similarities B=0.8, C=0.4; k=2, gamma=2,
  alpha=0.1 gives q[A]=0.9, q[B]=0.08, q[C]=0.02. Cover no-positive-neighbor
  fallback, alpha=0, ties, active-label restriction, finite loss/gradient and
  numerically stable extreme logits. Compare sparse and dense loss independently.
- Independently test metric oracle above, exact/micro equivalence, semantic-credit
  bounds, macro/pooled/fold distinctions, absent-train labels and empty subgroups.
- Validate headers, full pair coverage, reversal duplicates, diagonal rejection,
  complement tolerance, 81-style extra-label filtering, manifest corruption,
  conflicts/exclusion, alias unions, missing groups, invalid values and all-NA columns.
- Assert excluded SCD/accession/target never enter model inputs. Verify training-only
  imputation/knots/centering using extreme held-out examples; constant/tied features,
  additive margin reconstruction and score reload parity. No test-fold epoch tuning.
- Five-fold coverage/group disjointness and seeded reproducibility; difficult rare
  classes must be reported rather than leaking groups or silently dropping examples.
  Repeat agreement/semantic stability have independent small known examples.
- Small synthetic known-additive problem must learn its held-out rule; validate
  interfaces and integration on tiny five-group-per-class fixtures. CPU required;
  serial/parallel agreement, worker errors, checkpoint corruption/interruption,
  no partial success publication, lazy imports and CLI/library equivalence.
  GPU comparison if available; explicitly report untested GPU paths otherwise.
  CPU serial/parallel softmax probabilities must agree at atol=1e-6, rtol=1e-5;
  CPU/GPU probabilities at atol=1e-4, rtol=1e-4. Separately enumerate prediction
  disagreements, including near ties; do not hide them behind aggregate tolerance.
  Repeated CPU single-thread fits with identical seeds must have equal split plans,
  epochs and predicted labels. Preserve near-tie evidence if backend variation occurs.
- Run relevant pytest suites, packaging/help checks and a bounded synthetic pilot.
  Full data training is a separate explicit experiment with recorded identities;
  do not report synthetic recovery as protein classification performance.

## Alternatives and follow-up decisions

1. Retain random stratified five-fold CV as the primary current exploration. For stronger per-protein stability use
   repeated five-fold CV. A fixed external homology-separated test is preferable
   after repeated development; group bootstrap intervals on locked OOF predictions
   quantify conditional evaluation uncertainty, not full retraining uncertainty.
2. Alpha=0 is the required matched alternative, not a new algorithm search. If
   similarity smoothing harms exact hits, report the tradeoff; tune only internally.
   Expected-similarity decoding and label-embedding regression are later experiments.
3. If homology groups make five-fold support too sparse, use a documented three-fold
   grouped or fixed grouped holdout experiment with a separate contract; do not
   silently replace the requested five folds or relax homology separation.
4. Excluding conflicting targets enables a single-label pilot but changes scope.
   Upstream correction is preferable if memberships are an export error. If they
   are valid, specify set-valued/multilabel targets and scoring before retaining them.
5. No full-data deployment fit or claim about proteins lacking nearest neighbors
   follows automatically. Evaluate that intended deployment subset explicitly later.

## References

- [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html): grouped evaluation and split limitations.
- [scikit-learn SplineTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html): spline bases and extrapolation.
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html): soft probability targets; validate distributions explicitly.
- [Nested validation guidance](https://inria.github.io/scikit-learn-mooc/python_scripts/cross_validation_nested.html): separate selection from outer assessment.

## Implementation and validation record

Implemented the experimental script, root command, Python API and all listed
reports under the SIAC name. No production inference or source data was modified.
The feature input excludes SCD; continuous curve parameters persist as numerical
B-spline arrays, not plot files or pickled estimators.

Actual engineering refinements from the draft:

- Worker baseline memory uses conservative coordinator high-water RSS rather than
  launching an extra worker solely for a memory probe. Expanded training bases
  are cached once per fit in bounded arrays and accounted for in admission.
- `fit_siac` returns `(model, epoch_records)`; metrics are explicit dictionaries.
  They remain directly importable operations; both user interfaces are thin adapters.
- Rejected resumes stage validation independently and preserve existing input array
  inodes, committed folds and prior audit. Matching resumes never replace immutable
  inputs. Unknown files in interrupted stages are preserved and cause refusal.
- Models enforce float32 independently of the caller's Torch default dtype.
  Serialized metadata/array shape/type/gauge checks run before model reuse.
- Different large-cohort summation orders use 1e-8 tolerance for similarity sums;
  rates and individual semantic credits retain 1e-12 comparison tolerance.
- Source-only wheel/sdist packaging was checked in a temporary source copy to
  avoid copying the checkout's selected multi-GB reference bundles into test releases.
  No release was published and no selected data artifact was changed.

Validation on the current CPU environment:

- SIAC suite: **24 passed, 1 skipped** (GPU unavailable), including analytical
  target/loss/gradient and metric examples, held-out synthetic additive recovery,
  train-only splines, model reload/margin identity, five-fold/group coverage,
  repeated stability, unsupported labels, exclusion/alias handling, script/CLI/API,
  serial/parallel agreement, corrupt/incompatible resume rejection and interrupted
  aggregate publication. Added input-inode preservation, foreign-stage protection
  and Torch-default-dtype regression checks.
- Broader feature/structure/CLI selection: **164 passed, 2 skipped** before the
  final dtype regression addition; final SIAC suite reran after that refinement.
- Full repository run: **760 passed, 4 skipped, 1 fixture setup error** in
  765.87 seconds. The error was sandbox-denied loopback socket creation in the
  existing Swiss worker test; isolated rerun with loopback access: **1 passed**.
  The final SIAC regression selection ran after the last model changes (24 passed,
  1 GPU skip). Formatting, whitespace, source-only wheel/sdist content and
  installed-wheel CLI help passed.
- CPU pilot: 90 seeded synthetic examples, three artificial clusters, five folds,
  no homology claim; known rule recovered. This is a software correctness result,
  not evidence about protein classification. Reproducible commands and artifact
  identities are recorded in the progress log.

Full-reference training, GPU numerical validation, runtime/memory measurements at
380k examples, biological generalization and calibration remain unestablished.
Data prerequisites remain explicit: resolve conflicting labels or opt into audited
exclusion. Random splitting is now the default and homology uncontrolled;
optional grouped evaluation requires explicitly selecting homology mode. No automatic best-model or deployment fit occurs.


### 2026-11-07 — Random splitting becomes the default

At the user's request, homology grouping is no longer required for ordinary
exploration. Both API and CLI default to accession mode. Outer and inner folds use
seeded shuffled StratifiedKFold on deduplicated sequence/accession examples;
optional explicitly requested homology mode keeps the original grouped splitter.
Random splitting may put homologous proteins in different folds and is recorded
as homology uncontrolled. Existing conflict/exclusion policy is unchanged.
Historical experiment fingerprints remain historical; changed source/split contracts
require a fresh output directory rather than mixing cached models.


### 2026-11-07 — Eight requested workers by default

API and CLI now request eight workers (valid range 1..8). CPU admission limits
actual concurrency by available CPUs divided by threads, 5*repeats tasks and the
memory estimate. One repeat admits at most five independent fold jobs; repeated
CV may admit all eight. GPU execution still admits one worker. Record both
requested and effective counts; low-CPU machines clamp workers instead of rejecting
an otherwise valid default configuration. Serial mode remains `--workers 1`.


### 2026-11-07 — Visible progress and actionable conflict diagnostics

The script/root command emit timestamped INFO progress to stderr by default:
input snapshot/verification, feature and sequence-index scans, conflict count/audit
location, both similarity passes, split planning, admitted device/workers, task
start/completion/reuse, model/phase/epoch loss and early stopping, aggregation and
final summary location. Long scans and training batches are throttled to roughly
five seconds; epoch completion is always reported. Spawned CPU workers configure
matching stderr handlers through the executor initializer; pure Python library
imports/calls do not install logging handlers. CLI handlers are scoped to the call
and removed afterward. Logging does not change targets, predictions or artifact
schemas; persistent numerical epoch histories remain training.tsv.

Conflict errors name multiple labels per sequence identity and the explicit
`--conflict-policy exclude` option; the run error points to conflicts.tsv and notes
that retrying an existing invalid output needs --resume. No default exclusion is
introduced. Traced prepared example: Q2FH94/cluster_0868 and Q2FYU1/cluster_1372
share one sequence hash and representative Q2FH94; the feature writer deliberately
preserves both retained sequence memberships. This is not evidence of multiple
embedding-window targets or a rounding artifact.


### 2026-11-07 — Default identical-sequence prefilter and explicit removal counts

The exploration now runs an audited sequence-identity prefilter before splitting.
Run/API config and CLI default conflict_policy to exclude: retain one example for
same-target identical sequences and remove the entire identity if targets differ.
Never choose the first label or infer sequence equality from rounded features.
Use the sequence index hashes when available; accession-only inputs explicitly
report that cross-accession sequence identity is unverified. The low-level
load_cohort helper retains its strict default for direct validation callers.
Explicit --conflict-policy error preserves the previous fail-on-ambiguity behavior.

Persist prefilter_counts in population/preflight/summary: source_rows,
source_accessions, unique_identities, repeated_identity_rows (rows beyond one per
identity, including ambiguous ones), same_target_rows_collapsed,
ambiguous_identities_removed, ambiguous_source_rows_removed, total_rows_removed,
retained_sequences and identity_basis. Counters reconcile:
source_rows - retained_sequences == same_target_rows_collapsed +
ambiguous_source_rows_removed. Log these categories separately to avoid counting
membership rows as additional unique sequences or double-counting removals.
The source feature table remains unchanged; exclusions.tsv and aliases.tsv audit
removed identities and retained representatives. This user-requested policy
supersedes earlier requirements for explicit exclusion in the exploration defaults.


### 2026-11-07 — Unambiguous training stage labels and shorter epoch logs

Each training progress line identifies repeat/total, fold/5 and a numbered stage
within the fold's four fits: 1/4 SIAC epoch-selection, 2/4 SIAC outer-refit,
3/4 HARD epoch-selection, 4/4 HARD outer-refit. Model stages and phases are logged
at transitions; periodic batch counters and epoch completion carry the same prefix.
Prediction/report and checkpoint phases have distinct named messages.
Terminal epoch lines show epoch/limit and training loss only; omit validation loss
and best-epoch fields to shorten the line. Validation/epoch selection behavior and
training.tsv (including validation_loss/best_epoch) remain unchanged. Add regression
checks for all four stage labels and the absence of terminal validation_loss.


### 2026-11-07 — GPU consistency test exercised during final-model work

The previously skipped balanced two-class toy test failed its original 1e-4 logit
criterion on the available AMD GPU. Measured CPU/GPU drift: 0.0006954 logits,
0.0003477 softmax score, no label disagreements. Cancellation-scale near-zero bias
gradients are amplified by AdamW eps=1e-8. Both initial derivatives match the
analytical zero within float32 epsilon. The asymmetric 11/9 class example agrees
within 3.73e-9 logits. Tests now assess actual score/label behavior: 1e-4 tolerance
for the nondegenerate example, explicit 5e-4 bound for the balanced near-zero case,
and no changed labels on either analytic fixture. This is a documented numerical
exception, not an optimizer change or a universal cross-device guarantee. See
[SPEC-0023](0023-final-ml-model.md). Also, full-data deployment fitting is now
available under that separate specification; the original CV run is not modified.

## Completed-run device admission correction

Completed resume must apply the same requested-device probe and _resources memory
admission policy as initial preflight before comparing the actual device against
the saved contract. Use verified saved X/y/S arrays via read-only memory maps and
the recorded label count, avoiding cohort reconstruction or fitting. An auto GPU
that still lacks sufficient memory must fall back to CPU and reuse a completed
CPU run. A newly admitted GPU remains a changed actual device and is rejected;
explicit GPU memory failures retain their actionable error. Never modify completed
artifacts during these checks. Regression uses mocked GPU probing/memory with real
CPU training, verifying reuse and unchanged report bytes.

Resume correction validation: exploration suite 32 passed, 2 GPU tests skipped;
model/interpretation suites 23 passed. Mocked memory admission covered CPU reuse
and genuine device-change rejection with immutable artifacts. Ruff and diff
checks passed; actual GPU execution was not exercised.

## Sparse-target memory admission correction

Include full-label sparse targets even for hard targets (allocation still occurs):
K=min(neighbors,C-1), one table C*K*(8+4)+C*(8+4) bytes for int64
indices/active labels and float32 weights/alpha. Host admission reserves three
table copies for original arrays, transient NumPy copies and CPU tensors (also
covering overlap with prior-stage targets), plus 24*batch_size*K bytes for sparse
batch gathers/products and their gradients. GPU admission reserves one table plus
the same batch workspace. Apply existing 1.5 safety factor and per-worker scaling;
use the same pure estimator in exploration and final fitting. No target allocation
is needed for admission. Tests compare small actual target sizes, cap neighbor
width, and mock large C / memory limits to verify worker reduction, host refusal,
GPU auto fallback and explicit-GPU refusal.

Sparse-target admission validation: 58 tests passed, 2 GPU tests skipped across
memory/exploration/final-model/interpretation suites. Analytical actual-array
size checks and mocked 10,000-label admission cover worker reduction, host budget
refusal, auto CPU fallback and explicit GPU refusal without large allocations.
Ruff and git diff --check passed; real GPU execution was not exercised.

## Semantic-constant report memory correction

Compute semantic-constant scores as (train_counts @ S)[active], rather than
train_counts @ S[:,active]. The full similarity matrix remains read-only/mmap
backed; only score vectors are materialized. Preserve active-label restriction,
training-only counts, ordering and first-active-label tie breaking. Validate via
small independent weighted-sum examples and a read-only memmap view that rejects
full-column advanced indexing, including tied scores and an inactive best label.

Semantic-constant correction validation: 60 passed, 2 GPU tests skipped across
exploration, final-model, memory and interpretation suites. Read-only mmap
regression covers active-only predictions and deterministic ties without column
copying. Ruff and git diff --check passed.
