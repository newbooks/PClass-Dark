# SPEC-0023: Full-cohort additive model and query prediction

- Date: 2026-11-07.
- Status: **Implemented and validated — full reference model fitted on GPU.**
- Parent: [SPEC-0022](0022-explore-ml.md).
- Purpose: fit one reusable classifier using all eligible examples of a completed
  exploration, then predict arbitrary query features or full protein sequences.
- Interfaces: `scripts/train-ml.py`, `scripts/predict-ml.py`, equivalent
  `pclass-dark train-ml` / `predict-ml`, and `pclass_dark.ml_model` functions.

## Scientific scope and defaults

Use all N unique examples from the verified CV snapshot exactly once, including
all five outer folds. Do not concatenate repeats or average trained fold parameters:
fold preprocessing differs and duplicate training weights would be incorrect.
Refit spline preprocessing on the full cohort, then fit a fresh additive classifier.

Default configuration is `hard`, because its recorded exact and weighted CV scores
exceeded SIAC in every fold of the assessed reference run. `--config siac` remains
available. This is a choice based on already inspected CV scores, not independent
new evidence of optimality. Preserve the source CV scores as evidence; report no
new held-out accuracy or stability for this full-data model. No training-set accuracy
is substituted for CV. No automatic ensemble or production `infer` replacement.

Default epochs: nearest-integer median (floor(median+0.5)) of the selected inner
validation epoch counts in the source configuration across every repeat/fold.
Require 5*R verified fold model metadata records; for the current run all are 100.
An explicit positive `--epochs` overrides the budget and is identified in provenance.
Inherit knots, learning rate, decay, batch size, similarity alpha/k/power and seed
from the completed experiment. HARD sets alpha=0. Train on fixed epochs with no
held-out final-fit selection. Seed is derived_seed(seed,0,0,"final-<config>").

## Verified input and data contracts

- Input is a complete exploration directory; default `./explore-ml`. Validate
  complete schema/status, identity(contract)==experiment_id and all consumed file
  hashes, with safe relative paths and no symlinks. No current-checkout-code equality
  requirement: source CV source hashes remain historical, and final fitting records
  the current implementation separately. Current/model interface compatibility is
  enforced by the saved model schema and explicit feature contracts.
- Verify arrays/X.npy, y.npy, S.npy, labels.tsv, folds.tsv, summary.json and all
  source-configuration fold model JSONs against the exploration manifest. Recheck
  consumed files before publication to detect concurrent edits. Do not mutate the
  exploration, extraction reports or selected production artifacts.
- Read numerical arrays as read-only mmap with allow_pickle=False. Validate exact
  float64 X[N,18], int64 y[N], float64 S[C,C], finite-or-NaN X, finite S in [0,1],
  symmetry/diagonal, ordered unique labels and active y in 0..C-1. Require each label
  to have support; global class counts match labels.tsv; summary N/C match arrays.
- Check folds.tsv gives each example one outer assignment per repeat, example ID
  sets match across repeats, and exactly five outer fold IDs occur per repeat.
  This confirms fitting N examples, not N*repeats. Array ordering comes from the
  immutable CV snapshot; no sorting/reloading of the live feature table is allowed.
- Read the extraction manifest snapshotted by CV when available. Verify it as a
  consumed artifact; save its scientific input contract plus export precision.
  Export-only serialization metadata overrides original computation precision.
  For this reference the effective precision is .3f, despite original .17g receipts.
  Without extraction metadata, precomputed-feature prediction is supported but raw
  sequence prediction is refused (unknown scientific extraction contract).

## Model bundle, resource use and recovery

Default output `./ml-model`, outside the source experiment and its parents/inputs.
Reject source/output overlap, symlinks and nonempty unowned destinations. Sibling
nonblocking writer lock; never overwrite an existing model implicitly. `--resume`
reuses a verified matching completed bundle, or restarts an owned incomplete fit
from its seed. No partial-epoch optimizer resume. Changed inputs/config/code/runtime
require another output directory. Incomplete run marker binds source contract,
chosen settings, source inventory, feature contract and actual device.

Stage inside owned output `.fit-stage/`, with an owner marker written before
model/report files; refuse unknown staged files instead of deleting them. Publish
validated model/JSON/training history/feature contract by per-file atomic replace,
complete manifest last. Incomplete publication can restart from the owned stage;
no predictions may load an incomplete bundle. A malformed complete manifest or
changed committed file is corruption, not a reason to overwrite it.

Plain-array `model.npz` / `model.json` use the existing SIAC schema, including
label map, medians, spline knots/centering, gauge-normalized weights and biases.
Also publish feature_contract.json and training.tsv (epoch,n,mean_loss,
validation_loss,best_epoch; validation columns NA for fixed final fit).
Final manifest schema `pclass-dark.ml-model.v1`, status complete, with content ID,
contract, files{sha256,size_bytes}, source_experiment_id, source selected CV metrics,
N/C, configuration, chosen epochs/rule, actual device, seed, dependency versions,
scientific source hashes, timestamps and resource estimates. Include all metadata
needed for prediction; runtime prediction does not need the source experiment.

Reuse GPU-first deterministic fit_siac and bounded batch/basis behavior, explicit
CPU and CPU fallback; threads default 8, bounded by available CPUs. One full fit
has no independent fold workers. Memory budget default 8 GiB; estimate X/S and
training/basis copies, weights/optimizer and four batch-logit buffers with 50%
reserve. Check GPU free memory for numerical buffers, and CPU budget separately;
auto may choose CPU before fitting, never silently change device mid-fit. Training
logs identify final configuration, full cohort size, epoch/budget and training loss.
Save/reload verifies score parity on a bounded sample before publication.

## Prediction and extraction parity

`load_final_model(path)` verifies manifest identity, inventory, strict plain-array
model schema, feature order, selected config/epochs and label map before scoring.
Return an immutable wrapper with model, labels, feature contract and provenance.
Prediction itself is CPU NumPy spline evaluation, regardless of training device.
Softmax scores are uncalibrated scores, not membership confidence estimates.

- Feature queries: exact TSV header query_id plus the ordered 18 SPEC-0022 input
  columns. Unique nonempty IDs, valid finite numbers or literal NA. Preserve order.
  Reject target/extra/reordered/missing columns; never treat IDs as predictors.
  Python API accepts [N,18] arrays plus aligned unique query IDs, with NaN allowed
  and infinity rejected. Apply the recorded export precision before saved
  imputation/splines (3 decimals here). Standalone unbound feature experiments use
  identity precision; raw sequence prediction is not permitted without provenance.
- Sequence queries: `--seq` with default query ID `query`, or `--fasta`. Same
  StructuralPredictor and summarize_structural_features as extraction: full coverage
  windows, pinned ESM layer/checkpoint, frozen ALL-CEF and Q3 heads, original
  denominators/missingness, then discard SCD and apply recorded precision.
  Check model ID, ESM checkpoint policy and feature-scientific module hashes against
  the saved extraction contract before initializing prediction. No pooled embedding
  substitution. Lazy one predictor reused for a streamed FASTA; GPU-first feature
  extraction, explicit CPU option, offline default (opt out with --no-offline).
  Bound FASTA lengths to 100,000, headers to 4096 bytes, file to 256 MiB and 100,000
  queries. Invalid/duplicate IDs and invalid sequences error before partial output
  publication. Use source-stable FASTA reading; no target annotations are required.
- At most five ranked classes, decreasing logit with cluster-ID tie breaking.
  Expose logits and softmax scores separately. Return top1-versus-runner-up margin,
  bias and 18 signed feature differences, which sum to margin at SIAC tolerance.
  Include imputed/outside-support flags and rounded features in JSON output.
  Similarity credit is absent for queries with no known target.

## Interfaces and output

```bash
# From repository root, train all eligible references once using the best CV config.
python scripts/train-ml.py --experiment work/explore-ml --output-dir work/ml-model

# Query raw sequence or a FASTA; feature extractor assets must be cached offline.
python scripts/predict-ml.py --model-dir work/ml-model --seq MDEKRH
python scripts/predict-ml.py --model-dir work/ml-model --fasta query.fasta --format jsonl

# Precomputed features need only the trained numerical model.
pclass-dark predict-ml --model-dir work/ml-model --features query_features.tsv
```

Train flags: --experiment PATH, --output-dir PATH, --config hard|siac (hard),
--epochs INT (median), --device auto|cpu|cuda[:index] (auto), --threads INT (8),
--memory-budget-gib FLOAT (8), --resume (false).
Predict flags: --model-dir PATH (ml-model), exactly one --features/--seq/--fasta,
--query-id (query; only with --seq), --device (auto; feature extraction), --offline
(default true), --threads (8), --format tsv|jsonl (tsv), --output PATH (stdout).
TSV fields query_id,rank,cluster_id,logit,softmax_score. JSONL one object per query:
query_id,model_id,configuration,ranked list, features, margin,bias_difference,
feature_contributions,imputed_features,outside_support_features,feature_device.
Logs go to stderr. Stage even stdout results in a temporary file, so a late invalid
FASTA row does not publish partial success. Existing file outputs are refused;
atomic publication must not overwrite an input/model file. CLI error codes 2 for
options, 1 for data/IO/training, 130 for interrupt.

Library functions in ml_model.py: fit_final_model(...)->FinalFitResult,
load_final_model(...)->FinalMLModel, predict_features(model,X,query_ids)->iterator
of QueryPrediction, predict_sequences(model,queries,...)->iterator, and
read_query_features(path)->(IDs,X). No argument parsing, downloads, model loads or
computation at import. Optional ML/embedding dependencies stay lazy; query features
can score on CPU without initializing Torch or ESM. API ValueError/IntegrityError,
OSError and RuntimeError are explicit. Frozen dataclasses, read-only model arrays.
Script adapters and root commands call the same functions; include scripts/guide
in sdist, synchronize README and API documentation.

## Acceptance and experiment completion

- Synthetic completed-CV fixture: verify final train receives every example once,
  refits preprocessing from all examples, handles repeated-CV snapshots without
  duplicating examples, inherits settings, median/override epochs and hard alpha=0.
- Analytical logits/ranking and explanation margins; save/load parity, strict
  feature headers/shape/IDs/invalid inputs, canonical rounding of query values,
  NA imputation/out-of-support flags, no raw queries with missing/mismatched
  extraction provenance. Mock residue-head example independently checks raw-feature
  parity and uses the same 3-decimal rounding as reference export.
- Tampered source/bundle/labels, unsupported config/epochs, partial/foreign output,
  failed fit/restart, completed reuse, output atomicity and lazy imports. CLI/script
  help, API equivalence and packaging inclusion. CPU serial required; exercise GPU
  fit when local hardware is available and report device actually used.
- Run appropriate pytest suites and a bounded synthetic pilot before real fitting.
  Then perform the explicitly requested full-cohort fit, record exact source/model
  IDs, seed, settings, device, logs/artifacts and smoke-test model reload/query output.
  Distinguish these training/smoke results from the source held-out CV measurements.


## Implementation and actual validation

Implemented ml_model.py, ml_model_cli.py, both scripts/root commands and the
[model guide](../ml-model.md). Complete final bundle is `work/ml-model`, trained
on all 379,539 examples and 3,443 labels from source experiment
`4683ddea21e305608fc8e81f38c2f71276d83d5e194fe2547890acec98db8163`.
HARD configuration, 100 epochs (median of all five inner-selected counts), seed
and source settings recorded in the manifest. GPU cuda:0, AMD Radeon Graphics,
8 numerical CPU threads, 8 GiB budget, approximately 124 seconds after full-fit
validation. Model recipe ID:
`e24b23cbe6240e5d3519ff1a3efdff174b6f33e74fdb7037b88e7b5ebbdc16f9`.
Weights SHA-256:
`d267fd34a7ee6394488573f3c76945ccbf23649a2fc892b15af0d05f8bf27789`.
Model recipe identity includes scientific inputs/settings/code; output byte hashes
are separately bound through its manifest. Prediction needs only the portable
bundle and applicable optional feature-extraction assets.

Validated save/reload, whole-cohort N rather than N*repeats, default/override epoch
budgets, strict source/bundle integrity, failure/restart/publication behavior,
query ranking/margins, export rounding, offline raw-feature recipe, lazy imports,
CLI/script equivalence and atomic output. Real raw-sequence smoke query Q2FH94
(207 aa, ambiguity excluded from fitting) reproduced all 18 rounded reference
features exactly. Top predicted cluster_0868, score 0.367643; this is neither an
independent accuracy estimate nor calibrated membership confidence. Full model
training leaves the exploration and original source files unchanged.

An actually exercised CPU/GPU test exposed cancellation-scale bias-gradient
sensitivity in a balanced toy example with AdamW eps=1e-8: maximum logit difference
0.0006954, softmax-score difference 0.0003477, zero label disagreements. Initial
bias derivatives on both devices were within float32 epsilon of the independent
zero derivative. The nondegenerate (11/9 class balance) example differed by
3.73e-9 in logits and zero at reported float32 softmax precision. No optimizer or
trained-model changes were made to hide this. Regression tests now retain strict
1e-4 score tolerance for the nondegenerate case and explicitly bound the balanced
near-zero case at 5e-4, requiring identical predicted labels in both toy cases.
Per-device reproducibility does not imply universal cross-device retraining
identity. Full CPU-versus-GPU reference retraining was not performed; loading the
same saved weights is separately checked for prediction parity.

Validation outcomes and packaging checks are recorded in the dated progress log.
Source CV evidence (HARD exact 60.04%, weighted 71.15%) stays historical; the new
full-data artifact has no new held-out performance/stability measurement.

Human-readable reports and additive own-cluster explanation fields are implemented
in [SPEC-0024](0024-ml-interpretation.md); original comparative JSON fields retain
their semantics.

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
