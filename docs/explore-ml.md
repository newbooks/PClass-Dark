# SIAC machine-learning exploration

`explore-ml.py` evaluates the **Similarity-Informed Additive Classifier (SIAC)**
using five-fold cross-validation. It reads existing feature and medoid similarity
TSVs; it does not extract features or change production inference. Exactly 18
features enter the model: `sequence_charge_decoration`, accession identifiers and
target labels are excluded. The model learns a separate spline contribution curve
for every feature and cluster. Similarity informs training targets, not predictor
values or shared curve coefficients.

Install the optional dependencies:

```bash
pip install -e '.[ml]'
```

The default is seeded, shuffled, class-stratified random five-fold CV. No
homology files are required. Exact sequence aliases are deduplicated before
splitting when the sequence index is available. Homologous proteins may occur in
both training and test folds; reports mark this as **homology uncontrolled**.

```bash
python scripts/explore-ml.py \
  --features work/features.tsv --similarities work/cluster_dmatrix.tsv \
  --output-dir work/explore-ml --device auto
```

The equivalent command is `pclass-dark explore-ml`. Use `--help` for all options.
Paths default to `features.tsv`, `cluster_dmatrix.tsv`, and `explore-ml` in the
launch working directory. Source tables are never reformatted or modified.

## Resolve conflicting labels and inspect the population

Before any split or fit, an automatic prefilter keeps one example per identical
sequence with a consistent target and removes every row of identities whose target
labels conflict. Identity comes from the sequence-index hashes; when the index is
absent, only repeated accessions can be filtered and the limitation is logged.
Matching rounded feature vectors alone never establishes sequence identity.

The prefilter logs input rows, unique identities, same-target rows collapsed,
ambiguous identities/source rows removed, and final retained sequences. Counts
persist in `preflight.json` and `summary.json` under `population.prefilter_counts`.
`exclusions.tsv` and `aliases.tsv` provide the removal/retention audit. Source
inputs are unchanged. `--conflict-policy error` opts into strict validation instead
of excluding ambiguities; the default is now `exclude`.

For the inspected table: 381,627 rows describe 380,578 unique sequences; removing
1,039 ambiguous identities (2,088 rows) leaves 379,539 sequences. This is an
additional single-label training filter after the earlier FASTA deduplication.

Sequence deduplication and target uniqueness are different contracts. Original
accessions Q2FH94 (cluster_0868) and Q2FYU1 (cluster_1372) have an identical
207-residue sequence. Preparation retains representative Q2FH94 in FASTA and
preserves both cluster memberships. The feature exporter emits two Q2FH94 rows
with different targets. This is an ambiguous single-label target, not an
unremoved duplicate sequence record or a window/rounding issue.

A full preflight can be requested before optimization:

```bash
pclass-dark explore-ml --features work/features.tsv \
  --similarities work/cluster_dmatrix.tsv --split-mode accession \
  --conflict-policy exclude --output-dir work/explore-ml-pilot \
  --device cpu --validate-only
```

`--split-mode accession` is the default random split and is **homology uncontrolled**. It groups exact
sequence hashes when the extraction sequence index is available, but it cannot
establish performance on remote homologs. All evaluations are conditional on the
supplied fixed cluster system, medoid similarities and frozen feature predictor;
they do not independently evaluate their construction/training. It rejects supplied homology-map flags.
Optional homology-controlled evaluation remains available with explicit
`--split-mode homology`, `--groups`, and `--group-provenance`. The TSV header is
`accession_id<TAB>group_id`; provenance JSON records `method`, `tool_version`,
`command`, `source_sequences_sha256` and nonempty `parameters`. These files are
not needed for the default random exploration.
A ready validation-only output is continued using the same options with
`--resume`, without `--validate-only`. Invalid audits can be retried with `--resume`
after correcting data/policies before any fold has been committed.

## Training, metrics and stability

Each outer fold fits preprocessing inside training data. An inner random stratified split
(or grouped split in explicit homology mode)
selects epochs; a fresh model refits on all outer training examples. The outer
held-out examples never select parameters or epochs. SIAC is compared with the
same additive model using ordinary targets (`hard`), a majority constant, and a
similarity-central constant. All four use identical examples and folds.

Defaults: similarity target weight `--alpha 0.1`, at most `--neighbors 10` positive
neighbors, `--similarity-power 2`, `--knots 8`, `--max-epochs 100`, `--patience 10`,
`--learning-rate 0.001`, `--weight-decay 0.0001`, and `--batch-size 1024`.
These are experimental starting values. Any tuning must remain inside outer
training partitions; repeatedly choosing settings using CV test scores makes
those scores exploratory.

The primary metrics are:

- **Exact hit rate:** fraction whose predicted cluster equals the true cluster.
  This is accuracy/micro recall for this single-label experiment.
- **Weighted hit rate:** average `Similarity[true_cluster, predicted_cluster]`.
  Exact matches score 1; related misses receive partial credit. This is not
  support-weighted recall from a classification-report library.

`--repeats 1` reports variation across five fold scores. Use `--repeats 3` to give
each example three held-out predictions, enabling modal-label and pairwise semantic
agreement. Repeated training sets overlap; neither fold SD nor repeat SD is a
confidence interval. A single repeat leaves agreement fields `NA`.

## Reports

| File | Read it for |
| --- | --- |
| `summary.json` | Pooled exact/weighted hit rates in `repeat_metrics`, fold/repeat descriptive variability, paired baseline differences and stability summaries |
| `fold_metrics.tsv` | Fold scores, macro recall/similarity, class-support and cluster-size breakdowns |
| `per_cluster.tsv` | Per-cluster hits, recall and mean similarity, by fold and pooled |
| `oof_predictions.tsv` | One held-out prediction per example/configuration/repeat, exact hit, similarity credit and true-class training support |
| `oof_topk.tsv` | Up to five predicted clusters, logits and softmax scores for learned models |
| `explanations.tsv` | All 18 contributions plus bias to the predicted-versus-runner-up logit margin; missing/out-of-support flags |
| `stability.tsv` | Per-example modal and semantic agreement across repeats |
| `preflight.json` | Population, provenance, resource admission, unsupported labels and seed/split audit |
| `conflicts.tsv`, `exclusions.tsv` | Detected conflicting labels and prefiltered identities |
| `aliases.tsv`, `labels.tsv` | Evaluation-unit mapping and stable class index |
| `folds.tsv`, `inner_folds.tsv` | Reproducible outer test and inner selection assignments |
| `folds/rNNN/fNNN/` | Immutable task reports, training logs and both saved numerical spline models |
| `manifest.json` | Complete-run identity, input/code/version provenance and artifact checksums |

Example: read the main SIAC accuracy results without parsing predictions:

```python
import json
from pathlib import Path

summary = json.loads(Path("work/explore-ml/summary.json").read_text())
for result in summary["repeat_metrics"]:
    if result["config_id"] == "siac":
        print(result["repeat_id"], result["exact_hit_rate"], result["weighted_hit_rate"])
```

Softmax scores trained with soft targets are not automatically calibrated exact
membership probabilities. Correlated descriptors can share explanatory credit;
curves describe model behavior, not biological causation. No plots or deployment
model are produced in version 1.

## Progress logging

Timestamped stderr progress covers input verification/snapshotting, feature/index
and similarity scan counts, split/device/worker admission, folds, models, selection
and refit epochs/losses, early stopping, reused/committed tasks, aggregation and
completion. Long scans/batches report about every five seconds; epoch completion
always prints. Spawned worker messages include process and fold/model context.
Stdout retains the final report location. Library callers can enable normal Python
logging for `pclass_dark`; imports do not install handlers.

Training stages are numbered separately within each fold:

1. SIAC epoch selection.
2. SIAC refit on all outer training examples.
3. HARD control epoch selection.
4. HARD control refit on all outer training examples.

For example:

```text
[Repeat 1/1 | Fold 1/5 | Stage 3/4 HARD:epoch-selection] epoch 56/100 loss=0.123456
```

Every epoch/batch line keeps the stage label, so interleaved worker logs can be
identified. START/DONE messages mark transitions. Validation loss and best epoch
are saved in `training.tsv`; terminal epoch lines show only training loss.

After a conflict audit has created `explore-ml/`, retry with the default ambiguous-identity prefilter using:

```bash
# From work/; keep the combined terminal log outside the owned report directory.
../scripts/explore-ml.py --resume 2>&1 | tee explore-ml.log
```

Ambiguous-identity exclusion changes the population and is recorded in
`exclusions.tsv`; no arbitrary target is chosen. `conflicts.tsv` lists conflicting accession/cluster
memberships. Numerical epoch history persists in each model's `training.tsv`.
Optional terminal logs belong outside the report directory to avoid introducing
unrecognized experiment artifacts.

## Devices, resource bounds and recovery

`--device auto` probes a compatible GPU and falls back to CPU before fitting when
necessary. `--device cpu` is explicit CPU execution. Actual device and fallback
reasons appear in the audit. GPU folds are sequential; CPU `--workers 2` can run
fold tasks in spawned processes. Defaults are eight requested workers,
one numerical thread,
and an 8 GiB estimated memory budget. CPU concurrency is capped by available
CPUs, memory and the number of fold tasks: one repeat has at most five concurrent
fold jobs; repeated CV can use up to eight. Admission reduces effective workers
on smaller machines; `--workers 1` selects serial execution.
A single fit that cannot fit the budget fails with an estimate. Numerical/device
failures abort the fold rather than silently changing the scientific configuration.

A completed fold task is the resume unit; interrupted fits restart from their
recorded seeds. Use `--resume` with identical inputs, model settings, code,
dependency versions and actual device. Changing workers is allowed; recorded
per-task resource settings remain visible. Corrupt committed tasks are errors,
not silently reused or replaced. Existing foreign output files are never deleted.
Completion requires a verified `manifest.json`; a partial summary or `run.json`
alone does not establish success. Use a fresh output directory for a new experiment.

## Python API

```python
from pathlib import Path
from pclass_dark.explore_ml import ExploreMLConfig, run_exploration, validate_exploration

config = ExploreMLConfig(
    features=Path("work/features.tsv"),
    similarities=Path("work/cluster_dmatrix.tsv"),
    output_dir=Path("work/explore-ml"),
    device="cpu",
)
# Choose one: validation writes an audit; a run fits models.
# audit = validate_exploration(config)
result = run_exploration(config)
print(result.summary_path)
```

Lower-level reusable functions live in `ml_data`, `ml_splits`, `siac` and
`ml_reports`. `fit_siac` returns `(model, epoch_records)`;
`score_predictions` returns a metric dictionary. Saved research models use plain
NPZ arrays and schema-checked JSON, loaded with `siac.load_model`, without pickle.
Imports do not initialize Torch/sklearn or run computation.

See [SPEC-0022](specs/0022-explore-ml.md) for exact data contracts and acceptance
requirements. No full-reference classification performance is established by the
implementation or synthetic tests.
