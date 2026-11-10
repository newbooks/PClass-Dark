# Full-cohort additive model and query prediction

Train one portable model from all eligible reference examples in a completed
[SIAC exploration](explore-ml.md). The final fit uses the full deduplicated snapshot
once, including all five outer folds; it does not average fold-specific spline
parameters or duplicate examples across repeated CV runs.

```bash
pip install -e '.[ml]'
python scripts/train-ml.py --experiment work/explore-ml --output-dir work/ml-model
```

Default `--config hard` uses ordinary targets: it performed slightly better than
similarity-informed SIAC in the observed CV run. Use `--config siac` to retain
similarity-informed targets. Both are the same additive spline architecture.
Epoch budget defaults to the rounded median of the chosen configuration's
inner-selected epoch counts across all CV folds/repeats (100 in the reference run).
`--epochs` explicitly overrides it. Other fitting parameters come from the source
experiment, not newly changed exploration defaults. Splines/medians/centering are
refitted using every eligible reference example.

Training prefers a usable GPU (`--device auto`), with explicit CPU available.
`--threads 8` controls numerical CPU threads, bounded by available CPUs; this is a
single fit, not eight independent fold jobs. `--memory-budget-gib 8` bounds planned
CPU arrays/workspaces. Actual device, settings, source hashes, versions, seed and
resource estimates are saved. Logs go to stderr; for example:

```bash
python scripts/train-ml.py --experiment work/explore-ml --output-dir work/ml-model \
  2>&1 | tee work/ml-model-fit.log
```

A new output directory is required. `--resume` reuses a matching completed bundle
or restarts an owned interrupted fit from its recorded seed. It does not resume
mid-epoch. Corrupt or incompatible completed artifacts are refused. Source CV
artifacts remain unchanged. The finished bundle contains:

- `model.npz` / `model.json`: saved numerical classifier, labels and preprocessing.
- `feature_contract.json`: feature order, extraction provenance and input rounding.
- `training.tsv`: fixed-budget training history; validation loss is NA.
- `manifest.json`: complete identity/checksums, source CV evidence and provenance.

The model bundle is portable: query prediction does not need the CV directory.
There is **no new held-out estimate for the final fit**. Source CV measurements
are recorded as historical evidence, not training accuracy or a calibrated claim
for a new query. Choosing HARD using those scores is exploratory model selection.

## Predict precomputed features

A query TSV must have `query_id` followed by exactly the 18 input columns in the
[exploration specification](specs/0022-explore-ml.md), excluding SCD and cluster_id.
Use `NA` for undefined descriptors. Do not substitute pooled PLM embeddings or
features computed under another scientific contract.

```bash
python scripts/predict-ml.py --model-dir work/ml-model --features query_features.tsv
python scripts/predict-ml.py --model-dir work/ml-model --features query_features.tsv \
  --format tsv --output query_predictions.tsv
```

Prediction applies the reference export precision (three decimal places for the
current feature table) before saved median imputation and spline evaluation.
Scores are calculated on CPU without initializing Torch or ESM.

## Predict a sequence or FASTA

For raw sequences, also install the feature-extraction dependencies:

```bash
pip install -e '.[ml,embeddings]'
python scripts/predict-ml.py --model-dir work/ml-model --fasta query.fasta
python scripts/predict-ml.py --model-dir work/ml-model --seq MDEKRH --query-id example
```

Sequence prediction uses the same pinned ESM checkpoint, full-coverage windows,
frozen exposure/Q3 heads, 18 descriptor definitions and rounding as training.
It verifies the extraction contract before initializing the predictor. The model's
reference snapshot must include extraction provenance; otherwise supply precomputed
features. A changed feature implementation/model is an error, not silently accepted.

Offline mode is the default: the checkpoint must already be cached. `--no-offline`
permits resolving/downloading the pinned checkpoint if needed. `--device cpu`
selects CPU feature extraction; auto prefers an available compatible GPU. One
predictor is reused across FASTA entries. FASTA limits are 100,000 residues per
query, 100,000 queries and 256 MiB. Sequence IDs must be unique. No cluster label
or accession lookup is needed for a novel query.

## Prediction output and interpretation

Markdown interpretation is saved to a file by default: `seq2.fasta` produces
`seq2-interpretation.md` beside the input. Feature TSV inputs use the same naming
rule. Inline `--seq` uses `<query-id>-interpretation.md` in the current directory
(with filename-unsafe characters replaced). Use `--output report.md` to choose a
path or `--stdout` to print instead. Existing files are never overwritten. When saving Markdown, the query ID and
top-five prediction table are also printed in the terminal after successful
publication. `--stdout` prints the full report without a duplicate summary. Explicit `--format tsv` output has `query_id, rank, cluster_id, logit, softmax_score` (tab-separated),
with up to five clusters per query. Logits and softmax scores are distinct;
softmax scores are **not calibrated membership probabilities**.

For explanations, use one JSON object per query:

```bash
python scripts/predict-ml.py --model-dir work/ml-model --fasta query.fasta \
  --format jsonl --output query_predictions.jsonl
```

Each object retains the comparative `feature_contributions`, `margin` and
`bias_difference` fields: the contribution sum plus bias difference reconstructs
the chosen-versus-second-choice margin. New `chosen_feature_contributions`,
`chosen_bias` and `chosen_logit` explain the chosen cluster itself; their sum
reconstructs its logit. Each own-class feature curve is centered across all training
sequences (`training_example_count`). Positive means above its average training
contribution; negative means below. Neither measures biological causation.

Generate the shortened human-readable write-up directly:

```bash
# From work/:
../scripts/predict-ml.py --fasta seq2.fasta
# Writes seq2-interpretation.md; use --output to choose another path.
```

The report includes functional annotation when matching local provenance is
available, the top five predictions (cluster ID, description up to 24 characters,
and softmax score), a brief explanation, and all 18 feature contributions. Otherwise it
uses cluster IDs. Multiple queries produce separate sections. Exact sequence
length is retained for raw inputs; lengths inferred from rounded feature tables
are approximate. Missing and out-of-range descriptors are flagged. TSV is available with `--format tsv`. Existing trained bundles need no retraining.

The pure Python formatter is
`pclass_dark.ml_interpretation.render_interpretation(prediction, descriptions=None)`.
For supplied descriptions, callers are responsible for reference provenance;
`resolve_descriptions(model)` verifies available local annotations against the
model's preparation snapshot. See [SPEC-0024](specs/0024-ml-interpretation.md).

File outputs refuse existing files and are published only after the complete input
succeeds. Stdout is also staged so late invalid FASTA entries do not emit partial
success. Put prediction files outside the owned model directory.

## Python API

```python
from pclass_dark.ml_model import (
    fit_final_model, load_final_model, predict_features, predict_sequences,
    read_query_features,
)

# Optional explicit fitting; existing matching bundles use resume=True.
# result = fit_final_model("work/explore-ml", "work/ml-model", configuration="hard")
model = load_final_model("work/ml-model")
ids, X = read_query_features("query_features.tsv")
for result in predict_features(model, X, ids):
    print(result.query_id, result.ranked[0].cluster_id)

# Raw sequence queries, with lazy checkpoint initialization:
for result in predict_sequences(model, [("example", "MDEKRH")], offline=True):
    print(result.ranked, result.feature_contributions)
```

Imports perform no model loading, downloads, computation or argument parsing.
Model files use plain numerical arrays and schema-checked JSON, never pickle.
See [SPEC-0023](specs/0023-final-ml-model.md) for exact contracts and validation.

Standalone fitting/prediction scripts and their Python APIs remain supported;
there are no root `train-ml` or `predict-ml` subcommands.

## Production integration

For library-owned fitting without an exploration directory and combined signature/
additive inference, use [the integrated workflow](integrated-classification.md).
Standalone scripts and their existing model bundles remain supported.
