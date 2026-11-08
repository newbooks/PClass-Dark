# SIAC reference exploration: prediction accuracy and stability

Assessment date: 2026-11-07. Source: completed `work/explore-ml/`, experiment
`4683ddea21e305608fc8e81f38c2f71276d83d5e194fe2547890acec98db8163`.
Verified experiment contract identity and manifest checksums of summary.json,
fold_metrics.tsv, per_cluster.tsv and stability.tsv before summarizing. No models
were fitted or input/report files modified for this assessment.

## Population and protocol

379,539 unique sequences, 3,443 cluster targets, 18 descriptors excluding SCD.
The prefilter excluded 1,039 ambiguous identities (2,088 source membership rows)
from 380,578 identities/381,627 input rows. Results describe the retained population.
Five random class-stratified outer folds, one repeat, seed 42. Exact sequence
identity is controlled; homology is **uncontrolled**. Similarity generation
provenance is unverified in the supplied matrix; feature extraction provenance is
verified. Evaluation is conditional on the frozen feature predictor and existing
cluster/medoid system, not a fully held-out reconstruction of those resources.

Recorded configuration: alpha=0.1, neighbors=10, similarity power=2, eight cubic
spline knots, AdamW learning rate=0.001, weight decay=0.0001, batch=1024,
max_epochs=100, patience=10, min_delta=1e-5. Each inner split selects epochs before
refitting outer training examples. Actual device cuda:0, one effective worker
(eight requested), one numerical thread. The manifest records versions, scientific
source hashes, split fingerprints and dirty Git revision
7ca58d8f53cd7f1021649d1aca8d3b871cf50dfb. The recorded elapsed wall time includes
prior output-directory history; it is not interpreted here as isolated training time.

## Pooled held-out performance

| Configuration | Exact hit rate | Similarity-weighted credit | Top-5 exact hit rate | Macro recall |
| --- | --- | --- | --- | --- |
| SIAC, similarity-informed targets | 59.48% | 70.76% | 77.16% | 38.08% |
| HARD, ordinary-target additive control | 60.04% | 71.15% | 77.70% | 40.33% |
| Majority constant | 3.76% | 17.00% | NA | 0.03% |
| Similarity-central constant | 0.04% | 31.67% | NA | 0.03% |

Weighted credit is mean S[true,predicted], not an exact-hit fraction or a probability
of biological correctness. Both learned models substantially exceed the constant
controls. HARD makes 227,874 exact predictions versus 225,735 for SIAC: 2,139
additional exact hits, a 0.564 percentage-point improvement. HARD also improves
weighted credit by 0.391 points and macro recall by 2.257 points. It wins on exact
and weighted metrics in all five folds. These are observed paired differences,
not a significance test. At these settings, similarity-informed training did not
improve either primary metric.

## Fold variability and prediction stability

| Model/metric | Minimum–maximum across folds | Sample SD, percentage points |
| --- | --- | --- |
| SIAC exact | 59.30–59.73% | 0.193 |
| SIAC weighted | 70.60–71.01% | 0.175 |
| HARD exact | 59.84–60.31% | 0.208 |
| HARD weighted | 70.97–71.44% | 0.201 |

Fold scores vary little under this random-split protocol. Their SD is descriptive,
not a confidence interval: folds use overlapping training sets. Only one repeat
was performed. Each example has one held-out prediction; modal/semantic agreement
fields in stability.tsv are NA. **Individual-prediction stability has not been
measured.** It requires repeated held-out predictions, for example three repeats
of five-fold CV, rather than predictions from models trained on those examples.

## Cluster-size dependence and fitting limit

| Number of true-cluster training examples | Held-out examples | SIAC exact | HARD exact |
| --- | --- | --- | --- |
| 1–19 | 24,038 | 28.36% | 31.60% |
| 20–99 | 63,985 | 40.37% | 42.01% |
| 100–499 | 154,783 | 67.51% | 67.88% |
| >=500 | 136,733 | 64.80% | 64.60% |

No held-out examples had an absent training label. The lower macro recalls and
size breakdown show that overall accuracy is not representative of small clusters.
HARD's largest advantage is among the smallest clusters; SIAC has a small advantage
in the largest-size bin only.

All ten learned outer models (five per configuration) selected 100 epochs, the
configured maximum. This does not establish convergence. Any extension or tuning
must still use training/inner validation only, and must be reported as further
exploration rather than selecting parameters using these outer test scores.

## Interpretation

The features carry substantial cluster-predictive information under random
sequence-level splitting, with stable aggregate fold scores. The unsmoothed
additive control currently performs best overall. There is no demonstrated benefit
from the tested similarity-target smoothing configuration, no per-protein repeated
prediction stability result, and no remote-homology generalization estimate.

Reproduce the summary by reading repeat_metrics/fold_descriptive in summary.json,
selecting stratum=all or train_size in fold_metrics.tsv, and checking selected_epochs
in folds/r000/fNNN/{siac,hard}/model.json. Work artifacts and their manifest remain
unchanged; the original run configuration/source hashes are the reproducibility
record for fitting, not the subsequently edited checkout defaults.
