# SIAC CPU synthetic pilot — 2026-11-07

Purpose: validate the training, five-fold held-out prediction, control-model and
report pipeline on an independently known artificial classification rule.
**This is not protein classification performance.**

Retained artifacts: [inputs and machine-readable record](2026-11-07-siac-pilot/).
There are 90 artificial examples (30 per class), three classes and one repeat.
All 19 source descriptors are present; SCD is NA and excluded from the model.
Feature values are synthetic and need not describe physically realizable proteins.

Generation: NumPy default_rng(42), label c=0,1,2, 30 iterations per label in that
order. All descriptors start at 0.1. Set ln_length=3+c+Normal(0,0.03), acid_fraction
=0.1+0.1*c, mean_kyte_doolittle_hydropathy=-0.3+0.2*c, SCD=None. IDs are
`S<c>_<two-digit-index>`, labels `C<c>`. Similarities C0/C1=0.8, C0/C2=0.4,
C1/C2=0.25, with complementary distances and artificial medoids M0/M1/M2.
Source tables use the report writer's full numeric precision.

Input SHA-256:

- features.tsv: `71dbd546d729bdcc3b49a4bda5246b260339480d0806839cb4471ec299fc8416`.
- cluster_dmatrix.tsv: `e4fe255addbae99acdf48f67b77e5f22c1ccbce0fb88f654795682adfff899ed`.

Reproduce from the repository root into a fresh output directory:

```bash
python scripts/explore-ml.py \
  --features docs/research/notes/2026-11-07-siac-pilot/features.tsv \
  --similarities docs/research/notes/2026-11-07-siac-pilot/cluster_dmatrix.tsv \
  --split-mode accession --device cpu --max-epochs 20 --patience 5 \
  --learning-rate 0.05 --batch-size 64 --workers 1 \
  --output-dir /tmp/siac-pilot-reproduction
```

All other parameters use SPEC-0022 defaults: five outer folds, grouped inner epoch
selection, seed 42, eight knots, alpha=0.1, ten possible neighbors, similarity
power 2, AdamW decay 1e-4, one worker and one numerical thread. The retained
pilot.json records exact scientific source hashes, package versions, input hashes,
split fingerprint, runtime/hardware metadata and Git revision/dirty state. Starting
revision: 7ca58d8f with this implementation's working-tree changes.

Linux x86_64 CPU; Torch 2.9.1+rocm6.3, NumPy 2.4.4, SciPy 1.18.0,
scikit-learn 1.9.0. No GPU was available. Experiment ID:
`5e1a1cdc0914d3e856b038fcf3d0c983b5f07e1766912991da1dd54537879aed`.
Measured orchestration duration was 3.63 seconds; this tiny timing excludes the
Python interpreter's initial imports and is not a full-data throughput estimate.

Results pooled across the five outer test folds:

| Configuration | Exact hit rate | Similarity-weighted hit rate |
| --- | --- | --- |
| SIAC | 1.000 | 1.000 |
| Ordinary-target additive control | 1.000 | 1.000 |
| Majority constant | 0.333 | 0.733 |
| Similarity-central constant | 0.333 | 0.733 |

The learned models recover the deliberately easy rule. The two identical learned
results do not establish a benefit from similarity-informed training. The high
weighted baseline illustrates why semantic credit must be compared against
constant predictions as well as exact accuracy. No per-protein stability claim
is made from a single repeat; automated tests exercise repeated stability.

Full transient task/model artifacts are at
`/tmp/pclass-dark-siac-pilot-20261107/results-final/`. The retained directory contains
selected inputs and provenance only, not a resumable complete experiment family.
No reference proteins, homology-controlled biological benchmark or deployment
model was evaluated.


Subsequent split-policy revision: accession mode now uses shuffled StratifiedKFold
rather than StratifiedGroupKFold on singleton groups. The command above runs the
current random-split protocol; reproducing the historical fold fingerprint requires
the recorded original scientific code hashes. The retained results are unchanged.
