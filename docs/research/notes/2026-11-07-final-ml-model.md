# Full-cohort additive model and query smoke test — 2026-11-07

Status: trained, portable bundle verified, raw and precomputed query paths checked.
This is a full-data refit, not an additional held-out performance measurement.

## Inputs, settings and artifacts

Source completed exploration:
`4683ddea21e305608fc8e81f38c2f71276d83d5e194fe2547890acec98db8163`,
`work/explore-ml/`. Refit all 379,539 eligible unique sequences and 3,443 labels
once, irrespective of the source folds/repeats. Ambiguous identities excluded by
the source prefilter remain excluded. Original feature/CV artifacts were not edited.

Selected configuration HARD (ordinary targets) because it outperformed SIAC on
both primary metrics in all five source folds. Historical source CV exact/weighted
scores are 60.04% / 71.15%; these are not scores measured on the new model.

Epochs: median of the five inner-selected counts, all 100, so fixed budget 100.
Other settings inherited: 8 quantile knots, cubic splines, 18 source descriptors,
160 fitted basis columns, batch=1024, learning rate=0.001, AdamW decay=0.0001,
alpha=0 for HARD, optimizer eps=1e-8. Final seed 3525665611 derived from seed 42
and purpose final-hard. Medians/knots/centering are refitted on the entire cohort.

Actual device cuda:0, AMD Radeon Graphics, 8 numerical CPU threads; GPU VRAM
8,573,157,376 bytes. CPU memory budget 8 GiB, estimated peak 2,565,677,448 bytes.
Post-validation fit/publication elapsed approximately 123.97 seconds. Training
finished at mean cross-entropy 1.777485; this is training loss, not a hit rate.
Torch 2.9.1+rocm6.3, NumPy 2.4.4, SciPy 1.18.0, scikit-learn 1.9.0,
threadpoolctl 3.6.0. Starting revision 7ca58d8f plus working-tree implementation;
exact current scientific source hashes are recorded in the bundle manifest.

```bash
python scripts/train-ml.py --experiment work/explore-ml --output-dir work/ml-model \
  --config hard --device cuda:0 --threads 8 > work/ml-model-fit.log 2>&1
```

The local GPU required access outside the tool's restricted sandbox; no model
assets were downloaded for fitting or the offline sequence smoke test.

Bundle: `work/ml-model/` (approximately 2.3 MB). Model recipe identity:
`e24b23cbe6240e5d3519ff1a3efdff174b6f33e74fdb7037b88e7b5ebbdc16f9`.
Model NPZ SHA-256:
`d267fd34a7ee6394488573f3c76945ccbf23649a2fc892b15af0d05f8bf27789`.
Manifest checksums bind actual output bytes separately from recipe identity.
Load/reload prediction and contribution reconstruction checks passed.

[Retained provenance/history/smoke evidence](2026-11-07-final-ml-model/) contains
selected records, not a second complete model bundle. Full weights remain in
`work/ml-model`; terminal training log is `work/ml-model-fit.log`.

## Real query parity

Used the 207-aa representative Q2FH94 as an offline raw-sequence smoke query.
This exact sequence identity was excluded from fitting due to ambiguous targets;
its biological label remains ambiguous and this single query is not an accuracy
benchmark. GPU raw feature extraction reproduced all 18 rounded reference feature
values exactly. Effective reference precision is .3f, bound explicitly even though
the historical extraction computation contract originally said .17g.

```bash
python scripts/predict-ml.py --model-dir work/ml-model \
  --fasta work/ml-model-smoke/query.fasta --format jsonl \
  --output work/ml-model-smoke/prediction.jsonl --device cuda:0
```

Top result: cluster_0868, softmax score 0.3676432294. Runner-up: cluster_1183,
score 0.1317253525. Score margin 1.0263938904; contribution sum plus bias
1.0263917595 (within float32 reconciliation tolerance). The top label matches one
of the original memberships, which does not resolve the annotation conflict.
Scores are uncalibrated, not membership probabilities.

## Validation and numerical limitations

- Final ML/model/CLI selection with local GPU access: **70 passed** in 60.86 s.
- Restricted-sandbox selection: 68 passed, 2 GPU cases skipped. GPU cases were
  subsequently exercised with actual device access; no skipped path is claimed
  as tested solely from a mock.
- Source-only wheel/sdist built; verified new scripts, guide and modules included.
  Installed-wheel command help, loading the real final bundle and CPU feature
  prediction passed. Build log `/tmp/pclass-dark-final-ml-build.log`.
- Actual GPU testing first exposed the previously skipped balanced-toy consistency
  test's overly strict 1e-4 logit assumption: CPU/GPU difference 0.0006954 logits,
  0.0003477 softmax score, zero label disagreements. Initial balanced bias gradients
  agree with the analytical zero within float32 epsilon, but AdamW eps=1e-8 can
  amplify cancellation-scale differences. An asymmetric 11/9 toy case differs by
  3.73e-9 logits. Optimizer/trained weights were not changed to conceal this.
  Tests retain strict ordinary-case checks and explicitly document a 5e-4 score
  bound for that balanced near-zero case, with identical labels required in both.
  This is not a universal CPU/GPU retraining-equivalence guarantee. Full reference
  CPU/GPU retraining comparison was not run; scoring the same saved weights is
  separately validated.

No new homology-controlled performance, calibration or repeated-query stability
claim follows from this final fit. See [model guide](../../ml-model.md) and
[SPEC-0023](../../specs/0023-final-ml-model.md).
