# Scoped build-library benchmark — 2026-10-26

This implementation check used the existing complete embeddings in
`work/work/setup-data`, without generating new embeddings. Code was the
uncommitted implementation on branch `issue#10`. Machine platform, exact input
identities, pinned representation, numerical engine versions, thresholds,
selected dimensions, and all fit metrics are preserved in the accompanying
[JSON record](2026-10-26-build-library-benchmark.json).

```bash
PYTHONPATH=src python -m pclass_dark build-library \
  --data-dir work/work/setup-data \
  --output-dir work/library-benchmark --cluster cluster_0945
```

Defaults: one CPU worker, 1024 MiB workspace budget, epsilon and sigma floor
1e-8. The target was the median-sized cluster among targets with at least 640
windows: 915 C windows and 213,061 fixed B windows. Five discovery/validation
rotations and the all-C refit completed. The interval from the first checkpoint
receipt to the published manifest was 496.35 seconds; this excludes input
verification and is not the full command wall time. Peak RSS was not measured.
The run retained private disk-backed Gaussian caches for resumability.

The immutable output is
`work/library-benchmark/d5cc3bf6dda01fcbf8273bb2a47604b75173eee5941bbee2ea0ce58165ed22b7/`.
It contains 3,200 fold-dimension rows, 640 final-dimension rows, 915 out-of-fold
positive records, five fold signatures, and one usable final signature.

Five-fold training AUROC: **0.996319 ± 0.000825**. Validation AUROC:
**0.994898 ± 0.001078** (mean ± sample SD). The final all-C signature selected
21 dimensions, with in-sample AUROC 0.995025, precision 0.097486, recall 0.970492,
specificity 0.961415, and accuracy 0.961454 at its training-selected threshold.
The low precision despite high AUROC illustrates the prevalence and operating
point dependence of these metrics; balanced accuracy does not optimize precision.

These are internal window-level diagnostics against reused B, not independent
background, homology-controlled, or deployment validation. No all-cluster build
was run. This scoped target does not establish the resource requirements of the
largest target; assess those before committing to a full production build.
The benchmark process started before the final metadata-only addition of
software version/execution fields to newly published manifests; its algorithm
and artifact schema are unchanged.

On 2026-10-27, disposable Gaussian/prefix caches from this completed benchmark
were reclaimed during the user-authorized disk cleanup. Published results and
complete fit receipts remain; rerunning the kernel timing requires regenerating
the disposable caches from the original embeddings and stored parameters.
