# Removal-scoring optimization — 2026-10-26

The user approved cached partial sums, reusable numerical buffers, fast
leave-one-out composite scores, and first-improvement removal. These changes
are versioned as `first-improvement-logdiff-prefix-v2`. First-improvement can
produce different signatures from the original best-improvement algorithm.
Prior libraries remain readable, but new builds use distinct library identities
and private checkpoint namespaces; changing selection versions also invalidates
fit/selection receipt dependencies.

A focused timing comparison used the previously completed cluster_0945 fold_0
Gaussian caches from the [original scoped build](2026-10-26-build-library-benchmark.md).
Take the first 160 dimensions of its stored individual ranking, all 732 training
C rows, and the first 4,096 B rows. Load these arrays into RAM before timing.
Compute the current composite AUROC. Time all 158 non-core leave-one-out trials,
first using `composite_log` plus `auroc_counts`, then using a fresh `RemovalScores`
instance with the same selected subset and current count. The fast timing
includes cache construction, AUROC, cancellation/near-tie fallbacks, and
reference confirmation of proposed gains. Both loops evaluate all trials for
comparison; production selection stops at the first improvement.

Results: reference recomputation **1.2430 s**, optimized **0.3600 s**, or
**3.45×** faster for this one round. Every trial's AUROC count was identical.
The reference already uses reusable buffers, so this comparison isolates the
additional deletion-scoring cache optimization. Exact measurements and scope
are in the [JSON record](2026-10-26-removal-optimization.json).

This single timing is not a whole-library benchmark. It excludes cache reads,
checkpoint writes, profiling, and prefix screening. The smaller B is a timing
fixture only; production discovery continues using the full fixed B. A new
end-to-end benchmark was attempted but the selected setup-data input root was
locked by another invocation (`Setup busy: setup-transaction`). That invocation
was left undisturbed; no new end-to-end performance claim is made.

Numerical validation covers ordinary and extreme Gaussian logs, a removed term
dominating the sum, underflow, tied populations, subset-cache invalidation,
reusable scratch arrays, independent first-improvement examples, equivalence to
reference first-improvement paths on seeded fixtures, and interrupted trial
recovery. Near-tie fallback is a conservative heuristic, not a proof of bitwise
rank equivalence on every possible floating-point input. Every accepted gain is
confirmed with reference scores, and final metrics/thresholds/runtime scores
continue to use ordered reference accumulation.

On 2026-10-27, disposable Gaussian/prefix caches from this completed benchmark
were reclaimed during the user-authorized disk cleanup. Published results and
complete fit receipts remain; rerunning the kernel timing requires regenerating
the disposable caches from the original embeddings and stored parameters.
