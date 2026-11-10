# Integrated reranking smoke — 2026-11-08

This is a scientific/runtime smoke test, not a held-out accuracy measurement or
production deployment. It uses the existing full-cohort HARD model and a read-only
existing signature/distribution library. A small synthetic test separately covers
setup → direct fitting without exploration → immutable envelope → infer/report
publication and installed release inventory.

## Real query and results

Input is the 129-residue seq2 lysozyme demonstration sequence used in earlier
interpretation work. PClass-Dark accepts 37 clusters. The additive classifier
selects cluster_1096 (peptidoglycan muralytic with hydrolase activity), moving it
from original signature rank 4 to accepted-candidate rerank 1. Its full-vocabulary
softmax score is 0.8536225617748884. It is the only global top-five classifier
prediction accepted by PClass-Dark for this query. Other global suggestions remain
explicitly unaccepted. No acceptance or B/C value was changed.

The opening paragraph explains the selected accepted cluster using length,
predicted beta-strand fraction and mean hydropathy. Detailed signature/window
evidence precedes the additive reranking and centered feature contributions.

- [Human report](2026-11-08-integrated-reranking-smoke/interpretation.md)
- [Comprehensive JSON](2026-11-08-integrated-reranking-smoke/classification.json)
- [Terminal summary](2026-11-08-integrated-reranking-smoke/screen.txt)

## Reproduction and provenance

The local temporary parity harness was /tmp/integrated-real-smoke.py. It constructs
an IntegratedSession around the existing work/ml-model weights after checking
preparation and head identity, then calls the original signature compute path and
the new shared-backend supplemental callback. It does not modify selection.json,
train weights, or claim that the old CV-derived model is a newly fitted production
generation. The report's aggregate identity explicitly says saved-weight parity
harness. Actual model/library/reference IDs and numerical evidence are in JSON.

Execution: device=cuda:0, offline=True, workers=8, default signature batches and
workspace. One ESM backend supplies signature embeddings and structural residue
states; additive numerical scoring uses CPU NumPy/SciPy. GPU access required an
execution environment outside the restricted sandbox. The selected production
pointer was checked byte-for-byte unchanged after the run. No latency benchmark
was collected and no full-cohort retraining was performed.
