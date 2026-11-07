# Frozen exposure and secondary-structure prediction

`pclass-dark build-library` now prepares the frozen ProStruct prediction heads
alongside signatures and C/B distributions. Normal, scoped and reused builds
install a checksum-verified bundle at
`data/library/structural-models/<structural-model-id>/` and print its path.
No structural reference inference, training or ESM download runs during this step.
`--clean` neither creates nor repairs the structural bundle. Existing production
library IDs, selectors and infer B/C calculations are unchanged.

The same models are included in wheel/source distributions under
`pclass_dark/resources/structure/v1/`, so prediction needs no neighboring checkout.
The model card, original configuration, training history/metrics and provenance
travel with the unmodified weights. Shared-data packaging does not need to export
the duplicate output companion because the frozen package copy is always included.
A corrupt existing companion fails verification rather than being silently replaced.

## Models required for the planned 19 features

Only three pretrained components are needed:

1. The already used ESM-2 t30/150M checkpoint at pinned revision
   `a695f6045e2e32885fa60af20c13cb35398ce30c`.
2. ProStruct's exposure head, predicting raw ALL contextual exposure fraction
   (ALL CEF), called `rsa` in output.
3. ProStruct's Q3 head, predicting raw C/E/H secondary-structure labels.

The latter two share a 3,176,040-byte safetensors file, including task-specific
normalization. No extra electrostatic model, structure predictor or DSSP program
is needed to calculate the proposed descriptors. Eight descriptors use sequence
alone, eight additionally use exposure and three use Q3. Fitting cluster profiles
and selecting/calibrating a reranker remain separate future work.

ALL CEF is not conventional maximum-ASA-normalized RSA. Predictions are unclipped;
Q3 uses raw argmax in C/E/H order, without smoothing. These values do not imply a
measured 3D structure or physical electrostatic field.

## Python prediction interface

Install `pip install -e '.[embeddings]'` for prediction. Model preparation itself
uses only the core dependencies and does not load torch or transformers.

```python
from pclass_dark.library import build_library
from pclass_dark.structure import StructuralPredictor

built = build_library()  # Existing libraries are reused when compatible.
predictor = StructuralPredictor(
    model_dir=built.structural_model_dir,  # Omit to use packaged models.
    device="auto",       # Prefer GPU, CPU fallback; use "cpu" explicitly.
    offline=True,        # Require the pinned checkpoint in the local HF cache.
    window_batch_size=8, # 1..8; bound GPU memory, 1 for serial comparison.
    cpu_threads=8,       # 1..8; bound numerical-library CPU threads.
)
result = predictor.predict("ACDEKRHX")
print(result.sequence, result.rsa, result.q3, result.device, result.model_id)
# Reuse predictor for subsequent queries; ESM loads only once.
```

`StructuralPrediction` returns read-only arrays in original sequence order.
Noncanonical ASCII letters and `*?-` retain their positions, become X in ESM
context and receive `numpy.nan` exposure plus `"NA"` Q3. Lowercase is normalized;
empty sequences, embedded whitespace and other characters raise `ValueError`.
Entirely noncanonical queries do not load ESM. GPU execution is requested through
`device="cuda"` or `"cuda:<index>"`; explicit-device failures raise errors,
whereas automatic initialization can fall back to CPU. Actual device is returned.
GPU/CPU roundoff can affect nearly tied Q3 logits. The predictor owns mutable
model state; use separate instances rather than concurrent calls on one instance.

For integrations already holding compatible residue states:

```python
from pclass_dark.structure import StructuralHeads

heads = StructuralHeads()  # CPU; frozen heads only, no ESM download/load.
# states must be finite float32, shape (residue_count, 640), from the pinned
# ESM final layer and the specified structural window policy.
rsa, q3 = heads.predict(states, batch_size=4096, cpu_threads=1)
```

The returned exposure values are model outputs, not the final 19-feature vector.
The experimental [feature extractor](feature-extraction.md) now aggregates the
19 descriptors; the `rerank` command and ML evaluation remain future work.

## Sequence windows and existing accessions

Structural prediction uses ProStruct's full-coverage windows of at most 1021
residues with minimum overlap 100. A residue uses its most central window,
resolving equal centers toward the earlier start. This differs from Dark's
1022-residue terminal windows. Saved pooled Dark embeddings cannot reconstruct
per-residue states. The pinned checkpoint/cache is shared; the representation
must be recomputed unless verified residue predictions are already available.

Unique-sequence accession data already exists in the selected preparation.
Resolve its location through `data/setup-data/selection.json`:

- `prepared/<preparation-id>/sequences.fasta`: one representative accession and
  full sequence per retained unique sequence.
- `memberships.tsv`: every retained sequence/cluster relation and representative.
- `aliases.tsv`: original accession, original cluster, sequence hash,
  representative and retention flags. Filter retention flags for retained inputs.

The inspected selected preparation contains 380,578 unique sequences, 381,627
memberships and 3,443 retained clusters. Identical sequences can belong to multiple
clusters: compute features once per sequence hash while preserving every retained
membership. Exact deduplication does not remove nonidentical homologs.

See [SPEC-0020](specs/0020-structural-prediction-models.md) for provenance,
contracts, validation and remaining evaluation work.


For extraction runtime provenance, `StructuralPredictor.prepare()` initializes the
runtime without forwarding a dummy sequence and returns a fresh JSON-compatible
metadata dictionary. It reports the actual device, pinned embedding contract,
software/backend versions and numerical thread settings under the predictor's
configured CPU limits. Repeated calls reuse the same initialized model.
