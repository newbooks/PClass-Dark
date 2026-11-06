# ProStruct feature extraction for candidate reranking

Assessment date: 2026-11-06. Status: feasible; implementation and reranking
performance evaluation remain planned. No change to inference behavior.

## Inspected sources and available artifacts

ProStruct revision `fa8fa0da7a9f2ecb1ab096f979b3936dd2409d54`.
Its working tree also contains an untracked exact-sequence-deduplication note
and derived feature tables; these are observations, not committed interfaces.

Relevant reference implementations:

- `../ProStruct/src/prostruct/enrichment.py`: `compute_ele`, per-sequence
  RSA/ALL-CEF and Q3 predictions, X context and NA outputs for unknown residues,
  checksum-bound resumable outputs and CPU/GPU execution.
- `../ProStruct/src/prostruct/library.py`: validated dual-head loading,
  task-specific standardization, streaming residue embedding and prediction.
- `../ProStruct/src/prostruct/esm2.py`: full-coverage overlapping window planning
  and batched residue-state extraction.
- `../ProStruct/src/prostruct/features.py`: `summarize` and `extract`, the 19
  descriptor definitions, QC, threading and publication.
- `../ProStruct/scripts/{compute_ele,feature_extraction,profile_clusters,compose_signature}.py`
  and SPEC-0023, 0024, 0026, 0028, 0029 and 0030.
  `compose_signature.py` currently delegates to `prostruct.signatures`, whose
  selected-signature behavior extends the historical prefix-only SPEC-0029.

The actual dual-head artifact exists at `../ProStruct/artifacts/library/`:

- Library ID: `9ed4d06c9282266bc1b400f440e15ef61b9c5454195857750c1bd8a8dfa90315`.
- `models.safetensors`: 3,176,040 bytes, both heads and their fitted normalization.
- Each head: 640 -> 512 -> 128 -> output, exact GELU, evaluation mode.
- Exposure output: one raw, unclipped ALL contextual exposure fraction (ALL CEF).
- Q3 output: three logits, class order C/E/H, raw argmax without smoothing.

The bundled `resources/models/rsa_mlp_wide/v1/model.safetensors` is a different,
RSA-only artifact: it is insufficient for the complete feature set. Preserve
provenance and distribution terms when packaging the dual-head artifact; do not
silently substitute the bundled single-task model.

## Reuse and incompatibilities

The selected PClass-Dark preparation is
`8e5994726cb958fce5d2396835f9a3767da897007110a9fda4f2e3b06cccbe4e`.
Its manifest reports 380,578 retained unique full sequences, 381,627 memberships,
3,443 retained clusters and 393,548 windows. Reuse `sequences.fasta`,
`memberships.tsv` and `aliases.tsv`; a single sequence may belong to multiple
clusters. Compute features once per sequence hash and join all memberships.

Both projects use ESM-2 `facebook/esm2_t30_150M_UR50D`, revision
`a695f6045e2e32885fa60af20c13cb35398ce30c`, final layer 30, dimension 640.
However, saved PClass-Dark embeddings are residue-mean-pooled window vectors.
ProStruct heads require a vector for every residue: those vectors cannot be
reconstructed from the pooled cache, and applying the nonlinear heads to pooled
vectors does not reproduce residue predictions or the resulting descriptors.
The checkpoint/cache can be shared after identity validation; representations
must be generated again unless verified residue predictions are reused.

Window contracts differ:

| Property | PClass-Dark | ProStruct |
| --- | --- | --- |
| Maximum residue window | 1022 | 1021 |
| Long sequences | N and C terminal windows | Distributed overlapping windows covering the entire sequence |
| Overlap handling | Separate candidate evidence | Each residue owned by its most central window; ties prefer earlier start |
| Minimum overlap | Not a full-coverage tiling | 100 residues |
| Saved representation | One pooled vector/window | Streaming residue vectors for two heads |

The selected Dark manifest reports 2,104 proteins with omitted centers and
2,018,649 omitted residues. Those centers matter for whole-protein features.

ProStruct's existing `enrichment/features.tsv` and `raw_ele/` may avoid much of
reference inference, but only after sequence-hash, model/policy and completeness
validation. Its derived `features_dedup.tsv` keeps the first cluster label and
loses later cross-cluster memberships. It is not a drop-in Dark reference table;
join by exact sequence and restore Dark's membership relation. Existing ProStruct
profiles/backgrounds/signatures must be rebuilt for the selected Dark universe.

## The 19 descriptors

Sequence-only (8): `ln_length`, `acid_fraction`, `base_fraction`,
`net_charge_balance`, `charged_fraction`, `histidine_fraction`,
`mean_kyte_doolittle_hydropathy`, `sequence_charge_decoration`.

Exposure-dependent (8): `surface_acid_over_all`, `buried_acid_over_all`,
`surface_base_over_all`, `buried_base_over_all`, `surface_fraction`,
`surface_acid_over_surface`, `surface_base_over_surface`,
`surface_charged_over_surface`.

Q3-dependent (3): `q3_h_fraction`, `q3_e_fraction`, `q3_c_fraction`.

These are composition and predicted structural descriptors, including charge
proxies, rather than a physical electrostatic calculation. No 3D structure,
DSSP run, or surface-charge solver is required for prediction with existing heads.
The exposure target is ALL CEF, not conventional maximum-ASA-normalized RSA.
Surface means raw ALL CEF >0.05; equality is buried. Do not clip raw predictions.
Basic composition includes histidine; net-charge/charged proxies exclude it.
Preserve full-length versus surface-conditional denominators, unknown-residue
masking, undefined-feature NA rules, raw Q3 labels and output precision policy.

## Proposed reproducible workflow

1. Verify the selected preparation and export unique full sequences with all
   memberships. Bind sequence hashes and preparation identity in a manifest.
2. Load the pinned ESM checkpoint and validated dual-head artifact once. Generate
   ProStruct's full-coverage windows; extract float32 residue states in bounded
   batches (reference implementation uses eight windows), excluding special tokens.
3. Assign each residue to one central window, apply each head's own fitted
   normalization and constant-feature mask, and predict ALL CEF plus raw Q3.
4. Aggregate the 19 descriptors on the original complete sequence, once per hash.
   Save feature schema, QC, model/checkpoint/window policy and code identities.
   Optional residue files enable audit/resume; caching all residue embeddings is
   unnecessary. Keep deterministic ordering and atomic completion receipts.
5. For a new query, use exactly the same normalization, windows, predictions and
   aggregation. No cluster membership or known accession is scientifically needed
   for this operation. ProStruct's batch extractor currently expects membership
   metadata and accession-named files; adapt this wrapper for arbitrary query IDs
   and content-addressed caching rather than inventing functional labels.
6. Fit cluster feature profiles and a candidate compatibility score on training
   references only. Apply it to Dark's already accepted candidates, retaining
   original B/C evidence, original rank and the new score/rank. Missing or invalid
   feature evidence needs an explicit deterministic fallback to original order.

A distinct experiment could extract descriptors separately for Dark's accepted
N/C windows. That requires reference window profiles and matching query windows;
whole-protein profiles cannot be compared directly with fragment features.
Even a 1022-residue Dark window needs ProStruct's internal 1021-residue tiling.
Length, charge decoration and structural fractions can change substantially on
fragments. Choose and record the biological unit before implementation.

Keep the implementation, frozen constants, artifact loader/schema and tests in
PClass-Dark rather than importing a neighboring checkout at runtime. Add a
numbered implementation specification before coding. Provide reusable Python
operations and a thin user-facing adapter consistent with repository conventions.
GPU preference, explicit CPU mode, bounded queues, CPU aggregation worker count,
resume, provenance and CPU/GPU reporting are required. Shared ESM forward work
with Dark inference is possible only for exactly identical window/tokenization
inputs; the two window policies prevent assuming universal reuse.

## Reranking experiment and validation

Current `combine_matches` selects each cluster's minimum-B accepted window and
sorts clusters by ascending B, then cluster ID. ProStruct Gaussian compatibility
scores provide a plausible additional signal, but are not calibrated comparable
membership probabilities across clusters. Both methods depend on the same PLM,
so additional information and improved ranking are hypotheses, not established.

Keep candidate acceptance fixed in the first experiment. Measure candidate-set
truth coverage (the attainable ceiling), top-1 accuracy and reciprocal rank before
and after reranking, including the subset whose correct cluster was accepted.
Use parent-sequence/homology-separated evaluation; keep all aliases and windows
together. Fit profiles, feature choices, score combinations and thresholds only
on training data, tune on validation, and report final held-out results. Existing
ProStruct profiles/signature metrics are in-sample diagnostics and cannot establish
held-out reranking improvement. Include a sequence-only eight-feature ablation.

Software validation must cover analytical feature examples, all 19 columns,
reference-head agreement, overlap ownership, unknowns/zero surface, missing
features, cross-cluster membership preservation, CPU and worker consistency,
query/reference parity, interrupted resume and model/schema mismatch rejection.

## Checks actually executed

- Loaded the real dual-head library using `load_library`: complete file/hash and
  tensor-schema validation passed; both heads returned finite outputs on two
  synthetic standardized CPU vectors (shapes 2x1 and 2x3).
- Ran ProStruct `test_features.py`, `test_feature_extraction.py` and
  `test_esm2_batch.py`: **49 passed**. These cover extraction and padded small
  random-ESM batched/single agreement, not production-checkpoint accuracy.
- No GPU is visible to PyTorch. One existing AMD_SERIALIZE_KERNEL warning appeared.
- No production ESM inference, new reference extraction, model training or
  reranking experiment was run. Throughput and ranking improvement are unmeasured.
