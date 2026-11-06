# SPEC-0020: Portable RSA/ALL-CEF and Q3 predictors

Status: implemented, 2026-11-06; validation recorded below.

## Purpose and provenance

Prepare ProStruct's frozen dual prediction heads during `build-library`, and
provide importable sequence-to-residue prediction for future 19-feature reranking.
This does not extract the reference feature table, fit a reranker, change accepted
candidates, or change current infer B/C scores. No model retraining is required.
ProStruct revision: fa8fa0da7a9f2ecb1ab096f979b3936dd2409d54. Source library:
9ed4d06c9282266bc1b400f440e15ef61b9c5454195857750c1bd8a8dfa90315.
Reference implementations: library.py, esm2.py, enrichment.py and predict_rsa.py;
see [feasibility assessment](../research/prostruct-reranking-feasibility.md).

## Contracts and design

Bundle the original `models.safetensors` with a compact content-addressed manifest,
configuration, training provenance and model card. Do not depend on a neighboring
checkout, and do not copy its structural training dataset or DSSP coordinates.
The model artifact and standardization tensors are unchanged. Package the assets
in wheel and sdist. Keep optional torch/transformers/safetensors dependencies lazy;
ordinary builds verify hashes and install the model bundle without loading ESM,
requiring a GPU, downloading weights or importing torch.

Successful ordinary `build_library` calls, including reused/scoped builds, install
a validated companion at `<output>/structural-models/<bundle-id>`. Extend the
result with its ID/path, and report the path in CLI output. Keep production and
C/B artifact IDs/selectors unchanged. `--clean` remains validation/deletion only:
it must not create or repair structural model assets. Structural companions are
outside cleanup's eligible generation namespaces. Installed packages also have
the bundled models, independent of shared production-data export.

Prediction API: load a validated head bundle, predict from finite float32 residue
states shaped (N,640), and predict full sequences with lazy pinned ESM loading.
Return raw exposure and C/E/H labels, plus actual device and model identity.
RSA means raw ALL contextual exposure fraction, not conventional normalized RSA;
no clipping, smoothing or retraining. Each head uses its own float64 mean/scale,
float32 standardized inputs, constant dimensions zeroed, exact GELU architecture
640->512->128->{1,3}; Q3 argmax uses class order C/E/H and earliest-class ties.
Reject corrupt files, schema/contract/tensor mismatches, invalid/empty sequences,
nonfinite/wrong-width states and invalid batch/device requests.

Use frozen ESM-2 t30/150M revision a695f6045e2e32885fa60af20c13cb35398ce30c,
final layer 30, float32. Reuse the existing checked checkpoint loader, adding a
residue-state operation without changing mean pooling. ProStruct windows are
1021 residues with >=100 overlap, terminally anchored, distributed starts;
choose most-central ownership per residue, ties to earlier window. Bound forward
batches, preserve full coverage without averaging duplicate residue predictions.
Canonical alphabet ACDEFGHIKLMNPQRSTVWY; noncanonical ASCII letters and *?- are
X in model context and NA in both residue outputs, retaining original indices.
GPU-first auto mode with CPU fallback; explicit CPU/device and offline control.
CPU heads use bounded numerical-library threads; no per-residue worker processes.
Prediction memory is bounded by a window batch plus O(sequence length) outputs,
not a full sequence-by-640 matrix. A future extraction orchestrator may stream
multiple sequences and parallelize independent CPU aggregation.

No additional pretrained models are needed for the 19 descriptors: eight use
sequence only, eight use exposure predictions, three use Q3. Reranker cluster
profiles, feature choices and any calibration remain future fitted artifacts.
The current unique sequence accessions already exist in selected preparation:
sequences.fasta representative headers, aliases.tsv originals and memberships.tsv
all retained cluster relations. Preserve multi-cluster identical sequences.

## Validation expectations

Test analytical heads and independent normalization/argmax examples, real frozen
head parity with ProStruct, hash/schema/tensor corruption, full-coverage window
ownership against brute force, padding/token exclusion, unknown masking, CPU and
batch consistency, lazy imports, successful/reused build integration, --clean
non-creation, and packaged asset inclusion. Run relevant library/embedding/CLI,
cleanup and packaging regressions. Test GPU when available and report otherwise.
No ranking improvement or production throughput claim follows from these checks.

## Implementation and validation — 2026-11-06

Implemented `structural_models.py` (dependency-light artifact verification and
atomic companion installation), `structure.py` (validated heads, full-sequence
prediction, window ownership and masking), and `EsmBackend.residue_states`.
`LibraryResult` now includes `structural_model_id` and `structural_model_dir`;
ordinary build-library prepares/reuses them and the CLI reports the path. Model
preparation failure carries `production_result` so the completed signature pair
remains identifiable. CLI scientific prediction will be integrated through the
future rerank workflow; no extra root prediction command was introduced.

Frozen imported model SHA-256:
`1892cfd64abb13bafb6f199ae020f5d0e54664c15866a6c428a862603e10530e`.
Portable bundle ID:
`d0f2ef975fef2b86e7057643b250e467a77e81afc58aba42aba8c2f207e844b2`.
Both complete safetensors heads are copied byte-for-byte with normalization.
The compact manifest replaces ProStruct's large coordinate-file inventory:
training coordinates are not needed for inference. Original training provenance,
configuration, history and diagnostics are retained. ESM uses this repository's
checked asset/token loader rather than importing ProStruct's loader. Structural
windowing/ownership and task outputs retain ProStruct's policy.

Checks executed:

- `pytest -q tests/test_structure.py`: 46 passed, 1 skipped (GPU unavailable).
  Two subsequently added copy-failure/production-preservation tests: 2 passed.
  Includes analytical head normalization and unclipped outputs; explicit frozen
  tensor formula parity; brute-force overlap ownership; unknown masking;
  CPU/batch agreement; tensor/hash/contract rejection; auto-device fallback;
  no eager torch/ESM imports; clean non-creation; and CLI path reporting.
- The new suite builds a wheel from an sdist in an isolated temporary source
  tree, checks all model files in both archives, then loads both heads from the
  extracted wheel away from this repository and ProStruct. Passed.
- Real pinned ESM CPU/offline query `ACDEKRHX`: seven finite exposure predictions,
  raw Q3 labels and an NA/NA unknown position. No downloads occurred.
- Independent real-model comparison against original ProStruct
  `enrichment.write_accession`, sequence
  `MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEX` (41 residues), same pinned ESM
  tokenizer/model and original dual-head library: all Q3 outputs agreed;
  maximum exposure difference 4.930724994967051e-10 (reference TSV rounding).
  CPU, 4 numerical threads; new head implementation and reference head code
  both executed. No full reference inference or reranking evaluation.
- Read selected sequences.fasta headers independently: 380,578 entries and
  380,578 distinct accession headers.
- Final structural/shared-path/CLI run: 78 passed, 1 skipped. Separate
  structural/legacy-library/library-path run: 68 passed. The broader production,
  embedding, cleanup, packaging and shared-path run had 210 passes and two
  test-fixture failures; both were fixed and passed in the final run. See the
  progress log for exact suites. Ruff lint/format and whitespace checks passed.

GPU was unavailable to PyTorch; actual GPU comparison was skipped. Existing
AMD_SERIALIZE_KERNEL warning appeared. Transformers reports unused LM-head and
pooler parameters through the existing ESM loader; the pooler does not contribute
to residue states. This does not affect the measured reference agreement.
No CPU multiprocessing is added to this GPU/batched prediction operation;
independent CPU feature aggregation remains future work. No reference feature
matrix, reranker profiles, calibrated rank score or held-out ranking results were
produced. No manuscript changes.
