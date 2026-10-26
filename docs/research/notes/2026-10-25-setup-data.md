# Setup-data implementation and execution validation — 2026-10-25

Implemented [SPEC-0003](../../specs/0003-setup-data.md). No signature discovery,
calibration, or biological performance estimation was performed.

## Full reference preparation

Command:

```bash
PYTHONPATH=src python -m pclass_dark setup-data \
  --prepare-only --workers 2 --output-dir work/setup-data
```

The completed bundle retains 3,443 clusters, 380,578 unique full sequences,
381,627 cluster/sequence memberships, and 393,548 windows. All deduplicated
size bins match the earlier independent sequence audit. The original reference
bytes were unchanged. Run counts and identities are preserved in
[setup-data-counts.json](2026-10-25-setup-data-counts.json).
Verified reuse on the same snapshot took about 18.35 seconds. This is one local
measurement, not a general performance guarantee. The completed prepared bundle
occupies approximately 476 MiB, including derived FASTA and audit tables.

The preparation algorithm is `setup-v1`; reference generations are recorded in
its manifest. Code was the uncommitted implementation on branch `issue#4`, based
on the existing repository revision; no commit or push was performed.

## Actual model checks

Used the locally cached official `facebook/esm2_t30_150M_UR50D` checkpoint,
revision `a695f6045e2e32885fa60af20c13cb35398ce30c`, with float32 inference and
pooling on CPU. No model downloads were required. Software and asset identities
are in [pooling-check.json](2026-10-25-esm2-pooling-check.json).

Checks covered direct residue-slice pooling, single versus padded batches,
reordered batches, a 1022-residue window, and X/B/Z/U/O token preservation.
The maximum measured batching difference in the recorded smoke check was
9.54e-7. Required tolerance was rtol=1e-4, atol=1e-5. CUDA was not available;
GPU comparison was explicitly skipped.

A separate checkpoint recovery check used 66 synthetic unique sequences in two
synthetic clusters, seed 42, and no biological labels. The first run used batch
size 8 and was interrupted after its first committed 64-window shard. Restarting
with batch size 3 computed only the remaining two windows. An uninterrupted
batch-size-5 result agreed within tolerance (maximum difference 9.54e-7).
See [resume-check.json](2026-10-25-esm2-resume-check.json). Detailed local scripts,
references, shards, and output are under `work/real-model-resume/`.

Automated real-checkpoint checks use only cached assets:

```bash
PCLASS_DARK_TEST_CACHED_ESM=1 PYTHONPATH=src pytest -q
```

The normal test run skips opt-in model checks. Tests additionally cover compact
C/B corruption, malformed indices, multiprocessing reproducibility, exact
window boundaries, partial/missing/corrupt shards, OOM, finalization failure,
selection failure with forced-publication rollback, and model-free reuse.

## Packaging

Built both distribution formats in a temporary staged project containing the
real reference files and a small complete synthetic prepared/embedding pair.
Verified file inclusion, then installed the wheel into a separate temporary
virtual environment. From `/tmp`, loaded compact sets and verified the matrix
while blocking imports of Torch, Transformers, huggingface_hub, and safetensors.
No package-resource locks or writes were created. Model weights and resumable
state were excluded from the distributions. These fixture distributions are
packaging checks, not production data releases.

## Limitations and next step

Full-reference preparation is complete. Full-dataset ESM inference was not run:
production users can start it with `setup-data --offline` when the checkpoint is
cached, and resume normally after interruption. No production embeddings were
copied into package resources. GPU behavior remains untested here. Signature
selection, calibration, and the interpretation of fixed-background validation
remain for the next implementation specification.
