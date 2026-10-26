# SPEC-0003: Reference preparation and resumable embeddings with setup-data

- Date: 2026-10-25.
- Status: Implemented and validated on CPU; full-reference preparation complete.
  Production embedding of the full dataset and GPU validation have not been run.
- Related: [bundled references](0002-bundled-references.md),
  [sequence identity audit](../research/notes/2026-10-25-sequence-identity.md).

## Purpose and scope

Prepare a reproducible working dataset from the bundled profunc memberships and
sequences, then compute frozen PLM embeddings for the prepared windows. Use the
bundled references directly: do not copy the entire raw reference pair, modify
installed package resources, access profunc at runtime, or refresh reference data
from the Internet. Derived sequences and windows are intentional outputs.

Include exact sequence deduplication, cluster-size filtering, terminal windows,
training/validation assignments, checkpoint acquisition when necessary, and
resumable residue-pooled embeddings. Frozen embeddings may be computed for both
splits because this stage does not fit dataset-dependent parameters. Keep
normalization fitted to the dataset, signature discovery, feature selection,
calibration, and predictive evaluation in the next model-building step.

Prepare fixed cluster-capped background definitions for downstream comparisons.
Do not run homology searches. Homology is expected and intentionally uncontrolled in this initial protocol; validation
must not be described as distant-homolog evaluation.

The reference preparation phase remains entirely offline. The embedding phase
may download a pinned checkpoint and tokenizer if absent locally; `--offline`
forbids all network access and requires local assets. `--prepare-only` avoids
loading model dependencies or initializing a GPU. This supersedes the earlier
draft in which embedding was outside setup-data.

## Interfaces

Provide both a thin Typer subcommand and a reusable library API:

```bash
pclass-dark setup-data --output-dir work/setup-data --min-clusters 10
```

```python
from pclass_dark.setup_data import prepare_data

result = prepare_data(
    output_dir="work/setup-data",
    min_clusters=10,
    seed=42,
    workers=1,
    prepare_only=False,
    offline=False,
    model="facebook/esm2_t30_150M_UR50D",
    revision="a695f6045e2e32885fa60af20c13cb35398ce30c",
    device="auto",
    batch_size=8,
)
```

Command options (also keyword arguments to `prepare_data`):

| Option | Default | Meaning |
|---|---|---|
| `--output-dir` | `work/setup-data` | Derived dataset destination, relative to invocation directory |
| `--min-clusters` | `10` | Minimum unique full-sequence members per retained cluster |
| `--seed` | `42` | Nonnegative seed for random window assignment to five folds |
| `--workers` | `1` | Positive CPU worker count for suitable independent preparation work |
| `--prepare-only` | false | Stop after local preparation; do not load or download the PLM |
| `--offline` | false | Forbid network access, including model metadata requests |
| `--model` | `facebook/esm2_t30_150M_UR50D` | Frozen PLM checkpoint |
| `--revision` | `a695f6045e2e32885fa60af20c13cb35398ce30c` | Immutable checkpoint/tokenizer commit; reject floating names |
| `--device` | `auto` | Prefer compatible GPU; support explicit CPU or GPU |
| `--batch-size` | `8` | Positive inference batch size; may change on resume |
| `--force` | false | Archive and rebuild the selected prepared/embedding bundles and checkpoints |

The default checkpoint commit was verified against the official Hugging Face
repository on 2026-10-25. Initially support only this model/revision pair; reject
other pairs until their contracts and checksums are added to the supported-model
registry. The revision option makes the identity explicit, not permission to
load arbitrary untested checkpoints. No floating revisions or remote code.

Verified assets at this revision:

| Asset | SHA-256 | Size (bytes, if listed) |
|---|---|---:|
| `model.safetensors` | `c3f1da8aea53bddd32c246c86168c23b9fd72341fb9db9a94436f855f5053566` | 595257706 |
| `config.json` | `e512f68ec444d99477703a9806639ca83da3dbc19f6c5fe428d6e5b7460972dc` | — |
| `vocab.txt` | `0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03` | — |

Acquire only these assets plus `tokenizer_config.json` and
`special_tokens_map.json` from the pinned revision. Record hashes of the latter
two and validate the tokenizer contract. Never download duplicate PyTorch/TF
weights; use safetensors. Verify the weights' reported digest on acquisition.
Use the Hugging Face cache; offline resolution must use local files only.
The source for the asset identity is the official pinned repository metadata:
`https://huggingface.co/api/models/facebook/esm2_t30_150M_UR50D/revision/` followed
by the commit above and `?blobs=true`.

Retain the user-requested spelling `--min-clusters`, but explicitly explain in
help that it is a member-count threshold, not a minimum number of clusters.
Require a positive integer threshold and a nonnegative integer seed.
Create exactly five folds; do not expose a validation-fraction option.
The terminal window length is fixed at 1022 residues for this
initial implementation and recorded in metadata. Preparation uses CPU workers (spawn-based processes, bounded by requested workers, CPU count,
source cluster count, and a maximum of 32); embedding prefers CUDA with CPU fallback when automatic device selection cannot
initialize a compatible CUDA device. Support `auto`, `cpu`, `cuda`, and
`cuda:<nonnegative-index>`; reject other device strings in this version.
Explicit CUDA requests must fail clearly if unavailable.
Do not silently switch devices after a partially completed inference batch.
Preparation worker counts must not change results.

The library accepts an optional local `reference_dir` override (API only) for
manifest-bound integrations and tests; the CLI always uses packaged references.
The library returns a frozen `SetupResult` with `preparation_id`, `prepared_dir`,
`embedding_id` and `embedding_dir` (both null for prepare-only), `counts`, and
`reused_prepared`/`reused_embeddings` flags. Provide `force=False` in the API.
Use `ValueError` for invalid configuration/reference schemas, `IntegrityError`
(a `ValueError` subclass) for checksum/identity failure, `FileExistsError` for
conflicting complete outputs without force, `SetupBusyError` (`RuntimeError`)
for lock contention, and ordinary `OSError` for filesystem failures. Preserve
exception causes. CLI configuration errors exit 2, execution errors exit 1,
Ctrl-C exits 130, and success exits 0. Imports have no
side effects; CLI converts errors to clear messages and nonzero exits.

## Input verification

Resolve packaged resources through `importlib.resources`. Before processing:

- Verify the bundled manifest and its listed file sizes and SHA-256 digests;
  reject unresolved Git LFS pointers, missing files, and mismatched provenance.
- Require the exact membership header `member_id`, `cluster_id`, nonempty IDs,
  and exactly one source cluster membership per accession.
- Parse primary FASTA IDs, require unique IDs and nonempty sequences, and verify
  that every clustered accession has a sequence. Extra unclustered records are
  counted and excluded from the prepared training dataset.
- Preserve sequence characters and accession identifiers. Strip leading/trailing
  ASCII whitespace from sequence lines, concatenate them, and reject embedded
  whitespace, empty IDs, tab/newline-containing IDs, and nonalphabetic residues. Do not uppercase, replace
  residues, discard noncanonical sequences, or silently shorten them.
- Report noncanonical symbols and affected accessions. A model-specific token
  policy belongs to embedding generation; setup does not imply model eligibility.

## Exact deduplication and cluster filtering

1. Group clustered accessions by exact full-sequence equality. Use stable SHA-256
   identifiers for sequence groups, but verify actual string equality within
   digest groups and fail on a collision. Identity is evaluated before windowing.
2. Choose the lexicographically smallest clustered accession as the global
   representative for each identical-sequence group. Preserve every original
   accession and its original cluster label in an alias table.
3. Derive unique `(cluster_id, sequence_id)` memberships. A sequence shared across
   distinct clusters remains a member of each cluster; do not arbitrarily select
   one cluster or discard the other labels. Store its sequence once globally.
4. Count unique full-sequence members separately for each cluster. Drop clusters
   with counts strictly less than `min_clusters`; retain clusters equal to the
   threshold. Do not count windows or accession aliases as additional members.
5. Remove sequence groups with no retained cluster membership from prepared
   sequence/window outputs. Preserve their audit entries and exclusion reasons.
   Keep representative IDs chosen before filtering stable even when the
   representative accession's original cluster is excluded; aliases and retained
   membership rows explain the relationship.

For the current snapshot and default threshold, the audit predicts 81 excluded
clusters and 3,443 retained clusters. Retained membership count is 381,627
(382,268 unique cluster/sequence memberships minus 641 in excluded clusters).
This is not the globally unique retained-sequence count, which implementation
must compute. Store actual observed counts rather than hard-code these values.

Cross-cluster identical sequences imply multiple labels for the same input.
The output schema must support this explicitly; later scoring and validation
must not treat another retained true label as a negative by default.

## Terminal window construction

Follow `_plm_embedding_windows` in
`../PClass/src/pclass/build_models/workflow.py` and its corresponding tests,
with the explicit change from PClass's current 1024 limit to **1022** here.
For each retained full-sequence group, use representative accession `A`:

| Full length L | Embedding record ID | Sequence | One-based inclusive coordinates |
|---|---|---|---|
| L <= 1022 | `A` | Full sequence | 1..L |
| L > 1022 | `N-A` | `sequence[:1022]` | 1..1022 |
| L > 1022 | `C-A` | `sequence[-1022:]` | L-1021..L |

Exactly two windows are emitted for a long sequence. They may overlap when
1022 < L < 2044; for L > 2044, the central region is omitted. Record covered and
omitted residue counts and report how many proteins have omitted centers.
This is terminal sampling, not exhaustive sliding-window coverage.

Both windows inherit the parent sequence's complete retained cluster-label set.
Assign folds to window records independently after window construction; the N
and C windows of one parent may belong to the same or different folds. Keep explicit parent IDs and coordinates rather than
relying on
prefix stripping to reconstruct them. Detect generated ID collisions. If N and C
happen to contain identical strings, retain both named records for compatibility
and report their identity; no second deduplication changes cluster member counts.

## Embedding contract: residue-only pooling

The embedding phase of setup-data implements this contract. Window lengths and
IDs from preparation must remain consistent with tokenization. Preserve the
prepared windows, original characters, and audit records unchanged.

For `facebook/esm2_t30_150M_UR50D`, use final-layer encoder hidden states
(layer 30), with hidden dimension 640. Do not pool masked-language-model logits,
attention weights, or contact predictions. For a window of L residues, normal
ESM-2 tokenization produces `<cls>`, L residue tokens, and `<eos>`: L+2 positions.
Thus the 1022-residue maximum gives 1024 nonpadding tokens. Special tokens add
sequence positions, not feature dimensions; no feature columns are removed.

Request the tokenizer's attention mask and special-token mask explicitly, with
padding enabled and truncation disabled. For batch item b and token position t:

```text
A[b,t] = attention_mask[b,t]       # 1 for actual input, 0 for padding
S[b,t] = special_tokens_mask[b,t]  # 1 for special tokens, 0 otherwise
R[b,t] = bool(A[b,t]) AND NOT bool(S[b,t])
n[b]   = sum_t R[b,t]
E[b,d] = sum_{t where R[b,t]} H[b,t,d] / n[b]
```

Both masks are mandatory. The attention mask alone includes start/end tokens;
the special-token mask alone must not be relied upon to exclude padding.
Use logical operations, validate mask shapes and binary values, and require
alignment with the hidden-state token axis. Remove `special_tokens_mask` from
the model input dictionary if the encoder does not accept that argument, but
pass `attention_mask` into the model as well as using it for pooling.

For every window, require `n[b] == windows.tsv.length == len(window_sequence)`
and `n[b] > 0`. Verify one-to-one residue tokenization, including residue-token
order and the selected noncanonical-residue policy; a matching count alone
does not prove token identity. Reject silent truncation, unexpected special
symbols in sequence text, or tokenization changes that break this contract.
Do not silently skip unknown residue tokens: specify and test their handling
according to the supported alphabet below before processing references.

Use float32 model parameters, inference, pooling accumulation, and outputs in
this initial version. Disable autocast and CUDA TF32. Seed Torch from the
recorded setup seed and run in evaluation/inference mode; dropout is disabled.
The inference seed is execution provenance, not an embedding identity input,
because outputs must be deterministic within the numerical contract without
stochastic model operations. Reduced-precision inference requires a later
versioned contract, not an implicit runtime optimization.

Pool selected residue rows in float32. Exclude masked rows by selection or by replacing them with zero
before summation; multiplying by zero does not safely exclude NaN/Inf values.
Reject nonfinite selected residue states or pooled outputs. Run the encoder in
evaluation mode with gradients disabled. Return one finite 640-vector per
window; padding, batch composition, and batch order must not change it beyond
documented floating-point tolerances.

The embedding manifest must record model/tokenizer IDs and immutable revisions,
selected layer, hidden dimension, window length, residue-token policy, both-mask
rule, pooling algorithm version, accumulation dtype, inference dtype/device,
and prepared-data identity. Reference behavior: Meta's
[extraction script](https://github.com/facebookresearch/esm/blob/main/scripts/extract.py)
averages residue rows `1:L+1`; the
[model configuration](https://huggingface.co/facebook/esm2_t30_150M_UR50D/blob/main/config.json)
specifies 30 layers and 640 hidden dimensions. Use the pinned revision above
for executable assets; mutable links explain provenance only.

### Mandatory pooling tests

- **Independent analytical fixture:** use hand-assigned hidden vectors and masks
  with a manually calculated mean. Give BOS, EOS, and padding extreme values so
  including any of them causes a clear failure. Assert the complete output
  vector, not just its shape.
- **Mask truth table:** exercise all `(A,S)` pairs, including padding marked
  nonspecial, and confirm only `(1,0)` contributes. Test invalid mask shapes,
  nonbinary masks, zero selected residues, and mismatched expected lengths.
- **Masked-value robustness:** changing excluded rows, including to NaN/Inf,
  must not change pooling. Nonfinite selected rows must fail clearly. Verify
  float32 accumulation with an independently computed float32 reference.
- **Real tokenizer boundaries:** test one residue, mixed short lengths, and
  lengths 1021 and 1022; assert exact residue token IDs/order, mask positions,
  two special-token positions, and expected token counts. Reject over-limit
  windows before tokenization; no implicit truncation is permitted.
- **Noncanonical inputs:** test the explicit accepted/rejected policy for X, B,
  Z, U, O, and unknown tokens. Reject injected special-token text rather than
  treating it as a protein residue sequence.
- **Batched padding integration:** embed the same sequence alone, in an
  unequal-length padded batch, in a different batch order, and at different
  batch sizes. Require numerically equivalent vectors after reordering and
  identical selected-residue counts. Pass the attention mask to the encoder.
- **Direct slicing oracle:** for unpadded ESM-2 examples, compare mask pooling
  with the independent final-layer slice `H[b, 1:L+1, :].float().mean(0)`.
  Confirm a 640-vector, never a vocabulary-logit vector or 638-vector.
- **Window inheritance:** N-/C- inputs retain prepared IDs, parent labels, and
  independent window-fold assignments. Verify both are pooled over exactly 1022
  residue tokens.

Run analytical and tokenizer tests offline in routine CI. Use a deterministic
small local encoder fixture for ordinary batching integration tests. Before
shipping the embedding feature, also run the batching and slicing checks against
the actual pinned ESM-2 checkpoint, on CPU and available GPU configurations.
Record checkpoint identity, hardware, dtypes, and measured differences. Require
`allclose(rtol=1e-4, atol=1e-5)` for actual-checkpoint batching and CPU/CUDA
comparisons; analytical fixtures must use exact or independently justified
tighter tolerances. Fail acceptance rather than silently relax the tolerance.
Report unavailable device checks. Preparation-only tests must not
load models; routine tests must not download assets or treat mocked-model tests
as real-checkpoint validation.

## Resumable embedding computation

Resume is the default behavior; no `--resume` flag is needed. A restart must
never discard validated completed work or infer completion from file existence,
array length, a timestamp, or zero/nonzero matrix values alone.

### Identity and reusable work

- Compute a preparation identity from reference digests, filtering/window/split
  configuration, and preparation algorithm/schema versions. Record canonical
  ordering and hashes of the prepared sequence and window inventories.
- Compute an embedding identity from the ordered `(window_id, sequence_digest)`
  inventory, pinned model/tokenizer identity and asset digests, layer, pooling
  version, residue policy, inference dtype, output dtype, and embedding algorithm
  version. Do not use training/validation assignments or cluster labels as
  embedding inputs. A split-only change can reuse the same vectors and attach
  the new prepared-data identity through the selection pointer; do not mutate
  a completed embedding bundle.
- Record actual device, software versions, and batch sizes per execution.
  Changing batch size is allowed on resume. Allow CPU/GPU transitions only
  within the same tested inference-dtype/algorithm contract, record mixed-device
  provenance, and apply the specified numerical tolerances. Reject incompatible
  changes rather than silently mix embeddings.
- On a matching completed bundle, verify its checksums and return without model
  acquisition or inference. On a matching partial run, verify checkpoints and
  compute only missing rows. Changed identities select separate bundles;
  unrelated complete or partial runs must not be overwritten.

### Durable checkpoint format and commit order

Use immutable, bounded row-range shards in a hidden work directory adjacent to
the final embedding bundle. Shard boundaries are fixed by the algorithm version
(64 windows per shard in algorithm v1), independent of the adjustable model batch
size. A shard may take several inference batches to complete; at most 64 completed
window vectors in the active uncommitted shard can be lost on interruption.
Only its validated publication counts as completed work; interruption may recompute the active
shard, never previously committed shards.

Each committed shard consists of a little-endian float32 (`<f4`) `.npy` array and a JSON receipt with
the embedding identity, canonical zero-based half-open row range `[start, stop)`, inventory digest, array
shape/dtype, SHA-256, and execution provenance. For each shard:

1. Compute missing windows with the required masks and pooling checks. Validate
   vector count, width, finite values, row order, and expected window IDs.
2. Write the array to a uniquely named temporary file, flush and fsync, and
   calculate its digest. Rename it into the checkpoint directory atomically.
3. Write, flush, fsync, and atomically rename its receipt **after** the array.
   The receipt is the commit marker. Synchronize directory metadata where
   supported; a valid receipt and matching array are required for reuse.
4. Record progress after the durable receipt. Keep batch/range progress logs,
   but emit `Durably committed N/N windows` only once, after successful final
   bundle publication. Do not emit it per shard or on failed finalization.

Acquire an exclusive per-embedding lock before scanning, computing, or
publishing checkpoints. Reject concurrent writers with an actionable message;
use `filelock.FileLock` with nonblocking acquisition on a persistent lock-file
path. The OS releases ownership on process death; the remaining file is not
evidence of a live lock. Never unlink or replace lock files during acquisition
or release, and do not implement PID-based lock stealing.

On restart, require the receipt root, execution metadata, and execution contract
to be JSON objects. Structurally malformed valid JSON follows the same
quarantine-and-recompute path as invalid JSON or corrupt arrays.
Validate every committed shard against the expected identity,
nonoverlapping row range, row-inventory digest, shape, dtype, finite values,
and checksum. Preserve valid shards. Quarantine invalid arrays/receipts and
recompute their ranges; log the reason. Ignore incomplete temporary files and
arrays without receipts, keeping them outside published outputs. Reject path
aliases and symlinks that escape the owned checkpoint directory.

### Finalization and interruption behavior

After all shards validate, stream them in canonical row order into a staged,
memory-mappable `embeddings.npy` and `rows.tsv`. Avoid loading the entire matrix
or all shards into RAM. Verify complete, exactly-once row coverage, finite
float32 values, 640 columns, row-index joins, and final checksums. Publish the
embedding manifest last, retaining valid shards until final publication has
succeeded. An interruption during assembly must resume assembly from the shards
without re-running the PLM. Retain verified shards as a local recovery cache after success; they are never
packaged. They may be removed only after final-bundle verification, without
affecting complete-bundle consumption.

Ctrl-C, process termination, disk-full errors, model failures, and GPU OOM must
leave earlier valid shards reusable and existing completed bundles intact.
An OOM must not mark any affected rows complete; report that a smaller
`--batch-size` can resume. Do not automatically fall back to a different
precision or silently alter windowing. `--force` archives prior complete and
partial state before intentional recomputation; it is not needed for normal
recovery. Never require users to edit progress metadata manually.

## Target sets C, background sets B, and five-fold assignment

### ProStruct reference definitions

ProStruct's `src/prostruct/cluster_sets.py` and SPEC-0027 define:

- C_k: every member of target cluster k, with no cap.
- S_j: every member of source cluster j if its size is at most 200; otherwise
  200 members sampled uniformly without replacement.
- B_k: the union of S_j over all source clusters j other than target k.

The cap is **200 per contributing source cluster**, not 200 total background
members. Select S_j once and reuse it across all target comparisons and all
features. ProStruct sorts IDs and derives independent per-source PCG64 seeds
from a SHA-256 encoding of its base seed and cluster ID; it records selected
IDs and NumPy version. Its original accession memberships form a partition,
which makes C_k and B_k disjoint automatically.

Use this fixed capped-source design here. After deduplication, filtering, and
windowing, C_k contains all emitted window records assigned to retained cluster
k. C_k is never capped. Background source clusters are the retained clusters
only; excluded clusters and unclustered FASTA accessions do not contribute.
Keep B_k fixed throughout the five C-fold rounds, without resampling it or
assigning its members a global train/validation fold.

Unlike ProStruct, PClass-Dark preserves multiple cluster labels for identical
full sequences. Excluding source cluster k alone does not guarantee disjoint
C_k/B_k. Exclude known target parent-sequence members from B_k even when another
source cluster supplies them, and store the resulting exclusion counts. Preserve
multiple labels; never use another known label as grounds to call a target
member a negative. Exact terminal-window collisions between different parents
remain reported diagnostics under the user's accepted window-level policy.

### Per-cluster C folds

The user clarified that folding is **within each C set**, not across the entire
dataset. This supersedes the previous globally balanced fold assignment.

For each retained cluster k independently:

1. Collect its unique window IDs from retained cluster/sequence membership and
   the window inventory. A short protein contributes one window and a long
   protein contributes N/C windows. Do not repeat windows for accession aliases.
2. For each window, hash the UTF-8 canonical compact JSON array
   `["pclass-dark.c-folds.v1", seed, cluster_id, window_id]` with SHA-256.
   Sort by digest bytes, breaking ties by window ID.
3. At shuffled zero-based rank r, assign `fold_id = r % 5`. Fold IDs are 0..4;
   within this cluster, fold window counts differ by at most one. Do not
   rebalance by parent, sequence identity, length, or observed feature values.
4. Store one assignment per `(cluster_id, window_id)`. A shared window may have
   different fold IDs in different target clusters; these are separate target
   classification tasks. Canonical embedding rows remain globally unique.

N/C windows need not share a fold. Identical window strings from different
parents need not share a fold. Do not enforce the superseded connected-component
or parent-group restrictions. With the default threshold, every retained cluster
has at least 10 windows and five nonempty validation folds. For custom positive thresholds below five,
allow empty validation folds: still store all five count rows, report
`empty_validation` or `insufficient_training` when appropriate, and never alter
random assignment or silently raise the requested membership threshold.

Count cluster eligibility before windowing. Record per-cluster/per-fold counts
of training/validation windows and distinct parents, as well as cross-fold
sibling and identical-window diagnostics. Changing the fold seed changes
prepared-data identity but not unchanged embedding vectors.

### Fixed background sampling: 200 windows per source cluster

The user approved a cap of **200 window records**, not parent proteins, per
contributing source cluster. Fix `background_cap = 200` for this initial version;
record it in configuration and provenance. Each eligible N/C window is a separate
sampling candidate. A long protein can contribute zero, one, or both windows.

For every retained source cluster j, let W_j be its distinct window IDs sorted
lexicographically. Define one shared sample S_j:

1. If `len(W_j) <= 200`, select every ID; do not sample with replacement or pad
   the set to 200.
2. Otherwise encode `["pclass-dark.background-windows.v1", seed, cluster_id]`
   with `json.dumps(..., ensure_ascii=True, separators=(",", ":"))`, UTF-8,
   no trailing newline. Interpret its SHA-256 digest as an unsigned big-endian
   integer. Initialize `numpy.random.Generator(numpy.random.PCG64(derived_seed))`.
3. Select 200 indices with `rng.choice(len(W_j), size=200, replace=False)`.
   Sort and persist the selected indices into the exact source W_j array
   stored in `cluster_members.json`, as defined by the compact C/B contract below.

Use a separate namespace from C-fold randomization even though both use the
same base seed. Feature values, lengths, fold IDs, and observed performance must
not influence sampling. Record NumPy version, algorithm version, base seed,
per-source candidate count, and actual selected indices. Their resolution to
window IDs is bound to the exact C file. Persist selections as the authoritative
sampled result; they must not be regenerated differently on load.

For target k, expand B_k as the set union of S_j for all j != k, then remove
any window whose full parent-sequence ID is a known member of target k. Apply
this exclusion against the full C_k, not only its four training folds. Deduplicate
by window ID after union and sort the final IDs. Do not collapse distinct IDs
merely because their window strings coincide; report such identity collisions.
Do not resample or refill after target-member exclusion or duplicate removal.
Thus 200 is an upper bound on each source contribution, not a guarantee that
200 unique eligible background windows survive for every target.

Store compact shared selections once, not thousands of expanded B lists. The
loader accepts a target ID and deterministically expands the fixed background;
use that same B_k for all dimensions and five C-fold rounds. Verify selected IDs
belong to their source cluster, have no duplicate IDs, and respect the cap.
Persist per-target raw contribution counts, unique-union count, target-exclusion
count, final background size, and SHA-256 of the final sorted window-ID list.
If a target has no eligible B, retain its C/folds but flag background status as
`empty_background`; downstream signature discovery must reject unsupported
comparisons explicitly rather than fabricate scores.

### Compact C/B storage and index integrity

Follow ProStruct's two-file design, adapting membership to windows and permitting
shared labels. Store each C window-ID array once, and each source selection once.
Do not materialize a separate background-ID array for every target.

`cluster_members.json` (C) contains:

- Schema `pclass-dark.cluster-members.v1` and preparation identity.
- Clusters keyed in lexicographic order; each value is its sorted, unique,
  nonempty window-ID array W_j. A window may appear in different cluster arrays.
- Digests of the exact `windows.tsv` and `memberships.tsv` files, the canonical
  ordered `(window_id, parent_sequence_id, window_sequence_sha256)` inventory,
  reference identity, algorithm versions, and observed membership counts.

`background_members.json` (B) contains:

- Schema `pclass-dark.background-members.v1`, matching preparation/inventory
  identities, cap, seed, sampler/version, NumPy version, and exclusion/union rules.
- `cluster_members_sha256`: SHA-256 of the **exact stored C-file bytes**.
- `source_samples`: identical cluster-key coverage to C. Value `"all"` is
  mandatory when `len(W_j) <= 200`; otherwise store exactly 200 increasing,
  unique zero-based integer indices into that cluster's C array. Booleans,
  negative indices, nonintegers, duplicates, and out-of-range values are invalid.
- Per-target compact entries specifying excluded source cluster, removal of
  every target-parent member, final B count, final ordered-ID digest, and status.
- A pair identity computed over the preparation/inventory identities, C-file
  digest, source selections, cap, seed, algorithm versions, and expansion rules.
  Hash canonical JSON with sorted object keys, `ensure_ascii=True`, compact
  separators, UTF-8, and no trailing newline. Do not include the pair ID in its
  own hash input. The prepared manifest records the pair ID and hashes of both
  actual C/B files, avoiding circular file hashes.

The loader must verify before exposing either set or dereferencing any index:

1. Prepared manifest completion status, schema versions, both file-byte hashes,
   B's binding to the exact C bytes, matching identities, and pair identity.
   Reject duplicate JSON object keys and duplicate/invalid member IDs.
2. Exact cluster-key coverage, canonical order, valid membership/window joins,
   source counts, and selection shape/range/uniqueness/cap. Cross-check C against
   memberships and the window inventory, not just self-reported JSON counts.
3. Expand each B deterministically and verify final count and ordered-ID digest,
   exclusions, and unique IDs. A reordered C array, even with the same IDs, is
   invalid unless all dependent artifacts are rebuilt and identities updated.
4. Before mapping window IDs to vectors, verify the embedding inventory identity,
   `rows.tsv` digest, unique window IDs, contiguous zero-based row indices,
   parent IDs and sequence digests, matrix row count/width/dtype, and matrix
   checksum. Join by window ID. A source-sample index is **not** an embedding row
   index; the two address spaces must never be conflated.

Use one verified immutable snapshot per operation, under the publication/read
lock where writers can replace files. Do not validate one file version and then
reopen an unverified replacement while resolving indices. On any mismatch,
raise a clear integrity error; do not guess a new order, resample, or silently
repair files. Rebuilding through setup-data is the recovery path.

Checksum binding detects corruption, stale combinations, and accidental reorder;
it is not a cryptographic signature proving authenticity against an attacker
who can replace data and every manifest together. The bundled reference and
release identities establish which snapshot the package intends to distribute.

### Rotation and interpretation

For each target cluster, the next step runs five rounds. In round f, discover
the signature using C_k windows whose target-specific `fold_id != f`, and
validate on the remaining C_k fold. Use the same B_k in every round; it is not
partitioned by other clusters' C-fold assignments. Each `(cluster_id, window_id)`
has exactly one out-of-fold target-positive score.
Gaussian target parameters and feature selection must not use the held-out
C_k fold. Any background-dependent ranking, selection, or normalization follows
the fixed-B protocol and is recorded explicitly; B is shared across rounds.
Frozen PLM embedding can be shared across rounds.

Report fold-level metrics, their mean and sample standard deviation (`ddof=1`),
valid-fold counts, positive/background counts, and pooled out-of-fold results
per target cluster.
Do not replace undefined metrics with zero. With fewer than two valid folds,
report standard deviation as unavailable. Track selected-dimension frequencies
and signature sizes to describe discovery instability. The five estimates share
training observations; their standard deviation is fold variation, not a
confidence interval or an independent-repeat standard error.

Because sibling windows and identical strings may span folds, this is
window-level internal validation and can give optimistic results for new
proteins. It is not parent-independent, identity-separated, or homology-separated
validation. Count and report parents whose N/C records span folds and exact
window strings occurring in multiple folds. These are accepted diagnostics, not
errors. Random assignment can also place related windows in one fold; do not
force them into different folds.

Calibration remains in the next step. Before implementing its performance
reporting, decide whether thresholds are fixed in advance, fitted within each
round's training folds, or tuned using out-of-fold labels. Tuning and evaluating
on the same out-of-fold labels yields calibration diagnostics rather than
independent thresholded performance. Reusing B for selection and evaluation
also means background-side metrics are conditional on a fixed development
background, not an independently held-out negative population. A large B reduces
sampling variation but does not remove selection bias or biological dependence.
This does not block creating the five folds or frozen embeddings in setup-data.

## Distributable artifact layout

Write completed artifacts beneath the configurable output root, initially
`work/setup-data`, rather than modifying the installed package. Use:

```text
<output-root>/
  prepared/<preparation-id>/
  embeddings/<embedding-id>/
  .checkpoints/<embedding-id>/       # Private resumable work; not distributable
```

Completed preparation and embedding bundles must be portable and distributable:
no required absolute paths, dependence on the original checkout, model cache,
checkpoints, GPU, or neighboring repositories. Reference and prepared identities
link bundles; filesystem locations can be resolved relative to a supplied data
root or installed resources.

Release builders copy verified completed bundles into
`src/pclass_dark/resources/prepared/<preparation-id>/` and
`src/pclass_dark/resources/embeddings/<embedding-id>/`. Update setuptools
package-data rules and Git LFS attributes for both trees when implementing this
layout. Include full artifact bytes in wheel/source distributions, not LFS
pointers or checkpoint files. Package consumers can read supplied embeddings
without installing embedding-generation dependencies or retrieving the PLM.
The release manifest must explicitly identify which prepared/embedding pair is
distributed; do not select an arbitrary newest directory by timestamp.

## Prepared output bundle

Use stable order and UTF-8 TSV with headers; nested label sets belong in the
membership table rather than ambiguous comma-separated label columns.

| File | Required content |
|---|---|
| `sequences.fasta` | One full sequence per retained global representative |
| `aliases.tsv` | Original accession, original cluster, sequence ID, representative, retained/excluded status |
| `memberships.tsv` | Cluster ID, sequence ID, representative; one row per retained unique membership |
| `clusters.tsv` | Cluster ID, original accession count, deduplicated count, retained flag, exclusion reason |
| `cluster_folds.tsv` | Retained cluster ID, fold ID, validation/training window counts, validation/training distinct-parent counts; five rows per retained cluster, including zeros |
| `folds.tsv` | Cluster ID, window ID, parent sequence ID, representative, fold ID (0..4); one row per target-cluster/window membership |
| `cluster_members.json` | Compact C: sorted per-cluster window-ID arrays, preparation/inventory bindings, counts and provenance |
| `background_members.json` | Compact B: C-file digest, per-source `"all"` or 200 sampled C-array indices, fixed expansion/exclusion rules, per-target counts/digests, pair identity and provenance |
| `background_summary.tsv` | Target cluster ID, summed source selections before union, unique-union count, target-excluded count, final B size, final ID-list digest, status |
| `windows.fasta` | Full short records and named N/C records for long parents |
| `windows.tsv` | Record ID, sequence ID, representative, parent length, window kind, start/end, length, window SHA-256; fold IDs live in target-specific `folds.tsv` |
| `window_identity.tsv` | Repeated window digest groups with record/parent IDs and target cluster and fold from the membership join, including cross-fold flags |
| `manifest.json` | Input identities, output checksums/sizes, configuration, formulas, algorithm version, software version, counts and diagnostics |

Record all exclusion reasons and counts. Preserve source cluster IDs. Report
original and deduplicated member-count distributions using the audited bins:
<10, [10,20), [20,40), [40,80), [80,160), [160,320), [320,640), >=640.

### Serialization and identity conventions

Use UTF-8, LF line endings, and one terminal newline for persisted text files.
TSVs have exactly the headers listed below, no comments, no embedded tabs or
newlines, decimal integers, and lowercase `true`/`false`. Empty optional cells
mean not applicable; do not overload zero as missing. Persist JSON with sorted
object keys, `ensure_ascii=True`, compact separators, no NaN/Inf, and a terminal
newline; canonical JSON used as a hash input omits that final newline.

Use lowercase hexadecimal SHA-256 of the exact concatenated ASCII sequence as
`sequence_id`, and the same convention for `window_sha256`. Inventory digest
input is the canonical JSON array of `[window_id, sequence_id, window_sha256]`
rows sorted by window ID. Canonical embedding order and zero-based matrix row
indices follow this same order. Source-cluster arrays remain lexicographically
sorted by window ID, independent of embedding row indices.

Exact TSV headers (tab-separated in files):

| File | Header, in order |
|---|---|
| `aliases.tsv` | `accession, original_cluster_id, sequence_id, representative, retained_cluster, retained_sequence, exclusion_reason` |
| `memberships.tsv` | `cluster_id, sequence_id, representative` |
| `clusters.tsv` | `cluster_id, original_accessions, unique_sequences, retained, exclusion_reason` |
| `cluster_folds.tsv` | `cluster_id, fold_id, validation_windows, training_windows, validation_parents, training_parents, status` |
| `folds.tsv` | `cluster_id, window_id, sequence_id, representative, fold_id` |
| `background_summary.tsv` | `cluster_id, source_contributions, unique_union_count, target_excluded_count, background_count, background_ids_sha256, status` |
| `windows.tsv` | `window_id, sequence_id, representative, parent_length, kind, start, end, length, window_sha256` |
| `window_identity.tsv` | `window_sha256, window_id, sequence_id, cluster_id, fold_id, shared_across_folds` |
| Embedding `rows.tsv` | `row_index, window_id, sequence_id, window_sha256` |

The commas above separate column names for readability; files use tabs.
Sort accession rows by accession, membership/fold/diagnostic rows by cluster
then window/sequence ID then numeric fold ID where applicable, cluster summary
rows by cluster/fold, and window/embedding rows by window ID. Identity diagnostics
are ordered by digest, cluster ID, window ID. FASTA headers contain only the
representative or window ID, with sequences wrapped at 80 characters. `kind` is
`full`, `N`, or `C`. `exclusion_reason` is empty or `cluster_below_minimum`;
`retained_sequence` can be true even when an alias's original cluster is excluded.
Fold status is `ok`, `empty_validation` when validation is empty, or
`insufficient_training` when training has fewer than two windows; the latter
has priority if both conditions apply. Background status is `ok` or
`empty_background`. Identity diagnostics include all target-membership rows of
repeated actual window strings; their cross-fold flag is computed per target.

Preparation identity hashes a canonical object containing reference file
identities, `min_clusters`, seed, five-fold algorithm version, background cap,
sampling algorithm and NumPy version, and preparation schema/algorithm versions.
Do not include output locations, timestamps, worker count, device, or batch size.
Prepared manifests include actual output-byte hashes/sizes, schema/algorithm
versions, reference/preparation/inventory/pair identities, counts, diagnostics,
configuration, and execution provenance. The manifest never hashes itself.
Include complete configuration needed for regeneration, not just its digest.

Embedding identity uses the previously specified inventory/model/pooling inputs
plus exact Torch and Transformers versions and fixed float32 inference contract.
Device and batch size are provenance only; mixed-device reuse must satisfy the
numerical contract. Record tokenizer asset digests and the weight digest in the
model contract. On complete-bundle reuse, verify this recorded supported contract
without importing Torch or contacting the Hub; do not require a local weight
copy. The originating preparation identity remains immutable provenance; a
new split configuration links to the same vector inventory without editing the
completed embedding manifest. Same-digest reference data are reusable across
filesystem locations.

The selected window-ID digest for B is SHA-256 of the canonical JSON array of
sorted final IDs. The prepared manifest includes `background_summary.tsv` and
both C/B digests; C's preparation ID is based on configuration/input identity,
not the prepared-manifest hash, so there is no hash cycle. JSON record layout
must be documented with small schema fixtures in tests, including source
selection arrays, metadata, target definitions, and pair-identity input objects.
Unknown schema versions or required-field omissions are fatal.


## Completed embedding bundle

| File | Required content |
|---|---|
| `embeddings.npy` | Memory-mappable C-order little-endian float32 (`<f4`) matrix, one row per window and 640 columns |
| `rows.tsv` | Contiguous zero-based row index, window ID, parent sequence ID, window sequence digest |
| `manifest.json` | Embedding identity, compatible prepared-window inventory identity, originating prepared-data identity, model/tokenizer provenance, pooling contract, checksums, dimensions, counts, execution history, completion status |

Keep labels and split assignments in the prepared bundle. Consumers must join
by validated window/parent IDs and verify inventory compatibility; row position
alone is not a join key. Do not normalize vectors or fit a projection here.
Supported embedding characters are exactly `ACDEFGHIKLMNPQRSTVWYXBZUO`,
verified against the pinned vocabulary. Each must map to its own nonspecial
residue token; unknown-token fallback is forbidden. No case conversion or
substitution is performed. Preparation may retain and diagnose other alphabetic
characters, but full embedding must reject unsupported inputs before acquiring
weights or computing any shard. Report affected IDs/characters and leave the
complete prepared bundle usable for review. Do not drop such sequences and
silently change cluster eligibility. Validate all inventories before expensive
inference; tokenizer files alone may be acquired first to verify token behavior.

## Reuse, publication, and resources

Reuse a complete existing output only when input digests, configuration, schema,
and preparation algorithm version match and all output checksums verify.
Use distinct identity directories for different configurations. Require `--force`
to intentionally archive and replace a selected existing bundle. Partial prepared
files are not reusable; valid embedding shards are reusable only according to
the checkpoint protocol above. A complete preparation-only run can later acquire
embeddings without rewriting prepared outputs.

Compute in a sibling staging directory, hold an output lock during publication,
write the manifest last, and restore the prior bundle on ordinary replacement
failure. Reject input/output overlap and unsafe output aliases. Keep raw bundled
resources immutable. Bound memory and work queues; avoid copying the complete
sequence corpus into every worker. Parallelism is optional when overhead exceeds
benefit, and serial execution remains available.

Format CLI log records as `[LEVEL] message` (for example,
`[INFO] Background identities: 1/3443 targets`).
Log phase progress and summary counts: verification, grouping, filtering,
windowing, split assignment, diagnostics, checkpoint acquisition, embedding,
resume verification, and publication. Long passes must report progress. Network
access is restricted to model/tokenizer acquisition when needed and permitted;
`--offline` forbids it, and `--prepare-only` never acquires model assets.

### Publication, discovery, and dependencies

Use persistent nonblocking `filelock.FileLock` files under
`<output-root>/.locks/`. Every setup invocation first acquires the root-wide
`setup-transaction` lock and retains it throughout preparation, embedding,
selection publication, and any rollback. Acquire preparation/embedding identity
locks only inside that root transaction, in that order by phase; release the
preparation identity lock before acquiring embedding ownership. Root ownership
is reentrant within the same thread/process, never across invocations.
Concurrent invocations for the same output root fail promptly with SetupBusyError;
independent output roots may run concurrently. Mutable bundle verification and
C/B readers acquire the same root guard, so rollback-eligible generations cannot
be exposed to another reader or selected by a reusing invocation.
C/B readers additionally take the preparation lock while snapshotting;
validate retained bytes without reopening replaced files. For installed
read-only resources, `load_sets` automatically uses an immutable read; explicit
`immutable=True` supports other release directories that cannot be replaced.
Mutable working roots use the root transaction guard for both prepared/vector
verification, plus the preparation lock for C/B snapshots. Publish whole staged
bundle directories on the same filesystem after writing their manifests. On
force, archive the old bundle under `<output-root>/.archives/` before replacement;
leave enough recovery metadata to restore it if replacement fails. Never alter
an existing complete bundle in place. Corrupt completed output fails integrity
checks unless force explicitly archives it; only private corrupt checkpoints
can be quarantined and regenerated automatically.

Write `<output-root>/selection.json` atomically after each successful invocation,
with relative prepared/embedding paths, identities, inventory digest, schema
version, and both manifest-byte digests. Prepare-only selections have null
embedding fields; failure must not replace a previous successful selection.
This pointer selects artifacts explicitly, never by modification time. Release
builders copy its verified referenced bundles and selection file into package
resources and verify relative-path resolution after installation. Selection
publication completes before root ownership is released; rollback also finishes
before release on failure. A later successful invocation may update selection,
but it cannot reuse a bundle while its publisher could still roll it back.
A force rebuild preserves existing verified embedding shards until their own
archive/rebuild phase. Prepare-only force never modifies embedding bundles.
An ordinary downstream or selection-publication failure restores force-replaced
completed bundles from their archives; verify the replacement manifest digest
before rollback so another writer's output is never overwritten.

Implement reusable logic in `setup_data`, `reference_data`, `cluster_sets`, and
`embeddings` modules with thin CLI registration. Add `filelock` as a core
requirement; declare `torch`, `transformers`, `huggingface-hub`, and `safetensors`
in the embeddings extra, with tested compatible version constraints. Only load
them when new inference or asset acquisition is required. Preparation, C/B
loading, and completed-vector consumption must work without that extra.
Support Python 3.10+: do not use newer-only APIs without compatibility handling.
Add the pinned tokenizer's small asset fixtures for offline unit tests with
upstream provenance. Model weight binaries are external cached build inputs,
not part of distributable prepared/embedding bundles.

## Required tests and acceptance

- CLI and library defaults/options; preparation-only execution without profunc
  or model dependencies; missing/LFS-pointer/corrupt references and malformed
  schemas. Offline embedding must use only local assets and fail clearly if
  absent. Reuse of completed embeddings must not import model dependencies.
- Same sequence under different accessions, differing FASTA wrapping, short
  differences that must not deduplicate, noncanonical residues preserved,
  unclustered extras, digest collision handling, and deterministic representatives.
- Cross-cluster identical groups retain all labels and one stored sequence;
  representatives remain traceable after filtering their source cluster.
- Threshold boundaries: 9 excluded, 10 retained; apply after deduplication and
  before windows/splits. Empty retained dataset must fail clearly.
- Lengths 1021, 1022, 1023, 2043, 2044, and 2045: IDs, exact slices,
  coordinates, overlap, omitted centers, duplicate terminal windows, and collisions.
- Seeded per-C hash-shuffle folds: exactly five IDs per cluster, sizes differing by at most
  one, every window assigned once, fixed-seed/input-order/worker invariance,
  fixtures with N/C records in the same fold and different folds, and identical
  window strings permitted across folds. Test custom thresholds below five,
  empty folds, and explicit unavailable counts/statuses. Check
  shared windows may have different fold IDs in different target clusters, and
  every target/window membership rotates once. Verify fixed B across rounds,
  the per-source cap, target-member exclusion, union deduplication, reproducible
  selection, and compact background expansion without storing K full copies.
- Integrity fault tests: swap C/B from different runs; reorder or mutate C;
  alter B indices; use boolean, repeated, negative, and out-of-range indices;
  corrupt manifests or inventories; reorder embedding rows; mismatch matrix
  shape/content; and replace files during loading. Assert rejection before
  dereferencing/scoring. Include positive round-trip and ID-to-row join tests
  proving that local source indices never address the embedding matrix directly.
- Identical output data across input ordering and worker counts; independently
  hand-calculated fixtures for counts, bins, and split assignment.
- Reuse, changed parameters, corruption, lock contention, archive/rollback,
  output overlap, interrupted publication, and paths outside the checkout.
  Pause a force rebuild after bundle publication but before selection; prove
  concurrent reuse and mutable readers fail busy, rollback restores old digests,
  and a subsequent invocation succeeds. Test root reentrancy, cross-process
  exclusion, independent roots, and read-only release access.
- Implement the mandatory pooling tests above and checkpoint fault-injection
  tests: interruption before/after array and receipt publication, incomplete
  files, corrupted shards, reordered/missing/duplicate rows, changed input or
  checkpoint identity, concurrent writers, stale-lock recovery, OOM, disk-full,
  batch-size changes, and final assembly interruption. Instrument model calls
  to prove valid committed rows are not re-embedded. Compare resumed output
  with an uninterrupted deterministic fixture and validate real-model numerical
  equivalence within recorded tolerances. Test split-only vector reuse.
- Build source/wheel distributions containing a small complete prepared and
  embedding fixture. Install outside the checkout and verify resource checksums
  and joins without checkpoint/model files or generation dependencies. Assert
  hidden resumable state is excluded from distributions.
- Run the full relevant pytest suite and an end-to-end preparation run on the
  bundled snapshot. Reconcile observed counts with the audit; validate real
  embeddings, interrupted-run recovery, and packaged artifact access. Record
  timings/resources and update this spec, README examples, and progress log
  together. Full embedding production runs require their own execution record;
  synthetic tests alone do not demonstrate checkpoint/model compatibility.

## Implementation readiness and boundary

The setup-data contract is implemented. Model identity, accepted
residues, float32 inference, pooling masks, fold and background algorithms,
portable outputs, integrity binding, and resumable commit rules are specified.
Acceptance evidence is recorded below. Full-dataset production embedding and
GPU numerical validation remain outside the completed validation run.

The user-approved scope is bundled references; exact deduplication; default
minimum unique cluster membership of 10; 1022-residue N/C windows; five seeded
window-level folds separately within each C; fixed B selections capped at 200
windows per source cluster; distributable artifacts; and resumable embedding.
Signature discovery, calibration, and final performance interpretation belong
to the next step. Its threshold-tuning methodology is deferred and does not
block setup-data. Setup-data implementation is complete; signature discovery
and calibration are not implemented by this change.


## Implementation and validation — 2026-10-25

Implemented in `setup_data.py`, `reference_data.py`, `cluster_sets.py`,
`embeddings.py`, and `artifacts.py`, with the Typer command in `cli.py`.
Packaging includes prepared/embedding resource patterns and corresponding LFS
rules. The full production reference files were not modified.

Validation completed:

- 48 automated tests passed with the real-checkpoint tests enabled; one GPU
  comparison was skipped because no CUDA-compatible device was visible.
- Ruff and whitespace checks passed. Analytical pooling fixtures, tokenizer
  layout, a small local encoder, and the pinned actual ESM-2 CPU checkpoint were
  tested. Actual-checkpoint comparisons included direct residue slicing,
  unequal padding, reordered batches, 1022 residues, and X/B/Z/U/O inputs.
- An actual ESM-2 interruption check on 66 synthetic windows committed 64,
  restarted at a different batch size, and computed only two remaining windows.
  Maximum difference from an uninterrupted run was approximately 9.54e-7,
  within rtol=1e-4/atol=1e-5. This validates computation, not biological accuracy.
- Offline preparation of the entire bundled snapshot retained 3,443 clusters,
  380,578 distinct full sequences, 381,627 cluster/sequence memberships, and
  393,548 window records. All audited size bins reconciled. Verified reuse took
  about 18.35 seconds in this environment.
- A staged source distribution and wheel included the actual reference snapshot
  plus a small synthetic prepared/embedding fixture. Installed consumption from
  outside the checkout verified compact sets and vectors while imports of all
  model-generation libraries were blocked; installed resources required no locks
  or writes. Synthetic fixture bundles are validation artifacts, not shipped
  production biological data.

Evidence and identities are in `docs/research/notes/2026-10-25-setup-data.md`
and its linked JSON records. Full preparation is available locally under
`work/setup-data/selection.json`; the real-model resume fixture is under
`work/real-model-resume/`. These working directories are ignored by Git.
