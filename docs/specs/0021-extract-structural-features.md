# SPEC-0021: Extract 19 sequence and predicted structural features

- Date: 2026-11-06.
- Status: **Implemented and validated — experimental; full-reference extraction and ML evaluation not run.**
- Entry point: `scripts/extract_features.py`.
- Parent: [SPEC-0020](0020-structural-prediction-models.md).
- Purpose: prepare labeled features for an initial machine-learning experiment
  assessing whether accepted `infer` candidates can be ranked more accurately.

## Scope and scientific boundary

Extract one 19-dimensional vector per filtered, unique full protein sequence,
using the same operation for reference proteins and eventual queries. Export a
labeled TSV with `accession_id` first, 19 ordered feature columns, and `cluster_id`
last. This step fits no model or preprocessing, assigns no new functional labels,
and changes neither infer acceptance nor B/C evidence. Training, model selection,
held-out reranking evaluation and the `rerank` command need a subsequent spec.
A labeled feature table alone does not demonstrate reranking improvement.

Use only local PClass-Dark implementations and packaged heads; ProStruct is a
method/provenance reference, not a runtime dependency. Reference behavior is
`../ProStruct/src/prostruct/features.py` and ProStruct SPEC-0026; sequence inference
is already adapted in `pclass_dark.structure` under SPEC-0020. The ordered 19
features and formula conventions below follow those references. Meaningful
intentional differences include preservation of all retained
cluster memberships, and integrated prediction rather than mandatory saved
per-accession residue TSV inputs.

## Interfaces

The script is a thin adapter to `pclass_dark.feature_extraction` library
operations. Pure aggregation lives in `pclass_dark.structural_features`.
This remains an experimental, manually invoked script. Do not register a root
subcommand or run extraction automatically from setup-data, build-library or infer.
The script requires no accession list, FASTA or input path for its default run:
it locates the selected preparation from the imported package's shared-data
configuration, reads its already filtered/deduplicated representatives and full
sequences, and preserves the selected membership labels.

The script always writes `features.tsv` in `Path.cwd()` (the directory from which
the user launches it), with QC, identity, manifest and work/cache companions there.
Do not infer the output directory from the script/package location. Reject an
output family inside the selected preparation, setup root, shared library root,
structural model bundle or package resources; an ordinary experiment cwd or
repository root is allowed. No CLI `--output` option is
needed for this experiment; run from the desired results directory.
Include the script in the source distribution and document installed-package
requirements, following the existing experimental script conventions.

```bash
# Run from the desired experiment directory; no input arguments are required.
# Automatically use selected filtered unique accessions and their full sequences.
# Write ./features.tsv and companions here.
python /path/to/PClass-Dark/scripts/extract_features.py

# Explicit preparation and CPU execution, independent of the invoking cwd.
python /path/to/PClass-Dark/scripts/extract_features.py \
  --prepared-dir /path/to/verified/prepared/PREPARATION_ID \
  --device cpu --workers 1 --offline

# Resume the same interrupted extraction.
python /path/to/PClass-Dark/scripts/extract_features.py --resume --offline
```

| Option | Default | Contract |
| --- | --- | --- |
| `--prepared-dir PATH` | Resolve selected preparation through `default_setup_input_dir()` and `selection.json` | Verified complete prepared bundle; this overrides default selection lookup |
| `--model-dir PATH` | Packaged structural heads | Accept a SPEC-0020 verified bundle, including a build-library companion |
| `--device TEXT` | `auto` | `auto`, `cpu`, `cuda`, `cuda:<index>`; report device actually used |
| `--offline` | false | Forbid ESM checkpoint downloads; local preparation/models are always required |
| `--window-batch-size INT` | 8 | 1..8; SPEC-0020 forward-window memory bound |
| `--workers INT` | 8 | 1..8 requested CPU aggregation workers, further limited by available CPUs; 1 is serial |
| `--resume` | false | Reuse compatible, checksum-verified completed sequence records |
| `--force` | false | Archive and replace a completed output family; incompatible with `--resume` |

No independent FASTA or arbitrary accession-list flag in this first version:
selected filtering/deduplication and membership are authoritative. A different
cohort requires an explicit verified preparation, rather than silently changing
which proteins or labels enter a run. Paths supplied explicitly are cwd-relative;
default input resolution follows package/repository/cache rules, not cwd.

Python API, with equivalent default behavior and no subprocess invocation.
Its explicit `output` argument supports programmatic integrations; the experimental
script passes `Path.cwd() / "features.tsv"`:

```python
from pclass_dark.feature_extraction import extract_features

result = extract_features(
    prepared_dir=None, output="features.tsv", model_dir=None,
    device="auto", offline=True, window_batch_size=8,
    workers=1, resume=False, force=False,
)
# result: feature/QC/index/manifest paths, extraction ID, unique sequence count,
# membership row count, retained cluster count, reused/computed counts, devices.
```

Expose the following concrete, frozen dataclasses (Path fields are absolute):

- `FeatureSummary(values, qc)`: `values` is a tuple of exactly 19 float-or-None
  entries in the feature order below; `qc` is a `FeatureQC` record. Use None for
  undefined values internally. These records must be spawn-pickleable.
- `FeatureQC`: `length`, `canonical_count`, `noncanonical_count`,
  `predicted_residue_count`, `surface_count`, `buried_count`,
  `rsa_out_of_range_count`, `scd_status`, `scd_method`, `undefined_features`,
  `prediction_status`. Counts are nonnegative integers; undefined_features is an
  ordered tuple of feature names. Accessions/sequence IDs/device metadata are
  attached by the coordinator, not used to compute descriptors.
- `FeatureExtractionResult`: `tsv_path`, `qc_path`, `sequences_path`,
  `manifest_path`, `extraction_id`, `sequence_count`, `membership_count`,
  `cluster_count`, `computed_sequences`, `reused_sequences`, `devices` (tuple of
  device strings), and `reused_output` (boolean). This is returned only after
  complete output verification.

`summarize_structural_features(sequence, prediction) -> FeatureSummary` performs
no memberships, IO or model loading. Require a nonempty uppercase sequence of
ASCII letters or `*?-` (the generic predictor alphabet), and a SPEC-0020
`StructuralPrediction` with the same sequence and supported bundle ID, arrays of shape (L,), real floating RSA,
and string Q3. Promote raw RSA to float64 for calculation, require finite values
and C/E/H at canonical positions, and np.nan/`NA` at unknown positions. Reject
infinity or other Q3 placeholders. Validate even for C=0, rather than accepting
an unrelated/malformed prediction. This pure function must import without torch;
use TYPE_CHECKING annotations or the computation-free structure module.
Imports must not parse arguments, spawn workers, load models or download assets.
Invalid inputs raise ValueError/IntegrityError, IO failures propagate, and model
failures remain RuntimeError. CLI errors exit 1, argument errors 2, interruption
130; never report a partial table as successful.

## Authoritative inputs and row identity

Default resolution is exactly:

1. `root = default_setup_input_dir()`; never search cwd or neighboring repositories.
2. Read `root/selection.json`, require schema `pclass-dark.selection.v1`, a
   lowercase 64-hex preparation_id and `prepared == "prepared/" + preparation_id`.
3. Resolve that bundle with safe relative paths. Match its manifest SHA-256 to
   `prepared_manifest_sha256`, schema to `pclass-dark.prepared.v1`, status to
   complete, and ID to preparation_id. An embedding selection is not required;
   `embedding=null` is valid and saved pooled matrices are never opened.

For `--prepared-dir`, validate that exact bundle without consulting a selector.
An unavailable/invalid local default selection is an error, not a reason to
silently choose some older preparation. Use the installed bundled-data fallback
only when `default_setup_input_dir()` itself selects it under existing path rules.

Use `verify_bundle(path, PREP_SCHEMA)` to stream-check every manifest-listed file
hash/size, and recompute prepared identity exactly as setup_data.verify_prepared:
`identity({"reference": reference_identity(m["reference"]),
"configuration": m["configuration"]})`. Then perform the streaming extraction
joins below. **Do not call verify_prepared as the production indexing path**:
it eagerly materializes FASTA/windows/folds and is a small-fixture validation
oracle here, not a bounded implementation. This extractor checks required sequence,
alias, cluster and membership semantics; it does not recompute unused discovery
fold assignments. All unused files are still checksum-validated as bundle members.

Under the mutable setup root's transaction guard, copy normalized full sequences,
all alias metadata, cluster rows and memberships into the output-local SQLite
snapshot described below. Standard `<root>/prepared/<id>` bundles use `<root>`
as guard root; an explicit copied bundle elsewhere uses its containing directory.
Packaged immutable inputs need no lock-file writes. Hold guards only for validation
and snapshotting, not the multi-hour inference. Read the selector before/after
snapshot under its root guard, and recheck source manifest/file hashes after
streaming to detect noncooperating changes. Resume validates against the original
snapshot identity and current requested selection; a moved default selection
requires --prepared-dir for the old bundle or a fresh --force run.

Validation may create operational lock files and a private uncommitted work
snapshot, but may not download/load ESM, start workers, replace completed reports,
or change source data. Only mark the SQLite snapshot ready after all joins and
counts validate. Afterward extraction depends on that snapshot, not live inputs.

Read these files from the same preparation:

- `sequences.fasta`: normalized full sequences keyed by representative accession.
- `memberships.tsv`: exact header `cluster_id`, `sequence_id`, `representative`;
  one record per retained sequence/cluster relation.
- `aliases.tsv`: original accession/cluster, full sequence hash, representative,
  retention flags. Verify representative/alias joins; never expand all aliases
  into duplicated training sequences.
- `clusters.tsv` and manifest: retained-cluster and expected-count validation.

The sequence ID is SHA-256 of the original normalized ASCII full sequence,
including noncanonical letters. Prepared FASTA must already be uppercase ASCII
letters (the reference reader does not accept `*?-`); do not silently normalize
malformed input or broaden it to the general predictor's alphabet. Stream with
read_fasta(check_stable=True) and reject lowercase before hashing. Set its
max_header_bytes=4096 and max_sequence_length=10_000_000, rejecting oversized
records before joining them; these are recorded engineering limits, not biological
filters. A limit failure aborts, never truncates or skips a protein. Use the same
10,000,000-residue admission limit before allocating predictions in the extractor;
the reusable summary/predictor APIs need not acquire this exporter-only limit.
Validate one full sequence per representative,
unique sequence hashes/strings, exact membership joins, nonempty sequences,
nonempty IDs, and absence of tabs/newlines in exported IDs. Reject duplicate
memberships, missing/extra retained sequence records, inconsistent representatives
or retention flags, and mismatches with the verified preparation counts.
Do not fetch replacement sequences from UniProt/profunc or redo filtering.

Validate all headers exactly against setup_data.HEADERS; flags must be literal
`true`/`false`. Cluster retained status must agree with the stored unique-sequence
count and configuration.min_clusters. The distinct retained alias pairs
(original_cluster_id, sequence_id) must equal memberships. Every membership's
representative must equal its sequence FASTA ID and alias mapping. Every retained
sequence must be reachable from a retained alias/membership. Excluded aliases may
refer to sequences absent from retained FASTA; do not reject those expected rows.
A representative can originate in an excluded cluster yet represent a sequence
retained through another cluster: do not require its own original cluster to be
retained. Require retained_sequence iff the hash exists in retained FASTA,
retained_cluster iff the original cluster is retained, and a consistent
representative per alias sequence-hash group (lexicographically smallest original
accession, matching preparation). Check original accession/cluster totals and each cluster's original
accessions/distinct sequence counts against aliases; compare retained sequence,
membership and cluster totals with manifest counts. Repeated sequence hashes with
different strings are an integrity error. IDs are nonempty printable ASCII without
whitespace; never use accession strings as filenames.

**Row unit:** one `(representative accession_id, cluster_id)` membership. Compute
once per sequence hash, then repeat its identical feature vector for every retained
cluster membership. `accession_id` is not necessarily unique in the final table;
the pair is unique. Do not choose an arbitrary first cluster or concatenate cluster
IDs into one cell. Deterministic ordering: accession_id, then cluster_id, ascending
lexicographically. Preserve IDs exactly rather than reindexing clusters.

Current selected preparation (a reference snapshot, not a hard-coded constraint):
`8e5994726cb958fce5d2396835f9a3767da897007110a9fda4f2e3b06cccbe4e`;
380,578 unique sequences, 381,627 memberships, 3,443 retained clusters. Thus expect
380,578 feature computations and **381,627 main-table rows**, with 21 columns.
Other valid preparations derive expectations from their verified content.

## Sequence -> windows -> embeddings -> predictions -> features

1. Resolve the verified sequence/membership snapshot and structural model bundle.
   Copy the small complete head bundle into work/models, verify the copy and
   source hashes after copying, then use that immutable local copy for inference.
   Do not duplicate the ESM checkpoint or create per-membership sequence copies.
2. Reuse one `StructuralPredictor` for pending sequences, lazily loading the pinned
   ESM checkpoint once. Add its public `prepare()` method to initialize the runtime
   and return the execution metadata specified below, without a dummy prediction;
   it must share `_runtime`'s existing device/fallback behavior. Do not expose or
   use private model internals in the script. Entirely unknown cohorts skip
   prepare() and ESM entirely. Use the frozen SPEC-0020 model and normalization unchanged.
   The ESM identity is `facebook/esm2_t30_150M_UR50D`, revision
   `a695f6045e2e32885fa60af20c13cb35398ce30c`, layer 30, float32, width 640.
3. Window each full sequence with `structural_windows`: maximum 1021 residues,
   minimum overlap 100, distributed terminally anchored starts. Select each
   residue's most-central window, with earlier-start ties. Cover every position
   exactly once in the assembled predictions; do not average overlapping outputs.
4. Extract residue embeddings, remove special/padding tokens, and apply the two
   frozen task-specific normalized heads. Exposure is raw ALL contextual exposure
   fraction (ALL CEF), despite the historical `rsa` name; no clipping. Q3 is raw
   C/E/H argmax, no island smoothing. Noncanonical positions use X context and
   receive missing predictions, preserving original coordinates and full length.
5. Aggregate on the complete original sequence, not individual windows. Compute
   sequence-only features from its original residues, not the X-substituted context.
   Cache one full-precision feature/QC record per unique sequence, then join labels.
6. Write deterministic main/QC/index tables and a checksum-bound completion manifest.

Dark's existing 1022-residue N/C windows and saved pooled embeddings do not satisfy
this representation contract. Long-protein centers must not be omitted. Do not
recover residue states from pooled vectors, apply heads to pooled vectors, or
average already aggregated window-level ratios. Existing unrelated ProStruct
feature tables are not automatically trusted/reused by this exporter.

## Exact 21-column output contract

UTF-8 TSV, header, LF newlines, no dataframe index or comment lines. Order is
exactly the numbered table below between first column `accession_id` and last
column `cluster_id`. Numeric features use float64 aggregation and fixed `.3f` serialization (exactly
three decimal places); normalize rounded negative zero to `0.000`. Undefined
values are literal `NA`, never empty, NaN or infinity. Identifiers and integer
QC/index counts retain their original representation. Never round intermediate
predictions or cached receipts. Legacy `.17g` reports remain verifiable using
their recorded precision. An in-place export conversion records its serialization
precision and source table checksum in manifest metadata, preserving the original
computation identity; publication archives the prior report family. Do not promise
bitwise identity across devices; record the actual execution configuration.

Notation: L is original full length; C is canonical residue count, alphabet
ACDEFGHIKLMNPQRSTVWY. A = D/E positions; B = K/R/H positions; P = K/R positions;
Z = D/E/K/R positions. S is canonical positions with predicted ALL CEF >0.05;
T is canonical positions with ALL CEF <=0.05. Compare float64-promoted raw
predictions against Python float `0.05`, never float32-array weak-scalar comparison.
For example float32(0.05) promoted to float64 is approximately 0.050000000745 and
is surface; a true float64 value 0.05 is buried. This follows ProStruct's scalar
float parsing rather than a threshold rounded down to float32. H3/E3/C3 are canonical positions
with raw Q3 H/E/C. Histidine participates in basic composition but not the fixed
charge proxy. Unknown positions remain in L, but not C/S/T/Q3 counts.

| # | Feature column | Definition |
| --- | --- | --- |
| 1 | `ln_length` | natural log(L) |
| 2 | `acid_fraction` | count(A)/L |
| 3 | `base_fraction` | count(B)/L |
| 4 | `surface_acid_over_all` | count(S intersect A)/L |
| 5 | `buried_acid_over_all` | count(T intersect A)/L |
| 6 | `surface_base_over_all` | count(S intersect B)/L |
| 7 | `buried_base_over_all` | count(T intersect B)/L |
| 8 | `q3_h_fraction` | count(H3)/L |
| 9 | `q3_e_fraction` | count(E3)/L |
| 10 | `q3_c_fraction` | count(C3)/L |
| 11 | `net_charge_balance` | (count(P)-count(A))/L |
| 12 | `charged_fraction` | count(Z)/L |
| 13 | `histidine_fraction` | count(amino-acid H)/L |
| 14 | `surface_fraction` | count(S)/L |
| 15 | `surface_acid_over_surface` | count(S intersect A)/count(S) |
| 16 | `surface_base_over_surface` | count(S intersect B)/count(S) |
| 17 | `surface_charged_over_surface` | count(S intersect Z)/count(S) |
| 18 | `mean_kyte_doolittle_hydropathy` | sum of canonical KD scale values/C |
| 19 | `sequence_charge_decoration` | SCD below |

Keep the original Kyte–Doolittle scale as an explicit, tested local constant:
A 1.8, C 2.5, D -3.5, E -3.5, F 2.8, G -0.4, H -3.2, I 4.5, K -3.9,
L 3.8, M 1.9, N -3.5, P -1.6, Q -3.5, R -4.5, S -0.8, T -0.7,
V 4.2, W -0.9, Y -1.3. No scaling or sliding-window averaging.

For an entirely canonical sequence, q_i=+1 for K/R, -1 for D/E, otherwise 0:

```text
SCD = (1/L) * sum over i<j of q_i*q_j*sqrt(j-i)
```

Preserve residue spacing and sign. Use the equivalent charge autocorrelation
with float64 FFT, zero padding to at least 2L-1, excluding lag zero. Check finite
lag counts and absolute distance <=1e-6 from integers; round only these counts.
Use power-of-two FFT size `1 << (2*L - 2).bit_length()`; multiply rFFT by its
complex conjugate and inverse-transform before taking lags 1..L-1. Use a direct
lag-sum with math.fsum on failed FFT integrity checks and record the event. Short
circuits: noncanonical -> NA, then fewer than two charged residues -> 0. Return
only finite SCD; a nonfinite fallback is an error, not a missing descriptor.
This follows ProStruct's implementation/provenance (Sawle–Ghosh Eq. 14, as cited
in ProStruct SPEC-0026); no kappa, pH-dependent charge model or fitted correction.

Edge cases and invariants:

- Empty sequences are invalid. Canonical singleton: ln_length=0, SCD=0. Fewer
  than two charged residues also gives SCD=0. Any unknown residue makes SCD NA.
- Mean hydropathy uses C, excluding unknowns; C=0 gives NA.
- No surface positions: three surface-conditional fractions are NA. Other
  structural fractions are zero where appropriate if C>0.
- C=0: all 11 exposure/Q3 descriptors are NA, sequence-only composition/charge
  fractions retain their defined zero values, hydropathy/SCD are NA. Retain the
  record with QC; do not pretend structural inference succeeded or load ESM.
- Any malformed/incomplete/nonfinite prediction at a canonical residue is an
  extraction error, not ordinary NA. Abort publication rather than omit the row.
- Q3 fractions sum to C/L, not necessarily 1. Surface+buried acidic/basic
  fractions reconcile with their sequence composition fractions. Conditional
  surface charged fraction excludes H although surface basic fraction includes H.
- Out-of-range raw ALL CEF is retained and counted in QC. The threshold is fixed
  at >0.05, with equality buried; it is an operational descriptor convention,
  not a fitted or independently validated physical surface threshold.

## Exact output-family, state and provenance schemas

For Python output `P/name.tsv` (script: cwd/features.tsv), require existing parent
P and a nonempty stem. Derive all paths in P, without creating ancestor folders:

| Role | Filename |
| --- | --- |
| main | `name.tsv` |
| QC | `name.qc.tsv` |
| sequence index | `name.sequences.tsv` |
| completion marker | `name.tsv.manifest.json` |
| nonblocking FileLock | `.name.features.lock` |
| resumable work directory | `.name.features-work/` |
| replacement archives | `.name.features-archives/<UTC-timestamp>-<uuid>/` |

Reject symlinks, input aliases (including hardlinks by inode), overlapping family
paths and protected-directory destinations before writes. Reject an output stem
that makes paths coincide with protected inputs. Distinct output families have
distinct work/lock/archive paths. Unrelated cwd files remain untouched.

QC has exactly this ordered header, one row per representative, sorted by
accession_id:

```text
accession_id sequence_id length canonical_count noncanonical_count
predicted_residue_count surface_count buried_count rsa_out_of_range_count
scd_status scd_method undefined_feature_count undefined_features
prediction_status device
```

The block lists column names separated by spaces/newlines for readability; the
actual header is one tab-delimited line. Count fields are decimal integers.
`undefined_features` is a comma-separated list in canonical feature order, or
`none`. `scd_status` is `defined` or `noncanonical`; `scd_method` is `fft`,
`direct_fallback`, `zero_shortcut`, or `not_applicable`. `prediction_status` is
`complete` (C>0) or `no_canonical_residues`; device is the actual prediction device
or `not_run` for C=0. For C=0 all structural counts are 0 and 13 features are
undefined (11 structural, hydropathy and SCD). This differs from the generic
StructuralPrediction.device default of cpu for a query that ran no ESM.

Sequence index header: `accession_id`, `sequence_id`, `length`, `membership_count`.
Use the same representative ordering and one row per unique full sequence. No
metadata column is added to the requested 21-column main table.

Manifest schema `pclass-dark.structural-features.v1` has required fields:

- `status="complete"`, `id`, `identity_inputs` as defined below.
- `files`: only the three TSV names, each with sha256, size_bytes, rows and ordered
  columns; rows excludes the header. The marker does not hash itself or depend on work database byte layout.
- `inputs`: preparation ID, manifest hash and input-file hash/size inventory;
  model bundle ID/manifest hash and model tensor SHA-256; pinned ESM asset hashes.
- `counts`: unique_sequences, memberships, retained_clusters, with_noncanonical,
  no_canonical, with_undefined_features, rsa_out_of_range_residues, scd_fallbacks.
- `execution`: requested CLI/API configuration, invocation/timestamps, versions,
  code file hashes, device/runtime fingerprint or `not_run`, admitted workers,
  numerical thread limits, queue limits, elapsed wall time, computed/reused counts
  and ordered warning summaries. Output verification checks all counts/joins.

Record scientific constants and output policies in identity_inputs, not only
free-form prose. identity_inputs contains keys `contract_inputs` and
`execution_metadata` (with their full objects); it must recompute contract_id, execution_fingerprint and extraction_id exactly by the
rules below; id is that extraction_id. `execution_metadata="not_run"` for C=0
cohorts. Include both derived IDs as explicit manifest fields for inspection. Capture code SHA-256 for feature aggregation/orchestration,
structure, structural_models, embeddings, setup_data, reference_data, artifacts
and paths modules, plus every new internal extraction storage/publication module;
Git revision/dirty status are optional provenance, never the sole code identity.
Include Python, NumPy, SciPy, torch, transformers and safetensors versions
(`not_installed` where no model inference is needed). No learned parameters.

### Disk-backed snapshot and receipts

Work directory layout is fixed:

```text
.name.features-work/
  run.json                 # schema pclass-dark.feature-work.v1, status, identity
  state.sqlite             # coordinator is sole writer
  models/                  # verified local copy of the small frozen head bundle
  stage/                   # new complete output family; disposable before journal
  publication.json         # exists only during recoverable publication
```

SQLite uses foreign_keys=ON, journal_mode=WAL, synchronous=FULL. Tables:

- `sequences(sequence_id TEXT PRIMARY KEY, accession_id TEXT UNIQUE NOT NULL,
  sequence TEXT NOT NULL, length INTEGER NOT NULL, canonical_count INTEGER NOT NULL)`.
- `memberships(sequence_id TEXT, cluster_id TEXT, PRIMARY KEY(sequence_id,cluster_id))`
  with sequence foreign key. Sequence rows precede membership insertion.
- `aliases(accession TEXT PRIMARY KEY, original_cluster_id TEXT, sequence_id TEXT,
  representative TEXT, retained_cluster INTEGER, retained_sequence INTEGER)`;
  include excluded aliases too; do not foreign-key excluded sequence IDs.
- `clusters(cluster_id TEXT PRIMARY KEY, original_accessions INTEGER,
  unique_sequences INTEGER, retained INTEGER)`.
- `completed(sequence_id TEXT PRIMARY KEY REFERENCES sequences,
  record_json TEXT NOT NULL, record_sha256 TEXT NOT NULL)`.

Assign processing order by `ORDER BY accession_id COLLATE BINARY`; main output
joins memberships and orders accession_id, cluster_id using BINARY collation.
Avoid loading all strings/feature records into Python dictionaries. Set the
SQLite cache budget to 64 MiB and temp_store=FILE. Progress queries use SQL counts.

Before marking a snapshot ready, store run.json.snapshot_hashes for the four
input tables. For each, hash a stream of `canonical(list(row)) + b"\n"` in its
primary-key BINARY order, with exact schema column order and stored types. Resume
runs SQLite quick_check, verifies these hashes and semantic joins/counts by
streaming, and rehashes each sequence string to its sequence_id. Do not compare
SQLite file bytes: WAL/checkpoint/layout changes are not scientific identity.

A completed record is canonical artifacts.canonical JSON of: record schema
`pclass-dark.feature-record.v1`, extraction_id, sequence_id, accession_id, length,
19-element numeric/null values, QC fields and device. Counts/strings must have
exact types (bool is not an integer count); prohibit duplicate JSON keys and
nonstandard NaN/Infinity literals using artifacts.read_json/canonical semantics.
Its hash is SHA-256 of those exact UTF-8 bytes. Validate schema, identity, length/count relations, numeric ranges,
missingness, feature order and every hash before reuse. Require finite defined
values; composition/structural fractions in [0,1], net charge in [-1,1], hydropathy
in [-4.5,4.5], ln_length matching log(L), and finite signed SCD without an invented
range restriction. Recompute inexpensive sequence-only quantities from the
snapshot; compare count-derived ratios and reconciliation identities with
atol=1e-12, rtol=0 rather than requiring exact floating-point sums. Cache records contain
features/QC only; do not persist residue embeddings or predictions. Use fixed
SQLite schema user_version=1; enforce primary keys, foreign keys, positive lengths, nonnegative counts and boolean flag constraints. Commit batches
of at most 32 records or after 10 seconds since the last commit, whichever occurs
first, plus flush on graceful interruption/finalization. Abrupt death can recompute
up to 32 uncommitted sequences; only committed records count as reusable. Never
trust a partly written JSON record; SQLite transactions are the receipt boundary.

`run.json` has required schema `pclass-dark.feature-work.v1`, absolute output
family path, status, contract_inputs/contract_id, extraction_id and
execution_metadata/fingerprint (null before runtime binding), snapshot_hashes
(null while indexing), snapshot counts, SQLite schema version, created/updated
UTC times and invocation metadata. Validate path ownership and derived identities
before recovery or --force archival. It transitions
`indexing -> ready -> running -> complete`. Write it with atomic write_json. An interrupted `indexing` snapshot has no reusable computations:
on --resume discard/rebuild only that uncommitted snapshot after validating its
ownership/schema; do not delete arbitrary unknown files. Ready/running snapshots
are immutable except completed receipts and execution/progress metadata. Schema
or database corruption is fatal with an explicit error; --force may archive owned
work, but must not silently erase an unrecognized directory. On --force archive
any recognized prior work state (including complete work) before starting its
replacement; a prior valid public report remains at its original paths until
publication. Archive only when run.json path ownership/schema are valid.

### Identity and execution compatibility

Before runtime resolution, compute `contract_id = identity(contract_inputs)`.
contract_inputs contains preparation ID/manifest hash and three authoritative table
hashes (FASTA, aliases, memberships), feature schema/formulas/KD constants/SCD
algorithm/threshold/missingness/precision, model/checkpoint identities, relevant
code hashes/versions, window_batch_size and seed 42. Cluster and all other input
hashes remain bound through the preparation manifest. Absolute paths, timestamps,
output stem, resume/force/offline flags and aggregation worker count do not affect
this mathematical contract.

For a cohort with canonical residues, `StructuralPredictor.prepare()` returns a
public dictionary with actual device, EsmBackend.contract, torch version,
torch.version.cuda/hip, device name (CPU platform.machine/processor fallback;
CUDA name/capability), actual numerical thread count, deterministic-algorithm flag,
TF32 settings and seed. Normalize actual device to `cpu` or `cuda:<index>` based
on the model parameters' device, even if the request was `cuda`. Use that same
canonical device string in QC/receipts; unknown-only records use not_run.
Include window_batch_size and Python/NumPy versions in
`execution_fingerprint = identity(metadata)`. Validate this contract against the
pinned model, revision, assets, width/layer and float32 policy. Do not include
unstable timestamps, free memory or process IDs. This public method is an explicit
small SPEC-0020 API extension required by this implementation, with tests.
prepare() must execute metadata capture under self.cpu_threads numerical limits,
using the same context as predict(), and return a fresh JSON-serializable copy.
Repeated prepare() reuses the initialized model and reports the same fingerprint;
it never forwards a dummy sequence. Keep the declared return type
`dict[str, object]` with the exact fields above.

Compute `extraction_id = identity({"contract_id": contract_id,
"execution_fingerprint": execution_fingerprint})`; use
execution_fingerprint=`not_run` for an entirely unknown cohort. Bind once before
committing any canonical prediction-derived receipt. All-unknown records within
a mixed cohort still belong to its extraction ID but use device=not_run. Determine
whether the snapshot has any canonical residues in SQL, and initialize/bind the
mixed-cohort runtime before committing any records, including all-unknown ones.
For an entirely unknown cohort synthesize the structurally missing prediction
arrays without constructing StructuralHeads/Predictor; no optional model imports
are needed beyond installed distribution metadata. No per-sequence silent
device switching: if the chosen device fails mid-run, stop;
initial automatic GPU initialization may fall back to CPU as in SPEC-0020.

A resumed incomplete canonical run must initialize/verify its runtime and match
the stored execution fingerprint before using receipts or launching workers.
Changing --workers is allowed only if the admitted inference numerical thread
count and all other fingerprint fields still match; otherwise instruct the user
to restore that configuration or --force. --offline only controls asset resolution;
it does not invalidate matching content. No cross-output-directory receipt reuse
or automatic import of ProStruct tables in v1.

A complete family with --resume can be returned without torch/ESM initialization
if the requested contract matches, all table checksums/counts/schema/joins validate,
and the explicit requested device does not conflict with its recorded actual
device (`auto` accepts the recorded device; `cuda` accepts a recorded CUDA device;
`cuda:N` must match N, and cpu must match cpu). Use package distribution metadata to
check versions without importing torch. Preserve the original manifest bytes;
return reused_output=True, computed_sequences=0 and reused_sequences=N. If no
public complete family remains but every committed receipt is valid, incomplete
resume still follows runtime verification, then regenerates tables deterministically.

### CPU/GPU admission and memory bounds

One coordinator owns inference and database publication. Use the existing predictor
sequentially across sequences; do not require cross-sequence ESM batching in v1.
Window batches remain 1..8. No GPU model is loaded in a CPU worker.

Let A=max(1, os.process_cpu_count() when available, otherwise os.cpu_count() or 1).
Let W=min(requested_workers,8,max(0,A-1)). Use W spawned aggregation processes only
if requested_workers>1 and W>=2; otherwise serial aggregation and report
workers_used=1. In parallel mode cap coordinator model numerical libraries and
each worker to one thread. In serial mode use min(8,A) predictor threads. Apply
threadpool_limits inside the worker, and ensure spawn entry modules do not import
or load torch. Propagate all worker exceptions; no nested pools.

Bound submitted-but-not-consumed aggregation tasks by 2*W (serial: one). Reserve
payload bytes before predicting/dispatching a sequence: `64*L + 4096` per task,
covering sequence/prediction arrays, serialization and worker-copy allowance.
Release a task reservation only when the coordinator has consumed/validated its
result and written it to the pending database transaction; the separately bounded
32-record commit buffer contains only small feature/QC JSON. Total reserved
payload <=256 MiB. If one sequence exceeds this budget, drain all
other tasks, process that sequence serially with a warning, then resume normal
admission. Report maximum reserved bytes and oversized sequence lengths. This
bound excludes model weights, window hidden states/attention workspaces, SQLite
and O(L) SCD workspace; it is not a total RSS/VRAM promise. CPU/GPU OOM fails the
run with accession/length/device/batch context; do not silently truncate or drop
proteins or change batching during a compatible run. Support reducing
--window-batch-size for a fresh run. Use ordered final SQL queries rather than an
unbounded Python buffer for out-of-order completed results.

GPU-first auto and explicit CPU follow SPEC-0020. No throughput/speedup or GPU
agreement claim without measurement. Log input/model identities, cohort counts,
actual device and worker plan at start; progress every 1000 completed sequences
and at completion, plus errors/warnings immediately. Logs go to stderr; successful
stdout prints the main path and sequence/membership/cluster counts.

### Existing-state decisions and recoverable publication

Acquire the nonblocking family FileLock before inspecting/recovering/mutating work
or outputs; contention raises SetupBusyError/exit 1. --force and --resume together
are argument errors. Keep input/model/source validation ahead of expensive work.

| Existing state | Default | --resume | --force |
| --- | --- | --- | --- |
| No family/work | Start | Start | Start |
| Valid complete family | Refuse | Verify/reuse if matching | Compute replacement, preserve old family until publish |
| Owned ready/running work | Refuse | Verify snapshot, identity, receipts and continue | Archive work, start fresh |
| Partial family or publication journal | Refuse with recovery guidance | Recover journal, then verify/continue | Recover journal first, then replace owned work |
| Unrecognized work/partial family without journal | Fail | Fail | Fail; no deletion of unknown user files |

Decision order: recover an owned journal first, then inspect active ready/running
work, then a completed public family. Matching active replacement work may resume
while preserving a public family from an older contract; a mismatched active run
must not be ignored to reuse an older public table. A complete public family that
matches the active run can be finalized/reused without runtime initialization
once all hashes/contract checks pass. Completed work never overrides verification
of the public table. Default still refuses any preexisting owned state.

A partial family without a journal is not a recoverable committed output. A
malformed complete manifest or altered TSV is corruption: fail even under --force
until explicitly moved aside by the user. A recognizable interrupted initial
indexing snapshot is rebuilt under --resume as specified above.

Stage three TSVs and the complete manifest in work/stage, fsync them, then validate
headers, row ordering/joins, duplicate keys, feature invariants and complete input
coverage. Staging never modifies an existing public family. Before replacement,
copy the verified previous complete family to a fresh archive directory and verify
its hashes; the public old files stay readable until swaps begin.

Write work/publication.json atomically and fsync it before the first swap. Journal
schema `pclass-dark.feature-publication.v1` contains family absolute paths,
extraction_id, new/old file hash-size inventories, archive path (or null for first
publication), stage paths and state=`installing`. All journal paths must be derived
from this family, with safe basenames; never trust arbitrary stored paths. Install
main, QC and sequence index, then manifest last, via atomic per-file replacement;
fsync parent after each. Verify the entire public family, mark journal state
`committed` durably, then remove the journal. Mark run.json complete and retain
compact completed records for audit/reuse; remove only stage/temporary WAL files
that SQLite has safely checkpointed/closed. Archives are never auto-deleted.

On ordinary failure or --resume recovery: if public files exactly match the new
manifest/inventory, finalize the new family. Otherwise restore the complete old
family from verified archive, or remove only journal-owned new files for a first
publication; compare extant files against old/new hashes before overwriting or
removing. Any third-party hash or corrupt/missing required archive is an integrity
error requiring manual inspection. Keep stage/receipts so new tables can be
regenerated/retried without inference. Journal removal happens only after verified
restoration/finalization. Unexpected files outside the owned family are untouched.
Multiple paths are not one atomic filesystem transaction: all consumers must
honor the lock and verify manifest file hashes. Never publish a success marker
for missing sequences. Never edit preparation, production or C/B selectors.

## Future ML and reranking experiment requirements

Extraction is label-independent and fits no dataset statistics, so frozen features
may be computed before selecting evaluation splits. Subsequent model training must:

- Group identical sequence hashes, aliases and every repeated membership row in
  the same split; additionally control nonidentical homology across train/test.
  Existing window-level discovery folds are not suitable held-out ML splits.
- Treat multiple labels as a membership relation. A single-label classifier must
  define an ambiguity policy in its own spec; do not silently train duplicate
  identical feature vectors with conflicting supposedly exclusive labels.
- Fit imputation, scaling, feature selection, classifier/calibration parameters
  and reranking weights on training data, tune on validation, then freeze before
  evaluating held-out queries. Retain NA/QC for an explicit training-only policy.
- Generate the accepted infer candidate lists without leaking held-out queries
  into its fitted signatures/reference distributions. A classifier accuracy test
  alone is not a candidate-reranking evaluation. Record correct-cluster coverage
  among accepted candidates as the reranker's ceiling.
- Compare original B order with reranked order using top-1 accuracy and reciprocal
  rank, with counts/uncertainty and both overall and correct-candidate-present
  results. Include the eight sequence-only feature baseline. The same PLM supplies
  both methods; improved ranking must be measured rather than assumed.

## Implementation tasks and acceptance gates

1. `structural_features.py`: constants, FeatureQC/FeatureSummary, independent
   aggregation validation, KD and FFT/direct SCD; no model imports/IO.
2. `feature_extraction.py`: result dataclass, option/path validation, streaming
   verified snapshot, SQLite receipts, bounded spawn orchestration, manifest
   verifier, family locking and journaled publication/recovery. Split supporting
   internal storage/publication modules if needed; keep scientific logic separate.
3. Extend StructuralPredictor with public prepare()/runtime metadata; preserve
   existing predict() behavior, pooled embeddings and all current model identities.
   Add explicit metadata/mismatch and lazy-import tests.
4. `scripts/extract_features.py`: thin single-command Typer adapter, -h/--help,
   no required arguments, cwd/features.tsv, options above and defined exit codes.
   Export script through MANIFEST.in and add examples to an experimental extraction
   guide linked from README. Do not register pclass-dark extract-features.
5. Run acceptance suites, then a small real-model CPU pilot with an explicitly
   prepared tiny fixture (not a limited/silently sampled full run). Record source
   IDs, environment, commands, resources and artifact paths. GPU pilot when
   available. Full-reference extraction requires a separate explicit run; this
   specification defines implementation scope, not a completed experiment.

Tests must use analytical/reference oracles, not restate the implementation:

| Area | Required evidence |
| --- | --- |
| Formulas/schema | Exact 21-column main header and exact companion headers; all 19 order/definitions; float64 aggregation and three-decimal export |
| Small numeric oracle | DEKRH fixture below, KD means, singleton, independent O(L^2) charge pair sums versus FFT and injected FFT-integrity fallback |
| Threshold/missingness | True float64 0.05 buried; float32(0.05) promoted surface; out-of-range predictions; canonical/no-surface/mixed-unknown/all-unknown inputs; malformed shapes/labels/nonfinite canonical values rejected |
| Windowing | Existing ProStruct brute-force center-ownership oracle, 1021/1022 boundary and >2044 center coverage; distinguish whole-protein ratios from unweighted window means |
| Cohort | Filtered-only records; full alias joins including excluded aliases/representatives; multi-cluster sequence computed once but all labels emitted; duplicate/hash/schema/count/input-mutation rejection |
| Discovery/paths | Zero-argument default from unrelated launch cwd; no pooled embeddings required; source/package/cache fallback exactly matches paths module; all output/work files cwd-local; explicit input and Python output override; no scientific source modifications |
| Resources | workers=1 versus admitted spawn workers; bounded submitted tasks and byte reservations, oversized serial path, failure propagation and no torch/model in workers; no unbounded completed-result buffer |
| Resume | Empty/partial/all-complete receipts, checksum corruption, indexing restart, code/version/model/batch/thread/device incompatibilities, compatible worker/offline changes and complete-family reuse without model initialization |
| Publication | Fault injection before/after each TSV/marker swap and journal state update, first-publication cleanup, replacement restoration, post-marker finalization, third-party mutation rejection and lock contention |
| Interfaces/release | Library/script equivalence on tiny fixture; errors/interruption exit codes; lazy imports; three-decimal export matches rounded aggregation; full-precision receipts remain validated; script inclusion in sdist |
| Devices | CPU required; actual GPU/CPU feature/label comparisons when hardware exists with measured tolerance, otherwise explicit skip/report; mocked auto-device fallback/mid-run failure |

Analytical fixture: sequence DEKRH, exposure `[0.1, 0.05, 0.2, -0.1, 1.2]` as
float64, Q3 `[H,E,C,C,H]`. Expected vector in canonical column order:

```text
ln(5), 2/5, 3/5, 1/5, 1/5, 2/5, 1/5, 2/5, 1/5, 2/5,
0, 4/5, 1/5, 3/5, 1/3, 2/3, 2/3, -3.72,
(1 - 2*sqrt(2) - sqrt(3))/5
```

QC: L=C=5, surface=3, buried=2, out-of-range=2, predicted=5, no undefined
features. Appending X increases L to 6, retains C=5 and the same canonical
counts/hydropathy, makes SCD NA and Q3 fractions sum to 5/6. An XX query has
length 2, canonical=0, structural counts 0, exactly 13 undefined features and
no ESM initialization. Tests must not cast the threshold-equality oracle to
float32 inadvertently. For ProStruct parity, compare numeric values before its
three-decimal export, or compare rounded exports at their specified precision.

Completion requires tests for all gates above, relevant structure/embedding and
artifact/CLI/packaging regressions, synchronized docs/status/progress, and an
honest record of untested GPU/performance cases. Training models and measuring
reranking improvement remain outside this extraction implementation.

## Historical readiness review (superseded by implementation below)

**Implementation-ready as of 2026-11-06, before implementation.** Interfaces, output row semantics,
formulas, scientific precision, concrete state/receipt/manifest contracts,
resource admission, runtime compatibility and publication recovery are specified;
no unresolved scientific choice is delegated to implementation. This remains an
experimental script with automatic selected-cohort discovery and cwd-local output.

Inspected current paths/artifact/FASTA/preparation/model/predictor implementations
and ProStruct feature code. Checked exact feature order and KD constants against
that source; independently verified the analytical charge/hydropathy fixture,
21-column schema and local Markdown links. Earlier draft conclusions remain in
the chronological progress log; this revision replaces its unspecified cache and
runtime/publication details. At that review, extraction code/tests and experimental
runs had not been produced. The implementation results below supersede that
status; the earlier chronological progress entries remain intact.


## Implemented behavior and validation — 2026-11-06

Implemented the pure 19-feature aggregator, exact QC/header contracts, streaming
verified cohort snapshot, SQLite feature receipts, bounded spawn orchestration,
model/runtime compatibility, report validation, journaled publication/recovery and
thin standalone script. Supporting modules are `feature_store.py` and
`feature_reports.py`. No runtime import from ProStruct. Source packaging includes
the script and [operational guide](../feature-extraction.md); no root subcommand
or automatic setup/build/infer invocation was added. StructuralPredictor.prepare()
is implemented with fresh metadata and no dummy forward pass. Source selectors,
current production signatures and B/C behavior remain unchanged.

Implementation details/clarifications:

- SQL snapshot logical hashes are checked before reuse and again before export;
  prepared manifest byte hashes, all source member hashes, source code and package
  versions are rechecked at the relevant boundaries. Mutable state/unknown user
  files are not silently accepted or overwritten.
- Commits check the 32-record/10-second rule when completed aggregation results
  arrive, and flush on graceful interruption/finalization. An in-progress model
  forward may defer the timer check; the maximum uncommitted count remains 32.
- Runtime metadata retains EsmBackend.contract as loader provenance, including
  its generic mean-pooling API description and maximum 1022 support. The actual
  structural path is unpooled residue inference with the 1021/100 ownership policy
  recorded separately in contract_inputs.esm; pooled vectors are never used here.
- created_at/updated_at/finished_at are UTC Unix seconds. execution.elapsed_seconds
  measures validation/inference up to export staging; the pilot external wall
  measurement includes the complete command, export and publication.
- Reserved payload is an admission estimate, not a total memory limit. Actual
  model/GPU/FFT workspaces remain excluded as specified. Archives are retained;
  automatic model training and production-scale feature extraction are out of scope.

Automated command:

```bash
python -m pytest -q tests/test_structural_features.py tests/test_feature_extraction.py \
  tests/test_structure.py tests/test_embeddings.py tests/test_cli.py \
  tests/test_setup_data.py tests/test_reference_data.py tests/test_shared_paths.py \
  tests/test_shared_distribution.py
```

**193 passed, 1 skipped in 38.66 s.** GPU comparison was skipped because PyTorch
exposes no GPU; an existing AMD_SERIALIZE_KERNEL environment warning appeared.
The subsequently strengthened all-unknown test also passed separately, proving
fresh extraction and complete-report reuse with torch/transformers/safetensors
imports explicitly blocked. Ruff lint/format and whitespace/content checks passed.
Packaging tests build/inspect wheel/source archives and check script inclusion.
Fault-injection tests exercise each table/marker swap, interrupted recovery,
replacement during indexing, third-party mutations and preservation of old reports.

Independent ProStruct aggregation comparison: 15 fixtures, seed 21; all 19 defined
values agree at atol/rtol 1e-12 and NA policies agree, before ProStruct's export
rounding. Real offline pinned-model CPU pilot: four unique synthetic sequences,
five membership rows, two retained synthetic clusters. Exact table has 21 columns;
serial/parallel feature TSVs are byte-identical. Complete-report resume passed.
No full reference, classifier training, held-out performance or reranking run.

See [the reproducible pilot record](../research/notes/2026-11-06-extract-features-pilot.md)
for data, commands, code/model identities, output artifacts, timings and limitations.


### 2026-11-07 — Three-decimal export revision

Implemented fixed `.3f` numeric feature export in the shared report writer used by
`scripts/extract_features.py` and the Python API. This supersedes the original
full-precision TSV policy; aggregation and cache receipts retain full precision.
Report verification compares rounded exports against independently validated
full-precision receipts, rather than applying exact ratio identities to rounded
numbers. Legacy reports use their recorded `.17g` policy. Export-only conversion
uses explicit serialization metadata and refreshed file checksums without changing
the original computation identity; changed code/contracts still require matching
resume state or a fresh extraction.

Converted all 381,627 rows in `work/features.tsv` under the output lock using
staged, journaled publication. Original report family archived; QC/index content,
identifiers, row order and missingness preserved. See the dated progress entry.
Validation: extractor, structural-feature and structural-prediction suites:
115 passed, 1 skipped (GPU unavailable). Ruff and whitespace checks passed.


### 2026-11-07 — Complete-report reuse without work receipts

A complete report family is independently reusable after its hidden work directory
has been removed. After verifying the report family, current input/code contract
and requested device, `--resume` must return that report without creating a new
work directory, SQLite snapshot or predictor. Repeated reuse must remain possible.
When matching work state exists, retain snapshot and receipt-dependent checks.
Regression validation must delete the work directory, resume twice, verify all
report bytes are unchanged and confirm no snapshot/model initialization occurs.
Implemented the early return before work-state creation. Validation: regression
failed before the fix; extractor, structural-feature and structural-prediction
suites passed afterward: 116 passed, 1 skipped (GPU unavailable). Ruff checks,
formatting and whitespace checks passed.
