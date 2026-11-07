# Experimental 19-feature extraction

Run the standalone script from the directory where you want the results:

```bash
pip install -e '.[embeddings]'
mkdir -p work-features
cd work-features
python /path/to/PClass-Dark/scripts/extract_features.py --offline
```

No accession list, FASTA or output argument is needed. The script automatically
resolves the selected preparation through the installed package's shared-data
configuration, verifies it and reads its filtered, deduplicated full sequences,
representative accessions and retained memberships. It writes **features.tsv in
the launch working directory**. It does not register a pclass-dark subcommand or
run as part of setup-data, build-library or infer.

`--offline` requires the pinned ESM checkpoint already in the Hugging Face cache;
omit it to allow checkpoint resolution/download. The two structural heads are
packaged locally. GPU-first `--device auto` falls back to CPU during initialization.
For explicit CPU/serial execution, use `--device cpu --workers 1`.

```bash
# Resume the same extraction without recomputing validated sequence receipts.
python /path/to/PClass-Dark/scripts/extract_features.py --resume --offline

# Use another verified preparation; results still go to the launch cwd.
python /path/to/PClass-Dark/scripts/extract_features.py \
  --prepared-dir /path/to/setup/prepared/PREPARATION_ID \
  --device cpu --workers 1 --window-batch-size 1 --offline
```

Use `-h` or `--help` for options. The default is eight requested CPU aggregation
workers (further limited by available CPUs), and at most eight windows per forward
batch. One coordinator owns the models; spawned aggregation workers do not load
Torch. Queued payloads are bounded by task count and 256 MiB; oversized individual
sequences run aggregation serially. These limits exclude model weights, inference
workspaces and per-sequence SCD workspace. Lower the window batch size on a fresh
run if needed; incompatible existing receipts require --force or a new cwd.

## Output and identity

The main table has exactly 21 columns: accession_id, the 19 descriptors specified
in [SPEC-0021](specs/0021-extract-structural-features.md), and cluster_id. Numeric
features use exactly three decimal places in the TSV; calculations and cached
receipts retain float64 precision. Rounded negative zero is `0.000`; undefined
values are literal NA. Each row represents one retained sequence/cluster relation.
An accession with multiple labels has repeated, identical feature vectors in
separate rows. Computation is performed only once per sequence hash.

For the selected reference snapshot inspected during development, 380,578 unique
sequences produce 381,627 rows across 3,443 clusters. Counts are derived from the
verified inputs, not hard-coded. Keep all rows for the same sequence in the same
training/evaluation split; deduplication alone does not remove nonidentical homologs.

Other cwd-local files:

- features.qc.tsv: one row per unique sequence, prediction/missingness diagnostics.
- features.sequences.tsv: representative accession, sequence hash, length and label count.
- features.tsv.manifest.json: complete report marker, file hashes, input/model/code
  identities, runtime, counts and execution settings.
- .features.features-work/: SQLite snapshot/receipts, copied structural heads and
  publication journal/staging. Keep it to resume interrupted work.
- .features.features.lock: nonblocking output lock; concurrent writers fail promptly.
- .features.features-archives/: preserved earlier work/reports after --force.

Ordinary invocations refuse existing work/reports. --resume reuses matching data;
--force archives recognized work and computes a replacement while preserving an
older complete report until publication. They cannot be combined. Corrupt public
reports and unknown user files are never silently overwritten. An interrupted
publication is recovered from its owned journal on --resume/--force. Consumers
must honor the output lock and verify every manifest file hash; multiple files
cannot be installed by a single filesystem rename.

An incomplete resumed run must match the recorded model/checkpoint, code,
versions, window batch size, effective device and numerical thread configuration.
Changing workers can change the inference thread count and invalidate resume.
A verified complete report can be reused without initializing ESM. Entirely
noncanonical cohorts need no model runtime; their structural descriptors are NA.

## Scientific contract and Python API

The pipeline uses full-coverage ProStruct windows (maximum 1021 residues,
minimum overlap 100), pinned layer-30 ESM-2 residue embeddings and the frozen
exposure/Q3 heads. Dark's saved pooled N/C-window embeddings are not reused.
Exposure is raw ALL contextual exposure fraction (ALL CEF), not conventional
normalized RSA. It is unclipped; Q3 uses raw C/E/H labels without smoothing.
The descriptors are sequence/charge proxies and predicted structural summaries,
not a physical electrostatic calculation.

```python
from pclass_dark.feature_extraction import extract_features

result = extract_features(
    output="features.tsv",  # Python integrations may choose another existing directory.
    device="cpu", offline=True, workers=1, window_batch_size=8,
)
print(result.tsv_path, result.sequence_count, result.membership_count)

# Pure aggregation for future queries, using the same feature definitions:
from pclass_dark.structure import StructuralPredictor
from pclass_dark.structural_features import FEATURES, summarize_structural_features

predictor = StructuralPredictor(device="cpu", offline=True)
prediction = predictor.predict("DEKRHX")
summary = summarize_structural_features(prediction.sequence, prediction)
print(dict(zip(FEATURES, summary.values)))
```

`extract_features` raises ValueError/IntegrityError for invalid input/state,
SetupBusyError for contention, IO errors for failed file operations, and
RuntimeError for model/worker failures. The script exits 1 on operational errors,
2 on argument errors and 130 on interruption. Logs go to stderr; successful stdout
reports the output path and counts.

Changing the export precision changes the extraction contract. Historical complete
reports remain verifiable, but `--resume` requires matching code and contract;
export-only conversion does not relabel old cached computations as a new run.

This extraction fits no classifier or preprocessing and makes no reranking
performance claim. Model training, imputation/scaling, homology-controlled held-out
splits and accepted-candidate ranking evaluation are subsequent experimental work.
See [structural models](structural-prediction.md) for head provenance.
