# Experimental feature extraction: CPU pilot and software validation

Date: 2026-11-06. Revision: `5f4facf41a9ce541e3ad55fc6bc5cd860aa6ed25`
plus the extraction implementation. Exact module hashes and package versions are
in the [pilot metadata](2026-11-06-extract-features-pilot/pilot.json) and
[output manifest](2026-11-06-extract-features-pilot/features.tsv.manifest.json).
Specification: [SPEC-0021](../../specs/0021-extract-structural-features.md).

This is a synthetic implementation check, not protein-function performance or a
reranking experiment. No classifier or preprocessing was fitted and there are no
training/test splits, learned feature choices, or accuracy results.

## Inputs and reproduction

The reference fixture has six original accession records:

| Accession | Original synthetic cluster | Full sequence |
| --- | --- | --- |
| A0 | excluded | DEKRH |
| B1 | K1 | DEKRH |
| B2 | K1 | ACDEKR |
| C1 | K2 | DEKRH |
| C2 | K2 | KKDA |
| U1 | K1 | XX |

Prepare with min_clusters=2, seed=42, workers=1 and prepare_only=True.
Cluster excluded is removed, but its A0 accession is the global representative of
DEKRH, which remains in both K1 and K2. The retained cohort therefore contains
four unique sequences and five memberships. This deliberately tests the
excluded-representative and cross-cluster duplicate cases.

The fixture construction matches `cohort` in
[the integration tests](../../../tests/test_feature_extraction.py). A standalone
recipe from the repository root is:

```python
from pathlib import Path
import tempfile
from pclass_dark.artifacts import file_record, write_json
from pclass_dark.setup_data import prepare_data

root = Path(tempfile.mkdtemp(prefix="pclass-dark-feature-reproduction-"))
reference = root / "reference"
reference.mkdir()
records = [
    ("A0", "excluded", "DEKRH"), ("B1", "K1", "DEKRH"),
    ("B2", "K1", "ACDEKR"), ("C1", "K2", "DEKRH"),
    ("C2", "K2", "KKDA"), ("U1", "K1", "XX"),
]
(reference / "clusters.tsv").write_text(
    "member_id\tcluster_id\n" + "".join(f"{a}\t{c}\n" for a, c, s in records)
)
(reference / "accession.fasta").write_text(
    "".join(f">{a}\n{s}\n" for a, c, s in records)
)
write_json(reference / "manifest.json", {
    "schema_version": 1,
    "counts": {
        "fasta_records": 6, "fasta_residues": 27, "clustered_accessions": 6,
        "clusters": 3, "unclustered_fasta_accessions": 0,
        "clustered_noncanonical_sequences": 1,
    },
    "files": {n: file_record(reference / n)
              for n in ("clusters.tsv", "accession.fasta")},
})
prepared = prepare_data(
    root / "setup", reference_dir=reference, prepare_only=True,
    min_clusters=2, seed=42, workers=1,
)
(root / "results").mkdir()
print(prepared.prepared_dir, root / "results")
```

Run from that results directory, substituting the printed preparation path:

```bash
/usr/bin/time -v python /path/to/PClass-Dark/scripts/extract_features.py \
  --prepared-dir /path/to/pilot/setup/prepared/PREPARATION_ID \
  --device cpu --workers 1 --window-batch-size 1 --offline
```

Actual temporary root:
`/tmp/pclass-dark-feature-pilot-20261106-595td09r`.
Preparation ID:
`5e16555f574ce68c71a573e6499d3717e1da738c64068ee1379120aa07a9f175`.
Serial extraction ID:
`d9ae16e8ba255581d7ffa3c3c4ed44b6652775bae36d1f5de0ecff92ba61bbfa`.
Pinned ESM-2 t30/150M and the original packaged dual heads were used unchanged;
no downloads or retraining. Float32 residue inference, float64 aggregation,
full-precision TSV output. Main feature table SHA-256:
`7669a2d8e9329270e0c8001d2f69b0e4d8ce7c511458a7fb1b02c3beab23ef75`.

## Observations and artifacts

- [features.tsv](2026-11-06-extract-features-pilot/features.tsv): five rows,
  exactly 21 columns; duplicate A0 features preserve both K1/K2 labels.
- [QC](2026-11-06-extract-features-pilot/features.qc.tsv) and
  [sequence identity index](2026-11-06-extract-features-pilot/features.sequences.tsv):
  four rows each; XX has no canonical residues and 13 undefined descriptors,
  with no model forward for that sequence.
- Three canonical sequences were predicted; no raw exposure excursions or SCD
  fallbacks occurred. One sequence has unknown residues/undefined features.
- Complete-output --resume succeeded without model initialization or report edits.
- A second results directory with --workers 2 used two spawned aggregation workers
  and one inference numerical thread, versus eight threads in serial mode.
  Its main feature TSV was byte-identical; per-run metadata/identities differ.
  Parallel extraction ID:
  `52f75bfce8d7c6b094245c5c37380fff8e86ca91c4c64a584ceb923b348707bd`;
  [parallel manifest](2026-11-06-extract-features-pilot/parallel.manifest.json).
- Serial command: 5.86 seconds wall time, peak RSS 1,465,432 KiB, measured by
  /usr/bin/time -v. This includes interpreter/checkpoint loading and is not a
  full-reference throughput estimate. Runtime: Linux x86_64, Python 3.14.6,
  NumPy 2.4.4, PyTorch 2.9.1+rocm6.3, Transformers 5.13.1, CPU only.
- Independent aggregation parity: 15 seeded fixtures (seed 21), including unknown
  residues, compared with ProStruct's unrounded features.summarize outputs;
  all defined descriptors agreed at atol/rtol 1e-12 and undefined values agreed.

Intermediate pilots during validation produced the same feature TSV while state/
argument validation was being hardened; their transient work was retained under
other /tmp/pclass-dark-feature-pilot-20261106-* directories. The committed artifacts
here belong to the final code hash set above, not those intermediate identities.

## Automated checks and limits

Final relevant suite: 193 passed, 1 skipped (GPU unavailable), one existing AMD
runtime-environment warning. It covers analytical features, threshold precision,
FFT/direct agreement and fallback, all-unknown handling, automatic discovery/cwd,
serial/spawn consistency and worker imports, payload admission, receipts/resume,
input/code/snapshot mutation, publication recovery and third-party mutation,
packaging, plus existing structure/embedding/setup/reference/CLI/shared-path tests.
Exact command and intermediate failures are in the chronological progress log.

No actual GPU run or production-scale timing/memory test was possible here.
Synthetic labels and possible structural-head training overlap make this unsuitable
for biological generalization claims. The full reference was not processed, and
model training/reranking evaluation remain future work.
