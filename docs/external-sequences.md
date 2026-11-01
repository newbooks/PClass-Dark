# External experimentally annotated sequences

Run from the directory where the results should be written, with PClass-Dark
installed (`pip install -e /path/to/PClass-Dark`):

```bash
python /path/to/PClass-Dark/scripts/get-external-seq.py
```

The first run downloads `goa_uniprot_all.gaf.gz` (about 12 GB in historical
snapshots). It streams annotation parsing into a disk-backed SQLite database.
Allow approximately 50 GB free disk space for processing/refresh, including a
potential new download alongside the old one. Exact use depends on accession
counts. There is no sampling cap, embedding computation or GPU requirement.

Default references come from installed profunc's validated active dataset:
`go_graph.pkl` and the complete original `accession.fasta`, including proteins
outside retained clusters. Supply `--reference-fasta` if the portable runtime
omits that source file; it must match profunc's dataset manifest. Alternatively:

```bash
python /path/to/PClass-Dark/scripts/get-external-seq.py --gaf /data/goa_uniprot_all.gaf.gz \
  --go-graph /data/go_graph.pkl --swissprot-accessions /data/swissprot.txt \
  --workers 4 --batch-size 100 --retries 3
```

The graph override is a **trusted local profunc-format pickle**. The Swiss-Prot
list contains one accession per line, with optional blank lines and `#` comments.
It is mutually exclusive with `--reference-fasta`. `--gaf` avoids only the GOA
download: fetching selected sequences still needs network access. `--workers 1`
provides serial retrieval. The script always write to cwd, not the repository.

## Selection and outputs

Only positive UniProtKB MF annotations with evidence EXP, IDA, IPI, IMP, IGI, IEP,
HTP, HDA, HMP, HGI or HEP and a term present in the local active MF graph qualify.
Isoform IDs are normalized to parents; this is an explicit annotation-to-parent
approximation. All qualifying terms for each parent are retained. Accessions in
the local Swiss-Prot snapshot are excluded before full sequence retrieval.

For identical obtained sequences, keep the accession with the greatest **GAF
annotation date among its qualifying rows**. Date ties choose the smallest
accession lexicographically. Retain that winner's GO terms only; terms from other
accessions are not combined. Do not exclude exact Swiss-Prot sequence matches,
truncate long proteins or sample accessions. This is local accession exclusion,
not verification of current unreviewed status or homology-independent evaluation.

The two primary files are:

- `external_sequences.fasta`: full unique sequences, ordered by winner accession.
  Headers contain `go_mf_terms=GO:...|GO:...` and
  `latest_annotation_date=YYYYMMDD` after the accession token.
- `external_filtered.txt`: aligned stage counts with annotation-row, distinct
  accession and sequence columns, rejection counts and source provenance. Before
  retrieval, sequence counts are unknown and displayed as `-`.

Empty successful selections produce an empty FASTA and complete count report.
Missing response IDs and invalid sequences are excluded and counted. GAF rows
labelled UniProtKB but containing invalid/non-protein accessions are also skipped
and counted as `invalid_accession_rows`, with up to ten warning examples.
Structural/header/date errors, network failures and release inconsistency stop
publication and preserve old results. Parse-progress and GAF diagnostic line
numbers use comma separators.
Logs use stderr; final paths and the sequence count use stdout.

## Resume, refresh and consistency

`.external-seq/` holds private schema-versioned run state, original qualifying
annotation provenance, duplicate mappings, response caches and a completed
`manifest.json` with both output hashes. An interrupted parse restarts; completed
sequence batches are verified and reused. Repeat the same command to resume.
Compatible completed results are verified and returned without downloading again.

Use `--refresh` to start a new acquisition generation when sources/configuration
change or to replace stale cached responses. A source/retrieval release mismatch
requires refresh. Do not manually mix files between run directories. Historical
runs remain on disk; version 1 does not automatically prune them.

Unregistered output files are never overwritten. Modified tool-owned final files
must be restored to their recorded contents, or both outputs and `.external-seq/`
moved aside before restarting. This protects manual edits even with `--refresh`.
A cwd lock prevents concurrent writers. Interrupted paired publication is rolled
back or finished from a durable journal on the next invocation. Readers requiring
a consistent pair should hold `.external-seq/lock` or verify both file hashes
against the completed manifest; ordinary two-file reads are not atomic.

## Python API

```python
from pathlib import Path
from pclass_dark.external_sequences import collect_external_sequences

result = collect_external_sequences(work_dir=Path.cwd(), workers=4)
print(result.fasta_path, result.report_path, result.counts[-1].sequences)
```

Optional keyword arguments are `go_graph=None`, `swissprot_accessions=None`,
`reference_fasta=None`, `gaf=None`, `refresh=False`, `workers=4`, `batch_size=100`,
and `retries=3`. Paths may be strings or Path objects and resolve against the
caller's cwd. The work directory must exist. The frozen result provides
`fasta_path`, `report_path`, `manifest_path`, `run_id` and a tuple of frozen
`StageCount(stage, annotation_rows, accessions, sequences)` records; unavailable
counts are `None`.

Failures are `ValueError` for invalid configuration/data, `OSError` for I/O,
and public `ExternalDownloadError`, `ExternalSnapshotError` or
`ExternalRecoveryError` (all RuntimeError subclasses), importable from the same
module. Imports do not read references or access the network. CLI usage errors
exit 2, operational failures 1, interruption 130, success 0.

See [SPEC-0014](specs/0014-get-external-sequences.md) for algorithms, data contracts
and implementation validation.

GOA download progress uses comma-separated decimal KB (1000 bytes), for example
`Downloaded GOA: 18,790,481 / 28,598,433 KB (65.7%)`. When the HTTP response does
not supply a comparable total size, percentage is explicitly unavailable.

Some GOA rows describe protein complexes using `CPX-...` identifiers even under
DB `UniProtKB`. These cannot be retrieved as canonical protein sequences and are
skipped under `invalid_accession_rows`; their annotations need not be malformed.
After a failure from the earlier strict parser, rerun the same command without
`--refresh`: the validated download is reused, while GAF parsing restarts from
the beginning. Partial GAF parsing is not resumable.
