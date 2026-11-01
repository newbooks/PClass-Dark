"""Reference resolution, streaming annotation filters and exact sequence selection."""

import gzip
import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime

from .external_sequence_io import record
from .external_sequence_types import ExternalSnapshotError, StageCount

EVIDENCE = frozenset(
    {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"}
)
ACCESSION = re.compile(
    r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
)
RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYBZXUOJ")
STAGES = (
    "Input GAF records",
    "UniProtKB",
    "Without NOT",
    "MF aspect F",
    "Experimental evidence",
    "Present in local MF graph",
    "Grouped parent accessions",
    "After local Swiss-Prot exclusion",
    "Retrieved full sequences",
    "Valid full sequences",
    "Deduplicated (latest annotation)",
)


def parent_accession(value):
    value = re.sub(r"-[1-9][0-9]*$", "", value.strip())
    if not ACCESSION.fullmatch(value):
        raise ValueError(f"Invalid UniProt accession: {value!r}")
    return value


def header_accession(header, *, canonical=False):
    token = header.split()[0] if header.split() else ""
    if "|" in token:
        fields = token.split("|")
        if len(fields) != 3 or fields[0] not in {"sp", "tr"}:
            raise ValueError(f"Malformed FASTA header: {header!r}")
        token = fields[1]
    parent = parent_accession(token)
    if canonical and token != parent:
        raise ValueError(f"Unexpected isoform response: {token}")
    return parent


def references(graph_path, swiss_path, reference_fasta):
    """Return qualified graph nodes, complete exclusion set and source contract."""
    from profunc.go.inspect import _load_go_graph

    from .reference_data import (
        profunc_fasta_path,
        profunc_reference_contract,
        resolve_profunc,
    )

    runtime = resolve_profunc() if graph_path is None or swiss_path is None else None
    files = {}
    contract = {"runtime": runtime.manifest if runtime else None}
    if graph_path is None:
        graph_path = runtime.dataset / "go_graph.pkl"
        metadata = json.loads((runtime.dataset / "manifest.json").read_text())
        member = next(
            (m for m in metadata["members"] if m["path"] == "go_graph.pkl"), None
        )
        if member is None or record(graph_path) != {
            "size_bytes": member["size"],
            "sha256": member["sha256"],
        }:
            raise ExternalSnapshotError("Profunc graph differs from its manifest")
    if not graph_path.is_file() or graph_path.is_symlink():
        raise ValueError("GO graph must be a regular non-symlink file")
    files[str(graph_path)] = record(graph_path)
    graph = _load_go_graph(graph_path)
    nodes = set()
    for key, value in graph["terms"].items():
        if (
            not isinstance(key, str)
            or not re.fullmatch(r"GO:[0-9]{7}", key)
            or not isinstance(value, dict)
        ):
            raise ValueError("Malformed GO graph term")
        if (
            not value.get("obsolete", False)
            and not value.get("is_obsolete", False)
            and value.get("namespace", "molecular_function") == "molecular_function"
        ):
            nodes.add(key)
    excluded = set()
    if swiss_path is not None:
        files[str(swiss_path)] = record(swiss_path)
        for line in swiss_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                excluded.add(parent_accession(line))
        contract["exclusion_source"] = "explicit accession list"
    else:
        fasta = profunc_fasta_path(runtime, reference_fasta)
        if not fasta.is_file() or fasta.is_symlink():
            raise ValueError("Reference FASTA must be a regular non-symlink file")
        expected = profunc_reference_contract(runtime)["accession.fasta"]
        files[str(fasta)] = record(fasta)
        if files[str(fasta)] != expected:
            raise ExternalSnapshotError(
                "Swiss-Prot FASTA differs from profunc manifest"
            )
        seen_header = False
        with fasta.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(">"):
                    accession = header_accession(line[1:])
                    if accession in excluded:
                        raise ValueError(f"Duplicate Swiss-Prot accession: {accession}")
                    excluded.add(accession)
                    seen_header = True
                elif line.strip() and not seen_header:
                    raise ValueError("Sequence before Swiss-Prot FASTA header")
        contract["exclusion_source"] = "complete profunc dataset FASTA"
    contract["files"] = files
    return nodes, excluded, contract


def connect(path):
    db = sqlite3.connect(path)
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version not in {0, 1}:
        db.close()
        raise ValueError("Unsupported external sequence SQLite schema")
    db.execute("PRAGMA user_version=1")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-32768")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS stage_accessions (
        accession TEXT PRIMARY KEY, stage_mask INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS annotations (
        accession TEXT, source_id TEXT, go_id TEXT, evidence TEXT,
        annotation_date TEXT,
        PRIMARY KEY(accession,source_id,go_id,evidence,annotation_date));
    CREATE TABLE IF NOT EXISTS candidates (
        accession TEXT PRIMARY KEY, latest_date TEXT,
        excluded_swissprot INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS terms (
        accession TEXT, go_id TEXT, PRIMARY KEY(accession,go_id));
    CREATE TABLE IF NOT EXISTS batches (
        batch_id TEXT PRIMARY KEY, request_ids_json TEXT, response_path TEXT,
        response_sha256 TEXT, release TEXT, release_date TEXT, status TEXT);
    CREATE TABLE IF NOT EXISTS sequences (
        accession TEXT PRIMARY KEY, status TEXT, sequence TEXT, sequence_sha256 TEXT);
    CREATE INDEX IF NOT EXISTS sequence_hash ON sequences(sequence_sha256);
    CREATE TABLE IF NOT EXISTS representatives (
        accession TEXT PRIMARY KEY, winner TEXT);
    """)
    return db


def _log_parse_progress(rows, qualified):
    logging.getLogger(__name__).info(
        "Parsed %s GAF rows; %s graph-qualified rows", f"{rows:,}", f"{qualified:,}"
    )


def parse_gaf(db, path, nodes, excluded):
    for table in ("stage_accessions", "annotations", "candidates", "terms"):
        db.execute(f"DELETE FROM {table}")
    counts = [0] * 6
    version = None
    headers = []
    header_count = 0
    invalid_accession_rows = 0
    invalid_accession_examples = []
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as handle:
        for number, line in enumerate(handle, 1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("!"):
                header_count += 1
                if len(headers) < 1000:
                    headers.append(line)
                if line.startswith("!gaf-version:"):
                    declared = line.partition(":")[2].strip()
                    if declared not in {"2.1", "2.2"} or version not in {
                        None,
                        declared,
                    }:
                        raise ValueError(
                            f"GAF line {number:,}: unsupported/conflicting version"
                        )
                    version = declared
                continue
            try:
                if version is None:
                    raise ValueError("missing GAF version")
                columns = line.split("\t")
                if len(columns) != 17:
                    raise ValueError("expected 17 columns")
                counts[0] += 1
                if columns[0] != "UniProtKB":
                    continue
                try:
                    accession = parent_accession(columns[1])
                except ValueError:
                    invalid_accession_rows += 1
                    if len(invalid_accession_examples) < 10:
                        invalid_accession_examples.append(
                            {"line": number, "source_id": columns[1]}
                        )
                        logging.getLogger(__name__).warning(
                            "Skipping GAF line %s: invalid UniProtKB accession %r",
                            f"{number:,}",
                            columns[1],
                        )
                    continue
                passed = 1
                checks = (
                    "NOT" not in {v.strip() for v in columns[3].split("|")},
                    columns[8] == "F",
                    columns[6] in EVIDENCE,
                    columns[4] in nodes,
                )
                for check in checks:
                    if not check:
                        break
                    passed += 1
                for stage in range(passed):
                    counts[stage + 1] += 1
                db.execute(
                    "INSERT INTO stage_accessions VALUES (?,?) ON CONFLICT(accession) "
                    "DO UPDATE SET stage_mask=stage_mask | excluded.stage_mask",
                    (accession, (1 << passed) - 1),
                )
                if passed == 5:
                    date = columns[13]
                    if not re.fullmatch(r"[0-9]{8}", date):
                        raise ValueError("invalid annotation date")
                    datetime.strptime(date, "%Y%m%d")
                    db.execute(
                        "INSERT OR IGNORE INTO annotations VALUES (?,?,?,?,?)",
                        (accession, columns[1], columns[4], columns[6], date),
                    )
            except ValueError as exc:
                raise ValueError(f"GAF line {number:,}: {exc}") from exc
            finally:
                if counts[0] and counts[0] % 10000 == 0:
                    db.commit()
                    if counts[0] % 1000000 == 0:
                        _log_parse_progress(counts[0], counts[5])
    if version is None:
        raise ValueError("Missing GAF version declaration")
    if invalid_accession_rows:
        logging.getLogger(__name__).warning(
            "Skipped %s GAF rows with invalid UniProtKB accessions",
            f"{invalid_accession_rows:,}",
        )
    db.execute(
        "INSERT INTO candidates(accession,latest_date) "
        "SELECT accession,MAX(annotation_date) FROM annotations GROUP BY accession"
    )
    db.execute("INSERT INTO terms SELECT DISTINCT accession,go_id FROM annotations")
    db.executemany(
        "UPDATE candidates SET excluded_swissprot=1 WHERE accession=?",
        ((a,) for a in excluded),
    )
    db.commit()
    stages = [StageCount(STAGES[0], counts[0], None, None)]
    for i in range(5):
        n = db.execute(
            "SELECT COUNT(*) FROM stage_accessions WHERE stage_mask & ? != 0", (1 << i,)
        ).fetchone()[0]
        stages.append(StageCount(STAGES[i + 1], counts[i + 1], n, None))
    grouped = db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    selected = db.execute(
        "SELECT COUNT(*) FROM candidates WHERE excluded_swissprot=0"
    ).fetchone()[0]
    stages.extend(
        (
            StageCount(STAGES[6], None, grouped, None),
            StageCount(STAGES[7], None, selected, None),
        )
    )
    return stages, {
        "invalid_accession_rows": invalid_accession_rows,
        "invalid_accession_examples": invalid_accession_examples,
        "version": version,
        "comment_lines": header_count,
        "retained_comments": headers,
        "truncated": header_count > len(headers),
    }


def fasta_records(path, requested):
    """Validate one bounded response; preserve invalid requested records as such."""
    records = {}
    accession = None
    chunks = []

    def finish():
        if accession is None:
            return
        raw_sequence = "".join(chunks)
        sequence = raw_sequence.upper() if raw_sequence.isascii() else raw_sequence
        if accession in records and records[accession] != sequence:
            raise ValueError(f"Conflicting repeated response: {accession}")
        records[accession] = sequence

    with path.open(encoding="utf-8", errors="strict") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if not line:
                continue
            if line.startswith(">"):
                finish()
                accession = header_accession(line[1:], canonical=True)
                if accession not in requested:
                    raise ValueError(f"Unexpected response accession: {accession}")
                chunks = []
            elif accession is None:
                raise ValueError("Sequence before first FASTA header")
            else:
                chunks.append(line)
        finish()
    return records


def store_sequences(db, requested, records):
    for accession in requested:
        sequence = records.get(accession)
        status = "unresolved" if sequence is None else "invalid"
        if sequence and set(sequence) <= RESIDUES:
            status = "valid"
        db.execute(
            "INSERT OR REPLACE INTO sequences VALUES (?,?,?,?)",
            (
                accession,
                status,
                sequence if status == "valid" else None,
                hashlib.sha256(sequence.encode("ascii")).hexdigest()
                if status == "valid"
                else None,
            ),
        )


def deduplicate(db, stages):
    db.execute("DELETE FROM representatives")
    rows = db.execute(
        "SELECT s.accession,s.sequence_sha256,s.sequence FROM sequences s "
        "JOIN candidates c USING(accession) WHERE s.status='valid' "
        "ORDER BY s.sequence_sha256,c.latest_date DESC,s.accession"
    )
    for accession, digest, sequence in rows:
        # SQL checks exact strings as well as hashes, including deliberate collisions.
        winner = db.execute(
            "SELECT r.winner FROM representatives r JOIN sequences s "
            "ON r.accession=s.accession WHERE r.accession=r.winner "
            "AND s.sequence_sha256=? AND s.sequence=? LIMIT 1",
            (digest, sequence),
        ).fetchone()
        db.execute(
            "INSERT INTO representatives VALUES (?,?)",
            (accession, winner[0] if winner else accession),
        )
    db.commit()
    counts = dict(db.execute("SELECT status,COUNT(*) FROM sequences GROUP BY status"))
    valid = counts.get("valid", 0)
    retrieved = valid + counts.get("invalid", 0)
    winners = db.execute(
        "SELECT COUNT(*) FROM representatives WHERE accession=winner"
    ).fetchone()[0]
    selected = stages[-1].accessions
    if selected != retrieved + counts.get("unresolved", 0):
        raise ExternalSnapshotError("Retrieval count mismatch")
    stages = stages + [
        StageCount(STAGES[8], None, retrieved, retrieved),
        StageCount(STAGES[9], None, valid, valid),
        StageCount(STAGES[10], None, winners, winners),
    ]
    rejections = {
        "swissprot_accessions": stages[6].accessions - selected,
        "unresolved_accessions": counts.get("unresolved", 0),
        "invalid_sequences": counts.get("invalid", 0),
        "duplicate_sequences": valid - winners,
    }
    return stages, rejections


def write_fasta(db, path):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for accession, sequence, date in db.execute(
            "SELECT s.accession,s.sequence,c.latest_date FROM sequences s "
            "JOIN representatives r USING(accession) "
            "JOIN candidates c USING(accession) "
            "WHERE r.accession=r.winner ORDER BY s.accession"
        ):
            terms = "|".join(
                r[0]
                for r in db.execute(
                    "SELECT go_id FROM terms WHERE accession=? ORDER BY go_id",
                    (accession,),
                )
            )
            handle.write(
                f">{accession} go_mf_terms={terms} latest_annotation_date={date}\n"
            )
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")
