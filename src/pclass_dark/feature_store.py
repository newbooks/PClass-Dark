"""Streaming, checksum-bound cohort snapshots and sequence receipts (SPEC-0021)."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .artifacts import (
    IntegrityError,
    _pairs,
    canonical,
    identity,
    read_json,
    safe_file,
    schema_checked,
    sha256,
    transaction_guard,
    verify_bundle,
)
from .reference_data import read_fasta, sequence_id
from .setup_data import HEADERS, PREP_SCHEMA, reference_identity
from .structural_features import CANONICAL, FEATURES, HYDROPATHY, FeatureQC

TABLES = {
    "sequences": "sequence_id",
    "memberships": "sequence_id,cluster_id",
    "aliases": "accession",
    "clusters": "cluster_id",
}
LIMIT = 10_000_000


def loads(value):
    def invalid(_):
        raise IntegrityError("Nonfinite JSON number")

    try:
        return json.loads(value, object_pairs_hook=_pairs, parse_constant=invalid)
    except (ValueError, UnicodeError) as exc:
        raise IntegrityError("Invalid cached JSON") from exc


def connect(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA cache_size=-65536")
    db.execute("PRAGMA temp_store=FILE")
    return db


def create(db):
    db.executescript("""
    PRAGMA user_version=1;
    CREATE TABLE sequences(sequence_id TEXT PRIMARY KEY,
      accession_id TEXT UNIQUE NOT NULL,
      sequence TEXT NOT NULL,length INTEGER NOT NULL CHECK(length>0),
      canonical_count INTEGER NOT NULL CHECK(canonical_count>=0));
    CREATE TABLE memberships(sequence_id TEXT REFERENCES sequences,cluster_id TEXT,
      PRIMARY KEY(sequence_id,cluster_id));
    CREATE INDEX member_cluster ON memberships(cluster_id);
    CREATE TABLE aliases(accession TEXT PRIMARY KEY,original_cluster_id TEXT,
      sequence_id TEXT,representative TEXT,
      retained_cluster INTEGER CHECK(retained_cluster IN(0,1)),
      retained_sequence INTEGER CHECK(retained_sequence IN(0,1)));
    CREATE INDEX alias_sequence ON aliases(sequence_id);
    CREATE INDEX alias_cluster ON aliases(original_cluster_id);
    CREATE TABLE clusters(cluster_id TEXT PRIMARY KEY,
      original_accessions INTEGER CHECK(original_accessions>0),
      unique_sequences INTEGER CHECK(unique_sequences>0),
        retained INTEGER CHECK(retained IN(0,1)));
    CREATE TABLE completed(sequence_id TEXT PRIMARY KEY REFERENCES sequences,
      record_json TEXT NOT NULL,record_sha256 TEXT NOT NULL);
    """)


def identifier(value):
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or (not value.isprintable() or any(c.isspace() for c in value))
    ):
        raise IntegrityError(f"Invalid accession/cluster identifier: {value!r}")
    return value


def number(value):
    if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise IntegrityError("Invalid integer field")
    return int(value)


def flag(value):
    if value not in ("true", "false"):
        raise IntegrityError("Expected true/false flag")
    return int(value == "true")


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != HEADERS[path.name]:
            raise IntegrityError(f"Invalid prepared header: {path.name}")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise IntegrityError(f"Malformed prepared row: {path.name}")
            yield row


@schema_checked
def resolve_preparation(prepared_dir):
    from .paths import default_setup_input_dir

    if prepared_dir is not None:
        path = Path(prepared_dir).absolute()
        return (
            path,
            None,
            (path.parent.parent if path.parent.name == "prepared" else path.parent),
        )
    root = Path(default_setup_input_dir()).absolute()
    selection = read_json(safe_file(root, "selection.json"))
    pid = selection.get("preparation_id")
    if (
        selection.get("schema") != "pclass-dark.selection.v1"
        or not isinstance(pid, str)
        or not re.fullmatch(r"[0-9a-f]{64}", pid)
        or selection.get("prepared") != "prepared/" + pid
    ):
        raise IntegrityError("Invalid selected preparation")
    return safe_file(root, selection["prepared"]), selection, root


def prepared_guard(root):
    from contextlib import nullcontext

    from .paths import PACKAGE_DIR

    resources = PACKAGE_DIR / "resources"
    return (
        nullcontext()
        if root == resources or resources in root.parents
        else transaction_guard(root)
    )


def inspect_preparation(path, selection):
    manifest = verify_bundle(path, PREP_SCHEMA)
    expected = identity(
        {
            "reference": reference_identity(manifest["reference"]),
            "configuration": manifest["configuration"],
        }
    )
    if manifest.get("id") != expected:
        raise IntegrityError("Prepared identity mismatch")
    if selection is not None and (
        manifest["id"] != selection["preparation_id"]
        or sha256(path / "manifest.json") != selection.get("prepared_manifest_sha256")
    ):
        raise IntegrityError("Preparation differs from selected manifest")
    return manifest


def snapshot(db, path, manifest):
    """Stream authoritative files into SQLite, retaining every membership."""
    for accession, sequence in read_fasta(
        path / "sequences.fasta",
        check_stable=True,
        max_header_bytes=4096,
        max_sequence_length=LIMIT,
    ):
        identifier(accession)
        if sequence != sequence.upper():
            raise IntegrityError("Prepared FASTA must be uppercase")
        db.execute(
            "INSERT INTO sequences VALUES(?,?,?,?,?)",
            (
                sequence_id(sequence),
                accession,
                sequence,
                len(sequence),
                sum(r in CANONICAL for r in sequence),
            ),
        )
    for row in rows(path / "clusters.tsv"):
        retained = flag(row["retained"])
        if row["exclusion_reason"] != ("" if retained else "cluster_below_minimum"):
            raise IntegrityError("Invalid cluster exclusion reason")
        db.execute(
            "INSERT INTO clusters VALUES(?,?,?,?)",
            (
                identifier(row["cluster_id"]),
                number(row["original_accessions"]),
                number(row["unique_sequences"]),
                retained,
            ),
        )
    for row in rows(path / "aliases.tsv"):
        rc, rs = flag(row["retained_cluster"]), flag(row["retained_sequence"])
        if not re.fullmatch(r"[0-9a-f]{64}", row["sequence_id"]) or (
            row["exclusion_reason"] != ("" if rc else "cluster_below_minimum")
        ):
            raise IntegrityError("Invalid alias hash/exclusion")
        db.execute(
            "INSERT INTO aliases VALUES(?,?,?,?,?,?)",
            (
                identifier(row["accession"]),
                identifier(row["original_cluster_id"]),
                row["sequence_id"],
                identifier(row["representative"]),
                rc,
                rs,
            ),
        )
    for row in rows(path / "memberships.tsv"):
        match = db.execute(
            "SELECT accession_id FROM sequences WHERE sequence_id=?",
            (row["sequence_id"],),
        ).fetchone()
        if match is None or match[0] != row["representative"]:
            raise IntegrityError("Invalid membership representative")
        db.execute(
            "INSERT INTO memberships VALUES(?,?)",
            (row["sequence_id"], identifier(row["cluster_id"])),
        )
    validate_snapshot(db, manifest)
    db.commit()


def validate_snapshot(db, manifest):
    if db.execute("PRAGMA quick_check").fetchone()[0] != "ok" or (
        db.execute("PRAGMA user_version").fetchone()[0] != 1
    ):
        raise IntegrityError("Corrupt feature snapshot database")
    minimum = manifest["configuration"]["min_clusters"]
    if type(minimum) is not int or minimum < 1:
        raise IntegrityError("Invalid retained-cluster minimum")
    checks = [
        (
            "SELECT 1 FROM aliases a LEFT JOIN clusters c ON "
            "c.cluster_id=a.original_cluster_id "
            "LEFT JOIN sequences s ON s.sequence_id=a.sequence_id WHERE c.cluster_id "
            "IS NULL "
            "OR a.retained_cluster != c.retained OR a.retained_sequence != "
            "(s.sequence_id IS NOT NULL) "
            "OR (s.sequence_id IS NOT NULL AND a.representative!=s.accession_id) "
            "LIMIT 1",
            (),
        ),
        (
            "SELECT 1 FROM clusters c LEFT JOIN (SELECT original_cluster_id "
            "c,COUNT(*) n, "
            "COUNT(DISTINCT sequence_id) u FROM aliases GROUP BY original_cluster_id) "
            "a "
            "ON a.c=c.cluster_id WHERE a.n IS NULL OR a.n!=c.original_accessions OR "
            "a.u!=c.unique_sequences OR c.retained!=(c.unique_sequences>=?) LIMIT 1",
            (minimum,),
        ),
        (
            "SELECT 1 FROM aliases GROUP BY sequence_id HAVING "
            "MIN(representative)!=MIN(accession) "
            "OR MAX(representative)!=MIN(accession) LIMIT 1",
            (),
        ),
        (
            "SELECT sequence_id,original_cluster_id FROM aliases WHERE "
            "retained_cluster=1 "
            "EXCEPT SELECT sequence_id,cluster_id FROM memberships LIMIT 1",
            (),
        ),
        (
            "SELECT sequence_id,cluster_id FROM memberships EXCEPT SELECT sequence_id, "
            "original_cluster_id FROM aliases WHERE retained_cluster=1 LIMIT 1",
            (),
        ),
        (
            "SELECT 1 FROM sequences s WHERE NOT EXISTS (SELECT 1 FROM memberships m "
            "WHERE m.sequence_id=s.sequence_id) LIMIT 1",
            (),
        ),
    ]
    for sql, params in checks:
        if db.execute(sql, params).fetchone() is not None:
            raise IntegrityError("Inconsistent retained sequence/alias/cluster joins")
    for sid, accession, sequence, length, count in db.execute(
        "SELECT * FROM sequences"
    ):
        identifier(accession)
        if (
            not sequence
            or not sequence.isascii()
            or not sequence.isalpha()
            or sequence != sequence.upper()
            or len(sequence) != length
            or length > LIMIT
            or sequence_id(sequence) != sid
            or sum(r in CANONICAL for r in sequence) != count
        ):
            raise IntegrityError("Invalid snapshot sequence")
    expected = {
        "retained_sequences": db.execute("SELECT COUNT(*) FROM sequences").fetchone()[
            0
        ],
        "retained_memberships": db.execute(
            "SELECT COUNT(*) FROM memberships"
        ).fetchone()[0],
        "source_accessions": db.execute("SELECT COUNT(*) FROM aliases").fetchone()[0],
        "source_clusters": db.execute("SELECT COUNT(*) FROM clusters").fetchone()[0],
        "retained_clusters": db.execute(
            "SELECT COUNT(*) FROM clusters WHERE retained=1"
        ).fetchone()[0],
        "excluded_clusters": db.execute(
            "SELECT COUNT(*) FROM clusters WHERE retained=0"
        ).fetchone()[0],
    }
    if not expected["retained_sequences"] or any(
        manifest["counts"].get(k) != v for k, v in expected.items()
    ):
        raise IntegrityError("Prepared/snapshot count mismatch")
    return expected


def snapshot_hashes(db):
    result = {}
    for table, order in TABLES.items():
        digest = hashlib.sha256()
        for row in db.execute(f"SELECT * FROM {table} ORDER BY {order}"):
            digest.update(canonical(list(row)) + b"\n")
        result[table] = digest.hexdigest()
    return result


def make_record(sid, accession, sequence, summary, extraction_id, device):
    return {
        "schema": "pclass-dark.feature-record.v1",
        "extraction_id": extraction_id,
        "sequence_id": sid,
        "accession_id": accession,
        "length": len(sequence),
        "values": list(summary.values),
        "qc": asdict(summary.qc),
        "device": device,
    }


def validate_record(record, row, extraction_id):
    sid, accession, sequence, length, count = row
    if (
        record.get("schema") != "pclass-dark.feature-record.v1"
        or record.get("extraction_id") != extraction_id
        or record.get("sequence_id") != sid
        or record.get("accession_id") != accession
        or record.get("length") != length
    ):
        raise IntegrityError("Cached feature identity mismatch")
    values, qc = record["values"], record["qc"]
    if (
        set(qc) != set(FeatureQC.__dataclass_fields__)
        or type(record["length"]) is not int
    ):
        raise IntegrityError("Invalid feature receipt schema")
    if (
        not isinstance(values, list)
        or len(values) != 19
        or any(
            v is not None and (type(v) not in (float, int) or not math.isfinite(v))
            for v in values
        )
    ):
        raise IntegrityError("Invalid cached feature values")
    integer_keys = (
        "length",
        "canonical_count",
        "noncanonical_count",
        "predicted_residue_count",
        "surface_count",
        "buried_count",
        "rsa_out_of_range_count",
    )
    if any(type(qc.get(k)) is not int or qc[k] < 0 for k in integer_keys):
        raise IntegrityError("Invalid cached QC counts")
    if (
        qc["length"] != length
        or qc["canonical_count"] != count
        or qc["noncanonical_count"] != length - count
        or qc["predicted_residue_count"] != count
        or qc["surface_count"] + qc["buried_count"] != count
        or qc["rsa_out_of_range_count"] > count
    ):
        raise IntegrityError("Inconsistent cached QC counts")
    expected_missing = {FEATURES[18]} if count < length else set()
    if not count:
        expected_missing.update(FEATURES[i] for i in (*range(3, 10), *range(13, 18)))
    elif qc["surface_count"] == 0:
        expected_missing.update(FEATURES[14:17])
    missing = [f for f, v in zip(FEATURES, values, strict=True) if v is None]
    if set(missing) != expected_missing or qc["undefined_features"] != missing:
        raise IntegrityError("Invalid cached undefined features")
    if (
        qc["prediction_status"] != ("complete" if count else "no_canonical_residues")
        or qc["scd_status"] != ("defined" if count == length else "noncanonical")
        or qc["scd_method"]
        not in ("fft", "direct_fallback", "zero_shortcut", "not_applicable")
        or (count < length) != (qc["scd_method"] == "not_applicable")
        or (not count and record["device"] != "not_run")
        or (count and not re.fullmatch(r"cpu|cuda:\d+", record["device"]))
    ):
        raise IntegrityError("Invalid cached status/device")

    def equal(a, b):
        return a is not None and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)

    acid, base = sum(r in "DE" for r in sequence), sum(r in "KRH" for r in sequence)
    expected = {
        0: math.log(length),
        1: acid / length,
        2: base / length,
        10: (sum(r in "KR" for r in sequence) - acid) / length,
        11: sum(r in "DEKR" for r in sequence) / length,
        12: sequence.count("H") / length,
    }
    if count:
        expected[17] = (
            math.fsum(HYDROPATHY[r] for r in sequence if r in CANONICAL) / count
        )
        expected[13] = qc["surface_count"] / length
        if (
            not equal(values[3] + values[4], acid / length)
            or not equal(values[5] + values[6], base / length)
            or not equal(sum(values[7:10]), count / length)
        ):
            raise IntegrityError("Cached structural ratios do not reconcile")
        if qc["surface_count"]:
            if not equal(
                values[14], values[3] * length / qc["surface_count"]
            ) or not equal(values[15], values[5] * length / qc["surface_count"]):
                raise IntegrityError("Cached surface ratios do not reconcile")
    if any(not equal(values[i], v) for i, v in expected.items()):
        raise IntegrityError("Cached sequence-only feature mismatch")
    for i, v in enumerate(values):
        if v is None:
            continue
        if (
            (i in (*range(1, 10), 11, 12, *range(13, 17)) and not 0 <= v <= 1)
            or (i == 10 and not -1 <= v <= 1)
            or (i == 17 and not -4.5 <= v <= 4.5)
        ):
            raise IntegrityError("Cached feature outside valid range")
    return record


def read_record(db, row, extraction_id):
    cached = db.execute(
        "SELECT record_json,record_sha256 FROM completed WHERE sequence_id=?", (row[0],)
    ).fetchone()
    if cached is None:
        return None
    text, digest = cached
    if hashlib.sha256(text.encode()).hexdigest() != digest:
        raise IntegrityError("Feature receipt checksum mismatch")
    record = loads(text)
    if canonical(record).decode() != text:
        raise IntegrityError("Noncanonical receipt JSON")
    return validate_record(record, row, extraction_id)
