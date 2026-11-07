"""Experimental feature TSV schemas, verification and journaled publication."""

from __future__ import annotations

import csv
import itertools
import math
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import (
    IntegrityError,
    file_record,
    identity,
    read_json,
    safe_file,
    sync_dir,
    write_json,
)
from .feature_store import identifier, number, read_record, validate_record
from .structural_features import FEATURES, HEADER

QC_HEADER = tuple(
    (
        "accession_id sequence_id length canonical_count noncanonical_count "
        "predicted_residue_count surface_count buried_count rsa_out_of_range_count "
        "scd_status scd_method undefined_feature_count undefined_features "
        "prediction_status device"
    ).split()
)
INDEX_HEADER = ("accession_id", "sequence_id", "length", "membership_count")
SCHEMA = "pclass-dark.structural-features.v1"


@dataclass(frozen=True)
class Family:
    output: Path

    @property
    def parent(self):
        return self.output.parent

    @property
    def stem(self):
        return self.output.stem

    @property
    def names(self):
        return (
            self.output.name,
            f"{self.stem}.qc.tsv",
            f"{self.stem}.sequences.tsv",
            self.output.name + ".manifest.json",
        )

    @property
    def work(self):
        return self.parent / f".{self.stem}.features-work"

    @property
    def lock(self):
        return self.parent / f".{self.stem}.features.lock"

    @property
    def archives(self):
        return self.parent / f".{self.stem}.features-archives"

    def path(self, name):
        if name not in self.names:
            raise IntegrityError("Invalid feature family member")
        return safe_file(self.parent, name)


def extraction_identity(contract_id, metadata):
    fingerprint = "not_run" if metadata == "not_run" else identity(metadata)
    return identity(
        {"contract_id": contract_id, "execution_fingerprint": fingerprint}
    ), fingerprint


def manifest_identity(manifest):
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise IntegrityError("Incomplete feature manifest")
    inputs = manifest["identity_inputs"]
    cid = identity(inputs["contract_inputs"])
    eid, fingerprint = extraction_identity(cid, inputs["execution_metadata"])
    if (
        manifest.get("id") != eid
        or manifest.get("contract_id") != cid
        or manifest.get("execution_fingerprint") != fingerprint
    ):
        raise IntegrityError("Invalid extraction identity")


def formatted(value, *, precision=".3f"):
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        if precision == ".17g":
            return "0" if value == 0 else format(value, precision)
        rendered = format(value, precision)
        return "0.000" if rendered == "-0.000" else rendered
    return str(value)


def write_table(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        count = 0
        for row in rows:
            writer.writerow([formatted(v) for v in row])
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    return {**file_record(path), "rows": count, "columns": list(header)}


def table(path, header):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        if next(reader, None) != list(header):
            raise IntegrityError(f"Invalid feature table header: {path.name}")
        for row in reader:
            if len(row) != len(header):
                raise IntegrityError("Invalid feature table width")
            yield row


def numeric(value, *, precision=".3f"):
    if value == "NA":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise IntegrityError("Invalid feature number") from exc
    if not math.isfinite(parsed) or formatted(parsed, precision=precision) != value:
        raise IntegrityError("Noncanonical/nonfinite feature number")
    return parsed


def verify_family(family, *, directory=None, db=None):
    root = directory or family.parent
    paths = [safe_file(root, n) for n in family.names]
    exists = [p.exists() for p in paths]
    if not any(exists):
        return None
    if not all(exists):
        raise IntegrityError("Partial feature family without a valid complete marker")
    manifest = read_json(paths[-1])
    manifest_identity(manifest)
    policies = manifest["identity_inputs"]["contract_inputs"]["policies"]
    precision = manifest.get("serialization", {}).get(
        "precision", policies["precision"]
    )
    if precision not in (".17g", ".3f"):
        raise IntegrityError("Unsupported feature export precision")
    expected_headers = (HEADER, QC_HEADER, INDEX_HEADER)
    if set(manifest["files"]) != set(family.names[:3]):
        raise IntegrityError("Unexpected feature manifest files")
    for path, header in zip(paths[:3], expected_headers, strict=True):
        record = manifest["files"][path.name]
        if file_record(path) != {
            k: record[k] for k in ("sha256", "size_bytes")
        } or record["columns"] != list(header):
            raise IntegrityError("Feature output checksum/schema mismatch")
    main = itertools.groupby(table(paths[0], HEADER), key=lambda row: row[0])
    qc_iter, index_iter = table(paths[1], QC_HEADER), table(paths[2], INDEX_HEADER)
    counts = {
        "unique_sequences": 0,
        "memberships": 0,
        "retained_clusters": 0,
        "with_noncanonical": 0,
        "no_canonical": 0,
        "with_undefined_features": 0,
        "rsa_out_of_range_residues": 0,
        "scd_fallbacks": 0,
    }
    clusters = set()
    previous = None
    for triple in itertools.zip_longest(main, qc_iter, index_iter):
        grouped, qrow, index = triple
        if grouped is None or qrow is None or index is None:
            raise IntegrityError("Feature/QC/index coverage mismatch")
        accession, group = grouped
        identifier(accession)
        if previous is not None and accession <= previous:
            raise IntegrityError("Feature rows not uniquely ordered")
        previous = accession
        if (
            qrow[0] != accession
            or index[0] != accession
            or qrow[1] != index[1]
            or not re.fullmatch(r"[a-f0-9]{64}", index[1])
        ):
            raise IntegrityError("Feature sequence identity join mismatch")
        qc = dict(zip(QC_HEADER, qrow, strict=True))
        for key in QC_HEADER[2:9] + ("undefined_feature_count",):
            qc[key] = number(qc[key])
        length = number(index[2])
        membership_count = number(index[3])
        if (
            not length
            or qc["length"] != length
            or qc["canonical_count"] + qc["noncanonical_count"] != length
            or not membership_count
        ):
            raise IntegrityError("Invalid feature index/QC counts")
        undefined = (
            []
            if qc["undefined_features"] == "none"
            else qc["undefined_features"].split(",")
        )
        if qc["undefined_feature_count"] != len(undefined):
            raise IntegrityError("Invalid undefined-feature count")
        first = None
        actual_members = []
        for row in group:
            identifier(row[-1])
            if actual_members and row[-1] <= actual_members[-1]:
                raise IntegrityError("Duplicate/unsorted feature membership")
            actual_members.append(row[-1])
            clusters.add(row[-1])
            if first is None:
                first = [numeric(v, precision=precision) for v in row[1:-1]]
            elif [formatted(v, precision=precision) for v in first] != row[1:-1]:
                raise IntegrityError("One sequence has inconsistent feature vectors")
            counts["memberships"] += 1
        if len(actual_members) != membership_count or undefined != [
            f for f, v in zip(FEATURES, first, strict=True) if v is None
        ]:
            raise IntegrityError("Invalid feature membership/missingness counts")
        if db is not None:
            expected = db.execute(
                "SELECT * FROM sequences WHERE accession_id=?", (accession,)
            ).fetchone()
            expected_members = [
                r[0]
                for r in db.execute(
                    "SELECT cluster_id FROM memberships WHERE sequence_id=? ORDER BY "
                    "cluster_id",
                    (index[1],),
                )
            ]
            if (
                expected is None
                or expected[0] != index[1]
                or expected_members != actual_members
            ):
                raise IntegrityError(
                    "Output does not match prepared sequence memberships"
                )
            cached = read_record(db, expected, manifest["id"])
            if cached is None or [
                formatted(v, precision=precision) for v in cached["values"]
            ] != [formatted(v, precision=precision) for v in first]:
                raise IntegrityError("Feature output does not match cached values")
            qvalues = {k: qc[k] for k in QC_HEADER[2:11]}
            qvalues.update(
                undefined_features=undefined, prediction_status=qc["prediction_status"]
            )
            record = {
                "schema": "pclass-dark.feature-record.v1",
                "extraction_id": manifest["id"],
                "sequence_id": index[1],
                "accession_id": accession,
                "length": length,
                "values": cached["values"],
                "qc": qvalues,
                "device": qc["device"],
            }
            validate_record(record, expected, manifest["id"])
        counts["unique_sequences"] += 1
        counts["with_noncanonical"] += int(qc["noncanonical_count"] > 0)
        counts["no_canonical"] += int(qc["canonical_count"] == 0)
        counts["with_undefined_features"] += int(bool(undefined))
        counts["rsa_out_of_range_residues"] += qc["rsa_out_of_range_count"]
        counts["scd_fallbacks"] += int(qc["scd_method"] == "direct_fallback")
    counts["retained_clusters"] = len(clusters)
    if not counts["unique_sequences"] or counts != manifest["counts"]:
        raise IntegrityError("Feature report count mismatch")
    if (
        db is not None
        and counts["unique_sequences"]
        != db.execute("SELECT COUNT(*) FROM sequences").fetchone()[0]
    ):
        raise IntegrityError("Incomplete prepared sequence coverage")
    for name, count in zip(
        family.names[:3],
        (counts["memberships"], counts["unique_sequences"], counts["unique_sequences"]),
        strict=True,
    ):
        if manifest["files"][name]["rows"] != count:
            raise IntegrityError("Manifest table row count mismatch")
    return manifest


def archive_directory(family):
    family.archives.mkdir(exist_ok=True)
    name = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    )
    path = safe_file(family.archives, name)
    path.mkdir()
    return path


def atomic_copy(source, target):
    fd, name = tempfile.mkstemp(prefix=".feature-copy-", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        sync_dir(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def inventory(family, directory):
    return {n: file_record(safe_file(directory, n)) for n in family.names}


def recover(family):
    journal_path = safe_file(family.work, "publication.json")
    if not journal_path.exists():
        return
    journal = read_json(journal_path)
    if (
        journal.get("schema") != "pclass-dark.feature-publication.v1"
        or journal.get("family") != str(family.output)
        or journal.get("state") not in ("installing", "committed")
        or set(journal["new"]) != set(family.names)
    ):
        raise IntegrityError("Unrecognized feature publication journal")
    new, old = journal["new"], journal["old"]
    if not isinstance(journal.get("extraction_id"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", journal["extraction_id"]
    ):
        raise IntegrityError("Invalid publication extraction identity")
    for records in (new, old):
        if records is not None:
            for record in records.values():
                if (
                    not isinstance(record, dict)
                    or set(record) != {"sha256", "size_bytes"}
                    or not isinstance(record["sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                    or type(record["size_bytes"]) is not int
                    or record["size_bytes"] < 0
                ):
                    raise IntegrityError("Malformed publication file inventory")
    if old is not None and set(old) != set(family.names):
        raise IntegrityError("Invalid prior feature publication inventory")
    for name in family.names:
        path = family.path(name)
        if path.exists() and file_record(path) not in (
            new[name],
            None if old is None else old[name],
        ):
            raise IntegrityError(
                "Third-party mutation prevents feature publication recovery"
            )
    if all(
        family.path(n).exists() and file_record(family.path(n)) == new[n]
        for n in family.names
    ):
        if verify_family(family)["id"] != journal["extraction_id"]:
            raise IntegrityError("Published extraction differs from journal")
    elif old is not None:
        archive = journal["archive"]
        if not isinstance(archive, str) or not re.fullmatch(
            r"\d{8}T\d{6}Z-[0-9a-f]{32}", archive
        ):
            raise IntegrityError("Unsafe feature archive path")
        source = safe_file(family.archives, archive)
        if inventory(family, source) != old:
            raise IntegrityError("Corrupt feature publication backup")
        verify_family(family, directory=source)
        for name in family.names:
            target = family.path(name)
            if target.exists() and file_record(target) not in (new[name], old[name]):
                raise IntegrityError(
                    "Concurrent modification during report restoration"
                )
            atomic_copy(source / name, target)
        verify_family(family)
    else:
        if journal.get("archive") is not None:
            raise IntegrityError("Invalid first-publication archive")
        for name in family.names:
            target = family.path(name)
            if target.exists() and file_record(target) != new[name]:
                raise IntegrityError("Concurrent modification during report recovery")
            target.unlink(missing_ok=True)
        sync_dir(family.parent)
    journal_path.unlink()
    sync_dir(family.work)


def publish_family(family):
    stage = safe_file(family.work, "stage")
    manifest = verify_family(family, directory=stage)
    if manifest is None:
        raise IntegrityError("Missing staged feature report")
    previous = verify_family(family)
    old = None
    backup = None
    if previous is not None:
        backup = archive_directory(family)
        for name in family.names:
            atomic_copy(family.path(name), backup / name)
        old = inventory(family, backup)
        verify_family(family, directory=backup)
    journal = {
        "schema": "pclass-dark.feature-publication.v1",
        "family": str(family.output),
        "extraction_id": manifest["id"],
        "new": inventory(family, stage),
        "old": old,
        "archive": None if backup is None else backup.name,
        "state": "installing",
    }
    marker = safe_file(family.work, "publication.json")
    write_json(marker, journal)
    try:
        for name in family.names:
            target = family.path(name)
            if target.exists() and (old is None or file_record(target) != old[name]):
                raise IntegrityError(
                    "Concurrent modification during report publication"
                )
            os.replace(stage / name, target)
            sync_dir(family.parent)
        verify_family(family)
        journal["state"] = "committed"
        write_json(marker, journal)
        marker.unlink()
        sync_dir(family.work)
    except BaseException:
        recover(family)
        raise
