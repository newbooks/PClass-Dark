"""Content-addressed cohorts and recoverable single-table publication."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from pathlib import Path

from .artifacts import canonical, identity
from .external_evaluation_types import (
    RESIDUES,
    EvaluationIntegrityError,
    EvaluationRecoveryError,
    digest_id,
    go_set,
    query_id,
)
from .external_sequence_io import now as now
from .external_sequence_io import record, sync_dir, sync_file, write_json


def read(path):
    try:
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(
                f,
                parse_constant=lambda v: (_ for _ in ()).throw(
                    ValueError(f"Nonfinite JSON: {v}")
                ),
            )
        if (
            not isinstance(data, dict)
            or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1
        ):
            raise ValueError("unknown schema")
        return data
    except (OSError, ValueError, TypeError) as exc:
        raise EvaluationIntegrityError(
            f"Invalid/missing manifest {path}: {exc}"
        ) from exc


def checked(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or record(path) != expected:
        raise EvaluationIntegrityError(f"Missing/corrupt committed artifact: {path}")
    return path


def safe_path(state, relative):
    if not isinstance(relative, str):
        raise EvaluationIntegrityError("Non-string cache path")
    parts = Path(relative).parts
    if (
        not parts
        or Path(relative).is_absolute()
        or any(p in {".", ".."} for p in parts)
    ):
        raise EvaluationIntegrityError("Unsafe cache path")
    p = state.joinpath(*parts)
    if any(q.is_symlink() for q in (p, *p.parents)):
        raise EvaluationIntegrityError(f"Cache symlink: {p}")
    return p


def verify_files(directory, manifest):
    for name, expected in manifest.get("files", {}).items():
        checked(safe_path(directory, name), expected)


def publish_directory(staged, target, manifest):
    target.parent.mkdir(parents=True, exist_ok=True)
    for p in staged.iterdir():
        if p.is_file():
            sync_file(p)
    write_json(staged / "manifest.json", manifest)
    sync_dir(staged)
    if target.exists():
        existing = read(target / "manifest.json")
        verify_files(target, existing)
        # Matching identity, not provenance timestamps or SQLite byte layout.
        if existing.get("id") != manifest.get("id"):
            raise EvaluationIntegrityError(f"Conflicting immutable artifact: {target}")
        shutil.rmtree(staged)
    else:
        os.replace(staged, target)
        sync_dir(target.parent)


def snapshot_inputs(
    fasta,
    truth_tsv,
    state,
    *,
    truth_resolver=None,
    truth_source=None,
    accession_resolver=None,
):
    sources = {str(fasta): record(fasta)}
    if truth_tsv is not None:
        sources[str(truth_tsv)] = record(truth_tsv)
    staged = state / (".cohort-" + uuid.uuid4().hex)
    staged.mkdir()
    db = sqlite3.connect(staged / "queries.sqlite")
    try:
        db.executescript("""
        CREATE TABLE queries(ordinal INTEGER PRIMARY KEY,query_id TEXT UNIQUE,
            sequence TEXT,length INTEGER,sequence_sha256 TEXT);
        CREATE TABLE truth(query_id TEXT PRIMARY KEY,go_ids_json TEXT);
        CREATE TABLE supplied(query_id TEXT PRIMARY KEY,go_ids_json TEXT);
        """)
        if accession_resolver is not None:
            db.execute(
                "CREATE TABLE accessions(ordinal INTEGER PRIMARY KEY, "
                "query_id TEXT UNIQUE, primary_accession TEXT UNIQUE)"
            )
        if truth_tsv is not None:
            with truth_tsv.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                if reader.fieldnames != ["Query_id", "go_mf_terms"]:
                    raise ValueError(
                        "Truth TSV must have Query_id and go_mf_terms headers"
                    )
                for row in reader:
                    if None in row or any(v is None for v in row.values()):
                        raise ValueError("Malformed truth TSV row")
                    qid = query_id(row["Query_id"])
                    terms = go_set(row["go_mf_terms"].split("|"), nonempty=True)
                    db.execute(
                        "INSERT INTO supplied VALUES (?,?)", (qid, json.dumps(terms))
                    )
        input_digest = hashlib.sha256()
        truth_digest = hashlib.sha256()
        count = 0
        header = None
        seq = []
        seq_length = 0
        with (staged / "headers.jsonl").open("wb") as hf:

            def finish():
                nonlocal count
                if header is None:
                    return
                words = header.split()
                if not words:
                    raise ValueError("Empty FASTA ID")
                qid = query_id(words[0])
                sequence = "".join(seq)
                if not sequence or not sequence.isascii() or set(sequence) - RESIDUES:
                    raise ValueError(f"Invalid/empty sequence: {qid}")
                if len(sequence) > 1000000 or len((">" + qid + "\n").encode()) > 4096:
                    raise ValueError(f"Overlong query: {qid}")
                attrs = [
                    w.partition("=")[2]
                    for w in words[1:]
                    if w.startswith("go_mf_terms=")
                ]
                if len(attrs) > 1:
                    raise ValueError(f"Duplicate truth header attribute: {qid}")
                terms = go_set(attrs[0].split("|"), nonempty=True) if attrs else None
                supplied = db.execute(
                    "SELECT go_ids_json FROM supplied WHERE query_id=?", (qid,)
                ).fetchone()
                if supplied:
                    explicit = tuple(json.loads(supplied[0]))
                    if terms is not None and terms != explicit:
                        raise ValueError(f"Conflicting truth sources: {qid}")
                    terms = explicit
                elif truth_tsv is not None:
                    raise ValueError(f"Missing truth TSV query: {qid}")
                if truth_resolver is not None:
                    terms = truth_resolver(qid, terms)
                if not terms:
                    raise ValueError(f"Missing GO MF truth: {qid}")
                digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
                input_digest.update(
                    canonical(
                        {
                            "query_id": qid,
                            "sequence_sha256": digest,
                            "length": len(sequence),
                        }
                    )
                    + b"\n"
                )
                truth_digest.update(
                    canonical({"query_id": qid, "go_ids": terms}) + b"\n"
                )
                if accession_resolver is not None:
                    db.execute(
                        "INSERT INTO accessions VALUES (?,?,?)",
                        (count, qid, accession_resolver(qid)),
                    )
                db.execute(
                    "INSERT INTO queries VALUES (?,?,?,?,?)",
                    (count, qid, sequence, len(sequence), digest),
                )
                db.execute("INSERT INTO truth VALUES (?,?)", (qid, json.dumps(terms)))
                hf.write(
                    canonical({"ordinal": count, "query_id": qid, "header": header})
                    + b"\n"
                )
                count += 1
                if count % 1000 == 0:
                    db.commit()

            with fasta.open(encoding="utf-8") as f:
                for raw in f:
                    line = raw.rstrip("\r\n")
                    if line.startswith(">"):
                        finish()
                        header = line[1:]
                        seq = []
                        seq_length = 0
                    elif line:
                        if header is None:
                            raise ValueError("Sequence before FASTA header")
                        if not line.isascii():
                            raise ValueError("Non-ASCII sequence")
                        seq_length += len(line)
                        if seq_length > 1000000:
                            raise ValueError("Sequence exceeds 1000000 residues")
                        seq.append(line.upper())
                finish()
        if count == 0:
            raise ValueError("Empty evaluation FASTA")
        if db.execute(
            "SELECT COUNT(*) FROM supplied WHERE query_id NOT IN "
            "(SELECT query_id FROM queries)"
        ).fetchone()[0]:
            raise ValueError("Extra truth TSV query IDs")
        input_id = input_digest.hexdigest()
        truth_key = {
            "input_id": input_id,
            "truth_rows_sha256": truth_digest.hexdigest(),
        }
        if truth_source is not None:
            truth_key["source"] = truth_source
        truth_id = identity(truth_key)
        cohort_key = {
            "input_id": input_id,
            "truth_id": truth_id,
            "fasta_sha256": sources[str(fasta)]["sha256"],
            "truth_file_sha256": sources[str(truth_tsv)]["sha256"]
            if truth_tsv
            else None,
            "headers_sha256": record(staged / "headers.jsonl")["sha256"],
        }
        cohort_id = identity(cohort_key)
        for table in ["truth", "supplied"]:
            if table == "truth":
                truth_stage = state / (".truth-" + uuid.uuid4().hex)
                truth_stage.mkdir()
                with sqlite3.connect(truth_stage / "truth.sqlite") as out:
                    out.execute(
                        "CREATE TABLE truth(query_id TEXT PRIMARY KEY,go_ids_json TEXT)"
                    )
                    out.executemany(
                        "INSERT INTO truth VALUES (?,?)",
                        db.execute("SELECT * FROM truth"),
                    )
                publish_directory(
                    truth_stage,
                    state / "truth" / truth_id,
                    {
                        "schema_version": 1,
                        "id": truth_id,
                        "input_id": input_id,
                        **(
                            {"source": truth_source} if truth_source is not None else {}
                        ),
                        "files": {"truth.sqlite": record(truth_stage / "truth.sqlite")},
                    },
                )
            db.execute(f"DROP TABLE {table}")
        db.commit()
        db.close()
        header_stage = state / (".headers-" + uuid.uuid4().hex)
        header_stage.mkdir()
        os.replace(staged / "headers.jsonl", header_stage / "headers.jsonl")
        publish_directory(
            header_stage,
            state / "cohorts" / cohort_id,
            {
                "schema_version": 1,
                "id": cohort_id,
                **cohort_key,
                "sources": sources,
                "files": {"headers.jsonl": record(header_stage / "headers.jsonl")},
            },
        )
        publish_directory(
            staged,
            state / "inputs" / input_id,
            {
                "schema_version": 1,
                "id": input_id,
                "query_count": count,
                "files": {"queries.sqlite": record(staged / "queries.sqlite")},
            },
        )
        return {
            **cohort_key,
            "cohort_id": cohort_id,
            "query_count": count,
            "sources": sources,
        }
    except sqlite3.IntegrityError as exc:
        raise ValueError("Duplicate input/truth query ID") from exc
    finally:
        db.close()
        if staged.exists():
            shutil.rmtree(staged)


def source_unchanged(cohort):
    for name, expected in cohort["sources"].items():
        checked(Path(name), expected)


def publication_paths(output):
    return (
        output.with_name("." + output.name + ".evaluation.json"),
        output.with_name("." + output.name + ".evaluation-journal.json"),
    )


def owner_check(output, state):
    marker, _ = publication_paths(output)
    if marker.exists():
        m = read(marker)
        if m["workspace_uuid"] != read(state / "workspace.json")["workspace_uuid"]:
            raise EvaluationIntegrityError("Output belongs to another workspace")
        checked(output, m["tsv"])
        return m
    if output.exists():
        raise EvaluationIntegrityError("Unowned output; choose another --output path")
    return None


def _recover_publication(output, state):
    marker, journal = publication_paths(output)
    if not journal.exists():
        return
    j = read(journal)
    if j.get("workspace_uuid") != read(state / "workspace.json")["workspace_uuid"]:
        raise EvaluationRecoveryError("Foreign publication journal")
    old, new = j["old"], j["new"]
    current = record(output) if output.is_file() else None
    old_rec = old["tsv"] if old else None
    if current not in [old_rec, new["tsv"]]:
        raise EvaluationRecoveryError("Output modified during interrupted publication")
    marker_value = read(marker) if marker.exists() else None
    if marker_value not in [old, new]:
        raise EvaluationRecoveryError("Marker modified during interrupted publication")
    score_dir = state / "scores" / digest_id(new["score_id"])
    checked(score_dir / "manifest.json", new["score_manifest"])
    score = read(score_dir / "manifest.json")
    verify_files(score_dir, score)
    if any(
        j["selection"].get(k) != score.get(k)
        for k in [
            "input_id",
            "truth_id",
            "pclass_generation",
            "dark_generation",
            "graph_id",
            "mapping_id",
        ]
    ):
        raise EvaluationRecoveryError(
            "Journal selection does not match scored manifest"
        )
    if j["selection"].get("score_id") != new["score_id"]:
        raise EvaluationRecoveryError("Journal score identity mismatch")
    if current == new["tsv"] and marker_value == new:
        write_json(state / "selection.json", j["selection"])
    else:
        backup = safe_path(state, j["backup"])
        expected_parent = state / "publications" / identity(str(output))
        if backup.parent != expected_parent:
            raise EvaluationRecoveryError("Foreign publication backup path")
        required = {"table", "marker"} if old else set()
        if j["old_selection"] is not None:
            required.add("selection.json")
        if set(j["files"]) != required:
            raise EvaluationRecoveryError("Incomplete publication backup inventory")
        for name, expected in j["files"].items():
            checked(safe_path(backup, name), expected)
        if old and (
            read(backup / "marker") != old or j["files"]["table"] != old["tsv"]
        ):
            raise EvaluationRecoveryError("Inconsistent publication rollback marker")
        if (
            j["old_selection"] is not None
            and read(backup / "selection.json") != j["old_selection"]
        ):
            raise EvaluationRecoveryError("Inconsistent publication rollback selection")
        if old:
            restore = output.with_name("." + output.name + ".restore")
            shutil.copyfile(backup / "table", restore)
            sync_file(restore)
            os.replace(restore, output)
            write_json(marker, old)
        else:
            output.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
        if j["old_selection"] is None:
            (state / "selection.json").unlink(missing_ok=True)
        else:
            write_json(state / "selection.json", j["old_selection"])
    sync_dir(output.parent)
    journal.unlink()
    sync_dir(output.parent)


def recover_publication(output, state):
    try:
        return _recover_publication(output, state)
    except EvaluationRecoveryError:
        raise
    except (EvaluationIntegrityError, ValueError, KeyError, TypeError, OSError) as exc:
        raise EvaluationRecoveryError(
            f"Cannot recover evaluation publication: {exc}"
        ) from exc


def publish_table(
    output,
    state,
    score_dir,
    score_manifest,
    *,
    table_name="external_evaluation.tsv",
    evaluation_kind="external",
):
    old = owner_check(output, state)
    marker, journal = publication_paths(output)
    backup = state / "publications" / identity(str(output)) / uuid.uuid4().hex
    backup.mkdir(parents=True)
    old_selection = (
        read(state / "selection.json") if (state / "selection.json").exists() else None
    )
    files = {}
    if old:
        for name, path in [("table", output), ("marker", marker)]:
            shutil.copyfile(path, backup / name)
            sync_file(backup / name)
            files[name] = record(backup / name)
    if old_selection:
        write_json(backup / "selection.json", old_selection)
        files["selection.json"] = record(backup / "selection.json")
    sync_dir(backup)
    new = {
        "schema_version": 1,
        "workspace_uuid": read(state / "workspace.json")["workspace_uuid"],
        "score_id": score_manifest["id"],
        "score_manifest": record(score_dir / "manifest.json"),
        "tsv": record(score_dir / table_name),
    }
    if evaluation_kind == "swiss":
        new.update(
            marker_version=2,
            state_relative=os.path.relpath(state, output.parent),
            evaluation_kind=evaluation_kind,
        )
    selection = {
        "schema_version": 1,
        **{
            k: score_manifest[k]
            for k in [
                "input_id",
                "truth_id",
                "pclass_generation",
                "dark_generation",
                "graph_id",
                "mapping_id",
            ]
        },
        "score_id": score_manifest["id"],
        "score_manifest": new["score_manifest"],
    }
    staged = output.with_name("." + output.name + "." + uuid.uuid4().hex + ".tmp")
    shutil.copyfile(score_dir / table_name, staged)
    sync_file(staged)
    write_json(
        journal,
        {
            "schema_version": 1,
            "workspace_uuid": new["workspace_uuid"],
            "old": old,
            "new": new,
            "selection": selection,
            "old_selection": old_selection,
            "backup": str(backup.relative_to(state)),
            "files": files,
        },
    )
    os.replace(staged, output)
    sync_dir(output.parent)
    write_json(marker, new)
    write_json(state / "selection.json", selection)
    recover_publication(output, state)
