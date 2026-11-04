"""Frozen primary-accession truth lookup using profunc's function-query source."""

import json
import shutil
import sqlite3
import uuid
from pathlib import Path

from . import external_evaluation_store as store
from .artifacts import identity
from .external_evaluation_types import go_set
from .external_sequence_data import ACCESSION


def primary_accession(query_id):
    """Resolve bare or canonical Swiss-Prot IDs; never strip isoform suffixes."""
    accession = query_id
    if "|" in accession:
        parts = accession.split("|")
        if len(parts) != 3 or parts[0] != "sp" or not parts[2]:
            raise ValueError(f"Invalid Swiss-Prot header: {query_id}")
        accession = parts[1]
    if not ACCESSION.fullmatch(accession):
        raise ValueError(f"Invalid primary accession: {query_id}")
    return accession


def freeze_annotations(state):
    """Copy one validated profunc dataset's mapping and manifest before lookup."""
    from profunc.go.sets import load_accession_mf_go
    from profunc.runtime import resolve_runtime

    runtime = resolve_runtime()
    dataset = runtime.dataset
    manifest = json.loads((dataset / "manifest.json").read_text())
    expected = next(
        m for m in manifest["members"] if m["path"] == "accession_mf_go.tsv"
    )
    expected = {"size_bytes": expected["size"], "sha256": expected["sha256"]}
    source = dataset / "accession_mf_go.tsv"
    store.checked(source, expected)
    contract = {
        "resolver": "profunc-direct-mf-v1",
        "dataset": runtime.manifest,
        "mapping": expected,
        "manifest": store.record(dataset / "manifest.json"),
    }
    sid = identity(contract)
    target = state / "annotations" / sid
    if not target.exists():
        staged = state / (".annotations-" + uuid.uuid4().hex)
        staged.mkdir()
        try:
            shutil.copyfile(source, staged / "accession_mf_go.tsv")
            shutil.copyfile(dataset / "manifest.json", staged / "source_manifest.json")
            store.checked(source, expected)
            store.checked(staged / "accession_mf_go.tsv", expected)
            store.checked(staged / "source_manifest.json", contract["manifest"])
            store.publish_directory(
                staged,
                target,
                {
                    "schema_version": 1,
                    "id": sid,
                    "contract": contract,
                    "files": {
                        n: store.record(staged / n)
                        for n in ("accession_mf_go.tsv", "source_manifest.json")
                    },
                },
            )
        finally:
            if staged.exists():
                shutil.rmtree(staged)
    store.verify_files(target, store.read(target / "manifest.json"))
    validate_annotation_rows(target / "accession_mf_go.tsv")
    return (
        {"id": sid, **contract},
        load_accession_mf_go(target / "accession_mf_go.tsv"),
        runtime,
    )


def validate_annotation_rows(path):
    """Reject ambiguities the profunc convenience loader would otherwise hide."""
    seen = set()
    with Path(path).open() as stream:
        for i, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"Malformed annotation row {i}")
            accession = primary_accession(fields[0])
            if accession in seen:
                raise ValueError(
                    f"Duplicate annotation accession at row {i}: {accession}"
                )
            seen.add(accession)
            go_set(fields[1].split(";"), nonempty=True)


def validate_headers(fasta, mapping):
    """Collect bounded diagnostics before producing an exact-coverage snapshot."""
    seen, errors, count = set(), [], 0
    with Path(fasta).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith(">"):
                continue
            words = line[1:].split()
            qid = words[0] if words else "<empty>"
            try:
                accession = primary_accession(qid)
                if accession in seen:
                    raise ValueError("duplicate normalized accession")
                seen.add(accession)
                if not mapping.get(accession):
                    raise ValueError("missing MF truth")
            except ValueError as exc:
                count += 1
                if len(errors) < 20:
                    errors.append(f"{qid}: {exc}")
    if count:
        raise ValueError(
            f"Invalid Swiss query annotations/IDs ({count}): " + "; ".join(errors)
        )


class TruthResolver:
    """Exact primary accession join; uniqueness is checked independently of IDs."""

    def __init__(self, mapping):
        self.mapping, self.seen = mapping, set()

    def __call__(self, qid, supplied):
        accession = primary_accession(qid)
        if accession in self.seen:
            raise ValueError(f"Duplicate normalized accession: {accession}")
        self.seen.add(accession)
        terms = go_set(self.mapping.get(accession, ()), nonempty=True)
        if supplied is not None and terms != supplied:
            raise ValueError(f"Conflicting profunc/header truth: {qid}")
        return terms


def saved_resolver(state, selection):
    folder = state / "truth" / selection["truth_id"]
    manifest = store.read(folder / "manifest.json")
    store.verify_files(folder, manifest)
    db = sqlite3.connect(folder / "truth.sqlite")

    def resolve(qid, supplied):
        row = db.execute(
            "SELECT go_ids_json FROM truth WHERE query_id=?", (qid,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No saved truth for query: {qid}")
        terms = go_set(json.loads(row[0]), nonempty=True)
        if supplied is not None and terms != supplied:
            raise ValueError(f"Conflicting saved/header truth: {qid}")
        return terms

    return resolve, manifest["source"], db
