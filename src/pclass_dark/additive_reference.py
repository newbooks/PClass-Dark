"""Frozen, snapshot-bound additive references and recipes (SPEC-0026)."""

from __future__ import annotations

import copy
import csv
import gzip
import math
import sqlite3
import tempfile
from pathlib import Path

from .artifacts import (
    IntegrityError,
    file_record,
    identity,
    read_json,
    safe_file,
    schema_checked,
    sha256,
    write_json,
)
from .ml_data import FEATURES, PAIR_HEADER, load_similarity, rows

SOURCE_SCHEMA = "pclass-dark.additive-source.v1"
REFERENCE_SCHEMA = "pclass-dark.additive-reference.v1"
RECIPE_SCHEMA = "pclass-dark.additive-recipe.v1"
DEFAULT_RECIPE = dict(
    schema=RECIPE_SCHEMA,
    configuration="hard",
    epochs=100,
    knots=8,
    degree=3,
    learning_rate=0.001,
    weight_decay=0.0001,
    optimizer="adamw",
    optimizer_eps=1e-8,
    batch_size=1024,
    seed=42,
    alpha=0.0,
    neighbors=10,
    similarity_power=2.0,
    feature_precision=".3f",
    features=list(FEATURES),
    identity_conflicts="exclude",
)


def validate_recipe(recipe):
    """Return a detached validated complete recipe; reject unknown options."""
    if not isinstance(recipe, dict) or set(recipe) != set(DEFAULT_RECIPE):
        raise ValueError("Invalid additive recipe fields")
    r = copy.deepcopy(recipe)
    for key, lower in [
        ("epochs", 1),
        ("knots", 2),
        ("batch_size", 1),
        ("seed", 0),
        ("neighbors", 1),
    ]:
        if type(r[key]) is not int or r[key] < lower:
            raise ValueError(f"Invalid recipe {key}")
    for key, positive in [
        ("learning_rate", True),
        ("weight_decay", False),
        ("optimizer_eps", True),
        ("similarity_power", True),
        ("alpha", False),
    ]:
        v = r[key]
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v < 0
            or (positive and v == 0)
        ):
            raise ValueError(f"Invalid recipe {key}")
    if (
        r["schema"] != RECIPE_SCHEMA
        or r["configuration"] not in ("hard", "siac")
        or type(r["degree"]) is not int
        or r["degree"] != 3
        or r["optimizer"] != "adamw"
        or r["optimizer_eps"] != 1e-8
        or r["feature_precision"] != ".3f"
        or r["features"] != list(FEATURES)
        or r["identity_conflicts"] != "exclude"
        or (r["configuration"] == "hard" and r["alpha"] != 0)
        or (r["configuration"] == "siac" and not 0 < r["alpha"] < 1)
    ):
        raise ValueError("Unsupported/inconsistent additive recipe")
    return r


def resource_root():
    return Path(__file__).parent / "resources/additive-reference/v1"


@schema_checked
def verify_reference(path):
    """Verify all owned reference bytes and scientific identity."""
    path = Path(path)
    m = read_json(path / "manifest.json")
    if (
        m.get("schema") != REFERENCE_SCHEMA
        or m.get("status") != "complete"
        or identity(m["identity_inputs"]) != m["id"]
    ):
        raise IntegrityError("Invalid additive reference manifest")
    for name, record in m["files"].items():
        if file_record(safe_file(path, name)) != record:
            raise IntegrityError(f"Changed additive reference: {name}")
    if {p.name for p in path.iterdir()} != set(m["files"]) | {"manifest.json"}:
        raise IntegrityError("Unknown additive reference artifact")
    validate_recipe(read_json(path / "recipe.json"))
    return m


@schema_checked
def prepare_additive_reference(
    setup_root, prepared_dir, *, source=None, source_manifest=None, recipe=None
):
    """Import a bound six-column table and publish a companion selection.

    Raises ValueError/IntegrityError on invalid recipes/source; OSError on IO.
    Uses disk-backed ordering, and never modifies the source or prepared bundle.
    Caller holds the setup transaction guard.
    """
    root, prepared = Path(setup_root), Path(prepared_dir)
    pm = read_json(prepared / "manifest.json")
    if (source is None) != (source_manifest is None):
        raise ValueError("Provide both cluster similarity and its source manifest")
    recipe_value = validate_recipe(
        read_json(recipe) if recipe else dict(DEFAULT_RECIPE)
    )
    selection_path = root / "additive-selection.json"
    if source is None and selection_path.exists():
        selected = read_json(selection_path)
        previous = safe_file(root, selected["reference"])
        m = verify_reference(previous)
        if sha256(previous / "manifest.json") != selected["reference_manifest_sha256"]:
            raise IntegrityError("Changed additive reference selection")
        if (
            m["identity_inputs"]["prepared_manifest_sha256"]
            == sha256(prepared / "manifest.json")
            and recipe is None
            and m["identity_inputs"].get("canonicalization") == "retained-pairs-v1"
        ):
            return previous
    packaged = source is None
    if packaged:
        folder = resource_root()
        source = folder / "cluster_dmatrix.tsv.gz"
        source_manifest = folder / "source.json"
        if recipe is None:
            recipe_value = validate_recipe(read_json(folder / "recipe.json"))
    sm = read_json(source_manifest)
    if sm.get("schema") != SOURCE_SCHEMA or sm["source_reference"] != pm["reference"]:
        raise IntegrityError("Similarity source reference does not match preparation")
    source = Path(source)
    members = read_json(safe_file(Path(source_manifest).parent, sm["members_path"]))
    descriptions = read_json(
        safe_file(Path(source_manifest).parent, sm["descriptions_path"])
    )
    for field in ("members", "descriptions"):
        actual = file_record(
            safe_file(Path(source_manifest).parent, sm[field + "_path"])
        )
        if actual != sm[field + "_file"]:
            raise IntegrityError("Changed source metadata")
    source_record = file_record(source)
    expected = (
        sm["compressed_file"] if packaged or source.suffix == ".gz" else sm["raw_file"]
    )
    if source_record != expected:
        raise IntegrityError(
            "Changed/missing frozen similarity asset; materialize Git LFS data"
        )
    with (prepared / "clusters.tsv").open() as f:
        labels = tuple(
            sorted(
                r["cluster_id"]
                for r in csv.DictReader(f, delimiter="\t")
                if r.get("retained", "true") == "true"
            )
        )
    if set(labels) - members.keys():
        raise IntegrityError("Missing retained medoid metadata")
    parent = root / "additive-reference"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=parent) as tmp:
        stage = Path(tmp)
        raw = source
        if source.suffix == ".gz":
            raw = stage / "source.tsv"
            with gzip.open(source, "rb") as inp, raw.open("wb") as out:
                total = 0
                for chunk in iter(lambda: inp.read(2**20), b""):
                    total += len(chunk)
                    if total > sm["raw_file"]["size_bytes"]:
                        raise IntegrityError("Frozen source expansion limit exceeded")
                    out.write(chunk)
        if file_record(raw) != sm["raw_file"]:
            raise IntegrityError("Raw matrix checksum mismatch")
        db = sqlite3.connect(stage / "sort.sqlite")
        try:
            db.execute("PRAGMA cache_size=-65536")
            db.execute("PRAGMA max_page_count=524288")
            db.execute(
                "CREATE TABLE pairs(a TEXT,b TEXT,ma TEXT,mb TEXT,d "
                "REAL,s REAL,PRIMARY KEY(a,b)) WITHOUT ROWID"
            )
            keep = set(labels)
            count = 0
            for a, ma, b, mb, d, s in rows(raw, PAIR_HEADER):
                d, s = float(d), float(s)
                if (
                    a == b
                    or not all(math.isfinite(v) and 0 <= v <= 1 for v in (d, s))
                    or abs(d + s - 1) > 1.00000001e-5
                ):
                    raise IntegrityError("Invalid similarity scores/pair")
                for c, medoid in ((a, ma), (b, mb)):
                    if c not in members or members[c]["medoid_accession"] != medoid:
                        raise IntegrityError("Source medoid mismatch")
                if b < a:
                    a, b, ma, mb = b, a, mb, ma
                # Detect duplicates even for rows later excluded from the vocabulary.
                try:
                    db.execute(
                        "INSERT INTO pairs VALUES(?,?,?,?,?,?)", (a, b, ma, mb, d, s)
                    )
                except sqlite3.IntegrityError as exc:
                    raise IntegrityError("Duplicate similarity pair") from exc
                count += 1
            db.commit()
            output = stage / "cluster_dmatrix.tsv"
            with output.open("w", newline="") as f:
                writer = csv.writer(f, delimiter="\t", lineterminator="\n")
                writer.writerow(PAIR_HEADER)
                retained = 0
                for a, b, ma, mb, d, s in db.execute(
                    "SELECT * FROM pairs ORDER BY a,b"
                ):
                    if a in keep and b in keep:
                        writer.writerow(
                            (a, ma, b, mb, format(d, ".17g"), format(s, ".17g"))
                        )
                        retained += 1
            if retained != len(labels) * (len(labels) - 1) // 2:
                raise IntegrityError("Incomplete retained target similarity coverage")
        finally:
            db.close()
        (stage / "sort.sqlite").unlink()
        if raw != source:
            raw.unlink()
        load_similarity(output, labels=labels)
        with (stage / "clusters.tsv").open("w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(("cluster_id", "medoid_accession"))
            w.writerows((c, members[c]["medoid_accession"]) for c in labels)
        write_json(
            stage / "descriptions.json", {c: descriptions.get(c) for c in labels}
        )
        write_json(stage / "recipe.json", recipe_value)
        inputs = dict(
            canonicalization="retained-pairs-v1",
            preparation_id=pm["id"],
            prepared_manifest_sha256=sha256(prepared / "manifest.json"),
            source=sm,
            recipe_id=identity(recipe_value),
        )
        files = {p.name: file_record(p) for p in stage.iterdir()}
        inputs["files"] = files
        rid = identity(inputs)
        target = parent / rid
        write_json(
            stage / "manifest.json",
            dict(
                schema=REFERENCE_SCHEMA,
                status="complete",
                id=rid,
                identity_inputs=inputs,
                files=files,
                counts=dict(source_pairs=count, retained_pairs=retained),
            ),
        )
        if file_record(source) != source_record:
            raise IntegrityError("Similarity source changed during import")
        if target.exists():
            verify_reference(target)
        else:
            # Same-filesystem atomic installation under the setup guard.
            stage.rename(target)
        verify_reference(target)
        write_json(
            selection_path,
            dict(
                schema="pclass-dark.additive-selection.v1",
                preparation_id=pm["id"],
                prepared_manifest_sha256=sha256(prepared / "manifest.json"),
                reference_id=rid,
                reference=str(target.relative_to(root)),
                reference_manifest_sha256=sha256(target / "manifest.json"),
                recipe_id=identity(recipe_value),
            ),
        )
        return target
