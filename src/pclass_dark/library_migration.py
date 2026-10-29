"""Validated imports from legacy six-fit bundles and private final receipts."""

from __future__ import annotations

import copy
import csv
import gzip
import itertools
import json
import logging
from pathlib import Path

from .artifacts import (
    IntegrityError,
    identity,
    read_json,
    safe_file,
    sha256,
    write_json,
)
from .cluster_paths import stored_cluster_name
from .fit_records import validate_result
from .signatures import SELECTION_VERSION

LOG = logging.getLogger(__name__)


def prepare_sources(
    root, base, scopes, targets, *, include_folds=False, final_ids=None
):
    """Prove legacy hash preimages before handing sources to worker processes."""
    from .library import _load_library
    from .library_reports import _parse_profile, final_results

    engines = (
        "cpu-logdiff-reference-fallback-exact-count-v2",
        "cpu-left-logaddexp-exact-count-v1",
    )
    recipes = {}
    for scope in scopes:
        for engine in engines:
            recipe = copy.deepcopy(base)
            recipe["cluster_ids"] = sorted(scope)
            recipe["numerical"]["engine"] = engine
            recipes[identity(recipe)] = recipe
    sources = {}
    for path in sorted(root.iterdir()):
        if (
            not path.is_dir()
            or path.is_symlink()
            or not (path / "manifest.json").is_file()
        ):
            continue
        manifest = read_json(path / "manifest.json")
        if manifest.get("schema") == "pclass-dark.library.v2":
            matching = {
                c
                for c, f in manifest["identity_inputs"]["final_fits"]
                if final_ids is not None and c in targets and final_ids.get(c) == f
            }
            if not matching:
                continue
            _load_library(path)
            cache = safe_file(root, ".migration/" + sha256(path / "manifest.json"))
            for cluster, result in final_results(path, matching):
                directory = cache / ("c_" + cluster.encode().hex())
                directory.mkdir(parents=True, exist_ok=True)
                file = directory / "final.json"
                write_json(file, result)
                sources.setdefault((cluster, "final"), []).append(
                    {
                        "kind": "production",
                        "path": str(file),
                        "sha256": sha256(file),
                        "source_id": manifest["id"],
                        "recipe": manifest["identity_inputs"],
                        "source_manifest_sha256": sha256(path / "manifest.json"),
                    }
                )
            continue
        if manifest.get("schema") != "pclass-dark.library.v1":
            continue
        recipe = manifest["identity_inputs"]
        expected = copy.deepcopy(base)
        expected["cluster_ids"] = recipe.get("cluster_ids")
        expected["numerical"]["engine"] = recipe.get("numerical", {}).get("engine")
        if recipe != expected or expected["numerical"]["engine"] not in engines:
            continue
        _load_library(path)  # Published corruption is an error; never repair it.
        if manifest["id"] != identity(recipe):
            raise IntegrityError("Legacy manifest identity mismatch")
        recipes[manifest["id"]] = recipe
        selected = set(targets) & set(recipe["cluster_ids"])
        cache = safe_file(root, ".migration/" + sha256(path / "manifest.json"))
        for cluster, result in final_results(path, selected):
            directory = cache / ("c_" + cluster.encode().hex())
            directory.mkdir(parents=True, exist_ok=True)
            file = directory / "final.json"
            write_json(file, result)
            sources.setdefault((cluster, "final"), []).append(
                {
                    "kind": "published",
                    "path": str(file),
                    "sha256": sha256(file),
                    "source_id": manifest["id"],
                    "recipe": recipe,
                    "source_manifest_sha256": sha256(path / "manifest.json"),
                }
            )
        if include_folds:
            # Spool OOF by target/fold, not all 3443 targets in RAM.
            with gzip.open(
                path / "out_of_fold_positive_scores.tsv.gz", "rt", newline=""
            ) as handle:
                rows = csv.DictReader(handle, delimiter="\t")
                for (cluster, fold), group in itertools.groupby(
                    rows, key=lambda r: (r["cluster_id"], r["fold_id"])
                ):
                    if cluster not in selected:
                        continue
                    oof = []
                    signature_file = (
                        path
                        / "clusters"
                        / stored_cluster_name(cluster, manifest)
                        / f"fold_{fold}.json"
                    )
                    digest = sha256(signature_file)
                    for row in group:
                        if row["signature_digest"] != digest:
                            raise IntegrityError("Legacy OOF signature digest mismatch")
                        oof.append(
                            {
                                "window_id": row["window_id"],
                                "log_score": float(row["log_score"])
                                if row["log_score"]
                                else None,
                                "score": float(row["score"]) if row["score"] else None,
                                "accepted": row["accepted"] == "true"
                                if row["accepted"]
                                else None,
                                "status": row["status"],
                            }
                        )
                    directory = cache / ("c_" + cluster.encode().hex())
                    directory.mkdir(parents=True, exist_ok=True)
                    write_json(directory / f"oof_{fold}.json", oof)
            with gzip.open(path / "dimension_folds.tsv.gz", "rt", newline="") as handle:
                rows = csv.DictReader(handle, delimiter="\t")
                for (cluster, fold), group in itertools.groupby(
                    rows, key=lambda r: (r["cluster_id"], r["fold_id"])
                ):
                    if cluster not in selected:
                        continue
                    directory = (
                        path / "clusters" / stored_cluster_name(cluster, manifest)
                    )
                    with gzip.open(
                        directory / "selection_trace.json.gz", "rt"
                    ) as traces:
                        trace = json.load(traces)[int(fold)]
                    signature = read_json(directory / f"fold_{fold}.json")
                    oof_file = (
                        cache / ("c_" + cluster.encode().hex()) / f"oof_{fold}.json"
                    )
                    if oof_file.exists():
                        oof = read_json(oof_file)
                    elif signature["counts"]["validation_positive_n"] == 0:
                        oof = []
                    else:
                        raise IntegrityError(
                            "Missing legacy OOF rows for nonempty validation fold"
                        )
                    result = {
                        "signature": signature,
                        "profiles": [_parse_profile(r, False) for r in group],
                        "trace": trace,
                        "oof": oof,
                    }
                    file = cache / ("c_" + cluster.encode().hex()) / f"fold_{fold}.json"
                    write_json(file, result)
                    sources.setdefault((cluster, f"fold_{fold}"), []).append(
                        {
                            "kind": "published",
                            "path": str(file),
                            "sha256": sha256(file),
                            "source_id": manifest["id"],
                            "recipe": recipe,
                            "source_manifest_sha256": sha256(path / "manifest.json"),
                        }
                    )
    for source_id, recipe in recipes.items():
        directory = safe_file(root, ".checkpoints/" + source_id)
        if not directory.exists():
            continue
        for cluster in targets:
            for fit in (
                ["final", *[f"fold_{f}" for f in range(5)]]
                if include_folds
                else ["final"]
            ):
                path = safe_file(
                    directory, "c_" + cluster.encode().hex() + f"/{fit}/complete.json"
                )
                if path.is_file():
                    sources.setdefault((cluster, fit), []).append(
                        {
                            "kind": "private",
                            "path": str(path),
                            "source_id": source_id,
                            "recipe": recipe,
                        }
                    )
    known = set(recipes)
    private = root / ".checkpoints"
    if private.exists():
        for path in private.iterdir():
            if path.name not in known:
                LOG.info(
                    "Legacy checkpoint %s: recipe unproved; preserving without reuse",
                    path.name,
                )
    return sources


def import_source(
    source,
    *,
    cluster,
    fit,
    counts,
    validation_ids,
    input_identity,
    sigma_floor,
    normalized_inputs=None,
):
    """Read a proven recipe's result and verify target-specific dependencies."""
    recipe = source["recipe"]
    if identity(recipe) != source["source_id"]:
        raise IntegrityError("Unproved legacy recipe")
    file = Path(source["path"])
    if file.is_symlink():
        raise IntegrityError("Symlink migration source")
    if source["kind"] == "private":
        receipt = read_json(file)
        result = receipt["payload"]
        task = identity([source["source_id"], cluster, fit, input_identity])
        if (
            receipt["schema"] != "pclass-dark.library-checkpoint.v1"
            or receipt["task_id"] != task
            or receipt["dependency"] != identity([task, SELECTION_VERSION])
            or receipt["files"] != {}
            or receipt["arrays"] != {}
            or receipt["payload_id"] != identity(result)
        ):
            raise IntegrityError("Legacy complete receipt identity/dependency mismatch")
    else:
        if sha256(file) != source["sha256"]:
            raise IntegrityError("Migration source checksum mismatch")
        result = read_json(file)
    signature = result["signature"]
    if source["kind"] == "production":
        from .fit_records import payload_identity

        if (
            signature["fit_identity"] != identity(normalized_inputs)
            or signature["scientific_payload_id"] != payload_identity(result)
            or signature["library_id"] != source["source_id"]
        ):
            raise IntegrityError("Published production fit dependencies mismatch")
        validate_result(
            result,
            cluster=cluster,
            fit=fit,
            counts=counts,
            sigma_floor=sigma_floor,
            validation_ids=validation_ids,
        )
        return result
    if (
        signature["library_id"] != source["source_id"]
        or signature["input_identity"] != input_identity
        or recipe["algorithm"] != "library-" + SELECTION_VERSION
    ):
        raise IntegrityError("Legacy signature dependencies mismatch")
    validate_result(
        result,
        cluster=cluster,
        fit=fit,
        counts=counts,
        sigma_floor=sigma_floor,
        validation_ids=validation_ids,
    )
    return result
