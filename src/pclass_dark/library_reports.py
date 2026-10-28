"""Separate production/research schemas, deterministic reports and loaders."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from types import MappingProxyType

from . import __version__
from .artifacts import (
    IntegrityError,
    file_record,
    identity,
    read_json,
    safe_file,
    schema_checked,
    sha256,
    transaction_guard,
    verify_bundle,
    write_json,
)
from .fit_records import payload_identity, rebound_signature, validate_result
from .library_runtime import freeze, representation_contract, validate_signature
from .signature_metrics import METRICS

PRODUCTION = "pclass-dark.library.v2"
RESEARCH = "pclass-dark.library-research.v1"


def read_target(path):
    target = read_json(path)
    root = Path(path)
    while root.name not in {".production-checkpoints", ".research-checkpoints"}:
        if root.parent == root:
            raise IntegrityError("Invalid target checkpoint layout")
        root = root.parent
    root = root.parent
    result_path = safe_file(root, target["result"])
    if sha256(result_path) != target["result_sha256"]:
        raise IntegrityError("Target result receipt checksum mismatch")
    receipt = read_json(result_path)
    if receipt["payload_id"] != identity(receipt["payload"]):
        raise IntegrityError("Target fit payload checksum mismatch")
    return target, receipt["payload"]


def _write_metrics(table, cluster, signature):
    for population in (
        ("training",) if signature["fit_id"] == "final" else ("training", "validation")
    ):
        metrics = signature[f"{population}_metrics"]
        for metric in sorted(METRICS):
            table(
                {
                    "cluster_id": cluster,
                    "fit_id": signature["fit_id"],
                    "population": population,
                    "metric": metric,
                    "value": metrics.get(metric) if metrics else None,
                    "positive_n": signature["counts"][f"{population}_positive_n"],
                    "negative_n": signature["counts"]["background_n"],
                    "status": metrics["status"] if metrics else signature["status"],
                }
            )


def _manifest(stage, schema, inputs, contract, counts, execution):
    from .artifacts import sync_tree

    sync_tree(stage)
    manifest = {
        "schema": schema,
        "id": identity(inputs),
        "identity_inputs": inputs,
        "status": "complete",
        "representation_contract": contract,
        "counts": counts,
        "files": {
            str(p.relative_to(stage)): file_record(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        },
        "software_version": __version__,
        "execution": execution,
    }
    write_json(stage / "manifest.json", manifest)
    return manifest


def write_production(stage, paths, inputs, contract, execution):
    from .library import FINAL_PROFILE, HEADERS, _gzip_json, _table, encoded_cluster

    counts = {
        "targets": len(paths),
        "available_final": 0,
        "unsupported_final": 0,
        "dimension_final_rows": 0,
    }
    with (
        _table(
            stage / "dimension_final.tsv.gz", ["cluster_id", *FINAL_PROFILE]
        ) as dimensions,
        _table(
            stage / "cluster_metrics.tsv", HEADERS["cluster_metrics.tsv"]
        ) as metrics,
    ):
        for path in paths:
            target, result = read_target(path)
            cluster = target["cluster_id"]
            directory = stage / "clusters" / encoded_cluster(cluster)
            directory.mkdir(parents=True)
            signature = rebound_signature(
                result, identity(inputs), target["fit_id"], target.get("source")
            )
            write_json(directory / "final.json", signature)
            _gzip_json(directory / "selection_trace.json.gz", [result["trace"]])
            for row in result["profiles"]:
                dimensions({"cluster_id": cluster, **row})
            _write_metrics(metrics, cluster, signature)
            counts["dimension_final_rows"] += 640
            counts[
                "available_final"
                if signature["status"] == "ok"
                else "unsupported_final"
            ] += 1
    manifest = _manifest(stage, PRODUCTION, inputs, contract, counts, execution)
    manifest.update(
        evaluation_mode="all-C-in-sample",
        research_role="separate-optional-artifact",
        probability_calibrated=False,
    )
    write_json(stage / "manifest.json", manifest)
    return manifest


def _parse_profile(row, final):
    ints = {
        "dimension",
        "positive_n",
        "background_n",
        "rank",
        "training_auroc_numerator",
        "training_auroc_denominator",
        "validation_auroc_numerator",
        "validation_auroc_denominator",
    }
    result = {}
    for key, value in row.items():
        if key == "cluster_id" or key == "fold_id":
            continue
        if value == "":
            result[key] = None
        elif key in {"core", "candidate", "selected"}:
            if value not in {"true", "false"}:
                raise IntegrityError("Invalid profile boolean")
            result[key] = value == "true"
        elif key == "status":
            result[key] = value
        else:
            result[key] = int(value) if key in ints else float(value)
    if final:
        result.update(
            validation_auroc=None,
            validation_auroc_numerator=None,
            validation_auroc_denominator=0,
        )
    return result


def final_results(path, requested):
    """Stream a legacy/current final table once, preserving complete fit payloads."""
    import itertools

    from .library import encoded_cluster

    with gzip.open(path / "dimension_final.tsv.gz", "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for cluster, rows in itertools.groupby(reader, key=lambda r: r["cluster_id"]):
            rows = list(rows)
            if cluster not in requested:
                continue
            directory = path / "clusters" / encoded_cluster(cluster)
            signature = read_json(directory / "final.json")
            with gzip.open(directory / "selection_trace.json.gz", "rt") as trace_file:
                traces = json.load(trace_file)
            trace = next(t for t in traces if t["fit_id"] == "final")
            yield (
                cluster,
                {
                    "signature": signature,
                    "profiles": [_parse_profile(r, True) for r in rows],
                    "trace": trace,
                    "oof": [],
                },
            )


@schema_checked
def load_production(path):
    from .library import FINAL_PROFILE, HEADERS, Library, encoded_cluster

    manifest = verify_bundle(path, PRODUCTION)
    inputs = manifest["identity_inputs"]
    contract = representation_contract(manifest["representation_contract"])
    targets = inputs["cluster_ids"]
    if (
        manifest["id"] != identity(inputs)
        or targets != sorted(set(targets))
        or identity(contract) != inputs["representation_contract_id"]
    ):
        raise IntegrityError("Production identity/target/representation mismatch")
    required = {"dimension_final.tsv.gz", "cluster_metrics.tsv"}
    signatures = {}
    counts = {
        "targets": len(targets),
        "available_final": 0,
        "unsupported_final": 0,
        "dimension_final_rows": 640 * len(targets),
    }
    for cluster, result in final_results(path, set(targets)):
        signature = result["signature"]
        validate_signature(
            signature, library_id=manifest["id"], cluster_id=cluster, fit_id="final"
        )
        fit_id = dict(inputs["final_fits"])[cluster]
        if (
            signature["schema"] != "pclass-dark.signature.v2"
            or signature["fit_identity"] != fit_id
            or identity(signature["input_identity"]) != fit_id
            or signature["input_identity"]["cluster_id"] != cluster
            or signature["input_identity"]["representation_contract_id"]
            != identity(contract)
            or signature["scientific_payload_id"] != payload_identity(result)
            or signature["representation_contract_id"] != identity(contract)
            or signature["evaluation_mode"] != "all-C-in-sample"
        ):
            raise IntegrityError("Production signature/scientific payload mismatch")
        validate_result(
            result,
            cluster=cluster,
            fit="final",
            counts=signature["counts"],
            sigma_floor=inputs["sigma_floor"],
            validation_ids=[],
        )
        prefix = "clusters/" + encoded_cluster(cluster) + "/"
        with gzip.open(path / prefix / "selection_trace.json.gz", "rt") as handle:
            if len(json.load(handle)) != 1:
                raise IntegrityError("Production requires exactly one final trace")
        required.update({prefix + "final.json", prefix + "selection_trace.json.gz"})
        signatures[cluster] = freeze(signature)
        counts[
            "available_final" if signature["status"] == "ok" else "unsupported_final"
        ] += 1
    if (
        set(signatures) != set(targets)
        or set(manifest["files"]) != required
        or manifest["counts"] != counts
    ):
        raise IntegrityError("Production inventory/count mismatch")
    for name, header in {
        "dimension_final.tsv.gz": ["cluster_id", *FINAL_PROFILE],
        "cluster_metrics.tsv": HEADERS["cluster_metrics.tsv"],
    }.items():
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path / name, "rt", newline="") as handle:
            if next(csv.reader(handle, delimiter="\t"), None) != header:
                raise IntegrityError("Production report header mismatch")
    _verify_metrics(
        path / "cluster_metrics.tsv", {(c, "final"): s for c, s in signatures.items()}
    )
    return Library(
        manifest["id"],
        MappingProxyType(signatures),
        freeze(contract),
        freeze(counts),
        source_root=path.parent,
        manifest_sha256=sha256(path / "manifest.json"),
        source_path=path,
    )


def write_research(stage, paths, inputs, contract, execution, *, source_production):
    from contextlib import ExitStack

    from .library import HEADERS, STATS, _gzip_json, _stat, _table, encoded_cluster

    names = set(HEADERS) - {"dimension_final.tsv.gz"}
    counts = {
        "targets": len(paths),
        "valid_discovery_fits": 0,
        "unsupported_fits": 0,
        "dimension_fold_rows": 640 * 5 * len(paths),
        "oof_rows": 0,
    }
    with ExitStack() as stack:
        tables = {
            n: stack.enter_context(_table(stage / n, HEADERS[n])) for n in sorted(names)
        }
        for path in paths:
            target = read_json(path)
            cluster = target["cluster_id"]
            directory = stage / "clusters" / encoded_cluster(cluster)
            directory.mkdir(parents=True)
            fits, traces = [], []
            for reference in target["fits"]:
                _, result = read_target(reference)
                fits.append(result)
                signature = rebound_signature(
                    result, identity(inputs), read_json(reference)["fit_id"]
                )
                fit = signature["fit_id"]
                write_json(directory / f"{fit}.json", signature)
                digest = sha256(directory / f"{fit}.json")
                traces.append(result["trace"])
                counts[
                    "valid_discovery_fits"
                    if signature["status"] == "ok"
                    else "unsupported_fits"
                ] += 1
                for row in result["profiles"]:
                    tables["dimension_folds.tsv.gz"](
                        {"cluster_id": cluster, "fold_id": int(fit[-1]), **row}
                    )
                for row in result["oof"]:
                    tables["out_of_fold_positive_scores.tsv.gz"](
                        {
                            "cluster_id": cluster,
                            "fold_id": int(fit[-1]),
                            "signature_digest": digest,
                            **row,
                        }
                    )
                    counts["oof_rows"] += 1
                _write_metrics(tables["cluster_metrics.tsv"], cluster, signature)
            _gzip_json(directory / "selection_trace.json.gz", traces)
            valid = sum(f["signature"]["status"] == "ok" for f in fits)
            for d in range(640):
                rows = [f["profiles"][d] for f in fits]
                selected = sum(r["selected"] for r in rows)
                for statistic in sorted(STATS):
                    tables["dimension_summary.tsv.gz"](
                        {
                            "cluster_id": cluster,
                            "dimension": d,
                            "statistic": statistic,
                            **_stat([r[statistic] for r in rows]),
                            "selection_count": selected,
                            "selection_frequency": selected / valid if valid else None,
                            "core_count": sum(r["core"] for r in rows),
                        }
                    )
            for population in ("training", "validation"):
                for metric in sorted(METRICS):
                    values = [
                        f["signature"][population + "_metrics"].get(metric)
                        for f in fits
                        if f["signature"][population + "_metrics"] is not None
                        and f["signature"][population + "_metrics"]["status"] == "ok"
                    ]
                    tables["cluster_summary.tsv"](
                        {
                            "cluster_id": cluster,
                            "population": population,
                            "metric": metric,
                            **_stat(values),
                            "final_status": target["final_status"],
                        }
                    )
            tables["cluster_summary.tsv"](
                {
                    "cluster_id": cluster,
                    "population": "discovery",
                    "metric": "signature_size",
                    **_stat(
                        [
                            len(f["signature"]["selected_dimensions"])
                            for f in fits
                            if f["signature"]["status"] == "ok"
                        ]
                    ),
                    "final_status": target["final_status"],
                }
            )
    manifest = _manifest(stage, RESEARCH, inputs, contract, counts, execution)
    manifest["source_production"] = source_production
    write_json(stage / "manifest.json", manifest)
    return manifest


def _verify_metrics(path, signatures):
    from .library import HEADERS, _cell

    expected = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != HEADERS["cluster_metrics.tsv"]:
            raise IntegrityError("Metric header mismatch")
        for row in reader:
            key = row["cluster_id"], row["fit_id"], row["population"], row["metric"]
            if (
                key in expected
                or key[:2] not in signatures
                or row["metric"] not in METRICS
            ):
                raise IntegrityError("Metric inventory/key mismatch")
            signature = signatures[key[:2]]
            if row["population"] not in (
                ("training",) if key[1] == "final" else ("training", "validation")
            ):
                raise IntegrityError("Invalid metric population")
            m = signature[row["population"] + "_metrics"]
            if (
                row["value"] != _cell(m.get(row["metric"]) if m else None)
                or int(row["positive_n"])
                != signature["counts"][row["population"] + "_positive_n"]
                or int(row["negative_n"]) != signature["counts"]["background_n"]
                or row["status"] != (m["status"] if m else signature["status"])
            ):
                raise IntegrityError("Metric/scientific result mismatch")
            expected.add(key)
    size = sum(len(METRICS) * (1 if f == "final" else 2) for _, f in signatures)
    if len(expected) != size:
        raise IntegrityError("Missing metric rows")


@schema_checked
def load_research(research_dir, *, immutable=False):
    """Verify a separate five-fold report bundle, never a production scorer."""
    from contextlib import nullcontext

    from .library import HEADERS, encoded_cluster

    path = Path(research_dir).absolute()
    resources = Path(__file__).resolve().parent / "resources"
    immutable = immutable or resources in path.parents
    with nullcontext() if immutable else transaction_guard(path.parent.parent):
        manifest = verify_bundle(path, RESEARCH)
        if identity(manifest["identity_inputs"]) != manifest["id"]:
            raise IntegrityError("Research identity mismatch")
        required = set(HEADERS) - {"dimension_final.tsv.gz"}
        for cluster in manifest["identity_inputs"]["cluster_ids"]:
            prefix = "clusters/" + encoded_cluster(cluster) + "/"
            required.add(prefix + "selection_trace.json.gz")
            with gzip.open(path / prefix / "selection_trace.json.gz", "rt") as handle:
                traces = json.load(handle)
            if len(traces) != 5:
                raise IntegrityError("Research requires five traces")
            for f in range(5):
                name = prefix + f"fold_{f}.json"
                required.add(name)
                signature = read_json(path / name)
                validate_signature(
                    signature,
                    library_id=manifest["id"],
                    cluster_id=cluster,
                    fit_id=f"fold_{f}",
                )
                if identity(traces[f]) != signature["selection_trace_identity"]:
                    raise IntegrityError("Research trace mismatch")
        if set(manifest["files"]) != required:
            raise IntegrityError("Research inventory mismatch")
        if (
            manifest["source_production"]["id"] == ""
            or len(manifest["source_production"]["manifest_sha256"]) != 64
        ):
            raise IntegrityError("Missing research production provenance")
        _validate_research_tables(path, manifest)
        return freeze(manifest)


def _validate_research_tables(path, manifest):
    import itertools

    from .library import HEADERS, encoded_cluster

    signatures = {}
    fold_ids = dict(manifest["identity_inputs"]["fold_fits"])
    counts = {
        "targets": len(manifest["identity_inputs"]["cluster_ids"]),
        "valid_discovery_fits": 0,
        "unsupported_fits": 0,
        "dimension_fold_rows": 0,
        "oof_rows": 0,
    }
    expected = [
        (c, str(f))
        for c in manifest["identity_inputs"]["cluster_ids"]
        for f in range(5)
    ]
    seen = []
    with (
        gzip.open(path / "dimension_folds.tsv.gz", "rt", newline="") as dims,
        gzip.open(
            path / "out_of_fold_positive_scores.tsv.gz", "rt", newline=""
        ) as oofs,
    ):
        dimensions = csv.DictReader(dims, delimiter="\t")
        positives = csv.DictReader(oofs, delimiter="\t")
        if (
            dimensions.fieldnames != HEADERS["dimension_folds.tsv.gz"]
            or positives.fieldnames != HEADERS["out_of_fold_positive_scores.tsv.gz"]
        ):
            raise IntegrityError("Research table header mismatch")
        groups = iter(
            itertools.groupby(positives, key=lambda r: (r["cluster_id"], r["fold_id"]))
        )
        current = next(groups, None)
        for (cluster, fold), group in itertools.groupby(
            dimensions, key=lambda r: (r["cluster_id"], r["fold_id"])
        ):
            seen.append((cluster, fold))
            directory = path / "clusters" / encoded_cluster(cluster)
            signature = read_json(directory / f"fold_{fold}.json")
            with gzip.open(directory / "selection_trace.json.gz", "rt") as handle:
                trace = json.load(handle)[int(fold)]
            result = {
                "signature": signature,
                "profiles": [_parse_profile(r, False) for r in group],
                "trace": trace,
                "oof": [],
            }
            if current is not None and current[0] == (cluster, fold):
                digest = sha256(directory / f"fold_{fold}.json")
                for row in current[1]:
                    if row["signature_digest"] != digest or row["accepted"] not in {
                        "",
                        "true",
                        "false",
                    }:
                        raise IntegrityError("Invalid research OOF binding")
                    result["oof"].append(
                        {
                            "window_id": row["window_id"],
                            "status": row["status"],
                            "log_score": float(row["log_score"])
                            if row["log_score"]
                            else None,
                            "score": float(row["score"]) if row["score"] else None,
                            "accepted": row["accepted"] == "true"
                            if row["accepted"]
                            else None,
                        }
                    )
                current = next(groups, None)
            ids = [r["window_id"] for r in result["oof"]]
            inputs = signature["input_identity"]
            if (
                signature["fit_identity"] != fold_ids[cluster][int(fold)]
                or identity(inputs) != signature["fit_identity"]
                or signature["scientific_payload_id"] != payload_identity(result)
                or identity(ids) != inputs["V_ids_sha256"]
                or len(ids) != signature["counts"]["validation_positive_n"]
            ):
                raise IntegrityError("Research scientific identity mismatch")
            validate_result(
                result,
                cluster=cluster,
                fit=f"fold_{fold}",
                counts=signature["counts"],
                sigma_floor=inputs["sigma_floor"],
                validation_ids=ids,
            )
            signatures[(cluster, f"fold_{fold}")] = signature
            counts["dimension_fold_rows"] += len(result["profiles"])
            counts["oof_rows"] += len(ids)
            counts[
                "valid_discovery_fits"
                if signature["status"] == "ok"
                else "unsupported_fits"
            ] += 1
        if current is not None or seen != expected or counts != manifest["counts"]:
            raise IntegrityError("Research population/count inventory mismatch")
    _verify_metrics(path / "cluster_metrics.tsv", signatures)
