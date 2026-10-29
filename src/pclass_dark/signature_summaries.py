"""Checksum-bound production summaries without profiles, traces or score arrays."""

from collections.abc import Mapping
from contextlib import ExitStack, nullcontext
from pathlib import Path

from .artifacts import (
    IntegrityError,
    identity,
    read_json,
    safe_file,
    schema_checked,
    sha256,
    transaction_guard,
    verify_files,
)
from .cluster_descriptions import description_for, load_catalog
from .fit_records import thaw
from .library_runtime import representation_contract, validate_signature

METRIC_KEYS = ("auroc", "recall", "specificity", "precision")
UNDEFINED = "undefined_denominator_or_population"
SCHEMA = "pclass-dark.signature-summary-list.v1"


def require(condition, message):
    if not condition:
        raise IntegrityError("Signature summary: " + message)


def integer(value):
    return type(value) is int and value >= 0


def digest(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def fraction(value):
    return type(value) in (int, float) and 0 <= value <= 1


@schema_checked
def validate_signature_summary(signature, *, manifest, cluster_id):
    """Validate and return one compact row, without reconstructing fitting.

    Raises IntegrityError for malformed fields, invalid bindings/counts or
    inconsistent saved metrics. The manifest must already be checksum verified.
    """
    fit_id = dict(manifest["identity_inputs"]["final_fits"])[cluster_id]
    return _summary_row(
        signature, manifest=manifest, cluster_id=cluster_id, fit_id=fit_id
    )


@schema_checked
def _summary_row(signature, *, manifest, cluster_id, fit_id):
    validate_signature(
        signature, library_id=manifest["id"], cluster_id=cluster_id, fit_id="final"
    )
    inputs = thaw(signature["input_identity"])
    require(
        signature["schema"] == "pclass-dark.signature.v2"
        and signature["evaluation_mode"] == "all-C-in-sample"
        and signature["fit_identity"] == fit_id
        and identity(inputs) == signature["fit_identity"]
        and inputs["cluster_id"] == cluster_id
        and signature["representation_contract_id"]
        == inputs["representation_contract_id"]
        == manifest["identity_inputs"]["representation_contract_id"],
        "signature identity/representation mismatch",
    )
    counts = signature["counts"]
    c, b = counts["training_positive_n"], counts["background_n"]
    require(integer(c) and integer(b), "invalid window counts")
    require(
        type(inputs["C_n"]) is int
        and inputs["C_n"] == c
        and type(inputs["B_n"]) is int
        and inputs["B_n"] == b,
        "fit population counts mismatch",
    )
    status, reason = signature["status"], signature["reason"]
    require(
        reason is None or isinstance(reason, str) and bool(reason), "invalid reason"
    )
    row = {
        "cluster_id": cluster_id,
        "status": status,
        "reason": reason,
        "C_windows": c,
        "B_windows": b,
        "dimensions": len(signature["selected_dimensions"]),
        **{k: None for k in METRIC_KEYS},
        "unavailable_reasons": {},
    }
    metrics = signature["training_metrics"]
    if status != "ok":
        require(
            metrics is None
            and signature["threshold"] is None
            and not signature["selected_dimensions"]
            and isinstance(signature["selected_dimension_ids"], (list, tuple))
            and not signature["selected_dimension_ids"],
            "unsupported signature contains scoring results",
        )
        require(
            b == 0
            if status == "empty_background"
            else b > 0 and c < 2
            if status == "insufficient_training"
            else b > 0 and c >= 2,
            "unsupported status/population mismatch",
        )
        row["unavailable_reasons"] = {k: reason or status for k in METRIC_KEYS}
        return row
    require(
        c >= 2 and b > 0 and reason is None,
        "supported signature population/reason mismatch",
    )
    require(
        isinstance(metrics, Mapping) and metrics["status"] == "ok",
        "missing training metrics",
    )
    require(
        integer(metrics["positive_n"])
        and metrics["positive_n"] == c
        and integer(metrics["negative_n"])
        and metrics["negative_n"] == b,
        "metric population mismatch",
    )
    require(
        all(integer(metrics[k]) for k in ("tp", "fp", "tn", "fn")),
        "invalid confusion counts",
    )
    tp, fp, tn, fn = (metrics[k] for k in ("tp", "fp", "tn", "fn"))
    require(tp + fn == c and tn + fp == b, "confusion population mismatch")
    n, d = metrics["auroc_numerator"], metrics["auroc_denominator"]
    require(
        integer(n) and integer(d) and d == 2 * c * b and n <= d, "invalid AUROC counts"
    )
    expected = {
        "auroc": n / d,
        "recall": tp / c,
        "specificity": tn / b,
        "precision": tp / (tp + fp) if tp + fp else None,
    }
    reasons = metrics["unavailable_reasons"]
    require(isinstance(reasons, Mapping), "invalid unavailable reasons")
    for key, value in expected.items():
        actual = metrics[key]
        if value is None:
            require(
                actual is None and reasons.get(key) == UNDEFINED,
                "undefined metric/reason mismatch",
            )
            row["unavailable_reasons"][key] = reasons[key]
        else:
            require(
                fraction(actual) and actual == value and key not in reasons,
                "inconsistent " + key,
            )
        row[key] = actual
    if signature["threshold"]["kind"] == "above_maximum":
        require(
            tp == fp == 0 and fn == c and tn == b, "predict-none confusion mismatch"
        )
    return row


def list_signature_summaries(library_dir=None):
    """Load a compact, verified production-only summary snapshot.

    Raises IntegrityError for invalid artifacts, OSError for inaccessible inputs.
    Relative/research files are outside this operation's validation scope.
    """
    from .explanations import SignatureSummaryListing
    from .library import _library_immutable
    from .paths import default_library_input_dir

    try:
        root = Path(
            default_library_input_dir() if library_dir is None else library_dir
        ).absolute()
    except (TypeError, ValueError) as exc:
        raise ValueError("library_dir must be a filesystem path") from exc
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise IntegrityError("Symlink library roots are unsupported")
    if not root.is_dir():
        raise FileNotFoundError(f"Library root not found: {root}")
    if not safe_file(root, "selection.json").is_file():
        raise FileNotFoundError(
            f"Library selection not found: {root / 'selection.json'}"
        )
    immutable = _library_immutable(root, False)
    with nullcontext() if immutable else transaction_guard(root):
        return SignatureSummaryListing(_load_rows(root, immutable))


@schema_checked
def _load_rows(root, immutable):
    from .library import encoded_cluster

    selected = read_json(safe_file(root, "selection.json"))
    require(
        selected["schema"]
        in {
            "pclass-dark.library-selection.v1",
            "pclass-dark.library-selection.v2",
        },
        "unknown selection schema",
    )
    path = safe_file(root, selected["library"])
    with ExitStack() as stack:
        if not immutable and path.parent != root:
            stack.enter_context(transaction_guard(path.parent))
        manifest_file = safe_file(path, "manifest.json")
        checksum = sha256(manifest_file)
        require(
            checksum == selected["manifest_sha256"], "selected manifest digest mismatch"
        )
        manifest = read_json(manifest_file)
        if manifest["schema"] == "pclass-dark.library.v1":
            raise IntegrityError(
                "Legacy library: run pclass-dark build-library to migrate first"
            )
        require(
            manifest["schema"] == "pclass-dark.library.v2"
            and manifest["status"] == "complete"
            and manifest["evaluation_mode"] == "all-C-in-sample",
            "incomplete or unknown production schema",
        )
        inputs = manifest["identity_inputs"]
        require(
            digest(manifest["id"])
            and manifest["id"] == selected["library_id"] == identity(inputs),
            "production identity mismatch",
        )
        contract = representation_contract(manifest["representation_contract"])
        require(
            identity(contract) == inputs["representation_contract_id"],
            "representation mismatch",
        )
        targets = inputs["cluster_ids"]
        require(
            isinstance(targets, list)
            and all(isinstance(c, str) and c for c in targets),
            "invalid cluster inventory",
        )
        require(targets == sorted(set(targets)), "duplicate or unsorted clusters")
        fit_rows = inputs["final_fits"]
        require(
            isinstance(fit_rows, list)
            and all(
                isinstance(r, list)
                and len(r) == 2
                and isinstance(r[0], str)
                and digest(r[1])
                for r in fit_rows
            ),
            "malformed final-fit inventory",
        )
        fits = dict(fit_rows)
        require(
            len(fits) == len(fit_rows) and set(fits) == set(targets),
            "final-fit inventory mismatch",
        )
        counts = manifest["counts"]
        require(
            all(
                integer(counts[k])
                for k in (
                    "targets",
                    "available_final",
                    "unsupported_final",
                    "dimension_final_rows",
                )
            )
            and counts["targets"] == len(targets)
            and counts["dimension_final_rows"] == 640 * len(targets)
            and counts["available_final"] + counts["unsupported_final"] == len(targets),
            "manifest summary counts mismatch",
        )
        records = manifest["files"]
        require(isinstance(records, dict), "invalid file record mapping")
        catalog = load_catalog()
        rows = []
        for cluster in targets:
            name = "clusters/" + encoded_cluster(cluster) + "/final.json"
            record = records[name]
            require(
                isinstance(record, dict)
                and digest(record["sha256"])
                and integer(record["size_bytes"]),
                "invalid signature file record",
            )
            verify_files(path, {name: record})
            signature = read_json(safe_file(path, name))
            row = _summary_row(
                signature, manifest=manifest, cluster_id=cluster, fit_id=fits[cluster]
            )
            description, unavailable = description_for(signature, catalog)
            rows.append(
                {
                    "cluster_id": row.pop("cluster_id"),
                    "description": description,
                    "description_unavailable_reason": unavailable,
                    **row,
                }
            )
        available = sum(r["status"] == "ok" for r in rows)
        require(
            available == counts["available_final"], "available signature count mismatch"
        )
        return {
            "schema": SCHEMA,
            "source": {
                "library_id": manifest["id"],
                "manifest_sha256": checksum,
                "selection_schema": selected["schema"],
                "evaluation_mode": "all-C-in-sample",
                "validation_scope": "signature_summaries",
                "description_catalog_id": catalog["id"]
                if catalog is not None
                else None,
            },
            "counts": {
                "clusters": len(rows),
                "available": available,
                "unsupported": len(rows) - available,
            },
            "rows": rows,
            "limitations": [
                {
                    "code": "in_sample",
                    "message": (
                        "Reference populations were used during discovery "
                        "and threshold selection."
                    ),
                },
                {
                    "code": "reference_precision",
                    "message": (
                        "Precision depends on the sampled reference population balance."
                    ),
                },
                {
                    "code": "display_rounding",
                    "message": (
                        "Text/Markdown use three decimals; JSON retains full precision."
                    ),
                },
            ],
        }
