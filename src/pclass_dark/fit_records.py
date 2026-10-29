"""Stable scientific fit records shared by production, migration and research."""

from __future__ import annotations

import copy

from .artifacts import IntegrityError, identity
from .library_runtime import validate_signature
from .signatures import _validate_profiles

WRAPPERS = {
    "schema",
    "library_id",
    "input_identity",
    "fit_identity",
    "scientific_payload_id",
    "evaluation_mode",
    "source_provenance",
}


def thaw(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw(v) for v in value]
    return value


def payload_identity(result):
    scientific = dict(result)
    scientific["signature"] = {
        k: v for k, v in result["signature"].items() if k not in WRAPPERS
    }
    return identity(scientific)


def validate_result(result, *, cluster, fit, counts, sigma_floor, validation_ids):
    """Validate complete scientific content without mutating source receipts."""
    try:
        signature = result["signature"]
        validate_signature(signature, cluster_id=cluster, fit_id=fit)
        if signature["counts"] != counts:
            raise IntegrityError("Fit population counts mismatch")
        _validate_profiles(
            result["profiles"],
            range(640),
            counts["training_positive_n"],
            counts["background_n"],
            counts["validation_positive_n"],
            sigma_floor,
        )
        if (
            identity(result["trace"]) != signature["selection_trace_identity"]
            or [r["window_id"] for r in result["oof"]] != validation_ids
            or result["trace"]["fit_id"] != fit
            or result["trace"]["selection"]["selected"]
            != signature["selected_dimension_ids"]
        ):
            raise IntegrityError("Fit trace/OOF binding mismatch")
        selection = result["trace"]["selection"]
        order = sorted(
            (
                p["dimension"]
                for p in result["profiles"]
                if p["training_auroc"] is not None
            ),
            key=lambda d: (-result["profiles"][d]["training_auroc_numerator"], d),
        )
        if result["trace"]["order"] != order:
            raise IntegrityError("Fit ranking/profile mismatch")
        ranks = {d: i + 1 for i, d in enumerate(order)}
        for record in signature["selected_dimensions"]:
            profile = result["profiles"][record["dimension"]]
            for parameter, statistic in (
                ("mu", "mu_C"),
                ("sigma", "sigma_C"),
                ("effective_sigma", "effective_sigma"),
                ("R", "R"),
                ("D", "D"),
                ("individual_rank", "rank"),
            ):
                if record[parameter] != profile[statistic]:
                    raise IntegrityError("Signature parameter/profile mismatch")
            for population in ("training", "validation"):
                auc = record["individual_" + population + "_auroc"]
                for key, suffix in (
                    ("value", ""),
                    ("numerator", "_numerator"),
                    ("denominator", "_denominator"),
                ):
                    if auc[key] != profile[population + "_auroc" + suffix]:
                        raise IntegrityError(
                            "Signature individual AUROC/profile mismatch"
                        )
        for d, row in enumerate(result["profiles"]):
            if row["rank"] != ranks.get(d):
                raise IntegrityError("Profile rank/order mismatch")
            if (
                row["status"] != signature["status"]
                or row["selected"] != (d in selection["selected"])
                or row["core"] != (d in selection["core"])
                or row["candidate"] != (d in selection["candidates"])
            ):
                raise IntegrityError("Fit profile flags/status mismatch")
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise IntegrityError("Malformed scientific fit result") from exc


def rebound_signature(result, library_id, fit_id, source=None):
    signature = copy.deepcopy(result["signature"])
    signature.update(
        schema="pclass-dark.signature.v2",
        library_id=library_id,
        fit_identity=fit_id,
        scientific_payload_id=payload_identity(result),
        evaluation_mode="all-C-in-sample"
        if signature["fit_id"] == "final"
        else "four-fold-training-held-out-positive",
    )
    if source:
        signature["source_provenance"] = source
    return signature
