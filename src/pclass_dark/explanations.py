"""Verified, portable cluster explanations; no fitting or query evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from .artifacts import (
    IntegrityError,
    read_json,
    safe_file,
    schema_checked,
    transaction_guard,
)
from .explanation_trace import selection_evidence
from .fit_records import thaw
from .library_runtime import freeze

SCHEMA = "pclass-dark.signature-explanation.v1"
ABC = {
    "A": {
        "label": "Acceptance",
        "direction": None,
        "formula": "log_S >= fitted log cutoff; No for predict-none",
    },
    "B": {
        "label": "Background score",
        "direction": "lower",
        "formula": "(1 + count(B_log_scores >= log_S)) / (n_B + 1)",
    },
    "C": {
        "label": "Cluster score",
        "direction": "higher",
        "formula": "count(C_log_scores <= log_S) / n_C",
    },
}


@dataclass(frozen=True)
class SignatureExplanation:
    """Detached immutable report. ``to_dict`` returns an independent JSON tree."""

    data: Mapping

    def __post_init__(self):
        detached = thaw(self.data)
        json.dumps(detached, allow_nan=False)
        object.__setattr__(self, "data", freeze(detached))

    def to_dict(self):
        return thaw(self.data)


def explain_signature(cluster_id, library_dir=None, *, detailed=False):
    """Explain one selected v2 production signature, without setup-data inputs.

    Raises ValueError for invalid arguments/unknown clusters, IntegrityError for
    malformed or incompatible artifacts, and OSError for inaccessible inputs.
    """
    if not isinstance(cluster_id, str) or not cluster_id:
        raise ValueError("cluster_id must be a nonempty string")
    if type(detailed) is not bool:
        raise ValueError("detailed must be a boolean")
    from .library import _library_immutable
    from .paths import default_library_input_dir

    root = Path(
        default_library_input_dir() if library_dir is None else library_dir
    ).absolute()
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
        try:
            return _assemble(root, cluster_id, detailed, immutable)
        except (StopIteration, EOFError) as exc:
            raise IntegrityError(
                "Incomplete production result or compressed trace"
            ) from exc


@schema_checked
def _assemble(root, cluster_id, detailed, immutable):
    from .library_reports import final_results
    from .relative_scores import load_selected_library

    selector = read_json(safe_file(root, "selection.json"))
    if selector["schema"] not in {
        "pclass-dark.library-selection.v1",
        "pclass-dark.library-selection.v2",
    }:
        raise IntegrityError("Unknown library selection schema")
    manifest = read_json(
        safe_file(safe_file(root, selector["library"]), "manifest.json")
    )
    if manifest["schema"] == "pclass-dark.library.v1":
        raise IntegrityError(
            "Legacy library: run pclass-dark build-library to migrate first"
        )
    if manifest["schema"] != "pclass-dark.library.v2":
        raise IntegrityError("Unsupported production schema")
    pair = load_selected_library(root, immutable=immutable, cluster_id=cluster_id)
    library = pair.library
    if cluster_id not in library.signatures:
        raise ValueError(f"Unknown cluster: {cluster_id}")
    result = (
        thaw(library.target_result)
        if library.target_result is not None
        else next(final_results(library.source_path, {cluster_id}))[1]
    )
    signature = result["signature"]
    selection = selection_evidence(result, detailed=detailed)
    profiles = result["profiles"]
    weights = {r["dimension"]: r["weight"] for r in signature["selected_dimensions"]}
    dimensions = [{**p, "weight": weights.get(p["dimension"])} for p in profiles]
    counts = signature["counts"]
    populations = {
        "C_windows": counts["training_positive_n"],
        "C_parents": counts.get("training_parent_n"),
        "B_windows": counts["background_n"],
        "B_parents": counts.get("background_parent_n"),
    }
    populations["unavailable_reasons"] = {
        k: "not_recorded" for k, value in populations.items() if value is None
    }
    distributions = (
        thaw(pair.relative.summarize(cluster_id))
        if pair.relative
        else {
            "status": "unavailable",
            "reason": "no_selected_companion",
            "relative_id": None,
            "quantile_method": "linear-in-log-space-v1",
            "C": None,
            "B": None,
        }
    )
    limitations = [
        {
            "code": "in_sample",
            "message": "C and B were used during discovery and threshold selection.",
        },
        {
            "code": "held_out_not_included",
            "message": "Optional research results are separate and not included.",
        },
        {
            "code": "not_probability",
            "message": (
                "Gaussian similarity and ABC evidence are not membership probabilities."
            ),
        },
        {
            "code": "not_mechanism",
            "message": (
                "Embedding dimension IDs do not identify biochemical mechanisms."
            ),
        },
    ]
    if signature["status"] == "ok":
        limitations.append(
            {
                "code": "greedy_selection",
                "message": (
                    "This greedy stopping point does not prove "
                    "global subset optimality."
                ),
            }
        )
    if distributions["status"] != "ok":
        limitations.append(
            {
                "code": "distributions_unavailable",
                "message": str(distributions["reason"]),
            }
        )
    inputs = signature["input_identity"]
    return SignatureExplanation(
        {
            "schema": SCHEMA,
            "source": {
                "library_id": library.library_id,
                "manifest_sha256": library.manifest_sha256,
                "cluster_id": cluster_id,
                "fit_identity": signature["fit_identity"],
                "scientific_payload_id": signature["scientific_payload_id"],
                "evaluation_mode": signature["evaluation_mode"],
                "representation_contract_id": signature["representation_contract_id"],
                "relative_id": distributions["relative_id"],
                "selection_schema": pair.selection_schema,
                "validation_scope": "cluster",
            },
            "status": signature["status"],
            "reason": signature["reason"],
            "detailed": detailed,
            "populations": populations,
            "representation": thaw(library.embedding_contract),
            "dimensions": {
                "top": [dimensions[d] for d in selection["order"][:3]],
                "selected": [dimensions[d] for d in selection["selected"]],
                "all": dimensions if detailed else None,
            },
            "selection": selection,
            "scoring": {
                "epsilon": inputs["epsilon"],
                "sigma_floor": inputs["sigma_floor"],
                "threshold": signature["threshold"],
                "gaussian_formula": "member-mean-gaussian-v1",
                "composite_formula": "equal-weight-arithmetic-mean-v1",
                "gaussian_definition": (
                    "exp(-0.5 * ((x_j - mu_C,j) / max(sigma_C,j, sigma_floor))**2)"
                ),
                "composite_definition": (
                    "sum(g_j(x) for selected dimensions) / number_selected"
                ),
            },
            "metrics": {
                "training": signature["training_metrics"],
                "training_unavailable_reason": signature["reason"],
                "held_out": {"status": "not_included", "reason": "production_only"},
            },
            "distributions": distributions,
            "abc_definitions": ABC,
            "limitations": limitations,
        }
    )


def render_signature_explanation(report, *, format="text"):
    """Render a detached report as text, Markdown or JSON; performs no IO."""
    if format not in ("text", "markdown", "json"):
        raise ValueError("format must be text, markdown or json")
    if not isinstance(report, SignatureExplanation):
        raise ValueError("Expected SignatureExplanation")
    data = report.to_dict()
    if format == "json":
        return json.dumps(data, indent=2, allow_nan=False) + "\n"
    from .explanation_render import render_narrative

    return render_narrative(data, markdown=format == "markdown")
