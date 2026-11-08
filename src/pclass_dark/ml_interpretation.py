"""Deterministic chosen-cluster explanations and human-readable prediction reports."""

from __future__ import annotations

import html
import logging
import math
import re

import numpy as np

from .artifacts import IntegrityError, file_record, read_json, safe_file, sha256
from .ml_data import FEATURES

LOG = logging.getLogger(__name__)
EPSILON = 1e-6
LABELS = dict(
    zip(
        FEATURES,
        (
            "Protein length",
            "Acidic residues / full sequence",
            "Basic residues / full sequence",
            "Exposed acidic residues / full sequence",
            "Buried acidic residues / full sequence",
            "Exposed basic residues / full sequence",
            "Buried basic residues / full sequence",
            "Predicted alpha-helix fraction",
            "Predicted beta-strand fraction",
            "Predicted coil/loop fraction",
            "Charge-balance proxy",
            "Charged residues / full sequence",
            "Histidine fraction",
            "Predicted exposed-residue fraction",
            "Acidic residues among predicted exposed residues",
            "Basic residues among predicted exposed residues",
            "Charged residues among predicted exposed residues",
            "Mean hydropathy",
        ),
        strict=True,
    )
)


def chosen_support(classifier, X, classes):
    """Return [N,18] own-class centered terms and [N] biases for local class indices.

    Unlike pairwise margins, each term is relative to its curve's mean over the
    model's training examples. Summed terms plus bias reconstruct the chosen logit.
    """
    classes = np.asarray(classes)
    if (
        classes.ndim != 1
        or len(classes) != len(X)
        or not len(classes)
        or classes.dtype.kind not in "iu"
        or classes.min() < 0
        or classes.max() >= len(classifier.bias)
    ):
        raise ValueError("Expected aligned valid local class indices")
    basis = classifier.preprocess.transform(X)
    coefficients = classifier.weight[classes]
    terms = np.zeros((len(X), len(FEATURES)), dtype=np.float64)
    for j in range(len(FEATURES)):
        begin, end = classifier.preprocess.offsets[j : j + 2]
        terms[:, j] = (basis[:, begin:end] * coefficients[:, begin:end]).sum(axis=1)
    return terms, classifier.bias[classes].astype(np.float64)


def resolve_descriptions(model):
    """Resolve optional live labels only after binding them to training provenance.

    No preparation/catalog is needed to score the model. Missing or mismatched
    metadata produces ID-only reports and a reason on the configured logger.
    """
    try:
        extraction = model.feature_contract.get("extraction")
        if extraction is None:
            raise IntegrityError("model has no saved extraction/preparation provenance")
        from .cluster_descriptions import load_catalog
        from .paths import default_setup_input_dir
        from .reference_data import resolve_profunc

        path = safe_file(
            default_setup_input_dir(),
            f"prepared/{extraction['preparation_id']}/manifest.json",
        )
        if sha256(path) != extraction["prepared_manifest_sha256"]:
            raise IntegrityError("preparation manifest differs from training")
        preparation = read_json(path)
        if preparation["id"] != extraction["preparation_id"]:
            raise IntegrityError("preparation identity mismatch")
        reference = preparation["reference"]
        members = {r["path"]: r for r in reference["source"]["identity"]["members"]}
        runtime = resolve_profunc()
        bound = (
            (runtime.clusters / "clusters_meta.tsv", "clusters/clusters_meta.tsv"),
            (runtime.dataset / "go_graph.pkl", "dataset/go_graph.pkl"),
        )

        def verify_labels():
            for actual, relative in bound:
                record = members[relative]
                if file_record(actual) != {
                    "sha256": record["sha256"],
                    "size_bytes": record["size"],
                }:
                    raise IntegrityError(
                        f"functional-label reference differs: {relative}"
                    )

        verify_labels()
        catalog = load_catalog(model.labels)
        verify_labels()
        if catalog["reference"] != reference["files"]:
            raise IntegrityError(
                "live cluster membership/dataset differs from training"
            )
        return {
            cluster: row["description"]
            for cluster, row in catalog["clusters"].items()
            if row["description"]
        }
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        LOG.warning("Functional annotations unavailable for this model: %s", exc)
        return {}


def _escape(value):
    value = html.escape(str(value), quote=False)
    return re.sub(r"([\\`*_{}\[\]|])", r"\\\1", value)


def _length(prediction):
    if prediction.sequence_length is not None:
        return f"{prediction.sequence_length:,} amino acids"
    value = prediction.features["ln_length"]
    if value is None:
        return "NA (imputed)"
    if 0 <= value <= math.log(100000):
        return f"about {round(math.exp(value)):,} amino acids (from rounded log length)"
    return f"ln(length) = {value:.3f}"


def _value(prediction, feature):
    value = prediction.features[feature]
    if feature == "ln_length":
        return _length(prediction)
    if value is None:
        return "NA (imputed)"
    if feature == "mean_kyte_doolittle_hydropathy":
        label = (
            "hydrophilic" if value < 0 else "hydrophobic" if value > 0 else "neutral"
        )
        return f"{value:.3f}; {label} on average"
    if feature == "net_charge_balance":
        return f"{value:+.3f}"
    return f"{100 * value:.1f}%"


def _list(items):
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def render_top_predictions(prediction, *, descriptions=None):
    """Render the shared ranked table, with descriptions capped at 24 characters."""
    descriptions = descriptions or {}
    lines = [
        "",
        "## Top 5 predictions",
        "",
        "| Cluster ID | Short description | Softmax score |",
        "| --- | --- | ---: |",
    ]
    for rank in prediction.ranked[:5]:
        description = " ".join(
            (descriptions.get(rank.cluster_id) or "Unavailable").split()
        )
        if len(description) > 24:
            description = description[:23] + "…"
        lines.append(
            f"| {_escape(rank.cluster_id)} | {_escape(description)} | "
            f"{rank.softmax_score:.3f} |"
        )
    return "\n".join(lines) + "\n"


def render_interpretation(prediction, *, descriptions=None):
    """Render the user's shortened Markdown template from real chosen-class terms.

    Pure formatting: does not discover references, load a model or call a language
    model. Optional descriptions are caller-supplied, already-bound annotation text.
    """
    p = prediction
    support = p.chosen_feature_contributions
    if (
        set(support) != set(FEATURES)
        or set(p.features) != set(FEATURES)
        or p.training_example_count < 1
        or not p.ranked
        or p.chosen_bias is None
        or p.chosen_logit is None
        or not all(math.isfinite(v) for v in support.values())
        or not math.isfinite(p.chosen_bias)
        or not math.isfinite(p.chosen_logit)
        or not np.isclose(
            sum(support.values()) + p.chosen_bias, p.chosen_logit, atol=1e-4, rtol=1e-4
        )
        or not np.isclose(p.chosen_logit, p.ranked[0].logit, atol=1e-4, rtol=1e-4)
    ):
        raise ValueError(
            "Interpretation requires a reconciled chosen-cluster explanation"
        )
    descriptions = descriptions or {}
    chosen = p.ranked[0]

    def label(rank):
        text = f"**{_escape(rank.cluster_id)}**"
        description = descriptions.get(rank.cluster_id)
        if description:
            text += f", annotated as **{_escape(description)}**"
        return text

    first = f"The model assigns this protein to {label(chosen)}."
    if p.sequence_length is not None:
        first = (
            f"The model assigns this **{p.sequence_length:,}-amino-acid protein** "
            f"to {label(chosen)}."
        )
    elif p.features["ln_length"] is not None:
        first = f"The model assigns this protein ({_length(p)}) to {label(chosen)}."
    lines = [
        f"# Prediction interpretation: {_escape(p.query_id)}",
        "",
        "## Predicted function",
        "",
        first,
        "",
        f"The chosen cluster has a **softmax score of {chosen.softmax_score:.3f}**.",
    ]
    if len(p.ranked) > 1:
        second = p.ranked[1]
        lines.append(
            f"The **second choice** is {label(second)}, "
            f"with score {second.softmax_score:.3f}."
        )
    lines += [
        "These are normalized model scores, not calibrated probabilities "
        "of biological correctness."
    ]
    if chosen.cluster_id not in descriptions:
        lines += [
            "A matching functional annotation is unavailable; the model's "
            "cluster ID is reported above."
        ]
    lines.append(render_top_predictions(p, descriptions=descriptions).rstrip())
    positive = sorted(
        (f for f in FEATURES if support[f] > EPSILON and p.features[f] is not None),
        key=lambda f: -support[f],
    )
    negative = sorted(
        (f for f in FEATURES if support[f] < -EPSILON and p.features[f] is not None),
        key=lambda f: support[f],
    )

    def phrase(feature):
        return f"**{LABELS[feature].lower()} ({_value(p, feature)})**"

    lines += ["", "## Plain-language explanation", ""]
    if positive:
        lines += [
            "The strongest observed feature support comes from "
            + _list([phrase(f) for f in positive[:3]])
            + ".",
            "The model learned that these observed values are more supportive "
            "of the chosen cluster than their average contributions across the"
            " training proteins.",
        ]
        if len(positive) > 3:
            lines += [
                "",
                "Additional support comes from "
                + _list([phrase(f) for f in positive[3:6]])
                + ".",
            ]
    else:
        lines += [
            "None of the observed feature values provides positive support "
            "above its average training contribution. The chosen score also "
            "depends on the learned baseline and any imputed descriptors."
        ]
    if negative:
        lines += [
            "",
            (
                "The observed value of "
                if len(negative[:3]) == 1
                else "The observed values of "
            )
            + _list([phrase(f) for f in negative[:3]])
            + (" reduces" if len(negative[:3]) == 1 else " reduce")
            + " the chosen cluster's score relative to average training contributions.",
        ]
        if (
            positive
            and max(abs(support[f]) for f in negative) <= 0.1 * support[positive[0]]
        ):
            lines += [
                "These contrary effects are small compared with the strongest "
                "supporting feature."
            ]
    else:
        lines += [
            "",
            "No observed descriptor reduces the chosen score below its average"
            " training contribution.",
        ]
    lines += [
        "",
        "This explains the model's learned profile; it is not an "
        "experimental demonstration of activity.",
        "",
        "## Support from each feature",
        "",
        "Each value is the feature's support **for the chosen cluster itself**, "
        "relative to that curve's average over all "
        f"**{p.training_example_count:,} training sequences**. "
        "The reference does not depend on the second choice.",
        "",
        "- **Positive:** more supportive than the average training contribution.",
        "- **Negative:** less supportive than the same reference.",
        "- **Near zero:** little difference from that average.",
        "",
        "Support is in model-score (logit) units, not percentage "
        "confidence or independent biological importance. The signs "
        "describe these observed values; they do not imply a universally "
        "increasing or decreasing feature effect.",
        "",
        f"| Feature | Observed value | Support for {_escape(chosen.cluster_id)} |",
        "| --- | --- | ---: |",
    ]
    for feature in sorted(FEATURES, key=lambda f: -support[f]):
        lines.append(
            f"| {LABELS[feature]} | {_value(p, feature)} | {support[feature]:+.3f} |"
        )
    lines += [
        "",
        "Protein length uses the model's natural-log input and is "
        "displayed in amino acids when available. Charge balance is a "
        "residue-composition proxy, not charge at a particular pH. "
        "Exposure and secondary structure are predicted descriptors. "
        "Related charge/exposure features share information; these "
        "contributions are not independent biochemical mechanisms.",
    ]
    if p.imputed_features or p.outside_support_features:
        lines += ["", "## Data-quality notes", ""]
        if p.imputed_features:
            lines += [
                "Missing descriptors were replaced with saved training medians: "
                + _list([LABELS[f] for f in p.imputed_features])
                + ". Their support values describe imputed inputs, not observations."
            ]
        if p.outside_support_features:
            lines += [
                "Observed values outside the saved training ranges: "
                + _list([LABELS[f] for f in p.outside_support_features])
                + ". These require caution when interpreting the prediction."
            ]
    return "\n".join(lines) + "\n"
