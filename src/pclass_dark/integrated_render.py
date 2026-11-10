"""JSON, Markdown and terminal renderers for one integrated result stream."""

from __future__ import annotations

import json
from types import SimpleNamespace

from .fit_records import thaw
from .ml_interpretation import LABELS, _escape, _value


def render_query(query, *, coverage=None, markdown_stream=None):
    """Return detailed Markdown and compact terminal text from the same record."""
    d = query.to_dict() if hasattr(query, "to_dict") else query
    a = d["additive"]
    c = d["classification"]
    matches = d["pclass_dark"]["matches"]
    selected = c["selected_cluster_id"]
    reason = a["reason"]
    e = next((e for e in a["explanations"] if e["cluster_id"] == selected), None)
    if not selected:
        opening = "PClass-Dark finds no accepted cluster for this query."
        if a["global_top5"]:
            opening += (
                " The additive predictions below are supplemental suggestions only."
            )
    elif a["status"] != "available":
        opening = (
            f"PClass-Dark accepts {_escape(selected)} as its leading "
            f"cluster. Additive reranking is unavailable "
            f"({_escape(reason)}); the original ordering is retained."
        )
    else:
        desc = next(
            (
                r.get("description")
                for r in a["accepted_reranking"]
                if r["cluster_id"] == selected
            ),
            None,
        )
        label = _escape(selected) + (f" ({_escape(desc)})" if desc else "")
        if c["selection_rule"] == "pclass_original":
            opening = (
                f"PClass-Dark accepts {len(matches)} clusters. Its "
                f"leading cluster is {label}; none of the accepted "
                f"clusters has an additive score, so the original "
                f"ordering is retained."
            )
        else:
            opening = (
                f"PClass-Dark accepts {len(matches)} clusters. Additive "
                f"reranking places **{label}** first (original rank "
                f"{c['original_rank']}). "
            )
            opening += (
                "It is also in the classifier’s global top five."
                if c["in_additive_top5"]
                else "It is outside the classifier’s global top five."
            )
        if e:
            features = e["positive_observed_features"][:3]
            view = SimpleNamespace(features=a["features"], sequence_length=d["length"])
            if features:
                opening += (
                    " Supporting observed features are "
                    + ", ".join(
                        (
                            f"{LABELS[f]} ({_value(view, f)}, "
                            f"{e['feature_contributions'][f]:+.3f})"
                        )
                        for f in features
                    )
                    + ", relative to their average training contributions."
                )
            else:
                opening += (
                    " No observed feature has positive support above its "
                    "average training contribution."
                )
    head = f"# Classification: {_escape(d['query_id'])}\n\n{opening}\n"
    accepted = [
        (
            "| Cluster | Original rank | New rank | B ↓ | C ↑ | "
            "Softmax | Top-five agreement |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    byid = {m["cluster_id"]: m for m in matches}
    topids = set(a.get("top5_accepted_cluster_ids", []))
    for r in a["accepted_reranking"]:
        m = byid[r["cluster_id"]]
        score = f"{r['softmax_score']:.3f}" if r["score_available"] else "Unavailable"
        accepted.append(
            f"| {_escape(r['cluster_id'])} | {r['original_rank']} | "
            f"{r['position']} | {m['background_tail_fraction']:.6g} "
            f"| {m['cluster_percentile_le']:.6g} | {score} | "
            f"{'Yes' if r['cluster_id'] in topids else 'No'} |"
        )
    top = [
        "| Cluster ID | Short description | Softmax score | Accepted |",
        "| --- | --- | ---: | --- |",
    ]
    for r in a["global_top5"]:
        desc = " ".join((r.get("description") or "Unavailable").split())
        if len(desc) > 24:
            desc = desc[:23] + "…"
        top.append(
            f"| {_escape(r['cluster_id'])} | {_escape(desc)} | "
            f"{r['softmax_score']:.3f} | "
            f"{'Yes' if r['accepted_by_pclass_dark'] else 'No'} |"
        )
    compact = (
        head
        + "\n"
        + "\n".join(accepted)
        + "\n\n## Top 5 additive predictions\n\n"
        + "\n".join(top)
        + "\n"
    )

    class MarkdownOutput:
        def append(self, text):
            markdown_stream.write(text)

        def extend(self, texts):
            for text in texts:
                self.append(text)

    detailed = [] if markdown_stream is None else MarkdownOutput()
    detailed.extend([head, "\n## Detailed PClass-Dark analysis\n"])
    if coverage:
        detailed.append(
            f"\nEvaluated {coverage['clusters_evaluated']}/"
            f"{coverage['clusters_total']} "
            f"clusters; {len(coverage['unavailable'])} unavailable.\n"
        )
    for m in matches:
        detailed.append(
            f"\n### {_escape(m['cluster_id'])}\n\nOriginal rank "
            f"{m['original_rank']}; accepted: Yes; "
            f"B={m['background_tail_fraction']:.6g}; "
            f"C={m['cluster_percentile_le']:.6g}; C_n={m['C_n']}; "
            f"B_n={m['B_n']}.\n"
        )
        detailed.append(
            "\nRepresentative window: "
            + _escape(json.dumps(m["representative"], ensure_ascii=False))
            + ".\n"
        )
        detailed.append("\nSupporting windows:\n\n")
        for w in m["supporting_windows"]:
            detailed.append("- " + _escape(json.dumps(w, ensure_ascii=False)) + "\n")
    if not matches:
        detailed.append("\nNo accepted matches.\n")
    detailed.extend(
        [
            "\n## Additive reranking analysis\n\n",
            "\n".join(accepted),
            "\n\n### Global top five\n\n",
            "\n".join(top),
            "\n",
        ]
    )
    if a["model_id"]:
        detailed.append(
            f"\nClassifier configuration: **{_escape(a['configuration'])}**; "
            f"model: {_escape(a['model_id'])}.\n"
        )
    for explanation in a["explanations"]:
        detailed.append(
            f"\n### Feature support for {_escape(explanation['cluster_id'])}\n\n"
        )
        view = SimpleNamespace(features=a["features"], sequence_length=d["length"])
        for sign, key in [
            ("Supporting", "positive_observed_features"),
            ("Opposing", "negative_observed_features"),
        ]:
            fs = explanation[key][:3]
            detailed.append(
                sign
                + " observed evidence: "
                + (
                    ", ".join(
                        (
                            f"{LABELS[f]} ({_value(view, f)}, "
                            f"{explanation['feature_contributions'][f]:+.3f})"
                        )
                        for f in fs
                    )
                    if fs
                    else "none"
                )
                + ".\n\n"
            )
        detailed.append(
            "| Feature | Observed value | Support |\n| --- | --- | ---: |\n"
        )
        for f, v in sorted(
            explanation["feature_contributions"].items(), key=lambda item: -item[1]
        ):
            detailed.append(f"| {LABELS[f]} | {_value(view, f)} | {v:+.3f} |\n")
        detailed.append(
            f"\nReference: each curve’s mean over "
            f"{explanation['training_example_count']:,} training "
            f"sequences. Contributions plus bias reconstruct the "
            f"class logit.\n"
        )
    if a["imputed_features"]:
        detailed.append(
            "\nImputed features: " + ", ".join(a["imputed_features"]) + ".\n"
        )
    if a["outside_support_features"]:
        detailed.append(
            "\nOutside training support: "
            + ", ".join(a["outside_support_features"])
            + ".\n"
        )
    detailed.append(
        "\nPClass-Dark evidence concerns accepted windows; "
        "additive evidence describes the full sequence. Softmax "
        "scores are uncalibrated and are not renormalized over "
        "accepted clusters. Feature supports are statistical, "
        "not causal mechanisms. The methods are not independent "
        "experimental confirmations.\n"
    )
    detailed.append(
        "\nCharge balance is a composition proxy, not charge at a particular pH. "
        "Exposure and secondary structure are predicted. Related descriptors "
        "share information, so their contributions are not independent mechanisms.\n"
    )
    return ("".join(detailed) if markdown_stream is None else ""), compact


def write_integrated_reports(
    session, json_stream, markdown_stream, *, terminal_stream=None
):
    """Consume one session once; stream comprehensive JSON and matching prose."""
    root = dict(
        schema="pclass-dark.integrated-infer-report.v1",
        source=dict(
            pclass_dark=thaw(session.source),
            additive=dict(
                model_id=session.model.manifest["model_id"] if session.model else None,
                configuration=session.model.manifest["contract"]["configuration"]
                if session.model
                else None,
                integrated_library_id=session.envelope["id"]
                if session.envelope
                else None,
                reference_id=session.envelope["identity_inputs"]["reference_id"]
                if session.envelope
                else None,
                recipe_id=session.envelope["identity_inputs"]["recipe_id"]
                if session.envelope
                else None,
            ),
        ),
        execution=dict(
            pclass_dark=thaw(session.execution),
            additive=dict(mode=session.additive_mode, score_device="cpu"),
        ),
        coverage=dict(
            pclass_dark=thaw(session.coverage),
            additive=dict(
                label_count=len(session.model.labels) if session.model else 0
            ),
        ),
    )

    # The evidence database is static during rendering. Count UTF-8 output bytes
    # without repeatedly stat-ing it or materializing a full query JSON/string.
    from . import inference as infer

    base_usage = session._usage()
    written = sum(
        stream.tell()
        for stream in (json_stream, markdown_stream)
        + ((terminal_stream,) if terminal_stream else ())
    )

    class QuotaStream:
        def __init__(self, stream):
            self.stream = stream

        def write(self, text):
            nonlocal written
            size = len(text.encode("utf-8"))
            if base_usage + written + size > infer.MAX_TEMP:
                raise infer.InferenceResourceError("2 GiB report quota exceeded")
            self.stream.write(text)
            written += size

    js, md = QuotaStream(json_stream), QuotaStream(markdown_stream)
    terminal = QuotaStream(terminal_stream) if terminal_stream else None
    encoder = json.JSONEncoder(allow_nan=False, ensure_ascii=True)

    def encode(v):
        return json.dumps(v, allow_nan=False, ensure_ascii=True)

    js.write(encode(root)[:-1] + ',"queries":[')
    counts = dict(
        thaw(session.counts),
        additive_available_queries=0,
        reranked_queries=0,
        selected_top5_agreement_queries=0,
    )
    first = True
    for result in session:
        d = result.to_dict()
        if not first:
            js.write(",")
            md.write("\n---\n\n")
        first = False
        for chunk in encoder.iterencode(d):
            js.write(chunk)
        _, text = render_query(d, coverage=thaw(session.coverage), markdown_stream=md)
        if terminal:
            terminal.write(text + "\n")
        counts["additive_available_queries"] += d["additive"]["status"] == "available"
        counts["reranked_queries"] += (
            d["classification"]["selection_rule"] == "additive_reranked"
        )
        counts["selected_top5_agreement_queries"] += d["classification"][
            "in_additive_top5"
        ]
        session._quota(
            sum(
                s.tell()
                for s in [json_stream, markdown_stream]
                + ([terminal_stream] if terminal_stream else [])
            )
        )
    js.write('],"counts":' + encode(counts) + "}\n")
