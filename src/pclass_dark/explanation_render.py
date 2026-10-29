"""Pure deterministic narrative rendering for verified signature reports."""


def number(value):
    """Six significant digits, preserving small nonzero values."""
    return "unavailable" if value is None else format(value, ".6g")


def percent(value):
    return "unavailable" if value is None else number(100 * value) + "%"


def render_narrative(data, *, markdown=False):
    """Render evidence as paragraphs/tables without accessing its source files."""
    lines = []

    def paragraph(text):
        lines.extend([text, ""])

    def title(text):
        paragraph(("## " if markdown else "") + text)

    def table(headers, rows):
        # Plain text uses tabs: no ANSI, terminal sizing, or value truncation.
        if markdown:

            def escaped(value):
                return str(value).replace("|", "\\|").replace("\n", " ")

            lines.append("| " + " | ".join(map(escaped, headers)) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            lines.extend("| " + " | ".join(map(escaped, row)) + " |" for row in rows)
        else:
            lines.append("\t".join(map(str, headers)))
            lines.extend("\t".join(map(str, row)) for row in rows)
        lines.append("")

    source, pop = data["source"], data["populations"]
    selection, dimensions = data["selection"], data["dimensions"]
    ok = data["status"] == "ok"
    title("Signature explanation: " + source["cluster_id"])
    if ok:
        paragraph(
            f"This cluster's signature uses {len(selection['selected'])} of the "
            "640 PLM embedding dimensions to recognize a pattern "
            "among reference windows. "
            f"Evaluation mode: {source['evaluation_mode']}."
        )
    else:
        paragraph(
            f"This cluster has no usable signature: {data['reason']}. "
            "The available population and diagnostic evidence follows."
        )
    paragraph(f"Library: {source['library_id']}. Fit: {source['fit_identity']}.")

    title("Member and background windows")
    paragraph(
        "A protein chain of at most 1022 amino acids contributes one complete-sequence "
        "window. A longer chain contributes its first 1022 residues (N-<accession>) "
        "and last 1022 residues (C-<accession>). "
        "These windows overlap at chain lengths "
        "1023–2043, meet without overlap at 2044, and omit the central region above "
        "2044. This is terminal sampling, not exhaustive sliding-window coverage."
    )
    paragraph(
        "Windows, rather than whole chains, are the observations in C (members) "
        "and B (background). Both terminal windows inherit the parent chain's "
        "retained cluster memberships. Exact full-sequence deduplication precedes "
        "windowing; terminal windows are not independently deduplicated by sequence."
    )
    table(
        ["Population", "Windows", "Distinct parent chains"],
        [
            [
                "C — members",
                pop["C_windows"],
                pop["C_parents"] if pop["C_parents"] is not None else "unavailable",
            ],
            [
                "B — background",
                pop["B_windows"],
                pop["B_parents"] if pop["B_parents"] is not None else "unavailable",
            ],
        ],
    )
    paragraph(
        "B is the fixed out-of-cluster background, selected with up to 200 windows "
        "per source cluster, excluding target parents and deduplicating window IDs "
        "across contributing clusters. This cap limits large clusters' influence; "
        "it does not give every cluster the same contribution."
    )
    contract = data["representation"]
    paragraph(
        f"Recorded model: {contract['model']}, layer {contract['layer']}, width "
        f"{contract['width']}. Residue-only mean pooling uses positions where the "
        "attention mask is true AND the special-token mask is false, excluding "
        "padding and special tokens. Build-library consumes these fixed vectors "
        "and does not pool tokens again."
    )

    title("Evidence in individual dimensions")
    paragraph(
        "For each dimension, member and background means and sample standard "
        "deviations describe the reference populations. Relative conservation "
        "R = sigma_C / (sigma_B + epsilon): lower R means tighter member variation "
        "relative to background. Discrimination D = abs(mu_C - mu_B) / "
        "(hypot(sigma_C, sigma_B)/sqrt(2) + epsilon): higher D means greater "
        "standardized mean separation. R and D support interpretation; individual "
        "training AUROC determines the ranking."
    )
    paragraph(
        "Each dimension supplies a Gaussian similarity score for closeness to its "
        "member mean, with member variation setting tolerance. Individual AUROC "
        "measures how well this score ranks members above background. Numeric "
        "embedding dimensions do not by themselves identify biochemical mechanisms."
    )

    def profile_table(records, full=False):
        headers = ["Dimension (zero-based)", "Rank", "Member mean", "Member SD"]
        if full:
            headers += ["Background mean", "Background SD"]
        headers += [
            "Gaussian width",
            "R",
            "D",
            "Individual AUROC",
            "Protected",
            "Selected",
            "Weight",
        ]
        rows = []
        for p in records:
            row = [
                p["dimension"],
                number(p["rank"]),
                number(p["mu_C"]),
                number(p["sigma_C"]),
            ]
            if full:
                row += [number(p["mu_B"]), number(p["sigma_B"])]
            row += [
                number(p["effective_sigma"]),
                number(p["R"]),
                number(p["D"]),
                number(p["training_auroc"]),
                "Yes" if p["core"] else "No",
                "Yes" if p["selected"] else "No",
                number(p["weight"]),
            ]
            rows.append(row)
        if rows:
            table(headers, rows)
        else:
            paragraph(
                "No rankable or selected dimensions are available in this section."
            )

    paragraph("Top individual dimensions:")
    profile_table(dimensions["top"], full=True)
    paragraph("All selected dimensions and their fitted parameters:")
    profile_table(dimensions["selected"], full=True)
    if data["detailed"]:
        paragraph("Complete dimension profiles:")
        profile_table(dimensions["all"], full=True)

    title("Ranked prefixes and candidate cutoff")
    paragraph(
        "Dimensions are ranked by descending individual training AUROC, with exact "
        "ties resolved by numeric dimension ID. Each ranked prefix combines "
        "Gaussian similarity scores using an equal-weight mean; its composite "
        "training AUROC is evaluated separately. Individual AUROCs are not averaged. "
        "The earliest maximum-AUROC prefix sets an initial candidate cutoff. "
        "Candidates are the union of the top two, that prefix, and every dimension "
        "whose individual AUROC is strictly greater than 0.90. This candidate "
        "cutoff is different from the later acceptance threshold."
    )
    if ok:
        peak = selection["best_prefix_length"]
        extras = [
            d for d in selection["candidates"] if d not in selection["order"][:peak]
        ]
        high_extras = [d for d in extras if d not in selection["core"]]
        paragraph(
            f"Evaluated {len(selection['order'])} ranked prefixes. Best prefix length: "
            f"{peak}; candidate union: {len(selection['candidates'])} dimensions. "
            f"Added beyond the prefix: {len(extras)} dimensions, including "
            f"{len(high_extras)} non-core dimensions with individual AUROC >0.90."
        )
        table(
            ["Combination", "Dimensions", "Training AUROC"],
            [
                [
                    " / ".join(r["labels"]),
                    len(r["dimension_ids"]),
                    number(r["auroc"]["value"]),
                ]
                for r in selection["comparisons"]
            ],
        )
    else:
        paragraph(f"Candidate selection is unavailable: {data['reason']}.")

    title("Removal reasoning")
    if ok:
        paragraph(
            "Protected top two dimensions: "
            + ", ".join(map(str, selection["core"]))
            + ". "
            "Other candidates are tested for removal in individual-rank order. "
            "The first strict improvement in composite training AUROC is accepted, "
            "then the scan restarts. Protected dimensions are retained by policy."
        )
        rows = []
        for r in selection["accepted_removals"]:
            gain = r["gain_numerator"] / r["before_auroc"]["denominator"]
            rows.append(
                [
                    r["dimension"],
                    r["remaining_count"],
                    number(r["before_auroc"]["value"]),
                    number(r["after_auroc"]["value"]),
                    f"{gain:.6e}",
                ]
            )
        if rows:
            table(
                ["Removed", "Remaining", "Before AUROC", "After AUROC", "AUROC gain"],
                rows,
            )
        else:
            paragraph("No candidate dimension was removed.")
        paragraph(
            "Only the protected core remains; no removable dimensions remain."
            if selection["termination"] == "protected_core_only"
            else "The final complete scan found no remaining permitted "
            "single-dimension "
            "removal that improved AUROC."
        )
        paragraph(
            "Final dimension IDs: " + ", ".join(map(str, selection["selected"])) + ". "
            "This greedy stopping point does not prove global subset optimality. "
            "Removal does not imply a dimension lacks biological relevance."
        )
    else:
        paragraph("Removal reasoning is unavailable for this unsupported fit.")
    if data["detailed"] and selection["prefixes"]:
        paragraph("All evaluated prefixes:")
        table(
            ["Length", "AUROC", "Numerator", "Denominator", "Delta numerator"],
            [
                [
                    p["length"],
                    number(p["count"]["value"]),
                    p["count"]["numerator"],
                    p["count"]["denominator"],
                    p["delta_numerator"],
                ]
                for p in selection["prefixes"]
            ],
        )
        paragraph("Every recorded removal trial (iterations numbered from zero):")
        table(
            ["Iteration", "Dimension", "Trial AUROC", "Gain numerator", "Removed"],
            [
                [
                    i,
                    t["dimension"],
                    number(t["count"]["value"]),
                    t["gain_numerator"],
                    "Yes" if t["dimension"] == step["removed"] else "No",
                ]
                for i, step in enumerate(selection["iterations"])
                for t in step["trials"]
            ],
        )

    title("Composite score and acceptance threshold")
    paragraph(
        "For dimension j: width_j = max(sigma_C,j, sigma_floor); "
        "g_j(x) = exp(-0.5 * ((x_j - mu_C,j) / width_j)^2). "
        "Gaussian similarity is 1 at the member mean, about 0.607 at one effective "
        "width away, and about 0.135 at two widths away. "
        f"The numerical width floor is {number(data['scoring']['sigma_floor'])}."
    )
    paragraph(
        "The composite score S is the arithmetic mean of the selected Gaussian "
        "similarity scores: S(x) = sum(g_j(x)) / number_selected. Each selected "
        "dimension has equal weight. The score lies between 0 and 1. "
        "Higher scores indicate stronger average "
        "agreement with member patterns. This is not a product, Gaussian probability "
        "density, or membership probability. Computation uses stable log(S); "
        "the saved log cutoff remains authoritative if the ordinary score underflows."
    )
    threshold = data["scoring"]["threshold"]
    if threshold is None:
        paragraph(f"No acceptance threshold is available: {data['reason']}.")
    elif threshold["kind"] == "above_maximum":
        paragraph("The fitted threshold is predict-none: no windows are accepted.")
    else:
        score = threshold["score_value"]
        value = (
            number(score)
            if score
            else f"underflow; log score {number(threshold['log_value'])}"
        )
        paragraph(
            f"A composite score of {value} or higher is accepted "
            f"(log score >= {number(threshold['log_value'])})."
        )
    if threshold is not None:
        paragraph(
            "The threshold maximizes training balanced accuracy, with the highest "
            "cutoff chosen for ties. Balanced accuracy is (recall + specificity)/2; "
            "the objective does not maximize precision."
        )

    title("Metrics and their consequences")
    metrics = data["metrics"]["training"]
    if metrics is None:
        paragraph(
            "Training metrics are unavailable: "
            + str(data["metrics"]["training_unavailable_reason"])
            + "."
        )
    else:
        rows = []
        for key, label in [
            ("auroc", "AUROC"),
            ("recall", "Recall"),
            ("specificity", "Specificity"),
            ("balanced_accuracy", "Balanced accuracy"),
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
        ]:
            value = metrics[key]
            display = number(value) if key == "auroc" else percent(value)
            if value is None:
                display += ": " + str(
                    metrics["unavailable_reasons"].get(key, "not_defined")
                )
            rows.append([label, display])
        if data["detailed"]:
            rows += [
                ["Average precision", number(metrics["average_precision"])],
                ["F1", number(metrics["f1"])],
            ]
        table(["In-sample metric", "Value"], rows)
        table(
            ["TP", "FP", "TN", "FN"], [[metrics[k] for k in ("tp", "fp", "tn", "fn")]]
        )
        paragraph(
            f"At this threshold, {metrics['tp']} of {metrics['positive_n']} member "
            f"windows and {metrics['fp']} of {metrics['negative_n']} background "
            f"windows are accepted. Reference member prevalence is "
            f"{percent(metrics['positive_prevalence'])}. Precision depends on this "
            "class imbalance and does not establish deployment precision."
        )
    paragraph(
        "AUROC measures ranking separation, giving half credit to ties. These "
        "results are in-sample: C and B contributed to signature discovery and "
        "threshold selection. Held-out results are not included; optional research "
        "reports are separate."
    )

    title("Reference distributions and ABC match evidence")
    distributions = data["distributions"]
    if distributions["status"] != "ok":
        paragraph(
            "Reference distributions are unavailable: "
            + str(distributions["reason"])
            + "."
        )
    else:
        paragraph(
            "After the signature was fixed, its exact C/B windows were scored. "
            "The following descriptive quantiles use linear interpolation in "
            "sorted log scores, then exponentiation. They are not fitted Gaussian "
            "distributions; this convention can differ from interpolating "
            "ordinary scores."
        )
        rows = []
        for key in ("C", "B"):
            summary = distributions[key]
            paragraph(f"{key}: {summary['n']} windows; role: {summary['role']}.")
            for q in summary["quantiles"]:
                rows.append(
                    [
                        key,
                        percent(q["q"]),
                        "underflow; log score " + number(q["log_score"])
                        if q["score_underflow"]
                        else number(q["score"]),
                        number(q["log_score"]),
                    ]
                )
        table(["Population", "Percentile", "Composite score", "Log score"], rows)
    paragraph(
        "A — Acceptance compares a window's score with the fitted threshold. "
        "B — Background score (lower is stronger evidence) is "
        "(1 + count(background scores >= score)) / (n_B + 1). "
        "C — Cluster score (higher means higher rank among members) is "
        "count(member scores <= score) / n_C. These comparisons include exact ties. "
        "A high C percentile is not necessarily greater typicality; it can occur "
        "above every observed member score. No single ABC value belongs to a "
        "cluster independently of an evaluated window. These are not calibrated "
        "membership probabilities."
    )
    return "\n".join(lines).rstrip() + "\n"
