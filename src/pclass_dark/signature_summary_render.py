"""Aligned deterministic listing strings, independent of terminal dimensions."""

from rich.cells import cell_len

TITLE = "Final signatures — in-sample reference metrics"
HEADERS = (
    "Cluster",
    "Description",
    "Status",
    "C_windows",
    "B_windows",
    "Dimensions",
    "AUROC",
    "Recall%",
    "Specificity%",
    "Precision%",
)
NOTE = (
    "Metrics are in-sample; reference populations were used during signature "
    "discovery and threshold selection. Values are rounded to three decimal places; "
    "small positive values may display as 0.000. Use --cluster for an explanation. "
    "Precision depends on the sampled reference population balance."
)


def visible(value):
    """Escape control/nonprinting characters before measuring or displaying IDs."""
    return "".join(ascii(c)[1:-1] if not c.isprintable() else c for c in str(value))


def short_description(value):
    """Cap a visible description at 24 characters/cells, including literal dots."""
    if value is None:
        return "—"
    text = visible(value)
    if len(text) <= 24 and cell_len(text) <= 24:
        return text
    prefix = ""
    for char in text:
        if len(prefix + char) > 21 or cell_len(prefix + char) > 21:
            break
        prefix += char
    return prefix + "..."


def metric(value, *, percentage=False):
    if value is None:
        return "—"
    value = value * 100 if percentage else value
    return format(0.0 if value == 0 else value, ".3f")


def render_table(data, *, markdown=False):
    """Render all rows in display-cell-aligned columns; performs no IO."""
    show_status = not data["rows"] or any(row["status"] != "ok" for row in data["rows"])
    columns = [i for i in range(len(HEADERS)) if i != 2 or show_status]
    headers = tuple(HEADERS[i] for i in columns)
    rows = []
    for row in data["rows"]:
        rows.append(
            [
                visible(row["cluster_id"]),
                short_description(row.get("description")),
                visible(
                    "ok" if row["status"] == "ok" else row["reason"] or row["status"]
                ),
                str(row["C_windows"]),
                str(row["B_windows"]),
                str(row["dimensions"]),
                metric(row["auroc"]),
                *(
                    metric(row[k], percentage=True)
                    for k in ("recall", "specificity", "precision")
                ),
            ]
        )
    rows = [[row[i] for i in columns] for row in rows]
    widths = [
        max(cell_len(r[i]) for r in [headers, *rows]) for i in range(len(headers))
    ]

    def padded(cells):
        result = []
        for i, cell in enumerate(cells):
            padding = " " * (widths[i] - cell_len(cell))
            result.append(
                cell + padding
                if headers[i] in ("Cluster", "Description", "Status")
                else padding + cell
            )
        return result

    def line(cells):
        cells = padded(cells)
        if markdown:
            cells = [c.replace("\\", "\\\\").replace("|", "\\|") for c in cells]
            return "| " + " | ".join(cells) + " |"
        return "  ".join(cells).rstrip()

    lines = [("## " if markdown else "") + TITLE, "", line(headers)]
    if markdown:
        lines.append(
            "| "
            + " | ".join(
                ":---" if h in ("Cluster", "Description", "Status") else "---:"
                for h in headers
            )
            + " |"
        )
    lines.extend(line(row) for row in rows)
    if not rows:
        lines.extend(["", "No clusters in selected library."])
    lines.extend(
        ["", NOTE, "", render_distributions(data["distributions"], markdown=markdown)]
    )
    return "\n".join(lines) + "\n"


def render_distributions(distributions, *, markdown=False):
    """Render all 20 bins and per-metric availability without recomputing scores."""
    keys = ("auroc", "recall", "specificity", "precision")
    headers = (
        "AUROC_bin",
        "Rate_bin%",
        "AUROC_n",
        "Recall_n",
        "Specificity_n",
        "Precision_n",
    )
    rows = []
    edges = distributions["edges"]
    for index in range(distributions["bin_count"]):
        right = "]" if index == distributions["bin_count"] - 1 else ")"
        low, high = edges[index : index + 2]
        rows.append(
            [
                f"[{low:.3f},{high:.3f}{right}",
                f"[{low * 100:.3f},{high * 100:.3f}{right}",
                *(str(distributions["metrics"][key]["counts"][index]) for key in keys),
            ]
        )
    title = "In-sample metric distributions — cluster counts (20 bins)"
    lines = [("## " if markdown else "") + title, ""]

    def table(headings, values, left_columns):
        widths = [
            max(cell_len(row[i]) for row in [headings, *values])
            for i in range(len(headings))
        ]

        def line(row):
            cells = []
            for i, cell in enumerate(row):
                pad = " " * (widths[i] - cell_len(cell))
                cells.append(cell + pad if i < left_columns else pad + cell)
            return (
                "| " + " | ".join(cells) + " |"
                if markdown
                else "  ".join(cells).rstrip()
            )

        lines.append(line(headings))
        if markdown:
            lines.append(
                "| "
                + " | ".join(
                    ":---" if i < left_columns else "---:" for i in range(len(headings))
                )
                + " |"
            )
        lines.extend(line(row) for row in values)

    table(headers, rows, 2)
    lines.extend(
        [
            "",
            "One count per cluster per available metric; "
            "values are binned before rounding.",
            "",
        ]
    )
    table(
        ("Metric", "Valid_n", "Unavailable_n"),
        [
            [
                key.upper() if key == "auroc" else key.capitalize(),
                str(distributions["metrics"][key]["valid"]),
                str(distributions["metrics"][key]["unavailable"]),
            ]
            for key in keys
        ],
        1,
    )
    return "\n".join(lines)
