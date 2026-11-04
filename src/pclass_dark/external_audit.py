"""Descriptive accuracy/coverage audits of external-evaluation TSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .external_evaluation_types import HEADERS


@dataclass(frozen=True)
class AuditData:
    query_ids: tuple[str, ...]
    predicted: tuple[np.ndarray, np.ndarray]
    matches: tuple[np.ndarray, np.ndarray]
    evidence: tuple[np.ndarray, np.ndarray]


def read_audit_table(path: str | Path) -> AuditData:
    """Read a comparison table; raise ValueError for invalid rows or numbers."""
    ids, seen = [], set()
    predicted = ([], [])
    matches = ([], [])
    evidence = ([], [])
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(HEADERS) <= set(fields):
            raise ValueError("Expected unique external-evaluation TSV headers")
        for line, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Line {line}: malformed TSV row")
            query = row["Query_id"].strip()
            if not query or query == "NA" or query in seen:
                raise ValueError(f"Line {line}: empty or duplicate Query_id")
            seen.add(query)
            ids.append(query)
            for i, (go, match, confidence, maximum) in enumerate(
                (
                    ("PClass_go", "Pclass_match", "Embedding_d", 2),
                    ("Dark_go", "Dark_match", "Dark_Bscore", 1),
                )
            ):
                present = row[go].strip() not in {"", "NA"}
                values = []
                for name, upper in ((match, 1), (confidence, maximum)):
                    raw = row[name].strip()
                    try:
                        value = np.nan if raw in {"", "NA"} else float(raw)
                    except ValueError as exc:
                        raise ValueError(f"Line {line}: invalid {name}") from exc
                    if raw not in {"", "NA"} and (
                        not np.isfinite(value) or not 0 <= value <= upper
                    ):
                        raise ValueError(f"Line {line}: {name} outside [0,{upper}]")
                    values.append(value)
                if np.isfinite(values[0]) and not present:
                    raise ValueError(f"Line {line}: match score without GO prediction")
                predicted[i].append(present)
                matches[i].append(values[0])
                evidence[i].append(values[1])
    return AuditData(
        tuple(ids),
        tuple(np.asarray(v, dtype=bool) for v in predicted),
        tuple(np.asarray(v, dtype=float) for v in matches),
        tuple(np.asarray(v, dtype=float) for v in evidence),
    )


def _options(cutoff, bins=10):
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("hit cutoff must be finite and in [0,1]")
    if isinstance(bins, bool) or not isinstance(bins, int) or not 1 <= bins <= 100:
        raise ValueError("bins must be an integer between 1 and 100")


def confidence_bins(predicted, match, evidence, *, bins=10, cutoff=0.7):
    """Return labels and hit/non-hit/undefined counts; preserve overflow and NA."""
    _options(cutoff, bins)
    edges = np.linspace(0, 1, bins + 1)
    labels = [f"{a:.2g}–{b:.2g}" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    overflow = bool(np.any(predicted & (evidence > 1)))
    missing = bool(np.any(predicted & ~np.isfinite(evidence)))
    if overflow:
        labels.append(">1")
    if missing:
        labels.append("NA")
    counts = np.zeros((3, len(labels)), dtype=int)
    for i in np.flatnonzero(predicted):
        value = evidence[i]
        if not np.isfinite(value):
            column = len(labels) - 1
        elif value > 1:
            column = bins
        else:
            column = min(int(np.searchsorted(edges, value, side="right")) - 1, bins - 1)
        category = 2 if not np.isfinite(match[i]) else int(match[i] < cutoff)
        counts[category, column] += 1
    return labels, counts


def coverage_curve(predicted, match, evidence, *, cutoff=0.7):
    """Return coverage/hit-rate percentages at achievable tied thresholds."""
    _options(cutoff)
    selected = np.flatnonzero(predicted & np.isfinite(evidence))
    selected = selected[np.argsort(evidence[selected], kind="stable")]
    if not selected.size:
        return np.array([]), np.array([])
    end = np.r_[np.flatnonzero(np.diff(evidence[selected]) != 0), selected.size - 1]
    valid = np.cumsum(np.isfinite(match[selected]))[end]
    hits = np.cumsum(match[selected] >= cutoff)[end]
    rates = np.divide(
        100.0 * hits, valid, out=np.full(len(end), np.nan), where=valid > 0
    )
    return 100.0 * (end + 1) / len(predicted), rates


def summarize_audit(data: AuditData, *, cutoff=0.7):
    """Return per-method descriptive counts and shared query-level hit outcomes."""
    _options(cutoff)
    total = len(data.query_ids)
    hits = tuple(m >= cutoff for m in data.matches)
    rows = []
    for i, name in enumerate(("PClass", "PClass-Dark")):
        predicted, match, evidence = (
            data.predicted[i],
            data.matches[i],
            data.evidence[i],
        )
        n = int(predicted.sum())
        scorable = int(np.isfinite(match).sum())
        h = int(hits[i].sum())
        rows.append(
            dict(
                method=name,
                queries=total,
                predictions=n,
                scorable=scorable,
                hits=h,
                coverage_pct=100 * n / total if total else None,
                hit_rate_pct=100 * h / scorable if scorable else None,
                hits_per_query_pct=100 * h / total if total else None,
                undefined_matches=n - scorable,
                missing_confidence=int((predicted & ~np.isfinite(evidence)).sum()),
                both_hit=int((hits[0] & hits[1]).sum()),
                method_only_hit=int((hits[i] & ~hits[1 - i]).sum()),
            )
        )
    distance = data.evidence[0]
    finite = int(np.isfinite(distance).sum())
    over = int((distance > 1).sum())
    return rows, dict(
        distance_defined=finite,
        distance_over_one=over,
        distance_over_one_pct=100 * over / finite if finite else None,
        distance_max=float(np.nanmax(distance)) if finite else None,
        jointly_scorable=int(
            (np.isfinite(data.matches[0]) & np.isfinite(data.matches[1])).sum()
        ),
    )


def filter_query_go(data: AuditData, annotations, *, exclude_go="GO:0005515"):
    """Remove entire queries containing a truth term; never filter predicted GO."""
    from .external_evaluation_types import go_set

    go_set([exclude_go], nonempty=True)
    if set(annotations) != set(data.query_ids):
        raise ValueError(
            "Truth annotations must cover exactly the input Query_id values"
        )
    keep = np.asarray(
        [
            exclude_go not in go_set(annotations[q], nonempty=True)
            for q in data.query_ids
        ],
        dtype=bool,
    )
    return AuditData(
        tuple(q for q, retained in zip(data.query_ids, keep, strict=True) if retained),
        tuple(a[keep] for a in data.predicted),
        tuple(a[keep] for a in data.matches),
        tuple(a[keep] for a in data.evidence),
    )


def read_query_annotations(input_path, data: AuditData, truth_tsv=None):
    """Read explicit truth or hash-verified saved scoring truth for this table."""
    import json

    from .artifacts import file_record
    from .external_evaluation_types import digest_id, go_set

    annotations = {}

    def add(query, terms):
        if not query or query in annotations:
            raise ValueError("Missing or duplicate truth Query_id")
        annotations[query] = go_set(terms, nonempty=True)

    if truth_tsv is not None:
        with Path(truth_tsv).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames != ["Query_id", "go_mf_terms"]:
                raise ValueError("Truth TSV requires Query_id and go_mf_terms columns")
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("Malformed truth TSV row")
                add(row["Query_id"].strip(), row["go_mf_terms"].split("|"))
    else:
        path = Path(input_path)
        marker_path = path.with_name("." + path.name + ".evaluation.json")
        if not marker_path.is_file():
            raise ValueError(
                "Query GO annotations unavailable; supply --truth-tsv "
                "(Query_id and go_mf_terms) or use the published evaluation table"
            )

        def verified(file, expected):
            if (
                any(p.is_symlink() for p in (file, *file.parents))
                or file_record(file) != expected
            ):
                raise ValueError(f"Evaluation source identity mismatch: {file}")

        try:
            marker = json.loads(marker_path.read_text())
            verified(path, marker["tsv"])
            state = path.parent / ".external-evaluation"
            if marker.get("marker_version") == 2:
                relative = marker["state_relative"]
                if (
                    not isinstance(relative, str)
                    or not relative
                    or Path(relative).is_absolute()
                    or marker["evaluation_kind"] not in {"external", "swiss"}
                ):
                    raise ValueError("Invalid evaluation state location")
                state = path.parent / relative
                if any(p.is_symlink() for p in (state, *state.parents)):
                    raise ValueError("Evaluation state symlink")
                if (state / "workspace.json").is_symlink():
                    raise ValueError("Evaluation workspace metadata symlink")
                workspace = json.loads((state / "workspace.json").read_text())
                if workspace["workspace_uuid"] != marker["workspace_uuid"]:
                    raise ValueError("Evaluation workspace identity mismatch")
            elif "marker_version" in marker:
                raise ValueError("Unknown evaluation marker version")
            score = state / "scores" / digest_id(marker["score_id"])
            verified(score / "manifest.json", marker["score_manifest"])
            manifest = json.loads((score / "manifest.json").read_text())
            if manifest["id"] != marker["score_id"] or manifest["query_count"] != len(
                data.query_ids
            ):
                raise ValueError("Evaluation metadata does not match input queries")
            status = score / "query_status.jsonl"
            verified(status, manifest["files"]["query_status.jsonl"])
            with status.open() as stream:
                for ordinal, line in enumerate(stream):
                    row = json.loads(line)
                    if (
                        ordinal >= len(data.query_ids)
                        or row["ordinal"] != ordinal
                        or row["query_id"] != data.query_ids[ordinal]
                    ):
                        raise ValueError("Scoring metadata query order mismatch")
                    add(row["query_id"], row["truth_go"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed evaluation annotation metadata") from exc
    if set(annotations) != set(data.query_ids):
        raise ValueError(
            "Truth annotations must cover exactly the input Query_id values"
        )
    return annotations


def audit_external(
    input_path, output_dir=None, *, bins=10, cutoff=0.7, exclude_go=None, truth_tsv=None
):
    """Validate, analyze and publish PNG/TSV artifacts; return paths and counts.

    Raises ValueError/OSError for input/output errors and RuntimeError when the
    optional plotting dependency is unavailable. Existing audit outputs are replaced.
    With exclude_go, use query truth from verified metadata or truth_tsv, remove
    whole queries before analysis, and publish _v2 outputs. Without it, preserve
    the original unfiltered API behavior and filenames.
    """
    return _audit(
        input_path,
        output_dir,
        bins=bins,
        cutoff=cutoff,
        exclude_go=exclude_go,
        truth_tsv=truth_tsv,
        prefix="external_audit",
        summary_title="External audit",
    )


def _audit(
    input_path,
    output_dir=None,
    *,
    bins=10,
    cutoff=0.7,
    exclude_go=None,
    truth_tsv=None,
    prefix,
    summary_title,
):
    """Shared validated analysis, rendering and staged per-file publication."""
    import os
    import tempfile

    _options(cutoff, bins)
    data = read_audit_table(input_path)
    original_count = len(data.query_ids)
    if exclude_go is not None:
        annotations = read_query_annotations(input_path, data, truth_tsv)
        data = filter_query_go(data, annotations, exclude_go=exclude_go)
    rows, diagnostics = summarize_audit(data, cutoff=cutoff)
    if exclude_go is not None:
        diagnostics.update(
            original_queries=original_count,
            excluded_queries=original_count - len(data.query_ids),
            retained_queries=len(data.query_ids),
            excluded_go=exclude_go,
        )
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError(
            "Install plotting support: pip install 'pclass-dark[audit]'"
        ) from exc
    output = Path(output_dir) if output_dir is not None else Path(input_path).parent
    suffix = "_v2" if exclude_go is not None else ""
    names = [
        f"{prefix}_{name}{suffix}.png"
        for name in ("match_scores", "confidence", "coverage", "summary")
    ]
    names.append(f"{prefix}_summary{suffix}.tsv")
    sources = {Path(input_path).resolve()}
    if truth_tsv is not None:
        sources.add(Path(truth_tsv).resolve())
    if sources & {(output / name).resolve() for name in names}:
        raise ValueError("Audit output would overwrite the input table")
    for name in names:
        destination = output / name
        if destination.is_symlink():
            raise ValueError("Audit destination must not be a symlink")
        if destination.exists() and any(
            source.exists() and destination.samefile(source) for source in sources
        ):
            raise ValueError("Audit output would overwrite an input source")
    output.mkdir(parents=True, exist_ok=True)
    colors = ("#278878", "#df9262", "#a4a9b0")

    def figure(width=12, height=5):
        fig = Figure(figsize=(width, height), layout="constrained")
        FigureCanvasAgg(fig)
        return fig

    with tempfile.TemporaryDirectory(prefix=".external-audit-", dir=output) as temp:
        stage = Path(temp)
        fig = figure(7, 7)
        ax = fig.subplots()
        paired = np.isfinite(data.matches[0]) & np.isfinite(data.matches[1])
        ax.scatter(
            data.matches[0][paired],
            data.matches[1][paired],
            s=9,
            alpha=0.22,
            color="#355d86",
            rasterized=True,
        )
        ax.plot([0, 1], [0, 1], color="#777777", linewidth=1)
        ax.axvline(cutoff, color="#aaaaaa", linestyle="--")
        ax.axhline(cutoff, color="#aaaaaa", linestyle="--")
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="PClass GO match score",
            ylabel="PClass-Dark GO match score",
            title=f"GO match agreement · {int(paired.sum()):,} paired queries\n"
            f"{len(paired) - int(paired.sum()):,} excluded (missing score)",
        )
        ax.set_aspect("equal")
        fig.savefig(stage / names[0], dpi=180)
        fig.clear()

        fig = figure(14, 6)
        axes = fig.subplots(1, 2, sharey=True)
        max_bar = 1.0
        for i, ax in enumerate(axes):
            labels, counts = confidence_bins(
                data.predicted[i],
                data.matches[i],
                data.evidence[i],
                bins=bins,
                cutoff=cutoff,
            )
            positions = np.arange(len(labels))
            bottom = np.zeros(len(labels))
            for values, color, label in zip(
                counts, colors, ("Hit", "Non-hit", "Undefined match"), strict=True
            ):
                ax.bar(positions, values, bottom=bottom, color=color, label=label)
                bottom += values
            for j, total in enumerate(bottom):
                valid = counts[0, j] + counts[1, j]
                if total:
                    annotation = f"{100 * counts[0, j] / valid:.1f}%" if valid else "NA"
                    ax.annotate(
                        annotation,
                        (j, total),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )
            ax.set_xticks(positions, labels, rotation=45, ha="right")
            ax.set(
                xlabel=(
                    "Embedding distance" if i == 0 else "Background-tail fraction (B)"
                ),
                ylabel="Predictions (count)",
                title=rows[i]["method"] + " · smaller is stronger",
            )
            max_bar = max(max_bar, float(bottom.max()))
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        axes[0].set_ylim(0, max_bar * 1.2)
        fig.suptitle(
            f"Confidence distributions · hit = GO match ≥ {cutoff:g}\n"
            "Annotations: hits / defined matches; grey excluded from percentage"
        )
        fig.savefig(stage / names[1], dpi=180)
        fig.clear()

        fig = figure(12, 5)
        axes = fig.subplots(1, 2, sharex=True, sharey=True)
        for i, ax in enumerate(axes):
            x, y = coverage_curve(
                data.predicted[i], data.matches[i], data.evidence[i], cutoff=cutoff
            )
            ax.plot(x, y, color=("#355d86", colors[0])[i], linewidth=2)
            if len(x):
                ax.scatter(x[-1:], y[-1:], color=("#355d86", colors[0])[i], s=20)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No rankable predictions",
                    transform=ax.transAxes,
                    ha="center",
                )
            ax.set(
                xlim=(0, 100),
                ylim=(0, 101),
                title=rows[i]["method"],
                xlabel="Retained predictions / all queries (%)\nStrong to weak",
                ylabel="Hit rate (%)",
            )
            ax.grid(alpha=0.2)
        fig.suptitle(
            f"Coverage versus semantic hit rate · cutoff {cutoff:g}\n"
            "Smaller evidence retained first; ties kept together; "
            "undefined matches excluded from hit rate"
        )
        fig.savefig(stage / names[2], dpi=180)
        fig.clear()

        fig = figure(13, 5.7 if exclude_go else 5)
        ax = fig.subplots()
        ax.axis("off")
        metrics = [
            ("All queries", "queries"),
            ("Predictions", "predictions"),
            ("Prediction coverage (%)", "coverage_pct"),
            ("Scorable", "scorable"),
            ("Hits", "hits"),
            ("Hit rate (%)", "hit_rate_pct"),
            ("Hits / all queries (%)", "hits_per_query_pct"),
            ("Undefined matches", "undefined_matches"),
            ("Missing confidence", "missing_confidence"),
            ("Both methods hit", "both_hit"),
            ("Only this method hits", "method_only_hit"),
        ]

        if exclude_go is not None:
            metrics = [
                ("Original queries", "original_queries"),
                (f"Queries excluded ({exclude_go})", "excluded_queries"),
            ] + metrics
            rows = [
                {
                    **row,
                    "original_queries": original_count,
                    "excluded_queries": original_count - len(data.query_ids),
                }
                for row in rows
            ]

        def display(v):
            return (
                "NA" if v is None else f"{v:.2f}" if isinstance(v, float) else f"{v:,}"
            )

        table = ax.table(
            cellText=[
                [label, *(display(r[key]) for r in rows)] for label, key in metrics
            ],
            colLabels=["Metric", "PClass", "PClass-Dark"],
            cellLoc="center",
            loc="center",
            colWidths=[0.48, 0.24, 0.24],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        pct = diagnostics["distance_over_one_pct"]
        detail = "NA" if pct is None else f"{pct:.4f}%"
        ax.set_title(
            f"{summary_title} · semantic hit cutoff {cutoff:g}\n"
            f"Embedding distance >1: {diagnostics['distance_over_one']:,} / "
            f"{diagnostics['distance_defined']:,} ({detail})",
            pad=18,
        )
        fig.savefig(stage / names[3], dpi=180)
        fig.clear()
        with (stage / names[4]).open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(dict.fromkeys([*rows[0], *diagnostics])),
                delimiter="\t",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: "NA" if v is None else v
                        for k, v in {**row, **diagnostics}.items()
                    }
                )
        for name in names:
            os.replace(stage / name, output / name)
    return tuple(output / name for name in names), rows, diagnostics
