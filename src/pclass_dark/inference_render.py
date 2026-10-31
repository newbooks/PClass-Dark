"""Explicit inference report rendering; no interpretation or automatic files."""

import json

from rich.cells import cell_len

from .artifacts import IntegrityError
from .fit_records import thaw


def _display(value):
    return "".join(c if c.isprintable() else f"\\u{ord(c):04x}" for c in value)


def write_report(session, stream, *, format="text"):
    if format not in {"text", "json"}:
        raise ValueError("format must be text or json")
    if session._state != "active" or session._emitted:
        raise RuntimeError("Report requires a fresh entered, unconsumed session")
    written = 0

    def write(text):
        nonlocal written
        size = len(text.encode("utf-8"))
        session._quota(written + size)
        stream.write(text)
        written += size

    def encode(value):
        return json.dumps(
            thaw(value), ensure_ascii=True, allow_nan=False, separators=(",", ":")
        )

    if format == "json":
        write(
            '{"schema":"pclass-dark.infer-report.v1","source":' + encode(session.source)
        )
        write(',"execution":' + encode(session.execution))
        write(',"coverage":' + encode(session.coverage) + ',"queries":[')
    else:
        c = session.coverage
        write(
            f"Evaluated {c['clusters_evaluated']}/{c['clusters_total']} clusters; "
            f"{len(c['unavailable'])} unavailable\n"
        )
    count = matching = rows = windows = 0
    for result in session:
        if format == "json":
            if count:
                write(",")
            write(encode(result.to_dict()))
        else:
            write(
                f"Query {result.query_index + 1}: {_display(result.query_id)} "
                f"(length {result.length}; windows {len(result.windows)})\n"
            )
            if not result.matches:
                write("No accepted matches\n\n")
            else:
                table = [("Cluster", "A", "B ↓", "C ↑")]
                table.extend(
                    (
                        _display(m.cluster_id),
                        "Yes",
                        f"{100 * m.background_tail_fraction:.6g}%",
                        f"{100 * m.cluster_percentile_le:.6g}%",
                    )
                    for m in result.matches
                )
                widths = [max(cell_len(r[i]) for r in table) for i in range(4)]
                for row in table:
                    write(
                        "  ".join(
                            v + " " * (widths[i] - cell_len(v))
                            for i, v in enumerate(row)
                        ).rstrip()
                        + "\n"
                    )
                write("\n")
        count += 1
        matching += bool(result.matches)
        rows += len(result.matches)
        windows += len(result.windows)
    actual = {
        "queries_total": count,
        "queries_with_matches": matching,
        "matched_cluster_rows": rows,
        "windows_total": windows,
    }
    if actual != dict(session.counts) or count != session.query_count:
        raise IntegrityError("Inference report/query count mismatch")
    if format == "json":
        write('],"counts":' + encode(actual) + "}\n")
