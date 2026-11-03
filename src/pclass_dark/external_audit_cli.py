"""Thin CLI adapters for the external audit library."""

from pathlib import Path
from typing import Annotated

import typer


def external_audit_command(
    input_table: Annotated[
        Path, typer.Argument(help="External-evaluation TSV (any filename).")
    ],
    output_dir: Annotated[
        Path | None, typer.Option(help="Default: beside input table.")
    ] = None,
    bins: Annotated[int, typer.Option(min=1, max=100)] = 10,
    hit_cutoff: Annotated[float, typer.Option(min=0, max=1)] = 0.7,
    truth_tsv: Annotated[
        Path | None,
        typer.Option(
            help="Query_id/go_mf_terms TSV; default: saved evaluation annotations."
        ),
    ] = None,
) -> None:
    """Plot v2 audits excluding queries annotated with GO:0005515 (protein binding)."""
    from .external_audit import audit_external

    try:
        paths, _, counts = audit_external(
            input_table,
            output_dir,
            bins=bins,
            cutoff=hit_cutoff,
            exclude_go="GO:0005515",
            truth_tsv=truth_tsv,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"Excluded {counts['excluded_queries']} / {counts['original_queries']} "
        f"queries annotated GO:0005515; retained {counts['retained_queries']}"
    )
    typer.echo(
        f"Embedding distances >1: {counts['distance_over_one']} / "
        f"{counts['distance_defined']}; maximum: {counts['distance_max']}"
    )
    for path in paths:
        typer.echo(str(path))


def main():
    typer.run(external_audit_command)
