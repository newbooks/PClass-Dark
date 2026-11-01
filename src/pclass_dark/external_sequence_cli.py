"""Standalone external-sequence script adapter; not a package subcommand."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False)


@app.command(
    "get-external-seq", context_settings={"help_option_names": ["-h", "--help"]}
)
def get_external_seq(
    go_graph: Annotated[
        Path | None, typer.Option(help="Trusted local profunc MF graph pickle.")
    ] = None,
    swissprot_accessions: Annotated[
        Path | None, typer.Option(help="Local Swiss-Prot accession list.")
    ] = None,
    reference_fasta: Annotated[
        Path | None, typer.Option(help="Manifest-matched profunc source FASTA.")
    ] = None,
    gaf: Annotated[
        Path | None,
        typer.Option(help="Local GOA GAF gzip; sequence retrieval stays online."),
    ] = None,
    refresh: bool = typer.Option(
        False, help="Start a fresh annotation/sequence generation."
    ),
    workers: int = typer.Option(
        4, min=1, max=32, help="Bounded retrieval threads; 1 for serial."
    ),
    batch_size: int = typer.Option(
        100, min=1, max=100, help="Accessions per sequence request."
    ),
    retries: int = typer.Option(
        3, min=0, max=10, help="Additional transient HTTP attempts."
    ),
):
    """Collect experimental GO MF proteins outside local Swiss-Prot into cwd."""
    import logging

    from .external_sequences import collect_external_sequences

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = collect_external_sequences(
            work_dir=Path.cwd(),
            go_graph=go_graph,
            swissprot_accessions=swissprot_accessions,
            reference_fasta=reference_fasta,
            gaf=gaf,
            refresh=refresh,
            workers=workers,
            batch_size=batch_size,
            retries=retries,
        )
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f"{result.fasta_path}\n{result.report_path}\n"
        f"Sequences: {result.counts[-1].sequences}"
    )
