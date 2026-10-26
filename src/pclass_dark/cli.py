"""Command-line interface; scientific functionality belongs in library modules."""

import typer

from pclass_dark import __version__

app = typer.Typer(
    name="pclass-dark",
    help="Protein function classification using cluster-conserved PLM signatures.",
    no_args_is_help=True,
)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"pclass-dark {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_show_version,
        is_eager=True,
        help="Show the installed version and exit.",
    ),
) -> None:
    """Entry point for the command/subcommand hierarchy."""


def main() -> None:
    """Run the installed pclass-dark command."""
    app()


@app.command("setup-data")
def setup_data(
    output_dir: str = typer.Option("work/setup-data", help="Derived data destination."),
    min_clusters: int = typer.Option(
        10, min=1, help="Minimum unique full-sequence members per cluster."
    ),
    seed: int = typer.Option(
        42, min=0, help="Seed for per-cluster C folds and fixed B samples."
    ),
    workers: int = typer.Option(1, min=1, help="CPU workers for cluster preparation."),
    prepare_only: bool = typer.Option(
        False, help="Prepare offline without loading a PLM."
    ),
    offline: bool = typer.Option(
        False, help="Forbid model/tokenizer network requests."
    ),
    model: str = typer.Option(
        "facebook/esm2_t30_150M_UR50D", help="Supported frozen model."
    ),
    revision: str = typer.Option(
        "a695f6045e2e32885fa60af20c13cb35398ce30c", help="Pinned model revision."
    ),
    device: str = typer.Option("auto", help="auto, cpu, cuda, or cuda:<index>."),
    batch_size: int = typer.Option(
        8, min=1, help="Inference batch size; adjustable on resume."
    ),
    force: bool = typer.Option(False, help="Archive and recompute matching bundles."),
) -> None:
    """Prepare references, C folds, fixed B, and resumable PLM embeddings."""
    import logging

    from pclass_dark.setup_data import prepare_data

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        result = prepare_data(
            output_dir,
            min_clusters=min_clusters,
            seed=seed,
            workers=workers,
            prepare_only=prepare_only,
            offline=offline,
            model=model,
            revision=revision,
            device=device,
            batch_size=batch_size,
            force=force,
        )
    except KeyboardInterrupt:
        typer.echo("Interrupted; committed embedding shards can be resumed.", err=True)
        raise typer.Exit(130) from None
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Prepared: {result.prepared_dir}")
    if result.embedding_dir is not None:
        typer.echo(f"Embeddings: {result.embedding_dir}")
