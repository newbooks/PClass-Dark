"""Command-line interface; scientific functionality belongs in library modules."""

from typing import Annotated

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


@app.command("setup-data", context_settings={"help_option_names": ["-h", "--help"]})
def setup_data(
    output_dir: str | None = typer.Option(
        None, help="Destination; default: repository data/setup-data, or user cache."
    ),
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


@app.command("build-library", context_settings={"help_option_names": ["-h", "--help"]})
def build_library_command(
    data_dir: str | None = typer.Option(
        None, help="Input; default: repository data/setup-data, or bundled/user data."
    ),
    output_dir: str | None = typer.Option(
        None, help="Destination; default: repository data/library, or user cache."
    ),
    workers: int = typer.Option(1, min=1, help="Bounded CPU target workers."),
    memory_mib: int = typer.Option(
        1024, min=64, help="Total numeric workspace budget."
    ),
    epsilon: float = typer.Option(1e-8, help="Positive R/D safeguard."),
    sigma_floor: float = typer.Option(1e-8, help="Positive Gaussian width floor."),
    cluster: Annotated[
        list[str] | None, typer.Option(help="Exact target cluster ID; repeatable.")
    ] = None,
    cross_validate: bool = typer.Option(
        False, help="Also run five research folds; use --cluster to limit cost."
    ),
    force: bool = typer.Option(False, help="Archive and recompute this library."),
) -> None:
    """Build all-C signatures and C/B distributions; optionally research five folds."""
    import logging

    from pclass_dark.library import build_library

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        result = build_library(
            data_dir,
            output_dir,
            workers=workers,
            memory_mib=memory_mib,
            epsilon=epsilon,
            sigma_floor=sigma_floor,
            cluster_ids=cluster,
            cross_validate=cross_validate,
            force=force,
        )
    except KeyboardInterrupt:
        typer.echo(
            "Interrupted; validated library checkpoints can be resumed.", err=True
        )
        raise typer.Exit(130) from None
    except (ValueError, RuntimeError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Library: {result.library_dir}")
    if result.relative_dir is not None:
        typer.echo(f"Relative scores: {result.relative_dir}")
    if result.research_dir is not None:
        typer.echo(f"Research: {result.research_dir}")


def _one_cluster(values: list[str] | None) -> list[str] | None:
    if values is not None and len(values) > 1:
        raise typer.BadParameter("Provide --cluster exactly once.")
    return values


@app.command(
    "explain-signature", context_settings={"help_option_names": ["-h", "--help"]}
)
def explain_signature_command(
    cluster: Annotated[
        list[str] | None,
        typer.Option("--cluster", callback=_one_cluster, help="One exact cluster ID."),
    ] = None,
    list_mode: bool = typer.Option(
        False, "--list", help="List all signature summaries."
    ),
    library_dir: str | None = typer.Option(
        None,
        help="Library root; default: repository data/library, or bundled/user data.",
    ),
    detailed: bool = typer.Option(
        False, help="Include all profiles, prefixes and trials."
    ),
    format: str = typer.Option("text", help="Report format: text, markdown or json."),
) -> None:
    """Explain one cluster signature or list all in-sample signature summaries."""
    if bool(cluster) == list_mode:
        raise typer.BadParameter("Provide exactly one of --cluster or --list.")
    if list_mode and detailed:
        raise typer.BadParameter("--detailed applies only to --cluster.")
    if format not in ("text", "markdown", "json"):
        raise typer.BadParameter(
            "Must be text, markdown or json.", param_hint="--format"
        )
    from .explanations import (
        explain_signature,
        list_signature_summaries,
        render_signature_explanation,
        render_signature_summaries,
    )

    try:
        if list_mode:
            listing = list_signature_summaries(library_dir)
            rendered = render_signature_summaries(listing, format=format)
        else:
            report = explain_signature(cluster[0], library_dir, detailed=detailed)
            rendered = render_signature_explanation(report, format=format)
    except KeyboardInterrupt:
        typer.echo("Interrupted.", err=True)
        raise typer.Exit(130) from None
    except (ValueError, RuntimeError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(rendered, nl=False)
