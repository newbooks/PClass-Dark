"""Thin experimental script adapter; intentionally absent from the root CLI."""

import logging
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False)


@app.command(context_settings={"help_option_names": ["-h", "--help"]})
def command(
    prepared_dir: Path | None = None,
    model_dir: Path | None = None,
    device: str = "auto",
    offline: bool = False,
    window_batch_size: Annotated[int, typer.Option(min=1, max=8)] = 8,
    workers: Annotated[int, typer.Option(min=1, max=8)] = 8,
    resume: bool = False,
    force: bool = False,
):
    """Extract selected unique sequences into 19 features in ./features.tsv."""
    from .feature_extraction import extract_features

    if resume and force:
        raise typer.BadParameter("--resume and --force cannot be combined")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        result = extract_features(
            prepared_dir,
            Path.cwd() / "features.tsv",
            model_dir,
            device=device,
            offline=offline,
            window_batch_size=window_batch_size,
            workers=workers,
            resume=resume,
            force=force,
        )
    except KeyboardInterrupt:
        typer.echo("Interrupted; completed feature receipts can be resumed.", err=True)
        raise typer.Exit(130) from None
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f"{result.tsv_path}\nSequences: {result.sequence_count:,}\n"
        f"Memberships: {result.membership_count:,}\nClusters: {result.cluster_count:,}"
    )


def main():
    app()
