"""Thin Typer adapters for the experimental SIAC script and command."""

from pathlib import Path
from typing import Annotated

import typer

from .ml_progress import terminal_progress

app = typer.Typer(add_completion=False)


@app.command(context_settings={"help_option_names": ["-h", "--help"]})
def command(
    features: Path = Path("features.tsv"),
    similarities: Path = Path("cluster_dmatrix.tsv"),
    output_dir: Path = Path("explore-ml"),
    sequence_index: Path | None = None,
    groups: Path | None = None,
    group_provenance: Path | None = None,
    similarity_provenance: Path | None = None,
    split_mode: str = "accession",
    conflict_policy: str = "exclude",
    repeats: Annotated[int, typer.Option(min=1)] = 1,
    seed: Annotated[int, typer.Option(min=0, max=2**31 - 1)] = 42,
    alpha: float = 0.1,
    neighbors: Annotated[int, typer.Option(min=1)] = 10,
    similarity_power: float = 2.0,
    knots: Annotated[int, typer.Option(min=2)] = 8,
    max_epochs: Annotated[int, typer.Option(min=1)] = 100,
    patience: Annotated[int, typer.Option(min=1)] = 10,
    min_delta: float = 1e-5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: Annotated[int, typer.Option(min=1)] = 1024,
    device: str = "auto",
    workers: Annotated[int, typer.Option(min=1, max=8)] = 8,
    threads: Annotated[int, typer.Option(min=1)] = 1,
    memory_budget_gib: float = 8.0,
    validate_only: bool = False,
    resume: bool = False,
):
    """Explore SIAC with random five-fold CV, hit-rate reports and matched controls."""
    from .explore_ml import ExploreMLConfig, run_exploration

    values = locals().copy()
    values.pop("ExploreMLConfig")
    values.pop("run_exploration")
    try:
        config = ExploreMLConfig(**values)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        with terminal_progress():
            result = run_exploration(config)
    except KeyboardInterrupt:
        typer.echo("Interrupted; committed fold tasks can be resumed.", err=True)
        raise typer.Exit(130) from None
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if config.validate_only:
        typer.echo(f"{result.preflight_path}\nStatus: {result.status}")
        if result.status != "ready":
            raise typer.Exit(1)
    else:
        typer.echo(
            f"{result.summary_path}\nExamples: {result.example_count:,}\n"
            f"Reused folds: {result.reused_folds}"
        )


def main():
    app()
