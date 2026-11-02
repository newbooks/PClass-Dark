"""Thin standalone interface to existing classifiers and profunc scoring."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from click.core import ParameterSource

from .external_evaluation_types import (
    DarkConfig,
    EvaluationPreflight,
    ExecutionConfig,
    PClassConfig,
    ScoringConfig,
)

app = typer.Typer(add_completion=False)


@app.command(context_settings={"help_option_names": ["-h", "--help"]})
def evaluate(
    ctx: typer.Context,
    fasta: Annotated[
        Path, typer.Argument(help="FASTA with go_mf_terms truth annotations.")
    ],
    output: Annotated[Path | None, typer.Option()] = None,
    truth_tsv: Annotated[Path | None, typer.Option()] = None,
    pclass_batch_size: Annotated[int, typer.Option(min=1, max=1024)] = 512,
    dark_chunk_size: Annotated[int, typer.Option(min=1, max=1024)] = 512,
    pclass_server_url: Annotated[
        str | None, typer.Option(help="Optional override; otherwise pclass defaults.")
    ] = None,
    pclass_artifact_dir: Annotated[Path | None, typer.Option()] = None,
    dark_library_dir: Annotated[Path | None, typer.Option()] = None,
    device: Annotated[
        str | None, typer.Option(help="Optional Dark device override.")
    ] = None,
    go_graph: Annotated[
        Path | None, typer.Option(help="Trusted local profunc graph pickle.")
    ] = None,
    cluster_go_map: Annotated[Path | None, typer.Option()] = None,
    score_workers: Annotated[int, typer.Option(min=1, max=8)] = 8,
    process_timeout: Annotated[float, typer.Option(min=0.001)] = 3600,
    max_raw_chunk_mib: Annotated[int, typer.Option(min=1)] = 128,
    run_tag: Annotated[
        str,
        typer.Option(help="Change after updating external classifier models/indexes."),
    ] = "default",
    score_only: bool = False,
    preflight_only: bool = False,
):
    """Collect existing pclass/pclass-dark JSON and compare top-match GO sets."""
    from .external_evaluation import evaluate_external

    if score_only:
        inference_options = {
            "pclass_batch_size",
            "dark_chunk_size",
            "pclass_server_url",
            "pclass_artifact_dir",
            "dark_library_dir",
            "device",
            "process_timeout",
            "max_raw_chunk_mib",
            "run_tag",
        }
        if any(
            ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE
            for name in inference_options
        ):
            raise typer.BadParameter("Score-only does not accept inference options")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = evaluate_external(
            fasta,
            work_dir=Path.cwd(),
            output=output,
            truth_tsv=truth_tsv,
            pclass=None
            if score_only
            else PClassConfig(
                pclass_batch_size, pclass_server_url, pclass_artifact_dir
            ),
            dark=None
            if score_only
            else DarkConfig(dark_chunk_size, dark_library_dir, device),
            scoring=ScoringConfig(go_graph, cluster_go_map, score_workers),
            execution=None
            if score_only
            else ExecutionConfig(process_timeout, max_raw_chunk_mib, run_tag),
            score_only=score_only,
            preflight_only=preflight_only,
        )
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    if isinstance(result, EvaluationPreflight):
        typer.echo(json.dumps(asdict(result), indent=2))
    else:
        typer.echo(
            f"{result.tsv_path}\nQueries: {result.query_count:,}\n"
            f"Score: {result.score_id}"
        )


def main():
    app()
