"""Thin script/installed CLI adapters for native Swiss-Prot evaluation."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from click.core import ParameterSource

from .external_evaluation_types import EvaluationPreflight, ScoringConfig
from .swiss_evaluation_types import (
    SwissDarkConfig,
    SwissExecutionConfig,
    SwissPClassConfig,
)

app = typer.Typer(add_completion=False)


@app.command(context_settings={"help_option_names": ["-h", "--help"]})
def evaluate_swiss_command(
    ctx: typer.Context,
    fasta: Annotated[
        Path, typer.Argument(help="Swiss-Prot FASTA; truth comes from profunc.")
    ],
    output: Path | None = None,
    pclass_batch_size: Annotated[int, typer.Option(min=1, max=1024)] = 512,
    pclass_server_url: str | None = None,
    pclass_timeout: float = 60,
    dark_chunk_size: Annotated[int, typer.Option(min=1, max=1024)] = 512,
    dark_library_dir: Path | None = None,
    device: str = "auto",
    dark_batch_size: Annotated[int, typer.Option(min=1, max=64)] = 8,
    dark_score_batch_size: Annotated[int, typer.Option(min=1, max=65536)] = 4096,
    dark_memory_mib: Annotated[int, typer.Option(min=128)] = 1024,
    dark_workers: Annotated[int, typer.Option(min=1, max=8)] = 8,
    dark_embedding_cache: Path | None = None,
    dark_prepared_dir: Path | None = None,
    offline: bool = False,
    score_workers: Annotated[int, typer.Option(min=1, max=8)] = 8,
    go_graph: Path | None = None,
    cluster_go_map: Path | None = None,
    process_timeout: float = 3600,
    max_raw_chunk_mib: Annotated[int, typer.Option(min=1)] = 128,
    run_tag: str = "default",
    score_only: bool = False,
    preflight_only: bool = False,
):
    """Compare native PClass neighbors and resident Dark inference with Swiss truth."""
    from .swiss_evaluation import evaluate_swiss

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    inference_names = {
        "pclass_batch_size",
        "pclass_server_url",
        "pclass_timeout",
        "dark_chunk_size",
        "dark_library_dir",
        "device",
        "dark_batch_size",
        "dark_score_batch_size",
        "dark_memory_mib",
        "dark_workers",
        "dark_embedding_cache",
        "dark_prepared_dir",
        "offline",
        "process_timeout",
        "max_raw_chunk_mib",
        "run_tag",
    }
    if score_only and any(
        ctx.get_parameter_source(n) == ParameterSource.COMMANDLINE
        for n in inference_names
    ):
        raise typer.BadParameter("Score-only rejects inference options")
    try:
        pc = (
            None
            if score_only
            else SwissPClassConfig(
                **{
                    "batch_size": pclass_batch_size,
                    "timeout_seconds": pclass_timeout,
                    **(
                        {"server_url": pclass_server_url}
                        if pclass_server_url is not None
                        else {}
                    ),
                }
            )
        )
        dc = (
            None
            if score_only
            else SwissDarkConfig(
                dark_chunk_size,
                dark_library_dir,
                device,
                dark_batch_size,
                dark_score_batch_size,
                dark_memory_mib,
                dark_workers,
                dark_embedding_cache,
                dark_prepared_dir,
                offline,
            )
        )
        result = evaluate_swiss(
            fasta,
            work_dir=Path.cwd(),
            output=output,
            pclass=pc,
            dark=dc,
            scoring=ScoringConfig(go_graph, cluster_go_map, score_workers),
            execution=None
            if score_only
            else SwissExecutionConfig(process_timeout, max_raw_chunk_mib, run_tag),
            score_only=score_only,
            preflight_only=preflight_only,
        )
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
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
