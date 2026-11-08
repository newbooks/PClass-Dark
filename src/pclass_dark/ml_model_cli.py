"""Thin CLI adapters for full-cohort additive fitting and query prediction."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from .ml_progress import terminal_progress

train_app = typer.Typer(add_completion=False)
predict_app = typer.Typer(add_completion=False)


@train_app.command(context_settings={"help_option_names": ["-h", "--help"]})
def train_command(
    experiment: Path = Path("explore-ml"),
    output_dir: Path = Path("ml-model"),
    config: str = "hard",
    epochs: Annotated[int | None, typer.Option(min=1)] = None,
    device: str = "auto",
    threads: Annotated[int, typer.Option(min=1)] = 8,
    memory_budget_gib: float = 8.0,
    resume: bool = False,
):
    """Fit one portable classifier on all reference examples of a completed CV run."""
    from .ml_model import fit_final_model

    if config not in ("hard", "siac"):
        raise typer.BadParameter("--config must be hard or siac")
    try:
        with terminal_progress():
            result = fit_final_model(
                experiment,
                output_dir,
                configuration=config,
                epochs=epochs,
                device=device,
                threads=threads,
                memory_budget_gib=memory_budget_gib,
                resume=resume,
            )
    except KeyboardInterrupt:
        typer.echo(
            "Final fitting interrupted; use --resume to restart the owned fit.",
            err=True,
        )
        raise typer.Exit(130) from None
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"{result.model_dir}\nModel: {result.model_id}\n"
        f"Configuration: {result.configuration}; examples: {result.example_count:,}; "
        f"epochs: {result.epochs}; device: {result.device}"
    )


@predict_app.command(context_settings={"help_option_names": ["-h", "--help"]})
def predict_command(
    model_dir: Path = Path("ml-model"),
    features: Path | None = None,
    seq: str | None = None,
    fasta: Path | None = None,
    query_id: str = "query",
    device: str = "auto",
    offline: bool = True,
    threads: Annotated[int, typer.Option(min=1)] = 8,
    format: str = "markdown",
    output: Path | None = None,
    stdout: Annotated[
        bool, typer.Option(help="Print instead of saving a report.")
    ] = False,
):
    """Predict top clusters from a raw sequence/FASTA or query_id + 18-feature TSV."""
    from threadpoolctl import threadpool_limits

    from .ml_model import (
        load_final_model,
        predict_features,
        predict_sequences,
        read_query_features,
    )
    from .reference_data import read_fasta

    if sum(value is not None for value in (features, seq, fasta)) != 1:
        raise typer.BadParameter("Provide exactly one of --features, --seq or --fasta")
    if format not in ("tsv", "jsonl", "markdown"):
        raise typer.BadParameter("--format must be tsv, jsonl or markdown")
    if seq is None and query_id != "query":
        raise typer.BadParameter("--query-id is only used with --seq")
    if stdout and output is not None:
        raise typer.BadParameter("--stdout cannot be combined with --output")
    if format == "markdown" and output is None and not stdout:
        source = fasta if fasta is not None else features
        if source is not None:
            output = source.with_name(f"{source.stem}-interpretation.md")
        else:
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id).strip("._-")
            output = Path(f"{name or 'query'}-interpretation.md")
    temporary = None
    try:
        if output is not None:
            raw = output.absolute()
            if raw.exists() or raw.is_symlink():
                raise ValueError("Prediction output already exists")
            if model_dir.resolve() in raw.resolve().parents:
                raise ValueError("Prediction output must be outside the model bundle")
            output.parent.mkdir(parents=True, exist_ok=True)
        with terminal_progress(), threadpool_limits(limits=threads):
            model = load_final_model(model_dir)
            descriptions = None
            if format == "markdown":
                from .ml_interpretation import (
                    render_interpretation,
                    render_top_predictions,
                    resolve_descriptions,
                )

                descriptions = resolve_descriptions(model)
            if features is not None:
                ids, X = read_query_features(features)
                predictions = predict_features(model, X, ids)
            else:
                queries = (
                    ((query_id, seq),)
                    if seq is not None
                    else read_fasta(
                        fasta,
                        max_sequence_length=100000,
                        max_header_bytes=4096,
                        max_bytes=256 * 1024**2,
                        max_records=100000,
                        check_stable=True,
                    )
                )
                predictions = predict_sequences(
                    model, queries, device=device, offline=offline, threads=threads
                )
            # Even stdout is staged: a late input error must not emit partial success.
            with (
                tempfile.TemporaryFile(
                    mode="w+", encoding="utf-8", newline=""
                ) as stream,
                tempfile.TemporaryFile(mode="w+", encoding="utf-8") as summary,
            ):
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                if format == "tsv":
                    writer.writerow(
                        ("query_id", "rank", "cluster_id", "logit", "softmax_score")
                    )
                count = 0
                for prediction in predictions:
                    count += 1
                    if format == "markdown":
                        if output is not None:
                            summary.write(f"Query: {json.dumps(prediction.query_id)}\n")
                            summary.write(
                                render_top_predictions(
                                    prediction, descriptions=descriptions
                                )
                            )
                            summary.write("\n")
                        if count > 1:
                            stream.write("\n---\n\n")
                        stream.write(
                            render_interpretation(prediction, descriptions=descriptions)
                        )
                    elif format == "jsonl":
                        stream.write(
                            json.dumps(asdict(prediction), allow_nan=False) + "\n"
                        )
                    else:
                        for rank in prediction.ranked:
                            writer.writerow(
                                (
                                    prediction.query_id,
                                    rank.rank,
                                    rank.cluster_id,
                                    format_float(rank.logit),
                                    format_float(rank.softmax_score),
                                )
                            )
                if not count:
                    raise ValueError("No query records")
                stream.flush()
                stream.seek(0)
                if output is None:
                    import sys

                    shutil.copyfileobj(stream, sys.stdout)
                else:
                    fd, name = tempfile.mkstemp(
                        prefix=".ml-predictions-", dir=output.parent
                    )
                    temporary = Path(name)
                    with os.fdopen(fd, "w", encoding="utf-8", newline="") as target:
                        shutil.copyfileobj(stream, target)
                        target.flush()
                        os.fsync(target.fileno())
                    # Link publication refuses a concurrent creator of the target.
                    os.link(temporary, output)
                    temporary.unlink()
                    temporary = None
                    if format == "markdown":
                        import sys

                        summary.seek(0)
                        shutil.copyfileobj(summary, sys.stdout)
                    typer.echo(
                        f"Saved interpretation: {output}"
                        if format == "markdown"
                        else f"Saved predictions: {output}",
                        err=True,
                    )
    except KeyboardInterrupt:
        typer.echo("Prediction interrupted; no complete output published.", err=True)
        raise typer.Exit(130) from None
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def format_float(value):
    return f"{value:.8g}"


def train_main():
    train_app()


def predict_main():
    predict_app()
