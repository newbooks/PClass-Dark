"""Command-line interface; scientific functionality belongs in library modules."""

from pathlib import Path
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
    reference_fasta: str | None = typer.Option(
        None, help="Source FASTA matching profunc's dataset manifest."
    ),
    min_clusters: int = typer.Option(
        10, min=1, help="Minimum unique full-sequence members per cluster."
    ),
    seed: int = typer.Option(
        42, min=0, help="Seed for per-cluster C folds and fixed B samples."
    ),
    workers: int = typer.Option(8, min=1, help="CPU workers for cluster preparation."),
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
    additive: bool = typer.Option(True, help="Prepare frozen additive reference."),
    cluster_similarity: Path | None = None,
    cluster_similarity_manifest: Path | None = None,
    additive_recipe: Path | None = None,
) -> None:
    """Prepare references, C folds, fixed B, and resumable PLM embeddings."""
    import logging

    from pclass_dark.setup_data import prepare_data

    if not additive and any(
        v is not None
        for v in (cluster_similarity, cluster_similarity_manifest, additive_recipe)
    ):
        raise typer.BadParameter("Additive import options require --additive")
    if (cluster_similarity is None) != (cluster_similarity_manifest is None):
        raise typer.BadParameter(
            "Provide both --cluster-similarity and --cluster-similarity-manifest"
        )

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        result = prepare_data(
            output_dir,
            reference_fasta=reference_fasta,
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
            additive=additive,
            cluster_similarity=cluster_similarity,
            cluster_similarity_manifest=cluster_similarity_manifest,
            additive_recipe=additive_recipe,
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
    if getattr(result, "additive_reference_dir", None) is not None:
        typer.echo(f"Additive reference: {result.additive_reference_dir}")


@app.command("build-library", context_settings={"help_option_names": ["-h", "--help"]})
def build_library_command(
    workers: int = typer.Option(8, min=1, help="Bounded CPU target workers."),
    cluster: Annotated[
        list[str] | None, typer.Option(help="Exact target cluster ID; repeatable.")
    ] = None,
    cross_validate: bool = typer.Option(
        False, help="Also run five research folds; use --cluster to limit cost."
    ),
    force: bool = typer.Option(False, help="Archive and recompute this library."),
    additive: bool = typer.Option(
        True, help="Fit and publish supplemental additive evidence."
    ),
    additive_device: str = "auto",
    additive_memory_gib: float = 8.0,
    offline: bool = False,
    clean: bool = typer.Option(
        False,
        help=(
            "Validate the existing library without building, then delete obsolete "
            "preparations and exclusive results; "
            "includes superseded rollback archives; preserves Git LFS objects."
        ),
    ),
) -> None:
    """Prepare RSA/Q3 heads, all-C signatures and C/B distributions."""
    import logging

    from pclass_dark.library import build_library

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    data_dir = output_dir = None
    memory_mib = 1024
    epsilon = sigma_floor = 1e-8
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
            additive=additive,
            additive_device=additive_device,
            additive_memory_gib=additive_memory_gib,
            offline=offline,
            **({"clean": True} if clean else {}),
        )
    except KeyboardInterrupt:
        typer.echo(
            "Interrupted; validated library checkpoints can be resumed.", err=True
        )
        raise typer.Exit(130) from None
    except (ValueError, RuntimeError, OSError) as exc:
        if getattr(exc, "production_result", None) is not None:
            typer.echo(
                f"Production remains valid: {exc.production_result.library_dir}",
                err=True,
            )
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if getattr(result, "cleanup", None) is not None:
        typer.echo(
            f"Cleanup: removed {len(result.cleanup.removed)} generations, "
            f"{result.cleanup.logical_bytes_removed} logical bytes; "
            "superseded rollback archives are eligible; Git LFS objects retained."
        )
    typer.echo(f"Library: {result.library_dir}")
    if getattr(result, "structural_model_dir", None) is not None:
        typer.echo(f"Structural models: {result.structural_model_dir}")
    if getattr(result, "additive_model_dir", None) is not None:
        typer.echo(f"Additive model: {result.additive_model_dir}")
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


@app.command("infer", context_settings={"help_option_names": ["-h", "--help"]})
def infer_command(
    seq: str | None = typer.Option(None, help="One uppercase protein sequence."),
    fasta: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Uncompressed FASTA file; multiple queries allowed.",
        ),
    ] = None,
    library_dir: str | None = typer.Option(
        None, help="Selected production/C/B library root."
    ),
    device: str = typer.Option(
        "auto", help="auto (GPU first, CPU fallback), cpu, cuda, or cuda:<index>."
    ),
    workers: int = typer.Option(8, min=1, help="CPU thread budget; one model process."),
    batch_size: int = typer.Option(8, min=1, max=64, help="Windows per model forward."),
    score_batch_size: int = typer.Option(
        4096, min=1, max=65536, help="Windows per CPU scoring block."
    ),
    memory_mib: int = typer.Option(
        1024,
        min=128,
        help="Array/SQLite workspace MiB; excludes model/runtime/Python records.",
    ),
    offline: bool = typer.Option(
        False, help="Require cached pinned model assets; no network."
    ),
    format: str = typer.Option("text", help="text, json or markdown on stdout."),
    additive: str = typer.Option("auto", help="auto, off or required."),
    report_prefix: Path | None = None,
    reports: bool = typer.Option(True, help="Save combined JSON and Markdown reports."),
) -> None:
    """Classify accepted clusters with supplemental additive ranking and reports.

    Writes JSON and Markdown by default; --no-reports keeps stdout only.
    Caps: 1 GiB input, 100,000 queries,
    1,000,000 residues/record, 4 KiB FASTA headers, 2 GiB temporary storage.
    """
    import logging
    import re
    import shutil
    import sys

    if (seq is None) == (fasta is None):
        raise typer.BadParameter("Provide exactly one of --seq or --fasta")
    if format not in {"text", "json", "markdown"}:
        raise typer.BadParameter("Use text, json or markdown", param_hint="--format")
    if additive not in {"auto", "off", "required"}:
        raise typer.BadParameter("Use auto, off or required", param_hint="--additive")
    if not reports and report_prefix is not None:
        raise typer.BadParameter("--report-prefix cannot be combined with --no-reports")
    if not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", device):
        raise typer.BadParameter(
            "Use auto, cpu, cuda, or cuda:<index>", param_hint="--device"
        )
    if fasta is not None and (str(fasta) == "-" or not fasta.is_file()):
        raise typer.BadParameter(
            "Use a regular FASTA file, not stdin", param_hint="--fasta"
        )
    from .inference import QuerySequence, read_query_fasta
    from .integrated_inference import infer_integrated_queries
    from .integrated_render import write_integrated_reports
    from .integrated_reports import report_outputs
    from .paths import default_library_input_dir, default_setup_input_dir

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    queries = (
        (QuerySequence("query", seq),) if seq is not None else read_query_fasta(fasta)
    )
    prefix = None
    if reports:
        prefix = report_prefix or (fasta.with_suffix("") if fasta else Path("query"))
        resolved = prefix.absolute().resolve()
        for owned in (
            Path(library_dir or default_library_input_dir()).resolve(),
            default_setup_input_dir().resolve(),
        ):
            if resolved == owned or owned in resolved.parents:
                raise typer.BadParameter(
                    "Reports must be outside owned library/setup roots"
                )
    staged_stdout = None
    try:
        with report_outputs(prefix) as output:
            with infer_integrated_queries(
                queries,
                library_dir=library_dir,
                additive=additive,
                device=device,
                workers=workers,
                batch_size=batch_size,
                score_batch_size=score_batch_size,
                memory_mib=memory_mib,
                offline=offline,
            ) as run:
                output.query_count = run.query_count
                write_integrated_reports(
                    run, output.json, output.markdown, terminal_stream=output.text
                )
            staged_stdout = output.prepare_stdout(format)
        shutil.copyfileobj(staged_stdout, sys.stdout, length=65536)
        sys.stdout.flush()
        if prefix is not None:
            typer.echo(f"Saved classification: {prefix}-classification.json", err=True)
            typer.echo(f"Saved interpretation: {prefix}-interpretation.md", err=True)
    except BrokenPipeError:
        raise
    except KeyboardInterrupt:
        typer.echo("Inference interrupted; no completed report published.", err=True)
        raise typer.Exit(130) from None
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    finally:
        if staged_stdout is not None:
            staged_stdout.close()
