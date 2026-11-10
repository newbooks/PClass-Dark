"""Full-reference fitting and immutable integrated library generations."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from .additive_reference import verify_reference
from .artifacts import (
    IntegrityError,
    file_record,
    identity,
    read_json,
    safe_file,
    schema_checked,
    sha256,
    staging,
    write_json,
)


@schema_checked
def verify_integrated(root, selection):
    """Verify the selected envelope, including every nested model/head artifact."""
    path = safe_file(root, selection["integrated"])
    if any(p.is_symlink() for p in (path, *path.parents)) or any(
        p.is_symlink() for p in path.rglob("*")
    ):
        raise IntegrityError("Integrated library symlink rejected")
    m = read_json(path / "manifest.json")
    if (
        m.get("schema") != "pclass-dark.integrated-library.v1"
        or m.get("status") != "complete"
        or identity(m["identity_inputs"]) != m["id"]
        or sha256(path / "manifest.json") != selection["integrated_manifest_sha256"]
    ):
        raise IntegrityError("Invalid integrated library identity")
    for name, record in m["files"].items():
        if file_record(safe_file(path, name)) != record:
            raise IntegrityError(f"Changed integrated artifact: {name}")
    if {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()} != set(
        m["files"]
    ) | {"manifest.json"}:
        raise IntegrityError("Unknown integrated artifact")
    for key in (
        "library_id",
        "manifest_sha256",
        "relative_id",
        "relative_manifest_sha256",
        "preparation_id",
        "embedding_id",
    ):
        if selection[key] != m["identity_inputs"]["signature_selection"][key]:
            raise IntegrityError("Integrated signature selection mismatch")
    return path, m


def build_additive_library(data_root, output_root, **options):
    """Serialize integrated builds while preserving the selected generation."""
    from filelock import FileLock

    root = Path(output_root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Symlink library output rejected")
    data = Path(data_root).resolve()
    if (
        root.resolve() == data
        or root.resolve() in data.parents
        or data in root.resolve().parents
    ):
        raise ValueError("Library output must be separate from setup data")
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".additive-build.lock"), timeout=0):
        return _build_additive_library(data, root.resolve(), **options)


def _build_additive_library(
    data_root,
    output_root,
    *,
    workers=8,
    memory_mib=1024,
    epsilon=1e-8,
    sigma_floor=1e-8,
    cluster_ids=None,
    force=False,
    cross_validate=False,
    device="auto",
    memory_budget_gib=8.0,
    offline=False,
):
    """Build signatures and an additive envelope, selecting only after all succeed."""
    import os

    from .feature_extraction import _contract, extract_features
    from .library_build import _guards, build
    from .ml_data import load_cohort, load_similarity
    from .ml_model import _admit, _fit_code, _versions, fit_reference_model
    from .siac import resolve_device
    from .structural_models import prepare_structural_models

    root, data = Path(output_root), Path(data_root)
    selected = read_json(data / "selection.json")
    if not (data / "additive-selection.json").is_file():
        raise ValueError(
            "Additive reference unavailable; "
            "run pclass-dark setup-data --prepare-only first"
        )
    reference_selection = read_json(data / "additive-selection.json")
    reference = safe_file(data, reference_selection["reference"])
    rm = verify_reference(reference)
    prepared = safe_file(data, selected["prepared"])
    if (
        reference_selection["preparation_id"] != selected["preparation_id"]
        or rm["identity_inputs"]["prepared_manifest_sha256"]
        != sha256(prepared / "manifest.json")
        or sha256(reference / "manifest.json")
        != reference_selection["reference_manifest_sha256"]
    ):
        raise IntegrityError("Additive/prepared selection mismatch; run setup-data")
    recipe = read_json(reference / "recipe.json")
    # Fail on missing fitting extras before an expensive production build.
    try:
        import sklearn  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Additive building needs pip install 'pclass-dark[ml,embeddings]'"
        ) from exc

    if force:
        raise ValueError(
            "Integrated --force is unsupported; "
            "use a fresh library root for a forced rebuild"
        )
    if cross_validate:
        raise ValueError(
            "Run signature research separately with --no-additive --cross-validate"
        )
    before = (
        read_json(root / "selection.json")
        if (root / "selection.json").exists()
        else None
    )
    result = build(
        data,
        root,
        workers=workers,
        memory_mib=memory_mib,
        epsilon=epsilon,
        sigma_floor=sigma_floor,
        cluster_ids=cluster_ids,
        force=force,
        cross_validate=False,
        _defer_selection=True,
    )
    heads = prepare_structural_models(root)
    feature_key = identity(
        dict(
            contract=_contract(
                read_json(prepared / "manifest.json"),
                selected["prepared_manifest_sha256"],
                heads,
                8,
            ),
            requested_device=device,
        )
    )
    cache = root / "additive-features" / feature_key
    cache.mkdir(parents=True, exist_ok=True)
    features = extract_features(
        prepared,
        cache / "features.tsv",
        heads.path,
        device=device,
        offline=offline,
        workers=min(8, workers),
        resume=True,
        _library_root=root,
    )
    feature_inventory = {
        p: file_record(p)
        for p in (features.tsv_path, features.sequences_path, features.manifest_path)
    }
    cohort = load_cohort(
        features.tsv_path,
        sequence_index=features.sequences_path,
        conflict_policy="exclude",
    )
    from .ml_data import ALIAS_HEADER, EXCLUSION_HEADER
    from .ml_reports import write_tsv

    write_tsv(cache / "cohort-aliases.tsv", ALIAS_HEADER, cohort.aliases)
    write_tsv(cache / "cohort-exclusions.tsv", EXCLUSION_HEADER, cohort.exclusions)
    write_json(
        cache / "cohort.json",
        dict(
            schema="pclass-dark.additive-cohort.v1",
            reference_id=rm["id"],
            prefilter_counts=cohort.prefilter_counts,
            files={
                name: file_record(cache / name)
                for name in ("cohort-aliases.tsv", "cohort-exclusions.tsv")
            },
        ),
    )
    similarity = load_similarity(
        reference / "cluster_dmatrix.tsv", labels=cohort.labels
    )
    fm = read_json(features.manifest_path)
    fc = dict(
        schema="pclass-dark.ml-feature-contract.v1",
        feature_names=list(cohort.feature_names),
        precision=".3f",
        extraction=fm["identity_inputs"]["contract_inputs"],
    )
    available = max(1, getattr(os, "process_cpu_count", os.cpu_count)() or 1)
    threads = min(workers, available)
    actual, _ = resolve_device(device, threads=threads)
    actual, _ = _admit(
        cohort.X, cohort.y, similarity.S, recipe, actual, device, memory_budget_gib
    )
    model_cache = (
        root
        / "additive-models"
        / identity(
            dict(
                features=file_record(features.manifest_path),
                recipe=recipe,
                reference=rm["id"],
                code=_fit_code(),
                versions=_versions(),
                threads=threads,
                actual_device=actual,
            )
        )
    )
    fit = fit_reference_model(
        cohort.X,
        cohort.y,
        cohort.labels,
        similarity,
        fc,
        recipe,
        model_cache,
        provenance=dict(
            reference_id=rm["id"], preparation_id=selected["preparation_id"]
        ),
        device=device,
        threads=workers,
        memory_budget_gib=memory_budget_gib,
        resume=True,
    )
    sig = dict(
        schema="pclass-dark.library-selection.v3",
        library_id=result.library_id,
        library=str(result.library_dir.relative_to(root)),
        manifest_sha256=sha256(result.library_dir / "manifest.json"),
        relative_id=result.relative_id,
        relative=str(result.relative_dir.relative_to(root)),
        relative_manifest_sha256=sha256(result.relative_dir / "manifest.json"),
        preparation_id=selected["preparation_id"],
        embedding_id=selected["embedding_id"],
    )
    inputs = dict(
        signature_selection=sig,
        reference_id=rm["id"],
        recipe_id=identity(recipe),
        model_id=fit.model_id,
        heads_id=heads.model_id,
    )
    eid = identity(inputs)
    target = root / "integrated" / eid
    with _guards(data, root, False):
        current = (
            read_json(root / "selection.json")
            if (root / "selection.json").exists()
            else None
        )
        if (
            verify_reference(reference) != rm
            or read_json(data / "additive-selection.json") != reference_selection
            or any(file_record(p) != record for p, record in feature_inventory.items())
        ):
            raise IntegrityError("Training source changed during additive build")
        from .library import load_library
        from .relative_scores import load_relative_scores

        load_relative_scores(
            result.relative_dir, library=load_library(result.library_dir)
        )
        if current != before or read_json(data / "selection.json") != selected:
            raise IntegrityError(
                "Library/setup selection changed during additive build"
            )
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with staging(target.parent) as stage:
                shutil.copytree(fit.model_dir, stage / "additive/model")
                shutil.copytree(heads.path, stage / "structural")
                shutil.copyfile(
                    reference / "descriptions.json",
                    stage / "additive/descriptions.json",
                )
                shutil.copyfile(
                    reference / "manifest.json", stage / "additive/reference.json"
                )
                write_json(
                    stage / "additive/training.json",
                    dict(
                        recipe=recipe,
                        example_count=len(cohort.y),
                        label_count=len(cohort.labels),
                        prefilter_counts=cohort.prefilter_counts,
                        feature_manifest=file_record(features.manifest_path),
                        alias_count=len(cohort.aliases),
                        exclusion_count=len(cohort.exclusions),
                    ),
                )
                files = {
                    str(p.relative_to(stage)): file_record(p)
                    for p in stage.rglob("*")
                    if p.is_file()
                }
                write_json(
                    stage / "manifest.json",
                    dict(
                        schema="pclass-dark.integrated-library.v1",
                        id=eid,
                        status="complete",
                        identity_inputs=inputs,
                        files=files,
                    ),
                )
                stage.rename(target)
        sig.update(
            additive_status="complete",
            additive_model_id=fit.model_id,
            integrated=str(target.relative_to(root)),
            integrated_id=eid,
            integrated_manifest_sha256=sha256(target / "manifest.json"),
        )
        verify_integrated(root, sig)
        write_json(root / "selection.json", sig)
    return replace(
        result,
        structural_model_id=heads.model_id,
        structural_model_dir=target / "structural",
        additive_model_id=fit.model_id,
        additive_model_dir=target / "additive/model",
    )
