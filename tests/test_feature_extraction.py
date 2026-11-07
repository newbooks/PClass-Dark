"""Experimental extractor integration, receipts, publication and script checks."""

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pclass_dark import feature_extraction as extraction
from pclass_dark import feature_reports as reports
from pclass_dark import feature_store as store
from pclass_dark.artifacts import IntegrityError, read_json, write_json
from pclass_dark.feature_extraction_cli import app
from pclass_dark.setup_data import prepare_data
from pclass_dark.structural_models import BUNDLE_ID
from pclass_dark.structure import StructuralPrediction


@pytest.fixture
def cohort(reference_factory, tmp_path):
    reference = reference_factory(
        [
            ("A0", "excluded", "DEKRH"),
            ("B1", "K1", "DEKRH"),
            ("B2", "K1", "ACDEKR"),
            ("C1", "K2", "DEKRH"),
            ("C2", "K2", "KKDA"),
            ("U1", "K1", "XX"),
        ]
    )
    prepared = prepare_data(
        tmp_path / "setup",
        reference_dir=reference,
        prepare_only=True,
        min_clusters=2,
        workers=1,
    )
    output = tmp_path / "experiment"
    output.mkdir()
    return prepared.prepared_dir, output / "features.tsv"


@pytest.fixture
def fake_predictor(monkeypatch):
    class Fake:
        calls = []
        prepares = 0
        fail_after = None

        def __init__(self, model_dir=None, **options):
            self.options = options

        def prepare(self):
            from pclass_dark.embeddings import ASSET_HASHES
            from pclass_dark.setup_data import MODEL, REVISION

            type(self).prepares += 1
            return {
                "device": "cpu",
                "embedding_contract": {
                    "model": MODEL,
                    "revision": REVISION,
                    "assets": dict(ASSET_HASHES),
                    "layer": 30,
                    "width": 640,
                    "inference_dtype": "float32",
                },
                "numerical_threads": self.options["cpu_threads"],
                "window_batch_size": self.options["window_batch_size"],
            }

        def predict(self, seq):
            if self.fail_after is not None and len(self.calls) >= self.fail_after:
                raise RuntimeError("injected prediction failure")
            self.calls.append(seq)
            return StructuralPrediction(
                seq,
                np.array([0.1] * len(seq), np.float32),
                np.array(["C"] * len(seq)),
                BUNDLE_ID,
                "cpu",
            )

    monkeypatch.setattr(extraction, "StructuralPredictor", Fake)
    return Fake


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_membership_table_precision_qc_and_complete_resume(cohort, fake_predictor):
    prepared, output = cohort
    result = extraction.extract_features(prepared, output, workers=1, offline=True)
    assert (
        result.sequence_count == 4
        and result.membership_count == 5
        and result.cluster_count == 2
    )
    assert result.computed_sequences == 4 and len(fake_predictor.calls) == 3
    table = read(output)
    assert len(table[0]) == 21 and list(table[0]) == list(reports.HEADER)
    assert [(r["accession_id"], r["cluster_id"]) for r in table] == [
        ("A0", "K1"),
        ("A0", "K2"),
        ("B2", "K1"),
        ("C2", "K2"),
        ("U1", "K1"),
    ]
    assert table[0]["acid_fraction"] == "0.400"
    for row in table:
        for feature in reports.FEATURES:
            value = row[feature]
            if value != "NA":
                assert value == f"{float(value):.3f}"
    assert read_json(result.manifest_path)["inputs"]["policies"]["precision"] == ".3f"
    assert len([v for k, v in table[-1].items() if v == "NA"]) == 13
    assert read(result.qc_path)[-1]["device"] == "not_run"
    original = {
        n: output.with_name(n).read_bytes() for n in reports.Family(output).names
    }
    reused = extraction.extract_features(prepared, output, resume=True, offline=True)
    assert (
        reused.reused_output
        and reused.computed_sequences == 0
        and fake_predictor.prepares == 1
    )
    assert original == {n: output.with_name(n).read_bytes() for n in original}
    with pytest.raises(ValueError, match="Existing"):
        extraction.extract_features(prepared, output)


def test_interrupted_resume_and_corrupt_receipt(cohort, fake_predictor):
    prepared, output = cohort
    fake_predictor.fail_after = 1
    with pytest.raises(RuntimeError, match="A|B|C|failed"):
        extraction.extract_features(prepared, output, offline=True)
    assert not output.exists()
    family = reports.Family(output)
    db = store.connect(family.work / "state.sqlite")
    assert db.execute("SELECT COUNT(*) FROM completed").fetchone()[0] == 1
    db.close()
    fake_predictor.fail_after = None
    result = extraction.extract_features(prepared, output, resume=True, offline=True)
    assert result.reused_sequences == 1 and result.computed_sequences == 3
    assert len(fake_predictor.calls) == 3
    # Verify corruption is not concealed by complete-output reuse.
    db = store.connect(family.work / "state.sqlite")
    db.execute("UPDATE completed SET record_sha256='bad'")
    db.commit()
    db.close()
    # Incomplete receipt reuse must reject it.
    for n in family.names:
        family.path(n).unlink()
    with pytest.raises(IntegrityError, match="checksum"):
        extraction.extract_features(prepared, output, resume=True, offline=True)


def test_parallel_serial_agreement(cohort, fake_predictor):
    prepared, output = cohort
    serial = extraction.extract_features(prepared, output, workers=1)
    parallel = extraction.extract_features(
        prepared, output.with_name("parallel.tsv"), workers=2
    )
    assert output.read_bytes() == parallel.tsv_path.read_bytes()
    assert serial.qc_path.read_bytes() == parallel.qc_path.read_bytes()


def test_auto_discovery_cwd_script_help_and_options(
    cohort, fake_predictor, monkeypatch
):
    prepared, output = cohort
    monkeypatch.setattr(
        "pclass_dark.paths.default_setup_input_dir", lambda: prepared.parent.parent
    )
    monkeypatch.chdir(output.parent)
    result = CliRunner().invoke(app, ["--offline", "--workers", "1"])
    assert result.exit_code == 0, result.output
    assert output.exists() and "Memberships: 5" in result.stdout
    script = Path(__file__).resolve().parents[1] / "scripts/extract_features.py"
    help_run = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert help_run.returncode == 0 and "--output" not in help_run.stdout
    for args in (["--resume", "--force"], ["--workers", "0"]):
        assert CliRunner().invoke(app, args).exit_code == 2


def test_fingerprint_contract_and_snapshot_mismatch(cohort, fake_predictor):
    prepared, output = cohort
    fake_predictor.fail_after = 1
    with pytest.raises(RuntimeError):
        extraction.extract_features(prepared, output)
    fake_predictor.fail_after = None
    with pytest.raises(IntegrityError, match="contract"):
        extraction.extract_features(prepared, output, resume=True, window_batch_size=1)
    db = store.connect(reports.Family(output).work / "state.sqlite")
    db.execute("UPDATE aliases SET retained_cluster=0 WHERE accession='B1'")
    db.commit()
    db.close()
    with pytest.raises(IntegrityError):
        extraction.extract_features(prepared, output, resume=True)


def test_force_backup_and_copy_failure_restores_old_report(
    cohort, fake_predictor, monkeypatch
):
    prepared, output = cohort
    extraction.extract_features(prepared, output)
    original = output.read_bytes()
    replace = reports.os.replace
    fired = False

    def fail_one(source, target):
        nonlocal fired
        if (
            Path(source).parent.name == "stage"
            and Path(target) == output.with_name("features.qc.tsv")
            and not fired
        ):
            fired = True
            raise OSError("injected publication failure")
        return replace(source, target)

    monkeypatch.setattr(reports.os, "replace", fail_one)
    with pytest.raises(OSError, match="injected"):
        extraction.extract_features(prepared, output, force=True, window_batch_size=1)
    assert output.read_bytes() == original and reports.verify_family(
        reports.Family(output)
    )
    assert not (reports.Family(output).work / "publication.json").exists()
    result = extraction.extract_features(
        prepared, output, resume=True, window_batch_size=1
    )
    assert result.reused_sequences == 4 and result.computed_sequences == 0


def test_path_collision_and_lock(cohort, fake_predictor):
    from filelock import FileLock

    from pclass_dark.artifacts import SetupBusyError

    prepared, output = cohort
    with pytest.raises(ValueError):
        extraction.extract_features(prepared, prepared / "features.tsv")
    alias = output.parent / "alias.tsv"
    os.link(prepared / "memberships.tsv", alias)
    with pytest.raises(ValueError):
        extraction.extract_features(prepared, alias)
    with FileLock(str(reports.Family(output).lock)):
        with pytest.raises(SetupBusyError):
            extraction.extract_features(prepared, output)


def test_all_unknown_real_script_without_optional_models(reference_factory, tmp_path):
    reference = reference_factory(
        [
            ("X1", "K1", "XX"),
            ("X2", "K1", "XXX"),
            ("X3", "K2", "XXXX"),
            ("X4", "K2", "XXXXX"),
        ]
    )
    prepared = prepare_data(
        tmp_path / "setup", reference_dir=reference, prepare_only=True, min_clusters=2
    )
    out = tmp_path / "unknown"
    out.mkdir()
    script = Path(__file__).resolve().parents[1] / "scripts/extract_features.py"
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--prepared-dir",
            str(prepared.prepared_dir),
            "--offline",
            "--workers",
            "1",
        ],
        cwd=out,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert len(read(out / "features.tsv")) == 4
    assert (
        read_json(out / "features.tsv.manifest.json")["execution_fingerprint"]
        == "not_run"
    )


@pytest.mark.parametrize("name_index", [0, 1, 2, 3])
@pytest.mark.parametrize("existing", [False, True])
def test_publication_failure_after_each_swap(
    cohort, fake_predictor, monkeypatch, name_index, existing
):
    prepared, output = cohort
    family = reports.Family(output)
    if existing:
        extraction.extract_features(prepared, output)
    replace = reports.os.replace
    fired = False

    def fail_after(source, target):
        nonlocal fired
        value = replace(source, target)
        if (
            Path(source).parent.name == "stage"
            and Path(target).name == family.names[name_index]
            and Path(source).name == family.names[name_index]
            and not fired
        ):
            fired = True
            raise OSError("after swap")
        return value

    monkeypatch.setattr(reports.os, "replace", fail_after)
    with pytest.raises(OSError, match="after swap"):
        extraction.extract_features(
            prepared, output, force=existing, window_batch_size=1
        )
    public = reports.verify_family(family)
    assert public is not None if existing or name_index == 3 else public is None
    result = extraction.extract_features(
        prepared, output, resume=True, window_batch_size=1
    )
    assert result.sequence_count == 4 and reports.verify_family(family)


def test_crash_journal_recovery_and_third_party_mutation(
    cohort, fake_predictor, monkeypatch
):
    prepared, output = cohort
    family = reports.Family(output)
    extraction.extract_features(prepared, output)
    replace = reports.os.replace
    recover = reports.recover

    def fail_swap(source, target):
        value = replace(source, target)
        if Path(source).parent.name == "stage" and Path(target) == output:
            raise OSError("simulated death")
        return value

    def leave_journal(_):
        raise RuntimeError("recovery interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(reports.os, "replace", fail_swap)
        patch.setattr(reports, "recover", leave_journal)
        with pytest.raises(RuntimeError, match="recovery interrupted"):
            extraction.extract_features(
                prepared, output, force=True, window_batch_size=1
            )
    assert (family.work / "publication.json").exists()
    saved = output.read_bytes()
    output.write_text("user mutation")
    with pytest.raises(IntegrityError, match="Third-party"):
        recover(family)
    assert (
        output.read_text() == "user mutation"
        and (family.work / "publication.json").exists()
    )
    output.write_bytes(saved)
    resumed = extraction.extract_features(
        prepared, output, resume=True, window_batch_size=1
    )
    assert resumed.computed_sequences == 0 and resumed.reused_sequences == 4
    assert not (family.work / "publication.json").exists()


def test_resume_force_replacement_that_failed_during_indexing(
    cohort, fake_predictor, monkeypatch
):
    prepared, output = cohort
    extraction.extract_features(prepared, output)
    original = output.read_bytes()
    snapshot = store.snapshot
    with monkeypatch.context() as patch:
        patch.setattr(
            store,
            "snapshot",
            lambda *a: (_ for _ in ()).throw(OSError("indexing interrupted")),
        )
        with pytest.raises(OSError):
            extraction.extract_features(
                prepared, output, force=True, window_batch_size=1
            )
    assert output.read_bytes() == original
    assert snapshot is store.snapshot
    result = extraction.extract_features(
        prepared, output, resume=True, window_batch_size=1
    )
    assert result.computed_sequences == 4 and not result.reused_output


def test_indexing_resume_refuses_unowned_nested_file(
    cohort, fake_predictor, monkeypatch
):
    prepared, output = cohort
    family = reports.Family(output)
    with monkeypatch.context() as patch:
        patch.setattr(
            store,
            "snapshot",
            lambda *a: (_ for _ in ()).throw(OSError("indexing interrupted")),
        )
        with pytest.raises(OSError):
            extraction.extract_features(prepared, output)
    note = family.work / "models" / "user-note.txt"
    note.write_text("preserve")
    with pytest.raises(IntegrityError, match="Unrecognized nested"):
        extraction.extract_features(prepared, output, resume=True)
    assert note.read_text() == "preserve"


def test_incomplete_runtime_mismatch(cohort, fake_predictor, monkeypatch):
    prepared, output = cohort
    fake_predictor.fail_after = 1
    with pytest.raises(RuntimeError):
        extraction.extract_features(prepared, output)
    fake_predictor.fail_after = None
    prepare = fake_predictor.prepare

    def changed(self):
        result = prepare(self)
        result["numerical_threads"] += 1
        return result

    monkeypatch.setattr(fake_predictor, "prepare", changed)
    with pytest.raises(IntegrityError, match="execution fingerprint"):
        extraction.extract_features(prepared, output, resume=True)


def test_oversized_serial_path_and_worker_errors(cohort, fake_predictor, monkeypatch):
    prepared, output = cohort
    monkeypatch.setattr(extraction, "BUDGET", 1)
    result = extraction.extract_features(prepared, output, workers=2)
    manifest = read_json(result.manifest_path)
    assert len(manifest["execution"]["plan"]["oversized_sequences"]) == 4
    assert manifest["execution"]["plan"]["max_reserved_bytes"] > 1
    with monkeypatch.context() as patch:

        def failed(*args):
            raise ValueError("aggregation failed")

        patch.setattr(extraction, "_aggregate", failed)
        with pytest.raises(RuntimeError, match="aggregation failed"):
            extraction.extract_features(prepared, output.with_name("bad.tsv"))
    assert not output.with_name("bad.tsv").exists()


@pytest.mark.parametrize(
    "kind", ["alias", "cluster", "membership", "lowercase", "selection"]
)
def test_invalid_source_inputs_fail_before_inference(
    cohort, fake_predictor, kind, monkeypatch
):
    from pclass_dark.artifacts import file_record

    prepared, output = cohort
    if kind == "selection":
        monkeypatch.setattr(
            "pclass_dark.paths.default_setup_input_dir", lambda: prepared.parent.parent
        )
        path = prepared.parent.parent / "selection.json"
        sel = read_json(path)
        sel["prepared"] = "../outside"
        write_json(path, sel)
        with pytest.raises(IntegrityError):
            extraction.extract_features(output=output)
        assert not fake_predictor.calls
        return
    filename = {
        "alias": "aliases.tsv",
        "cluster": "clusters.tsv",
        "membership": "memberships.tsv",
        "lowercase": "sequences.fasta",
    }[kind]
    path = prepared / filename
    data = path.read_text()
    if kind == "alias":
        data = data.replace("A0\texcluded", "A0\tK1")
    if kind == "cluster":
        data = data.replace("excluded\t1\t1\tfalse", "excluded\t1\t1\ttrue")
    if kind == "membership":
        data += "\n".join(data.splitlines()[1:2]) + "\n"
    if kind == "lowercase":
        data = data.replace("DEKRH", "dekrh")
    path.write_text(data)
    marker = prepared / "manifest.json"
    manifest = read_json(marker)
    manifest["files"][filename] = file_record(path)
    write_json(marker, manifest)
    with pytest.raises(IntegrityError):
        extraction.extract_features(prepared, output)
    assert fake_predictor.prepares == 0 and not output.exists()


def probe_worker_no_torch(sequence, prediction):
    assert "torch" not in sys.modules
    from pclass_dark.structural_features import summarize_structural_features

    return summarize_structural_features(sequence, prediction)


def test_spawn_worker_has_no_torch(cohort, fake_predictor, monkeypatch):
    prepared, output = cohort
    monkeypatch.setattr(extraction, "_aggregate", probe_worker_no_torch)
    extraction.extract_features(prepared, output, workers=2)


def test_all_unknown_blocks_optional_imports_and_complete_resume(
    reference_factory, tmp_path
):
    reference = reference_factory(
        [
            ("X1", "K1", "XX"),
            ("X2", "K1", "XXX"),
            ("X3", "K2", "XXXX"),
            ("X4", "K2", "XXXXX"),
        ]
    )
    prepared = prepare_data(
        tmp_path / "setup",
        reference_dir=reference,
        prepare_only=True,
        min_clusters=2,
        workers=1,
    )
    output = tmp_path / "features.tsv"
    # Both fresh extraction and complete reuse must work with optional imports blocked.
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers','safetensors'}:
   raise AssertionError('optional import: '+fullname)
sys.meta_path.insert(0,Block())
import pclass_dark.structural_features
import pclass_dark.feature_extraction_cli
from pclass_dark.feature_extraction import extract_features
result = extract_features(sys.argv[1], sys.argv[2], offline=True,
                          workers=1, resume=bool(int(sys.argv[3])))
assert result.sequence_count == 4
assert result.reused_output == bool(int(sys.argv[3]))
assert 'torch' not in sys.modules
"""
    for resume in (0, 1):
        run = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(prepared.prepared_dir),
                str(output),
                str(resume),
            ],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stderr


def test_corrupt_completed_report_and_unknown_partial_family_are_not_overwritten(
    cohort, fake_predictor
):
    prepared, output = cohort
    extraction.extract_features(prepared, output)
    output.write_text("user edit")
    with pytest.raises(IntegrityError):
        extraction.extract_features(prepared, output, force=True)
    assert output.read_text() == "user edit"
    other = output.with_name("partial.tsv")
    other.write_text("keep")
    with pytest.raises(IntegrityError):
        extraction.extract_features(prepared, other, force=True)
    assert other.read_text() == "keep"


def test_checkpoint_free_completed_reuse_and_code_drift(
    cohort, fake_predictor, monkeypatch
):
    prepared, output = cohort
    extraction.extract_features(prepared, output)
    monkeypatch.setattr(
        extraction,
        "StructuralPredictor",
        lambda *a, **k: pytest.fail("initialized model on completed reuse"),
    )
    assert extraction.extract_features(prepared, output, resume=True).reused_output
    code = extraction._code_hashes
    calls = 0

    def drift():
        nonlocal calls
        calls += 1
        result = code()
        if calls > 1:
            result["feature_extraction"] = "changed"
        return result

    monkeypatch.setattr(extraction, "_code_hashes", drift)
    with pytest.raises(IntegrityError, match="changed during"):
        # Unknown-only cohort would avoid a model, so use the fake directly here.
        monkeypatch.setattr(extraction, "StructuralPredictor", fake_predictor)
        extraction.extract_features(prepared, output.with_name("drift.tsv"))
    assert not output.with_name("drift.tsv").exists()


def test_script_is_in_source_archive(tmp_path):
    import shutil
    import tarfile

    project = Path(__file__).resolve().parents[1]
    release = tmp_path / "release"
    release.mkdir()
    for name in (
        "pyproject.toml",
        "setup.py",
        "build_support.py",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
    ):
        shutil.copyfile(project / name, release / name)
    shutil.copytree(
        project / "src/pclass_dark",
        release / "src/pclass_dark",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (release / "scripts").mkdir()
    shutil.copyfile(
        project / "scripts/extract_features.py", release / "scripts/extract_features.py"
    )
    (release / "docs").mkdir()
    shutil.copyfile(
        project / "docs/feature-extraction.md", release / "docs/feature-extraction.md"
    )
    built = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--no-isolation"],
        cwd=release,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    with tarfile.open(next((release / "dist").glob("*.tar.gz"))) as archive:
        names = archive.getnames()
        assert any(n.endswith("/scripts/extract_features.py") for n in names)
        assert any(n.endswith("/docs/feature-extraction.md") for n in names)


def test_bad_runtime_contract_rejected_before_prediction(
    cohort, fake_predictor, monkeypatch
):
    prepared, output = cohort
    prepare = fake_predictor.prepare

    def wrong(self):
        metadata = prepare(self)
        metadata["embedding_contract"]["width"] = 32
        return metadata

    monkeypatch.setattr(fake_predictor, "prepare", wrong)
    with pytest.raises(IntegrityError, match="pinned representation"):
        extraction.extract_features(prepared, output)
    assert not fake_predictor.calls and not output.exists()


@pytest.mark.parametrize(
    "kwargs",
    [{"output": None}, {"output": 17}, {"prepared_dir": 17}, {"model_dir": False}],
)
def test_invalid_path_types_before_any_work(kwargs):
    with pytest.raises(ValueError, match="filesystem path"):
        extraction.extract_features(**kwargs)


def test_malformed_selector_schema_has_actionable_error(cohort, monkeypatch):
    prepared, output = cohort
    root = prepared.parent.parent
    write_json(root / "selection.json", [])
    monkeypatch.setattr("pclass_dark.paths.default_setup_input_dir", lambda: root)
    with pytest.raises(IntegrityError, match="Malformed artifact"):
        extraction.extract_features(output=output)
    assert not output.exists()


def test_manifest_bytes_and_snapshot_drift_are_detected(
    cohort, fake_predictor, monkeypatch
):
    prepared, output = cohort
    snapshot = store.snapshot
    with monkeypatch.context() as patch:

        def change_manifest(*args):
            snapshot(*args)
            path = prepared / "manifest.json"
            path.write_bytes(path.read_bytes() + b" ")

        patch.setattr(store, "snapshot", change_manifest)
        with pytest.raises(IntegrityError, match="source changed during indexing"):
            extraction.extract_features(prepared, output)
    assert not output.exists() and not fake_predictor.calls
    compute = extraction._compute

    def mutate_snapshot(db, *args):
        result = compute(db, *args)
        db.execute("UPDATE aliases SET representative='other' WHERE accession='A0'")
        db.commit()
        return result

    monkeypatch.setattr(extraction, "_compute", mutate_snapshot)
    with pytest.raises(IntegrityError, match="snapshot changed during"):
        extraction.extract_features(prepared, output.with_name("snapshot-drift.tsv"))
    assert not output.with_name("snapshot-drift.tsv").exists()


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, "0.000"),
        (-0.0, "0.000"),
        (-0.0001, "0.000"),
        (1.2346, "1.235"),
        (-1.2346, "-1.235"),
        (1.0, "1.000"),
        (None, "NA"),
        (12, "12"),
        ("0012", "0012"),
    ],
)
def test_three_decimal_export(value, expected):
    assert reports.formatted(value) == expected


def test_export_numeric_validation():
    assert reports.numeric("0.400") == 0.4
    assert reports.numeric("0.40000000000000002", precision=".17g") == 0.4
    for invalid in ("0.4", "-0.000", "nan", "inf", "0.4001"):
        with pytest.raises(IntegrityError):
            reports.numeric(invalid)


def test_complete_resume_without_work_receipts(cohort, fake_predictor, monkeypatch):
    prepared, output = cohort
    extraction.extract_features(prepared, output, workers=1, offline=True)
    family = reports.Family(output)
    original = {name: family.path(name).read_bytes() for name in family.names}
    shutil.rmtree(family.work)

    def unexpected_snapshot(*args, **kwargs):
        pytest.fail("Complete-report reuse must not create a snapshot")

    monkeypatch.setattr(store, "snapshot", unexpected_snapshot)
    calls = list(fake_predictor.calls)
    prepares = fake_predictor.prepares
    for _ in range(2):
        result = extraction.extract_features(
            prepared, output, resume=True, workers=1, offline=True
        )
        assert result.reused_output and result.computed_sequences == 0
        assert result.reused_sequences == 4
        assert not family.work.exists()
        assert original == {name: family.path(name).read_bytes() for name in original}
    assert fake_predictor.calls == calls
    assert fake_predictor.prepares == prepares
