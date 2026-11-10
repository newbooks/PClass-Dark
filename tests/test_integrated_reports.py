"""Atomic family completion, no overwrite and recovery ownership."""

from pathlib import Path

import pytest

from pclass_dark.artifacts import read_json
from pclass_dark.integrated_reports import report_outputs


def fill(output):
    output.json.write('{"queries":[]}\n')
    output.markdown.write("report\n")
    output.text.write("screen\n")


def test_publish_collision_and_stdout(tmp_path):
    prefix = tmp_path / "query"
    with report_outputs(prefix) as output:
        fill(output)
        stdout = output.prepare_stdout("text")
    assert stdout.read() == "screen\n"
    stdout.close()
    assert (tmp_path / "query-interpretation.md").read_text() == "report\n"
    assert len(read_json(tmp_path / "query-classification.complete.json")["files"]) == 2
    with pytest.raises(ValueError, match="already exist"):
        with report_outputs(prefix):
            pass


def test_late_failure_no_partial_files(tmp_path):
    with pytest.raises(ValueError):
        with report_outputs(tmp_path / "q") as output:
            fill(output)
            raise ValueError("late")
    assert not list(tmp_path.glob("*.json"))
    assert not list(tmp_path.glob("*.md"))


def test_publication_failure_recovery(tmp_path, monkeypatch):
    import pclass_dark.integrated_reports as r

    real = r.os.link
    calls = []

    def fail(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("failed publication")
        return real(source, target)

    monkeypatch.setattr(r.os, "link", fail)
    with pytest.raises(OSError):
        with report_outputs(tmp_path / "q") as output:
            fill(output)
    monkeypatch.setattr(r.os, "link", real)
    with report_outputs(tmp_path / "q") as output:
        fill(output)
    assert (tmp_path / "q-classification.complete.json").exists()


def test_foreign_publication_race_preserved(tmp_path, monkeypatch):
    import pclass_dark.integrated_reports as r

    real = r.os.link

    def race(source, target):
        Path(target).write_text("foreign content")
        return real(source, target)

    monkeypatch.setattr(r.os, "link", race)
    with pytest.raises(FileExistsError):
        with report_outputs(tmp_path / "q") as output:
            fill(output)
    assert (tmp_path / "q-classification.json").read_text() == "foreign content"
    monkeypatch.setattr(r.os, "link", real)
    with pytest.raises(Exception, match="recovery conflict"):
        with report_outputs(tmp_path / "q"):
            pass
    assert (tmp_path / "q-classification.json").read_text() == "foreign content"
    # Once the conflicting file is removed by its owner, owned staging recovers.
    (tmp_path / "q-classification.json").unlink()
    with report_outputs(tmp_path / "q") as output:
        fill(output)
    assert (tmp_path / "q-classification.complete.json").exists()
