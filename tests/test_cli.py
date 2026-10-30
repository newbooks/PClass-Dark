"""Check the initial public package and CLI interfaces."""

import subprocess
import sys
from importlib.metadata import version

import pytest
from typer.testing import CliRunner

import pclass_dark
from pclass_dark.cli import app

runner = CliRunner()


def test_version_matches_distribution():
    assert pclass_dark.__version__ == version("pclass-dark") == "0.0.1"
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "pclass-dark 0.0.1"


def test_help_and_unknown_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--version" in result.stdout
    assert "cluster-conserved" in result.stdout
    assert runner.invoke(app, ["unknown-command"]).exit_code != 0


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "pclass_dark", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "pclass-dark 0.0.1"


def test_setup_log_level_is_bracketed():
    script = """
import logging
import sys
from types import SimpleNamespace
import pclass_dark.setup_data as setup
from pclass_dark.cli import main

def prepare(*args, **kwargs):
    logging.getLogger('pclass_dark.setup_data').info(
        'Background identities: 1/3443 targets'
    )
    return SimpleNamespace(prepared_dir='prepared', embedding_dir=None)

setup.prepare_data = prepare
sys.argv = ['pclass-dark', 'setup-data', '--prepare-only']
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stderr.strip() == "[INFO] Background identities: 1/3443 targets"


@pytest.mark.parametrize("command", ["setup-data", "build-library"])
def test_subcommand_short_help_matches_long_help(command):
    short = runner.invoke(app, [command, "-h"])
    long = runner.invoke(app, [command, "--help"])
    assert short.exit_code == long.exit_code == 0
    assert short.stdout == long.stdout
    assert "--help" in short.stdout and "-h" in short.stdout
    assert "--workers" in short.stdout


def test_setup_reference_fasta_forwarded(monkeypatch):
    from types import SimpleNamespace

    calls = []

    def prepare(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(prepared_dir="prepared", embedding_dir=None)

    monkeypatch.setattr("pclass_dark.setup_data.prepare_data", prepare)
    result = runner.invoke(
        app, ["setup-data", "--prepare-only", "--reference-fasta", "source.fasta"]
    )
    assert result.exit_code == 0
    assert calls[0]["reference_fasta"] == "source.fasta"


@pytest.mark.parametrize("command", ["setup-data", "build-library"])
@pytest.mark.parametrize("workers", [None, 1])
def test_worker_default_and_serial_override(command, workers, monkeypatch):
    from types import SimpleNamespace

    calls = []

    def operation(*args, **kwargs):
        calls.append(kwargs["workers"])
        return SimpleNamespace(
            prepared_dir="prepared",
            embedding_dir=None,
            library_dir="library",
            relative_dir=None,
            research_dir=None,
        )

    target = (
        "pclass_dark.setup_data.prepare_data"
        if command == "setup-data"
        else "pclass_dark.library.build_library"
    )
    monkeypatch.setattr(target, operation)
    arguments = [command]
    if workers is not None:
        arguments += ["--workers", str(workers)]
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert calls == [8 if workers is None else workers]
