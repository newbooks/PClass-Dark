"""Check the initial public package and CLI interfaces."""

import subprocess
import sys
from importlib.metadata import version

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
