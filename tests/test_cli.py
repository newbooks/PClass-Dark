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
