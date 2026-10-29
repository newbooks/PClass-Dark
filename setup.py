"""Setuptools command hooks; project metadata lives in pyproject.toml."""

import importlib.util
import sys
from pathlib import Path

from setuptools import setup

# PEP 517 executes setup.py without guaranteeing the project is on sys.path.
spec = importlib.util.spec_from_file_location(
    "_pclass_dark_build_support", Path(__file__).resolve().with_name("build_support.py")
)
support = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = support
spec.loader.exec_module(support)
setup(cmdclass={"build_py": support.SharedBuildPy, "sdist": support.SharedSdist})
