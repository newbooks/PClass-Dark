# SPEC-0001: Package and command-line interface

## Purpose and design

Provide an installable Python package at version 0.0.1, following the neighboring
profunc and ProStruct command/subcommand convention with Typer. Use a `src/`
layout, setuptools, and Python 3.10 or newer.

- Distribution and console command: `pclass-dark`.
- Import namespace: `pclass_dark`.
- Console entry point: `pclass_dark.cli:main`.
- Equivalent module entry point: `python -m pclass_dark`.
- The installed distribution metadata is the source of the public `__version__`.
- Initial CLI options: `--help` and `--version`. No scientific subcommands exist
  yet. Unknown commands must fail with a nonzero exit status.
- Future scientific operations belong in importable library modules; CLI
  subcommands are thin adapters, with no requirement to invoke a subprocess
  from integrating Python code. Library modules must not parse command-line
  arguments or start workloads on import.

## Dependencies and scope

Initial core dependencies are NumPy and SciPy for planned numerical work, and
Typer for the CLI. The optional `embeddings` extra adds PyTorch and Transformers;
model choice, pooling, hardware selection, and compatible version constraints
will be refined when embedding generation is implemented. The `dev` extra adds
pytest, build, and Ruff. No dependency on a neighboring checkout is introduced.
No profiling, classification, GPU execution, or multiprocessing is implemented
by this scaffold.

## Validation and status

Implemented. Tests cover package/distribution version consistency, CLI help,
version output, unknown-command rejection, and the Python module entry point.
Build a wheel and smoke-test its installed console command and package import.
Validation results are recorded in the progress log.
