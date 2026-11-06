# SPEC-0019: Script-only evaluation and audits

## Purpose and contract

Remove `external-audit`, `swiss-audit`, and `evaluate-swiss` from the
`pclass-dark` hierarchy. Their existing scripts remain the supported command-line
interfaces. This supersedes the installed-subcommand portions of SPEC-0016,
SPEC-0017, and SPEC-0018; scientific contracts remain unchanged.

Keep the script adapters, importable scientific APIs, options, defaults, outputs,
validation, resource controls and dependencies. Scripts require the installed
package and are included in source distributions; wheel installation does not
install these scripts as commands. No data migration is required.

## Implementation and validation

Implemented: removed imports and registrations from the root CLI, migrated
adapter checks to standalone apps, added absent/rejected-command regression
checks and subprocess help checks for all three scripts. Operational guides and
README now use scripts. No scientific or device changes.

Validation (2026-11-06): combined CLI/audit/Swiss evaluation run: 107 passed,
one setup error because the sandbox blocked the loopback HTTP fixture. The
blocked test passed when rerun with permission to open the local server (1 passed).
The final CLI suite, including three additional script help cases, passed all
25 tests. Ruff lint/format and whitespace checks passed. No live inference or GPU
execution was needed for this interface-only change.
