# Shared project artifacts

Default commands use this directory regardless of the working directory:

- `setup-data/`: prepared references, frozen embeddings and their selection.
- `library/`: final signatures, C/B distributions, optional research and selectors.

Completed artifacts are tracked through Git LFS and automatically included in
wheel/source builds. Only selected bundles and their manifest-listed files are
exported. Checkpoints, locks, archives, staging files and model weights stay local.
Do not hand-edit selected manifests or data. Regenerate them with the commands.

```bash
pclass-dark setup-data --workers 8
pclass-dark build-library --workers 8
python -m build
```

A package build fails if selected data is corrupt/incomplete or a writer is active.
When no selection exists yet, that stage contributes no generated artifacts to
the package. Building does not run embedding or signature discovery. An installed
package reads bundled artifacts and writes new output to the user cache; explicit
CLI/API paths override these defaults.


To reclaim obsolete generations, preview with
`pclass-dark build-library --clean --dry-run`, then use
`pclass-dark build-library --clean` for a validated full build followed by cleanup.
The option permanently removes eligible old preparations and exclusive outputs,
while retaining shared embeddings, selected data, partial work and archives.
It does not prune local Git LFS objects or remote/history copies. See
[SPEC-0013](../docs/specs/0013-build-library-clean.md). Pending cleanup journals
block packaging until recovery with the same setup/library roots.
