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
