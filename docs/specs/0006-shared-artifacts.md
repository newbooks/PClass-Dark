# SPEC-0006: Shared repository artifacts and package distribution

- Date: 2026-10-27.
- Status: Implemented; verification recorded below.
- Extends SPEC-0003/0004/0005 without changing scientific algorithms or identities.

## Default paths

Resolve checkout location from the imported package's source path, never the
working directory. The source layout and root pyproject must identify this repo.
Defaults are `<repo>/data/setup-data` and `<repo>/data/library` for CLI and Python.
Explicit paths retain their current semantics; relative overrides use the caller's
working directory. No process-wide chdir or import-time directory creation.

Outside a checkout, setup/build outputs use `$XDG_CACHE_HOME/pclass-dark` or
`~/.cache/pclass-dark`. Build input prefers an existing user-cache selection,
then bundled setup resources, then the writable cache path (with the existing
missing-data error if unprepared). Selected-library loading similarly prefers
cache output then packaged library resources. Installed resources remain immutable.

## Distribution

Track completed `data/` artifacts with Git LFS; ignore hidden checkpoint, staging,
archive and lock directories. Build wheels and source archives from the selected
completed setup/library/research bundles, using selectors and manifest inventories
rather than recursively including the working directory. Static reference data
remains under package resources. Add no model weights or intermediate arrays.

A setuptools build helper must validate selectors, bundle status/schema, IDs,
manifest-byte digests and file hashes/size inventories before copying. Use the
same nonblocking root transaction locks as runtime publishers, in sorted order.
Fail clearly on active writers, missing/corrupt selected data, unsafe paths or
inconsistent setup/library selections; no silent partial releases. Missing selectors
mean no completed artifacts from that stage yet and are not a source-build error.

Wheel destinations retain existing resource layouts: setup selection at
`resources/selection.json`, prepared/embedding bundles below their namespaces,
and library/relative/research selectors and bundles below `resources/library`.
Source archives include only the selected shared data files plus the build helper,
so rebuilding their wheel reproduces the same artifact payload. Clear stale build
output resource namespaces before copying current source and shared resources.
Never write generated copies into the source package or mutate shared artifacts.

## Validation

Test defaults from repo root, nested work directory and unrelated cwd; explicit
path overrides; installed bundled-data and cache fallback; selected manifest/file
validation and private-file exclusion; build lock contention; wheel/source rebuild
with small completed setup/embedding/library fixtures; and installed access from
outside the checkout with PLM imports blocked and no writes to package files.
Keep README, data directory instructions, earlier spec defaults and progress log
synchronized. Do not launch full PLM inference/library discovery to change paths.

## Implementation and verification

`paths.py` detects the checkout from the imported package and supplies defaults
for preparation, library building and selected-library loading. CLI defaults use
None so resolution happens in the reusable API, not at import/help time. Explicit
paths remain unchanged. The package-resources root itself is treated as immutable,
including when it is the default installed setup input.

Setuptools hooks in `build_support.py`/`setup.py` export validated selected shared
artifacts into wheel resources. The source archive includes the selected data and
helper, so its wheel rebuild exports the same payload. A fingerprint check catches
selection changes between source inventory and release-tree creation. Build
requirements now include filelock. New Git LFS/ignore rules distinguish completed
`data/` files from private hidden directories; README remains normal text.

Full regression run: **135 passed, 2 opt-in tests skipped** in 202.13 s, with the
existing Torch environment warning. After the installed-input root fix, **13
focused path/selection/CLI tests passed**. Built source and wheel from a small
complete shared dataset/library/research fixture: 49 package-resource files,
selected artifacts present in both archives, dummy private checkpoint absent.
Installed wheel verification passed outside the checkout with PLM imports blocked
and package writes denied: bundled loading/query evidence, build-library reading
bundled embeddings and writing user-cache output, and setup-data prepare-only
writing the user cache. No full embedding/discovery run was launched.

An isolated editable install was also verified from outside its checkout: defaults
still selected that checkout's shared data. Editable builds skip artifact copying
because they read this data in place; normal wheels/source rebuilds export the
selected payload. Hidden temporary files as well as private directories are
excluded by the shared-data Git ignore rule.
