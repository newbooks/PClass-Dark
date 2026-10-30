"""Setuptools hooks exporting only manifest-selected shared data (no PLM imports)."""

from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path

from filelock import FileLock, Timeout
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

EXTENSIONS = {".json", ".tsv", ".fasta", ".gz", ".npy"}
NAMESPACES = ("prepared", "embeddings", "library")


def safe(root, name):
    relative = Path(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or any(p.startswith(".") for p in relative.parts)
    ):
        raise ValueError(f"Unsafe release member: {name}")
    path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symlink release member: {path}")
    return path


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate release metadata key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(project):
    """Return source paths and package-resource destinations under held root locks."""
    result = {}
    setup = project / "data/setup-data"
    library = project / "data/library"

    def add(source, destination):
        if not source.is_file() or source.suffix not in EXTENSIONS:
            raise ValueError(f"Unsupported/missing release artifact: {source}")
        old = result.get(destination)
        if old is not None and old != source:
            raise ValueError(f"Conflicting release destination: {destination}")
        result[destination] = source

    def bundle(root, relative, expected_id, expected_sha, schemas, prefix):
        folder = safe(root, relative)
        manifest_path = safe(folder, "manifest.json")
        m = read(manifest_path)
        if (
            m.get("schema") not in schemas
            or m.get("status") != "complete"
            or m.get("id") != expected_id
            or digest(manifest_path) != expected_sha
            or not isinstance(m.get("files"), dict)
            or not m["files"]
        ):
            raise ValueError(f"Invalid selected release bundle: {folder}")
        add(manifest_path, prefix / relative / "manifest.json")
        for name, record in m["files"].items():
            file = safe(folder, name)
            if (
                not isinstance(record, dict)
                or not file.is_file()
                or file.stat().st_size != record.get("size_bytes")
                or digest(file) != record.get("sha256")
            ):
                raise ValueError(f"Selected release checksum/size mismatch: {file}")
            add(file, prefix / relative / name)
        return m

    selected_setup = None
    if (setup / "selection.json").is_file():
        selected_setup = read(safe(setup, "selection.json"))
        s = selected_setup
        if s["schema"] != "pclass-dark.selection.v1":
            raise ValueError("Invalid setup-data release selector")
        if s["prepared"] != "prepared/" + s["preparation_id"]:
            raise ValueError("Unexpected prepared release path")
        pm = bundle(
            setup,
            s["prepared"],
            s["preparation_id"],
            s["prepared_manifest_sha256"],
            {"pclass-dark.prepared.v1"},
            Path(),
        )
        if pm["inventory_id"] != s["inventory_id"]:
            raise ValueError("Prepared release inventory mismatch")
        if s["embedding"] is not None:
            if s["embedding"] != "embeddings/" + s["embedding_id"]:
                raise ValueError("Unexpected embeddings release path")
            em = bundle(
                setup,
                s["embedding"],
                s["embedding_id"],
                s["embedding_manifest_sha256"],
                {"pclass-dark.embeddings.v1"},
                Path(),
            )
            if em["inventory_id"] != s["inventory_id"]:
                raise ValueError("Embedding release inventory mismatch")
        add(setup / "selection.json", Path("selection.json"))
    if (library / "selection.json").is_file():
        s = read(safe(library, "selection.json"))
        if s["schema"] not in {
            "pclass-dark.library-selection.v1",
            "pclass-dark.library-selection.v2",
        }:
            raise ValueError("Invalid library release selector")
        if s["library"] != s["library_id"]:
            raise ValueError("Unexpected production release path")
        bundle(
            library,
            s["library"],
            s["library_id"],
            s["manifest_sha256"],
            {"pclass-dark.library.v1", "pclass-dark.library.v2"},
            Path("library"),
        )
        if selected_setup is not None and (
            s["preparation_id"] != selected_setup["preparation_id"]
            or s["embedding_id"] != selected_setup["embedding_id"]
        ):
            raise ValueError(
                "Setup-data and library differ; update selections before release"
            )
        if s["schema"].endswith(".v2"):
            if s["relative"] != "relative-scores/" + s["relative_id"]:
                raise ValueError("Unexpected relative-score release path")
            rm = bundle(
                library,
                s["relative"],
                s["relative_id"],
                s["relative_manifest_sha256"],
                {"pclass-dark.relative-score-library.v1"},
                Path("library"),
            )
            if (
                rm["source_production"]["id"] != s["library_id"]
                or rm["source_production"]["manifest_sha256"] != s["manifest_sha256"]
            ):
                raise ValueError(
                    "Relative release is bound to another production manifest"
                )
        add(library / "selection.json", Path("library/selection.json"))
        if (library / "research-selection.json").is_file():
            r = read(safe(library, "research-selection.json"))
            if (
                r["schema"] != "pclass-dark.library-research-selection.v1"
                or r["research"] != "research/" + r["research_id"]
                or r["production_library_id"] != s["library_id"]
                or r["production_manifest_sha256"] != s["manifest_sha256"]
            ):
                raise ValueError("Research selection differs from selected production")
            rm = bundle(
                library,
                r["research"],
                r["research_id"],
                r["manifest_sha256"],
                {"pclass-dark.library-research.v1"},
                Path("library"),
            )
            if rm["source_production"]["manifest_sha256"] != s["manifest_sha256"]:
                raise ValueError("Research release production binding mismatch")
            add(
                library / "research-selection.json",
                Path("library/research-selection.json"),
            )
    return [(source, destination) for destination, source in sorted(result.items())]


@contextmanager
def shared_files(project):
    with ExitStack() as stack:
        for root in sorted(project / "data" / n for n in ("setup-data", "library")):
            if not root.exists():
                continue
            if any(p.is_symlink() for p in (root, *root.parents)):
                raise ValueError("Symlink shared artifact root")
            locks = root / ".locks"
            if locks.is_symlink():
                raise ValueError("Symlink shared lock directory")
            locks.mkdir(exist_ok=True)
            try:
                stack.enter_context(
                    FileLock(str(locks / "setup-transaction.lock"), timeout=0)
                )
            except Timeout as exc:
                raise RuntimeError(
                    f"Artifact writer active at {root}; retry when it finishes"
                ) from exc
        yield inventory(project)


class SharedBuildPy(build_py):
    def run(self):
        # Editable installs read shared data in place; do not copy multi-GB data.
        if getattr(self, "editable_mode", False):
            return super().run()
        project = Path.cwd().resolve()
        target = Path(self.build_lib).resolve() / "pclass_dark/resources"
        if target == project / "src/pclass_dark/resources" or target.is_symlink():
            raise ValueError("Build output must not be the source resources directory")
        # Don't ship leftovers from an earlier invocation of build_py.
        for name in (
            *NAMESPACES,
            "selection.json",
            "reference",
            "cluster_descriptions.json",
        ):
            path = target / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        super().run()
        with shared_files(project) as files:
            # Override manually staged resource versions when shared selections exist.
            groups = {destination.parts[0] for _, destination in files}
            clear = (
                {"prepared", "embeddings", "selection.json"}
                if "selection.json" in groups
                else set()
            )
            if "library" in groups:
                clear.add("library")
            for name in clear:
                path = target / name
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            self._shared_outputs = []
            for source, destination in files:
                path = target / destination
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, path)
                self._shared_outputs.append(str(path))

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + getattr(
            self, "_shared_outputs", []
        )


class SharedSdist(sdist):
    def make_distribution(self):
        project = Path.cwd().resolve()
        with shared_files(project) as files:
            # SOURCES.txt may have come from an earlier selection.
            self.filelist.files = [
                f
                for f in self.filelist.files
                if (
                    not f.replace("\\", "/").startswith("data/")
                    or f == "data/README.md"
                )
                and not f.replace("\\", "/").startswith(
                    "src/pclass_dark/resources/reference/"
                )
                and f != "src/pclass_dark/resources/cluster_descriptions.json"
            ]
            self.filelist.extend(
                str(source.relative_to(project)) for source, _ in files
            )
            self.filelist.sort()
            self.filelist.remove_duplicates()
            self._shared_snapshot = [
                (str(p.relative_to(project)), digest(p)) for p, _ in files
            ]
            egg = self.get_finalized_command("egg_info")
            (Path(egg.egg_info) / "SOURCES.txt").write_text(
                "\n".join(self.filelist.files) + "\n"
            )
        super().make_distribution()

    def make_release_tree(self, base_dir, files):
        project = Path.cwd().resolve()
        with shared_files(project) as selected:
            snapshot = [(str(p.relative_to(project)), digest(p)) for p, _ in selected]
            if snapshot != self._shared_snapshot:
                raise RuntimeError(
                    "Shared artifact generation changed during sdist; retry"
                )
            super().make_release_tree(base_dir, files)
