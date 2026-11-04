"""Resident bounded Dark inference and verified prepared-vector reuse.

Construction/import performs no work. Enter explicitly, then consume each chunk
inside its context. Source libraries remain guarded for the runtime lifetime.
"""

import json
import math
import sqlite3
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import inference as infer
from .artifacts import IntegrityError, identity, sha256, transaction_guard
from .embeddings import verify_embeddings
from .inference_backend import backend_context, cpu_budget
from .inference_records import QuerySequence, make_query_windows
from .library_runtime import freeze, representation_contract
from .paths import default_library_input_dir
from .setup_data import verify_prepared


class EmbeddingCache:
    """Verified coordinate-aware index. The supplied bundles are never mutated."""

    def __init__(self, embedding_dir, prepared_dir, contract, index_dir):
        self.matrix = self.db = None
        embedding_dir, prepared_dir = Path(embedding_dir), Path(prepared_dir)
        self._source_files = [
            p
            for folder in (embedding_dir, prepared_dir)
            for p in folder.iterdir()
            if p.is_file()
        ]
        if any(
            p.is_symlink()
            for source in self._source_files
            for p in (source, *source.parents)
        ):
            raise IntegrityError("Embedding cache source symlink")
        self._stamps = self._source_stamps()
        preparation, windows, _ = verify_prepared(prepared_dir)
        manifest = verify_embeddings(embedding_dir, windows)
        if manifest["inventory_id"] != preparation[
            "inventory_id"
        ] or representation_contract(
            manifest["identity_inputs"]["contract"]
        ) != representation_contract(contract):
            raise IntegrityError("Embedding cache preparation/representation mismatch")
        self.source = {
            "embedding_id": manifest["id"],
            "originating_preparation_id": manifest.get("originating_preparation_id"),
            "manifest_sha256": sha256(embedding_dir / "manifest.json"),
            "prepared_id": preparation["id"],
            "prepared_manifest_sha256": sha256(prepared_dir / "manifest.json"),
            "algorithm": "swiss-coordinate-index-v1",
        }
        matrix = np.load(
            embedding_dir / "embeddings.npy", mmap_mode="r", allow_pickle=False
        )
        self.matrix = matrix
        folder = Path(index_dir) / identity(self.source)
        folder.mkdir(parents=True, exist_ok=True)
        # Rebuild after source verification. Atomic replacement prevents adopting
        # an interrupted build; query coverage never relies on a partial index.
        import os
        import uuid

        staged = folder / (uuid.uuid4().hex + ".sqlite")
        db = sqlite3.connect(staged)
        try:
            db.execute("CREATE TABLE vectors(key TEXT PRIMARY KEY,row_index INTEGER)")
            for i, w in enumerate(windows):
                key = self.key(w["sequence_id"], int(w["parent_length"]), w)
                prior = db.execute(
                    "SELECT row_index FROM vectors WHERE key=?", (key,)
                ).fetchone()
                if prior:
                    if matrix[prior[0]].tobytes() != matrix[i].tobytes():
                        raise IntegrityError(
                            "Conflicting duplicate embedding cache key"
                        )
                else:
                    db.execute("INSERT INTO vectors VALUES (?,?)", (key, i))
            db.commit()
            db.close()
            os.replace(staged, folder / "index.sqlite")
            self.db = sqlite3.connect(folder / "index.sqlite")
            self.check()
        except BaseException:
            db.close()
            staged.unlink(missing_ok=True)
            self.close()
            raise

    def _source_stamps(self):
        return [
            (
                p.stat().st_dev,
                p.stat().st_ino,
                p.stat().st_size,
                p.stat().st_mtime_ns,
                p.stat().st_ctime_ns,
            )
            for p in self._source_files
        ]

    def check(self):
        if self._source_stamps() != self._stamps:
            raise IntegrityError("Embedding cache sources changed during inference")

    @staticmethod
    def key(sequence_hash, length, window):
        return json.dumps(
            [
                sequence_hash,
                length,
                window["kind"],
                int(window["start"]),
                int(window["end"]),
                window["window_sha256"],
            ]
        )

    def lookup(self, sequence_hash, length, info):
        key = self.key(sequence_hash, length, info.to_dict())
        row = self.db.execute(
            "SELECT row_index FROM vectors WHERE key=?", (key,)
        ).fetchone()
        return None if row is None else self.matrix[row[0]]

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.matrix is not None:
            self.matrix._mmap.close()
            self.matrix = None


class DarkInferenceRuntime:
    """Own a library and lazy model across disk-backed inference chunks.

    `infer_chunk` returns the same one-shot context manager as `infer_queries`.
    Consume/close it before another chunk. Raises existing inference/integrity
    errors; memory_mib admits scoring workspace, not total model/process RSS.
    """

    def __init__(
        self,
        library_dir=None,
        *,
        device="auto",
        workers=8,
        batch_size=8,
        score_batch_size=4096,
        memory_mib=1024,
        offline=False,
        embedding_cache=None,
        prepared_dir=None,
        _index_dir=None,
    ):
        self.root = Path(library_dir or default_library_input_dir()).absolute()
        self.options = dict(
            device=device,
            workers=workers,
            batch_size=batch_size,
            score_batch_size=score_batch_size,
            memory_mib=memory_mib,
            offline=offline,
        )
        self.cache_paths = embedding_cache, prepared_dir
        self.index_dir = _index_dir
        self._stack = self._backend = self.cache = None
        self.execution = {}
        self._active = False

    def __enter__(self):
        if self._stack is not None:
            raise RuntimeError("Runtime already entered")
        infer._options(**self.options)
        if (self.cache_paths[0] is None) != (self.cache_paths[1] is None):
            raise ValueError("Embedding cache and prepared directory required together")
        self._stack = ExitStack()
        try:
            from .library import _library_immutable

            self.cpu_threads = cpu_budget(self.options["workers"])
            self._stack.enter_context(threadpool_limits(limits=self.cpu_threads))
            if not _library_immutable(self.root, False):
                self._stack.enter_context(transaction_guard(self.root))
            self.pair = infer.load_selected_library(self.root)
            source_root = self.pair.library.source_root
            if (
                not self.pair.library.immutable
                and source_root.resolve() != self.root.resolve()
            ):
                self._stack.enter_context(transaction_guard(source_root))
            self.metadata, self.available, self.coverage = infer._metadata(self.pair)
            self.contract = representation_contract(
                self.pair.library.embedding_contract
            )
            self.required_workspace = self._workspace_bytes()
            if self.required_workspace > self.options["memory_mib"] * 2**20:
                raise infer.InferenceResourceError(
                    f"Need {math.ceil(self.required_workspace / 2**20)} MiB workspace"
                )
            self.snapshot = self._snapshot()
            if (
                self.snapshot[1] != self.pair.library.manifest_sha256
                or self.snapshot[2] != self.pair.relative.manifest_sha256
            ):
                raise IntegrityError("Library changed during initialization")
            self.source = freeze(
                {
                    "library_dir": str(self.root.resolve()),
                    "production_id": self.pair.library.library_id,
                    "production_manifest_sha256": self.snapshot[1],
                    "relative_id": self.pair.relative.relative_id,
                    "relative_manifest_sha256": self.snapshot[2],
                    "representation_contract_id": identity(self.contract),
                    "probability_calibrated": False,
                }
            )
            if self.cache_paths[0] is not None:
                index_dir = self.index_dir
                if index_dir is None:
                    index_dir = self._stack.enter_context(tempfile.TemporaryDirectory())
                self.cache = EmbeddingCache(*self.cache_paths, self.contract, index_dir)
                self._stack.callback(self.cache.close)
            return self
        except BaseException:
            self.close()
            raise

    def _snapshot(self):
        infer.check_cleanup_pending(self.root)
        return (
            sha256(self.root / "selection.json"),
            sha256(self.pair.library.source_path / "manifest.json"),
            sha256(self.pair.relative.path / "manifest.json"),
        )

    def infer_chunk(self, queries):
        if self._stack is None or self._active:
            raise RuntimeError("Enter runtime and close previous chunk first")
        return _ResidentSession(
            queries, library_dir=self.root, _runtime=self, **self.options
        )

    def _workspace_bytes(self):
        s = self.options["score_batch_size"]
        largest = max(
            8 * (self.metadata[c]["C_n"] + self.metadata[c]["B_n"])
            for c in self.available
        )
        return 96 * 2**20 + largest + 16 * 640 * s + 32 * s

    def _compute_session(self, session):
        if self.cache is not None:
            self.cache.check()
        if self._snapshot() != self.snapshot:
            raise IntegrityError("Selected library changed during inference")
        session._metadata, session.coverage = self.metadata, self.coverage
        session.source = self.source
        session._quota(2048 + session._window_count * 640 * 4)
        self._embed_session(session)
        session._score(self.pair, self.available, self.contract)
        if self.cache is not None:
            self.cache.check()
        if self._snapshot() != self.snapshot:
            raise IntegrityError("Selected library changed during inference")

    def _model(self):
        if self._backend is None:
            backend, execution = self._stack.enter_context(
                backend_context(
                    self.contract,
                    device=self.options["device"],
                    workers=self.options["workers"],
                    offline=self.options["offline"],
                )
            )
            if representation_contract(backend.contract) != self.contract:
                raise IntegrityError("Backend representation mismatch")
            self._backend, self._model_execution = backend, execution
        return self._backend

    def inspect_queries(self, queries):
        hits = misses = windows = 0
        for query in queries:
            q = make_query_windows(query)
            for window in q.windows:
                windows += 1
                cached = (
                    None
                    if self.cache is None
                    else self.cache.lookup(q.sequence_sha256, q.length, window.info)
                )
                if cached is None:
                    misses += 1
                else:
                    hits += 1
        return {"windows": windows, "cache_hits": hits, "cache_misses": misses}

    def _embed_session(self, session):
        path = session._directory / "vectors.npy"
        session._quota(session._window_count * 640 * 4)
        matrix = np.lib.format.open_memmap(
            path, mode="w+", dtype="<f4", shape=(session._window_count, 640)
        )
        batch, positions = [], []
        hits = misses = offset = 0

        def flush():
            if not batch:
                return
            values = np.asarray(self._model().embed(batch))
            if (
                values.shape != (len(batch), 640)
                or values.dtype.str != "<f4"
                or not np.isfinite(values).all()
            ):
                raise IntegrityError("Invalid model window matrix")
            matrix[positions] = values
            batch.clear()
            positions.clear()

        try:
            with (session._directory / "queries.jsonl").open() as stream:
                for i, line in enumerate(stream):
                    qid, sequence = json.loads(line)
                    q = make_query_windows(QuerySequence(qid, sequence), query_index=i)
                    for window in q.windows:
                        cached = (
                            None
                            if self.cache is None
                            else self.cache.lookup(
                                q.sequence_sha256, q.length, window.info
                            )
                        )
                        if cached is None:
                            misses += 1
                            positions.append(offset)
                            batch.append(window.sequence)
                            if len(batch) == self.options["batch_size"]:
                                flush()
                        else:
                            hits += 1
                            matrix[offset] = cached
                        offset += 1
            flush()
            if offset != session._window_count:
                raise IntegrityError("Window coverage mismatch")
            matrix.flush()
        finally:
            matrix._mmap.close()
        self.execution = freeze(
            {
                **(
                    self._model_execution
                    if misses
                    else {"embedding_device": "not_used"}
                ),
                "cache_hits": hits,
                "cache_misses": misses,
                "cpu_threads": self.cpu_threads,
                "cache_source": self.cache.source if self.cache else None,
                "batch_size": self.options["batch_size"],
                "score_batch_size": self.options["score_batch_size"],
            }
        )
        session.execution = self.execution
        (session._directory / "queries.jsonl").unlink()

    def close(self):
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        self._backend = self.cache = None

    def __exit__(self, *exc):
        self.close()


class _ResidentSession(infer.InferenceSession):
    def __enter__(self):
        if self._runtime._active:
            raise RuntimeError("Previous chunk is still active")
        self._runtime._active = True
        return super().__enter__()

    def close(self):
        super().close()
        self._runtime._active = False
