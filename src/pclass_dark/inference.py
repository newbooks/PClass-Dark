"""Inference orchestration using pinned embeddings and accepted-window evidence.

No model, argument parsing or filesystem work occurs on import/construction.
Use infer_queries() as a context manager; results are streamed from private
working storage after all inference succeeds. See SPEC-0007 for resource limits.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .artifacts import (
    IntegrityError,
    check_cleanup_pending,
    identity,
    sha256,
    transaction_guard,
)
from .inference_backend import backend_context as _backend_context
from .inference_records import (
    MAX_HEADER,
    MAX_INPUT,
    MAX_QUERIES,
    MAX_SEQUENCE,
    MAX_TEMP,
    ClusterMatch,
    InferenceError,
    InferenceResourceError,
    QueryResult,
    QuerySequence,
    QueryWindow,
    QueryWindows,
    WindowInfo,
    WindowMatch,
    combine_matches,
    make_query_windows,
    validate_windows,
)
from .library_runtime import freeze, representation_contract
from .paths import default_library_input_dir
from .reference_data import FastaLimitError, read_fasta
from .relative_scores import load_selected_library

__all__ = [
    "QuerySequence",
    "QueryWindow",
    "QueryWindows",
    "WindowInfo",
    "WindowMatch",
    "ClusterMatch",
    "QueryResult",
    "InferenceSession",
    "InferenceError",
    "InferenceResourceError",
    "make_query_windows",
    "score_query_windows",
    "infer_queries",
    "read_query_fasta",
    "write_inference_report",
]
LOG = logging.getLogger(__name__)


def read_query_fasta(path):
    """Yield bounded query records; format/resource failures never skip records."""
    try:
        for name, sequence in read_fasta(
            path,
            max_sequence_length=MAX_SEQUENCE,
            max_header_bytes=MAX_HEADER,
            max_bytes=MAX_INPUT,
            max_records=MAX_QUERIES,
            check_stable=True,
        ):
            yield QuerySequence(name, sequence)
    except FastaLimitError as exc:
        raise InferenceResourceError(str(exc)) from exc


def _metadata(pair):
    if pair.relative is None:
        raise IntegrityError("infer requires production/C/B data; run build-library")
    metadata = pair.relative.describe()
    available, unavailable = [], []
    if set(metadata) != set(pair.library.signatures):
        raise IntegrityError("Signature/relative target inventory mismatch")
    for cluster in sorted(metadata):
        meta = metadata[cluster]
        if pair.library.signatures[cluster]["status"] != meta["status"]:
            raise IntegrityError("Signature/distribution status mismatch")
        if meta["status"] == "ok":
            available.append(cluster)
        else:
            unavailable.append(
                {
                    "cluster_id": cluster,
                    "status": meta["status"],
                    "reason": meta["reason"],
                }
            )
    if not available:
        raise InferenceError("No usable cluster signatures/distributions")
    return (
        metadata,
        tuple(available),
        freeze(
            {
                "clusters_total": len(metadata),
                "clusters_evaluated": len(available),
                "unavailable": unavailable,
            }
        ),
    )


def _scored(pair, cluster, vectors, contract):
    scores, accepted = pair.library.score(cluster, vectors, embedding_contract=contract)
    if (
        scores.shape != (len(vectors),)
        or accepted.shape != scores.shape
        or accepted.dtype.kind != "b"
        or not np.isfinite(scores).all()
    ):
        raise IntegrityError("Invalid/nonfinite query scores or acceptance flags")
    return scores, np.flatnonzero(accepted)


def score_query_windows(pair, queries, vectors, *, embedding_contract):
    """Score precomputed windows; no fitting/model loading or implicit printing.

    Raises ValueError for invalid window/vector contracts and IntegrityError for
    incompatible generations. Uses the same reducer as database-backed inference.
    """
    queries = tuple(queries)
    vectors = validate_windows(queries, vectors)
    if identity(representation_contract(embedding_contract)) != identity(
        representation_contract(pair.library.embedding_contract)
    ):
        raise IntegrityError("Query embedding representation differs from the library")
    metadata, available, _ = _metadata(pair)
    rows = [[] for _ in queries]
    keys = [(i, w) for i, q in enumerate(queries) for w in range(len(q.windows))]
    for cluster in available:
        scores, indices = _scored(pair, cluster, vectors, embedding_contract)
        if not len(indices):
            continue
        evidence = pair.relative.evaluate(cluster, scores[indices])
        for k, row in enumerate(indices):
            q, w = keys[int(row)]
            rows[q].append(
                (
                    cluster,
                    w,
                    evidence.background_tail_fraction[k],
                    evidence.cluster_percentile_le[k],
                )
            )
    return tuple(
        combine_matches(
            q.query_index,
            q.query_id,
            q.length,
            q.sequence_sha256,
            tuple(w.info for w in q.windows),
            values,
            metadata,
        )
        for q, values in zip(queries, rows, strict=True)
    )


def _options(device, workers, batch_size, score_batch_size, memory_mib, offline):
    if not isinstance(device, str) or not re.fullmatch(
        r"auto|cpu|cuda(?::[0-9]+)?", device
    ):
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
    for name, value, low, high in (
        ("workers", workers, 1, None),
        ("batch_size", batch_size, 1, 64),
        ("score_batch_size", score_batch_size, 1, 65536),
        ("memory_mib", memory_mib, 128, None),
    ):
        if type(value) is not int or value < low or high is not None and value > high:
            raise ValueError(
                f"Invalid {name}; minimum {low}, "
                f"maximum {high or 'available resources'}"
            )
    if type(offline) is not bool:
        raise ValueError("offline must be a boolean")


class InferenceSession:
    """One-shot context-managed inference with private bounded disk working data.

    All computation completes on entry; iteration materializes one query result.
    Fixed caps: 1 GiB input, 100k queries, 1M residues/record, 2 GiB total working
    storage. memory_mib budgets arrays/SQLite, not model or total process memory.
    """

    def __init__(
        self,
        queries,
        *,
        library_dir=None,
        device="auto",
        workers=8,
        batch_size=8,
        score_batch_size=4096,
        memory_mib=1024,
        offline=False,
    ):
        self._queries = queries
        self._library_dir = library_dir
        self._options = dict(
            device=device,
            workers=workers,
            batch_size=batch_size,
            score_batch_size=score_batch_size,
            memory_mib=memory_mib,
            offline=offline,
        )
        self._state, self._emitted = "new", 0
        self._temporary = self._db = None

    def __enter__(self):
        if self._state != "new":
            raise RuntimeError("InferenceSession cannot be restarted")
        self._state = "entering"
        try:
            _options(**self._options)
            self._temporary = tempfile.TemporaryDirectory(prefix="pclass-dark-infer-")
            self._directory = Path(self._temporary.name)
            self._db_path = self._directory / "working.sqlite"
            self._db = sqlite3.connect(self._db_path, isolation_level=None)
            self._db.executescript("""
                PRAGMA cache_size=-32768;
                PRAGMA temp_store=MEMORY;
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                CREATE TABLE queries(query_index INTEGER PRIMARY KEY,
                    query_id TEXT UNIQUE NOT NULL, length INTEGER NOT NULL,
                    sequence_sha256 TEXT NOT NULL);
                CREATE TABLE windows(row_index INTEGER PRIMARY KEY,
                    query_index INTEGER NOT NULL,
                    window_index INTEGER NOT NULL, window_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    start INTEGER NOT NULL, end INTEGER NOT NULL,
                    window_sha256 TEXT NOT NULL,
                    UNIQUE(query_index,window_index));
                CREATE TABLE accepted(query_index INTEGER NOT NULL,
                    cluster_id TEXT NOT NULL,
                    window_index INTEGER NOT NULL,
                    background_tail_fraction REAL NOT NULL,
                    cluster_percentile_le REAL NOT NULL,
                    CHECK(background_tail_fraction>0 AND background_tail_fraction<=1),
                    CHECK(cluster_percentile_le>=0 AND cluster_percentile_le<=1),
                    PRIMARY KEY(query_index,cluster_id,window_index)) WITHOUT ROWID;
            """)
            self._preflight()
            self._compute()
            self._state = "active"
            return self
        except BaseException as exc:
            self.close()
            if isinstance(exc, sqlite3.Error):
                if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL:
                    raise InferenceResourceError(
                        "Temporary quota exceeded; split the input"
                    ) from exc
                raise InferenceError("Temporary evidence database failed") from exc
            raise

    def _usage(self):
        return sum(p.stat().st_size for p in self._directory.iterdir() if p.is_file())

    def _quota(self, extra=0):
        if self._usage() + extra > MAX_TEMP:
            raise InferenceResourceError(
                "2 GiB temporary quota exceeded; split the input"
            )

    def _db_limit(self, reserve=0):
        page_size = self._db.execute("PRAGMA page_size").fetchone()[0]
        size = self._db_path.stat().st_size
        allowed = (MAX_TEMP - (self._usage() - size) - reserve) // page_size
        pages = self._db.execute("PRAGMA page_count").fetchone()[0]
        if allowed < pages or allowed < 1:
            raise InferenceResourceError(
                "Temporary database quota exceeded; split the input"
            )
        self._db.execute(f"PRAGMA max_page_count={allowed}")

    def _preflight(self):
        LOG.info("Inference: validating query input")
        spool = self._directory / "queries.jsonl"
        row_index = raw_bytes = count = 0
        with spool.open("wb") as handle:
            for index, query in enumerate(self._queries):
                if index >= MAX_QUERIES:
                    raise InferenceResourceError(
                        "Query count exceeds 100,000; split the input"
                    )
                if not isinstance(query, QuerySequence) or not isinstance(
                    query.sequence, str
                ):
                    raise ValueError("Expected QuerySequence records")
                raw_bytes += (
                    len(query.sequence) + len(query.query_id.encode("utf-8"))
                    if isinstance(query.query_id, str)
                    else 0
                )
                if raw_bytes > MAX_INPUT:
                    raise InferenceResourceError("Input exceeds 1 GiB; split the input")
                windows = make_query_windows(query, query_index=index)
                data = (
                    json.dumps(
                        [query.query_id, query.sequence.strip(" \t\r\n\v\f")],
                        ensure_ascii=True,
                    )
                    + "\n"
                ).encode("utf-8")
                if handle.tell() + len(data) > MAX_INPUT:
                    raise InferenceResourceError("Normalized input exceeds 1 GiB")
                self._quota(len(data))
                self._db_limit(reserve=len(data))
                try:
                    self._db.execute(
                        "INSERT INTO queries VALUES(?,?,?,?)",
                        (
                            index,
                            windows.query_id,
                            windows.length,
                            windows.sequence_sha256,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"Duplicate query ID: {query.query_id}") from exc
                for ordinal, window in enumerate(windows.windows):
                    info = window.info
                    self._db.execute(
                        "INSERT INTO windows VALUES(?,?,?,?,?,?,?,?)",
                        (
                            row_index,
                            index,
                            ordinal,
                            info.window_id,
                            info.kind,
                            info.start,
                            info.end,
                            info.window_sha256,
                        ),
                    )
                    row_index += 1
                handle.write(data)
                handle.flush()
                self._quota()
                count += 1
        self._queries = None
        if not count:
            raise ValueError("Expected at least one query")
        self.query_count, self._window_count = count, row_index

    def _snapshot(self, pair, root):
        try:
            return (
                sha256(root / "selection.json"),
                sha256(pair.library.source_path / "manifest.json"),
                sha256(pair.relative.path / "manifest.json"),
            )
        except OSError as exc:
            raise IntegrityError("Library generation removed; retry inference") from exc

    def _compute(self):
        root = Path(
            default_library_input_dir()
            if self._library_dir is None
            else self._library_dir
        ).absolute()
        check_cleanup_pending(root)
        LOG.info("Inference: validating selected production/C/B library")
        from .library import _library_immutable

        with ExitStack() as initial:
            if not _library_immutable(root, False):
                initial.enter_context(transaction_guard(root))
            pair = load_selected_library(root)
            self._metadata, available, self.coverage = _metadata(pair)
            contract = representation_contract(pair.library.embedding_contract)
            snapshot = self._snapshot(pair, root)
            if (
                snapshot[1] != pair.library.manifest_sha256
                or snapshot[2] != pair.relative.manifest_sha256
            ):
                raise IntegrityError("Library snapshot changed; retry inference")
        s = self._options["score_batch_size"]
        largest = max(
            8 * (self._metadata[c]["C_n"] + self._metadata[c]["B_n"]) for c in available
        )
        required = 96 * 2**20 + largest + 16 * 640 * s + 32 * s
        if required > self._options["memory_mib"] * 2**20:
            raise InferenceResourceError(
                f"Need {math.ceil(required / 2**20)} MiB scoring workspace; "
                "reduce score batch or raise memory budget"
            )
        self.source = freeze(
            {
                "library_dir": str(root.resolve()),
                "production_id": pair.library.library_id,
                "production_manifest_sha256": snapshot[1],
                "relative_id": pair.relative.relative_id,
                "relative_manifest_sha256": snapshot[2],
                "representation_contract_id": identity(contract),
                "probability_calibrated": False,
            }
        )
        self._quota(2048 + self._window_count * 640 * 4)
        with _backend_context(
            contract,
            device=self._options["device"],
            workers=self._options["workers"],
            offline=self._options["offline"],
        ) as (backend, execution):
            if identity(representation_contract(backend.contract)) != identity(
                contract
            ):
                raise IntegrityError(
                    "Backend embedding representation differs from library"
                )
            self.execution = freeze(
                {
                    **execution,
                    "batch_size": self._options["batch_size"],
                    "score_batch_size": s,
                }
            )
            with ExitStack() as guards:
                for p in sorted({root.resolve(), pair.library.source_root.resolve()}):
                    check_cleanup_pending(p)
                    if not pair.library.immutable:
                        guards.enter_context(transaction_guard(p))
                    check_cleanup_pending(p)
                if self._snapshot(pair, root) != snapshot:
                    raise IntegrityError(
                        "Selected library changed during model initialization; "
                        "retry inference"
                    )
                self._embed(backend)
                self._score(pair, available, contract)
                for p in {root, pair.library.source_root}:
                    check_cleanup_pending(p)
                if self._snapshot(pair, root) != snapshot:
                    raise IntegrityError(
                        "Library snapshot changed during inference; retry"
                    )
        previous_query = previous_match = None
        rows = matched_queries = 0
        for query, cluster in self._db.execute(
            "SELECT query_index,cluster_id FROM accepted "
            "ORDER BY query_index,cluster_id,window_index"
        ):
            if query != previous_query:
                matched_queries += 1
                previous_query = query
            if (query, cluster) != previous_match:
                rows += 1
                previous_match = query, cluster
        self.counts = freeze(
            {
                "queries_total": self.query_count,
                "queries_with_matches": matched_queries,
                "matched_cluster_rows": rows,
                "windows_total": self._window_count,
            }
        )

    def _embed(self, backend):
        LOG.info("Inference: embedding %s windows", self._window_count)
        path = self._directory / "vectors.npy"
        matrix = np.lib.format.open_memmap(
            path, mode="w+", dtype="<f4", shape=(self._window_count, 640)
        )
        batch, offset = [], 0
        try:
            self._quota()
            with (self._directory / "queries.jsonl").open(encoding="utf-8") as source:
                for index, line in enumerate(source):
                    name, sequence = json.loads(line)
                    windows = make_query_windows(
                        QuerySequence(name, sequence), query_index=index
                    )
                    for window in windows.windows:
                        batch.append(window.sequence)
                        if len(batch) == self._options["batch_size"]:
                            offset = self._forward(backend, matrix, batch, offset)
                            batch = []
                if batch:
                    offset = self._forward(backend, matrix, batch, offset)
            if offset != self._window_count:
                raise IntegrityError("Embedding row coverage mismatch")
            matrix.flush()
        finally:
            matrix._mmap.close()
        (self._directory / "queries.jsonl").unlink()
        self._quota()

    def _forward(self, backend, matrix, batch, offset):
        try:
            values = backend.embed(batch)
        except Exception as exc:
            raise InferenceError(
                "Embedding failed; reduce --batch-size or use --device cpu"
            ) from exc
        values = np.asarray(values)
        if (
            values.shape != (len(batch), 640)
            or values.dtype.str != "<f4"
            or not np.isfinite(values).all()
        ):
            raise IntegrityError("Invalid model window matrix shape/dtype/values")
        matrix[offset : offset + len(batch)] = values
        return offset + len(batch)

    def _score(self, pair, available, contract):
        path = self._directory / "vectors.npy"
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            for number, cluster in enumerate(available, 1):
                if number == 1 or number % 100 == 0 or number == len(available):
                    LOG.info("Inference: scoring cluster %s/%s", number, len(available))
                with pair.relative.for_cluster(cluster) as reference:
                    for start in range(
                        0, len(matrix), self._options["score_batch_size"]
                    ):
                        stop = min(
                            start + self._options["score_batch_size"], len(matrix)
                        )
                        scores, indices = _scored(
                            pair, cluster, matrix[start:stop], contract
                        )
                        if not len(indices):
                            continue
                        evidence = reference.evaluate(scores[indices])
                        keys = self._db.execute(
                            "SELECT query_index,window_index FROM windows "
                            "WHERE row_index>=? AND row_index<? ORDER BY row_index",
                            (start, stop),
                        ).fetchall()
                        if len(keys) != stop - start:
                            raise IntegrityError("Window row mapping mismatch")
                        records = [
                            (
                                keys[int(i)][0],
                                cluster,
                                keys[int(i)][1],
                                float(evidence.background_tail_fraction[k]),
                                float(evidence.cluster_percentile_le[k]),
                            )
                            for k, i in enumerate(indices)
                        ]
                        self._db_limit()
                        self._db.execute("BEGIN")
                        self._db.executemany(
                            "INSERT INTO accepted VALUES(?,?,?,?,?)", records
                        )
                        self._db.execute("COMMIT")
                        self._quota()
        finally:
            matrix._mmap.close()
        path.unlink()

    def __iter__(self):
        if self._state != "active":
            raise RuntimeError("Use InferenceSession inside its context")
        return self

    def __next__(self):
        if self._state != "active":
            raise RuntimeError("Use InferenceSession inside its context")
        if self._emitted >= self.query_count:
            raise StopIteration
        index = self._emitted
        row = self._db.execute(
            "SELECT query_id,length,sequence_sha256 FROM queries WHERE query_index=?",
            (index,),
        ).fetchone()
        if row is None:
            raise IntegrityError("Query result coverage mismatch")
        windows = tuple(
            WindowInfo(*w)
            for w in self._db.execute(
                "SELECT window_id,kind,start,end,window_sha256 FROM windows "
                "WHERE query_index=? ORDER BY window_index",
                (index,),
            )
        )
        accepted = self._db.execute(
            "SELECT cluster_id,window_index,background_tail_fraction,"
            "cluster_percentile_le FROM accepted WHERE query_index=? "
            "ORDER BY cluster_id,window_index",
            (index,),
        )
        result = combine_matches(index, *row, windows, accepted, self._metadata)
        self._emitted += 1
        return result

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self._queries = None
        self._state = "closed"

    def __exit__(self, *exc):
        self.close()


def infer_queries(
    queries,
    *,
    library_dir=None,
    device="auto",
    workers=8,
    batch_size=8,
    score_batch_size=4096,
    memory_mib=1024,
    offline=False,
):
    """Create a lazy context-managed run. See InferenceSession for caps/errors."""
    return InferenceSession(
        queries,
        library_dir=library_dir,
        device=device,
        workers=workers,
        batch_size=batch_size,
        score_batch_size=score_batch_size,
        memory_mib=memory_mib,
        offline=offline,
    )


def write_inference_report(session, stream, *, format="text"):
    """Explicitly render a complete fresh session; no implicit stdout or file path.

    Does not roll back the caller's stream. CLI uses a private report spool for
    failure-atomic publication. B/C remain native unrounded fractions in JSON.
    """
    from .inference_render import write_report

    return write_report(session, stream, format=format)
