"""Query contracts and shared deterministic accepted-window reduction."""

from dataclasses import asdict, dataclass, field

import numpy as np

from .artifacts import IntegrityError
from .embeddings import ALPHABET
from .reference_data import sequence_id
from .setup_data import WINDOW_LENGTH, window_records

MAX_SEQUENCE = 1_000_000
MAX_QUERIES = 100_000
MAX_INPUT = 2**30
MAX_HEADER = 4096
MAX_TEMP = 2 * 2**30


class InferenceError(RuntimeError):
    """Backend execution failed; no query is silently skipped."""


class InferenceResourceError(RuntimeError):
    """A query batch exceeds a documented resource/admission limit."""


class Record:
    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class QuerySequence(Record):
    query_id: str
    sequence: str


@dataclass(frozen=True)
class WindowInfo(Record):
    window_id: str
    kind: str
    start: int
    end: int
    window_sha256: str


@dataclass(frozen=True)
class QueryWindow(Record):
    info: WindowInfo
    sequence: str = field(repr=False)


@dataclass(frozen=True)
class QueryWindows(Record):
    query_index: int
    query_id: str
    length: int
    sequence_sha256: str
    windows: tuple[QueryWindow, ...]


@dataclass(frozen=True)
class WindowMatch(Record):
    window: WindowInfo
    background_tail_fraction: float
    cluster_percentile_le: float
    accepted: bool = field(default=True, init=False)


@dataclass(frozen=True)
class ClusterMatch(Record):
    cluster_id: str
    background_tail_fraction: float
    cluster_percentile_le: float
    C_n: int
    B_n: int
    representative: WindowInfo
    supporting_windows: tuple[WindowMatch, ...]
    accepted: bool = field(default=True, init=False)


@dataclass(frozen=True)
class QueryResult(Record):
    query_index: int
    query_id: str
    length: int
    sequence_sha256: str
    windows: tuple[WindowInfo, ...]
    status: str
    matches: tuple[ClusterMatch, ...]
    schema: str = field(default="pclass-dark.infer-query.v1", init=False)


def make_query_windows(query, *, query_index=0):
    """Validate one query and use the exact reference terminal-window algorithm.

    Raises ValueError for invalid identifiers/residues and InferenceResourceError
    for sequence limits. No model, filesystem or worker is started.
    """
    if not isinstance(query, QuerySequence):
        raise ValueError("Expected QuerySequence")
    if type(query_index) is not int or query_index < 0:
        raise ValueError("query_index must be a nonnegative integer")
    name, sequence = query.query_id, query.sequence
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 256
        or not name.isprintable()
        or any(c.isspace() for c in name)
    ):
        raise ValueError("Query ID must be printable, without whitespace, length 1–256")
    if not isinstance(sequence, str):
        raise ValueError(f"Sequence must be text: {name}")
    sequence = sequence.strip(" \t\r\n\v\f")
    if len(sequence) > MAX_SEQUENCE:
        raise InferenceResourceError(f"Sequence {name} exceeds {MAX_SEQUENCE} residues")
    if not sequence or not sequence.isascii() or set(sequence) - ALPHABET:
        raise ValueError(
            f"Invalid/empty protein sequence: {name}; use uppercase reference residues"
        )
    windows = tuple(
        QueryWindow(WindowInfo(wid, kind, start, end, sequence_id(seq)), seq)
        for wid, seq, kind, start, end in window_records(name, sequence)
    )
    return QueryWindows(
        query_index, name, len(sequence), sequence_id(sequence), windows
    )


def validate_windows(queries, vectors):
    if not queries:
        raise ValueError("Expected at least one QueryWindows")
    indices, names = [], set()
    for q in queries:
        if (
            not isinstance(q, QueryWindows)
            or type(q.query_index) is not int
            or q.query_index < 0
        ):
            raise ValueError("Invalid query window record/index")
        if (
            not isinstance(q.query_id, str)
            or not q.query_id
            or len(q.query_id) > 256
            or not q.query_id.isprintable()
            or any(c.isspace() for c in q.query_id)
            or type(q.length) is not int
            or not 1 <= q.length <= MAX_SEQUENCE
        ):
            raise ValueError("Invalid query ID/length")
        if q.query_id in names or q.query_index in indices:
            raise ValueError("Duplicate query ID/index")
        names.add(q.query_id)
        indices.append(q.query_index)
        import re

        if not isinstance(q.sequence_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", q.sequence_sha256
        ):
            raise ValueError("Invalid full query hash")
        if q.length <= WINDOW_LENGTH:
            expected = [(q.query_id, "full", 1, q.length)]
        else:
            expected = [
                ("N-" + q.query_id, "N", 1, WINDOW_LENGTH),
                ("C-" + q.query_id, "C", q.length - WINDOW_LENGTH + 1, q.length),
            ]
        if not isinstance(q.windows, tuple) or len(q.windows) != len(expected):
            raise ValueError("Invalid reference query window count")
        for w, record in zip(q.windows, expected, strict=True):
            if (
                not isinstance(w, QueryWindow)
                or not isinstance(w.info, WindowInfo)
                or not isinstance(w.sequence, str)
                or type(w.info.start) is not int
                or type(w.info.end) is not int
            ):
                raise ValueError("Invalid query window record")
            info = w.info
            if (
                (info.window_id, info.kind, info.start, info.end) != record
                or len(w.sequence) != info.end - info.start + 1
                or not w.sequence
                or set(w.sequence) - ALPHABET
                or sequence_id(w.sequence) != info.window_sha256
            ):
                raise ValueError("Invalid query window coordinates, residues or hash")
    if indices != sorted(indices):
        raise ValueError("Query indices must be strictly increasing")
    vectors = np.asarray(vectors)
    if (
        vectors.shape != (sum(len(q.windows) for q in queries), 640)
        or vectors.dtype.kind != "f"
        or not np.isfinite(vectors).all()
    ):
        raise ValueError(
            "Expected finite floating vectors matching flattened (windows,640)"
        )
    return vectors


def combine_matches(index, name, length, digest, windows, rows, metadata):
    """One reduction rule shared by in-memory and SQLite-backed inference.

    rows are (cluster_id, window_index, B, C); they contain accepted windows only.
    """
    grouped = {}
    for cluster, ordinal, b, c in rows:
        b, c = float(b), float(c)
        if not (np.isfinite(b) and np.isfinite(c) and 0 < b <= 1 and 0 <= c <= 1):
            raise IntegrityError("Invalid accepted B/C evidence")
        if type(ordinal) is not int or not 0 <= ordinal < len(windows):
            raise IntegrityError("Evidence window index mismatch")
        bucket = grouped.setdefault(cluster, {})
        if ordinal in bucket:
            raise IntegrityError("Duplicate accepted window evidence")
        bucket[ordinal] = WindowMatch(windows[ordinal], b, c)
    matches = []
    for cluster, bucket in grouped.items():
        support = tuple(v for _, v in sorted(bucket.items()))
        representative = min(
            support,
            key=lambda w: (
                w.background_tail_fraction,
                w.window.start,
                w.window.end,
                w.window.window_id,
            ),
        )
        meta = metadata[cluster]
        matches.append(
            ClusterMatch(
                cluster,
                representative.background_tail_fraction,
                representative.cluster_percentile_le,
                int(meta["C_n"]),
                int(meta["B_n"]),
                representative.window,
                support,
            )
        )
    matches.sort(key=lambda m: (m.background_tail_fraction, m.cluster_id))
    return QueryResult(
        index,
        name,
        length,
        digest,
        tuple(windows),
        "matches" if matches else "no_accepted_matches",
        tuple(matches),
    )
