"""Typed interfaces for SPEC-0015; imports perform no evaluation work."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HEADERS = (
    "Query_id",
    "PClass_go",
    "Embedding_d",
    "Pclass_match",
    "Dark_go",
    "Dark_Bscore",
    "Dark_match",
)
GO_PATTERN = re.compile(r"GO:[0-9]{7}\Z")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYBOUXZ")


class EvaluationInferenceError(RuntimeError):
    """An inference child failed; completed observations are preserved."""


class EvaluationSnapshotError(RuntimeError):
    """A runtime identity changed during collection."""


class EvaluationIntegrityError(RuntimeError):
    """A committed observation is missing, incompatible or corrupt."""


class EvaluationResourceError(RuntimeError):
    """An input or child exceeds an admission/resource limit."""


class EvaluationRecoveryError(RuntimeError):
    """Publication cannot be safely recovered."""


def integer(name, value, low, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be an integer in {low}..{high or 'unbounded'}")


def number(name, value, low, high=None, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < low
        or (positive and value == low)
        or (high is not None and value > high)
    ):
        raise ValueError(f"Invalid {name}: {value!r}")
    return float(value)


def go_set(values, *, nonempty=False):
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise ValueError("GO set must be a collection of identifiers")
    if any(not isinstance(v, str) or not GO_PATTERN.fullmatch(v) for v in values):
        raise ValueError("Malformed GO identifier")
    result = tuple(sorted(set(values)))
    if nonempty and not result:
        raise ValueError("GO truth/composition must be nonempty")
    return result


def query_id(value):
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 256
        or not value.isprintable()
        or any(c.isspace() for c in value)
    ):
        raise ValueError(
            "Query ID must be printable, without whitespace, length 1..256"
        )
    return value


def digest_id(value):
    if not isinstance(value, str) or not DIGEST_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid SHA-256 identity: {value!r}")
    return value


@dataclass(frozen=True)
class PClassConfig:
    batch_size: int = 512
    server_url: str | None = None
    artifact_dir: str | Path | None = None

    def __post_init__(self):
        integer("pclass batch size", self.batch_size, 1, 1024)
        if self.server_url is not None and not isinstance(self.server_url, str):
            raise ValueError("server_url must be text")


@dataclass(frozen=True)
class DarkConfig:
    chunk_size: int = 512
    library_dir: str | Path | None = None
    device: str | None = None

    def __post_init__(self):
        integer("dark chunk size", self.chunk_size, 1, 1024)
        if self.device is not None and not re.fullmatch(
            r"auto|cpu|cuda(?::[0-9]+)?", self.device
        ):
            raise ValueError("Invalid Dark device")


@dataclass(frozen=True)
class ScoringConfig:
    go_graph: str | Path | None = None
    cluster_go_map: str | Path | None = None
    workers: int = 8

    def __post_init__(self):
        integer("score_workers", self.workers, 1, 8)


@dataclass(frozen=True)
class ExecutionConfig:
    process_timeout: float = 3600
    max_raw_chunk_mib: int = 128
    run_tag: str = "default"

    def __post_init__(self):
        number("process_timeout", self.process_timeout, 0, positive=True)
        integer("max_raw_chunk_mib", self.max_raw_chunk_mib, 1)
        if not isinstance(self.run_tag, str) or not self.run_tag:
            raise ValueError("run_tag must be nonempty text")


@dataclass(frozen=True)
class EvaluationQuery:
    ordinal: int
    query_id: str
    length: int
    sequence_sha256: str
    truth_go: tuple[str, ...]


@dataclass(frozen=True)
class SavedPClassQuery:
    query_id: str
    length: int
    sequence_sha256: str
    result: Mapping[str, Any]


@dataclass(frozen=True)
class SavedDarkQuery:
    query_id: str
    length: int
    sequence_sha256: str
    result: Mapping[str, Any]


@dataclass(frozen=True)
class ClusterGoMap:
    clusters: Mapping[str, Mapping[str, Any]]
    reference: Mapping[str, Any] = field(default_factory=dict)

    def terms(self, cluster_id):
        if cluster_id not in self.clusters:
            raise ValueError(f"No profunc GO mapping for Dark cluster: {cluster_id}")
        return go_set(self.clusters[cluster_id]["go_mf_terms"])


@dataclass(frozen=True)
class ComparisonRow:
    query_id: str
    pclass_go: tuple[str, ...]
    embedding_d: float | None
    pclass_match: float | None
    dark_go: tuple[str, ...]
    dark_bscore: float | None
    dark_match: float | None

    def cells(self):
        def numeric(value):
            return "NA" if value is None else repr(float(value))

        return (
            self.query_id,
            "|".join(self.pclass_go) or "NA",
            numeric(self.embedding_d),
            numeric(self.pclass_match),
            "|".join(self.dark_go) or "NA",
            numeric(self.dark_bscore),
            numeric(self.dark_match),
        )


@dataclass(frozen=True)
class EvaluationResult:
    tsv_path: Path
    score_manifest_path: Path
    score_id: str
    input_id: str
    truth_id: str
    pclass_generation: str
    dark_generation: str
    query_count: int


@dataclass(frozen=True)
class EvaluationPreflight:
    input_id: str
    truth_id: str
    query_count: int
    commands: Mapping[str, Any]
    graph_id: str
    mapping_id: str
