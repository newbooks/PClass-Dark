"""Immutable public cleanup records and actionable failures (no filesystem work)."""

from dataclasses import asdict, dataclass

from .artifacts import IntegrityError


class CleanupStalePlanError(IntegrityError):
    """Selectors, artifact ownership or file inventories changed after planning."""


class CleanupRecoveryRequiredError(RuntimeError):
    """A managed root has an unfinished cleanup; recover with the same roots."""


class CleanupExecutionError(RuntimeError):
    """Cleanup failed; an already published production result remains valid."""

    def __init__(self, message, *, result=None, production_result=None):
        super().__init__(message)
        self.result = result
        self.production_result = production_result


class Record:
    def to_dict(self):
        """Return a detached, JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class CleanupEntry(Record):
    root: str
    path: str
    kind: str
    artifact_id: str
    disposition: str
    reason: str
    dependencies: tuple[str, ...]
    tree_sha256: str | None
    files: int
    logical_bytes: int
    allocated_bytes_estimate: int | None

    @property
    def key(self):
        return self.root + ":" + self.path


@dataclass(frozen=True)
class CleanupPlan(Record):
    schema: str
    data_dir: str
    library_dir: str
    snapshot: tuple[tuple[str, str, str], ...]
    entries: tuple[CleanupEntry, ...]
    inventory_sha256: str
    plan_id: str


@dataclass(frozen=True)
class CleanupResult(Record):
    schema: str
    plan_id: str
    status: str
    removed: tuple[str, ...]
    protected: tuple[tuple[str, str], ...]
    failed: tuple[tuple[str, str], ...]
    logical_bytes_removed: int
    allocated_bytes_estimate: int | None
    recovery_required: bool
