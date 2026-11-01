"""Public records and failures for external sequence collection (SPEC-0014)."""

from dataclasses import dataclass
from pathlib import Path


class ExternalDownloadError(RuntimeError):
    """A download failed after bounded retries."""


class ExternalSnapshotError(RuntimeError):
    """Sources, responses or cached generations are incompatible."""


class ExternalRecoveryError(RuntimeError):
    """Publication recovery cannot safely proceed."""


@dataclass(frozen=True)
class StageCount:
    stage: str
    annotation_rows: int | None
    accessions: int | None
    sequences: int | None


@dataclass(frozen=True)
class ExternalSequenceResult:
    fasta_path: Path
    report_path: Path
    manifest_path: Path
    run_id: str
    counts: tuple[StageCount, ...]
