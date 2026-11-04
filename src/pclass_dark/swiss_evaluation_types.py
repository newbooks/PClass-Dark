"""Native Swiss-Prot evaluation contracts (SPEC-0017); no runtime side effects."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from .external_evaluation_types import ExecutionConfig, integer, number

FORMAT = "swiss-native-v1"
SCHEMAS = {
    "pclass": "pclass-dark.swiss-pclass-neighbors.v1",
    "dark": "pclass-dark.swiss-dark-evidence.v1",
}


@dataclass(frozen=True)
class SwissPClassConfig:
    batch_size: int = 512
    server_url: str = field(
        default_factory=lambda: os.environ.get(
            "PCLASS_SERVER_URL", "http://127.0.0.1:8000"
        )
    )
    timeout_seconds: float = 60

    def __post_init__(self):
        integer("PClass batch_size", self.batch_size, 1, 1024)
        number("HTTP timeout", self.timeout_seconds, 0, positive=True)
        if not isinstance(self.server_url, str):
            raise ValueError("server_url must be text")
        url = urlsplit(self.server_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Invalid PClass server URL")
        _ = url.port


@dataclass(frozen=True)
class SwissDarkConfig:
    chunk_size: int = 512
    library_dir: str | Path | None = None
    device: str = "auto"
    batch_size: int = 8
    score_batch_size: int = 4096
    memory_mib: int = 1024
    workers: int = 8
    embedding_cache: str | Path | None = None
    prepared_dir: str | Path | None = None
    offline: bool = False

    def __post_init__(self):
        for name, low, high in (
            ("chunk_size", 1, 1024),
            ("batch_size", 1, 64),
            ("score_batch_size", 1, 65536),
            ("memory_mib", 128, None),
            ("workers", 1, 8),
        ):
            integer(name, getattr(self, name), low, high)
        if not isinstance(self.device, str) or not re.fullmatch(
            r"auto|cpu|cuda(?::[0-9]+)?", self.device
        ):
            raise ValueError("Invalid device")
        if (self.embedding_cache is None) != (self.prepared_dir is None):
            raise ValueError(
                "Embedding cache and prepared directory are required together"
            )
        if type(self.offline) is not bool:
            raise ValueError("offline must be boolean")


@dataclass(frozen=True)
class SwissExecutionConfig(ExecutionConfig):
    """Same resource caps/run tag as external evaluation; separate public type."""
