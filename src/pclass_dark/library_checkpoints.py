"""Content-bound private library checkpoints; malformed state is quarantined."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .artifacts import (
    IntegrityError,
    file_record,
    identity,
    read_json,
    safe_file,
    verify_files,
    write_json,
)
from .embeddings import _save_array

LOG = logging.getLogger(__name__)


class Checkpoints:
    def __init__(self, path, task_id):
        self.path, self.task_id = Path(path), task_id
        if self.path.is_symlink():
            raise IntegrityError("Symlink checkpoint root")
        self.path.mkdir(parents=True, exist_ok=True)
        self.reused = 0

    def read(self, name, dependency, validate=lambda payload: None):
        path = safe_file(self.path, name + ".json")
        if not path.exists():
            return None
        receipt = None
        try:
            receipt = read_json(path)
            if not isinstance(receipt, dict) or not isinstance(
                receipt.get("payload"), dict
            ):
                raise IntegrityError("Checkpoint receipt/payload must be objects")
            if (
                receipt["schema"] != "pclass-dark.library-checkpoint.v1"
                or receipt["task_id"] != self.task_id
                or receipt["dependency"] != dependency
                or receipt["payload_id"] != identity(receipt["payload"])
            ):
                raise IntegrityError("Stale checkpoint identity/dependency")
            if receipt["files"]:
                verify_files(self.path, receipt["files"])
            arrays = {}
            if set(receipt["files"]) != {a["file"] for a in receipt["arrays"].values()}:
                raise IntegrityError("Unbound checkpoint array files")
            for key, metadata in receipt["arrays"].items():
                if not isinstance(metadata, dict):
                    raise IntegrityError("Invalid checkpoint array metadata")
                array = np.load(
                    safe_file(self.path, metadata["file"]),
                    mmap_mode="r",
                    allow_pickle=False,
                )
                if (
                    list(array.shape) != metadata["shape"]
                    or array.dtype.str != metadata["dtype"]
                ):
                    raise IntegrityError("Checkpoint array shape/dtype mismatch")
                if array.dtype.str != "<f8":
                    raise IntegrityError("Invalid checkpoint numeric dtype")
                flat = array.reshape(-1)
                for begin in range(0, flat.size, 1048576):
                    if not np.isfinite(flat[begin : begin + 1048576]).all():
                        raise IntegrityError("Nonfinite checkpoint numeric payload")
                arrays[key] = array
            validate(receipt["payload"])
            self.reused += 1
            return receipt["payload"], arrays
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            IndexError,
        ) as exc:
            LOG.warning("Quarantining library checkpoint %s: %s", name, exc)
            quarantine = self.path / ".quarantine"
            quarantine.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            members = [path]
            if isinstance(receipt, dict) and isinstance(receipt.get("files"), dict):
                for member in receipt["files"]:
                    try:
                        members.append(safe_file(self.path, member))
                    except IntegrityError:
                        continue
            for member in members:
                if member.exists() and not member.is_symlink():
                    os.replace(member, quarantine / f"{member.name}-{stamp}")
            return None

    def write(self, name, dependency, payload, arrays=None, external_arrays=None):
        files, metadata = {}, {}
        for key, array in (arrays or {}).items():
            filename = name + "." + key + ".npy"
            path = safe_file(self.path, filename)
            _save_array(path, np.asarray(array, dtype="<f8"))
            files[filename] = file_record(path)
            metadata[key] = {
                "file": filename,
                "shape": list(array.shape),
                "dtype": "<f8",
            }
        for key, path in (external_arrays or {}).items():
            path = safe_file(self.path, Path(path).name)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            files[path.name] = file_record(path)
            metadata[key] = {
                "file": path.name,
                "shape": list(array.shape),
                "dtype": array.dtype.str,
            }
        receipt = {
            "schema": "pclass-dark.library-checkpoint.v1",
            "task_id": self.task_id,
            "dependency": dependency,
            "payload": payload,
            "payload_id": identity(payload),
            "files": files,
            "arrays": metadata,
        }
        write_json(safe_file(self.path, name + ".json"), receipt)
