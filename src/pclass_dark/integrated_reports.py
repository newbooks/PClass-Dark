"""Recoverable no-overwrite publication for combined JSON/Markdown reports."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock

from .artifacts import IntegrityError, file_record, read_json, sync_dir, write_json


@contextmanager
def _artifact_errors():
    try:
        yield
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise IntegrityError(f"Malformed report journal: {exc}") from exc


@contextmanager
def report_outputs(prefix=None):
    """Yield staged JSON/Markdown/text streams; commit only after successful exit.

    If prefix is None no persistent report files/locks are created. Consumers copy
    staged stdout after this context exits successfully, using read_staged().
    """
    if prefix is None:
        with tempfile.TemporaryDirectory(prefix="pclass-integrated-") as tmp:
            staged = StagedReports(Path(tmp))
            try:
                yield staged
            finally:
                staged.close()
        return
    prefix = Path(prefix).absolute()
    parent = prefix.parent
    for p in (prefix, *prefix.parents):
        if p.is_symlink():
            raise ValueError("Report symlink path rejected")
    parent.mkdir(parents=True, exist_ok=True)
    targets = [
        Path(str(prefix) + s)
        for s in (
            "-classification.json",
            "-interpretation.md",
            "-classification.complete.json",
        )
    ]
    lock = Path(str(prefix) + "-classification.lock")
    journal = Path(str(prefix) + "-classification.journal.json")
    if lock.is_symlink() or journal.is_symlink():
        raise ValueError("Report control symlink rejected")
    with _artifact_errors(), FileLock(str(lock), timeout=0):
        if journal.exists():
            recovery = read_json(journal)
            stage = parent / recovery["stage"]
            if (
                stage.parent != parent
                or not stage.name.startswith(".integrated-")
                or stage.is_symlink()
            ):
                raise IntegrityError("Invalid report recovery stage")
            if recovery.get("schema") != "pclass-dark.integrated-report-journal.v1":
                raise IntegrityError("Invalid report journal")
            if not stage.is_dir() or (stage.stat().st_dev, stage.stat().st_ino) != (
                recovery["stage_device"],
                recovery["stage_inode"],
            ):
                raise IntegrityError("Report recovery stage ownership mismatch")
            allowed = {"report.json", "report.md", "report.txt", "complete.json"}
            if any(
                p.is_symlink() or not p.is_file() or p.name not in allowed
                for p in stage.iterdir()
            ):
                raise IntegrityError("Unknown report recovery stage content")
            for name, record in recovery["members"].items():
                staged_file = stage / record["staged_name"]
                if record["staged_name"] not in allowed or not staged_file.is_file():
                    raise IntegrityError("Missing owned recovery payload")
                stat = staged_file.stat()
                if (stat.st_dev, stat.st_ino) != (
                    record["dev"],
                    record["ino"],
                ) or file_record(staged_file) != record["file"]:
                    raise IntegrityError("Changed owned recovery payload")
                target = parent / name
                if name not in {p.name for p in targets}:
                    raise IntegrityError("Unowned report recovery member")
                if target.exists() or target.is_symlink():
                    stat = target.lstat()
                    if (
                        target.is_symlink()
                        or (stat.st_dev, stat.st_ino) != (record["dev"], record["ino"])
                        or file_record(target) != record["file"]
                    ):
                        raise IntegrityError(f"Report recovery conflict: {target}")
            complete = targets[-1].exists()
            if not complete:
                for name in recovery["members"]:
                    (parent / name).unlink(missing_ok=True)
            shutil.rmtree(stage, ignore_errors=True)
            journal.unlink()
        if any(p.exists() or p.is_symlink() for p in targets):
            raise ValueError(
                "Classification reports already exist; choose --report-prefix"
            )
        stage = Path(tempfile.mkdtemp(prefix=".integrated-", dir=parent))
        staged = StagedReports(stage)
        try:
            yield staged
            staged.close()
            receipt = stage / "complete.json"
            write_json(
                receipt,
                dict(
                    schema="pclass-dark.integrated-report-completion.v1",
                    transaction_id=uuid.uuid4().hex,
                    query_count=staged.query_count,
                    files={
                        targets[0].name: file_record(stage / "report.json"),
                        targets[1].name: file_record(stage / "report.md"),
                    },
                ),
            )
            sources = [stage / "report.json", stage / "report.md", receipt]
            members = {
                t.name: dict(
                    dev=s.stat().st_dev,
                    ino=s.stat().st_ino,
                    file=file_record(s),
                    staged_name=s.name,
                )
                for s, t in zip(sources, targets, strict=True)
            }
            write_json(
                journal,
                dict(
                    schema="pclass-dark.integrated-report-journal.v1",
                    stage=stage.name,
                    stage_device=stage.stat().st_dev,
                    stage_inode=stage.stat().st_ino,
                    members=members,
                ),
            )
            for source, target in zip(sources, targets, strict=True):
                with source.open("rb") as f:
                    os.fsync(f.fileno())
                os.link(source, target)
                sync_dir(parent)
            journal.unlink()
            sync_dir(parent)
        except BaseException:
            staged.close()
            if journal.exists():
                record = read_json(journal)
                owned = []
                conflict = False
                for name, member in record["members"].items():
                    target = parent / name
                    if target.exists() or target.is_symlink():
                        stat = target.lstat()
                        if (
                            target.is_symlink()
                            or (stat.st_dev, stat.st_ino)
                            != (member["dev"], member["ino"])
                            or file_record(target) != member["file"]
                        ):
                            conflict = True
                            break
                        owned.append(target)
                if not conflict:
                    for target in owned:
                        target.unlink()
                    journal.unlink()
                    sync_dir(parent)
            if not journal.exists():
                shutil.rmtree(stage, ignore_errors=True)
            raise
        finally:
            staged.close()
            if not journal.exists():
                shutil.rmtree(stage, ignore_errors=True)


class StagedReports:
    def __init__(self, path):
        self.path = path
        self.json = (path / "report.json").open("w+", encoding="utf-8")
        self.markdown = (path / "report.md").open("w+", encoding="utf-8")
        self.text = (path / "report.txt").open("w+", encoding="utf-8")
        self.stdout = None
        self.query_count = 0

    def prepare_stdout(self, format):
        """Detach staged stdout into an anonymous stream surviving publication."""
        from .inference_records import MAX_TEMP

        payload_bytes = sum(
            p.stat().st_size for p in self.path.iterdir() if p.is_file()
        )
        chosen = {"json": self.json, "markdown": self.markdown, "text": self.text}[
            format
        ]
        chosen.flush()
        chosen.seek(0, 2)
        if payload_bytes + chosen.tell() + 8192 > MAX_TEMP:
            raise ValueError("2 GiB report staging quota exceeded")
        self.stdout = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        stream = {"json": self.json, "markdown": self.markdown, "text": self.text}[
            format
        ]
        stream.flush()
        stream.seek(0)
        shutil.copyfileobj(stream, self.stdout)
        self.stdout.seek(0)
        return self.stdout

    def close(self):
        for stream in (self.json, self.markdown, self.text):
            stream.close()
