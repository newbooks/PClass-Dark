"""Consume the working classifier CLIs without importing their dependencies."""

import json
import logging
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

from .artifacts import identity
from .external_evaluation_scoring import project_dark, project_pclass
from .external_evaluation_store import (
    checked,
    read,
    record,
    sync_dir,
    sync_file,
    write_json,
)
from .external_evaluation_types import (
    EvaluationInferenceError,
    EvaluationIntegrityError,
    EvaluationResourceError,
    SavedDarkQuery,
    SavedPClassQuery,
    digest_id,
)

LOG = logging.getLogger(__name__)
FORMAT = "command-output-v2"


def input_rows(state, input_id):
    with sqlite3.connect(state / "inputs" / input_id / "queries.sqlite") as db:
        for ordinal, qid, sequence, length, digest in db.execute(
            "SELECT ordinal,query_id,sequence,length,sequence_sha256 "
            "FROM queries ORDER BY ordinal"
        ):
            yield {
                "ordinal": ordinal,
                "id": qid,
                "sequence": sequence,
                "length": length,
                "sequence_sha256": digest,
            }


def command_resource(engine, config, execution):
    executable = shutil.which("pclass" if engine == "pclass" else "pclass-dark")
    if executable is None:
        raise ValueError(f"Missing classifier command on PATH: {engine}")
    args = [str(Path(executable).resolve()), "infer"]
    options = (
        [("server_url", "--server-url"), ("artifact_dir", "--artifact-dir")]
        if engine == "pclass"
        else [("library_dir", "--library-dir"), ("device", "--device")]
    )
    for name, flag in options:
        value = getattr(config, name)
        if value is not None:
            args += [flag, str(value)]
    contract = {
        "format": FORMAT,
        "engine": engine,
        "argv": args,
        "working_directory": str(Path.cwd()),
        "executable": record(executable),
        "run_tag": execution.run_tag,
        "environment": {
            k: os.environ[k]
            for k in (
                "PCLASS_SERVER_URL",
                "PCLASS_QUERY_TIMEOUT_SECONDS",
                "CUDA_VISIBLE_DEVICES",
            )
            if k in os.environ
        },
    }
    return {"engine": engine, "contract": contract, "contract_id": identity(contract)}


def plan_chunks(rows, engine, count):
    chunk, size = [], 0
    for row in rows:
        # Conservative JSON encoding of each sequence; extra 4 KiB covers options.
        row_size = (
            len(
                json.dumps(
                    {"id": row["id"], "sequence": row["sequence"]}, ensure_ascii=True
                ).encode()
            )
            + 2
        )
        if chunk and (
            len(chunk) >= count
            or (engine == "pclass" and size + row_size > 8 * 2**20 - 4096)
        ):
            yield chunk
            chunk, size = [], 0
        if engine == "pclass" and row_size > 8 * 2**20 - 4096:
            raise EvaluationResourceError(
                f"PClass singleton exceeds request bound: {row['id']}"
            )
        chunk.append(row)
        size += row_size
    if chunk:
        yield chunk


def _pairs(pairs):
    obj = {}
    for k, v in pairs:
        if k in obj:
            raise ValueError(f"Duplicate JSON key: {k}")
        obj[k] = v
    return obj


def load_report(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(
            f,
            object_pairs_hook=_pairs,
            parse_constant=lambda v: (_ for _ in ()).throw(
                ValueError(f"Nonfinite JSON: {v}")
            ),
        )


def report_results(path, engine, members):
    payload = load_report(path)
    if not isinstance(payload, dict):
        raise ValueError("Inference report must be a JSON object")
    if engine == "dark" and payload.get("schema") != "pclass-dark.infer-report.v1":
        raise ValueError("Unsupported Dark report schema")
    rows = payload.get("results" if engine == "pclass" else "queries")
    if not isinstance(rows, list):
        raise ValueError("Missing inference result array")
    expected = {m["id"]: m for m in members}
    seen = set()
    for result in rows:
        projected = (
            project_pclass(result) if engine == "pclass" else project_dark(result)
        )
        qid = projected["id" if engine == "pclass" else "query_id"]
        if qid not in expected or qid in seen:
            raise ValueError(f"Duplicate/unexpected {engine} query: {qid}")
        seen.add(qid)
        if engine == "dark" and (
            result.get("length") != expected[qid]["length"]
            or result.get("sequence_sha256") != expected[qid]["sequence_sha256"]
        ):
            raise ValueError(f"Dark query sequence mismatch: {qid}")
    if seen != set(expected):
        raise ValueError(f"Missing {engine} query results")
    return rows


def _stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_child(resource, fasta, execution, attempt):
    output, error = attempt / "response.json", attempt / "stderr.log"
    argv = resource["contract"]["argv"] + ["--fasta", str(fasta), "--format", "json"]
    start = time.monotonic()
    with output.open("wb") as stdout, error.open("wb") as stderr:
        process = subprocess.Popen(
            argv, stdout=stdout, stderr=stderr, start_new_session=True
        )
        try:
            while process.poll() is None:
                if time.monotonic() - start > execution.process_timeout:
                    raise EvaluationInferenceError(
                        f"{resource['engine']} timeout; log {error}"
                    )
                if output.stat().st_size > execution.max_raw_chunk_mib * 2**20:
                    raise EvaluationResourceError(
                        f"Raw stdout cap exceeded; log {error}"
                    )
                time.sleep(0.1)
            if output.stat().st_size > execution.max_raw_chunk_mib * 2**20:
                raise EvaluationResourceError(f"Raw stdout cap exceeded; log {error}")
            if process.returncode:
                with error.open("r", errors="replace") as f:
                    first = f.readline(4096).strip()
                size_rejected = resource["engine"] == "pclass" and (
                    process.returncode == 4
                    and first.startswith("REQUEST_TOO_LARGE:")
                    or process.returncode == 7
                    and re.match(r"SERVICE_ERROR: Service returned HTTP 413\b", first)
                )
                if size_rejected:
                    return None
                raise EvaluationInferenceError(
                    f"{resource['engine']} exited {process.returncode}; log {error}"
                )
        except BaseException:
            _stop(process)
            raise
    return output


def raw_receipt(state, engine, chunk_id):
    digest_id(chunk_id)
    root = state / "raw" / engine / chunk_id
    receipt = read(root / "receipt.json")
    if (
        receipt.get("format") != FORMAT
        or receipt["engine"] != engine
        or identity(receipt["contract"]) != receipt["contract_id"]
        or identity(
            {"contract_id": receipt["contract_id"], "members": receipt["members"]}
        )
        != chunk_id
    ):
        raise EvaluationIntegrityError("Raw receipt identity mismatch")
    checked(root / "response.json", receipt["response"])
    return root, receipt


def collect_generation(state, cohort, resource, count, execution):
    engine = resource["engine"]
    key = {
        "format": FORMAT,
        "engine": engine,
        "input_id": cohort["input_id"],
        "contract_id": resource["contract_id"],
        "chunk_count": count,
    }
    gid = identity(key)
    target = state / "generations" / engine / gid
    if (target / "manifest.json").exists():
        return verify_generation(state, engine, gid, cohort["input_id"])
    target.mkdir(parents=True, exist_ok=True)
    chunks = []
    completed = 0

    def execute(rows):
        nonlocal completed
        members = [{k: v for k, v in r.items() if k != "sequence"} for r in rows]
        cid = identity({"contract_id": resource["contract_id"], "members": members})
        root = state / "raw" / engine / cid
        split = target / (cid + ".split.json")
        if root.exists():
            _, receipt = raw_receipt(state, engine, cid)
        elif split.exists():
            if read(split).get("members") != members:
                raise EvaluationIntegrityError("Invalid saved split")
            middle = len(rows) // 2
            if not middle:
                raise EvaluationIntegrityError("Invalid singleton split")
            execute(rows[:middle])
            execute(rows[middle:])
            return
        else:
            attempt = state / "attempts" / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            fasta = attempt / "queries.fasta"
            with fasta.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(f">{row['id']}\n{row['sequence']}\n")
            output = run_child(resource, fasta, execution, attempt)
            if output is None:
                if len(rows) == 1:
                    raise EvaluationResourceError("PClass rejected a singleton request")
                write_json(split, {"schema_version": 1, "members": members})
                middle = len(rows) // 2
                execute(rows[:middle])
                execute(rows[middle:])
                return
            report_results(output, engine, members)
            receipt = {
                "schema_version": 1,
                "format": FORMAT,
                "engine": engine,
                "contract": resource["contract"],
                "contract_id": resource["contract_id"],
                "members": members,
                "response": record(output),
            }
            staged = attempt / "commit"
            staged.mkdir()
            os.replace(output, staged / "response.json")
            sync_file(staged / "response.json")
            write_json(staged / "receipt.json", receipt)
            sync_dir(staged)
            root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, root)
            sync_dir(root.parent)
        chunks.append({"id": cid, "receipt": record(root / "receipt.json")})
        completed += len(rows)
        LOG.info(
            "%s: completed %s queries (%s); %.1f%% finished",
            engine,
            f"{len(rows):,}",
            cid[:12],
            100 * completed / cohort["query_count"],
        )

    for rows in plan_chunks(input_rows(state, cohort["input_id"]), engine, count):
        execute(rows)
    manifest = {"schema_version": 1, **key, "id": gid, "chunks": chunks}
    write_json(target / "manifest.json", manifest)
    return verify_generation(state, engine, gid, cohort["input_id"])


def verify_generation(state, engine, gid, input_id):
    digest_id(gid)
    manifest = read(state / "generations" / engine / gid / "manifest.json")
    if manifest.get("format") != FORMAT:
        raise EvaluationIntegrityError(
            "Legacy evaluator generation; collect a command-output run first"
        )
    key = {
        k: manifest[k]
        for k in ("format", "engine", "input_id", "contract_id", "chunk_count")
    }
    if (
        identity(key) != gid
        or manifest.get("id") != gid
        or manifest["engine"] != engine
        or manifest["input_id"] != input_id
    ):
        raise EvaluationIntegrityError("Invalid engine generation")
    expected = {
        r["id"]: {k: v for k, v in r.items() if k != "sequence"}
        for r in input_rows(state, input_id)
    }
    seen = set()
    for chunk in manifest["chunks"]:
        root, receipt = raw_receipt(state, engine, chunk["id"])
        checked(root / "receipt.json", chunk["receipt"])
        if receipt["contract_id"] != manifest["contract_id"]:
            raise EvaluationIntegrityError("Chunk contract mismatch")
        for member in receipt["members"]:
            qid = member["id"]
            if qid in seen or expected.get(qid) != member:
                raise EvaluationIntegrityError("Invalid generation query coverage")
            seen.add(qid)
    if seen != set(expected):
        raise EvaluationIntegrityError("Incomplete generation")
    return manifest


def saved_queries(state, engine, manifest):
    for chunk in manifest["chunks"]:
        root, receipt = raw_receipt(state, engine, chunk["id"])
        members = {m["id"]: m for m in receipt["members"]}
        for result in report_results(
            root / "response.json", engine, receipt["members"]
        ):
            qid = result["id" if engine == "pclass" else "query_id"]
            member = members[qid]
            cls = SavedPClassQuery if engine == "pclass" else SavedDarkQuery
            yield cls(qid, member["length"], member["sequence_sha256"], result)
