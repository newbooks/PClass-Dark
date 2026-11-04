"""Supervised resident workers and immutable native-evidence checkpoints."""

import json
import logging
import multiprocessing
import os
import signal
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from . import external_evaluation_collect as common
from . import external_evaluation_store as store
from .artifacts import identity
from .external_evaluation_scoring import project_dark, project_pclass
from .external_evaluation_types import (
    EvaluationInferenceError,
    EvaluationIntegrityError,
    EvaluationResourceError,
    SavedDarkQuery,
    SavedPClassQuery,
    digest_id,
    number,
)
from .swiss_evaluation_types import FORMAT, SCHEMAS

LOG = logging.getLogger(__name__)


class NativeWorker:
    """One outstanding task; parent alone validates and commits evidence."""

    def __init__(self, engine, config, state, execution, mapping_clusters):
        self.engine, self.config, self.state = engine, config, Path(state)
        self.execution, self.mapping_clusters = execution, mapping_clusters
        self.process = self.connection = None
        self.task_id = 0

    def __enter__(self):
        from .swiss_evaluation_worker import worker_main

        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=worker_main,
            args=(
                child,
                self.engine,
                asdict(self.config),
                str(self.state),
                asdict(self.execution),
                list(self.mapping_clusters),
            ),
        )
        self.process.start()
        child.close()
        try:
            ready = self._receive("ready")
            self.contract, self.contract_id = ready["contract"], ready["contract_id"]
            self.details = ready.get("details", {})
            if identity(self.contract) != self.contract_id:
                raise EvaluationIntegrityError("Worker contract mismatch")
            return self
        except BaseException:
            self.close()
            raise

    def _receive(self, kind, attempt=None):
        deadline = time.monotonic() + self.execution.process_timeout
        while time.monotonic() < deadline:
            if attempt is not None:
                response = attempt / "response.json"
                if response.exists() and response.stat().st_size > (
                    self.execution.max_raw_chunk_mib * 2**20
                ):
                    raise EvaluationResourceError(
                        "Native evidence cap exceeded; reduce chunk size"
                    )
            if self.connection.poll(0.1):
                try:
                    reply = self.connection.recv()
                except EOFError as exc:
                    raise EvaluationInferenceError(
                        "Worker closed protocol unexpectedly"
                    ) from exc
                if (
                    not isinstance(reply, dict)
                    or reply.get("protocol") != 1
                    or reply.get("task_id") != self.task_id
                ):
                    raise EvaluationIntegrityError("Malformed native worker reply")
                if reply.get("kind") == "error":
                    raise EvaluationInferenceError(
                        f"{self.engine}: {reply.get('category')}: "
                        f"{reply.get('message')}; "
                        f"log {self.state / 'worker-stderr.log'}"
                    )
                if reply.get("kind") != kind:
                    raise EvaluationIntegrityError("Unexpected worker state")
                return reply
            if not self.process.is_alive():
                raise EvaluationInferenceError(f"{self.engine} worker crashed")
        raise EvaluationResourceError(f"{self.engine} worker timed out")

    def task(self, rows, attempt, *, inspect_only=False):
        self.task_id += 1
        path = attempt / "input.json"
        path.write_text(json.dumps(rows))
        self.connection.send(
            {
                "protocol": 1,
                "task_id": self.task_id,
                "kind": "inspect" if inspect_only else "chunk",
                "input": str(path),
                "attempt": str(attempt),
            }
        )
        reply = self._receive("done", attempt)
        if not inspect_only and not reply.get("split"):
            output = attempt / "response.json"
            if reply.get("path") != str(output):
                raise EvaluationIntegrityError(
                    "Worker returned unexpected evidence path"
                )
            store.checked(output, reply["record"])
            if output.stat().st_size > self.execution.max_raw_chunk_mib * 2**20:
                raise EvaluationResourceError(
                    "Native evidence cap exceeded; reduce chunk size"
                )
        return reply

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                try:
                    self.task_id += 1
                    self.connection.send(
                        {"protocol": 1, "kind": "close", "task_id": self.task_id}
                    )
                except (BrokenPipeError, OSError):
                    pass
                self.process.join(0.5)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    if self.process.is_alive():
                        self.process.terminate()
                self.process.join(1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __exit__(self, *exc):
        self.close()


def members(rows):
    return [
        {k: r[k] for k in ("ordinal", "id", "length", "sequence_sha256")} for r in rows
    ]


def evidence(path, engine, rows, contract):
    """Validate exact coverage and reconstruct Dark reductions independently."""
    envelope = common.load_report(path)
    expected = members(rows)
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"schema", "source", "execution", "members", "payload"}
        or envelope["schema"] != SCHEMAS[engine]
        or envelope["source"] != contract
        or envelope["members"] != expected
        or not isinstance(envelope["execution"], dict)
    ):
        raise EvaluationIntegrityError("Native evidence envelope mismatch")
    payload = envelope["payload"]
    if engine == "pclass" and not isinstance(payload, dict):
        raise EvaluationIntegrityError("PClass payload must be a neighbor response")
    results = payload.get("results") if engine == "pclass" else payload
    if not isinstance(results, list):
        raise EvaluationIntegrityError("Missing native query array")
    by_id = {r["id"]: r for r in rows}
    seen = set()
    for result in results:
        projected = (
            project_pclass(result) if engine == "pclass" else project_dark(result)
        )
        qid = projected["id" if engine == "pclass" else "query_id"]
        if qid not in by_id or qid in seen:
            raise EvaluationIntegrityError("Native evidence query coverage mismatch")
        seen.add(qid)
        row = by_id[qid]
        if engine == "pclass":
            for neighbor in result["neighbors"]:
                number("PClass confidence", neighbor.get("confidence"), 0, 1)
            normalization = result.get("normalization")
            if (
                normalization is not None
                and normalization.get("normalized_length") != row["length"]
            ):
                raise EvaluationIntegrityError("PClass normalization/sequence mismatch")
            accessions = [n["accession"] for n in projected["neighbors"]]
            if len(accessions) != len(set(accessions)):
                raise EvaluationIntegrityError("Duplicate PClass neighbors")
        else:
            from .inference_records import (
                QuerySequence,
                combine_matches,
                make_query_windows,
            )

            i = next(i for i, r in enumerate(rows) if r["id"] == qid)
            q = make_query_windows(QuerySequence(qid, row["sequence"]), query_index=i)
            accepted, metadata = [], {}
            for m in result["matches"]:
                metadata[m["cluster_id"]] = {"C_n": m["C_n"], "B_n": m["B_n"]}
                if (
                    type(m["C_n"]) is not int
                    or type(m["B_n"]) is not int
                    or min(m["C_n"], m["B_n"]) < 1
                ):
                    raise EvaluationIntegrityError("Invalid Dark sample counts")
                for support in m["supporting_windows"]:
                    w = support["window"]
                    j = next(
                        (
                            j
                            for j, window in enumerate(q.windows)
                            if window.info.to_dict() == w
                        ),
                        None,
                    )
                    if j is None or support.get("accepted") is not True:
                        raise EvaluationIntegrityError("Dark support window mismatch")
                    accepted.append(
                        (
                            m["cluster_id"],
                            j,
                            support["background_tail_fraction"],
                            support["cluster_percentile_le"],
                        )
                    )
            rebuilt = combine_matches(
                i,
                qid,
                q.length,
                q.sequence_sha256,
                tuple(w.info for w in q.windows),
                accepted,
                metadata,
            ).to_dict()
            if json.loads(json.dumps(rebuilt)) != result:
                raise EvaluationIntegrityError(
                    "Dark evidence reduction/sequence mismatch"
                )
    if seen != set(by_id):
        raise EvaluationIntegrityError("Incomplete native query coverage")
    return results


def raw_receipt(state, engine, cid):
    root = state / "raw" / engine / digest_id(cid)
    receipt = store.read(root / "receipt.json")
    key = {k: receipt[k] for k in ("format", "contract_id", "members")}
    if (
        receipt["format"] != FORMAT
        or identity(key) != cid
        or identity(receipt["contract"]) != receipt["contract_id"]
    ):
        raise EvaluationIntegrityError("Invalid native receipt identity")
    store.checked(root / "response.json", receipt["response"])
    return root, receipt


def collect_generation(state, cohort, worker, count):
    engine = worker.engine
    key = {
        "format": FORMAT,
        "engine": engine,
        "input_id": cohort["input_id"],
        "contract_id": worker.contract_id,
        "chunk_count": count,
    }
    gid = identity(key)
    target = state / "generations" / engine / gid
    if (target / "manifest.json").exists():
        return verify_generation(state, engine, gid, cohort["input_id"])
    target.mkdir(parents=True, exist_ok=True)
    chunks, completed = [], 0

    def execute(rows):
        nonlocal completed
        key = {
            "format": FORMAT,
            "contract_id": worker.contract_id,
            "members": members(rows),
        }
        cid = identity(key)
        root = state / "raw" / engine / cid
        split = target / (cid + ".split.json")
        if root.exists():
            _, receipt = raw_receipt(state, engine, cid)
        elif split.exists():
            if store.read(split)["members"] != members(rows) or len(rows) < 2:
                raise EvaluationIntegrityError("Invalid native split plan")
            middle = len(rows) // 2
            execute(rows[:middle])
            execute(rows[middle:])
            return
        else:
            attempt = state / "attempts" / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            reply = worker.task(rows, attempt)
            if reply.get("split"):
                if engine != "pclass" or len(rows) < 2:
                    raise EvaluationResourceError("Rejected native singleton")
                store.write_json(split, {"schema_version": 1, "members": members(rows)})
                middle = len(rows) // 2
                execute(rows[:middle])
                execute(rows[middle:])
                return
            evidence(attempt / "response.json", engine, rows, worker.contract)
            receipt = {
                "schema_version": 1,
                **key,
                "contract": worker.contract,
                "response": store.record(attempt / "response.json"),
            }
            staged = attempt / "commit"
            staged.mkdir()
            os.replace(attempt / "response.json", staged / "response.json")
            store.write_json(staged / "receipt.json", receipt)
            # Raw format has no directory manifest: publish files then rename once.
            for p in staged.iterdir():
                store.sync_file(p)
            store.sync_dir(staged)
            root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, root)
            store.sync_dir(root.parent)
        chunks.append({"id": cid, "receipt": store.record(root / "receipt.json")})
        completed += len(rows)
        LOG.info(
            "%s: completed %s queries (%s); %.1f%% finished",
            engine,
            f"{len(rows):,}",
            cid[:12],
            100 * completed / cohort["query_count"],
        )

    for rows in common.plan_chunks(
        common.input_rows(state, cohort["input_id"]), engine, count
    ):
        execute(rows)
    store.write_json(
        target / "manifest.json",
        {"schema_version": 1, **key, "id": gid, "chunks": chunks},
    )
    return verify_generation(state, engine, gid, cohort["input_id"])


def verify_generation(state, engine, gid, input_id):
    manifest = store.read(
        state / "generations" / engine / digest_id(gid) / "manifest.json"
    )
    key = {
        k: manifest[k]
        for k in ("format", "engine", "input_id", "contract_id", "chunk_count")
    }
    if (
        key["format"] != FORMAT
        or key["engine"] != engine
        or key["input_id"] != input_id
        or identity(key) != gid
        or manifest["id"] != gid
    ):
        raise EvaluationIntegrityError("Native generation identity mismatch")
    expected = {
        r["id"]: m for r in common.input_rows(state, input_id) for m in members([r])
    }
    seen = set()
    for chunk in manifest["chunks"]:
        root, receipt = raw_receipt(state, engine, chunk["id"])
        store.checked(root / "receipt.json", chunk["receipt"])
        if receipt["contract_id"] != key["contract_id"]:
            raise EvaluationIntegrityError("Native chunk contract mismatch")
        for m in receipt["members"]:
            if m["id"] in seen or expected.get(m["id"]) != m:
                raise EvaluationIntegrityError("Native generation coverage mismatch")
            seen.add(m["id"])
    if seen != set(expected):
        raise EvaluationIntegrityError("Native generation is incomplete")
    return manifest


def saved_queries(state, engine, manifest):
    # Read only each chunk's sequences from the immutable input DB.
    import sqlite3

    with sqlite3.connect(
        state / "inputs" / manifest["input_id"] / "queries.sqlite"
    ) as db:
        for chunk in manifest["chunks"]:
            root, receipt = raw_receipt(state, engine, chunk["id"])
            rows = []
            for m in receipt["members"]:
                row = db.execute(
                    "SELECT sequence FROM queries WHERE query_id=?", (m["id"],)
                ).fetchone()
                if row is None:
                    raise EvaluationIntegrityError("Missing saved query")
                rows.append({**m, "sequence": row[0]})
            by_id = {r["id"]: r for r in rows}
            for result in evidence(
                root / "response.json", engine, rows, receipt["contract"]
            ):
                qid = result["id" if engine == "pclass" else "query_id"]
                m = by_id[qid]
                cls = SavedPClassQuery if engine == "pclass" else SavedDarkQuery
                yield cls(qid, m["length"], m["sequence_sha256"], result)


def progress_summary(state, cohort, worker, count):
    """Count resumable receipt coverage without contacting a classifier."""
    key = {
        "format": FORMAT,
        "engine": worker.engine,
        "input_id": cohort["input_id"],
        "contract_id": worker.contract_id,
        "chunk_count": count,
    }
    target = state / "generations" / worker.engine / identity(key)
    cached = pending = 0

    def inspect(rows):
        nonlocal cached, pending
        cid = identity(
            {
                "format": FORMAT,
                "contract_id": worker.contract_id,
                "members": members(rows),
            }
        )
        root = state / "raw" / worker.engine / cid
        split = target / (cid + ".split.json")
        if root.exists():
            raw_receipt(state, worker.engine, cid)
            cached += len(rows)
        elif split.exists():
            if len(rows) < 2 or store.read(split)["members"] != members(rows):
                raise EvaluationIntegrityError("Invalid saved native split")
            middle = len(rows) // 2
            inspect(rows[:middle])
            inspect(rows[middle:])
        else:
            pending += len(rows)

    for rows in common.plan_chunks(
        common.input_rows(state, cohort["input_id"]), worker.engine, count
    ):
        inspect(rows)
    return {"cached_queries": cached, "pending_queries": pending}
