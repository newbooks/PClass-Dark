"""Explicit spawned native-worker entry point; no classifier imports on import."""

import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import traceback
from contextlib import ExitStack
from pathlib import Path

from .artifacts import identity
from .external_evaluation_store import record
from .swiss_evaluation_types import SCHEMAS, SwissDarkConfig


def fingerprint(modules):
    return {
        name: record(Path(importlib.import_module(name).__file__))["sha256"]
        for name in modules
    }


def pclass_contract():
    """Inspect installed native APIs without calling the service or infer CLI."""
    version = importlib.metadata.version("pclass")
    if not re.match(r"^0\.3\.", version):
        raise ValueError(f"Unsupported PClass version {version}; require 0.3.x")
    client = importlib.import_module("pclass.query.client")
    schemas = importlib.import_module("pclass.serving.schemas")
    if not callable(client.request_neighbors) or not callable(schemas.SequenceInput):
        raise ValueError("Unsupported PClass native API")
    return {
        "version": version,
        "code": fingerprint(
            ["pclass.query.client", "pclass.serving.schemas", "pclass.query.errors"]
        ),
    }


def plain(value):
    """Convert frozen runtime mappings/tuples to JSON-compatible containers."""
    from collections.abc import Mapping

    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def worker_main(connection, engine, config, state, execution, mapping_clusters):
    os.setsid()
    task_id = 0
    config = plain(config)
    state = Path(state)
    with ExitStack() as stack:
        try:
            # Keep tracebacks out of protocol messages and preserve attempt logs.
            log = stack.enter_context((state / "worker-stderr.log").open("a"))
            os.dup2(log.fileno(), 2)
            if engine == "pclass":
                source = pclass_contract()
                runtime = None
            else:
                from .inference_runtime import DarkInferenceRuntime

                SwissDarkConfig(**config)
                options = {k: v for k, v in config.items() if k != "chunk_size"}
                runtime = stack.enter_context(
                    DarkInferenceRuntime(**options, _index_dir=state / "vector-index")
                )
                unknown = set(runtime.metadata) - set(mapping_clusters)
                if unknown:
                    raise ValueError(
                        f"Dark clusters absent from GO map: {sorted(unknown)[:20]}"
                    )
                source = {
                    "library": plain(runtime.source),
                    "cache": runtime.cache.source if runtime.cache else None,
                }
            modules = [
                "pclass_dark.swiss_evaluation_worker",
                "pclass_dark.swiss_evaluation_collect",
                "pclass_dark.swiss_evaluation_types",
                "pclass_dark.external_evaluation_scoring",
            ]
            if engine == "dark":
                modules += [
                    "pclass_dark.inference_runtime",
                    "pclass_dark.inference",
                    "pclass_dark.inference_records",
                    "pclass_dark.library_runtime",
                    "pclass_dark.relative_scores",
                    "pclass_dark.signature_metrics",
                    "pclass_dark.inference_backend",
                    "pclass_dark.embeddings",
                ]
            code = fingerprint(modules)
            versions = {}
            for package in (
                ("numpy", "torch", "transformers")
                if engine == "dark"
                else ("requests", "pydantic")
            ):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = None
            contract = {
                "engine": engine,
                "format": SCHEMAS[engine],
                "config": config,
                "source": source,
                "code": code,
                "package_versions": versions,
                "run_tag": execution["run_tag"],
                "environment": {
                    k: os.environ.get(k)
                    for k in (
                        "CUDA_VISIBLE_DEVICES",
                        "HIP_VISIBLE_DEVICES",
                        "ROCR_VISIBLE_DEVICES",
                        "OMP_NUM_THREADS",
                    )
                },
            }
            details = (
                {"service_availability": "unverified"}
                if runtime is None
                else {
                    "scoring_workspace_bytes": runtime.required_workspace,
                    "max_input_bytes": 2**30,
                    "max_temp_bytes": 2 * 2**30,
                    "embedding_dependencies_installed": all(
                        importlib.util.find_spec(n) is not None
                        for n in ("torch", "transformers")
                    ),
                    "device": "unverified_until_forward",
                }
            )
            contract_id = identity(contract)
            connection.send(
                {
                    "protocol": 1,
                    "kind": "ready",
                    "task_id": 0,
                    "contract": contract,
                    "contract_id": contract_id,
                    "details": details,
                }
            )
            while True:
                message = connection.recv()
                task_id = message["task_id"]
                if message.get("protocol") != 1:
                    raise ValueError("Unknown worker protocol")
                if message["kind"] == "close":
                    return
                if message["kind"] not in {"chunk", "inspect"}:
                    raise ValueError("Invalid task kind")
                # Task spool is a parent-owned, private file, bounded to one chunk.
                rows = json.loads(Path(message["input"]).read_text())
                if runtime is not None:
                    from .inference_records import QuerySequence

                if message["kind"] == "inspect":
                    if runtime is None:
                        raise ValueError("Only Dark has window preflight")
                    details = runtime.inspect_queries(
                        QuerySequence(r["id"], r["sequence"]) for r in rows
                    )
                    connection.send(
                        {
                            "protocol": 1,
                            "kind": "done",
                            "task_id": task_id,
                            "details": details,
                        }
                    )
                    continue
                attempt = Path(message["attempt"])
                members = [
                    {k: r[k] for k in ("ordinal", "id", "length", "sequence_sha256")}
                    for r in rows
                ]
                path = attempt / "response.json"
                if engine == "pclass":
                    payload = neighbors(rows, config)
                    if payload is None:
                        connection.send(
                            {
                                "protocol": 1,
                                "kind": "done",
                                "task_id": task_id,
                                "split": True,
                            }
                        )
                        continue
                    envelope = {
                        "schema": SCHEMAS[engine],
                        "source": contract,
                        "execution": {"service_device": "unverified"},
                        "members": members,
                        "payload": payload,
                    }
                    with path.open("w") as stream:
                        json.dump(
                            envelope, stream, allow_nan=False, separators=(",", ":")
                        )
                else:
                    with (
                        runtime.infer_chunk(
                            QuerySequence(r["id"], r["sequence"]) for r in rows
                        ) as session,
                        path.open("w") as stream,
                    ):
                        envelope = {
                            "schema": SCHEMAS[engine],
                            "source": contract,
                            "execution": plain(session.execution),
                            "members": members,
                        }
                        prefix = json.dumps(
                            envelope, allow_nan=False, separators=(",", ":")
                        )
                        stream.write(prefix[:-1] + ',"payload":[')
                        for i, result in enumerate(session):
                            if i:
                                stream.write(",")
                            json.dump(
                                result.to_dict(),
                                stream,
                                allow_nan=False,
                                separators=(",", ":"),
                            )
                        stream.write("]}")
                connection.send(
                    {
                        "protocol": 1,
                        "kind": "done",
                        "task_id": task_id,
                        "path": str(path),
                        "record": record(path),
                    }
                )
        except EOFError:
            return
        except BaseException as exc:
            traceback.print_exc()
            try:
                connection.send(
                    {
                        "protocol": 1,
                        "kind": "error",
                        "task_id": task_id,
                        "category": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            except (BrokenPipeError, EOFError, OSError):
                pass
        finally:
            connection.close()


def neighbors(rows, config):
    from pclass.query.client import request_neighbors
    from pclass.query.errors import QueryRequestTooLargeError, QueryServiceError
    from pclass.serving.schemas import SequenceInput

    try:
        result = request_neighbors(
            [SequenceInput(id=r["id"], sequence=r["sequence"]) for r in rows],
            server_url=config["server_url"],
            timeout_seconds=config["timeout_seconds"],
        )
    except QueryRequestTooLargeError:
        return None
    except QueryServiceError as exc:
        if re.match(r"Service returned HTTP 413\b", str(exc)):
            return None
        raise
    return result.model_dump(mode="json")
