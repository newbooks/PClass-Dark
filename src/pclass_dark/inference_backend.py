"""Scoped model execution; Torch and model downloads occur only on entry."""

import logging
import os
import threading
from contextlib import contextmanager

from threadpoolctl import threadpool_limits

from .embeddings import EsmBackend
from .inference_records import InferenceError

LOG = logging.getLogger(__name__)
_EXECUTION_LOCK = threading.RLock()


def cpu_budget(requested):
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(1, min(requested, available))


@contextmanager
def backend_context(contract, *, device, workers, offline):
    """One model, bounded native threads, precision restoration even on failure."""
    try:
        import torch
    except ImportError as exc:
        raise InferenceError(
            "Inference needs pip install 'pclass-dark[embeddings]'"
        ) from exc
    with _EXECUTION_LOCK:
        previous = torch.get_num_threads()
        settings = []
        for owner, attr in (
            (torch.backends.cuda.matmul, "allow_tf32"),
            (torch.backends.cudnn, "allow_tf32"),
        ):
            settings.append((owner, attr, getattr(owner, attr)))
        budget = cpu_budget(workers)
        backend = None
        try:
            torch.set_num_threads(budget)
            with threadpool_limits(limits=budget):
                available = False
                if device == "auto":
                    try:
                        available = torch.cuda.is_available()
                    except RuntimeError:
                        LOG.warning("CUDA availability failed; using CPU")
                requested = "cpu" if device == "auto" and not available else device
                try:
                    backend = EsmBackend(
                        model=contract["model"],
                        revision=contract["revision"],
                        offline=offline,
                        device=requested,
                        seed=42,
                    )
                except RuntimeError as exc:
                    raise InferenceError(f"Model initialization failed: {exc}") from exc
                actual = backend.device
                if actual == "cuda":
                    actual = f"cuda:{torch.cuda.current_device()}"
                fallback = None
                if device == "auto" and actual == "cpu":
                    fallback = (
                        "cuda_initialization_failed"
                        if available
                        else "cuda_unavailable"
                    )
                    LOG.info("Embedding fallback: %s", fallback)
                execution = {
                    "requested_device": device,
                    "device": actual,
                    "fallback_reason": fallback,
                    "workers_requested": workers,
                    "workers_used": budget,
                    "model": contract["model"],
                    "revision": contract["revision"],
                    "seed": 42,
                    "inference_dtype": "float32",
                    "torch_version": backend.contract["torch_version"],
                    "transformers_version": backend.contract["transformers_version"],
                }
                LOG.info(
                    "Inference embedding device: %s; CPU thread budget: %s",
                    actual,
                    budget,
                )
                yield backend, execution
        finally:
            backend = None
            for owner, attr, value in settings:
                setattr(owner, attr, value)
            torch.set_num_threads(previous)
