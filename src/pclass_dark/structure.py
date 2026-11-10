"""Frozen sequence-to-residue exposure and secondary-structure prediction.

Adapted from ProStruct library/esm2/enrichment, revision
fa8fa0da7a9f2ecb1ab096f979b3936dd2409d54.
Imports are computation-free. See SPEC-0020 for scientific and artifact contracts.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

from .artifacts import IntegrityError
from .structural_models import verify_structural_models

CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")
WINDOW_LENGTH = 1021
MINIMUM_OVERLAP = 100
LOG = logging.getLogger(__name__)


def structural_windows(length: int) -> tuple[tuple[int, int], ...]:
    """Return one-based inclusive ProStruct windows covering every residue."""
    if type(length) is not int or length < 1:
        raise ValueError("Sequence length must be a positive integer")
    if length <= WINDOW_LENGTH:
        return ((1, length),)
    count = math.ceil((length - MINIMUM_OVERLAP) / (WINDOW_LENGTH - MINIMUM_OVERLAP))
    starts = [1 + i * (length - WINDOW_LENGTH) // (count - 1) for i in range(count)]
    return tuple((start, start + WINDOW_LENGTH - 1) for start in starts)


def _owned_ranges(windows):
    # Midpoint between consecutive window centers, equality to earlier start.
    boundaries = [
        (sum(a) + sum(b)) // 4 for a, b in zip(windows, windows[1:], strict=False)
    ]
    return tuple(
        (
            1 if i == 0 else boundaries[i - 1] + 1,
            windows[-1][1] if i == len(windows) - 1 else boundaries[i],
        )
        for i in range(len(windows))
    )


def _network(torch, task):
    return torch.nn.Sequential(
        torch.nn.Linear(640, 512),
        torch.nn.GELU(approximate="none"),
        torch.nn.Dropout(0.1 if task == "rsa" else 0.2),
        torch.nn.Linear(512, 128),
        torch.nn.GELU(approximate="none"),
        torch.nn.Linear(128, 1 if task == "rsa" else 3),
    )


@dataclass(frozen=True)
class StructuralPrediction:
    """Original sequence order; unknown positions have np.nan RSA and 'NA' Q3.

    ``rsa`` is raw ALL contextual exposure fraction, intentionally unclipped.
    Arrays are read-only; indices correspond to sequence positions 1..length.
    """

    sequence: str
    rsa: np.ndarray
    q3: np.ndarray
    model_id: str
    device: str


class StructuralHeads:
    """Load frozen heads on CPU; predict on caller-supplied residue states.

    Requires the embeddings extra. ValueError/IntegrityError report invalid data
    or artifacts; RuntimeError reports missing optional dependencies/device errors.
    No ESM weights are loaded by this class.
    """

    def __init__(self, model_dir=None):
        self.bundle = verify_structural_models(model_dir)
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "Structural prediction needs pip install 'pclass-dark[embeddings]'"
            ) from exc
        self.torch = torch
        tensors = load_file(str(self.bundle.path / "models.safetensors"), device="cpu")
        self.models = {}
        self.normalization = {}
        expected = set()
        # Constructing a module must not alter the application's random sequence.
        with threadpool_limits(limits=8), torch.random.fork_rng(devices=[]):
            for task in ("rsa", "q3"):
                model = _network(torch, task)
                state = model.state_dict()
                for name, template in state.items():
                    key = f"{task}.{name}"
                    expected.add(key)
                    tensor = tensors.get(key)
                    if (
                        tensor is None
                        or tensor.shape != template.shape
                        or tensor.dtype != torch.float32
                        or not torch.isfinite(tensor).all()
                    ):
                        raise IntegrityError(f"Invalid structural model tensor: {key}")
                model.load_state_dict({k: tensors[f"{task}.{k}"] for k in state})
                self.models[task] = model.eval()
                values = {}
                for name, dtype in (
                    ("mean", torch.float64),
                    ("scale", torch.float64),
                    ("constant", torch.bool),
                ):
                    key = f"{task}.{name}"
                    expected.add(key)
                    value = tensors.get(key)
                    if value is None or value.shape != (640,) or value.dtype != dtype:
                        raise IntegrityError(f"Invalid structural normalization: {key}")
                    if name != "constant" and (
                        not torch.isfinite(value).all()
                        or name == "scale"
                        and not (value > 0).all()
                    ):
                        raise IntegrityError(f"Invalid structural normalization: {key}")
                    values[name] = value.numpy()
                self.normalization[task] = values
        if set(tensors) != expected:
            raise IntegrityError("Unexpected structural model tensors")
        self.device = "cpu"

    def to(self, device):
        """Move both heads; callers implement any auto-device fallback."""
        try:
            for model in self.models.values():
                model.to(device)
        except (RuntimeError, ValueError, AssertionError):
            for model in self.models.values():
                model.to("cpu")
            self.device = "cpu"
            raise
        self.device = str(device)

    def predict(self, states, *, batch_size=4096, cpu_threads=8):
        """Return (raw ALL-CEF float32, C/E/H strings) for an (N,640) matrix.

        Empty matrices are allowed. Standardization uses frozen task-specific
        float64 statistics before casting to float32, as in ProStruct.
        """
        if type(batch_size) is not int or not 1 <= batch_size <= 65536:
            raise ValueError("Head batch_size must be between 1 and 65536")
        if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8:
            raise ValueError("cpu_threads must be between 1 and 8")
        states = np.asarray(states)
        if (
            states.ndim != 2
            or states.shape[1] != 640
            or states.dtype != np.float32
            or not np.isfinite(states).all()
        ):
            raise ValueError("Expected finite float32 residue states shaped (N,640)")
        rsa = np.empty(len(states), dtype=np.float32)
        q3 = np.empty(len(states), dtype="<U2")
        torch = self.torch
        with (
            threadpool_limits(limits=cpu_threads),
            torch.inference_mode(),
            torch.autocast(device_type=torch.device(self.device).type, enabled=False),
        ):
            for start in range(0, len(states), batch_size):
                stop = min(start + batch_size, len(states))
                for task, model in self.models.items():
                    norm = self.normalization[task]
                    values = (
                        (states[start:stop].astype(np.float64) - norm["mean"])
                        / norm["scale"]
                    ).astype(np.float32)
                    values[:, norm["constant"]] = 0
                    if not np.isfinite(values).all():
                        raise ValueError("Structural standardization overflow")
                    result = (
                        model(torch.from_numpy(values).to(self.device)).cpu().numpy()
                    )
                    if not np.isfinite(result).all():
                        raise RuntimeError("Nonfinite structural predictions")
                    if task == "rsa":
                        rsa[start:stop] = result[:, 0]
                    else:
                        q3[start:stop] = np.array(["C", "E", "H"])[
                            result.argmax(axis=1)
                        ]
        return rsa, q3


class StructuralPredictor:
    """Reusable sequence predictor sharing one ESM instance and the two heads.

    ESM is loaded lazily on the first query with canonical residues. Use
    device='cpu' for explicit CPU and offline=True to forbid checkpoint downloads.
    Memory: bounded window batches plus O(sequence length) output arrays.
    Instances own mutable model state and are not intended for concurrent calls.
    """

    def __init__(
        self,
        model_dir=None,
        *,
        device="auto",
        offline=False,
        window_batch_size=8,
        cpu_threads=8,
        _backend=None,
    ):
        if not isinstance(device, str) or not re.fullmatch(
            r"auto|cpu|cuda(?::\d+)?", device
        ):
            raise ValueError("Device must be auto, cpu, cuda or cuda:<index>")
        if type(offline) is not bool:
            raise ValueError("offline must be boolean")
        if type(window_batch_size) is not int or not 1 <= window_batch_size <= 8:
            raise ValueError("window_batch_size must be between 1 and 8")
        if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8:
            raise ValueError("cpu_threads must be between 1 and 8")
        self.heads = StructuralHeads(model_dir)
        self.requested_device = device
        self.offline = offline
        self.window_batch_size = window_batch_size
        self.cpu_threads = cpu_threads
        self._embedder = _backend
        if _backend is not None:
            from .setup_data import MODEL, REVISION

            if (
                _backend.contract["model"] != MODEL
                or _backend.contract["revision"] != REVISION
            ):
                raise ValueError("Borrowed structural backend contract mismatch")
            try:
                self.heads.to(_backend.device)
            except (RuntimeError, ValueError, AssertionError):
                if device != "auto" or not _backend.device.startswith("cuda"):
                    raise
                _backend.model.to("cpu")
                _backend.device = "cpu"
                self.heads.to("cpu")
                LOG.warning("Borrowed structural heads fell back to CPU")

    def _runtime(self):
        if self._embedder is None:
            from .embeddings import EsmBackend
            from .setup_data import MODEL, REVISION

            backend = EsmBackend(
                model=MODEL,
                revision=REVISION,
                offline=self.offline,
                device=self.requested_device,
                seed=42,
            )
            try:
                self.heads.to(backend.device)
            except (RuntimeError, ValueError, AssertionError) as exc:
                if self.requested_device != "auto" or not backend.device.startswith(
                    "cuda"
                ):
                    raise RuntimeError(
                        "Cannot initialize structural heads on device"
                    ) from exc
                LOG.warning("Structural head device failed; using CPU: %s", exc)
                backend.model.to("cpu")
                backend.device = "cpu"
                self.heads.to("cpu")
            self._embedder = backend
            LOG.info("Structural prediction device: %s", backend.device)
        return self._embedder

    def prepare(self) -> dict[str, object]:
        """Initialize once, returning a fresh JSON-compatible runtime fingerprint.

        Captured under the same numerical thread limits as predict; no dummy
        forward pass is used. This may resolve/download the pinned checkpoint
        unless offline=True. Initialization/device errors follow predict().
        """
        import copy
        import platform

        with threadpool_limits(limits=self.cpu_threads):
            backend = self._runtime()
            torch = self.heads.torch
            actual = str(next(backend.model.parameters()).device)
            if actual.startswith("cuda"):
                name = torch.cuda.get_device_name(actual)
                capability = list(torch.cuda.get_device_capability(actual))
            else:
                name = platform.processor() or platform.machine()
                capability = None
            return copy.deepcopy(
                {
                    "device": actual,
                    "embedding_contract": backend.contract,
                    "torch_version": str(torch.__version__),
                    "cuda_version": torch.version.cuda,
                    "hip_version": torch.version.hip,
                    "device_name": name,
                    "device_capability": capability,
                    "numerical_threads": torch.get_num_threads(),
                    "deterministic_algorithms": (
                        torch.are_deterministic_algorithms_enabled()
                    ),
                    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                    "seed": 42,
                    "window_batch_size": self.window_batch_size,
                    "python_version": platform.python_version(),
                    "numpy_version": np.__version__,
                }
            )

    def predict(self, sequence: str) -> StructuralPrediction:
        """Predict original positions, preserving and masking noncanonical symbols.

        Accept a nonempty ASCII sequence of letters or *?-. Uppercase is
        normalized; whitespace is rejected. No accession/cluster label is needed.
        """
        if not isinstance(sequence, str) or not re.fullmatch(
            r"[A-Za-z*?\-]+", sequence
        ):
            raise ValueError(
                "Expected a nonempty ASCII protein sequence without whitespace"
            )
        sequence = sequence.upper()
        known = np.fromiter((r in CANONICAL for r in sequence), dtype=bool)
        rsa = np.full(len(sequence), np.nan, dtype=np.float32)
        q3 = np.full(len(sequence), "NA", dtype="<U2")
        if known.any():
            with threadpool_limits(limits=self.cpu_threads):
                backend = self._runtime()
                context = "".join(r if r in CANONICAL else "X" for r in sequence)
                windows = structural_windows(len(sequence))
                ranges = _owned_ranges(windows)
                for start in range(0, len(windows), self.window_batch_size):
                    batch = windows[start : start + self.window_batch_size]
                    blocks = backend.residue_states(
                        [context[a - 1 : b] for a, b in batch]
                    )
                    if len(blocks) != len(batch):
                        raise IntegrityError("ESM returned an invalid window count")
                    for ordinal, ((a, b), states) in enumerate(
                        zip(batch, blocks, strict=True), start
                    ):
                        if np.shape(states) != (b - a + 1, 640):
                            raise IntegrityError("ESM returned invalid residue states")
                        left, right = ranges[ordinal]
                        raw, labels = self.heads.predict(
                            states[left - a : right - a + 1],
                            cpu_threads=self.cpu_threads,
                        )
                        rsa[left - 1 : right] = raw
                        q3[left - 1 : right] = labels
            rsa[~known] = np.nan
            q3[~known] = "NA"
        rsa.flags.writeable = q3.flags.writeable = False
        return StructuralPrediction(
            sequence,
            rsa,
            q3,
            self.heads.bundle.model_id,
            self._embedder.device if self._embedder is not None else "cpu",
        )
