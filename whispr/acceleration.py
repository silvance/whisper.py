"""Choose the compute device for transcription: CPU, or an NVIDIA GPU if present.

CPU is the supported baseline - every deployed machine has one, results do not
depend on the choice, and nothing here is required for the application to work.
A GPU is an optional speed-up: on a field laptop with a CUDA card, a long
recording finishes in a fraction of the time.

Detection is entirely local. ``ctranslate2.get_cuda_device_count()`` asks the
already-bundled inference library what it can see; it loads no models, contacts
nothing, and returns 0 rather than raising when there is no driver. A build with
CPU-only CTranslate2 wheels simply reports no GPU, which is the truth for that
bundle.

The failure mode that matters is a run that dies half-way because CUDA was
selected on a machine that cannot serve it. So an explicit GPU request is
checked before the run starts and reported in plain language, and
:func:`fallback_to_cpu` gives the caller a CPU device to retry with rather than
losing the job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

# Selection modes, in the order the GUI offers them.
MODE_AUTO = "auto"
MODE_CUDA = "cuda"
MODE_CPU = "cpu"

MODE_LABELS: List[Tuple[str, str]] = [
    (MODE_AUTO, "Auto (use a GPU if one is available)"),
    (MODE_CUDA, "NVIDIA GPU (CUDA)"),
    (MODE_CPU, "CPU"),
]
DEFAULT_MODE = MODE_AUTO

# Quantization per device. int8 is the right default on CPU; float16 is what a
# CUDA card is good at. Neither changes what the model is, only how it computes.
CPU_COMPUTE_TYPE = "int8"
CUDA_COMPUTE_TYPE = "float16"


@dataclass(frozen=True)
class Device:
    """A resolved compute device, ready to hand to faster-whisper."""

    device: str
    compute_type: str
    label: str
    # Why this device was chosen, or why the requested one was not used.
    note: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.device == "cuda"

    def describe(self) -> str:
        return f"{self.label} ({self.device}/{self.compute_type})"


def cpu_device(note: str = "") -> Device:
    """The always-available baseline."""
    return Device("cpu", CPU_COMPUTE_TYPE, "CPU", note)


def cuda_device_count() -> int:
    """How many CUDA devices CTranslate2 can see. 0 when there are none.

    Purely a local capability query - no model is loaded and nothing is
    downloaded. Any failure (no library, no driver, a CPU-only build) counts as
    no GPU rather than an error, because that is what it means operationally.
    """
    try:
        import ctranslate2
    except ImportError:
        return 0
    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 - a broken/absent driver means "no GPU"
        return 0


def cuda_available() -> bool:
    return cuda_device_count() > 0


def describe_hardware() -> str:
    """One line for the self-test and status log."""
    count = cuda_device_count()
    if count <= 0:
        return "CPU only (no CUDA device detected)"
    plural = "device" if count == 1 else "devices"
    return f"CUDA available ({count} {plural}); CPU also supported"


def resolve(mode: str = DEFAULT_MODE) -> Device:
    """Turn a selection mode into the device that will actually be used.

    ``auto`` prefers a GPU when one is present and silently uses the CPU when
    not. An explicit ``cuda`` request on a machine without one does *not*
    silently downgrade - the returned device is the CPU with a note saying why,
    so the caller can tell the operator instead of quietly ignoring the setting.
    """
    mode = (mode or DEFAULT_MODE).strip().lower()
    if mode == MODE_CPU:
        return cpu_device()
    available = cuda_device_count()
    if mode == MODE_CUDA:
        if available > 0:
            return Device("cuda", CUDA_COMPUTE_TYPE, "NVIDIA GPU (CUDA)")
        return cpu_device(
            "No CUDA device was detected, so this run uses the CPU. "
            "Results are unaffected; it will simply take longer."
        )
    # Auto.
    if available > 0:
        return Device(
            "cuda", CUDA_COMPUTE_TYPE, "NVIDIA GPU (CUDA)", "selected automatically"
        )
    return cpu_device()


def fallback_to_cpu(failed: Device, error: BaseException) -> Optional[Device]:
    """A CPU device to retry a failed GPU run with, or ``None`` if it was CPU.

    A GPU can be visible and still fail at load time - too little VRAM, a driver
    mismatch, a card busy with something else. Losing the transcription to that
    would be worse than running it slowly, so the caller retries on the CPU and
    says so.
    """
    if not failed.is_gpu:
        return None
    reason = str(error).strip() or error.__class__.__name__
    return cpu_device(f"GPU transcription failed ({reason}); retried on the CPU.")
