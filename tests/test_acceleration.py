"""Choosing the compute device: CPU baseline, optional GPU, graceful failure."""

import pytest

from whispr import acceleration
from whispr.acceleration import (
    DEFAULT_MODE,
    MODE_CPU,
    MODE_CUDA,
    MODE_LABELS,
    Device,
    cpu_device,
    describe_hardware,
    fallback_to_cpu,
    resolve,
)


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(acceleration, "cuda_device_count", lambda: 0)


@pytest.fixture
def one_gpu(monkeypatch):
    monkeypatch.setattr(acceleration, "cuda_device_count", lambda: 1)


def test_cpu_is_the_default_when_no_gpu_is_present(no_gpu):
    device = resolve(DEFAULT_MODE)
    assert device.device == "cpu"
    assert device.compute_type == "int8"
    assert not device.is_gpu
    # Auto falling back to CPU is normal, not a warning.
    assert device.note == ""


def test_auto_prefers_a_gpu_when_one_is_available(one_gpu):
    device = resolve("auto")
    assert device.device == "cuda"
    assert device.compute_type == "float16"
    assert device.is_gpu


def test_cpu_is_honoured_even_with_a_gpu_present(one_gpu):
    # CPU stays fully supported; asking for it must not be overridden.
    assert resolve(MODE_CPU).device == "cpu"


def test_an_impossible_gpu_request_says_so_instead_of_failing_the_run(no_gpu):
    device = resolve(MODE_CUDA)
    assert device.device == "cpu"
    # Not a silent downgrade: the operator is told why their choice was not used.
    assert "No CUDA device was detected" in device.note
    assert "Results are unaffected" in device.note


def test_an_unknown_or_empty_mode_falls_back_to_auto(no_gpu):
    assert resolve("").device == "cpu"
    assert resolve("wildly-invalid").device == "cpu"


def test_a_failed_gpu_run_offers_a_cpu_retry():
    gpu = Device("cuda", "float16", "NVIDIA GPU (CUDA)")
    retry = fallback_to_cpu(gpu, RuntimeError("out of memory"))
    assert retry is not None
    assert retry.device == "cpu"
    assert "out of memory" in retry.note
    assert "retried on the CPU" in retry.note


def test_a_failed_cpu_run_has_nothing_to_fall_back_to():
    # Retrying the CPU on the CPU would just fail again; the caller re-raises.
    assert fallback_to_cpu(cpu_device(), RuntimeError("boom")) is None


def test_an_error_with_no_message_still_produces_a_usable_note():
    retry = fallback_to_cpu(Device("cuda", "float16", "GPU"), RuntimeError())
    assert retry is not None and "RuntimeError" in retry.note


def test_detection_is_local_and_treats_any_failure_as_no_gpu(monkeypatch):
    class _Broken:
        @staticmethod
        def get_cuda_device_count():
            raise OSError("no driver")

    monkeypatch.setitem(__import__("sys").modules, "ctranslate2", _Broken)
    # A missing driver, a CPU-only wheel or a broken install all mean "no GPU"
    # operationally - never an exception that stops the app.
    assert acceleration.cuda_device_count() == 0
    assert not acceleration.cuda_available()


def test_hardware_description_is_honest_either_way(no_gpu):
    assert "CPU only" in describe_hardware()


def test_hardware_description_counts_devices(one_gpu, monkeypatch):
    assert "1 device" in describe_hardware()
    monkeypatch.setattr(acceleration, "cuda_device_count", lambda: 2)
    assert "2 devices" in describe_hardware()


def test_every_offered_mode_resolves(no_gpu):
    for mode, label in MODE_LABELS:
        assert resolve(mode).device in {"cpu", "cuda"}
        assert label  # each mode is presented with a human label


def test_describe_names_the_device_and_precision():
    assert Device("cuda", "float16", "NVIDIA GPU (CUDA)").describe() == (
        "NVIDIA GPU (CUDA) (cuda/float16)"
    )
