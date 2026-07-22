"""Unit tests for vit.utils.device."""

from __future__ import annotations

import torch

import pytest

from vit.utils.device import get_device_info, resolve_device


class TestResolveDevice:
    def test_cpu_preference(self) -> None:
        assert resolve_device("cpu") == torch.device("cpu")

    def test_auto_preference_returns_valid_device(self) -> None:
        device = resolve_device("auto")
        assert device.type in ("cpu", "cuda")
        if not torch.cuda.is_available():
            assert device.type == "cpu"

    def test_case_insensitive(self) -> None:
        assert resolve_device("CPU") == torch.device("cpu")

    def test_invalid_preference_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_device("tpu")

    def test_cuda_requested_without_cuda_available_raises(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA is available on this machine; cannot test unavailability path")
        with pytest.raises(RuntimeError):
            resolve_device("cuda")

    def test_cuda_indexed_requested_without_cuda_available_raises(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA is available on this machine; cannot test unavailability path")
        with pytest.raises(RuntimeError):
            resolve_device("cuda:0")


class TestGetDeviceInfo:
    def test_contains_expected_keys(self) -> None:
        info = get_device_info()
        assert "platform" in info
        assert "python_version" in info
        assert "torch_version" in info
        assert "cuda_available" in info
        assert info["cuda_available"] in ("True", "False")

    def test_cuda_fields_absent_when_no_cuda(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA is available on this machine")
        info = get_device_info()
        assert "gpu_name" not in info
        assert "cuda_version" not in info

    def test_accepts_explicit_device(self) -> None:
        device = resolve_device("cpu")
        info = get_device_info(device)
        assert info["cuda_available"] == "False" or torch.cuda.is_available()