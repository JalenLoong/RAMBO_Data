"""Static and pure-Python contracts for the M2.1 CUDA evidence utility."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "rambo" / "verify_cuda_m2.py"
    spec = importlib.util.spec_from_file_location("rambo_m2_cuda_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m2_cuda_utility_never_imports_a_simulator_or_physics_backend() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts" / "rambo" / "verify_cuda_m2.py").read_text(encoding="utf-8")

    assert "isaaclab" not in source
    assert "isaacsim" not in source
    assert "newton" not in source.lower()
    assert "physics_backend_selected" in source
    assert "sm_89" in source
    assert "sm_86" in source
    assert "native_sm89_compiled" in source
    assert "sm_120" in source
    assert "native_sm120_compiled" in source
    assert "RUNS_ROOT" in source


def test_m2_cuda_utility_writes_a_checksum_covered_json_summary(tmp_path: Path) -> None:
    module = _load_module()
    payload = {"passed": True, "physics_backend_selected": False}

    module._write_artifact(tmp_path, payload)

    summary_path = tmp_path / "summary.json"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == payload
    assert (tmp_path / "checksums.sha256").read_text(encoding="utf-8") == (
        f"{hashlib.sha256(summary_path.read_bytes()).hexdigest()}  summary.json\n"
    )


class _Scalar:
    def item(self) -> float:
        return 1.0


class _Values:
    def square(self):
        return self

    def sum(self) -> _Scalar:
        return _Scalar()


class _Cuda:
    def __init__(self, capability: tuple[int, int], architectures: list[str]) -> None:
        self.capability = capability
        self.architectures = architectures
        self.synchronized = False

    def is_available(self) -> bool:
        return True

    def get_device_capability(self, _index: int) -> tuple[int, int]:
        return self.capability

    def get_arch_list(self) -> list[str]:
        return self.architectures

    def get_device_name(self, _index: int) -> str:
        return "synthetic GPU"

    def synchronize(self) -> None:
        self.synchronized = True


class _Torch:
    float32 = "float32"

    def __init__(self, capability: tuple[int, int], architectures: list[str]) -> None:
        self.cuda = _Cuda(capability, architectures)
        self.version = type("Version", (), {"cuda": "12.8"})()
        self.__version__ = "synthetic"

    def arange(self, *_args, **_kwargs) -> _Values:
        return _Values()


def test_blackwell_profile_requires_capability_120_and_native_sm120() -> None:
    module = _load_module()
    torch = _Torch((12, 0), ["sm_120"])
    result = module.collect_cuda_evidence(torch, hardware_profile="blackwell")
    assert result["hardware_profile"] == "blackwell"
    assert result["native_sm120_compiled"] is True
    assert result["accepted_compiled_arch"] == "sm_120"
    assert result["ada_compatible_compiled_arch"] is None
    assert torch.cuda.synchronized is True

    with pytest.raises(RuntimeError, match="lacks a compiled target"):
        module.collect_cuda_evidence(_Torch((12, 0), ["sm_90"]), hardware_profile="blackwell")
    with pytest.raises(RuntimeError, match=r"expected \(12, 0\)"):
        module.collect_cuda_evidence(_Torch((8, 9), ["sm_120"]), hardware_profile="blackwell")


def test_ada_profile_retains_reviewed_forward_compatible_arches() -> None:
    module = _load_module()
    result = module.collect_cuda_evidence(_Torch((8, 9), ["sm_86"]), hardware_profile="ada")
    assert result["accepted_compiled_arch"] == "sm_86"
    assert result["ada_compatible_compiled_arch"] == "sm_86"
