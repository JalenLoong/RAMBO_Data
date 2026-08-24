"""Static and pure-Python contracts for the M2.1 CUDA evidence utility."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


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


def test_m2_cuda_utility_writes_a_checksum_covered_json_summary(tmp_path: Path) -> None:
    module = _load_module()
    payload = {"passed": True, "physics_backend_selected": False}

    module._write_artifact(tmp_path, payload)

    summary_path = tmp_path / "summary.json"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == payload
    assert (tmp_path / "checksums.sha256").read_text(encoding="utf-8") == (
        f"{hashlib.sha256(summary_path.read_bytes()).hexdigest()}  summary.json\n"
    )
