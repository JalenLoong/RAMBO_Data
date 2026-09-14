"""Static contracts for the isolated official Unitree Go2 smoke gate.

These tests intentionally parse source only: they do not import Isaac Lab,
start Kit, create a physics context, or require an EULA environment variable.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


def _source() -> tuple[Path, str, ast.Module]:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "rambo60" / "official_go2_smoke.py"
    contents = path.read_text(encoding="utf-8")
    return path, contents, ast.parse(contents, filename=str(path))


def _load_module_without_simulator_imports():
    path, _, _ = _source()
    spec = importlib.util.spec_from_file_location("official_go2_smoke_static", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_go2_gate_has_no_project_package_imports() -> None:
    _, _, tree = _source()
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert "rambo" not in imported_roots
    assert "crl2" not in imported_roots


def test_official_go2_gate_is_explicitly_physx_only() -> None:
    _, contents, _ = _source()
    assert 'PHYSX_CFG_FQN = "isaaclab_physx.physics.physx_manager_cfg.PhysxCfg"' in contents
    assert 'PHYSX_MANAGER_FQN = "isaaclab_physx.physics.physx_manager.PhysxManager"' in contents
    assert "physics=runtime[\"PhysxCfg\"]()" in contents
    assert "use_newton_actuators=False" in contents
    assert "isaaclab_newton" not in contents
    assert "import newton" not in contents
    assert "backend_before" in contents
    assert "backend_after" in contents


def test_official_go2_gate_has_bounded_rgb_and_artifact_contracts() -> None:
    _, contents, _ = _source()
    assert "DEFAULT_STEPS = 1000" in contents
    assert "PHYSICS_DT_S = 0.002" in contents
    assert "IMAGE_WIDTH = 640" in contents
    assert "IMAGE_HEIGHT = 480" in contents
    assert "UNITREE_GO2_CFG" in contents
    assert "GroundPlaneCfg" in contents
    assert "IsaacRtxRendererCfg" in contents
    assert "go2_rgb.png" in contents
    assert "checksums.sha256" in contents
    assert "RUNS_ROOT" in contents
    assert "Refusing to overwrite existing artifact directory" in contents
    assert "STATE_TENSOR_FIELDS" in contents
    assert "state_finite_checks" in contents


def test_official_go2_gate_rejects_unaudited_launcher_options() -> None:
    _, contents, _ = _source()
    assert "--viz none" in contents
    assert "--headless is forbidden" in contents
    assert "--experience is forbidden" in contents
    assert "--kit_args is forbidden" in contents
    assert "os._exit" not in contents
    assert "skip_cleanup" not in contents


def test_official_go2_artifact_helpers_are_fresh_and_checksum_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise only standard-library artifact helpers; no Kit import is allowed."""

    module = _load_module_without_simulator_imports()
    artifact_root = tmp_path / "M2"
    monkeypatch.setattr(module, "ARTIFACT_ROOT", artifact_root)
    output_dir = module._prepare_output_dir(artifact_root / "fresh-go2")
    module._write_summary(output_dir, {"passed": True})
    (output_dir / "go2_rgb.png").write_bytes(b"synthetic-static-test-frame")
    checksums = module._write_checksums(output_dir)

    assert sorted(checksums) == ["go2_rgb.png", "summary.json"]
    manifest = (output_dir / "checksums.sha256").read_text(encoding="utf-8")
    assert "go2_rgb.png" in manifest
    assert "summary.json" in manifest
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module._prepare_output_dir(output_dir)
