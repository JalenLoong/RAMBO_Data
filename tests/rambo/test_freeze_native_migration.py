"""Pure-Python contracts for the M11 native-freeze generator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "rambo" / "freeze_native_migration.py"
    spec = importlib.util.spec_from_file_location("rambo_native_freeze", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _installation_verification() -> str:
    return (
        json.dumps(
            {
                "verified": True,
                "verification_mode": "gpu-required",
                "physics_policy": {"required_runtime_backend": "PhysX"},
            }
        )
        + "\nISAACSIM60_INSTALL_VERIFIED\n"
    )


def _host_manifest(artifact_root: Path, accepted_artifact: Path, m3_artifact: Path) -> dict[str, object]:
    return {
        "host": {
            "memory": {
                "probe": "/proc/meminfo",
                "physical_total_bytes": 1024,
                "swap_total_bytes": 0,
            },
            "toolchain": {
                "uv": {"path": "/usr/bin/uv", "version": "0.11.19"},
                "git": {"path": "/usr/bin/git", "version": "2.43.0"},
            },
        },
        "runtime": {
            "physics": {
                "required_backend": "PhysX",
                "required_manager": "isaaclab_physx.physics.physx_manager.PhysxManager",
                "use_newton_actuators": False,
                "optional_isaaclab_newton_extras_requested": False,
            }
        },
        "evidence": {
            "artifact_root": str(artifact_root),
            "m3_clean_inventory_artifact": str(m3_artifact),
            "accepted_runtime_artifacts": [{"milestone": "M10", "path": str(accepted_artifact)}],
        },
        "container_status": {
            "docker_engine_or_nvidia_container_toolkit_available_on_validation_host": False,
            "docker_build_or_runtime_validated": False,
            "probe_scope": "static executable and package inventory only",
            "docker_cli": {"executable_present": False, "path": None, "version": None},
            "nvidia_container_toolkit": {
                "nvidia_container_cli_present": False,
                "nvidia_container_runtime_present": False,
                "package_probe": {"manager": "dpkg-query", "packages_checked": [], "installed_packages": []},
            },
        },
    }


def _make_repo(tmp_path: Path, module) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    artifact_root = tmp_path / "artifacts"
    accepted_artifact = artifact_root / "M10" / "accepted"
    m3_artifact = artifact_root / "M3" / "accepted"
    accepted_artifact.mkdir(parents=True)
    m3_artifact.mkdir(parents=True)
    for relative in module.freeze_input_paths():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")
    (repo / "host-manifest.json").write_text(
        json.dumps(_host_manifest(artifact_root, accepted_artifact, m3_artifact), indent=2) + "\n",
        encoding="utf-8",
    )
    verification = tmp_path / "install-verification.txt"
    verification.write_text(_installation_verification(), encoding="utf-8")
    return repo, artifact_root, verification


def test_canonical_freeze_inputs_cover_wrappers_and_migration_records() -> None:
    module = _load_module()
    paths = set(module.freeze_input_paths())
    assert {
        "scripts/rambo/run.sh",
        "scripts/rambo/run60.sh",
        "scripts/rambo/run60_contract.sh",
        "scripts/rambo/run_runtime_artifact.sh",
        "scripts/rambo/finalize_runtime_artifact.py",
        "scripts/rambo/validate_runtime_artifact.py",
        "scripts/rambo/physx_quadruped_policy_smoke.py",
        "scripts/rambo/qp_reference.py",
        "migration-report.md",
        "MIGRATION_STATUS.md",
        "MIGRATION_FINAL_MATRIX.md",
    } <= paths
    assert module.FREEZE_INPUT_GROUPS["migration_records"][-1] == "MIGRATION_FINAL_MATRIX.md"
    assert set(module.FREEZE_INPUT_GROUPS["deferred_gui_gate_contracts"]) == {
        "README.md",
        "scripts/rambo/official_physx_smoke.py",
        "scripts/rambo/m2_cartpole_gui_artifact.py",
        "scripts/rambo/run_m2_cartpole_gui_gate.sh",
        "scripts/rambo/record_m2_cartpole_gui_attestation.py",
        "scripts/rambo/validate_m2_cartpole_gui_gate.py",
        "scripts/rambo/teleop_loco_manip.py",
        "scripts/rambo/run_gui_keyboard_artifact.sh",
        "scripts/rambo/finalize_gui_keyboard_artifact.py",
        "scripts/rambo/record_gui_keyboard_attestation.py",
        "source/rambo/rambo/validation/gui_keyboard_artifact.py",
    }


def test_create_and_verify_native_freeze_binds_every_canonical_input(tmp_path: Path) -> None:
    module = _load_module()
    repo, artifact_root, verification = _make_repo(tmp_path, module)
    artifact = artifact_root / "M11" / "freeze"

    result = module.create_native_freeze_artifact(
        repo,
        artifact,
        verification,
        git_head="a" * 40,
        git_status="",
        generated_at_utc="2026-08-24T00:00:00Z",
    )

    assert result["freeze_input_count"] == len(module.freeze_input_paths())
    assert result["accepted_runtime_artifact_count"] == 2
    records = module._parse_digest_lines(
        (artifact / "freeze-inputs.sha256").read_text(encoding="utf-8"),
        label="fixture freeze inputs",
    )
    assert set(records) == set(module.freeze_input_paths())
    metadata = json.loads((artifact / "freeze-metadata.json").read_text(encoding="utf-8"))
    assert metadata["execution_boundary"] == {
        "docker_or_nvidia_container_toolkit_started_by_generator": False,
        "installation_verification_source_sha256": hashlib.sha256(verification.read_bytes()).hexdigest(),
        "isaac_sim_or_kit_started_by_generator": False,
    }
    assert module.verify_native_freeze_artifact(repo, artifact)["freeze_input_count"] == len(records)

    (repo / "scripts" / "rambo" / "run60.sh").write_text("changed wrapper\n", encoding="utf-8")
    with pytest.raises(module.NativeFreezeError, match="source input changed: scripts/rambo/run60.sh"):
        module.verify_native_freeze_artifact(repo, artifact)


def test_native_freeze_rejects_missing_final_matrix_and_dirty_source(tmp_path: Path) -> None:
    module = _load_module()
    repo, artifact_root, verification = _make_repo(tmp_path, module)
    (repo / "MIGRATION_FINAL_MATRIX.md").unlink()
    with pytest.raises(module.NativeFreezeError, match="source input is missing: MIGRATION_FINAL_MATRIX.md"):
        module.create_native_freeze_artifact(
            repo,
            artifact_root / "M11" / "missing-matrix",
            verification,
            git_head="b" * 40,
            git_status="",
        )

    repo, artifact_root, verification = _make_repo(tmp_path / "dirty", module)
    with pytest.raises(module.NativeFreezeError, match="source-clean Git worktree"):
        module.create_native_freeze_artifact(
            repo,
            artifact_root / "M11" / "dirty",
            verification,
            git_head="c" * 40,
            git_status=" M host-manifest.json\n",
        )


def test_real_host_manifest_records_static_capacity_toolchain_and_container_state() -> None:
    module = _load_module()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "host-manifest.json").read_text(encoding="utf-8"))

    module._validate_host_manifest(manifest)
    host = manifest["host"]
    assert host["memory"]["physical_total_bytes"] == 66479996928
    assert host["memory"]["swap_total_bytes"] == 8589930496
    assert host["toolchain"]["uv"] == {"path": "/usr/local/bin/uv", "version": "0.11.19"}
    assert host["toolchain"]["git"] == {"path": "/usr/bin/git", "version": "2.43.0"}
    container = manifest["container_status"]
    assert container["docker_build_or_runtime_validated"] is False
    assert container["docker_cli"]["executable_present"] is False
    assert container["nvidia_container_toolkit"]["nvidia_container_cli_present"] is False
    assert container["nvidia_container_toolkit"]["nvidia_container_runtime_present"] is False
