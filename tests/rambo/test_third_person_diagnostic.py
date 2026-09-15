"""Simulator-free contracts for final third-person validation evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from rambo.validation.third_person_diagnostic import (  # noqa: E402
    FINAL_ACTION_STEPS,
    FINAL_FRONT_RGB_FRAMES,
    FINAL_PHYSICS_TICKS,
    MANIFEST_FILENAME,
    PROVENANCE_SCHEMA_VERSION,
    TARGET_ISAACLAB_COMMIT,
    TARGET_ISAACLAB_TAG,
    TARGET_ISAACSIM_VERSION,
    ThirdPersonDiagnosticError,
    build_manifest,
    final_capture_timing,
    validate_final_artifact,
    validate_final_capture_record,
    validate_prelaunch_source_provenance,
    validate_runtime_provenance,
    verify_checksum_manifest,
    write_checksum_manifest,
    write_json,
)


def _rollout() -> dict:
    return {
        "steps": FINAL_ACTION_STEPS,
        "policy_timestep_s": 0.01,
        "physics_timestep_s": 0.002,
        "expected_camera_frame_count": FINAL_FRONT_RGB_FRAMES,
    }


def _rgb() -> dict:
    return {
        "passed": True,
        "frame_count": FINAL_FRONT_RGB_FRAMES,
        "expected_frame_count": FINAL_FRONT_RGB_FRAMES,
    }


def _capture(timing: dict) -> dict:
    return {
        "action_step": FINAL_ACTION_STEPS,
        "physics_ticks": FINAL_PHYSICS_TICKS,
        "timestamp_s": 30.0,
        "front_camera_ticks_before_sensor_initialization": [FINAL_PHYSICS_TICKS],
        "front_camera_ticks_after_sensor_initialization": [FINAL_PHYSICS_TICKS],
        "camera_target_is_final_robot_root": True,
        "final_state_preserved_during_sensor_initialization": True,
        "front_camera_counter_preserved_during_sensor_initialization": True,
        "robot_and_scene_visual_review_required": True,
        "rgb": {"shape": [1, 480, 640, 3], "mean": 48.0, "std": 31.0},
        "distance_to_image_plane": {"shape": [1, 480, 640, 1], "positive_fraction": 0.8},
        "timing_copy": timing,
    }


def _provenance(*, clean: bool = True) -> dict:
    status = [] if clean else [" M scripts/rambo/validate.py"]
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "collection_phase": "pre_app_launcher_source_then_runtime",
        "command": [
            "/workspace/envs/rambo-isaac60-py312/bin/python",
            "/workspace/repos/RAMBO_Data/scripts/rambo/validate.py",
            "--task",
            "Isaac-RAMBO-Quadruped-Go2-v0",
            "--checkpoint",
            "/workspace/checkpoints/rambo/go2/quadruped/model_2000.pt",
            "--steps",
            "3000",
            "--output-dir",
            "/workspace/runs/audit/isaac60/M7/synthetic",
            "--third-person-diagnostic",
            "final",
        ],
        "rambo": {
            "module_path": "/workspace/repos/RAMBO_Data/source/rambo/rambo/__init__.py",
            "git_root": "/workspace/repos/RAMBO_Data",
            "git_head": "a" * 40,
            "pre_run_worktree_clean": clean,
            "pre_run_status_porcelain_v1": status,
            "runtime_module_path": "/workspace/repos/RAMBO_Data/source/rambo/rambo/__init__.py",
            "runtime_git_root": "/workspace/repos/RAMBO_Data",
            "runtime_git_head": "a" * 40,
            "runtime_worktree_clean": clean,
            "runtime_status_porcelain_v1": status,
        },
        "isaaclab": {
            "module_path": "/workspace/third_party/IsaacLab/3.0.0-beta2.patch1/source/isaaclab/isaaclab/__init__.py",
            "package_version": "6.1.14",
            "module_version": "3.0.0-beta2.patch1",
            "git_root": "/workspace/third_party/IsaacLab/3.0.0-beta2.patch1",
            "git_head": TARGET_ISAACLAB_COMMIT,
            "git_exact_tag": TARGET_ISAACLAB_TAG,
        },
        "isaacsim": {
            "package_version": TARGET_ISAACSIM_VERSION,
        },
        "runtime": {
            "python_version": "3.12.3",
            "python_executable": "/workspace/envs/rambo-isaac60-py312/bin/python",
            "platform": "Linux-6.8-x86_64",
            "torch_version": "2.10.0+cu128",
            "cuda_runtime": "12.8",
            "cuda_device": "Synthetic RTX",
        },
    }


def _load_validate_entrypoint():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "rambo" / "validate.py"
    spec = importlib.util.spec_from_file_location("rambo_third_person_validate", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_timing_requires_one_complete_m7_m9_rollout() -> None:
    timing = final_capture_timing(requested_steps=FINAL_ACTION_STEPS, rollout=_rollout(), rgb=_rgb())

    assert timing["physics_ticks"] == FINAL_PHYSICS_TICKS
    assert timing["front_rgb_frames"] == FINAL_FRONT_RGB_FRAMES
    assert timing["physics_ticks_per_action"] == 5
    assert timing["final_timestamp_s"] == 30.0


def test_final_timing_rejects_short_or_incomplete_evidence() -> None:
    with pytest.raises(ThirdPersonDiagnosticError, match="3000"):
        final_capture_timing(requested_steps=16, rollout=_rollout(), rgb=_rgb())

    incomplete_rgb = _rgb()
    incomplete_rgb["frame_count"] = 374
    with pytest.raises(ThirdPersonDiagnosticError, match="375"):
        final_capture_timing(requested_steps=FINAL_ACTION_STEPS, rollout=_rollout(), rgb=incomplete_rgb)


def test_final_capture_record_requires_final_timing_targeting_and_rgbd_geometry() -> None:
    timing = final_capture_timing(requested_steps=FINAL_ACTION_STEPS, rollout=_rollout(), rgb=_rgb())
    capture = _capture(timing)
    validate_final_capture_record(capture, timing)

    capture["physics_ticks"] = FINAL_PHYSICS_TICKS - 5
    with pytest.raises(ThirdPersonDiagnosticError, match="final physics tick"):
        validate_final_capture_record(capture, timing)


def test_runtime_provenance_requires_prelaunch_clean_source_and_pinned_runtime() -> None:
    provenance = _provenance()
    validate_prelaunch_source_provenance(provenance)
    validate_runtime_provenance(provenance)

    dirty = _provenance(clean=False)
    with pytest.raises(ThirdPersonDiagnosticError, match="clean RAMBO Git worktree"):
        validate_prelaunch_source_provenance(dirty)
    # A caller can inspect a diagnostic record with a dirty tree explicitly,
    # but build_manifest below always applies the formal clean-source gate.
    validate_runtime_provenance(dirty, require_clean_source=False)

    wrong_command = _provenance()
    wrong_command["command"].remove("--third-person-diagnostic")
    wrong_command["command"].remove("final")
    with pytest.raises(ThirdPersonDiagnosticError, match="third-person-diagnostic final"):
        validate_runtime_provenance(wrong_command)


def test_prelaunch_collector_records_dirty_status_without_starting_kit(monkeypatch: pytest.MonkeyPatch) -> None:
    entrypoint = _load_validate_entrypoint()

    def fake_git_output(_cwd: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "--show-toplevel"):
            return "/workspace/repos/RAMBO_Data"
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return " M scripts/rambo/validate.py\n?? tests/rambo/test_third_person_diagnostic.py"
        if arguments == ("rev-parse", "HEAD"):
            return "c" * 40
        raise AssertionError(arguments)

    monkeypatch.setattr(entrypoint, "_git_output", fake_git_output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "/workspace/repos/RAMBO_Data/scripts/rambo/validate.py",
            "--steps=3000",
            "--third-person-diagnostic=final",
        ],
    )
    provenance = entrypoint._collect_pre_app_launcher_provenance()

    assert provenance["rambo"]["git_head"] == "c" * 40
    assert provenance["rambo"]["module_path"].endswith("/source/rambo/rambo/__init__.py")
    assert provenance["rambo"]["pre_run_worktree_clean"] is False
    assert provenance["rambo"]["pre_run_status_porcelain_v1"] == [
        " M scripts/rambo/validate.py",
        "?? tests/rambo/test_third_person_diagnostic.py",
    ]
    validate_prelaunch_source_provenance(provenance, require_clean_source=False)


def test_manifest_and_checksums_cover_every_final_evidence_file(tmp_path: Path) -> None:
    timing = final_capture_timing(requested_steps=FINAL_ACTION_STEPS, rollout=_rollout(), rgb=_rgb())
    capture = _capture(timing)
    physx = {
        "requested_cfg": "isaaclab_physx.physics.physx_manager_cfg.PhysxCfg",
        "actual_manager": "isaaclab_physx.physics.physx_manager.PhysxManager",
        "configured_manager": "isaaclab_physx.physics.physx_manager:PhysxManager",
        "use_newton_actuators": False,
    }
    summary = {
        "passed": True,
        "task": "Isaac-RAMBO-Quadruped-Go2-v0",
        "checkpoint": {"sha256": "a" * 64},
        "provenance": _provenance(),
        "backend_before": physx,
        "backend_after_rollout": physx,
        "backend_after": physx,
        "validation_only_overrides": {"episode_length_s": {"before": 10.0, "after": 31.0}},
        "third_person_diagnostic": {
            "production_front_camera_contract_unchanged": True,
            "timing": timing,
            "capture": capture,
            "backend_before_sensor_initialization": physx,
            "backend_after_capture": physx,
        },
    }
    manifest = build_manifest(summary)
    assert manifest["timing"]["physics_ticks"] == FINAL_PHYSICS_TICKS
    assert manifest["files"]["third_person_manifest"] == MANIFEST_FILENAME
    assert manifest["provenance"]["rambo"]["pre_run_worktree_clean"] is True

    write_json(tmp_path / "summary.json", summary)
    write_json(tmp_path / MANIFEST_FILENAME, manifest)
    (tmp_path / "rgb").mkdir()
    for frame_index in range(FINAL_FRONT_RGB_FRAMES):
        (tmp_path / "rgb" / f"{frame_index:06d}.png").write_bytes(b"front-frame")
    (tmp_path / "rgb_timestamps.npy").write_bytes(b"timestamps")
    (tmp_path / "rgb_frame_ids.npy").write_bytes(b"frame-ids")
    (tmp_path / "third_person_final_robot_scene_rgb.png").write_bytes(b"third-person-rgb")
    (tmp_path / "third_person_final_robot_scene_distance_to_image_plane.npy").write_bytes(b"third-person-depth")
    process_exit = {
        "schema_version": 1,
        "captured_by": "scripts/rambo/run_runtime_artifact.sh",
        "shell_exit_status": 0,
        "exit_status_zero": True,
        "signal_hint": None,
        "workload_summary_passed_before_exit": True,
        "acceptance_passed": True,
    }
    manifest["process_exit"] = process_exit
    write_json(tmp_path / MANIFEST_FILENAME, manifest)
    write_json(tmp_path / "process_exit.json", process_exit)

    checksum_path = write_checksum_manifest(tmp_path)
    assert checksum_path.name == "checksums.sha256"
    assert validate_final_artifact(tmp_path)["front_rgb_frame_count"] == FINAL_FRONT_RGB_FRAMES
    assert verify_checksum_manifest(tmp_path) == 382
    checksum_path.write_text(
        checksum_path.read_text(encoding="utf-8").replace("summary.json", "missing.json"),
        encoding="utf-8",
    )
    with pytest.raises(ThirdPersonDiagnosticError, match="missing"):
        verify_checksum_manifest(tmp_path)


def test_final_artifact_rejects_missing_required_image_or_nonzero_exit(tmp_path: Path) -> None:
    timing = final_capture_timing(requested_steps=FINAL_ACTION_STEPS, rollout=_rollout(), rgb=_rgb())
    physx = {
        "requested_cfg": "isaaclab_physx.physics.physx_manager_cfg.PhysxCfg",
        "actual_manager": "isaaclab_physx.physics.physx_manager.PhysxManager",
        "configured_manager": "isaaclab_physx.physics.physx_manager:PhysxManager",
        "use_newton_actuators": False,
    }
    summary = {
        "passed": True,
        "task": "Isaac-RAMBO-Quadruped-Go2-v0",
        "checkpoint": {"sha256": "a" * 64},
        "provenance": _provenance(),
        "backend_before": physx,
        "backend_after_rollout": physx,
        "backend_after": physx,
        "validation_only_overrides": {},
        "third_person_diagnostic": {
            "production_front_camera_contract_unchanged": True,
            "timing": timing,
            "capture": _capture(timing),
            "backend_before_sensor_initialization": physx,
            "backend_after_capture": physx,
        },
    }
    manifest = build_manifest(summary)
    write_json(tmp_path / "summary.json", summary)
    write_json(tmp_path / MANIFEST_FILENAME, manifest)
    (tmp_path / "rgb").mkdir()
    for frame_index in range(FINAL_FRONT_RGB_FRAMES - 1):
        (tmp_path / "rgb" / f"{frame_index:06d}.png").write_bytes(b"front-frame")
    (tmp_path / "rgb_timestamps.npy").write_bytes(b"timestamps")
    (tmp_path / "rgb_frame_ids.npy").write_bytes(b"frame-ids")
    (tmp_path / "third_person_final_robot_scene_rgb.png").write_bytes(b"third-person-rgb")
    (tmp_path / "third_person_final_robot_scene_distance_to_image_plane.npy").write_bytes(b"third-person-depth")
    process_exit = {"shell_exit_status": 1, "exit_status_zero": False, "acceptance_passed": False}
    manifest["process_exit"] = process_exit
    write_json(tmp_path / MANIFEST_FILENAME, manifest)
    write_json(tmp_path / "process_exit.json", process_exit)
    write_checksum_manifest(tmp_path)

    with pytest.raises(ThirdPersonDiagnosticError, match="375 front RGB PNG"):
        validate_final_artifact(tmp_path)


def test_validate_entrypoint_makes_final_capture_explicit_and_post_rollout_only() -> None:
    root = Path(__file__).resolve().parents[2]
    contents = (root / "scripts" / "rambo" / "validate.py").read_text(encoding="utf-8")

    assert '"--third-person-diagnostic"' in contents
    assert 'choices=("final",)' in contents
    assert "Disabled by default." in contents
    assert "requires exactly --steps 3000" in contents
    assert "validation_only_overrides" in contents
    assert "write_checksum_manifest" in contents
    assert "_collect_pre_app_launcher_provenance" in contents
    assert "validate_runtime_provenance" in contents
    assert contents.index("_collect_pre_app_launcher_provenance()") < contents.index("app_launcher =")
    assert contents.index('rollout_metrics = runtime["run_policy_rollout"]') < contents.index(
        "rgb_metrics = rgb_recorder.finalize()"
    ) < contents.index("capture = _capture_final_third_person_diagnostic")
    capture_body = contents.split("def _capture_final_third_person_diagnostic", maxsplit=1)[1].split(
        "def main", maxsplit=1
    )[0]
    assert "env.reset(" not in capture_body
    assert "base_env.sim.reset()" in capture_body
    assert "camera_target_is_final_robot_root" in capture_body
    assert "front_camera_counter_preserved_during_sensor_initialization" in capture_body
