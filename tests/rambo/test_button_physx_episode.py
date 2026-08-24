"""Pure-Python contracts for the isolated M10 Button PhysX episode recorder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from rambo.validation import button_episode_artifact as artifact_schema
from rambo.validation.button_episode_artifact import (
    ACTION_DIMENSION,
    ACTION_SCHEMA_VERSION,
    BUTTON_HOLD_STEPS,
    BUTTON_PRESS_THRESHOLD_M,
    BUTTON_REBOUND_THRESHOLD_M,
    CAMERA_INTERVAL_ACTION_STEPS,
    DECIMATION,
    GPU_SLOPE_LIMIT_BYTES_PER_100_STEPS,
    M10_PROMPT,
    MIB,
    MOTION_MINIMUM_DELTA_M,
    OBSERVATION_DIMENSION,
    OBSERVATION_SCHEMA_VERSION,
    PHYSICS_DT_NS,
    PHYSX_CFG_FQN,
    PHYSX_MANAGER_FQN,
    POLICY_DT_NS,
    QUADRUPED_CHECKPOINT_SHA256,
    RSS_SLOPE_LIMIT_BYTES_PER_100_STEPS,
    SCHEMA_VERSION,
    TASK_ID,
    TARGET_ISAACLAB_COMMIT,
    TARGET_ISAACLAB_TAG,
    TARGET_ISAACSIM_VERSION,
    TRACE_FILENAMES,
    ButtonEpisodeArtifactError,
    validate_button_episode_artifact,
    write_checksums,
    write_json,
)


def _load_recorder_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "rambo" / "record_button_physx_episode.py"
    specification = importlib.util.spec_from_file_location("rambo_button_episode_recorder", script_path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _physx_evidence() -> dict[str, object]:
    return {
        "requested_cfg": PHYSX_CFG_FQN,
        "actual_manager": PHYSX_MANAGER_FQN,
        "configured_manager": PHYSX_MANAGER_FQN,
        "use_newton_actuators": False,
        "physics_dt_s": 0.002,
        "device": "cuda:0",
    }


def _timing() -> dict[str, float | int]:
    return {
        "physics_dt_s": PHYSICS_DT_NS / 1_000_000_000,
        "control_decimation": DECIMATION,
        "control_dt_s": POLICY_DT_NS / 1_000_000_000,
        "camera_update_period_s": 0.08,
        "camera_rate_hz": 12.5,
        "camera_interval_physics_ticks": 40,
        "camera_interval_action_steps": CAMERA_INTERVAL_ACTION_STEPS,
    }


def _policy_schema() -> dict[str, dict[str, str | int]]:
    return {
        "observation": {
            "version": OBSERVATION_SCHEMA_VERSION,
            "dimension": OBSERVATION_DIMENSION,
            "dtype": "torch.float32",
        },
        "action": {
            "version": ACTION_SCHEMA_VERSION,
            "dimension": ACTION_DIMENSION,
            "dtype": "torch.float32",
        },
    }


def _provenance() -> dict[str, object]:
    return {
        "rambo_git_commit": "b" * 40,
        "rambo_git_dirty": False,
        "isaaclab": {"tag": TARGET_ISAACLAB_TAG, "commit": TARGET_ISAACLAB_COMMIT},
        "isaacsim_version": TARGET_ISAACSIM_VERSION,
        "gpu": {"model": "Synthetic GPU", "driver": "999.0"},
    }


def _empty_acceptance() -> dict[str, object]:
    return {
        "button": {
            "press_threshold_m": BUTTON_PRESS_THRESHOLD_M,
            "hold_steps_required": BUTTON_HOLD_STEPS,
            "rebound_threshold_m": BUTTON_REBOUND_THRESHOLD_M,
            "max_displacement_m": 0.0,
            "max_consecutive_pressed_action_steps": 0,
            "first_success_action_step": None,
            "first_rebound_after_success_action_step": None,
            "min_displacement_after_success_m": None,
        },
        "motion": {
            "minimum_required_delta_m": MOTION_MINIMUM_DELTA_M,
            "baseline_action_step": 1,
            "max_base_root_link_displacement_m": 0.0,
            "max_fl_foot_link_displacement_m": 0.0,
        },
    }


def _write_valid_short_artifact(output_dir: Path, *, steps: int = 8) -> None:
    assert steps % CAMERA_INTERVAL_ACTION_STEPS == 0
    (output_dir / "rgb").mkdir(parents=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_ID,
        "prompt": M10_PROMPT,
        "provenance": _provenance(),
        "requested_action_steps": steps,
        "short_smoke": True,
        "timebase": {
            "policy_dt_ns": POLICY_DT_NS,
            "physics_dt_ns": 2_000_000,
            "decimation": DECIMATION,
            "camera_interval_action_steps": CAMERA_INTERVAL_ACTION_STEPS,
        },
        "timing": _timing(),
        "policy_schema": _policy_schema(),
        "files": {**TRACE_FILENAMES, "rgb_directory": "rgb"},
    }
    runtime = {
        "python": "3.12.0",
        "torch": "2.7.0",
        "cuda_runtime": "12.8",
        "gpu": "Synthetic GPU",
        "gpu_index": 0,
        "driver": "999.0",
        "isaacsim_version": TARGET_ISAACSIM_VERSION,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "task": TASK_ID,
        "prompt": M10_PROMPT,
        "provenance": _provenance(),
        "seed": 42,
        "requested_action_steps": steps,
        "expected_rgb_frames": steps // CAMERA_INTERVAL_ACTION_STEPS,
        "short_smoke": True,
        "viz": "none",
        "enable_cameras": True,
        "runtime": runtime,
        "checkpoint": {
            "sha256": QUADRUPED_CHECKPOINT_SHA256,
            "allowlist_sha256": QUADRUPED_CHECKPOINT_SHA256,
            "observation_dim": OBSERVATION_DIMENSION,
            "action_dim": ACTION_DIMENSION,
        },
        "timing": _timing(),
        "policy_schema": _policy_schema(),
        "configured_physics": {
            "cfg": PHYSX_CFG_FQN,
            "use_newton_actuators": False,
        },
        "backend_before": _physx_evidence(),
        "backend_after": _physx_evidence(),
        "trace_counts": {
            "actions": steps,
            "observations": steps,
            "post_states": steps,
            "rgb_frames": steps // CAMERA_INTERVAL_ACTION_STEPS,
        },
        "terminal": {"terminal_count": 0},
        "success": {"success_seen": False, "release_after_success_seen": False},
        "acceptance": _empty_acceptance(),
        "rgb_cpu_staging": {
            "reused": True,
            "allocations": 1,
            "copies": steps // CAMERA_INTERVAL_ACTION_STEPS,
            "shape": [480, 640, 3],
            "dtype": "torch.uint8",
            "pinned_memory": True,
        },
        "memory_monitor": {
            "interval_action_steps": 100,
            "analysis_start_action_step": 1000,
            "first_last_window_action_steps": 500,
            "limits": {
                "gpu_slope_mib_per_100_steps": 1.0,
                "gpu_median_delta_mib": 128.0,
                "rss_slope_mib_per_100_steps": 4.0,
                "rss_median_delta_mib": 512.0,
            },
            "samples": [
                {
                    "action_step": 0,
                    "timestamp_ns": 0,
                    "gpu_allocated_bytes": 32 * MIB,
                    "rss_bytes": 1024 * MIB,
                }
            ],
            "analysis": {"evaluated": False, "reason": "requires action_steps >= 1000"},
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", summary)
    with (output_dir / TRACE_FILENAMES["actions"]).open("w", encoding="utf-8") as actions, (
        output_dir / TRACE_FILENAMES["observations"]
    ).open("w", encoding="utf-8") as observations, (
        output_dir / TRACE_FILENAMES["post_states"]
    ).open("w", encoding="utf-8") as states:
        for step in range(1, steps + 1):
            actions.write(
                json.dumps(
                    {
                        "action_step": step,
                        "start_timestamp_ns": (step - 1) * POLICY_DT_NS,
                        "end_timestamp_ns": step * POLICY_DT_NS,
                        "policy_action": [[0.0] * ACTION_DIMENSION],
                        "command": {},
                    }
                )
                + "\n"
            )
            observations.write(
                json.dumps(
                    {
                        "action_step": step,
                        "timestamp_ns": step * POLICY_DT_NS,
                        "policy_observation": [[0.0] * OBSERVATION_DIMENSION],
                    }
                )
                + "\n"
            )
            states.write(
                json.dumps(
                    {
                        "action_step": step,
                        "timestamp_ns": step * POLICY_DT_NS,
                        "robot": {
                            "root_link_pos_w_m": [[0.0, 0.0, 0.3]],
                            "fl_foot_body_name": "FL_foot",
                            "fl_foot_link_pos_w_m": [[0.2, 0.14, 0.0]],
                        },
                        "button": {
                            "displacement_m": [0.0],
                            "success": [False],
                            "released": [True],
                        },
                        "terminal": [False],
                    }
                )
                + "\n"
            )
    frame = output_dir / "rgb" / "000001.png"
    frame.write_bytes(b"not-decoded-by-the-offline-validator")
    (output_dir / TRACE_FILENAMES["rgb_frames"]).write_text(
        json.dumps(
            {
                "frame_index": 1,
                "action_step": 8,
                "timestamp_ns": 8 * POLICY_DT_NS,
                "physics_ticks": 8 * DECIMATION,
                "camera_frame_id": 1,
                "path": "rgb/000001.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_checksums(output_dir)


def test_offline_validator_accepts_a_complete_short_physx_artifact(tmp_path: Path) -> None:
    _write_valid_short_artifact(tmp_path)

    result = validate_button_episode_artifact(tmp_path)

    assert result["passed"] is True
    assert result["short_smoke"] is True
    assert result["rgb_frames"] == 1
    assert result["memory_monitor"] == {"evaluated": False, "sample_count": 1}


def test_offline_validator_rejects_an_unsynchronized_trace_even_before_checksum_check(tmp_path: Path) -> None:
    _write_valid_short_artifact(tmp_path)
    action_path = tmp_path / TRACE_FILENAMES["actions"]
    records = action_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(records[0])
    first["end_timestamp_ns"] = 1
    records[0] = json.dumps(first)
    action_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    with pytest.raises(ButtonEpisodeArtifactError, match="Action end timestamp"):
        validate_button_episode_artifact(tmp_path)


def test_offline_validator_requires_the_complete_m10_provenance_contract(tmp_path: Path) -> None:
    _write_valid_short_artifact(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("provenance")
    write_json(summary_path, summary)
    write_checksums(tmp_path)

    with pytest.raises(ButtonEpisodeArtifactError, match="summary.provenance is required"):
        validate_button_episode_artifact(tmp_path)


def test_offline_validator_requires_the_declared_405_by_18_policy_trace_shapes(tmp_path: Path) -> None:
    _write_valid_short_artifact(tmp_path)
    action_path = tmp_path / TRACE_FILENAMES["actions"]
    records = action_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(records[0])
    first["policy_action"] = [[0.0] * (ACTION_DIMENSION - 1)]
    records[0] = json.dumps(first)
    action_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    with pytest.raises(ButtonEpisodeArtifactError, match="action.policy_action must have shape"):
        validate_button_episode_artifact(tmp_path)


def test_offline_full_acceptance_gate_requires_press_hold_rebound_and_motion() -> None:
    acceptance = {
        "button": {
            "max_displacement_m": BUTTON_PRESS_THRESHOLD_M,
            "max_consecutive_pressed_action_steps": BUTTON_HOLD_STEPS,
            "min_displacement_after_success_m": BUTTON_REBOUND_THRESHOLD_M,
        },
        "motion": {
            "max_base_root_link_displacement_m": MOTION_MINIMUM_DELTA_M,
            "max_fl_foot_link_displacement_m": MOTION_MINIMUM_DELTA_M,
        },
    }
    final_state = {"button": {"success": [True], "released": [True]}}

    artifact_schema._validate_full_acceptance(
        success_seen=True,
        release_after_success_seen=True,
        final_state=final_state,
        acceptance=acceptance,
        rgb_count=375,
    )

    acceptance["button"]["max_consecutive_pressed_action_steps"] = BUTTON_HOLD_STEPS - 1
    with pytest.raises(ButtonEpisodeArtifactError, match="five action steps"):
        artifact_schema._validate_full_acceptance(
            success_seen=True,
            release_after_success_seen=True,
            final_state=final_state,
            acceptance=acceptance,
            rgb_count=375,
        )


def _acceptance_state(
    step: int,
    *,
    displacement: float,
    success: bool,
    base_x: float,
    fl_x: float,
) -> dict[str, object]:
    return {
        "action_step": step,
        "robot": {
            "root_link_pos_w_m": [[base_x, 0.0, 0.3]],
            "fl_foot_link_pos_w_m": [[fl_x, 0.14, 0.0]],
        },
        "button": {"displacement_m": [displacement], "success": [success]},
    }


def test_recorder_accumulator_proves_the_m10_button_and_loco_motion_evidence() -> None:
    module = _load_recorder_module()
    accumulator = module._ButtonAcceptanceAccumulator(
        press_threshold_m=BUTTON_PRESS_THRESHOLD_M,
        hold_steps=BUTTON_HOLD_STEPS,
        rebound_threshold_m=BUTTON_REBOUND_THRESHOLD_M,
        minimum_motion_m=MOTION_MINIMUM_DELTA_M,
    )
    for step in range(1, 8):
        accumulator.update(
            _acceptance_state(
                step,
                displacement=BUTTON_PRESS_THRESHOLD_M if 2 <= step <= 6 else (0.001 if step == 7 else 0.0),
                success=step == 6,
                base_x=0.051 if step == 7 else 0.0,
                fl_x=0.051 if step == 7 else 0.0,
            )
        )
    accumulator.assert_full_acceptance()
    evidence = accumulator.evidence()
    assert evidence["button"]["max_consecutive_pressed_action_steps"] == BUTTON_HOLD_STEPS
    assert evidence["button"]["min_displacement_after_success_m"] == 0.001
    assert evidence["motion"]["max_base_root_link_displacement_m"] >= MOTION_MINIMUM_DELTA_M
    assert evidence["motion"]["max_fl_foot_link_displacement_m"] >= MOTION_MINIMUM_DELTA_M

    insufficient_hold = module._ButtonAcceptanceAccumulator(
        press_threshold_m=BUTTON_PRESS_THRESHOLD_M,
        hold_steps=BUTTON_HOLD_STEPS,
        rebound_threshold_m=BUTTON_REBOUND_THRESHOLD_M,
        minimum_motion_m=MOTION_MINIMUM_DELTA_M,
    )
    for step in range(1, 7):
        insufficient_hold.update(
            _acceptance_state(
                step,
                displacement=BUTTON_PRESS_THRESHOLD_M if 2 <= step <= 5 else (0.001 if step == 6 else 0.0),
                success=step == 5,
                base_x=0.051 if step == 6 else 0.0,
                fl_x=0.051 if step == 6 else 0.0,
            )
        )
    with pytest.raises(module.ButtonEpisodeRecorderError, match="five action steps"):
        insufficient_hold.assert_full_acceptance()


def test_m10_memory_gate_uses_the_approved_100_step_post_warmup_thresholds() -> None:
    module = _load_recorder_module()
    stable_samples = [
        {
            "action_step": step,
            "timestamp_ns": step * POLICY_DT_NS,
            "gpu_allocated_bytes": 32 * MIB,
            "rss_bytes": 1024 * MIB,
        }
        for step in range(0, 3001, 100)
    ]
    analysis = module._memory_analysis(stable_samples, 3000)
    assert analysis["evaluated"] is True
    assert analysis["gpu"]["slope_bytes_per_100_steps"] <= GPU_SLOPE_LIMIT_BYTES_PER_100_STEPS
    assert analysis["rss"]["slope_bytes_per_100_steps"] <= RSS_SLOPE_LIMIT_BYTES_PER_100_STEPS

    leaking_gpu = [
        {
            **sample,
            "gpu_allocated_bytes": sample["gpu_allocated_bytes"] + 2 * MIB * (sample["action_step"] // 100),
        }
        for sample in stable_samples
    ]
    with pytest.raises(module.ButtonEpisodeRecorderError, match="gpu allocation slope"):
        module._memory_analysis(leaking_gpu, 3000)


def test_offline_validator_recomputes_the_long_memory_gate_from_raw_100_step_samples() -> None:
    samples = [
        {
            "action_step": step,
            "timestamp_ns": step * POLICY_DT_NS,
            "gpu_allocated_bytes": 32 * MIB,
            "rss_bytes": 1024 * MIB,
        }
        for step in range(0, 3001, 100)
    ]
    monitor = {
        "interval_action_steps": 100,
        "analysis_start_action_step": 1000,
        "first_last_window_action_steps": 500,
        "limits": {
            "gpu_slope_mib_per_100_steps": 1.0,
            "gpu_median_delta_mib": 128.0,
            "rss_slope_mib_per_100_steps": 4.0,
            "rss_median_delta_mib": 512.0,
        },
        "samples": samples,
        "analysis": {"evaluated": True},
    }
    assert artifact_schema._validate_memory_monitor({"memory_monitor": monitor}, 3000)["evaluated"] is True

    monitor["samples"] = [
        {
            **sample,
            "rss_bytes": sample["rss_bytes"] + 5 * MIB * (sample["action_step"] // 100),
        }
        for sample in samples
    ]
    with pytest.raises(ButtonEpisodeArtifactError, match="rss allocation slope"):
        artifact_schema._validate_memory_monitor({"memory_monitor": monitor}, 3000)


def test_m10_recorder_keeps_the_physx_only_and_non_overwrite_static_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    contents = (root / "scripts" / "rambo" / "record_button_physx_episode.py").read_text(encoding="utf-8")

    assert "validate_rambo_visualizer_args" in contents
    assert "configure_physx" in contents
    assert "assert_physx_environment" in contents
    assert "use_newton_actuators=False" in contents
    assert "--short-smoke" in contents
    assert "Refusing to overwrite existing output directory" in contents
    assert "MEMORY_MONITOR_INTERVAL_STEPS = 100" in contents
    assert "MEMORY_ANALYSIS_START_STEP = 1000" in contents
    assert "QUADRUPED_CHECKPOINT_SHA256" in contents
    assert "_RgbCpuStaging" in contents
    assert "pin_memory=True" in contents
    assert "copy_from(rgb[0])" in contents
    assert "detach().cpu().numpy()" not in contents
    assert "isaaclab_newton" not in contents
