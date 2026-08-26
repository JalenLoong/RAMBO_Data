"""Pure-Python contracts for LingBot-VA -> RAMBO Dataset V1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rambo.validation import lingbot_dataset_v1 as schema


def _valid_actions() -> dict[str, np.ndarray]:
    raw = np.arange(
        schema.RAW_COMMAND_COUNT * schema.RAW_COMMAND_DIM, dtype=np.float32
    ).reshape(schema.RAW_COMMAND_COUNT, schema.RAW_COMMAND_DIM)
    expected = schema.expected_action_indices()
    return {
        "raw": raw,
        "raw_timestamps": expected["raw_timestamps"],
        "raw_indices": expected["raw_indices"],
        "train": raw[::2].copy(),
        "train_timestamps": expected["train_timestamps"],
        "train_indices": expected["train_indices"],
        "train_source_raw_indices": expected["train_source_raw_indices"],
    }


def test_command_schema_is_exactly_the_approved_nine_columns() -> None:
    assert [column["index"] for column in schema.COMMAND_COLUMNS] == list(range(9))
    assert [column["name"] for column in schema.COMMAND_COLUMNS] == [
        "base_vx",
        "base_vy",
        "base_yaw_rate",
        "fl_ee_x",
        "fl_ee_y",
        "fl_ee_z",
        "fl_ee_fx",
        "fl_ee_fy",
        "fl_ee_fz",
    ]
    assert [column["unit"] for column in schema.COMMAND_COLUMNS] == [
        "m/s", "m/s", "rad/s", "m", "m", "m", "N", "N", "N"
    ]


def test_action_downsample_and_all_time_indices_are_exact() -> None:
    arrays = _valid_actions()
    schema.validate_action_arrays(arrays)
    assert arrays["raw_timestamps"][0] == 0
    assert arrays["raw_timestamps"][-1] == 29_990_000_000
    assert arrays["train_timestamps"][-1] == 29_980_000_000
    assert np.array_equal(arrays["train"], arrays["raw"][::2])


@pytest.mark.parametrize("mutation", ["shape", "downsample", "nan", "timestamp"])
def test_action_validator_rejects_contract_violations(mutation: str) -> None:
    arrays = _valid_actions()
    if mutation == "shape":
        arrays["raw"] = arrays["raw"][:-1]
    elif mutation == "downsample":
        arrays["train"][4, 2] += 1.0
    elif mutation == "nan":
        arrays["raw"][0, 0] = np.nan
    else:
        arrays["train_timestamps"][4] += 1
    with pytest.raises(schema.DatasetValidationError):
        schema.validate_action_arrays(arrays)


def test_rgb_policy_and_50hz_mappings() -> None:
    assert schema.rgb_mapping(0) == {
        "policy_step": 8,
        "timestamp_ns": 80_000_000,
        "preceding_train_indices": [0, 1, 2, 3],
    }
    assert schema.rgb_mapping(374) == {
        "policy_step": 3000,
        "timestamp_ns": 30_000_000_000,
        "preceding_train_indices": [1496, 1497, 1498, 1499],
    }
    assert schema.rgb_frame_index_for_policy_step(1) == 0
    assert schema.rgb_frame_index_for_policy_step(8) == 0
    assert schema.rgb_frame_index_for_policy_step(9) == 1
    assert schema.rgb_frame_index_for_policy_step(3000) == 374


def test_ego_pose_composes_the_fixed_front_camera_mount() -> None:
    root = np.asarray(
        [[0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    actual = schema.ego_pose_from_root_pose(root)
    np.testing.assert_allclose(actual[0], [0.3, 0.0, 0.48, 0.0, 0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(actual[1], [0.7, 2.0, 3.08, 0.0, 0.0, 1.0, 0.0], atol=1e-6)


def test_button_event_steps_are_derived_not_hard_coded() -> None:
    displacement = np.zeros(schema.POLICY_STEPS, dtype=np.float32)
    displacement[99:109] = 0.013
    success = np.zeros(schema.POLICY_STEPS, dtype=np.bool_)
    success[103:] = True
    displacement[109:] = 0.001
    events = schema.derive_button_events(displacement, success)
    assert events["first_press_threshold_step"] == 100
    assert events["computed_hold_success_step"] == 104
    assert events["detector_success_step"] == 104
    assert events["first_rebound_step"] == 110


def test_checksum_manifest_has_exact_coverage_and_detects_tamper(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "nested" / "b.txt").write_text("b", encoding="utf-8")
    schema.write_checksums(tmp_path)
    schema._validate_checksums(tmp_path)
    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(schema.DatasetValidationError, match="checksum mismatch"):
        schema._validate_checksums(tmp_path)


def test_wrong_frame_mapping_is_rejected_before_image_io(tmp_path: Path) -> None:
    frames = []
    for index in range(schema.RGB_FRAME_COUNT):
        mapping = schema.rgb_mapping(index)
        frames.append(
            {
                "frame_index": index,
                "policy_step": mapping["policy_step"],
                "timestamp_ns": mapping["timestamp_ns"],
                "preceding_train_indices": mapping["preceding_train_indices"],
                "views": {
                    view: {
                        "path": f"video/{view}/frame_{index:06d}.png",
                        "sensor_frame_id": index,
                    }
                    for view in schema.VIEWS
                },
            }
        )
    frames[0]["policy_step"] = 7
    (tmp_path / "video").mkdir()
    schema.write_json(tmp_path / "video" / "frame_index.json", {"schema_version": 1, "frames": frames})
    with pytest.raises(schema.DatasetValidationError, match="policy_step"):
        schema._validate_frame_index(tmp_path, verify_images=False)


def test_black_png_is_rejected(tmp_path: Path) -> None:
    frames = []
    for index in range(schema.RGB_FRAME_COUNT):
        mapping = schema.rgb_mapping(index)
        frames.append(
            {
                "frame_index": index,
                "policy_step": mapping["policy_step"],
                "timestamp_ns": mapping["timestamp_ns"],
                "preceding_train_indices": mapping["preceding_train_indices"],
                "views": {
                    view: {
                        "path": f"video/{view}/frame_{index:06d}.png",
                        "sensor_frame_id": index,
                    }
                    for view in schema.VIEWS
                },
            }
        )
    for view in schema.VIEWS:
        directory = tmp_path / "video" / view
        directory.mkdir(parents=True)
        Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(directory / "frame_000000.png")
    schema.write_json(tmp_path / "video" / "frame_index.json", {"schema_version": 1, "frames": frames})
    with pytest.raises(schema.DatasetValidationError, match="black"):
        schema._validate_frame_index(tmp_path, verify_images=True)


def test_failed_outcome_is_rejected(tmp_path: Path) -> None:
    metadata = {
        "dataset": schema.DATASET_NAME,
        "dataset_version": schema.DATASET_VERSION,
        "schema_version": schema.SCHEMA_VERSION,
        "episode_id": schema.EPISODE_ID,
        "task_id": schema.TASK_ID,
        "task_instruction": schema.TASK_INSTRUCTION,
        "seed": schema.SEED,
        "duration_s": 30.0,
        "counts": {
            "raw_commands": 3000,
            "train_commands": 1500,
            "policy_steps": 3000,
            "rgb_frames_per_view": 375,
        },
        "command_columns": list(schema.COMMAND_COLUMNS),
    }
    task = {
        "instruction": schema.TASK_INSTRUCTION,
        "task_id": schema.TASK_ID,
        "command_program_id": schema.COMMAND_PROGRAM_ID,
        "goal": {"type": "button_press"},
        "button": {
            "stroke_m": 0.02,
            "press_threshold_m": 0.012,
            "release_threshold_m": 0.002,
            "hold_policy_steps": 5,
            "contact_guard_m": 0.002,
            "status_marker_rendered": False,
        },
        "initial_state": {
            "robot_root_pose_w_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "button_cap_pose_w_xyzw": [0.99, 0.142, 0.30, 0.0, 0.0, 0.0, 1.0],
        },
    }
    outcome = {
        "success": False,
        "terminal": False,
        "terminal_reason": "fixed_duration_complete",
        "duration_s": 30.0,
    }
    sync = {
        "raw_command_index_base": 0,
        "policy_step_index_base": 1,
        "rgb_frame_index_base": 0,
        "command_timestamp_semantics": "interval_start",
        "rgb_timestamp_semantics": "post_step",
    }
    for name, value in (("metadata", metadata), ("task", task), ("outcome", outcome), ("sync", sync)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(schema.DatasetValidationError, match="outcome.success"):
        schema._validate_metadata(tmp_path)
