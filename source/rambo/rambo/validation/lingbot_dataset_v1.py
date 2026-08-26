"""Pure-Python schema and validation for LingBot-VA -> RAMBO Dataset V1.

This module deliberately has no Isaac Sim imports.  A completed episode can be
validated on an ordinary Python host from its on-disk PNG, NumPy, and JSON
artifacts alone.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import numpy as np


DATASET_NAME = "lingbot_rambo"
DATASET_VERSION = "1.0"
SCHEMA_VERSION = 1
EPISODE_ID = "episode_000001"
TASK_ID = "Isaac-RAMBO-Quadruped-Button-Go2-v0"
TASK_INSTRUCTION = "Press the button"
COMMAND_PROGRAM_ID = "rambo-button-script-v1"
SEED = 42

POLICY_STEPS = 3_000
POLICY_RATE_HZ = 100
POLICY_DT_NS = 10_000_000
TRAIN_RATE_HZ = 50
TRAIN_DT_NS = 20_000_000
PHYSICS_RATE_HZ = 500
PHYSICS_DT_NS = 2_000_000
CONTROL_DECIMATION = 5
RGB_RATE_HZ = 12.5
RGB_INTERVAL_POLICY_STEPS = 8
RGB_DT_NS = 80_000_000
RAW_COMMAND_DIM = 9
POLICY_ACTION_DIM = 18
OBSERVATION_DIM = 405
RAW_COMMAND_COUNT = POLICY_STEPS
TRAIN_COMMAND_COUNT = POLICY_STEPS // 2
RGB_FRAME_COUNT = POLICY_STEPS // RGB_INTERVAL_POLICY_STEPS
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
VIEWS = ("ego", "left", "right")

BUTTON_PRESS_THRESHOLD_M = 0.012
BUTTON_RELEASE_THRESHOLD_M = 0.002
BUTTON_HOLD_STEPS = 5

COMMAND_COLUMNS = (
    {"index": 0, "name": "base_vx", "unit": "m/s", "frame": "projected_com_controller"},
    {"index": 1, "name": "base_vy", "unit": "m/s", "frame": "projected_com_controller"},
    {"index": 2, "name": "base_yaw_rate", "unit": "rad/s", "frame": "projected_com_controller"},
    {"index": 3, "name": "fl_ee_x", "unit": "m", "frame": "projected_com"},
    {"index": 4, "name": "fl_ee_y", "unit": "m", "frame": "projected_com"},
    {"index": 5, "name": "fl_ee_z", "unit": "m", "frame": "projected_com"},
    {"index": 6, "name": "fl_ee_fx", "unit": "N", "frame": "projected_com"},
    {"index": 7, "name": "fl_ee_fy", "unit": "N", "frame": "projected_com"},
    {"index": 8, "name": "fl_ee_fz", "unit": "N", "frame": "projected_com"},
)

ACTION_FILES = {
    "raw": "actions/high_level_command_raw.npy",
    "raw_timestamps": "actions/high_level_command_raw_timestamp_ns.npy",
    "raw_indices": "actions/high_level_command_raw_index.npy",
    "train": "actions/high_level_command_train.npy",
    "train_timestamps": "actions/high_level_command_train_timestamp_ns.npy",
    "train_indices": "actions/high_level_command_train_index.npy",
    "train_source_raw_indices": "actions/high_level_command_train_source_raw_index.npy",
}

RAW_DEBUG_FILES = {
    "policy_action": "raw_debug/policy_action_18d.npy",
    "observation": "raw_debug/observation_405d.npy",
    "reward": "raw_debug/reward.npy",
    "root_pose": "raw_debug/root_pose_w.npy",
    "joint_pos": "raw_debug/joint_pos.npy",
    "joint_vel": "raw_debug/joint_vel.npy",
    "fl_foot_position": "raw_debug/fl_foot_position_w.npy",
    "button_displacement": "raw_debug/button_displacement_m.npy",
    "button_success": "raw_debug/button_success.npy",
    "button_released": "raw_debug/button_released.npy",
    "terminal": "raw_debug/terminal.npy",
    "contact_max_force": "raw_debug/contact_max_force_n.npy",
}


class DatasetValidationError(RuntimeError):
    """Raised when an episode violates the Dataset V1 golden-sample contract."""


def write_json(path: Path, value: Any) -> Path:
    """Write stable UTF-8 JSON and return the resolved path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(episode_dir: Path) -> Path:
    """Hash every episode file except the checksum manifest itself."""

    episode_dir = episode_dir.resolve()
    manifest = episode_dir / "checksums.sha256"
    paths = sorted(
        path for path in episode_dir.rglob("*") if path.is_file() and path.resolve() != manifest
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(episode_dir).as_posix()}" for path in paths]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def expected_action_indices() -> dict[str, np.ndarray]:
    raw_indices = np.arange(RAW_COMMAND_COUNT, dtype=np.int64)
    train_indices = np.arange(TRAIN_COMMAND_COUNT, dtype=np.int64)
    return {
        "raw_indices": raw_indices,
        "raw_timestamps": raw_indices * POLICY_DT_NS,
        "train_indices": train_indices,
        "train_timestamps": train_indices * TRAIN_DT_NS,
        "train_source_raw_indices": train_indices * 2,
    }


def rgb_frame_index_for_policy_step(policy_step: int) -> int:
    """Return the first RGB frame at or after a 1-based policy step."""

    if not 1 <= policy_step <= POLICY_STEPS:
        raise ValueError(f"policy_step must be in [1, {POLICY_STEPS}], got {policy_step}")
    return min(
        RGB_FRAME_COUNT - 1,
        math.ceil(policy_step / RGB_INTERVAL_POLICY_STEPS) - 1,
    )


def rgb_mapping(frame_index: int) -> dict[str, int | list[int]]:
    if not 0 <= frame_index < RGB_FRAME_COUNT:
        raise ValueError(f"frame_index must be in [0, {RGB_FRAME_COUNT - 1}], got {frame_index}")
    policy_step = RGB_INTERVAL_POLICY_STEPS * (frame_index + 1)
    first_train = 4 * frame_index
    return {
        "policy_step": policy_step,
        "timestamp_ns": policy_step * POLICY_DT_NS,
        "preceding_train_indices": list(range(first_train, first_train + 4)),
    }


def ego_pose_from_root_pose(root_pose: np.ndarray) -> np.ndarray:
    """Compose the fixed Go2 front-camera mount with xyzw root poses."""

    root_pose = np.asarray(root_pose, dtype=np.float32)
    if root_pose.shape[-1] != 7 or not np.isfinite(root_pose).all():
        raise DatasetValidationError("root pose must end in one finite [xyz,xyzw] vector")
    quaternion = root_pose[..., 3:7].copy()
    norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if bool(np.any(norms <= 0.0)):
        raise DatasetValidationError("root pose contains a zero quaternion")
    quaternion /= norms
    xyz = quaternion[..., :3]
    w = quaternion[..., 3:4]
    offset = np.broadcast_to(np.asarray([0.30, 0.0, 0.08], dtype=np.float32), xyz.shape)
    cross = 2.0 * np.cross(xyz, offset)
    camera_position = root_pose[..., :3] + offset + w * cross + np.cross(xyz, cross)
    return np.concatenate((camera_position, quaternion), axis=-1).astype(np.float32)


def derive_button_events(
    displacement: np.ndarray,
    success: np.ndarray,
    *,
    press_threshold_m: float = BUTTON_PRESS_THRESHOLD_M,
    release_threshold_m: float = BUTTON_RELEASE_THRESHOLD_M,
    hold_steps: int = BUTTON_HOLD_STEPS,
) -> dict[str, int | float | None | bool]:
    """Recompute press, detector success, and post-success rebound from traces."""

    displacement = np.asarray(displacement)
    success = np.asarray(success)
    if displacement.shape != (POLICY_STEPS,):
        raise DatasetValidationError(f"button displacement shape is {displacement.shape}, expected {(POLICY_STEPS,)}")
    if success.shape != (POLICY_STEPS,):
        raise DatasetValidationError(f"button success shape is {success.shape}, expected {(POLICY_STEPS,)}")
    if not np.issubdtype(displacement.dtype, np.floating) or not np.isfinite(displacement).all():
        raise DatasetValidationError("button displacement must be a finite floating-point trace")
    if success.dtype != np.bool_:
        raise DatasetValidationError(f"button success dtype is {success.dtype}, expected bool")

    threshold_indices = np.flatnonzero(displacement >= press_threshold_m)
    first_threshold_step = int(threshold_indices[0] + 1) if threshold_indices.size else None
    consecutive = 0
    computed_success_step: int | None = None
    for index, value in enumerate(displacement):
        consecutive = consecutive + 1 if float(value) >= press_threshold_m else 0
        if consecutive >= hold_steps:
            computed_success_step = index + 1
            break
    detector_indices = np.flatnonzero(success)
    detector_success_step = int(detector_indices[0] + 1) if detector_indices.size else None
    rebound_step: int | None = None
    if detector_success_step is not None:
        rebound_indices = np.flatnonzero(
            displacement[detector_success_step:] <= release_threshold_m
        )
        if rebound_indices.size:
            rebound_step = detector_success_step + int(rebound_indices[0]) + 1
    return {
        "success": detector_success_step is not None,
        "first_press_threshold_step": first_threshold_step,
        "computed_hold_success_step": computed_success_step,
        "detector_success_step": detector_success_step,
        "first_rebound_step": rebound_step,
        "max_displacement_m": float(np.max(displacement)),
    }


def validate_action_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    required = set(ACTION_FILES)
    if set(arrays) != required:
        raise DatasetValidationError(
            f"action array keys differ: missing={sorted(required - set(arrays))}, "
            f"extra={sorted(set(arrays) - required)}"
        )
    raw = arrays["raw"]
    train = arrays["train"]
    if raw.shape != (RAW_COMMAND_COUNT, RAW_COMMAND_DIM) or raw.dtype != np.float32:
        raise DatasetValidationError(
            f"raw command must be float32 {(RAW_COMMAND_COUNT, RAW_COMMAND_DIM)}, got {raw.dtype} {raw.shape}"
        )
    if train.shape != (TRAIN_COMMAND_COUNT, RAW_COMMAND_DIM) or train.dtype != np.float32:
        raise DatasetValidationError(
            f"train command must be float32 {(TRAIN_COMMAND_COUNT, RAW_COMMAND_DIM)}, got {train.dtype} {train.shape}"
        )
    if not np.isfinite(raw).all() or not np.isfinite(train).all():
        raise DatasetValidationError("command arrays contain NaN or Inf")
    expected = expected_action_indices()
    for key in (
        "raw_timestamps",
        "raw_indices",
        "train_timestamps",
        "train_indices",
        "train_source_raw_indices",
    ):
        value = arrays[key]
        if value.dtype != np.int64 or value.shape != expected[key].shape:
            raise DatasetValidationError(
                f"{key} must be int64 {expected[key].shape}, got {value.dtype} {value.shape}"
            )
        if not np.array_equal(value, expected[key]):
            raise DatasetValidationError(f"{key} does not match the exact Dataset V1 index/time formula")
        if "timestamps" in key and not bool(np.all(np.diff(value) > 0)):
            raise DatasetValidationError(f"{key} is not strictly increasing")
    if not np.array_equal(train, raw[::2]):
        raise DatasetValidationError("train command is not exactly raw[::2]")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetValidationError(f"cannot read valid JSON from {path}: {error}") from error


def _load_npy(root: Path, relative: str) -> np.ndarray:
    path = root / relative
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise DatasetValidationError(f"cannot load NumPy array {relative}: {error}") from error
    if not isinstance(value, np.ndarray):
        raise DatasetValidationError(f"{relative} did not contain a NumPy array")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise DatasetValidationError(f"{label} is {actual!r}, expected {expected!r}")


def _validate_checksums(root: Path) -> None:
    manifest = root / "checksums.sha256"
    if not manifest.is_file():
        raise DatasetValidationError("checksums.sha256 is missing")
    recorded: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise DatasetValidationError(f"malformed checksum line {line_number}")
        digest, relative = parts
        try:
            int(digest, 16)
        except ValueError as error:
            raise DatasetValidationError(f"non-hex checksum on line {line_number}") from error
        if relative in recorded or relative == "checksums.sha256" or relative.startswith("/") or ".." in Path(relative).parts:
            raise DatasetValidationError(f"invalid or duplicate checksum path: {relative!r}")
        recorded[relative] = digest
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    if set(recorded) != actual_paths:
        raise DatasetValidationError(
            "checksum coverage differs: "
            f"missing={sorted(actual_paths - set(recorded))}, extra={sorted(set(recorded) - actual_paths)}"
        )
    for relative, expected in recorded.items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise DatasetValidationError(f"checksum mismatch for {relative}")


def _validate_metadata(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata = _load_json(root / "metadata.json")
    task = _load_json(root / "task.json")
    outcome = _load_json(root / "outcome.json")
    sync = _load_json(root / "sync.json")
    _require_equal(metadata.get("dataset"), DATASET_NAME, "metadata.dataset")
    _require_equal(metadata.get("dataset_version"), DATASET_VERSION, "metadata.dataset_version")
    _require_equal(metadata.get("schema_version"), SCHEMA_VERSION, "metadata.schema_version")
    episode_id = metadata.get("episode_id")
    if not isinstance(episode_id, str) or re.fullmatch(r"episode_\d{6}", episode_id) is None:
        raise DatasetValidationError("metadata.episode_id must match episode_000001-style naming")
    if root.name.startswith("episode_"):
        _require_equal(episode_id, root.name.removesuffix(".inprogress"), "metadata.episode_id")
    _require_equal(metadata.get("task_id"), TASK_ID, "metadata.task_id")
    _require_equal(metadata.get("task_instruction"), TASK_INSTRUCTION, "metadata.task_instruction")
    seed = metadata.get("seed")
    if not isinstance(seed, int) or seed < 0:
        raise DatasetValidationError("metadata.seed must be a non-negative integer")
    episode_number = int(episode_id.rsplit("_", 1)[1])
    if episode_number == 1:
        _require_equal(seed, SEED, "metadata.seed")
    else:
        _require_equal(seed, episode_number, "metadata.seed")
        _require_equal(metadata.get("runtime_seed"), SEED, "metadata.runtime_seed")
    _require_equal(metadata.get("duration_s"), 30.0, "metadata.duration_s")
    counts = metadata.get("counts", {})
    for key, expected in {
        "raw_commands": RAW_COMMAND_COUNT,
        "train_commands": TRAIN_COMMAND_COUNT,
        "policy_steps": POLICY_STEPS,
        "rgb_frames_per_view": RGB_FRAME_COUNT,
    }.items():
        _require_equal(counts.get(key), expected, f"metadata.counts.{key}")
    _require_equal(metadata.get("command_columns"), list(COMMAND_COLUMNS), "metadata.command_columns")
    _require_equal(task.get("instruction"), TASK_INSTRUCTION, "task.instruction")
    _require_equal(task.get("task_id"), TASK_ID, "task.task_id")
    _require_equal(task.get("command_program_id"), COMMAND_PROGRAM_ID, "task.command_program_id")
    _require_equal(task.get("goal", {}).get("type"), "button_press", "task.goal.type")
    button = task.get("button", {})
    for key, expected in {
        "stroke_m": 0.02,
        "press_threshold_m": BUTTON_PRESS_THRESHOLD_M,
        "release_threshold_m": BUTTON_RELEASE_THRESHOLD_M,
        "hold_policy_steps": BUTTON_HOLD_STEPS,
        "contact_guard_m": 0.002,
        "status_marker_rendered": False,
    }.items():
        _require_equal(button.get(key), expected, f"task.button.{key}")
    initial_state = task.get("initial_state", {})
    for key in ("robot_root_pose_w_xyzw", "button_cap_pose_w_xyzw"):
        pose = np.asarray(initial_state.get(key), dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise DatasetValidationError(f"task.initial_state.{key} must be one finite 7D pose")
    randomization = metadata.get("randomization")
    if episode_number == 1:
        if randomization not in (None, {"enabled": False}):
            raise DatasetValidationError("golden episode must not declare button randomization")
    else:
        if not isinstance(randomization, dict) or randomization.get("enabled") is not True:
            raise DatasetValidationError("randomized episode must declare enabled randomization metadata")
        _require_equal(randomization.get("parameter"), "button_position", "metadata.randomization.parameter")
        _require_equal(randomization.get("seed"), seed, "metadata.randomization.seed")
        for key in ("reference_position", "sampled_position", "delta"):
            value = np.asarray(randomization.get(key), dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise DatasetValidationError(f"metadata.randomization.{key} must be a finite XYZ vector")
        if not np.allclose(
            np.asarray(randomization["sampled_position"]), np.asarray(initial_state["button_cap_pose_w_xyzw"])[:3], atol=1e-6
        ):
            raise DatasetValidationError("randomized sampled Button position differs from task initial cap pose")
    _require_equal(outcome.get("success"), True, "outcome.success")
    _require_equal(outcome.get("terminal"), False, "outcome.terminal")
    _require_equal(outcome.get("terminal_reason"), "fixed_duration_complete", "outcome.terminal_reason")
    _require_equal(outcome.get("duration_s"), 30.0, "outcome.duration_s")
    _require_equal(sync.get("raw_command_index_base"), 0, "sync.raw_command_index_base")
    _require_equal(sync.get("policy_step_index_base"), 1, "sync.policy_step_index_base")
    _require_equal(sync.get("rgb_frame_index_base"), 0, "sync.rgb_frame_index_base")
    _require_equal(sync.get("command_timestamp_semantics"), "interval_start", "sync.command_timestamp_semantics")
    _require_equal(sync.get("rgb_timestamp_semantics"), "post_step", "sync.rgb_timestamp_semantics")
    return metadata, task, outcome


def _validate_frame_index(root: Path, *, verify_images: bool) -> dict[str, Any]:
    document = _load_json(root / "video" / "frame_index.json")
    _require_equal(document.get("schema_version"), SCHEMA_VERSION, "frame_index.schema_version")
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != RGB_FRAME_COUNT:
        raise DatasetValidationError(f"frame_index must contain {RGB_FRAME_COUNT} frames")
    previous_sensor_ids: dict[str, int | None] = {view: None for view in VIEWS}
    image_stats: dict[str, dict[str, float]] = {}
    if verify_images:
        try:
            from PIL import Image
        except ModuleNotFoundError as error:
            raise DatasetValidationError("Pillow is required for PNG validation") from error
    for index, frame in enumerate(frames):
        mapping = rgb_mapping(index)
        _require_equal(frame.get("frame_index"), index, f"frame[{index}].frame_index")
        _require_equal(frame.get("policy_step"), mapping["policy_step"], f"frame[{index}].policy_step")
        _require_equal(frame.get("timestamp_ns"), mapping["timestamp_ns"], f"frame[{index}].timestamp_ns")
        _require_equal(
            frame.get("preceding_train_indices"),
            mapping["preceding_train_indices"],
            f"frame[{index}].preceding_train_indices",
        )
        views = frame.get("views")
        if not isinstance(views, dict) or set(views) != set(VIEWS):
            raise DatasetValidationError(f"frame[{index}] does not contain exactly ego/left/right")
        for view in VIEWS:
            record = views[view]
            expected_path = f"video/{view}/frame_{index:06d}.png"
            _require_equal(record.get("path"), expected_path, f"frame[{index}].{view}.path")
            frame_id = record.get("sensor_frame_id")
            if not isinstance(frame_id, int):
                raise DatasetValidationError(f"frame[{index}].{view}.sensor_frame_id is not an integer")
            previous = previous_sensor_ids[view]
            if previous is not None and frame_id != previous + 1:
                raise DatasetValidationError(
                    f"{view} sensor frame ID jumped from {previous} to {frame_id} at RGB frame {index}"
                )
            previous_sensor_ids[view] = frame_id
            path = root / expected_path
            if not path.is_file():
                raise DatasetValidationError(f"missing PNG: {expected_path}")
            if verify_images:
                try:
                    with Image.open(path) as loaded:
                        loaded.load()
                        array = np.asarray(loaded.convert("RGB"))
                except (OSError, ValueError) as error:
                    raise DatasetValidationError(f"unreadable PNG {expected_path}: {error}") from error
                if array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3) or array.dtype != np.uint8:
                    raise DatasetValidationError(
                        f"PNG {expected_path} has {array.dtype} {array.shape}, expected uint8 {(IMAGE_HEIGHT, IMAGE_WIDTH, 3)}"
                    )
                mean = float(array.mean())
                std = float(array.std())
                if not math.isfinite(mean) or not math.isfinite(std) or mean <= 2.0 or std <= 1.0:
                    raise DatasetValidationError(f"PNG {expected_path} is black/non-finite: mean={mean}, std={std}")
                image_stats[expected_path] = {"mean": mean, "std": std}
    for view in VIEWS:
        expected_names = {f"frame_{index:06d}.png" for index in range(RGB_FRAME_COUNT)}
        actual_names = {path.name for path in (root / "video" / view).glob("*.png") if path.is_file()}
        if actual_names != expected_names:
            raise DatasetValidationError(
                f"{view} PNG set differs: missing={sorted(expected_names - actual_names)}, "
                f"extra={sorted(actual_names - expected_names)}"
            )
    return {"image_stats": image_stats, "last_sensor_frame_ids": previous_sensor_ids}


def _validate_cameras(root: Path, metadata: dict[str, Any]) -> None:
    camera_metadata = metadata.get("cameras")
    if not isinstance(camera_metadata, dict) or set(camera_metadata) != set(VIEWS):
        raise DatasetValidationError("metadata.cameras must contain exactly ego/left/right")
    for view in VIEWS:
        record = camera_metadata[view]
        pose_rel = record.get("pose_trace_file")
        intrinsics_rel = record.get("intrinsics_file")
        expected_pose = f"video/camera/{view}_pose_w.npy"
        expected_intrinsics = f"video/camera/{view}_intrinsics.npy"
        _require_equal(pose_rel, expected_pose, f"metadata.cameras.{view}.pose_trace_file")
        _require_equal(intrinsics_rel, expected_intrinsics, f"metadata.cameras.{view}.intrinsics_file")
        _require_equal(record.get("resolution"), [IMAGE_WIDTH, IMAGE_HEIGHT], f"metadata.cameras.{view}.resolution")
        for key, expected in {
            "focal_length_mm": 18.0,
            "horizontal_aperture_mm": 20.955,
            "clipping_range_m": [0.1, 20.0],
            "renderer": "IsaacRtxRendererCfg",
            "world_fixed": view != "ego",
        }.items():
            _require_equal(record.get(key), expected, f"metadata.cameras.{view}.{key}")
        pose = _load_npy(root, expected_pose)
        intrinsics = _load_npy(root, expected_intrinsics)
        if pose.shape != (RGB_FRAME_COUNT, 7) or pose.dtype != np.float32 or not np.isfinite(pose).all():
            raise DatasetValidationError(
                f"{view} pose must be finite float32 {(RGB_FRAME_COUNT, 7)}, got {pose.dtype} {pose.shape}"
            )
        if intrinsics.shape != (3, 3) or intrinsics.dtype != np.float32 or not np.isfinite(intrinsics).all():
            raise DatasetValidationError(f"{view} intrinsics must be finite float32 (3, 3)")
        if float(intrinsics[0, 0]) <= 0 or float(intrinsics[1, 1]) <= 0 or not np.isclose(intrinsics[2, 2], 1.0):
            raise DatasetValidationError(f"{view} intrinsics are not a valid pinhole matrix")
        norms = np.linalg.norm(pose[:, 3:7], axis=1)
        if not np.allclose(norms, 1.0, atol=1.0e-4, rtol=0.0):
            raise DatasetValidationError(f"{view} quaternion trace is not normalized")
        if view in ("left", "right") and not np.allclose(pose, pose[0], atol=1.0e-6, rtol=0.0):
            raise DatasetValidationError(f"{view} camera is not world-fixed")
        if view in ("left", "right"):
            expected_eye = [0.45, 2.40, 1.20] if view == "left" else [0.45, -2.40, 1.20]
            _require_equal(record.get("configured_eye_w_m"), expected_eye, f"metadata.cameras.{view}.configured_eye_w_m")
            _require_equal(record.get("configured_target_w_m"), [0.60, 0.00, 0.35], f"metadata.cameras.{view}.configured_target_w_m")
            if not np.allclose(pose[0, :3], np.asarray(expected_eye), atol=1.0e-5, rtol=0.0):
                raise DatasetValidationError(f"{view} actual world position differs from its fixed configured eye")
        else:
            root_pose = _load_npy(root, RAW_DEBUG_FILES["root_pose"])
            expected_ego = ego_pose_from_root_pose(root_pose[np.arange(7, POLICY_STEPS, 8)])
            if not np.allclose(pose, expected_ego, atol=1.0e-5, rtol=0.0):
                raise DatasetValidationError("ego pose trace differs from post-step root pose plus its fixed mount")


def _validate_mp4(root: Path, view: str) -> None:
    path = root / "video" / f"{view}.mp4"
    if not path.is_file() or path.stat().st_size <= 0:
        raise DatasetValidationError(f"missing or empty convenience video: {path.relative_to(root)}")
    try:
        process = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DatasetValidationError(f"cannot inspect {view}.mp4 with ffprobe: {error}") from error
    if process.returncode != 0:
        raise DatasetValidationError(f"ffprobe rejected {view}.mp4: {process.stderr.strip()}")
    try:
        streams = json.loads(process.stdout).get("streams", [])
    except json.JSONDecodeError as error:
        raise DatasetValidationError(f"ffprobe emitted invalid JSON for {view}.mp4") from error
    if len(streams) != 1:
        raise DatasetValidationError(f"{view}.mp4 does not contain exactly one selected video stream")
    stream = streams[0]
    expected = {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "avg_frame_rate": "25/2",
        "nb_frames": str(RGB_FRAME_COUNT),
    }
    for key, value in expected.items():
        _require_equal(stream.get(key), value, f"video/{view}.mp4 {key}")


def _validate_debug_and_outcome(root: Path, metadata: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    arrays = {key: _load_npy(root, relative) for key, relative in RAW_DEBUG_FILES.items()}
    expected_shapes: dict[str, tuple[int, ...]] = {
        "policy_action": (POLICY_STEPS, POLICY_ACTION_DIM),
        "observation": (POLICY_STEPS, OBSERVATION_DIM),
        "reward": (POLICY_STEPS,),
        "root_pose": (POLICY_STEPS, 7),
        "fl_foot_position": (POLICY_STEPS, 3),
        "button_displacement": (POLICY_STEPS,),
        "button_success": (POLICY_STEPS,),
        "button_released": (POLICY_STEPS,),
        "terminal": (POLICY_STEPS,),
        "contact_max_force": (POLICY_STEPS,),
    }
    joint_dimension = int(metadata.get("dimensions", {}).get("joint_state", -1))
    expected_shapes["joint_pos"] = (POLICY_STEPS, joint_dimension)
    expected_shapes["joint_vel"] = (POLICY_STEPS, joint_dimension)
    bool_keys = {"button_success", "button_released", "terminal"}
    for key, value in arrays.items():
        if value.shape != expected_shapes[key]:
            raise DatasetValidationError(f"{key} shape is {value.shape}, expected {expected_shapes[key]}")
        if key in bool_keys:
            if value.dtype != np.bool_:
                raise DatasetValidationError(f"{key} dtype is {value.dtype}, expected bool")
        elif value.dtype != np.float32:
            raise DatasetValidationError(f"{key} dtype is {value.dtype}, expected float32")
        if key not in bool_keys and not np.isfinite(value).all():
            raise DatasetValidationError(f"{key} contains NaN or Inf")
    if bool(np.any(arrays["terminal"])):
        raise DatasetValidationError("episode contains a terminal policy step")
    events = derive_button_events(arrays["button_displacement"], arrays["button_success"])
    if not events["success"]:
        raise DatasetValidationError("button detector never reported success")
    if events["computed_hold_success_step"] != events["detector_success_step"]:
        raise DatasetValidationError(
            "button detector success step differs from the first five-step >=12 mm displacement run"
        )
    if events["first_rebound_step"] is None:
        raise DatasetValidationError("button never rebounded to <=2 mm after success")
    detector_step = int(events["detector_success_step"])
    rebound_step = int(events["first_rebound_step"])
    _require_equal(outcome.get("first_success_policy_step"), detector_step, "outcome.first_success_policy_step")
    _require_equal(outcome.get("first_success_timestamp_ns"), detector_step * POLICY_DT_NS, "outcome.first_success_timestamp_ns")
    _require_equal(outcome.get("first_rebound_policy_step"), rebound_step, "outcome.first_rebound_policy_step")
    _require_equal(outcome.get("first_rebound_timestamp_ns"), rebound_step * POLICY_DT_NS, "outcome.first_rebound_timestamp_ns")
    if float(events["max_displacement_m"]) < BUTTON_PRESS_THRESHOLD_M:
        raise DatasetValidationError("maximum button displacement did not reach 12 mm")
    return events


def validate_episode(
    episode_dir: str | Path,
    *,
    verify_checksums: bool = True,
    verify_images: bool = True,
) -> dict[str, Any]:
    """Validate one complete golden sample and return a machine-readable summary."""

    root = Path(episode_dir).expanduser().resolve()
    if not root.is_dir():
        raise DatasetValidationError(f"episode directory does not exist: {root}")
    metadata, _task, outcome = _validate_metadata(root)
    action_arrays = {key: _load_npy(root, relative) for key, relative in ACTION_FILES.items()}
    validate_action_arrays(action_arrays)
    frame_summary = _validate_frame_index(root, verify_images=verify_images)
    _validate_cameras(root, metadata)
    events = _validate_debug_and_outcome(root, metadata, outcome)
    for view in VIEWS:
        _validate_mp4(root, view)
    contact_sheet = root / "contact_sheet.png"
    if not contact_sheet.is_file() or contact_sheet.stat().st_size <= 0:
        raise DatasetValidationError("contact_sheet.png is missing or empty")
    if verify_checksums:
        _validate_checksums(root)
    return {
        "passed": True,
        "episode_dir": str(root),
        "episode_id": metadata["episode_id"],
        "instruction": TASK_INSTRUCTION,
        "raw_commands": RAW_COMMAND_COUNT,
        "train_commands": TRAIN_COMMAND_COUNT,
        "rgb_frames_per_view": RGB_FRAME_COUNT,
        "success_policy_step": events["detector_success_step"],
        "rebound_policy_step": events["first_rebound_step"],
        "max_button_displacement_m": events["max_displacement_m"],
        "last_sensor_frame_ids": frame_summary["last_sensor_frame_ids"],
        "checksum_verified": verify_checksums,
        "images_verified": verify_images,
    }


__all__ = (
    "ACTION_FILES",
    "BUTTON_HOLD_STEPS",
    "BUTTON_PRESS_THRESHOLD_M",
    "BUTTON_RELEASE_THRESHOLD_M",
    "COMMAND_COLUMNS",
    "COMMAND_PROGRAM_ID",
    "CONTROL_DECIMATION",
    "DATASET_NAME",
    "DATASET_VERSION",
    "DatasetValidationError",
    "EPISODE_ID",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "OBSERVATION_DIM",
    "PHYSICS_DT_NS",
    "PHYSICS_RATE_HZ",
    "POLICY_ACTION_DIM",
    "POLICY_DT_NS",
    "POLICY_RATE_HZ",
    "POLICY_STEPS",
    "RAW_COMMAND_COUNT",
    "RAW_COMMAND_DIM",
    "RAW_DEBUG_FILES",
    "RGB_DT_NS",
    "RGB_FRAME_COUNT",
    "RGB_INTERVAL_POLICY_STEPS",
    "RGB_RATE_HZ",
    "SCHEMA_VERSION",
    "SEED",
    "TASK_ID",
    "TASK_INSTRUCTION",
    "TRAIN_COMMAND_COUNT",
    "TRAIN_DT_NS",
    "TRAIN_RATE_HZ",
    "VIEWS",
    "derive_button_events",
    "ego_pose_from_root_pose",
    "expected_action_indices",
    "rgb_frame_index_for_policy_step",
    "rgb_mapping",
    "sha256_file",
    "validate_action_arrays",
    "validate_episode",
    "write_checksums",
    "write_json",
)
