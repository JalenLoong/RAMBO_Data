"""Pure-Python validation shared by the three procedural LingBot object tasks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import lingbot_dataset_v1 as common


PROFILES: dict[str, dict[str, Any]] = {
    "lift_basket": {
        "task_id": "Isaac-RAMBO-Quadruped-Lift-Basket-Go2-v0", "instruction": "Lift the basket",
        "command_program_id": "rambo-lift-basket-script-v1", "goal_type": "basket_lift", "primary_key": "basket", "hold_steps": 25,
        "eyes": {"left": [0.50, 2.40, 1.25], "right": [0.50, -2.40, 1.25]}, "target": [0.65, 0.10, 0.25],
        "geometry": {"basket_size_m": [0.34, 0.28, 0.18], "basket_mass_kg": 0.10, "u_handle_crossbar_size_m": [0.35, 0.18, 0.035], "u_handle_mass_kg": 0.03, "vertical_prismatic_guide_stroke_m": 0.35, "friction": "PhysX default material", "success": {"minimum_bottom_lift_m": 0.06, "maximum_tilt_deg": 35.0, "hold_policy_steps": 25}},
    },
    "pull_object_into_basket": {
        "task_id": "Isaac-RAMBO-Quadruped-Pull-Object-Into-Basket-Go2-v0", "instruction": "Pull the object on the table into the basket",
        "command_program_id": "rambo-pull-object-into-basket-script-v1", "goal_type": "object_into_basket", "primary_key": "table_object", "hold_steps": 25,
        "eyes": {"left": [0.70, 2.70, 1.40], "right": [0.70, -2.70, 1.40]}, "target": [0.85, 0.00, 0.38],
        "geometry": {"table_top_size_m": [0.62, 0.52, 0.04], "object_size_m": [0.07, 0.07, 0.07], "object_mass_kg": 0.08, "receiver_inner_half_extents_m": [0.105, 0.085, 0.16], "friction": "PhysX default material", "success": {"maximum_speed_m_s": 0.15, "hold_policy_steps": 25}},
    },
    "shoot_ball_into_goal": {
        "task_id": "Isaac-RAMBO-Quadruped-Shoot-Ball-Into-Goal-Go2-v0", "instruction": "Shoot the ball into the goal",
        "command_program_id": "rambo-shoot-ball-into-goal-script-v1", "goal_type": "ball_into_goal", "primary_key": "ball", "hold_steps": 10,
        "eyes": {"left": [0.90, 3.00, 1.50], "right": [0.90, -3.00, 1.50]}, "target": [1.00, 0.00, 0.25],
        "geometry": {"ball_radius_m": 0.10, "ball_mass_kg": 0.20, "goal_opening_half_width_m": 0.33, "goal_height_m": 0.56, "catch_depth_m": 0.30, "friction": "PhysX default material", "success": {"capture_policy_steps": 10}},
    },
}

POLICY_STEPS = common.POLICY_STEPS
RAW_COMMAND_DIM = common.RAW_COMMAND_DIM


class ObjectDatasetValidationError(RuntimeError):
    pass


def profile(name: str) -> dict[str, Any]:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ObjectDatasetValidationError(f"unknown task profile: {name}") from error


def first_consecutive(mask: np.ndarray, steps: int) -> int | None:
    count = 0
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        count = count + 1 if value else 0
        if count >= steps:
            return index + 1
    return None


def derive_events(task_name: str, metrics: dict[str, np.ndarray], success: np.ndarray) -> dict[str, Any]:
    cfg = profile(task_name)
    success = np.asarray(success)
    if success.shape != (POLICY_STEPS,) or success.dtype != np.bool_:
        raise ObjectDatasetValidationError("task success trace must be bool[3000]")
    contact = np.asarray(metrics["fl_contact"], dtype=bool)
    if contact.shape != (POLICY_STEPS,):
        raise ObjectDatasetValidationError("FL contact trace must be bool[3000]")
    if task_name == "lift_basket":
        clearance = np.asarray(metrics["clearance_m"], dtype=np.float32)
        tilt = np.asarray(metrics["tilt_rad"], dtype=np.float32)
        valid = (clearance >= 0.06) & (tilt <= math.radians(35.0)) & contact
        evidence = {"max_clearance_m": float(np.max(clearance)), "max_tilt_rad": float(np.max(tilt))}
    elif task_name == "pull_object_into_basket":
        inside = np.asarray(metrics["inside_basket"], dtype=bool)
        speed = np.asarray(metrics["speed_m_s"], dtype=np.float32)
        left_table = np.asarray(metrics["left_table"], dtype=bool)
        valid = inside & (speed <= 0.15) & contact & left_table
        evidence = {"max_speed_m_s": float(np.max(speed)), "inside_basket_steps": int(np.count_nonzero(inside))}
    else:
        scored = np.asarray(metrics["scored"], dtype=bool)
        valid = scored & contact
        evidence = {"scored_steps": int(np.count_nonzero(scored)), "max_behind_goal_m": float(np.max(np.asarray(metrics["behind_goal_m"], dtype=np.float32)))}
    computed = first_consecutive(valid, int(cfg["hold_steps"]))
    detector = np.flatnonzero(success)
    detector_step = int(detector[0] + 1) if detector.size else None
    if computed != detector_step:
        raise ObjectDatasetValidationError(f"{task_name} detector disagrees with raw telemetry: {computed} != {detector_step}")
    return {"success": detector_step is not None, "first_success_step": detector_step, "first_contact_step": int(np.flatnonzero(contact)[0] + 1) if contact.any() else None, **evidence}


def _load(root: Path, relative: str) -> np.ndarray:
    path = root / relative
    if not path.is_file():
        raise ObjectDatasetValidationError(f"missing {relative}")
    return np.load(path, allow_pickle=False)


def _require_finite_float(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ObjectDatasetValidationError(f"{name} must be finite float32 {shape}, got {array.dtype} {array.shape}")


def _require_bool(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if array.shape != shape or array.dtype != np.bool_:
        raise ObjectDatasetValidationError(f"{name} must be bool {shape}, got {array.dtype} {array.shape}")


def _validate_cameras(root: Path, metadata: dict[str, Any], cfg: dict[str, Any]) -> None:
    cameras = metadata.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != set(common.VIEWS):
        raise ObjectDatasetValidationError("metadata.cameras must contain ego/left/right")
    root_trace = _load(root, "raw_debug/root_pose_w.npy")
    for view in common.VIEWS:
        record = cameras[view]
        if record.get("resolution") != [640, 480] or record.get("focal_length_mm") != 18.0:
            raise ObjectDatasetValidationError(f"{view} camera resolution/focal contract failure")
        if record.get("horizontal_aperture_mm") != 20.955 or record.get("clipping_range_m") != [0.1, 20.0]:
            raise ObjectDatasetValidationError(f"{view} camera aperture/clip contract failure")
        if record.get("renderer") != "IsaacRtxRendererCfg" or record.get("world_fixed") is not (view != "ego"):
            raise ObjectDatasetValidationError(f"{view} camera renderer/fixed contract failure")
        pose = _load(root, f"video/camera/{view}_pose_w.npy")
        intrinsics = _load(root, f"video/camera/{view}_intrinsics.npy")
        _require_finite_float(pose, (common.RGB_FRAME_COUNT, 7), f"{view} pose")
        _require_finite_float(intrinsics, (3, 3), f"{view} intrinsics")
        if not np.allclose(np.linalg.norm(pose[:, 3:], axis=1), 1.0, atol=1e-4, rtol=0.0):
            raise ObjectDatasetValidationError(f"{view} pose quaternion is not normalized")
        if float(intrinsics[0, 0]) <= 0.0 or float(intrinsics[1, 1]) <= 0.0 or not np.isclose(intrinsics[2, 2], 1.0):
            raise ObjectDatasetValidationError(f"{view} intrinsic matrix is invalid")
        if view == "ego":
            expected = common.ego_pose_from_root_pose(root_trace[np.arange(7, common.POLICY_STEPS, 8)])
            if not np.allclose(pose, expected, atol=1e-5, rtol=0.0):
                raise ObjectDatasetValidationError("ego pose trace is not aligned to post-step robot state")
        else:
            if not np.allclose(pose, pose[0], atol=1e-6, rtol=0.0):
                raise ObjectDatasetValidationError(f"{view} camera is not world-fixed")
            if record.get("configured_eye_w_m") != cfg["eyes"][view] or record.get("configured_target_w_m") != cfg["target"]:
                raise ObjectDatasetValidationError(f"{view} configured eye/target mismatch")
            if not np.allclose(pose[0, :3], np.asarray(cfg["eyes"][view], dtype=np.float32), atol=1e-5, rtol=0.0):
                raise ObjectDatasetValidationError(f"{view} actual eye differs from metadata")


def validate_episode(root: Path, *, verify_checksums: bool = True, verify_images: bool = True) -> dict[str, Any]:
    root = root.expanduser().resolve()
    try:
        metadata = json.loads((root / "metadata.json").read_text())
        task = json.loads((root / "task.json").read_text())
        outcome = json.loads((root / "outcome.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ObjectDatasetValidationError(f"invalid episode JSON: {error}") from error
    task_name = metadata.get("task_profile")
    cfg = profile(str(task_name))
    if metadata.get("task_id") != cfg["task_id"] or task.get("task_id") != cfg["task_id"]:
        raise ObjectDatasetValidationError("task ID mismatch")
    if metadata.get("task_instruction") != cfg["instruction"] or task.get("instruction") != cfg["instruction"]:
        raise ObjectDatasetValidationError("instruction mismatch")
    if task.get("command_program_id") != cfg["command_program_id"]:
        raise ObjectDatasetValidationError("command program mismatch")
    episode_id = metadata.get("episode_id")
    if not isinstance(episode_id, str) or root.name.removesuffix(".inprogress") != episode_id:
        raise ObjectDatasetValidationError("episode ID mismatch")
    number = int(episode_id.rsplit("_", 1)[1])
    if metadata.get("runtime_seed") != common.SEED or metadata.get("seed") != (common.SEED if number == 1 else number):
        raise ObjectDatasetValidationError("seed convention mismatch")
    if metadata.get("dataset") != common.DATASET_NAME or metadata.get("dataset_version") != common.DATASET_VERSION:
        raise ObjectDatasetValidationError("dataset/version mismatch")
    if metadata.get("duration_s") != 30.0 or metadata.get("command_columns") != list(common.COMMAND_COLUMNS):
        raise ObjectDatasetValidationError("duration or 9D command semantics mismatch")
    counts = metadata.get("counts", {})
    if counts != {"policy_steps": 3000, "raw_commands": 3000, "train_commands": 1500, "rgb_frames_per_view": 375}:
        raise ObjectDatasetValidationError("episode count contract failure")
    try:
        action_arrays = {key: _load(root, relative) for key, relative in common.ACTION_FILES.items()}
        common.validate_action_arrays(action_arrays)
        common._validate_frame_index(root, verify_images=verify_images)
    except common.DatasetValidationError as error:
        raise ObjectDatasetValidationError(str(error)) from error
    pose = _load(root, "raw_debug/primary_pose_w.npy")
    velocity = _load(root, "raw_debug/primary_velocity_w.npy")
    success = _load(root, "raw_debug/task_success.npy")
    contact = _load(root, "raw_debug/fl_contact.npy")
    _require_finite_float(pose, (POLICY_STEPS, 7), "primary pose")
    _require_finite_float(velocity, (POLICY_STEPS, 6), "primary velocity")
    _require_bool(success, (POLICY_STEPS,), "task success")
    _require_bool(contact, (POLICY_STEPS,), "FL contact")
    for name, shape in (("policy_action_18d", (POLICY_STEPS, common.POLICY_ACTION_DIM)), ("observation_405d", (POLICY_STEPS, common.OBSERVATION_DIM)), ("reward", (POLICY_STEPS,)), ("root_pose_w", (POLICY_STEPS, 7)), ("fl_foot_position_w", (POLICY_STEPS, 3)), ("contact_max_force_n", (POLICY_STEPS,))):
        _require_finite_float(_load(root, f"raw_debug/{name}.npy"), shape, name)
    terminal = _load(root, "raw_debug/terminal.npy")
    _require_bool(terminal, (POLICY_STEPS,), "terminal")
    if np.any(terminal):
        raise ObjectDatasetValidationError("episode contains a terminal policy step")
    metrics: dict[str, np.ndarray] = {"fl_contact": contact}
    for name in ("clearance_m", "tilt_rad", "inside_basket", "speed_m_s", "left_table", "scored", "behind_goal_m"):
        candidate = root / f"raw_debug/{name}.npy"
        if candidate.is_file():
            metrics[name] = np.load(candidate, allow_pickle=False)
    events = derive_events(task_name, metrics, success)
    if not events["success"] or outcome.get("success") is not True or outcome.get("first_success_policy_step") != events["first_success_step"]:
        raise ObjectDatasetValidationError("outcome success mismatch")
    if outcome.get("terminal") is not False or outcome.get("terminal_reason") != "fixed_duration_complete":
        raise ObjectDatasetValidationError("terminal contract failure")
    if outcome.get("first_contact_policy_step") != events["first_contact_step"]:
        raise ObjectDatasetValidationError("outcome contact step disagrees with telemetry")
    _validate_cameras(root, metadata, cfg)
    for view in common.VIEWS:
        try:
            common._validate_mp4(root, view)
        except common.DatasetValidationError as error:
            raise ObjectDatasetValidationError(str(error)) from error
    if not (root / "contact_sheet.png").is_file() or (root / "contact_sheet.png").stat().st_size <= 0:
        raise ObjectDatasetValidationError("contact sheet is missing")
    if verify_checksums:
        try:
            common._validate_checksums(root)
        except common.DatasetValidationError as error:
            raise ObjectDatasetValidationError(str(error)) from error
    return {"passed": True, "episode_id": episode_id, "task_profile": task_name, "success_step": events["first_success_step"]}
