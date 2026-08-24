"""Schema and offline validator for the RAMBO Button PhysX episode evidence.

The module deliberately has no Isaac Sim imports.  The recorder imports it
only after Kit starts, while the validator and unit tests can inspect an
artifact on an ordinary Python interpreter.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterator


SCHEMA_VERSION = 2
TASK_ID = "Isaac-RAMBO-Quadruped-Button-Go2-v0"
PHYSX_CFG_FQN = "isaaclab_physx.physics.physx_manager_cfg.PhysxCfg"
PHYSX_MANAGER_FQN = "isaaclab_physx.physics.physx_manager.PhysxManager"
M10_PROMPT = (
    "Record one deterministic 30-second RAMBO Button loco-manipulation episode: "
    "3000 aligned actions, observations, and post-states plus 375 RGB frames; "
    "explicit PhysX only with Newton actuators disabled; use the released quadruped "
    "checkpoint; prove a 12 mm five-action-step press, rebound at or below 2 mm, "
    "and at least 0.05 m of both base and FL-foot motion."
)
TARGET_ISAACLAB_TAG = "v3.0.0-beta2.patch1"
TARGET_ISAACLAB_COMMIT = "ffff603eafc6b74264a5261cc0183d6a65390d78"
TARGET_ISAACSIM_VERSION = "6.0.1.0"
QUADRUPED_CHECKPOINT_SHA256 = "1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706"
OBSERVATION_SCHEMA_VERSION = "rambo-go2-policy-observation-v1"
ACTION_SCHEMA_VERSION = "rambo-go2-policy-action-v1"
OBSERVATION_DIMENSION = 405
ACTION_DIMENSION = 18
POLICY_DT_NS = 10_000_000
PHYSICS_DT_NS = 2_000_000
DECIMATION = 5
CAMERA_INTERVAL_ACTION_STEPS = 8
ACCEPTANCE_STEPS = 3000
ACCEPTANCE_RGB_FRAMES = 375
BUTTON_PRESS_THRESHOLD_M = 0.012
BUTTON_HOLD_STEPS = 5
BUTTON_REBOUND_THRESHOLD_M = 0.002
MOTION_MINIMUM_DELTA_M = 0.05
MEMORY_MONITOR_INTERVAL_STEPS = 100
MEMORY_ANALYSIS_START_STEP = 1000
MEMORY_WINDOW_STEPS = 500
MIB = 1024 * 1024
GPU_SLOPE_LIMIT_BYTES_PER_100_STEPS = 1 * MIB
GPU_MEDIAN_DELTA_LIMIT_BYTES = 128 * MIB
RSS_SLOPE_LIMIT_BYTES_PER_100_STEPS = 4 * MIB
RSS_MEDIAN_DELTA_LIMIT_BYTES = 512 * MIB
TRACE_FILENAMES = {
    "actions": "actions.jsonl",
    "observations": "observations.jsonl",
    "post_states": "post_states.jsonl",
    "rgb_frames": "rgb_frames.jsonl",
}


class ButtonEpisodeArtifactError(RuntimeError):
    """Raised when a Button episode artifact is incomplete or inconsistent."""


def sha256_file(path: str | Path) -> str:
    """Hash an artifact file without retaining the file content in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    """Convert common scalar-like values into JSON-safe data."""

    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: str | Path, value: Any) -> Path:
    """Write stable UTF-8 JSON evidence."""

    output = Path(path)
    output.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def write_checksums(output_dir: str | Path) -> Path:
    """Write a SHA-256 manifest for every evidence file except itself."""

    root = Path(output_dir)
    checksum_path = root / "checksums.sha256"
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in paths),
        encoding="utf-8",
    )
    return checksum_path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ButtonEpisodeArtifactError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ButtonEpisodeArtifactError(f"{label} must contain a JSON object")
    return value


def _records(path: Path, label: str) -> Iterator[dict[str, Any]]:
    try:
        file = path.open("r", encoding="utf-8")
    except OSError as error:
        raise ButtonEpisodeArtifactError(f"Missing {label}: {path}") from error
    with file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ButtonEpisodeArtifactError(f"{label} contains a blank line at {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ButtonEpisodeArtifactError(
                    f"{label} contains invalid JSON at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ButtonEpisodeArtifactError(
                    f"{label} line {line_number} must be a JSON object"
                )
            yield record


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ButtonEpisodeArtifactError(message)


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ButtonEpisodeArtifactError(f"{label} must be an integer, got bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ButtonEpisodeArtifactError(f"{label} must be an integer") from error
    if result != value:
        raise ButtonEpisodeArtifactError(f"{label} must be an exact integer, got {value!r}")
    return result


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ButtonEpisodeArtifactError(f"{label} must be a finite number, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ButtonEpisodeArtifactError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ButtonEpisodeArtifactError(f"{label} must be finite")
    return result


def _require_close(value: Any, expected: float, label: str, *, tolerance: float = 1.0e-8) -> float:
    result = _as_float(value, label)
    _require(math.isclose(result, expected, rel_tol=0.0, abs_tol=tolerance), f"{label} must equal {expected}")
    return result


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _require_git_commit(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 40 and all(char in "0123456789abcdef" for char in value),
        f"{label} must be a full lowercase Git commit",
    )
    return value


def _relative_artifact_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ButtonEpisodeArtifactError(f"{label} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ButtonEpisodeArtifactError(f"{label} must remain inside the artifact directory")
    path = root / candidate
    if not path.is_file():
        raise ButtonEpisodeArtifactError(f"{label} does not exist: {candidate}")
    return path


def _validate_provenance(manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    """Require reproducible source/runtime identities in both top-level records."""

    _require(manifest.get("prompt") == M10_PROMPT, "manifest.prompt does not match the fixed M10 prompt")
    _require(summary.get("prompt") == M10_PROMPT, "summary.prompt does not match the fixed M10 prompt")
    manifest_provenance = manifest.get("provenance")
    summary_provenance = summary.get("provenance")
    _require(isinstance(manifest_provenance, dict), "manifest.provenance is required")
    _require(isinstance(summary_provenance, dict), "summary.provenance is required")
    _require(manifest_provenance == summary_provenance, "Manifest and summary provenance disagree")
    _require_git_commit(manifest_provenance.get("rambo_git_commit"), "provenance.rambo_git_commit")
    _require(
        isinstance(manifest_provenance.get("rambo_git_dirty"), bool),
        "provenance.rambo_git_dirty must be bool",
    )
    isaaclab = manifest_provenance.get("isaaclab")
    _require(isinstance(isaaclab, dict), "provenance.isaaclab is required")
    _require(isaaclab.get("tag") == TARGET_ISAACLAB_TAG, "Isaac Lab tag is not the pinned target")
    _require(
        isaaclab.get("commit") == TARGET_ISAACLAB_COMMIT,
        "Isaac Lab commit is not the pinned target",
    )
    _require(
        manifest_provenance.get("isaacsim_version") == TARGET_ISAACSIM_VERSION,
        "Isaac Sim version is not the pinned target",
    )
    gpu = manifest_provenance.get("gpu")
    _require(isinstance(gpu, dict), "provenance.gpu is required")
    _require(isinstance(gpu.get("model"), str) and gpu["model"], "provenance.gpu.model is required")
    _require(isinstance(gpu.get("driver"), str) and gpu["driver"], "provenance.gpu.driver is required")
    runtime = summary.get("runtime")
    _require(isinstance(runtime, dict), "summary.runtime is required")
    _require(gpu["model"] == runtime.get("gpu"), "Provenance GPU model disagrees with runtime")
    _require(gpu["driver"] == runtime.get("driver"), "Provenance GPU driver disagrees with runtime")
    _require(runtime.get("isaacsim_version") == TARGET_ISAACSIM_VERSION, "Runtime Isaac Sim version is invalid")


def _validate_timing_and_schema(manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    expected_timing = {
        "physics_dt_s": PHYSICS_DT_NS / 1_000_000_000,
        "control_decimation": DECIMATION,
        "control_dt_s": POLICY_DT_NS / 1_000_000_000,
        "camera_update_period_s": 0.08,
        "camera_rate_hz": 12.5,
        "camera_interval_physics_ticks": 40,
        "camera_interval_action_steps": CAMERA_INTERVAL_ACTION_STEPS,
    }
    manifest_timing = manifest.get("timing")
    summary_timing = summary.get("timing")
    _require(isinstance(manifest_timing, dict), "manifest.timing is required")
    _require(isinstance(summary_timing, dict), "summary.timing is required")
    _require(manifest_timing == summary_timing, "Manifest and summary timing disagree")
    _require(set(manifest_timing) == set(expected_timing), "Timing evidence has missing or unexpected fields")
    for key, expected in expected_timing.items():
        if isinstance(expected, int):
            _require(_as_int(manifest_timing.get(key), f"timing.{key}") == expected, f"timing.{key} is invalid")
        else:
            _require_close(manifest_timing.get(key), expected, f"timing.{key}")

    expected_schema = {
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
    manifest_schema = manifest.get("policy_schema")
    summary_schema = summary.get("policy_schema")
    _require(manifest_schema == expected_schema, "manifest.policy_schema is invalid")
    _require(summary_schema == expected_schema, "summary.policy_schema is invalid")


def _validate_backend(summary: dict[str, Any]) -> None:
    configured = summary.get("configured_physics")
    _require(isinstance(configured, dict), "summary.configured_physics is required")
    _require(configured.get("cfg") == PHYSX_CFG_FQN, "recorder did not explicitly configure PhysxCfg")
    _require(
        configured.get("use_newton_actuators") is False,
        "recorder did not explicitly set use_newton_actuators=False",
    )
    for stage in ("backend_before", "backend_after"):
        evidence = summary.get(stage)
        _require(isinstance(evidence, dict), f"summary.{stage} PhysX evidence is required")
        _require(
            evidence.get("actual_manager") == PHYSX_MANAGER_FQN,
            f"summary.{stage} did not observe the actual PhysxManager",
        )
        _require(
            evidence.get("requested_cfg") == PHYSX_CFG_FQN,
            f"summary.{stage} did not retain the requested PhysxCfg",
        )
        _require(
            evidence.get("use_newton_actuators") is False,
            f"summary.{stage} did not retain use_newton_actuators=False",
        )


def _validate_runtime(summary: dict[str, Any]) -> None:
    checkpoint = summary.get("checkpoint")
    _require(isinstance(checkpoint, dict), "summary.checkpoint is required")
    digest = _require_sha256(checkpoint.get("sha256"), "summary.checkpoint.sha256")
    _require(
        digest == QUADRUPED_CHECKPOINT_SHA256,
        "summary.checkpoint.sha256 is not the released quadruped allowlist SHA",
    )
    _require(
        checkpoint.get("allowlist_sha256") == QUADRUPED_CHECKPOINT_SHA256,
        "summary.checkpoint.allowlist_sha256 is not the released quadruped allowlist SHA",
    )
    _require(
        _as_int(checkpoint.get("observation_dim"), "summary.checkpoint.observation_dim") == OBSERVATION_DIMENSION,
        "summary.checkpoint.observation_dim is invalid",
    )
    _require(
        _as_int(checkpoint.get("action_dim"), "summary.checkpoint.action_dim") == ACTION_DIMENSION,
        "summary.checkpoint.action_dim is invalid",
    )
    runtime = summary.get("runtime")
    _require(isinstance(runtime, dict), "summary.runtime is required")
    for key in ("python", "torch", "cuda_runtime", "gpu", "driver"):
        _require(isinstance(runtime.get(key), str) and runtime[key], f"summary.runtime.{key} is required")
    _require(_as_int(summary.get("seed"), "summary.seed") >= 0, "summary.seed must be non-negative")


def _validate_checksums(root: Path) -> int:
    checksum_path = root / "checksums.sha256"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ButtonEpisodeArtifactError("checksums.sha256 is required") from error
    _require(lines, "checksums.sha256 must not be empty")
    expected: dict[Path, str] = {}
    for line_number, line in enumerate(lines, start=1):
        try:
            digest, relative = line.split("  ", maxsplit=1)
        except ValueError as error:
            raise ButtonEpisodeArtifactError(
                f"checksums.sha256 line {line_number} must be '<sha256>  <path>'"
            ) from error
        _require(
            len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
            f"checksums.sha256 line {line_number} has an invalid digest",
        )
        path = _relative_artifact_path(root, relative, f"checksums.sha256 line {line_number}")
        _require(path.relative_to(root) not in expected, f"Duplicate checksum path: {relative}")
        expected[path.relative_to(root)] = digest
    actual_files = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    _require(set(expected) == actual_files, "checksums.sha256 does not cover exactly the artifact files")
    for relative, digest in expected.items():
        _require(sha256_file(root / relative) == digest, f"Checksum mismatch for {relative}")
    return len(expected)


def _validate_action_trace(root: Path, steps: int) -> int:
    count = 0
    for expected_step, record in enumerate(_records(root / TRACE_FILENAMES["actions"], "actions trace"), start=1):
        _require(_as_int(record.get("action_step"), "action.action_step") == expected_step, "Action steps are not contiguous")
        _require(
            _as_int(record.get("start_timestamp_ns"), "action.start_timestamp_ns")
            == (expected_step - 1) * POLICY_DT_NS,
            "Action start timestamp is not synchronized to the policy timebase",
        )
        _require(
            _as_int(record.get("end_timestamp_ns"), "action.end_timestamp_ns")
            == expected_step * POLICY_DT_NS,
            "Action end timestamp is not synchronized to the policy timebase",
        )
        _validate_policy_matrix(
            record.get("policy_action"),
            "action.policy_action",
            ACTION_DIMENSION,
        )
        _require(isinstance(record.get("command"), dict), "Action record is missing command evidence")
        count += 1
    _require(count == steps, f"Expected {steps} action records, found {count}")
    return count


def _validate_observation_trace(root: Path, steps: int) -> int:
    count = 0
    for expected_step, record in enumerate(_records(root / TRACE_FILENAMES["observations"], "observations trace"), start=1):
        _require(_as_int(record.get("action_step"), "observation.action_step") == expected_step, "Observation steps are not contiguous")
        _require(
            _as_int(record.get("timestamp_ns"), "observation.timestamp_ns") == expected_step * POLICY_DT_NS,
            "Observation timestamp is not synchronized to the action post-state",
        )
        _validate_policy_matrix(
            record.get("policy_observation"),
            "observation.policy_observation",
            OBSERVATION_DIMENSION,
        )
        count += 1
    _require(count == steps, f"Expected {steps} observation records, found {count}")
    return count


def _validate_policy_matrix(value: Any, label: str, dimension: int) -> None:
    """Require the single-environment policy payload to match its declared schema."""

    _require(isinstance(value, list) and len(value) == 1, f"{label} must have shape (1, {dimension})")
    row = value[0]
    _require(isinstance(row, list) and len(row) == dimension, f"{label} must have shape (1, {dimension})")
    for index, component in enumerate(row):
        _as_float(component, f"{label}[0][{index}]")


def _single_vector3(value: Any, label: str) -> tuple[float, float, float]:
    _require(isinstance(value, list) and len(value) == 1, f"{label} must contain one environment vector")
    vector = value[0]
    _require(isinstance(vector, list) and len(vector) == 3, f"{label} must have XYZ shape (1, 3)")
    return tuple(_as_float(component, f"{label}[0][{index}]") for index, component in enumerate(vector))  # type: ignore[return-value]


def _single_scalar(value: Any, label: str) -> float:
    _require(isinstance(value, list) and len(value) == 1, f"{label} must contain one environment scalar")
    return _as_float(value[0], f"{label}[0]")


def _distance3(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((first - second) ** 2 for first, second in zip(left, right)))


def _validate_post_state_trace(
    root: Path, steps: int
) -> tuple[int, bool, bool, dict[str, Any], dict[str, Any]]:
    count = 0
    success_seen = False
    release_after_success_seen = False
    final_record: dict[str, Any] = {}
    initial_base: tuple[float, float, float] | None = None
    initial_fl: tuple[float, float, float] | None = None
    max_base_delta = 0.0
    max_fl_delta = 0.0
    max_displacement = 0.0
    current_pressed_steps = 0
    max_pressed_steps = 0
    first_success_step: int | None = None
    first_rebound_after_success_step: int | None = None
    min_displacement_after_success: float | None = None
    for expected_step, record in enumerate(_records(root / TRACE_FILENAMES["post_states"], "post-state trace"), start=1):
        _require(_as_int(record.get("action_step"), "post_state.action_step") == expected_step, "Post-state steps are not contiguous")
        _require(
            _as_int(record.get("timestamp_ns"), "post_state.timestamp_ns") == expected_step * POLICY_DT_NS,
            "Post-state timestamp is not synchronized to the action post-state",
        )
        robot = record.get("robot")
        _require(isinstance(robot, dict), "Post-state record is missing robot state")
        _require(robot.get("fl_foot_body_name") == "FL_foot", "Post-state FL-foot body identity is invalid")
        base_position = _single_vector3(robot.get("root_link_pos_w_m"), "robot.root_link_pos_w_m")
        fl_position = _single_vector3(robot.get("fl_foot_link_pos_w_m"), "robot.fl_foot_link_pos_w_m")
        if initial_base is None:
            initial_base = base_position
            initial_fl = fl_position
        else:
            max_base_delta = max(max_base_delta, _distance3(base_position, initial_base))
            assert initial_fl is not None
            max_fl_delta = max(max_fl_delta, _distance3(fl_position, initial_fl))
        button = record.get("button")
        _require(isinstance(button, dict), "Post-state record is missing button state")
        success = button.get("success")
        released = button.get("released")
        terminal = record.get("terminal")
        _require(isinstance(success, list) and success, "Button success state is missing")
        _require(isinstance(released, list) and released, "Button release state is missing")
        _require(isinstance(terminal, list) and terminal, "Terminal state is missing")
        _require(not any(bool(value) for value in terminal), f"Episode terminated at action step {expected_step}")
        displacement = _single_scalar(button.get("displacement_m"), "button.displacement_m")
        _require(displacement >= 0.0, "Button displacement must be non-negative")
        max_displacement = max(max_displacement, displacement)
        if displacement >= BUTTON_PRESS_THRESHOLD_M:
            current_pressed_steps += 1
        else:
            current_pressed_steps = 0
        max_pressed_steps = max(max_pressed_steps, current_pressed_steps)
        step_success = any(bool(value) for value in success)
        step_released = any(bool(value) for value in released)
        _require(
            step_released == (displacement <= BUTTON_REBOUND_THRESHOLD_M),
            f"Button released state disagrees with {BUTTON_REBOUND_THRESHOLD_M} m rebound threshold",
        )
        if success_seen:
            if min_displacement_after_success is None:
                min_displacement_after_success = displacement
            else:
                min_displacement_after_success = min(min_displacement_after_success, displacement)
            if displacement <= BUTTON_REBOUND_THRESHOLD_M and first_rebound_after_success_step is None:
                first_rebound_after_success_step = expected_step
        release_after_success_seen |= success_seen and step_released
        if step_success and first_success_step is None:
            first_success_step = expected_step
        success_seen |= step_success
        final_record = record
        count += 1
    _require(count == steps, f"Expected {steps} post-state records, found {count}")
    return (
        count,
        success_seen,
        release_after_success_seen,
        final_record,
        {
            "button": {
                "press_threshold_m": BUTTON_PRESS_THRESHOLD_M,
                "hold_steps_required": BUTTON_HOLD_STEPS,
                "rebound_threshold_m": BUTTON_REBOUND_THRESHOLD_M,
                "max_displacement_m": max_displacement,
                "max_consecutive_pressed_action_steps": max_pressed_steps,
                "first_success_action_step": first_success_step,
                "first_rebound_after_success_action_step": first_rebound_after_success_step,
                "min_displacement_after_success_m": min_displacement_after_success,
            },
            "motion": {
                "minimum_required_delta_m": MOTION_MINIMUM_DELTA_M,
                "baseline_action_step": 1,
                "max_base_root_link_displacement_m": max_base_delta,
                "max_fl_foot_link_displacement_m": max_fl_delta,
            },
        },
    )


def _validate_acceptance_summary(summary: dict[str, Any], expected: dict[str, Any]) -> None:
    reported = summary.get("acceptance")
    _require(isinstance(reported, dict), "summary.acceptance is required")
    expected_button = expected["button"]
    expected_motion = expected["motion"]
    button = reported.get("button")
    motion = reported.get("motion")
    _require(isinstance(button, dict), "summary.acceptance.button is required")
    _require(isinstance(motion, dict), "summary.acceptance.motion is required")
    _require_close(button.get("press_threshold_m"), BUTTON_PRESS_THRESHOLD_M, "acceptance.button.press_threshold_m")
    _require(
        _as_int(button.get("hold_steps_required"), "acceptance.button.hold_steps_required") == BUTTON_HOLD_STEPS,
        "Button hold-step requirement is invalid",
    )
    _require_close(
        button.get("rebound_threshold_m"),
        BUTTON_REBOUND_THRESHOLD_M,
        "acceptance.button.rebound_threshold_m",
    )
    _require_close(
        button.get("max_displacement_m"),
        _as_float(expected_button["max_displacement_m"], "expected max displacement"),
        "summary acceptance max_displacement_m",
    )
    _require(
        _as_int(
            button.get("max_consecutive_pressed_action_steps"),
            "summary acceptance max_consecutive_pressed_action_steps",
        )
        == expected_button["max_consecutive_pressed_action_steps"],
        "summary acceptance max_consecutive_pressed_action_steps disagrees with post-state trace",
    )
    for key in ("first_success_action_step", "first_rebound_after_success_action_step"):
        _require(
            button.get(key) == expected_button[key],
            f"summary acceptance {key} disagrees with post-state trace",
        )
    expected_rebound = expected_button["min_displacement_after_success_m"]
    if expected_rebound is None:
        _require(
            button.get("min_displacement_after_success_m") is None,
            "summary acceptance min_displacement_after_success_m disagrees with post-state trace",
        )
    else:
        _require_close(
            button.get("min_displacement_after_success_m"),
            _as_float(expected_rebound, "expected rebound displacement"),
            "summary acceptance min_displacement_after_success_m",
        )
    _require_close(
        motion.get("minimum_required_delta_m"),
        MOTION_MINIMUM_DELTA_M,
        "acceptance.motion.minimum_required_delta_m",
    )
    _require(
        _as_int(motion.get("baseline_action_step"), "acceptance.motion.baseline_action_step") == 1,
        "Motion baseline must be the first post-state",
    )
    for key in ("max_base_root_link_displacement_m", "max_fl_foot_link_displacement_m"):
        _require_close(
            motion.get(key),
            _as_float(expected_motion[key], f"expected {key}"),
            f"summary acceptance {key}",
        )


def _validate_rgb_staging(summary: dict[str, Any], rgb_count: int) -> None:
    staging = summary.get("rgb_cpu_staging")
    _require(isinstance(staging, dict), "summary.rgb_cpu_staging is required")
    _require(staging.get("reused") is True, "RGB recorder must reuse one CPU staging buffer")
    _require(
        _as_int(staging.get("allocations"), "rgb_cpu_staging.allocations") == 1,
        "RGB recorder must allocate exactly one CPU staging buffer",
    )
    _require(
        _as_int(staging.get("copies"), "rgb_cpu_staging.copies") == rgb_count,
        "RGB staging-copy count disagrees with RGB trace",
    )
    _require(staging.get("shape") == [480, 640, 3], "RGB CPU staging shape is invalid")
    _require(staging.get("dtype") == "torch.uint8", "RGB CPU staging dtype is invalid")
    _require(isinstance(staging.get("pinned_memory"), bool), "RGB CPU staging pinning evidence is required")


def _validate_full_acceptance(
    *,
    success_seen: bool,
    release_after_success_seen: bool,
    final_state: dict[str, Any],
    acceptance: dict[str, Any],
    rgb_count: int,
) -> None:
    """Fail closed unless the streamed full episode proves every M10 outcome."""

    _require(success_seen, "Full Button episode never reached the success state")
    _require(release_after_success_seen, "Full Button episode never rebounded after success")
    final_button = final_state["button"]
    _require(any(bool(value) for value in final_button["success"]), "Full Button episode did not retain success")
    _require(any(bool(value) for value in final_button["released"]), "Full Button episode did not rebound")
    button_acceptance = acceptance["button"]
    motion_acceptance = acceptance["motion"]
    _require(
        button_acceptance["max_displacement_m"] >= BUTTON_PRESS_THRESHOLD_M,
        "Full Button episode never reached the 12 mm press threshold",
    )
    _require(
        button_acceptance["max_consecutive_pressed_action_steps"] >= BUTTON_HOLD_STEPS,
        "Full Button episode did not hold the 12 mm press for five action steps",
    )
    _require(
        button_acceptance["min_displacement_after_success_m"] is not None
        and button_acceptance["min_displacement_after_success_m"] <= BUTTON_REBOUND_THRESHOLD_M,
        "Full Button episode did not rebound to 2 mm or less after success",
    )
    _require(
        motion_acceptance["max_base_root_link_displacement_m"] >= MOTION_MINIMUM_DELTA_M,
        "Full Button episode lacks 0.05 m base-motion evidence",
    )
    _require(
        motion_acceptance["max_fl_foot_link_displacement_m"] >= MOTION_MINIMUM_DELTA_M,
        "Full Button episode lacks 0.05 m FL-foot-motion evidence",
    )
    _require(rgb_count == ACCEPTANCE_RGB_FRAMES, "Full Button episode must contain exactly 375 RGB frames")


def _validate_rgb_trace(root: Path, steps: int) -> int:
    expected_count = steps // CAMERA_INTERVAL_ACTION_STEPS
    count = 0
    previous_frame_id: int | None = None
    for expected_index, record in enumerate(_records(root / TRACE_FILENAMES["rgb_frames"], "RGB frame trace"), start=1):
        expected_action_step = expected_index * CAMERA_INTERVAL_ACTION_STEPS
        _require(_as_int(record.get("frame_index"), "rgb.frame_index") == expected_index, "RGB indices are not contiguous")
        _require(
            _as_int(record.get("action_step"), "rgb.action_step") == expected_action_step,
            "RGB action step is not aligned to the exact eight-step camera cadence",
        )
        _require(
            _as_int(record.get("timestamp_ns"), "rgb.timestamp_ns") == expected_action_step * POLICY_DT_NS,
            "RGB timestamp is not synchronized to the action post-state",
        )
        _require(
            _as_int(record.get("physics_ticks"), "rgb.physics_ticks") == expected_action_step * DECIMATION,
            "RGB physics tick count is not synchronized to the camera cadence",
        )
        frame_id = _as_int(record.get("camera_frame_id"), "rgb.camera_frame_id")
        if previous_frame_id is not None:
            _require(frame_id == previous_frame_id + 1, "RGB camera frame IDs are not consecutive")
        previous_frame_id = frame_id
        frame_path = _relative_artifact_path(root, record.get("path"), "rgb.path")
        _require(frame_path.suffix.lower() == ".png", "RGB evidence must be a PNG")
        count += 1
    _require(count == expected_count, f"Expected {expected_count} RGB records, found {count}")
    return count


def _memory_slope_bytes_per_100_steps(samples: list[dict[str, int]], field: str) -> float:
    if len(samples) < 2:
        raise ButtonEpisodeArtifactError(f"Need at least two memory samples to calculate {field} slope")
    xs = [float(sample["action_step"]) for sample in samples]
    ys = [float(sample[field]) for sample in samples]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0.0:
        raise ButtonEpisodeArtifactError(f"Memory sample steps are degenerate for {field} slope")
    return 100.0 * sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def _validate_memory_monitor(summary: dict[str, Any], steps: int) -> dict[str, Any]:
    """Recompute the 100-step post-warm-up allocation gate from raw samples."""

    monitor = summary.get("memory_monitor")
    _require(isinstance(monitor, dict), "summary.memory_monitor is required")
    _require(
        _as_int(monitor.get("interval_action_steps"), "memory_monitor.interval_action_steps")
        == MEMORY_MONITOR_INTERVAL_STEPS,
        "Memory monitor must sample every 100 action steps",
    )
    _require(
        _as_int(monitor.get("analysis_start_action_step"), "memory_monitor.analysis_start_action_step")
        == MEMORY_ANALYSIS_START_STEP,
        "Memory analysis must begin at action step 1000",
    )
    _require(
        _as_int(monitor.get("first_last_window_action_steps"), "memory_monitor.first_last_window_action_steps")
        == MEMORY_WINDOW_STEPS,
        "Memory analysis must use 500-step comparison windows",
    )
    limits = monitor.get("limits")
    _require(isinstance(limits, dict), "memory_monitor.limits is required")
    expected_limits = {
        "gpu_slope_mib_per_100_steps": GPU_SLOPE_LIMIT_BYTES_PER_100_STEPS / MIB,
        "gpu_median_delta_mib": GPU_MEDIAN_DELTA_LIMIT_BYTES / MIB,
        "rss_slope_mib_per_100_steps": RSS_SLOPE_LIMIT_BYTES_PER_100_STEPS / MIB,
        "rss_median_delta_mib": RSS_MEDIAN_DELTA_LIMIT_BYTES / MIB,
    }
    _require(limits == expected_limits, "memory_monitor.limits does not match the M10 gate")
    raw_samples = monitor.get("samples")
    _require(isinstance(raw_samples, list), "memory_monitor.samples must be a list")
    expected_steps = list(range(0, steps + 1, MEMORY_MONITOR_INTERVAL_STEPS))
    _require(len(raw_samples) == len(expected_steps), "Unexpected count of 100-step memory samples")
    samples: list[dict[str, int]] = []
    for expected_step, raw_sample in zip(expected_steps, raw_samples):
        _require(isinstance(raw_sample, dict), "Each memory sample must be an object")
        sample = {
            "action_step": _as_int(raw_sample.get("action_step"), "memory_sample.action_step"),
            "timestamp_ns": _as_int(raw_sample.get("timestamp_ns"), "memory_sample.timestamp_ns"),
            "gpu_allocated_bytes": _as_int(
                raw_sample.get("gpu_allocated_bytes"), "memory_sample.gpu_allocated_bytes"
            ),
            "rss_bytes": _as_int(raw_sample.get("rss_bytes"), "memory_sample.rss_bytes"),
        }
        _require(sample["action_step"] == expected_step, "Memory sample action steps are not contiguous 100-step samples")
        _require(
            sample["timestamp_ns"] == expected_step * POLICY_DT_NS,
            "Memory sample timestamp is not synchronized to the policy timebase",
        )
        _require(sample["gpu_allocated_bytes"] >= 0, "GPU allocation sample must be non-negative")
        _require(sample["rss_bytes"] > 0, "RSS sample must be positive")
        samples.append(sample)

    reported = monitor.get("analysis")
    _require(isinstance(reported, dict), "memory_monitor.analysis is required")
    if steps < MEMORY_ANALYSIS_START_STEP:
        _require(reported.get("evaluated") is False, "Short memory run must not claim post-warm-up analysis")
        return {"evaluated": False, "sample_count": len(samples)}

    _require(reported.get("evaluated") is True, "Long memory run must evaluate the post-step-1000 gate")
    post_warmup = [sample for sample in samples if sample["action_step"] >= MEMORY_ANALYSIS_START_STEP]
    first_window = [sample for sample in samples if sample["action_step"] <= MEMORY_WINDOW_STEPS]
    last_window = [sample for sample in samples if sample["action_step"] > steps - MEMORY_WINDOW_STEPS]
    _require(len(post_warmup) >= 2 and first_window and last_window, "Insufficient M10 memory samples")
    result: dict[str, Any] = {"evaluated": True, "sample_count": len(samples)}
    for label, field, slope_limit, median_delta_limit in (
        (
            "gpu",
            "gpu_allocated_bytes",
            GPU_SLOPE_LIMIT_BYTES_PER_100_STEPS,
            GPU_MEDIAN_DELTA_LIMIT_BYTES,
        ),
        ("rss", "rss_bytes", RSS_SLOPE_LIMIT_BYTES_PER_100_STEPS, RSS_MEDIAN_DELTA_LIMIT_BYTES),
    ):
        slope = _memory_slope_bytes_per_100_steps(post_warmup, field)
        median_delta = float(statistics.median(sample[field] for sample in last_window)) - float(
            statistics.median(sample[field] for sample in first_window)
        )
        _require(slope <= slope_limit, f"{label} allocation slope exceeds the M10 threshold")
        _require(median_delta <= median_delta_limit, f"{label} median growth exceeds the M10 threshold")
        result[label] = {
            "slope_bytes_per_100_steps": slope,
            "median_delta_bytes": median_delta,
        }
    return result


def validate_button_episode_artifact(
    output_dir: str | Path, *, require_acceptance: bool = False
) -> dict[str, Any]:
    """Validate a complete or explicitly marked short Button episode artifact."""

    root = Path(output_dir).expanduser().resolve()
    _require(root.is_dir(), f"Artifact directory does not exist: {root}")
    manifest = _read_json(root / "manifest.json", "manifest.json")
    summary = _read_json(root / "summary.json", "summary.json")
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "Unsupported manifest schema")
    _require(summary.get("schema_version") == SCHEMA_VERSION, "Unsupported summary schema")
    _require(manifest.get("task") == TASK_ID and summary.get("task") == TASK_ID, "Unexpected task")
    _require(
        manifest.get("files") == {**TRACE_FILENAMES, "rgb_directory": "rgb"},
        "Artifact file manifest does not match the M10 evidence contract",
    )
    _validate_provenance(manifest, summary)
    _validate_timing_and_schema(manifest, summary)
    _validate_backend(summary)
    _validate_runtime(summary)

    requested_steps = _as_int(manifest.get("requested_action_steps"), "manifest.requested_action_steps")
    short_smoke = manifest.get("short_smoke")
    _require(isinstance(short_smoke, bool), "manifest.short_smoke must be bool")
    _require(requested_steps > 0 and requested_steps % CAMERA_INTERVAL_ACTION_STEPS == 0, "Invalid action-step cadence")
    _require(
        manifest.get("timebase")
        == {
            "policy_dt_ns": POLICY_DT_NS,
            "physics_dt_ns": PHYSICS_DT_NS,
            "decimation": DECIMATION,
            "camera_interval_action_steps": CAMERA_INTERVAL_ACTION_STEPS,
        },
        "Artifact timebase does not match the fixed RAMBO episode contract",
    )
    _require(short_smoke == (requested_steps != ACCEPTANCE_STEPS), "short_smoke does not match requested steps")
    _require(summary.get("requested_action_steps") == requested_steps, "Summary action-step count disagrees with manifest")
    _require(summary.get("short_smoke") is short_smoke, "Summary short-smoke flag disagrees with manifest")
    _require(
        summary.get("expected_rgb_frames") == requested_steps // CAMERA_INTERVAL_ACTION_STEPS,
        "Summary RGB count is invalid",
    )
    _require(summary.get("enable_cameras") is True, "Recorder must explicitly enable the RGB camera")
    _require(summary.get("viz") in {"none", "kit"}, "Recorder visualization must be none or kit")
    if require_acceptance:
        _require(not short_smoke, "Full 30-second acceptance artifact is required")
    if not short_smoke:
        _require(requested_steps == ACCEPTANCE_STEPS, "Full artifact must contain exactly 3000 action steps")
        provenance = manifest["provenance"]
        _require(
            provenance["rambo_git_dirty"] is False,
            "Full M10 acceptance artifact must use a clean RAMBO Git HEAD",
        )

    action_count = _validate_action_trace(root, requested_steps)
    observation_count = _validate_observation_trace(root, requested_steps)
    post_state_count, success_seen, release_after_success_seen, final_state, acceptance = _validate_post_state_trace(
        root, requested_steps
    )
    rgb_count = _validate_rgb_trace(root, requested_steps)
    _validate_rgb_staging(summary, rgb_count)
    memory = _validate_memory_monitor(summary, requested_steps)
    expected_rgb = requested_steps // CAMERA_INTERVAL_ACTION_STEPS
    _require(summary.get("passed") is True, "Recorder summary is not marked passed")
    counts = summary.get("trace_counts")
    _require(isinstance(counts, dict), "summary.trace_counts is required")
    _require(
        counts == {
            "actions": action_count,
            "observations": observation_count,
            "post_states": post_state_count,
            "rgb_frames": rgb_count,
        },
        "summary.trace_counts does not match streamed evidence",
    )
    success = summary.get("success")
    _require(isinstance(success, dict), "summary.success is required")
    _require(bool(success.get("success_seen")) == success_seen, "summary success state disagrees with post-state trace")
    _require(
        bool(success.get("release_after_success_seen")) == release_after_success_seen,
        "summary rebound state disagrees with post-state trace",
    )
    _validate_acceptance_summary(summary, acceptance)
    terminal = summary.get("terminal")
    _require(isinstance(terminal, dict), "summary.terminal is required")
    _require(
        _as_int(terminal.get("terminal_count"), "summary.terminal.terminal_count") == 0,
        "Episode terminal count must be zero",
    )
    if not short_smoke:
        _validate_full_acceptance(
            success_seen=success_seen,
            release_after_success_seen=release_after_success_seen,
            final_state=final_state,
            acceptance=acceptance,
            rgb_count=rgb_count,
        )
    checksum_count = _validate_checksums(root)
    return {
        "passed": True,
        "requested_action_steps": requested_steps,
        "short_smoke": short_smoke,
        "rgb_frames": expected_rgb,
        "success_seen": success_seen,
        "acceptance": acceptance,
        "memory_monitor": memory,
        "checksummed_files": checksum_count,
    }


__all__ = [
    "ACCEPTANCE_RGB_FRAMES",
    "ACCEPTANCE_STEPS",
    "ACTION_DIMENSION",
    "ACTION_SCHEMA_VERSION",
    "BUTTON_HOLD_STEPS",
    "BUTTON_PRESS_THRESHOLD_M",
    "BUTTON_REBOUND_THRESHOLD_M",
    "ButtonEpisodeArtifactError",
    "CAMERA_INTERVAL_ACTION_STEPS",
    "DECIMATION",
    "GPU_MEDIAN_DELTA_LIMIT_BYTES",
    "GPU_SLOPE_LIMIT_BYTES_PER_100_STEPS",
    "MEMORY_ANALYSIS_START_STEP",
    "MEMORY_MONITOR_INTERVAL_STEPS",
    "MEMORY_WINDOW_STEPS",
    "MIB",
    "M10_PROMPT",
    "MOTION_MINIMUM_DELTA_M",
    "OBSERVATION_DIMENSION",
    "OBSERVATION_SCHEMA_VERSION",
    "PHYSICS_DT_NS",
    "PHYSX_CFG_FQN",
    "PHYSX_MANAGER_FQN",
    "POLICY_DT_NS",
    "QUADRUPED_CHECKPOINT_SHA256",
    "SCHEMA_VERSION",
    "RSS_MEDIAN_DELTA_LIMIT_BYTES",
    "RSS_SLOPE_LIMIT_BYTES_PER_100_STEPS",
    "TASK_ID",
    "TARGET_ISAACLAB_COMMIT",
    "TARGET_ISAACLAB_TAG",
    "TARGET_ISAACSIM_VERSION",
    "TRACE_FILENAMES",
    "jsonable",
    "sha256_file",
    "validate_button_episode_artifact",
    "write_checksums",
    "write_json",
]
