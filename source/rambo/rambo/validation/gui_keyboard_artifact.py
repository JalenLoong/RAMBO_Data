"""Offline schema for auditable RAMBO GUI keyboard teleoperation evidence.

This module deliberately imports only the Python standard library.  It can
therefore validate an artifact without starting Kit, constructing a simulator,
or selecting any physics backend.  The runtime recorder observes the existing
Carb keyboard callback; it never creates keyboard events itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 3
ARTIFACT_KIND = "rambo_gui_keyboard_teleop"
TASK_ID = "Isaac-RAMBO-Quadruped-Button-Go2-v0"
MODE = "explicit_gui_keyboard_callback"
VISUALIZER = "kit"
MAX_ACTION_STEPS = 3000
PHYSX_CFG_FQN = "isaaclab_physx.physics.physx_manager_cfg.PhysxCfg"
PHYSX_MANAGER_FQN = "isaaclab_physx.physics.physx_manager.PhysxManager"
QUADRUPED_CHECKPOINT_SHA256 = "1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706"
BUTTON_PRESS_THRESHOLD_M = 0.012
BUTTON_REBOUND_THRESHOLD_M = 0.002
FL_TARGET_MINIMUM_DELTA_M = 1.0e-4

MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "summary.json"
EVENTS_FILENAME = "events.jsonl"
STATES_FILENAME = "states.jsonl"
ATTESTATION_FILENAME = "attestation.json"
CHECKSUMS_FILENAME = "checksums.sha256"
PROCESS_EXIT_FILENAME = "process_exit.json"
POST_CLOSE_RUNNER = "scripts/rambo/run_gui_keyboard_artifact.sh"
PROCESS_EXIT_SCHEMA_VERSION = 1
ATTESTATION_CONFIRMATION_METHOD = "interactive_tty_exact_phrase_after_kit_close"
TERMINAL_CONFIRMATION_PHRASE = "I OBSERVED THE M8 PHYSICAL KEYBOARD GUI"
OBSERVED_CARB_EVENT_TYPES = frozenset({"KEY_PRESS", "KEY_RELEASE", "KEY_REPEAT", "CHAR"})

OPERATOR_ATTESTATION_STATEMENT = (
    "I personally focused the Isaac Sim GUI viewport and operated my physical keyboard through "
    "the live Selkies WebRTC session to issue base and FL commands, press SPACE, and observe a "
    "simulated button press and rebound. I did not use a synthetic, scripted, replayed, automated, "
    "or injected keyboard-event source."
)
PHYSICALITY_LIMITATION = (
    "This artifact records callback observations and simulation states, but it cannot "
    "independently prove that a physical keyboard caused those callbacks. It requires "
    "the named operator's attestation and reviewer judgment."
)


class GuiKeyboardArtifactError(RuntimeError):
    """Raised when GUI keyboard evidence is incomplete or inconsistent."""


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


def sha256_file(path: str | Path) -> str:
    """Hash one file without retaining its content in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> Path:
    """Write stable UTF-8 JSON evidence and reject non-finite values."""

    output = Path(path)
    output.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def write_checksums(output_dir: str | Path) -> Path:
    """Write SHA-256 entries for every evidence file except the checksum manifest."""

    root = Path(output_dir).expanduser().resolve()
    checksum_path = root / CHECKSUMS_FILENAME
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path != checksum_path)
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in paths),
        encoding="utf-8",
    )
    return checksum_path


class GuiKeyboardArtifactWriter:
    """Append callback/state records and finish pre-close runtime evidence.

    The caller owns all semantics of each record.  This class only gives each
    stream a contiguous sequence number, flushes each line promptly, and writes
    a checksum manifest once the in-process workload has ended.  The dedicated
    parent runner must subsequently record the real child exit after Kit has
    closed; until then this directory is deliberately not an accepted artifact.
    """

    def __init__(self, output_dir: str | Path, *, manifest: dict[str, Any]) -> None:
        self.root = Path(output_dir).expanduser().resolve()
        if self.root.exists():
            raise GuiKeyboardArtifactError(f"Refusing to overwrite existing artifact directory: {self.root}")
        self.root.mkdir(parents=True)
        write_json(self.root / MANIFEST_FILENAME, manifest)
        self._events = (self.root / EVENTS_FILENAME).open("x", encoding="utf-8")
        self._states = (self.root / STATES_FILENAME).open("x", encoding="utf-8")
        self._event_count = 0
        self._state_count = 0
        self._finalized = False

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def state_count(self) -> int:
        return self._state_count

    def _append(self, file: Any, record: dict[str, Any], *, sequence: int) -> None:
        payload = dict(record)
        if "sequence" in payload:
            raise GuiKeyboardArtifactError("Artifact records must not provide their own sequence")
        payload["sequence"] = sequence
        file.write(json.dumps(jsonable(payload), sort_keys=True, allow_nan=False) + "\n")
        file.flush()

    def record_event(self, record: dict[str, Any]) -> None:
        if self._finalized:
            raise GuiKeyboardArtifactError("Cannot append an event after artifact finalization")
        self._event_count += 1
        self._append(self._events, record, sequence=self._event_count)

    def record_state(self, record: dict[str, Any]) -> None:
        if self._finalized:
            raise GuiKeyboardArtifactError("Cannot append a state after artifact finalization")
        self._state_count += 1
        self._append(self._states, record, sequence=self._state_count)

    def finalize(self, summary: dict[str, Any]) -> Path:
        """Close trace streams, write the summary, then checksum pre-close files.

        ``process_exit.json`` is intentionally absent at this point: it can
        only be written by the parent process after the Isaac Sim child exits.
        """

        if self._finalized:
            raise GuiKeyboardArtifactError("Artifact has already been finalized")
        self._events.close()
        self._states.close()
        write_json(self.root / SUMMARY_FILENAME, summary)
        self._finalized = True
        return write_checksums(self.root)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GuiKeyboardArtifactError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise GuiKeyboardArtifactError(f"{label} must contain a JSON object")
    return value


def _records(path: Path, label: str) -> Iterator[dict[str, Any]]:
    try:
        file = path.open("r", encoding="utf-8")
    except OSError as error:
        raise GuiKeyboardArtifactError(f"Missing {label}: {path}") from error
    with file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise GuiKeyboardArtifactError(f"{label} contains a blank line at {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise GuiKeyboardArtifactError(
                    f"{label} contains invalid JSON at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise GuiKeyboardArtifactError(f"{label} line {line_number} must be a JSON object")
            yield record


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GuiKeyboardArtifactError(message)


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise GuiKeyboardArtifactError(f"{label} must be an integer, got bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise GuiKeyboardArtifactError(f"{label} must be an integer") from error
    if result != value:
        raise GuiKeyboardArtifactError(f"{label} must be an exact integer, got {value!r}")
    return result


def _as_finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise GuiKeyboardArtifactError(f"{label} must be a finite number, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise GuiKeyboardArtifactError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise GuiKeyboardArtifactError(f"{label} must be finite")
    return result


def _vector(value: Any, label: str, *, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise GuiKeyboardArtifactError(f"{label} must be a length-{length} list")
    return [_as_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _validate_checksums(root: Path) -> int:
    checksum_path = root / CHECKSUMS_FILENAME
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise GuiKeyboardArtifactError(f"Missing checksum manifest: {checksum_path}") from error
    _require(lines, "Checksum manifest is empty")
    listed: dict[Path, str] = {}
    for line_number, line in enumerate(lines, start=1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise GuiKeyboardArtifactError(f"Malformed checksum line {line_number}") from error
        _require(
            len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
            f"Checksum line {line_number} has an invalid SHA-256 digest",
        )
        path = Path(relative)
        _require(not path.is_absolute() and ".." not in path.parts and str(path) != ".", f"Checksum path escapes artifact: {relative}")
        _require(path not in listed, f"Checksum path is duplicated: {relative}")
        target = root / path
        _require(target.is_file(), f"Checksummed file is missing: {relative}")
        _require(sha256_file(target) == digest, f"Checksum mismatch: {relative}")
        listed[path] = digest
    # Exclude only this artifact's root manifest.  A nested file that happens
    # to share the filename is still evidence and must be checksummed.
    checksum_path = root / CHECKSUMS_FILENAME
    actual = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    _require(set(listed) == actual, "Checksum manifest does not cover exactly all evidence files")
    return len(listed)


def _validate_physx_evidence(value: Any, label: str) -> None:
    _require(isinstance(value, dict), f"{label} is required")
    _require(value.get("requested_cfg") == PHYSX_CFG_FQN, f"{label}.requested_cfg is not PhysxCfg")
    _require(value.get("actual_manager") == PHYSX_MANAGER_FQN, f"{label}.actual_manager is not PhysxManager")
    _require(value.get("use_newton_actuators") is False, f"{label}.use_newton_actuators must be false")


def _validate_manifest(manifest: dict[str, Any]) -> tuple[int, list[float]]:
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "Manifest schema version is invalid")
    _require(manifest.get("artifact_kind") == ARTIFACT_KIND, "Manifest artifact kind is invalid")
    _require(manifest.get("task") == TASK_ID, "Manifest task is invalid")
    _require(manifest.get("mode") == MODE, "Manifest mode is invalid")
    _require(manifest.get("visualizer") == VISUALIZER, "Manifest must record --viz kit")
    _require(manifest.get("smoke_mode") is False, "GUI artifact must not be a smoke run")
    max_steps = _as_int(manifest.get("max_action_steps"), "manifest.max_action_steps")
    _require(0 < max_steps <= MAX_ACTION_STEPS, "Manifest max_action_steps is outside the finite GUI limit")
    arm_timeout_s = _as_finite(manifest.get("arm_timeout_s"), "manifest.arm_timeout_s")
    _require(0.0 < arm_timeout_s <= 300.0, "Manifest arm_timeout_s is outside the finite GUI limit")
    initial_leg_target = _vector(manifest.get("initial_leg_target"), "manifest.initial_leg_target", length=3)
    files = manifest.get("files")
    _require(isinstance(files, dict), "manifest.files is required")
    _require(
        files == {
            "events": EVENTS_FILENAME,
            "states": STATES_FILENAME,
            "attestation": ATTESTATION_FILENAME,
            "process_exit": PROCESS_EXIT_FILENAME,
        },
        "Manifest file names are invalid",
    )
    post_close_exit = manifest.get("post_close_exit")
    _require(isinstance(post_close_exit, dict), "manifest.post_close_exit is required")
    _require(post_close_exit.get("required") is True, "Manifest must require a post-close exit record")
    _require(post_close_exit.get("file") == PROCESS_EXIT_FILENAME, "Manifest post-close exit file is invalid")
    _require(post_close_exit.get("captured_by") == POST_CLOSE_RUNNER, "Manifest post-close runner is invalid")
    post_close_attestation = manifest.get("post_close_attestation")
    _require(isinstance(post_close_attestation, dict), "manifest.post_close_attestation is required")
    _require(
        post_close_attestation.get("required") is True,
        "Manifest must require a post-close operator attestation",
    )
    _require(
        post_close_attestation.get("file") == ATTESTATION_FILENAME,
        "Manifest post-close attestation file is invalid",
    )
    _require(
        post_close_attestation.get("confirmation_method") == ATTESTATION_CONFIRMATION_METHOD,
        "Manifest post-close attestation method is invalid",
    )
    input_capture = manifest.get("input_capture")
    _require(isinstance(input_capture, dict), "manifest.input_capture is required")
    _require(input_capture.get("source") == "carb_keyboard_callback", "Input source is not the Carb callback")
    _require(input_capture.get("observed_only") is True, "Input capture must be observation-only")
    _require(input_capture.get("synthetic_input_injected") is False, "Synthetic input must be false")
    _require(
        input_capture.get("physical_keyboard_independently_proven") is False,
        "Artifact must not claim independent physical-keyboard proof",
    )
    _require(input_capture.get("limitation") == PHYSICALITY_LIMITATION, "Input limitation statement is invalid")
    telemetry = manifest.get("telemetry")
    _require(isinstance(telemetry, dict), "manifest.telemetry is required")
    _require(telemetry.get("states_file") == STATES_FILENAME, "Manifest telemetry must use states.jsonl")
    _require(telemetry.get("interval_action_steps") == 1, "Manifest telemetry interval must be one action step")
    configured = manifest.get("configured_physics")
    _require(isinstance(configured, dict), "manifest.configured_physics is required")
    _require(configured.get("cfg") == PHYSX_CFG_FQN, "Configured physics is not PhysxCfg")
    _require(configured.get("use_newton_actuators") is False, "Configured physics must disable Newton actuators")
    _require(isinstance(manifest.get("started_at_utc"), str) and manifest["started_at_utc"], "Manifest start time is required")
    _require(_as_int(manifest.get("started_monotonic_ns"), "manifest.started_monotonic_ns") > 0, "Manifest monotonic start must be positive")
    return max_steps, initial_leg_target


def _validate_attestation(attestation: dict[str, Any], *, expected_binding: dict[str, str]) -> None:
    _require(attestation.get("schema_version") == SCHEMA_VERSION, "Attestation schema version is invalid")
    _require(attestation.get("kind") == "operator_attestation", "Attestation kind is invalid")
    _require(
        isinstance(attestation.get("operator_name"), str) and attestation["operator_name"].strip(),
        "Attestation operator name is required",
    )
    _require(attestation.get("operator_acknowledged") is True, "Operator acknowledgement is required")
    _require(
        attestation.get("confirmation_method") == ATTESTATION_CONFIRMATION_METHOD,
        "Attestation must use an interactive post-close TTY confirmation",
    )
    _require(
        attestation.get("confirmation_phrase") == TERMINAL_CONFIRMATION_PHRASE,
        "Attestation confirmation phrase is invalid",
    )
    _require(attestation.get("statement") == OPERATOR_ATTESTATION_STATEMENT, "Attestation statement is invalid")
    _require(attestation.get("limitation") == PHYSICALITY_LIMITATION, "Attestation limitation is invalid")
    _require(
        attestation.get("physical_keyboard_independently_proven") is False,
        "Attestation must not claim independent physical-keyboard proof",
    )
    _require(isinstance(attestation.get("recorded_at_utc"), str) and attestation["recorded_at_utc"], "Attestation time is required")
    _require(
        attestation.get("pre_attestation_binding") == expected_binding,
        "Attestation does not bind the sealed post-close artifact",
    )


def _validate_post_close_exit(process_exit: dict[str, Any]) -> None:
    """Require the parent-observed successful exit that follows Kit shutdown."""

    _require(
        process_exit.get("schema_version") == PROCESS_EXIT_SCHEMA_VERSION,
        "Post-close exit schema version is invalid",
    )
    _require(process_exit.get("captured_by") == POST_CLOSE_RUNNER, "Post-close exit runner is invalid")
    _require(process_exit.get("captured_after_child_exit") is True, "Post-close exit was not captured after child exit")
    _require(
        process_exit.get("workload_summary_completed_before_exit") is True,
        "Workload summary was not completed before process exit",
    )
    _require(process_exit.get("summary_outcome_before_exit") == "completed", "Post-close summary outcome is invalid")
    _require(process_exit.get("shell_exit_status") == 0, "GUI process exit status is not zero")
    _require(process_exit.get("exit_status_zero") is True, "GUI process exit is not marked zero")
    _require(process_exit.get("signal_hint") is None, "Successful GUI process exit has a signal hint")
    _require(process_exit.get("acceptance_passed") is True, "GUI artifact is not accepted after process exit")
    _require(
        isinstance(process_exit.get("captured_at_utc"), str) and process_exit["captured_at_utc"],
        "Post-close exit capture time is required",
    )


def _validate_events(root: Path, *, started_monotonic_ns: int, finished_monotonic_ns: int) -> dict[str, int]:
    count = 0
    prior_timestamp = 0
    handled_presses = 0
    unhandled_space_presses = 0
    for expected_sequence, record in enumerate(_records(root / EVENTS_FILENAME, "events"), start=1):
        _require(_as_int(record.get("sequence"), "event.sequence") == expected_sequence, "Event sequence is not contiguous")
        timestamp = _as_int(record.get("callback_monotonic_ns"), "event.callback_monotonic_ns")
        _require(timestamp > 0 and timestamp >= prior_timestamp, "Event callback timestamps are not monotonic")
        _require(
            started_monotonic_ns <= timestamp <= finished_monotonic_ns,
            "Event callback timestamp is outside the recorded run interval",
        )
        prior_timestamp = timestamp
        _require(record.get("source") == "carb_keyboard_callback", "Event source is invalid")
        _require(record.get("callback_observed_only") is True, "Event must be callback observation-only")
        _require(isinstance(record.get("key"), str) and record["key"], "Event key is required")
        _require(record.get("event_type") in OBSERVED_CARB_EVENT_TYPES, "Event type is invalid")
        _require(isinstance(record.get("handled_by_loco_manip"), bool), "Event handled flag is invalid")
        _vector(record.get("base_command"), "event.base_command", length=3)
        held = record.get("held_leg_keys")
        _require(isinstance(held, list) and all(isinstance(key, str) and key for key in held), "Event held keys are invalid")
        _require(isinstance(record.get("clear_success_requested"), bool), "Event clear-success flag is invalid")
        if record["event_type"] == "KEY_PRESS" and record["handled_by_loco_manip"]:
            handled_presses += 1
        if (
            record["key"] == "SPACE"
            and record["event_type"] == "KEY_PRESS"
            and record["handled_by_loco_manip"] is False
        ):
            unhandled_space_presses += 1
        count += 1
    _require(count > 0, "Artifact did not record a keyboard callback")
    _require(handled_presses > 0, "Artifact did not record a handled keyboard press")
    _require(unhandled_space_presses > 0, "Artifact did not record an unhandled SPACE press")
    return {
        "event_count": count,
        "handled_key_press_count": handled_presses,
        "unhandled_space_press_count": unhandled_space_presses,
    }


def _validate_states(
    root: Path,
    *,
    max_steps: int,
    event_count: int,
    initial_leg_target: list[float],
    started_monotonic_ns: int,
    finished_monotonic_ns: int,
) -> dict[str, Any]:
    count = 0
    prior_timestamp = 0
    prior_simulation_time = -math.inf
    active_command_states = 0
    base_nonzero_states = 0
    fl_nonzero_velocity_states = 0
    observed_event_state = False
    max_fl_target_delta_m = 0.0
    initial_root_position: list[float] | None = None
    max_root_position_delta_m = 0.0
    max_button_displacement_m = 0.0
    press_threshold_seen = False
    success_after_threshold_seen = False
    first_success_action_step: int | None = None
    rebound_after_success_seen = False
    first_rebound_action_step: int | None = None
    for expected_sequence, record in enumerate(_records(root / STATES_FILENAME, "states"), start=1):
        _require(_as_int(record.get("sequence"), "state.sequence") == expected_sequence, "State sequence is not contiguous")
        _require(_as_int(record.get("action_step"), "state.action_step") == expected_sequence, "State action step is not contiguous")
        timestamp = _as_int(record.get("monotonic_ns"), "state.monotonic_ns")
        _require(timestamp > 0 and timestamp >= prior_timestamp, "State timestamps are not monotonic")
        _require(
            started_monotonic_ns <= timestamp <= finished_monotonic_ns,
            "State timestamp is outside the recorded run interval",
        )
        prior_timestamp = timestamp
        simulation_time = _as_finite(record.get("simulation_time_s"), "state.simulation_time_s")
        _require(simulation_time >= prior_simulation_time, "State simulation times are not monotonic")
        prior_simulation_time = simulation_time
        base_command = _vector(record.get("base_command"), "state.base_command", length=3)
        leg_velocity = _vector(record.get("leg_velocity"), "state.leg_velocity", length=3)
        leg_target = _vector(record.get("leg_target"), "state.leg_target", length=3)
        root_position = _vector(record.get("root_position_m"), "state.root_position_m", length=3)
        button_displacement_m = _as_finite(record.get("button_displacement_m"), "state.button_displacement_m")
        for field in ("button_success", "button_released", "manipulator_ready", "terminal"):
            _require(isinstance(record.get(field), bool), f"state.{field} must be bool")
        _require(record["terminal"] is False, "GUI artifact contains a terminal step")
        callbacks = _as_int(record.get("callback_events_observed"), "state.callback_events_observed")
        _require(0 <= callbacks <= event_count, "State callback count is outside the event range")
        observed_event_state = observed_event_state or callbacks > 0
        base_nonzero = any(abs(component) > 1.0e-9 for component in base_command)
        fl_nonzero_velocity = any(abs(component) > 1.0e-9 for component in leg_velocity)
        if base_nonzero or fl_nonzero_velocity:
            active_command_states += 1
        if base_nonzero:
            base_nonzero_states += 1
        if fl_nonzero_velocity:
            fl_nonzero_velocity_states += 1
        max_fl_target_delta_m = max(
            max_fl_target_delta_m,
            math.sqrt(sum((current - initial) ** 2 for current, initial in zip(leg_target, initial_leg_target))),
        )
        if initial_root_position is None:
            initial_root_position = root_position
        max_root_position_delta_m = max(
            max_root_position_delta_m,
            math.sqrt(
                sum((current - initial) ** 2 for current, initial in zip(root_position, initial_root_position))
            ),
        )
        max_button_displacement_m = max(max_button_displacement_m, button_displacement_m)
        if button_displacement_m >= BUTTON_PRESS_THRESHOLD_M:
            press_threshold_seen = True
        if record["button_success"] and press_threshold_seen:
            success_after_threshold_seen = True
            if first_success_action_step is None:
                first_success_action_step = expected_sequence
        if (
            first_success_action_step is not None
            and expected_sequence > first_success_action_step
            and record["button_released"]
            and button_displacement_m <= BUTTON_REBOUND_THRESHOLD_M
        ):
            rebound_after_success_seen = True
            if first_rebound_action_step is None:
                first_rebound_action_step = expected_sequence
        count += 1
    _require(count == max_steps, f"State count {count} does not equal requested finite max steps {max_steps}")
    _require(observed_event_state, "No state was recorded after a keyboard callback")
    _require(base_nonzero_states > 0, "No state records a non-zero base command")
    _require(fl_nonzero_velocity_states > 0, "No state records a non-zero FL velocity")
    _require(
        max_fl_target_delta_m >= FL_TARGET_MINIMUM_DELTA_M,
        "FL target did not change by the minimum auditable amount",
    )
    _require(press_threshold_seen, "Button never reached the required 12-mm press threshold")
    _require(success_after_threshold_seen, "Button success was not observed after reaching press threshold")
    _require(rebound_after_success_seen, "Button did not rebound and report released after success")
    return {
        "state_count": count,
        "active_command_state_count": active_command_states,
        "base_nonzero_state_count": base_nonzero_states,
        "fl_nonzero_velocity_state_count": fl_nonzero_velocity_states,
        "max_fl_target_delta_m": max_fl_target_delta_m,
        "max_root_position_delta_m": max_root_position_delta_m,
        "max_button_displacement_m": max_button_displacement_m,
        "button_press_threshold_seen": press_threshold_seen,
        "button_success_after_threshold_seen": success_after_threshold_seen,
        "first_button_success_action_step": first_success_action_step,
        "button_rebound_after_success_seen": rebound_after_success_seen,
        "first_button_rebound_action_step": first_rebound_action_step,
    }


def _validate_activity_summary(
    summary: dict[str, Any],
    *,
    event_evidence: dict[str, int],
    state_evidence: dict[str, Any],
) -> None:
    """Require summary counters to match the independently recomputed M8 evidence."""

    telemetry = summary.get("telemetry")
    _require(isinstance(telemetry, dict), "Summary telemetry is required")
    _require(telemetry.get("states_file") == STATES_FILENAME, "Summary telemetry must use states.jsonl")
    _require(telemetry.get("interval_action_steps") == 1, "Summary telemetry interval must be one action step")
    _require(
        _as_int(telemetry.get("state_records"), "summary.telemetry.state_records")
        == state_evidence["state_count"],
        "Summary telemetry state count disagrees with states",
    )
    activity = summary.get("m8_evidence")
    _require(isinstance(activity, dict), "Summary M8 evidence is required")
    integer_fields = {
        "base_nonzero_state_count": state_evidence["base_nonzero_state_count"],
        "fl_nonzero_velocity_state_count": state_evidence["fl_nonzero_velocity_state_count"],
        "unhandled_space_press_count": event_evidence["unhandled_space_press_count"],
        "first_button_success_action_step": state_evidence["first_button_success_action_step"],
        "first_button_rebound_action_step": state_evidence["first_button_rebound_action_step"],
    }
    for field, expected in integer_fields.items():
        _require(
            _as_int(activity.get(field), f"summary.m8_evidence.{field}") == expected,
            f"Summary M8 evidence disagrees for {field}",
        )
    bool_fields = {
        "button_press_threshold_seen": state_evidence["button_press_threshold_seen"],
        "button_success_after_threshold_seen": state_evidence["button_success_after_threshold_seen"],
        "button_rebound_after_success_seen": state_evidence["button_rebound_after_success_seen"],
    }
    for field, expected in bool_fields.items():
        _require(
            activity.get(field) is expected,
            f"Summary M8 evidence disagrees for {field}",
        )
    float_fields = {
        "max_fl_target_delta_m": state_evidence["max_fl_target_delta_m"],
        "max_root_position_delta_m": state_evidence["max_root_position_delta_m"],
        "max_button_displacement_m": state_evidence["max_button_displacement_m"],
    }
    for field, expected in float_fields.items():
        actual = _as_finite(activity.get(field), f"summary.m8_evidence.{field}")
        _require(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-8),
            f"Summary M8 evidence disagrees for {field}",
        )


def _attestation_binding(root: Path) -> dict[str, str]:
    """Bind a human declaration to the sealed workload and real child exit."""

    return {
        "summary_sha256": sha256_file(root / SUMMARY_FILENAME),
        "process_exit_sha256": sha256_file(root / PROCESS_EXIT_FILENAME),
    }


def validate_pre_attestation_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    """Verify all post-close M8 evidence that must exist before a human attests.

    This standard-library-only helper deliberately does not require
    ``attestation.json``.  The dedicated runner calls it through the interactive
    recorder only after the Kit child has returned and its parent-observed zero
    exit has been sealed.
    """

    root = Path(artifact_dir).expanduser().resolve()
    _require(root.is_dir(), f"Artifact directory does not exist: {root}")
    required = {
        MANIFEST_FILENAME,
        SUMMARY_FILENAME,
        EVENTS_FILENAME,
        STATES_FILENAME,
        PROCESS_EXIT_FILENAME,
        CHECKSUMS_FILENAME,
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    _require(not missing, f"Artifact is missing required pre-attestation files: {', '.join(missing)}")
    # Verify the immutable evidence set before interpreting any of its records.
    checksum_count = _validate_checksums(root)
    manifest = _read_json(root / MANIFEST_FILENAME, "manifest")
    summary = _read_json(root / SUMMARY_FILENAME, "summary")
    process_exit = _read_json(root / PROCESS_EXIT_FILENAME, "post-close process exit")
    max_steps, initial_leg_target = _validate_manifest(manifest)

    # The order matters for the interactive recorder: a declaration is never
    # requested or written unless a parent has already observed a clean child
    # exit and the workload says it completed before that exit.
    _validate_post_close_exit(process_exit)
    _require(summary.get("schema_version") == SCHEMA_VERSION, "Summary schema version is invalid")
    _require(summary.get("artifact_kind") == ARTIFACT_KIND, "Summary artifact kind is invalid")
    _require(summary.get("task") == TASK_ID, "Summary task is invalid")
    _require(summary.get("outcome") == "completed", "GUI artifact outcome is not completed")
    _require(summary.get("visualizer") == VISUALIZER, "Summary visualizer is invalid")
    _require(summary.get("smoke_mode") is False, "Summary must not label a smoke run")
    _require(summary.get("max_action_steps") == max_steps, "Summary max steps disagree with manifest")
    _require(
        _as_finite(summary.get("arm_timeout_s"), "summary.arm_timeout_s")
        == _as_finite(manifest.get("arm_timeout_s"), "manifest.arm_timeout_s"),
        "Summary arm timeout disagrees with manifest",
    )
    _require(summary.get("executed_action_steps") == max_steps, "GUI artifact did not reach its finite step limit")
    _require(summary.get("operator_attestation_required") is True, "Summary must require operator attestation")
    _require(
        summary.get("physical_keyboard_independently_proven") is False,
        "Summary must not claim independent physical-keyboard proof",
    )
    _require(summary.get("limitation") == PHYSICALITY_LIMITATION, "Summary limitation is invalid")
    _require(summary.get("post_close_exit_required") is True, "Summary must require a post-close exit record")
    _require(summary.get("post_close_exit_file") == PROCESS_EXIT_FILENAME, "Summary post-close exit file is invalid")
    _require(summary.get("post_close_exit_runner") == POST_CLOSE_RUNNER, "Summary post-close runner is invalid")
    _require(
        summary.get("post_close_attestation_required") is True,
        "Summary must require a post-close operator attestation",
    )
    _require(
        summary.get("post_close_attestation_file") == ATTESTATION_FILENAME,
        "Summary post-close attestation file is invalid",
    )
    _require(
        summary.get("post_close_attestation_confirmation_method") == ATTESTATION_CONFIRMATION_METHOD,
        "Summary post-close attestation method is invalid",
    )
    _validate_physx_evidence(summary.get("backend_before"), "summary.backend_before")
    _validate_physx_evidence(summary.get("backend_after"), "summary.backend_after")
    checkpoint = summary.get("checkpoint")
    _require(isinstance(checkpoint, dict), "Summary checkpoint is required")
    _require(checkpoint.get("sha256") == QUADRUPED_CHECKPOINT_SHA256, "Summary checkpoint SHA-256 is invalid")
    _require(isinstance(summary.get("finished_at_utc"), str) and summary["finished_at_utc"], "Summary finish time is required")
    started_monotonic_ns = _as_int(manifest["started_monotonic_ns"], "manifest.started_monotonic_ns")
    finished_monotonic_ns = _as_int(summary.get("finished_monotonic_ns"), "summary.finished_monotonic_ns")
    _require(finished_monotonic_ns >= started_monotonic_ns, "Summary finish precedes manifest start")
    event_evidence = _validate_events(
        root,
        started_monotonic_ns=started_monotonic_ns,
        finished_monotonic_ns=finished_monotonic_ns,
    )
    state_evidence = _validate_states(
        root,
        max_steps=max_steps,
        event_count=event_evidence["event_count"],
        initial_leg_target=initial_leg_target,
        started_monotonic_ns=started_monotonic_ns,
        finished_monotonic_ns=finished_monotonic_ns,
    )
    _require(
        summary.get("event_count") == event_evidence["event_count"],
        "Summary event count disagrees with events",
    )
    _require(
        summary.get("state_count") == state_evidence["state_count"],
        "Summary state count disagrees with states",
    )
    _require(
        summary.get("handled_key_press_count") == event_evidence["handled_key_press_count"],
        "Summary handled-key press count disagrees with events",
    )
    _require(
        summary.get("active_command_state_count") == state_evidence["active_command_state_count"],
        "Summary active-command state count disagrees with states",
    )
    _validate_activity_summary(summary, event_evidence=event_evidence, state_evidence=state_evidence)
    return {
        "artifact_dir": str(root),
        "schema_version": SCHEMA_VERSION,
        "max_action_steps": max_steps,
        "event_count": event_evidence["event_count"],
        "state_count": state_evidence["state_count"],
        "checksum_count": checksum_count,
        "process_exit_status": process_exit["shell_exit_status"],
        "m8_evidence": state_evidence | {"unhandled_space_press_count": event_evidence["unhandled_space_press_count"]},
        "attestation_binding": _attestation_binding(root),
    }


def write_post_close_operator_attestation(
    artifact_dir: str | Path, *, operator_name: str, confirmation_phrase: str
) -> dict[str, Any]:
    """Add one human declaration only after all post-close evidence validates.

    The caller must pass the exact phrase entered at the interactive TTY.  The
    dedicated recorder obtains it only after its preflight succeeds.
    """

    root = Path(artifact_dir).expanduser().resolve()
    evidence = validate_pre_attestation_artifact(root)
    _require(not (root / ATTESTATION_FILENAME).exists(), "Refusing to overwrite an existing operator attestation")
    name = operator_name.strip()
    _require(name, "Operator name must not be empty")
    _require(len(name) <= 160, "Operator name must be at most 160 characters")
    _require(
        confirmation_phrase == TERMINAL_CONFIRMATION_PHRASE,
        "Operator acknowledgement phrase did not match",
    )
    attestation = {
        "schema_version": SCHEMA_VERSION,
        "kind": "operator_attestation",
        "operator_name": name,
        "operator_acknowledged": True,
        "confirmation_method": ATTESTATION_CONFIRMATION_METHOD,
        "confirmation_phrase": confirmation_phrase,
        "statement": OPERATOR_ATTESTATION_STATEMENT,
        "limitation": PHYSICALITY_LIMITATION,
        "physical_keyboard_independently_proven": False,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pre_attestation_binding": evidence["attestation_binding"],
    }
    write_json(root / ATTESTATION_FILENAME, attestation)
    write_checksums(root)
    return {
        "artifact_dir": str(root),
        "operator_name": name,
        "checksummed_files": _validate_checksums(root),
        "process_exit_status": evidence["process_exit_status"],
    }


def validate_gui_keyboard_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    """Validate a completed GUI-keyboard artifact without importing Kit.

    A passing return value proves the structure, explicit PhysX evidence, and
    a post-close operator declaration are internally consistent. It
    intentionally does *not* claim to independently prove physical keyboard
    use.
    """

    evidence = validate_pre_attestation_artifact(artifact_dir)
    root = Path(evidence["artifact_dir"])
    attestation_path = root / ATTESTATION_FILENAME
    _require(attestation_path.is_file(), f"Artifact is missing required file: {ATTESTATION_FILENAME}")
    attestation = _read_json(attestation_path, "attestation")
    _validate_attestation(attestation, expected_binding=evidence["attestation_binding"])
    return {
        **evidence,
        "operator_attestation_required": True,
        "operator_name": attestation["operator_name"],
        "physical_keyboard_independently_proven": False,
        "limitation": PHYSICALITY_LIMITATION,
    }


__all__ = [
    "ARTIFACT_KIND",
    "ATTESTATION_CONFIRMATION_METHOD",
    "ATTESTATION_FILENAME",
    "BUTTON_PRESS_THRESHOLD_M",
    "BUTTON_REBOUND_THRESHOLD_M",
    "CHECKSUMS_FILENAME",
    "EVENTS_FILENAME",
    "FL_TARGET_MINIMUM_DELTA_M",
    "GuiKeyboardArtifactError",
    "GuiKeyboardArtifactWriter",
    "MANIFEST_FILENAME",
    "MAX_ACTION_STEPS",
    "MODE",
    "OPERATOR_ATTESTATION_STATEMENT",
    "PHYSICALITY_LIMITATION",
    "PHYSX_CFG_FQN",
    "PHYSX_MANAGER_FQN",
    "POST_CLOSE_RUNNER",
    "PROCESS_EXIT_FILENAME",
    "PROCESS_EXIT_SCHEMA_VERSION",
    "QUADRUPED_CHECKPOINT_SHA256",
    "SCHEMA_VERSION",
    "STATES_FILENAME",
    "SUMMARY_FILENAME",
    "TASK_ID",
    "TERMINAL_CONFIRMATION_PHRASE",
    "VISUALIZER",
    "jsonable",
    "sha256_file",
    "validate_gui_keyboard_artifact",
    "validate_pre_attestation_artifact",
    "write_checksums",
    "write_json",
    "write_post_close_operator_attestation",
]
