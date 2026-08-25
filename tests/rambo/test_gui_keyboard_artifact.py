"""Pure-Python contracts for the auditable GUI keyboard artifact pathway."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rambo.validation import gui_keyboard_artifact as schema


def test_offline_schema_imports_only_standard_library_modules() -> None:
    source_path = Path(schema.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    from_imports = {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
    }
    assert imports | from_imports <= {"__future__", "datetime", "hashlib", "json", "math", "pathlib", "typing"}


def _backend_evidence() -> dict[str, object]:
    return {
        "requested_cfg": schema.PHYSX_CFG_FQN,
        "actual_manager": schema.PHYSX_MANAGER_FQN,
        "configured_manager": schema.PHYSX_MANAGER_FQN,
        "use_newton_actuators": False,
    }


def _manifest(*, max_steps: int) -> dict[str, object]:
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "artifact_kind": schema.ARTIFACT_KIND,
        "task": schema.TASK_ID,
        "mode": schema.MODE,
        "visualizer": schema.VISUALIZER,
        "smoke_mode": False,
        "max_action_steps": max_steps,
        "arm_timeout_s": 120.0,
        "initial_leg_target": [0.1934, 0.142, 0.05],
        "files": {
            "events": schema.EVENTS_FILENAME,
            "states": schema.STATES_FILENAME,
            "attestation": schema.ATTESTATION_FILENAME,
            "process_exit": schema.PROCESS_EXIT_FILENAME,
        },
        "post_close_exit": {
            "required": True,
            "file": schema.PROCESS_EXIT_FILENAME,
            "captured_by": schema.POST_CLOSE_RUNNER,
        },
        "post_close_attestation": {
            "required": True,
            "file": schema.ATTESTATION_FILENAME,
            "confirmation_method": schema.ATTESTATION_CONFIRMATION_METHOD,
        },
        "input_capture": {
            "source": "carb_keyboard_callback",
            "observed_only": True,
            "synthetic_input_injected": False,
            "physical_keyboard_independently_proven": False,
            "limitation": schema.PHYSICALITY_LIMITATION,
        },
        "telemetry": {"states_file": schema.STATES_FILENAME, "interval_action_steps": 1},
        "configured_physics": {"cfg": schema.PHYSX_CFG_FQN, "use_newton_actuators": False},
        "backend_before": _backend_evidence(),
        "started_at_utc": "2026-08-24T12:00:00Z",
        "started_monotonic_ns": 1,
    }


def _load_gui_finalizer():
    script = Path(__file__).resolve().parents[2] / "scripts/rambo/finalize_gui_keyboard_artifact.py"
    spec = importlib.util.spec_from_file_location("rambo_gui_keyboard_artifact_finalizer", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seal_post_close(output_dir: Path, *, exit_status: int = 0) -> dict[str, object]:
    finalizer = _load_gui_finalizer()
    return finalizer.finalize_gui_keyboard_artifact(output_dir, exit_status=exit_status)


def _write_valid_artifact(output_dir: Path, *, max_steps: int = 3, seal_post_close: bool = True) -> Path:
    if max_steps < 3:
        raise ValueError("The complete M8 fixture needs at least three states")
    writer = schema.GuiKeyboardArtifactWriter(
        output_dir,
        manifest=_manifest(max_steps=max_steps),
    )
    writer.record_event(
        {
            "callback_monotonic_ns": 10,
            "source": "carb_keyboard_callback",
            "callback_observed_only": True,
            "key": "UP",
            "event_type": "KEY_PRESS",
            "handled_by_loco_manip": True,
            "held_leg_keys": [],
            "base_command": [0.4, 0.0, 0.0],
            "clear_success_requested": False,
        }
    )
    writer.record_event(
        {
            "callback_monotonic_ns": 11,
            "source": "carb_keyboard_callback",
            "callback_observed_only": True,
            "key": "W",
            "event_type": "KEY_PRESS",
            "handled_by_loco_manip": True,
            "held_leg_keys": ["W"],
            "base_command": [0.4, 0.0, 0.0],
            "clear_success_requested": False,
        }
    )
    writer.record_event(
        {
            "callback_monotonic_ns": 11,
            "source": "carb_keyboard_callback",
            "callback_observed_only": True,
            "key": "W",
            "event_type": "KEY_REPEAT",
            "handled_by_loco_manip": True,
            "held_leg_keys": ["W"],
            "base_command": [0.4, 0.0, 0.0],
            "clear_success_requested": False,
        }
    )
    writer.record_event(
        {
            "callback_monotonic_ns": 11,
            "source": "carb_keyboard_callback",
            "callback_observed_only": True,
            "key": "w",
            "event_type": "CHAR",
            "handled_by_loco_manip": False,
            "held_leg_keys": ["W"],
            "base_command": [0.4, 0.0, 0.0],
            "clear_success_requested": False,
        }
    )
    writer.record_event(
        {
            "callback_monotonic_ns": 12,
            "source": "carb_keyboard_callback",
            "callback_observed_only": True,
            "key": "SPACE",
            "event_type": "KEY_PRESS",
            "handled_by_loco_manip": False,
            "held_leg_keys": ["W"],
            "base_command": [0.4, 0.0, 0.0],
            "clear_success_requested": False,
        }
    )
    for step in range(1, max_steps + 1):
        if step == 1:
            base_command = [0.4, 0.0, 0.0]
            leg_velocity = [0.0, 0.0, 0.1]
            leg_target = [0.1934, 0.142, 0.051]
            button_displacement_m = 0.0
            button_success = False
            button_released = True
        elif step == 2:
            base_command = [0.4, 0.0, 0.0]
            leg_velocity = [0.12, 0.0, 0.0]
            leg_target = [0.1944, 0.142, 0.051]
            button_displacement_m = schema.BUTTON_PRESS_THRESHOLD_M
            button_success = True
            button_released = False
        else:
            base_command = [0.0, 0.0, 0.0]
            leg_velocity = [-0.12, 0.0, 0.0] if step == 3 else [0.0, 0.0, 0.0]
            leg_target = [0.1934, 0.142, 0.051]
            button_displacement_m = 0.001
            button_success = True
            button_released = True
        writer.record_state(
            {
                "action_step": step,
                "monotonic_ns": 10 + step,
                "simulation_time_s": step * 0.01,
                "base_command": base_command,
                "leg_velocity": leg_velocity,
                "leg_target": leg_target,
                "root_position_m": [0.02 * (step - 1), 0.0, 0.3],
                "button_displacement_m": button_displacement_m,
                "button_success": button_success,
                "button_released": button_released,
                "manipulator_ready": True,
                "terminal": False,
                "callback_events_observed": writer.event_count,
            }
        )
    writer.finalize(
        {
            "schema_version": schema.SCHEMA_VERSION,
            "artifact_kind": schema.ARTIFACT_KIND,
            "task": schema.TASK_ID,
            "outcome": "completed",
            "visualizer": schema.VISUALIZER,
            "smoke_mode": False,
            "max_action_steps": max_steps,
            "arm_timeout_s": 120.0,
            "executed_action_steps": max_steps,
            "event_count": writer.event_count,
            "state_count": writer.state_count,
            "handled_key_press_count": 2,
            "active_command_state_count": 3,
            "configured_physics": {"cfg": schema.PHYSX_CFG_FQN, "use_newton_actuators": False},
            "backend_before": _backend_evidence(),
            "backend_after": _backend_evidence(),
            "checkpoint": {"path": "/trusted/model_2000.pt", "sha256": schema.QUADRUPED_CHECKPOINT_SHA256},
            "runtime": {"python": "3.12.3", "torch": "2.10.0+cu128", "device": "cuda:0"},
            "telemetry": {
                "states_file": schema.STATES_FILENAME,
                "interval_action_steps": 1,
                "state_records": writer.state_count,
            },
            "m8_evidence": {
                "base_nonzero_state_count": 2,
                "fl_nonzero_velocity_state_count": 3,
                "unhandled_space_press_count": 1,
                "max_fl_target_delta_m": 0.0014142135623730972,
                "max_root_position_delta_m": 0.04,
                "max_button_displacement_m": schema.BUTTON_PRESS_THRESHOLD_M,
                "button_press_threshold_seen": True,
                "button_success_after_threshold_seen": True,
                "first_button_success_action_step": 2,
                "button_rebound_after_success_seen": True,
                "first_button_rebound_action_step": 3,
            },
            "operator_attestation_required": True,
            "physical_keyboard_independently_proven": False,
            "limitation": schema.PHYSICALITY_LIMITATION,
            "post_close_exit_required": True,
            "post_close_exit_file": schema.PROCESS_EXIT_FILENAME,
            "post_close_exit_runner": schema.POST_CLOSE_RUNNER,
            "post_close_attestation_required": True,
            "post_close_attestation_file": schema.ATTESTATION_FILENAME,
            "post_close_attestation_confirmation_method": schema.ATTESTATION_CONFIRMATION_METHOD,
            "finished_at_utc": "2026-08-24T12:00:01Z",
            "finished_monotonic_ns": 100,
        }
    )
    if seal_post_close:
        _seal_post_close(output_dir)
        schema.write_post_close_operator_attestation(
            output_dir,
            operator_name="test-operator",
            confirmation_phrase=schema.TERMINAL_CONFIRMATION_PHRASE,
        )
    return output_dir


def test_offline_validator_accepts_complete_callback_and_state_evidence(tmp_path: Path) -> None:
    artifact = _write_valid_artifact(tmp_path / "artifact")

    result = schema.validate_gui_keyboard_artifact(artifact)

    assert result["event_count"] == 5
    assert result["state_count"] == 3
    assert result["checksum_count"] == 6
    assert result["process_exit_status"] == 0
    assert result["m8_evidence"]["unhandled_space_press_count"] == 1
    assert result["m8_evidence"]["button_rebound_after_success_seen"] is True
    assert result["operator_attestation_required"] is True
    assert result["operator_name"] == "test-operator"
    assert result["physical_keyboard_independently_proven"] is False
    assert "cannot independently prove" in result["limitation"]


def test_offline_validator_requires_dedicated_post_close_zero_exit(tmp_path: Path) -> None:
    artifact = _write_valid_artifact(tmp_path / "artifact", seal_post_close=False)

    with pytest.raises(schema.GuiKeyboardArtifactError, match="process_exit.json"):
        schema.validate_gui_keyboard_artifact(artifact)

    record = _seal_post_close(artifact)["process_exit"]
    assert record["captured_by"] == schema.POST_CLOSE_RUNNER
    assert record["captured_after_child_exit"] is True
    assert record["workload_summary_completed_before_exit"] is True
    assert record["acceptance_passed"] is True
    with pytest.raises(schema.GuiKeyboardArtifactError, match="attestation.json"):
        schema.validate_gui_keyboard_artifact(artifact)
    attestation = schema.write_post_close_operator_attestation(
        artifact,
        operator_name="test-operator",
        confirmation_phrase=schema.TERMINAL_CONFIRMATION_PHRASE,
    )
    assert attestation["process_exit_status"] == 0
    assert schema.validate_gui_keyboard_artifact(artifact)["process_exit_status"] == 0


def test_post_close_attestation_refuses_unsealed_or_failed_evidence(tmp_path: Path) -> None:
    artifact = _write_valid_artifact(tmp_path / "unsealed", seal_post_close=False)

    with pytest.raises(schema.GuiKeyboardArtifactError, match="pre-attestation files"):
        schema.write_post_close_operator_attestation(
            artifact,
            operator_name="test-operator",
            confirmation_phrase=schema.TERMINAL_CONFIRMATION_PHRASE,
        )
    assert not (artifact / schema.ATTESTATION_FILENAME).exists()

    artifact = _write_valid_artifact(tmp_path / "nonzero", seal_post_close=False)
    _seal_post_close(artifact, exit_status=139)
    with pytest.raises(schema.GuiKeyboardArtifactError, match="GUI process exit status is not zero"):
        schema.write_post_close_operator_attestation(
            artifact,
            operator_name="test-operator",
            confirmation_phrase=schema.TERMINAL_CONFIRMATION_PHRASE,
        )
    assert not (artifact / schema.ATTESTATION_FILENAME).exists()

    artifact = _write_valid_artifact(tmp_path / "bad-phrase", seal_post_close=False)
    _seal_post_close(artifact)
    with pytest.raises(schema.GuiKeyboardArtifactError, match="phrase did not match"):
        schema.write_post_close_operator_attestation(
            artifact,
            operator_name="test-operator",
            confirmation_phrase="I did not enter the required acknowledgement",
        )
    assert not (artifact / schema.ATTESTATION_FILENAME).exists()


def test_offline_validator_rejects_nonzero_or_precompletion_post_close_exit(tmp_path: Path) -> None:
    artifact = _write_valid_artifact(tmp_path / "nonzero", seal_post_close=False)
    record = _seal_post_close(artifact, exit_status=139)["process_exit"]
    assert record["acceptance_passed"] is False
    assert record["signal_hint"] == 11
    with pytest.raises(schema.GuiKeyboardArtifactError, match="GUI process exit status is not zero"):
        schema.validate_gui_keyboard_artifact(artifact)

    artifact = _write_valid_artifact(tmp_path / "precompletion")
    exit_path = artifact / schema.PROCESS_EXIT_FILENAME
    process_exit = json.loads(exit_path.read_text(encoding="utf-8"))
    process_exit["workload_summary_completed_before_exit"] = False
    schema.write_json(exit_path, process_exit)
    schema.write_checksums(artifact)
    with pytest.raises(schema.GuiKeyboardArtifactError, match="not completed before process exit"):
        schema.validate_gui_keyboard_artifact(artifact)


def test_offline_validator_rejects_synthetic_or_independent_physicality_claim(tmp_path: Path) -> None:
    artifact = _write_valid_artifact(tmp_path / "artifact")
    manifest_path = artifact / schema.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_capture"]["physical_keyboard_independently_proven"] = True
    schema.write_json(manifest_path, manifest)
    schema.write_checksums(artifact)

    with pytest.raises(schema.GuiKeyboardArtifactError, match="independent physical-keyboard proof"):
        schema.validate_gui_keyboard_artifact(artifact)


def _rewrite_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_offline_validator_requires_base_fl_and_unhandled_space_evidence(tmp_path: Path) -> None:
    artifact = _write_valid_artifact(tmp_path / "artifact")
    events_path = artifact / schema.EVENTS_FILENAME
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events[-1]["handled_by_loco_manip"] = True
    _rewrite_jsonl(events_path, events)
    schema.write_checksums(artifact)

    with pytest.raises(schema.GuiKeyboardArtifactError, match="unhandled SPACE"):
        schema.validate_gui_keyboard_artifact(artifact)

    artifact = _write_valid_artifact(tmp_path / "no-base")
    states_path = artifact / schema.STATES_FILENAME
    states = [json.loads(line) for line in states_path.read_text(encoding="utf-8").splitlines()]
    for state in states:
        state["base_command"] = [0.0, 0.0, 0.0]
    _rewrite_jsonl(states_path, states)
    schema.write_checksums(artifact)

    with pytest.raises(schema.GuiKeyboardArtifactError, match="non-zero base command"):
        schema.validate_gui_keyboard_artifact(artifact)

    artifact = _write_valid_artifact(tmp_path / "no-fl")
    states_path = artifact / schema.STATES_FILENAME
    states = [json.loads(line) for line in states_path.read_text(encoding="utf-8").splitlines()]
    for state in states:
        state["leg_velocity"] = [0.0, 0.0, 0.0]
        state["leg_target"] = [0.1934, 0.142, 0.05]
    _rewrite_jsonl(states_path, states)
    schema.write_checksums(artifact)

    with pytest.raises(schema.GuiKeyboardArtifactError, match="non-zero FL velocity"):
        schema.validate_gui_keyboard_artifact(artifact)


def test_offline_validator_requires_button_press_success_and_rebound(tmp_path: Path) -> None:
    artifact = _write_valid_artifact(tmp_path / "no-press")
    states_path = artifact / schema.STATES_FILENAME
    states = [json.loads(line) for line in states_path.read_text(encoding="utf-8").splitlines()]
    for state in states:
        state["button_displacement_m"] = 0.001
        state["button_success"] = False
        state["button_released"] = True
    _rewrite_jsonl(states_path, states)
    schema.write_checksums(artifact)

    with pytest.raises(schema.GuiKeyboardArtifactError, match="12-mm press threshold"):
        schema.validate_gui_keyboard_artifact(artifact)

    artifact = _write_valid_artifact(tmp_path / "no-rebound")
    states_path = artifact / schema.STATES_FILENAME
    states = [json.loads(line) for line in states_path.read_text(encoding="utf-8").splitlines()]
    for state in states[2:]:
        state["button_displacement_m"] = schema.BUTTON_PRESS_THRESHOLD_M
        state["button_released"] = False
    _rewrite_jsonl(states_path, states)
    schema.write_checksums(artifact)

    with pytest.raises(schema.GuiKeyboardArtifactError, match="did not rebound"):
        schema.validate_gui_keyboard_artifact(artifact)


def test_offline_validator_rejects_no_callback_or_no_keyboard_driven_state(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    writer = schema.GuiKeyboardArtifactWriter(
        artifact,
        manifest=_manifest(max_steps=1),
    )
    writer.record_state(
        {
            "action_step": 1,
            "monotonic_ns": 11,
            "simulation_time_s": 0.01,
            "base_command": [0.0, 0.0, 0.0],
            "leg_velocity": [0.0, 0.0, 0.0],
            "leg_target": [0.1934, 0.142, 0.05],
            "root_position_m": [0.0, 0.0, 0.3],
            "button_displacement_m": 0.0,
            "button_success": False,
            "button_released": True,
            "manipulator_ready": True,
            "terminal": False,
            "callback_events_observed": 0,
        }
    )
    writer.finalize(
        {
            "schema_version": schema.SCHEMA_VERSION,
            "artifact_kind": schema.ARTIFACT_KIND,
            "task": schema.TASK_ID,
            "outcome": "completed",
            "visualizer": schema.VISUALIZER,
            "smoke_mode": False,
            "max_action_steps": 1,
            "arm_timeout_s": 120.0,
            "executed_action_steps": 1,
            "event_count": 0,
            "state_count": 1,
            "handled_key_press_count": 0,
            "active_command_state_count": 0,
            "configured_physics": {"cfg": schema.PHYSX_CFG_FQN, "use_newton_actuators": False},
            "backend_before": _backend_evidence(),
            "backend_after": _backend_evidence(),
            "checkpoint": {"sha256": schema.QUADRUPED_CHECKPOINT_SHA256},
            "operator_attestation_required": True,
            "physical_keyboard_independently_proven": False,
            "limitation": schema.PHYSICALITY_LIMITATION,
            "post_close_exit_required": True,
            "post_close_exit_file": schema.PROCESS_EXIT_FILENAME,
            "post_close_exit_runner": schema.POST_CLOSE_RUNNER,
            "post_close_attestation_required": True,
            "post_close_attestation_file": schema.ATTESTATION_FILENAME,
            "post_close_attestation_confirmation_method": schema.ATTESTATION_CONFIRMATION_METHOD,
            "finished_at_utc": "2026-08-24T12:00:01Z",
            "finished_monotonic_ns": 100,
        }
    )
    _seal_post_close(artifact)

    with pytest.raises(schema.GuiKeyboardArtifactError, match="did not record a keyboard callback"):
        schema.validate_gui_keyboard_artifact(artifact)


def test_offline_validator_detects_tampering_after_checksums_are_written(tmp_path: Path) -> None:
    artifact = _write_valid_artifact(tmp_path / "artifact")
    states = artifact / schema.STATES_FILENAME
    states.write_text(states.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(schema.GuiKeyboardArtifactError, match="Checksum mismatch"):
        schema.validate_gui_keyboard_artifact(artifact)


def test_offline_validator_rejects_unchecksummed_nested_checksum_named_file(tmp_path: Path) -> None:
    artifact = _write_valid_artifact(tmp_path / "artifact")
    nested = artifact / "nested" / schema.CHECKSUMS_FILENAME
    nested.parent.mkdir()
    nested.write_text("unchecksummed evidence\n", encoding="utf-8")

    with pytest.raises(schema.GuiKeyboardArtifactError, match="does not cover exactly all evidence files"):
        schema.validate_gui_keyboard_artifact(artifact)


def _load_teleop_module():
    script = Path(__file__).resolve().parents[2] / "scripts/rambo/teleop_loco_manip.py"
    spec = importlib.util.spec_from_file_location("rambo_gui_artifact_teleop", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Parser:
    def error(self, message: str) -> None:
        raise ValueError(message)


def _gui_args(teleop, **overrides):
    values = {
        "gui_artifact_dir": teleop.GUI_ARTIFACT_ROOT / "pytest-fresh-artifact",
        "operator_name": "operator",
        "rambo_visualizer": ["kit"],
        "smoke_walk": False,
        "smoke_press": False,
        "smoke_loco_manip": False,
        "max_steps": 8,
        "gui_arm_timeout_s": 120.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gui_artifact_arguments_fail_closed_before_app_launcher() -> None:
    teleop = _load_teleop_module()
    parser = _Parser()
    assert teleop.GUI_ARTIFACT_MAX_STEPS == schema.MAX_ACTION_STEPS
    assert teleop._keyboard_event_type_name(SimpleNamespace(type=SimpleNamespace(name="KEY_PRESS"))) == "KEY_PRESS"
    assert teleop._keyboard_event_type_name(SimpleNamespace(type="KeyboardEventType.KEY_RELEASE")) == "KEY_RELEASE"
    valid = _gui_args(teleop)
    teleop._validate_gui_artifact_args(parser, valid)
    assert valid.gui_artifact_dir.is_absolute()

    with pytest.raises(ValueError, match="--viz kit"):
        teleop._validate_gui_artifact_args(parser, _gui_args(teleop, rambo_visualizer=["none"]))
    with pytest.raises(ValueError, match="forbids all scripted smoke"):
        teleop._validate_gui_artifact_args(parser, _gui_args(teleop, smoke_press=True))
    with pytest.raises(ValueError, match="finite --max-steps"):
        teleop._validate_gui_artifact_args(parser, _gui_args(teleop, max_steps=0))
    with pytest.raises(ValueError, match="operator-name"):
        teleop._validate_gui_artifact_args(parser, _gui_args(teleop, operator_name=" "))
    with pytest.raises(ValueError, match="gui-arm-timeout"):
        teleop._validate_gui_artifact_args(parser, _gui_args(teleop, gui_arm_timeout_s=0.0))
    with pytest.raises(ValueError, match="below the M8 artifact root"):
        teleop._validate_gui_artifact_args(parser, _gui_args(teleop, gui_artifact_dir=Path("/tmp/not-m8")))
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        teleop._validate_gui_artifact_args(parser, _gui_args(teleop, gui_artifact_dir=teleop.GUI_ARTIFACT_ROOT))


def _call_name(node: ast.Call) -> str:
    target = node.func
    parts: list[str] = []
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if isinstance(target, ast.Name):
        parts.append(target.id)
    return ".".join(reversed(parts))


def test_gui_recorder_observes_existing_callback_without_input_injection_api() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/rambo/teleop_loco_manip.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "pyautogui" not in imported_modules
    call_names = {_call_name(node).lower() for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not any("inject" in name or "xdotool" in name for name in call_names)
    source = script.read_text(encoding="utf-8")
    assert "event_observer=observe_gui_keyboard_event" in source
    assert '"source": "carb_keyboard_callback"' in source
    assert "--operator-attestation" not in source
    assert "configure_physx(env_cfg)" in source
    assert source.count("assert_physx_environment(env)") >= 2


def test_operator_statement_allows_live_selkies_human_keyboard_but_rejects_automation() -> None:
    statement = schema.OPERATOR_ATTESTATION_STATEMENT
    assert "physical keyboard" in statement
    assert "live Selkies WebRTC session" in statement
    assert "scripted" in statement
    assert "replayed" in statement
    assert "automated" in statement
    assert "injected" in statement


def test_dedicated_m8_runner_rejects_non_gui_visualizer_before_python(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "scripts/rambo/run_gui_keyboard_artifact.sh"
    artifact = tmp_path / "artifact"
    result = subprocess.run(
        [
            str(runner),
            "--gui-artifact-dir",
            str(artifact),
            "--operator-name",
            "operator",
            "--viz",
            "none",
        ],
        cwd=root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "require explicit --viz kit" in result.stderr
    assert not artifact.exists()


def test_dedicated_m8_runner_requires_eula_and_visible_display_before_python(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "scripts/rambo/run_gui_keyboard_artifact.sh"
    arguments = [
        str(runner),
        "--gui-artifact-dir",
        str(tmp_path / "artifact"),
        "--operator-name",
        "operator",
        "--viz",
        "kit",
    ]

    missing_eula = os.environ.copy()
    missing_eula.pop("OMNI_KIT_ACCEPT_EULA", None)
    missing_eula["DISPLAY"] = ":test-visible-display"
    result = subprocess.run(
        arguments,
        cwd=root,
        env=missing_eula,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "OMNI_KIT_ACCEPT_EULA=Y" in result.stderr
    assert not (tmp_path / "artifact").exists()

    missing_display = os.environ.copy()
    missing_display["OMNI_KIT_ACCEPT_EULA"] = "Y"
    missing_display.pop("DISPLAY", None)
    result = subprocess.run(
        arguments,
        cwd=root,
        env=missing_display,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires DISPLAY" in result.stderr
    assert not (tmp_path / "artifact").exists()

    obsolete_flag = os.environ.copy()
    obsolete_flag.pop("OMNI_KIT_ACCEPT_EULA", None)
    obsolete_flag.pop("DISPLAY", None)
    result = subprocess.run(
        [*arguments, "--operator-attestation"],
        cwd=root,
        env=obsolete_flag,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "operator-attestation is obsolete" in result.stderr
    assert not (tmp_path / "artifact").exists()


def test_dedicated_m8_runner_uses_post_close_finalizer_recorder_and_validator() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/rambo/run_gui_keyboard_artifact.sh").read_text(encoding="utf-8")
    assert 'rambo_validate_launch_contract "$@"' in source
    assert '"${SCRIPT_DIR}/run60.sh" "${TELEOP_SCRIPT}" "$@"' in source
    assert '"${VENV_DIR}/bin/python" "${FINALIZER}"' in source
    assert '"${VENV_DIR}/bin/python" "${RECORDER}"' in source
    assert '"${VENV_DIR}/bin/python" "${VALIDATOR}" --artifact-dir "${artifact_dir}"' in source
    assert "--process-exit-status" in source
    assert "OMNI_KIT_ACCEPT_EULA=Y" in source
    assert "requires DISPLAY" in source
    assert "--operator-attestation is obsolete" in source
    assert "unset OMNI_KIT_ACCEPT_EULA" in source
    assert source.index('if [[ "${child_status}" -ne 0 ]]; then') < source.index(
        '"${VENV_DIR}/bin/python" "${RECORDER}"'
    )


def test_post_close_recorder_requires_tty_phrase_and_preflight() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/rambo/record_gui_keyboard_attestation.py").read_text(encoding="utf-8")
    assert "validate_pre_attestation_artifact" in source
    assert "write_post_close_operator_attestation" in source
    assert "sys.stdin.isatty()" in source
    assert "TERMINAL_CONFIRMATION_PHRASE" in source


def test_post_close_recorder_refuses_redirected_acknowledgement(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    recorder = root / "scripts/rambo/record_gui_keyboard_attestation.py"
    environment = os.environ.copy()
    source_paths = [str(root / "source/rambo"), str(root / "source/crl2")]
    if existing := environment.get("PYTHONPATH"):
        source_paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(source_paths)

    unsealed = _write_valid_artifact(tmp_path / "unsealed", seal_post_close=False)
    result = subprocess.run(
        [sys.executable, str(recorder), "--artifact-dir", str(unsealed), "--operator-name", "operator"],
        cwd=root,
        env=environment,
        input=f"{schema.TERMINAL_CONFIRMATION_PHRASE}\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "pre-attestation files" in result.stderr
    assert "interactive TTY" not in result.stderr
    assert not (unsealed / schema.ATTESTATION_FILENAME).exists()

    artifact = _write_valid_artifact(tmp_path / "artifact", seal_post_close=False)
    _seal_post_close(artifact)

    result = subprocess.run(
        [sys.executable, str(recorder), "--artifact-dir", str(artifact), "--operator-name", "operator"],
        cwd=root,
        env=environment,
        input=f"{schema.TERMINAL_CONFIRMATION_PHRASE}\n",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "interactive TTY is required" in result.stderr
    assert not (artifact / schema.ATTESTATION_FILENAME).exists()
