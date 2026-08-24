"""Offline schema checks for the finite biped front-leg independence gate.

This module deliberately has no Isaac Sim, Kit, or physics-backend imports.
The runtime smoke writes small JSON records that this module can validate in a
plain Python process, so missing runtime fields are evidence failures rather
than a reason to quietly weaken the gate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = 2
ARTIFACT_KIND = "rambo_physx_biped_front_leg_independence"
TASK_ID = "Isaac-RAMBO-Biped-Go2-v0"
REQUIRED_STEPS = 2
NUM_ENVS = 1
NUM_FEET = 4
NUM_LOGICAL_JOINTS = 12
FRONT_LEG_LOGICAL_JOINT_SLOTS = (0, 1, 4, 5, 8, 9)
PHYSX_CFG_FQN = "isaaclab_physx.physics.physx_manager_cfg.PhysxCfg"
PHYSX_MANAGER_FQN = "isaaclab_physx.physics.physx_manager.PhysxManager"
CONTACT_SOURCE_SCHEDULE = "contact_generator.desired_contact_state"
PERMANENT_WRENCH_OUTPUT_API = "robot.permanent_wrench_composer.out_force_b"
WRENCH_ZERO_TOLERANCE = 1.0e-6
WRENCH_NONZERO_TOLERANCE = 1.0e-4
BIPED_BASELINE_TERMINATION_FLAGS = {
    "terminate_on_undesired_foot_contact": False,
    "terminate_on_low_base_height": True,
    "terminate_on_large_orientation_error": True,
    "terminate_on_body_contact": True,
    "terminate_on_limb_contact": True,
}


class BipedIndependenceError(RuntimeError):
    """Raised when an independence artifact lacks unambiguous runtime evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BipedIndependenceError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _integer(value: Any, label: str) -> int:
    _require(not isinstance(value, bool), f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise BipedIndependenceError(f"{label} must be an integer") from error
    _require(result == value, f"{label} must be an exact integer")
    return result


def _finite(value: Any, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise BipedIndependenceError(f"{label} must be numeric") from error
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _nested_shape(value: Any, label: str) -> list[int]:
    """Return the shape of a JSON list while rejecting ragged values."""

    if not isinstance(value, list):
        return []
    _require(value, f"{label} must not be an empty list")
    child_shape = _nested_shape(value[0], f"{label}[0]")
    for index, item in enumerate(value[1:], start=1):
        _require(
            _nested_shape(item, f"{label}[{index}]") == child_shape,
            f"{label} must not be ragged",
        )
    return [len(value), *child_shape]


def _validate_numeric_tree(value: Any, label: str) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_numeric_tree(item, f"{label}[{index}]")
        return
    _finite(value, label)


def _validate_boolean_tree(value: Any, label: str) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_boolean_tree(item, f"{label}[{index}]")
        return
    _require(isinstance(value, bool), f"{label} must be bool")


def validate_tensor_record(
    record: Any,
    label: str,
    *,
    expected_shape: Sequence[int] | None = None,
    booleans: bool = False,
    require_values: bool = True,
) -> Mapping[str, Any]:
    """Validate a compact, JSON-safe tensor record emitted by the smoke."""

    value = _mapping(record, label)
    shape = value.get("shape")
    _require(isinstance(shape, list), f"{label}.shape must be a list")
    actual_shape = [_integer(item, f"{label}.shape[{index}]") for index, item in enumerate(shape)]
    _require(all(item > 0 for item in actual_shape), f"{label}.shape must contain positive dimensions")
    if expected_shape is not None:
        _require(actual_shape == list(expected_shape), f"{label}.shape is not {list(expected_shape)}")
    _require(isinstance(value.get("dtype"), str) and value["dtype"], f"{label}.dtype is required")

    if require_values:
        _require("values" in value, f"{label}.values is required")
        values = value["values"]
        _require(_nested_shape(values, f"{label}.values") == actual_shape, f"{label}.values shape disagrees")
        if booleans:
            _validate_boolean_tree(values, f"{label}.values")
        else:
            _validate_numeric_tree(values, f"{label}.values")
    else:
        for metric in ("min", "max", "abs_max"):
            _finite(value.get(metric), f"{label}.{metric}")
        _require(value.get("finite") is True, f"{label}.finite must be true")
    return value


def _vector(record: Any, label: str) -> list[float]:
    checked = validate_tensor_record(record, label, expected_shape=(NUM_ENVS, 3))
    values = checked["values"]
    return [float(item) for item in values[0]]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def validate_command_buffers(command_buffers: Any) -> None:
    """Require distinct front-foot positions and isolated FL/FR force commands."""

    buffers = _mapping(command_buffers, "command_buffers")
    fl_position = _vector(buffers.get("fl_position"), "command_buffers.fl_position")
    fr_position = _vector(buffers.get("fr_position"), "command_buffers.fr_position")
    fl_force = _vector(buffers.get("fl_force"), "command_buffers.fl_force")
    fr_force = _vector(buffers.get("fr_force"), "command_buffers.fr_force")
    _require(fl_position != fr_position, "FL and FR position command buffers must differ")
    _require(
        _norm(fl_force) <= WRENCH_ZERO_TOLERANCE,
        "FL force command buffer must be near zero",
    )
    _require(
        _norm(fr_force) > WRENCH_NONZERO_TOLERANCE,
        "FR force command buffer must be nonzero",
    )


def validate_qp_contact_trace(trace: Any) -> None:
    """Require direct proof that the six FL/FR joint selections were overridden."""

    item = _mapping(trace, "qp_contact_trace")
    _require(_integer(item.get("step"), "qp_contact_trace.step") > 0, "trace step must be positive")
    _require(
        item.get("contact_source") == CONTACT_SOURCE_SCHEDULE,
        "QP trace must name the desired-contact schedule as its source",
    )
    _require(item.get("use_actual_contact") is False, "QP trace must use the configured contact schedule")
    _require(item.get("front_leg_override_applied") is True, "QP trace lacks the front-leg override marker")
    slots = item.get("front_leg_logical_joint_slots")
    _require(slots == list(FRONT_LEG_LOGICAL_JOINT_SLOTS), "QP trace front-leg logical joint slots are invalid")

    raw = validate_tensor_record(
        item.get("source_contact_state"),
        "qp_contact_trace.source_contact_state",
        expected_shape=(NUM_ENVS, NUM_FEET),
        booleans=True,
    )["values"]
    expanded = validate_tensor_record(
        item.get("contact_state_expanded"),
        "qp_contact_trace.contact_state_expanded",
        expected_shape=(NUM_ENVS, NUM_LOGICAL_JOINTS),
        booleans=True,
    )["values"]
    _require(raw[0][0] is False and raw[0][1] is False, "raw FL/FR desired contacts must be false")
    for slot in FRONT_LEG_LOGICAL_JOINT_SLOTS:
        _require(expanded[0][slot] is True, f"front logical joint slot {slot} was not forced into contact")

    validate_tensor_record(
        item.get("contact_schedule_mode"),
        "qp_contact_trace.contact_schedule_mode",
        expected_shape=(NUM_ENVS, NUM_FEET),
    )
    validate_tensor_record(
        item.get("contact_schedule_phase"),
        "qp_contact_trace.contact_schedule_phase",
        expected_shape=(NUM_ENVS, NUM_FEET),
    )
    validate_tensor_record(item.get("grf"), "qp_contact_trace.grf", expected_shape=(NUM_ENVS, NUM_LOGICAL_JOINTS))
    validate_tensor_record(
        item.get("desired_joint_torque"),
        "qp_contact_trace.desired_joint_torque",
        expected_shape=(NUM_ENVS, NUM_LOGICAL_JOINTS),
    )
    validate_tensor_record(item.get("desired_tor"), "qp_contact_trace.desired_tor", expected_shape=(NUM_ENVS, NUM_LOGICAL_JOINTS))


def validate_permanent_wrench_output(record: Any) -> None:
    """Check the public composed wrench output on the two name-resolved feet."""

    output = _mapping(record, "permanent_wrench_output")
    _require(output.get("api") == PERMANENT_WRENCH_OUTPUT_API, "wrench output must use the public composer output")
    _require(output.get("active") is True, "permanent wrench composer is not active")
    shape = output.get("shape")
    _require(isinstance(shape, list) and len(shape) == 3, "wrench output shape is required")
    output_shape = [_integer(value, f"permanent_wrench_output.shape[{index}]") for index, value in enumerate(shape)]
    _require(
        output_shape[0] == NUM_ENVS and output_shape[1] > 0 and output_shape[2] == 3,
        "wrench output shape is invalid",
    )
    feet = _mapping(output.get("front_feet"), "permanent_wrench_output.front_feet")
    fl = _mapping(feet.get("fl"), "permanent_wrench_output.front_feet.fl")
    fr = _mapping(feet.get("fr"), "permanent_wrench_output.front_feet.fr")
    _require(fl.get("body_name") == "FL_foot", "FL wrench record resolves the wrong body")
    _require(fr.get("body_name") == "FR_foot", "FR wrench record resolves the wrong body")
    fl_id = _integer(fl.get("body_id"), "permanent_wrench_output.front_feet.fl.body_id")
    fr_id = _integer(fr.get("body_id"), "permanent_wrench_output.front_feet.fr.body_id")
    _require(fl_id != fr_id, "FL and FR physical body ids must differ")
    _require(fl_id < output_shape[1] and fr_id < output_shape[1], "front foot id is outside wrench output shape")
    fl_force = _vector(fl.get("force_body"), "permanent_wrench_output.front_feet.fl.force_body")
    fr_force = _vector(fr.get("force_body"), "permanent_wrench_output.front_feet.fr.force_body")
    _require(_norm(fl_force) <= WRENCH_ZERO_TOLERANCE, "composed FL wrench must be near zero")
    _require(_norm(fr_force) > WRENCH_NONZERO_TOLERANCE, "composed FR wrench must be nonzero")


def _validate_physx_evidence(record: Any, label: str) -> None:
    evidence = _mapping(record, label)
    _require(evidence.get("requested_cfg") == PHYSX_CFG_FQN, f"{label}.requested_cfg is not PhysxCfg")
    _require(evidence.get("actual_manager") == PHYSX_MANAGER_FQN, f"{label}.actual_manager is not PhysxManager")
    _require(evidence.get("use_newton_actuators") is False, f"{label}.use_newton_actuators must be false")


def _validate_clean_source_provenance(record: Any) -> None:
    """Bind a final M9 independence artifact to its actual clean RAMBO checkout."""

    provenance = _mapping(record, "summary.provenance")
    _require(
        provenance.get("schema_version") == 1,
        "summary.provenance schema version is invalid",
    )
    _require(
        provenance.get("collection_phase") == "pre_app_launcher_source",
        "summary.provenance was not collected before AppLauncher",
    )
    for field in ("module_path", "git_root"):
        value = provenance.get(field)
        _require(isinstance(value, str) and Path(value).is_absolute(), f"summary.provenance.{field} is invalid")
    head = provenance.get("git_head")
    _require(isinstance(head, str) and re.fullmatch(r"[0-9a-f]{40}", head) is not None, "summary.provenance.git_head is invalid")
    _require(
        provenance.get("pre_run_worktree_clean") is True,
        "summary.provenance requires a clean RAMBO worktree",
    )
    _require(
        provenance.get("pre_run_status_porcelain_v1") == [],
        "summary.provenance requires an empty pre-run porcelain status",
    )


def _validate_front_foot_body_ids(record: Any) -> dict[str, int]:
    feet = _mapping(record, "summary.front_foot_body_ids")
    expected_names = {"fl": "FL_foot", "fr": "FR_foot"}
    ids: dict[str, int] = {}
    for leg, expected_name in expected_names.items():
        foot = _mapping(feet.get(leg), f"summary.front_foot_body_ids.{leg}")
        _require(foot.get("body_name") == expected_name, f"summary front {leg} foot name is invalid")
        ids[leg] = _integer(foot.get("body_id"), f"summary.front_foot_body_ids.{leg}.body_id")
    _require(ids["fl"] != ids["fr"], "summary front foot body ids must differ")
    return ids


def validate_biped_independence_summary(summary: Any) -> None:
    """Validate the self-contained two-step artifact written by the runtime gate."""

    record = _mapping(summary, "summary")
    _require(_integer(record.get("schema_version"), "summary.schema_version") == SCHEMA_VERSION, "schema version is invalid")
    _require(record.get("artifact_kind") == ARTIFACT_KIND, "artifact kind is invalid")
    _require(record.get("task") == TASK_ID, "task is invalid")
    _require(record.get("passed") is True, "summary.passed must be true")
    _validate_clean_source_provenance(record.get("provenance"))
    _require(_integer(record.get("requested_steps"), "summary.requested_steps") == REQUIRED_STEPS, "requested steps is invalid")
    _require(_integer(record.get("steps_completed"), "summary.steps_completed") == REQUIRED_STEPS, "completed steps is invalid")
    configured = _mapping(record.get("configured_physics"), "summary.configured_physics")
    _require(configured.get("cfg") == PHYSX_CFG_FQN, "configured physics is not PhysxCfg")
    _require(configured.get("use_newton_actuators") is False, "configured Newton actuators must be disabled")
    _validate_physx_evidence(record.get("backend_before"), "summary.backend_before")
    _validate_physx_evidence(record.get("backend_after"), "summary.backend_after")
    determinism = _mapping(record.get("determinism"), "summary.determinism")
    _require(
        determinism.get("baseline_termination_flags") == BIPED_BASELINE_TERMINATION_FLAGS,
        "biped baseline termination flags were not preserved",
    )
    validate_command_buffers(record.get("command_buffers_before_steps"))
    validate_command_buffers(record.get("command_buffers_after_steps"))
    front_foot_body_ids = _validate_front_foot_body_ids(record.get("front_foot_body_ids"))
    validate_tensor_record(record.get("zero_action"), "summary.zero_action", expected_shape=(NUM_ENVS, 18))
    for value in validate_tensor_record(record.get("zero_action"), "summary.zero_action", expected_shape=(NUM_ENVS, 18))["values"][0]:
        _require(value == 0.0, "zero_action must contain only zeros")

    steps = record.get("steps")
    _require(isinstance(steps, list) and len(steps) == REQUIRED_STEPS, "summary must contain exactly two step records")
    for expected_step, step in enumerate(steps, start=1):
        step_record = _mapping(step, f"summary.steps[{expected_step - 1}]")
        _require(_integer(step_record.get("step"), f"summary.steps[{expected_step - 1}].step") == expected_step, "step order is invalid")
        validate_qp_contact_trace(step_record.get("qp_contact_trace"))
        wrench_output = _mapping(step_record.get("permanent_wrench_output"), "step permanent wrench output")
        validate_permanent_wrench_output(wrench_output)
        for leg, body_id in front_foot_body_ids.items():
            _require(
                _integer(
                    _mapping(wrench_output["front_feet"].get(leg), f"step {leg} wrench").get("body_id"),
                    f"step {leg} wrench body id",
                )
                == body_id,
                f"step {leg} wrench body id disagrees with name-resolved summary id",
            )
        validate_tensor_record(
            step_record.get("observation"),
            f"summary.steps[{expected_step - 1}].observation",
            require_values=False,
        )


__all__ = (
    "ARTIFACT_KIND",
    "BIPED_BASELINE_TERMINATION_FLAGS",
    "BipedIndependenceError",
    "CONTACT_SOURCE_SCHEDULE",
    "FRONT_LEG_LOGICAL_JOINT_SLOTS",
    "NUM_ENVS",
    "NUM_FEET",
    "NUM_LOGICAL_JOINTS",
    "PERMANENT_WRENCH_OUTPUT_API",
    "PHYSX_CFG_FQN",
    "PHYSX_MANAGER_FQN",
    "REQUIRED_STEPS",
    "SCHEMA_VERSION",
    "TASK_ID",
    "WRENCH_NONZERO_TOLERANCE",
    "WRENCH_ZERO_TOLERANCE",
    "validate_biped_independence_summary",
    "validate_command_buffers",
    "validate_permanent_wrench_output",
    "validate_qp_contact_trace",
    "validate_tensor_record",
)
