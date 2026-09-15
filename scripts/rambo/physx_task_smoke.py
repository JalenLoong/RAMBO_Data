#!/usr/bin/env python3
"""Run a finite RAMBO task smoke test with an explicit, audited PhysX backend.

The launcher deliberately accepts only ``--viz none`` and ``--viz kit``. It
never offers a Newton visualizer or backend option, and it records the requested
and active physics classes before and after the rollout.

``--m6-zero-action-contract`` is an intentionally opt-in quadruped-only
evidence run. It does not change any production task configuration: it makes a
fresh, deterministic *all-stance* smoke configuration and validates either a
100-step or a 1000-step all-zero-action rollout through the public Isaac Lab
scene APIs.  The production walking schedule is deliberately left untouched;
checkpoint replay validates that separately.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any


TASKS = {
    "quadruped": ("Isaac-RAMBO-Quadruped-Go2-v0", 405),
    "button": ("Isaac-RAMBO-Quadruped-Button-Go2-v0", 405),
}
M6_ZERO_ACTION_STEPS = (100, 1000)
M6_TRACE_FILENAME = "m6_zero_action_trace.jsonl"
M6_ALL_STANCE_FEET = ("FL", "FR", "RL", "RR")
M6_ALL_STANCE_CONTACT_MODE = 1.0
M6_BASELINE_TERMINATION_FLAGS = {
    "terminate_on_undesired_foot_contact": False,
    "terminate_on_low_base_height": True,
    "terminate_on_large_orientation_error": True,
    "terminate_on_body_contact": True,
    "terminate_on_limb_contact": True,
}
M6_EXPLOSION_LIMITS = {
    "root_position_w_m": 100.0,
    "root_linear_velocity_w_m_s": 100.0,
    "root_angular_velocity_w_rad_s": 100.0,
    "joint_position_rad": 100.0,
    "joint_velocity_rad_s": 1000.0,
    "contact_force_w_n": 1_000_000.0,
}
M6_ALL_STANCE_STABILITY_LIMITS = {
    # The native low-height safety termination is still enabled at 0.10 m.
    # This tighter independent floor prevents a barely non-terminal collapse
    # from being reported as a stationary all-stance success.
    "minimum_root_height_m": 0.15,
    "maximum_root_height_m": 1.0,
}


class TaskSmokeError(RuntimeError):
    """Raised when an opt-in task-level evidence contract is violated."""


def _build_parser() -> tuple[argparse.ArgumentParser, type]:
    """Build the CLI without importing Isaac Sim at module-import time."""

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:  # pragma: no cover - target-runtime guard.
        raise RuntimeError("physx_task_smoke.py requires the RAMBO Isaac Lab runtime") from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--disable-events", action="store_true", help="Use a deterministic no-domain-randomization smoke scene."
    )
    parser.add_argument(
        "--m6-zero-action-contract",
        action="store_true",
        help=(
            "Opt-in quadruped-only M6 evidence run. Requires --steps 100 or 1000; "
            "forces an event-free deterministic smoke configuration."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser, AppLauncher


def _has_option(name: str, argv: list[str] | None = None) -> bool:
    arguments = sys.argv[1:] if argv is None else argv
    return any(argument == name or argument.startswith(f"{name}=") for argument in arguments)


def _finite_tree(value: Any, label: str) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{label} contains NaN or Inf")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    # Do not import torch in the static path.  This handles diagnostic tensors
    # only after a target-runtime failure has already occurred.
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        try:
            return value.detach().cpu().tolist()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _validate_m6_zero_action_request(task_alias: str, steps: int) -> None:
    """Reject ambiguous M6 invocations before a Kit application is launched."""

    if task_alias != "quadruped":
        raise TaskSmokeError("--m6-zero-action-contract only permits --task quadruped")
    if steps not in M6_ZERO_ACTION_STEPS:
        supported = ", ".join(str(value) for value in M6_ZERO_ACTION_STEPS)
        raise TaskSmokeError(
            f"--m6-zero-action-contract requires --steps to be one of {supported}, got {steps}"
        )


def _public_torch_view(value: Any, label: str, torch: Any) -> Any:
    """Read a documented Isaac Lab ProxyArray only through its ``.torch`` view."""

    tensor = getattr(value, "torch", None)
    if not isinstance(tensor, torch.Tensor):
        raise TaskSmokeError(
            f"Required public Isaac Lab field {label} does not expose a torch.Tensor .torch view"
        )
    if tensor.numel() == 0:
        raise TaskSmokeError(f"Required public Isaac Lab field {label} is empty")
    if not bool(torch.isfinite(tensor).all()):
        raise TaskSmokeError(f"Required public Isaac Lab field {label} contains NaN or Inf")
    return tensor


def _tensor_sha256(tensor: Any, label: str, torch: Any) -> str:
    """Hash a finite public tensor with shape and dtype domain separation."""

    if not isinstance(tensor, torch.Tensor):
        raise TaskSmokeError(f"{label} must be a torch.Tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise TaskSmokeError(f"{label} contains NaN or Inf")
    detached = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": str(detached.dtype), "shape": list(detached.shape)}).encode("utf-8"))
    digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _tensor_metrics(tensor: Any, label: str, torch: Any, *, sample_elements: int = 12) -> dict[str, Any]:
    """Return compact finite-state evidence without retaining the full rollout."""

    if not isinstance(tensor, torch.Tensor):
        raise TaskSmokeError(f"{label} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.numel() == 0:
        raise TaskSmokeError(f"{label} must not be empty")
    if not bool(torch.isfinite(tensor).all()):
        raise TaskSmokeError(f"{label} contains NaN or Inf")
    detached = tensor.detach()
    flat = detached.reshape(-1)
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "min": float(detached.min().cpu()),
        "max": float(detached.max().cpu()),
        "mean": float(detached.mean().cpu()),
        "abs_max": float(detached.abs().max().cpu()),
        "sample": flat[:sample_elements].cpu().tolist(),
        "sha256": _tensor_sha256(detached, label, torch),
    }


def _int_list(value: Any, label: str, torch: Any) -> list[int]:
    """Normalize public Isaac Lab index collections for JSON evidence."""

    if isinstance(value, torch.Tensor):
        raw_values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raise TaskSmokeError(f"{label} must be a tensor, list, or tuple")
    try:
        result = [int(item) for item in raw_values]
    except (TypeError, ValueError) as exc:
        raise TaskSmokeError(f"{label} contains a non-integer index") from exc
    if not result:
        raise TaskSmokeError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise TaskSmokeError(f"{label} contains duplicate indices: {result}")
    return result


def _names_for_ids(names: Any, ids: list[int], label: str) -> list[str]:
    if not isinstance(names, (list, tuple)) or not all(isinstance(name, str) for name in names):
        raise TaskSmokeError(f"{label} names are unavailable from the public articulation data")
    if any(index < 0 or index >= len(names) for index in ids):
        raise TaskSmokeError(f"{label} contains an out-of-range index: ids={ids}, names={list(names)}")
    return [str(names[index]) for index in ids]


def _public_scene_objects(base_env: Any) -> tuple[Any, Any]:
    """Resolve the robot and contact sensor without touching private task fields."""

    scene = getattr(base_env, "scene", None)
    articulations = getattr(scene, "articulations", None)
    sensors = getattr(scene, "sensors", None)
    if not isinstance(articulations, Mapping):
        raise TaskSmokeError("Required public base_env.scene.articulations mapping is unavailable")
    if not isinstance(sensors, Mapping):
        raise TaskSmokeError("Required public base_env.scene.sensors mapping is unavailable")
    robot = articulations.get("robot")
    contact_sensor = sensors.get("contact_sensor")
    if robot is None:
        raise TaskSmokeError("Required public scene.articulations['robot'] is unavailable")
    if contact_sensor is None:
        raise TaskSmokeError("Required public scene.sensors['contact_sensor'] is unavailable")
    if getattr(robot, "data", None) is None:
        raise TaskSmokeError("Required public Articulation.data is unavailable")
    if getattr(contact_sensor, "data", None) is None:
        raise TaskSmokeError("Required public ContactSensor.data is unavailable")
    return robot, contact_sensor


def _resolve_m6_scene_contract(
    base_env: Any,
    *,
    torch: Any,
    body_order: tuple[str, ...],
    joint_order: tuple[str, ...],
    resolve_go2_indices: Any,
    ordered_jacobians: Any,
) -> dict[str, Any]:
    """Resolve and prove the public articulation/contact/Jacobian ordering once.

    ``ordered_jacobians`` is mandatory evidence, not an optional diagnostic.
    If Isaac Lab changes that public data contract, the M6 run fails rather
    than silently falling back to RAMBO's private ``_robot`` fields.
    """

    robot, contact_sensor = _public_scene_objects(base_env)
    contact_generator = getattr(base_env, "contact_generator", None)
    if contact_generator is None:
        raise TaskSmokeError("M6 zero-action contract requires RAMBO contact_generator")
    data = robot.data
    indices = resolve_go2_indices(robot)
    body_ids = _int_list(getattr(indices, "body_ids", None), "resolved articulation body ids", torch)
    joint_ids = _int_list(getattr(indices, "joint_ids", None), "resolved articulation joint ids", torch)
    foot_body_ids = _int_list(getattr(indices, "foot_body_ids", None), "resolved articulation foot ids", torch)
    head_body_ids_value = getattr(indices, "head_body_ids", None)
    if isinstance(head_body_ids_value, torch.Tensor) and head_body_ids_value.numel() == 0:
        head_body_ids: list[int] = []
    else:
        head_body_ids = _int_list(head_body_ids_value, "resolved articulation head ids", torch)

    if len(body_ids) != len(body_order) or len(joint_ids) != len(joint_order):
        raise TaskSmokeError(
            "Resolved Go2 logical ordering has unexpected dimensions: "
            f"bodies={len(body_ids)}/{len(body_order)}, joints={len(joint_ids)}/{len(joint_order)}"
        )
    body_names = _names_for_ids(getattr(data, "body_names", None), body_ids, "articulation body ids")
    joint_names = _names_for_ids(getattr(data, "joint_names", None), joint_ids, "articulation joint ids")
    if tuple(body_names) != body_order:
        raise TaskSmokeError(
            f"Resolved articulation body names are not RAMBO logical order: {body_names}"
        )
    if tuple(joint_names) != joint_order:
        raise TaskSmokeError(
            f"Resolved articulation joint names are not RAMBO logical order: {joint_names}"
        )

    contact_ids_value, contact_names_value = contact_sensor.find_sensors(list(body_order), preserve_order=True)
    contact_ids = _int_list(contact_ids_value, "resolved contact-sensor body ids", torch)
    contact_names = [str(name) for name in contact_names_value]
    if len(contact_ids) != len(body_order) or tuple(contact_names) != body_order:
        raise TaskSmokeError(
            "Resolved contact-sensor body names are not RAMBO logical order: "
            f"ids={contact_ids}, names={contact_names}"
        )
    contact_head_ids_value, contact_head_names_value = contact_sensor.find_sensors("Head_.*")
    contact_head_ids = _int_list(contact_head_ids_value, "resolved contact-sensor head ids", torch)
    contact_head_names = [str(name) for name in contact_head_names_value]
    if not contact_head_names:
        raise TaskSmokeError("Contact sensor did not resolve required Head_* bodies")

    raw_jacobians = _public_torch_view(
        getattr(data, "body_com_jacobian_w", None), "robot.data.body_com_jacobian_w", torch
    )
    if raw_jacobians.ndim != 4:
        raise TaskSmokeError(
            "Public robot.data.body_com_jacobian_w must have rank 4, got "
            f"{tuple(raw_jacobians.shape)}"
        )
    ordered = ordered_jacobians(robot, indices)
    jacobian_metrics = _tensor_metrics(ordered, "ordered_jacobians", torch)
    if ordered.ndim != 4 or ordered.shape[1] != len(body_order):
        raise TaskSmokeError(
            "RAMBO ordered_jacobians produced an unexpected logical body dimension: "
            f"{tuple(ordered.shape)}"
        )

    return {
        "robot": robot,
        "contact_sensor": contact_sensor,
        "contact_generator": contact_generator,
        "joint_ids": getattr(indices, "joint_ids"),
        "contact_ids": contact_ids,
        "summary": {
            "access_contract": {
                "articulation": "base_env.scene.articulations['robot'].data.<field>.torch",
                "contact_sensor": "base_env.scene.sensors['contact_sensor'].data.net_forces_w_history.torch",
            },
            "articulation": {
                "body_ids": body_ids,
                "body_names": body_names,
                "joint_ids": joint_ids,
                "joint_names": joint_names,
                "foot_body_ids": foot_body_ids,
                "foot_body_names": _names_for_ids(
                    getattr(data, "body_names", None), foot_body_ids, "articulation foot ids"
                ),
                "head_body_ids": head_body_ids,
                "head_body_names": (
                    _names_for_ids(getattr(data, "body_names", None), head_body_ids, "articulation head ids")
                    if head_body_ids
                    else []
                ),
            },
            "contact_sensor": {
                "sensor_key": "contact_sensor",
                "body_ids": contact_ids,
                "body_names": contact_names,
                "head_ids": contact_head_ids,
                "head_names": contact_head_names,
            },
            "jacobian_ordering": {
                "helper": "rambo.utils.articulation.ordered_jacobians",
                "logical_body_order": list(body_order),
                "resolved_body_ids": body_ids,
                "raw_public_field": "robot.data.body_com_jacobian_w.torch",
                "raw_shape": list(raw_jacobians.shape),
                "raw_omits_root_row": bool(
                    raw_jacobians.shape[1] == len(getattr(data, "body_names", ())) - 1
                ),
                "ordered_shape": list(ordered.shape),
                "ordered_metrics": jacobian_metrics,
            },
        },
    }


def _require_shape(tensor: Any, expected: tuple[int, ...], label: str) -> None:
    if tuple(tensor.shape) != expected:
        raise TaskSmokeError(f"{label} has shape {tuple(tensor.shape)}, expected {expected}")


def _m6_all_stance_scheduler_record(contact_generator: Any, *, num_envs: int, torch: Any) -> dict[str, Any]:
    """Fail closed unless the effective M6-only schedule remains all stance."""

    desired_contact_mode = getattr(contact_generator, "desired_contact_mode", None)
    desired_contact_state = getattr(contact_generator, "desired_contact_state", None)
    if not isinstance(desired_contact_mode, torch.Tensor):
        raise TaskSmokeError("M6 contact_generator.desired_contact_mode must be a torch.Tensor")
    if not isinstance(desired_contact_state, torch.Tensor):
        raise TaskSmokeError("M6 contact_generator.desired_contact_state must be a torch.Tensor")
    _require_shape(desired_contact_mode, (num_envs, len(M6_ALL_STANCE_FEET)), "contact_generator.desired_contact_mode")
    _require_shape(desired_contact_state, (num_envs, len(M6_ALL_STANCE_FEET)), "contact_generator.desired_contact_state")
    if desired_contact_state.dtype != torch.bool:
        raise TaskSmokeError(
            "M6 contact_generator.desired_contact_state must have dtype torch.bool, got "
            f"{desired_contact_state.dtype}"
        )
    if not bool(torch.all(desired_contact_mode == M6_ALL_STANCE_CONTACT_MODE).item()):
        raise TaskSmokeError("M6 all-stance scheduler emitted a non-stance contact mode")
    if not bool(torch.all(desired_contact_state).item()):
        raise TaskSmokeError("M6 all-stance scheduler emitted a non-contact foot state")
    return {
        "foot_order": list(M6_ALL_STANCE_FEET),
        "desired_contact_mode": desired_contact_mode.detach().cpu().tolist(),
        "desired_contact_state": desired_contact_state.detach().cpu().tolist(),
        "all_four_feet_stance": True,
        "all_four_feet_contact": True,
    }


def _collect_m6_public_state(
    scene_contract: dict[str, Any], *, step: int, num_envs: int, torch: Any
) -> dict[str, Any]:
    """Collect finite root/joint/contact state using only public ``.torch`` data."""

    data = scene_contract["robot"].data
    root_position = _public_torch_view(
        getattr(data, "root_link_pos_w", None), "robot.data.root_link_pos_w", torch
    )
    root_quaternion = _public_torch_view(
        getattr(data, "root_link_quat_w", None), "robot.data.root_link_quat_w", torch
    )
    root_linear_velocity = _public_torch_view(
        getattr(data, "root_link_lin_vel_w", None), "robot.data.root_link_lin_vel_w", torch
    )
    root_angular_velocity = _public_torch_view(
        getattr(data, "root_link_ang_vel_w", None), "robot.data.root_link_ang_vel_w", torch
    )
    joint_position_all = _public_torch_view(
        getattr(data, "joint_pos", None), "robot.data.joint_pos", torch
    )
    joint_velocity_all = _public_torch_view(
        getattr(data, "joint_vel", None), "robot.data.joint_vel", torch
    )
    contact_history_all = _public_torch_view(
        getattr(scene_contract["contact_sensor"].data, "net_forces_w_history", None),
        "contact_sensor.data.net_forces_w_history",
        torch,
    )

    _require_shape(root_position, (num_envs, 3), "robot.data.root_link_pos_w")
    _require_shape(root_quaternion, (num_envs, 4), "robot.data.root_link_quat_w")
    _require_shape(root_linear_velocity, (num_envs, 3), "robot.data.root_link_lin_vel_w")
    _require_shape(root_angular_velocity, (num_envs, 3), "robot.data.root_link_ang_vel_w")
    if joint_position_all.ndim != 2 or joint_position_all.shape[0] != num_envs:
        raise TaskSmokeError(
            "robot.data.joint_pos must have shape (num_envs, joint_count), got "
            f"{tuple(joint_position_all.shape)}"
        )
    if tuple(joint_velocity_all.shape) != tuple(joint_position_all.shape):
        raise TaskSmokeError(
            "robot.data.joint_vel must match robot.data.joint_pos shape, got "
            f"{tuple(joint_velocity_all.shape)} vs {tuple(joint_position_all.shape)}"
        )
    if contact_history_all.ndim != 4 or contact_history_all.shape[0] != num_envs or contact_history_all.shape[-1] != 3:
        raise TaskSmokeError(
            "contact_sensor.data.net_forces_w_history must have shape "
            f"(num_envs, history, bodies, 3), got {tuple(contact_history_all.shape)}"
        )

    joint_ids = scene_contract["joint_ids"]
    if not isinstance(joint_ids, torch.Tensor):
        raise TaskSmokeError("Resolved articulation joint ids are no longer a torch.Tensor")
    joint_ids = joint_ids.to(device=joint_position_all.device, dtype=torch.long)
    joint_position = joint_position_all.index_select(1, joint_ids)
    joint_velocity = joint_velocity_all.index_select(1, joint_ids)
    contact_ids = torch.tensor(
        scene_contract["contact_ids"], dtype=torch.long, device=contact_history_all.device
    )
    if int(contact_ids.max().detach().cpu()) >= contact_history_all.shape[2]:
        raise TaskSmokeError(
            "Resolved contact-sensor ids exceed net_forces_w_history body dimension: "
            f"ids={scene_contract['contact_ids']}, shape={tuple(contact_history_all.shape)}"
        )
    contact_history = contact_history_all.index_select(2, contact_ids)
    contact_force_norm = torch.linalg.vector_norm(contact_history, dim=-1)
    root_height = root_position[:, 2]
    minimum_root_height = float(root_height.min().detach().cpu())
    maximum_root_height = float(root_height.max().detach().cpu())
    if minimum_root_height < M6_ALL_STANCE_STABILITY_LIMITS["minimum_root_height_m"]:
        raise TaskSmokeError(
            "M6 all-stance root height fell below the independent stationary stability floor: "
            f"{minimum_root_height:.6g} < {M6_ALL_STANCE_STABILITY_LIMITS['minimum_root_height_m']:.6g}"
        )
    if maximum_root_height > M6_ALL_STANCE_STABILITY_LIMITS["maximum_root_height_m"]:
        raise TaskSmokeError(
            "M6 all-stance root height exceeded the independent stationary stability ceiling: "
            f"{maximum_root_height:.6g} > {M6_ALL_STANCE_STABILITY_LIMITS['maximum_root_height_m']:.6g}"
        )

    scheduler_record = _m6_all_stance_scheduler_record(
        scene_contract["contact_generator"], num_envs=num_envs, torch=torch
    )

    metrics = {
        "root_position_w_m": _tensor_metrics(root_position, "root_position_w_m", torch),
        "root_height_m": _tensor_metrics(root_height, "root_height_m", torch),
        "root_quaternion_xyzw": _tensor_metrics(root_quaternion, "root_quaternion_xyzw", torch),
        "root_linear_velocity_w_m_s": _tensor_metrics(
            root_linear_velocity, "root_linear_velocity_w_m_s", torch
        ),
        "root_angular_velocity_w_rad_s": _tensor_metrics(
            root_angular_velocity, "root_angular_velocity_w_rad_s", torch
        ),
        "joint_position_rad": _tensor_metrics(joint_position, "joint_position_rad", torch),
        "joint_velocity_rad_s": _tensor_metrics(joint_velocity, "joint_velocity_rad_s", torch),
        "contact_force_w_n": _tensor_metrics(contact_history, "contact_force_w_n", torch),
        "contact_force_norm_n": _tensor_metrics(contact_force_norm, "contact_force_norm_n", torch),
    }
    for name, limit in M6_EXPLOSION_LIMITS.items():
        if metrics[name]["abs_max"] > limit:
            raise TaskSmokeError(
                f"M6 zero-action explosion guard {name}={metrics[name]['abs_max']:.6g} exceeds {limit:.6g} at step {step}"
            )

    return {
        "step": step,
        "state_finite": True,
        "metrics": metrics,
        "sample": {
            "root_position_w_m_env0": root_position[0].detach().cpu().tolist(),
            "root_height_m_env0": float(root_height[0].detach().cpu()),
            "root_quaternion_xyzw_env0": root_quaternion[0].detach().cpu().tolist(),
            "root_linear_velocity_w_m_s_env0": root_linear_velocity[0].detach().cpu().tolist(),
            "root_angular_velocity_w_rad_s_env0": root_angular_velocity[0].detach().cpu().tolist(),
            "joint_position_rad_env0": joint_position[0].detach().cpu().tolist(),
            "joint_velocity_rad_s_env0": joint_velocity[0].detach().cpu().tolist(),
            "contact_force_w_last_history_env0": contact_history[0, -1].detach().cpu().tolist(),
        },
        "m6_all_stance_scheduler": scheduler_record,
    }


def _update_m6_state_progress(progress: dict[str, Any], record: dict[str, Any]) -> None:
    """Track summary extrema while the JSONL trace retains per-step samples."""

    progress["records"] += 1
    if "post_reset_sample" not in progress:
        progress["post_reset_sample"] = record["sample"]
    progress["last_sample"] = record["sample"]
    extrema = progress.setdefault("extrema", {})
    for name, metric in record["metrics"].items():
        if name not in extrema:
            extrema[name] = {
                "min": metric["min"],
                "max": metric["max"],
                "abs_max": metric["abs_max"],
            }
            continue
        extrema[name]["min"] = min(float(extrema[name]["min"]), float(metric["min"]))
        extrema[name]["max"] = max(float(extrema[name]["max"]), float(metric["max"]))
        extrema[name]["abs_max"] = max(float(extrema[name]["abs_max"]), float(metric["abs_max"]))


def _zero_action_dones(transition: Any, *, num_envs: int, torch: Any) -> dict[str, Any]:
    """Fail closed if the standard Gymnasium transition reports a reset trigger."""

    if not isinstance(transition, (tuple, list)) or len(transition) != 5:
        raise TaskSmokeError(
            "M6 zero-action contract requires a five-value Gymnasium transition "
            "(observations, rewards, terminated, truncated, info)"
        )
    terminated, truncated = transition[2], transition[3]
    for label, tensor in (("terminated", terminated), ("truncated", truncated)):
        if not isinstance(tensor, torch.Tensor):
            raise TaskSmokeError(f"M6 {label} must be a torch.Tensor")
        _require_shape(tensor, (num_envs,), f"M6 {label}")
        if bool(torch.any(tensor).item()):
            raise TaskSmokeError(f"M6 zero-action contract observed {label}=True")
    return {
        "terminated": terminated.detach().cpu().to(dtype=torch.bool).tolist(),
        "truncated": truncated.detach().cpu().to(dtype=torch.bool).tolist(),
        "all_dones_false": True,
    }


def _m6_episode_counter(base_env: Any, *, expected_step: int, num_envs: int, torch: Any) -> list[int]:
    """Prove DirectRLEnv did not conceal a reset behind its transition output."""

    episode_length = getattr(base_env, "episode_length_buf", None)
    if not isinstance(episode_length, torch.Tensor):
        raise TaskSmokeError("M6 base_env.episode_length_buf must be a torch.Tensor")
    _require_shape(episode_length, (num_envs,), "M6 episode_length_buf")
    values = [int(value) for value in episode_length.detach().cpu().tolist()]
    if values != [expected_step] * num_envs:
        raise TaskSmokeError(
            "M6 episode_length_buf did not advance monotonically without a reset: "
            f"expected {[expected_step] * num_envs}, got {values}"
        )
    return values


class _StateTraceWriter:
    """Stream M6 state evidence without retaining a 1000-step trace in RAM."""

    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("x", encoding="utf-8")
        self.records = 0

    def write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(_jsonable(record), separators=(",", ":"), sort_keys=True) + "\n")
        self.records += 1

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def _configure_m6_zero_action_scene(env_cfg: Any, *, seed: int, steps: int) -> dict[str, Any]:
    """Create an event-free deterministic evidence configuration after PhysX selection.

    This mutates only the fresh config passed to this smoke launcher. It does
    not alter the task class or its normal training/playback defaults.
    """

    _validate_m6_zero_action_request("quadruped", steps)
    env_cfg.seed = seed
    env_cfg.events = None
    required_false = (
        "randomize_initial_state",
        "obs_noise",
        "randomize_episode_progress",
        "enable_sampled_velocity_commands",
        "enable_sampled_pos_commands",
        "enable_sampled_force_commands",
        "enable_rgb_camera",
    )
    for attribute in required_false:
        if not hasattr(env_cfg, attribute):
            raise TaskSmokeError(f"M6 deterministic scene requires env_cfg.{attribute}")
        setattr(env_cfg, attribute, False)
        if getattr(env_cfg, attribute) is not False:
            raise TaskSmokeError(f"M6 deterministic scene requires env_cfg.{attribute}=False")
    for attribute in ("velocity_debug_vis", "pos_debug_vis", "force_debug_vis"):
        if hasattr(env_cfg, attribute):
            setattr(env_cfg, attribute, False)
    for attribute in ("action_noise_model", "observation_noise_model"):
        if hasattr(env_cfg, attribute):
            setattr(env_cfg, attribute, None)
            if getattr(env_cfg, attribute) is not None:
                raise TaskSmokeError(f"M6 deterministic scene requires env_cfg.{attribute}=None")
    if env_cfg.events is not None:
        raise TaskSmokeError("M6 deterministic scene requires env_cfg.events=None")

    try:
        step_dt_s = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TaskSmokeError("M6 zero-action contract cannot determine policy-step duration") from exc
    if not math.isfinite(step_dt_s) or step_dt_s <= 0.0:
        raise TaskSmokeError(f"M6 zero-action contract needs a positive policy-step duration, got {step_dt_s}")
    required_duration_s = steps * step_dt_s + 1.0
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), required_duration_s)

    action_scale = getattr(env_cfg, "action_scale", None)
    if not isinstance(action_scale, (list, tuple)) or len(action_scale) != 18:
        raise TaskSmokeError("M6 deterministic scene requires the unchanged 18-value action_scale contract")
    try:
        action_scale_values = [float(value) for value in action_scale]
    except (TypeError, ValueError) as exc:
        raise TaskSmokeError("M6 deterministic scene action_scale must contain numeric values") from exc
    if action_scale_values != [5.0] * 6 + [0.15] * 12:
        raise TaskSmokeError(
            "M6 deterministic scene action_scale differs from the RAMBO 6D-base/12D-joint contract"
        )

    termination_flags: dict[str, bool] = {}
    for attribute, expected in M6_BASELINE_TERMINATION_FLAGS.items():
        if not hasattr(env_cfg, attribute) or getattr(env_cfg, attribute) is not expected:
            raise TaskSmokeError(
                f"M6 deterministic scene requires the baseline safety setting env_cfg.{attribute}={expected}"
            )
        termination_flags[attribute] = expected

    contact_generator_config = getattr(env_cfg, "contact_generator_config", None)
    if not isinstance(contact_generator_config, dict):
        raise TaskSmokeError("M6 deterministic scene requires dict env_cfg.contact_generator_config")
    copied_contact_generator_config = copy.deepcopy(contact_generator_config)
    copied_contact_generator_config["contact_generator_debug_vis"] = False
    source_contact_sequence = copied_contact_generator_config.get("contact_sequence")
    if not isinstance(source_contact_sequence, dict) or not source_contact_sequence:
        raise TaskSmokeError("M6 deterministic scene requires a non-empty contact sequence")
    if tuple(source_contact_sequence) != M6_ALL_STANCE_FEET:
        raise TaskSmokeError(
            "M6 deterministic scene requires the canonical FL/FR/RL/RR contact sequence order, got "
            f"{tuple(source_contact_sequence)}"
        )
    for foot_name, sequence in source_contact_sequence.items():
        if not isinstance(sequence, list) or not sequence:
            raise TaskSmokeError(f"M6 contact sequence for {foot_name!r} must be a non-empty list")
        if any(not isinstance(segment, list) or len(segment) < 5 for segment in sequence):
            raise TaskSmokeError(f"M6 contact sequence for {foot_name!r} has an invalid segment")
    source_contact_sequence = copy.deepcopy(source_contact_sequence)
    source_contact_sequence_sha256 = hashlib.sha256(
        json.dumps(source_contact_sequence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    # A policy action of zero is not the walking policy's residual action: the
    # production walking schedule intentionally changes support after one
    # second.  The M6.3 diagnostic instead exercises a reproducible stationary
    # four-foot-support state.  It is a fresh smoke-only config and never
    # changes QPEnvCfg, production gait replay, safety termination, or PhysX.
    effective_contact_sequence = {
        foot_name: [["stance", required_duration_s, 0.0, 0.0, 0.0]]
        for foot_name in M6_ALL_STANCE_FEET
    }
    copied_contact_generator_config["contact_sequence"] = effective_contact_sequence
    env_cfg.contact_generator_config = copied_contact_generator_config
    if contact_generator_config["contact_sequence"] != source_contact_sequence:
        raise TaskSmokeError("M6 deterministic scene mutated the production contact schedule")

    for config_attribute, debug_key in (
        ("joint_position_controller_config", "joint_position_controller_debug_vis"),
        ("qp_torque_optimizer_config", "qp_debug_vis"),
    ):
        config = getattr(env_cfg, config_attribute, None)
        if isinstance(config, dict):
            copied_config = copy.deepcopy(config)
            copied_config[debug_key] = False
            setattr(env_cfg, config_attribute, copied_config)

    return {
        "events": None,
        "seed": seed,
        "randomize_initial_state": False,
        "obs_noise": False,
        "randomize_episode_progress": False,
        "enable_sampled_velocity_commands": False,
        "enable_sampled_pos_commands": False,
        "enable_sampled_force_commands": False,
        "enable_rgb_camera": False,
        "action_scale": action_scale_values,
        "safety_termination_flags": termination_flags,
        "policy_step_dt_s": step_dt_s,
        "episode_length_s": float(env_cfg.episode_length_s),
        "contact_schedule_profile": "m6_diagnostic_all_stance",
        "production_contact_schedule_mutated": False,
        "production_contact_sequence_sha256": source_contact_sequence_sha256,
        "production_contact_sequence": source_contact_sequence,
        "effective_contact_sequence": effective_contact_sequence,
        "contact_sequence_total_duration_s": {
            str(foot_name): sum(float(segment[1]) for segment in sequence)
            for foot_name, sequence in effective_contact_sequence.items()
        },
        "debug_visualization_disabled": {
            "contact_generator": bool(
                not env_cfg.contact_generator_config["contact_generator_debug_vis"]
            ),
            "joint_position_controller": bool(
                not getattr(env_cfg, "joint_position_controller_config", {}).get(
                    "joint_position_controller_debug_vis", False
                )
            ),
            "qp_torque_optimizer": bool(
                not getattr(env_cfg, "qp_torque_optimizer_config", {}).get("qp_debug_vis", False)
            ),
        },
    }


def _seed_m6_runtime(seed: int, torch: Any) -> None:
    """Seed only the opt-in evidence process before constructing its environment."""

    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_summary_and_checksums(
    output_dir: Path, summary: dict[str, Any], evidence_paths: tuple[Path, ...] = ()
) -> None:
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths = (summary_path, *[path for path in evidence_paths if path.is_file()])
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{_sha256_path(path)}  {path.name}\n" for path in paths), encoding="utf-8"
    )
    print(f"SUMMARY_PATH={summary_path}", flush=True)


def main() -> int:
    parser, app_launcher_type = _build_parser()
    args_cli = parser.parse_args()
    if args_cli.steps <= 0:
        parser.error("--steps must be positive")
    if args_cli.m6_zero_action_contract:
        try:
            _validate_m6_zero_action_request(args_cli.task, args_cli.steps)
        except TaskSmokeError as error:
            parser.error(str(error))
    if _has_option("--headless"):
        parser.error("--headless is forbidden; use --viz none")
    if not (_has_option("--viz") or _has_option("--visualizer")):
        parser.error("--viz must be explicitly set to none or kit")
    if args_cli.visualizer is None or args_cli.visualizer == ["none"]:
        visualizer_selection = ["none"]
    elif args_cli.visualizer == ["kit"]:
        visualizer_selection = ["kit"]
    else:
        parser.error("only --viz none and --viz kit are permitted for RAMBO PhysX execution")
    if args_cli.experience:
        parser.error("--experience is forbidden for RAMBO PhysX execution")
    if args_cli.kit_args:
        parser.error("--kit_args is forbidden for RAMBO PhysX execution")
    if args_cli.livestream not in (-1, 0):
        parser.error("--livestream may only be omitted or set to 0 for RAMBO PhysX execution")

    args_cli.fast_shutdown = True
    args_cli.livestream = 0
    output_dir = args_cli.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "passed": False,
        "task_alias": args_cli.task,
        "task": TASKS[args_cli.task][0],
        "requested_steps": args_cli.steps,
        "seed": args_cli.seed,
        "viz": visualizer_selection,
        "eula_acceptance": "explicit_user_consent",
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
    }
    m6_contract: dict[str, Any] | None = None
    if args_cli.m6_zero_action_contract:
        m6_contract = {
            "requested": True,
            "task_alias": "quadruped",
            "approved_step_counts": list(M6_ZERO_ACTION_STEPS),
            "requested_steps": args_cli.steps,
            "zero_actions_only": True,
            "initial_reset_calls": 1,
            "resets_after_initial_verified": False,
            "dones_false_steps": 0,
            "explosion_guard_limits": M6_EXPLOSION_LIMITS,
            "all_stance_stability_limits": M6_ALL_STANCE_STABILITY_LIMITS,
            "passed": False,
        }
        summary["m6_zero_action_contract"] = m6_contract

    simulation_app = None
    env = None
    base_env_for_m6 = None
    torch = None
    m6_trace_writer: _StateTraceWriter | None = None
    m6_state_progress: dict[str, Any] | None = None
    exit_code = 0
    try:
        app_launcher = app_launcher_type(args_cli)
        simulation_app = app_launcher.app
        from rambo.torch_runtime import ensure_cuda_linalg_loaded

        ensure_cuda_linalg_loaded()
        import gymnasium as gym
        import rambo
        import torch as torch_module
        from rambo.utils import assert_physx_environment, configure_physx, parse_env_cfg
        from rambo.utils.articulation import (
            GO2_BODY_ORDER,
            GO2_JOINT_ORDER,
            ordered_jacobians,
            resolve_go2_indices,
        )

        torch = torch_module
        task, expected_observation_dim = TASKS[args_cli.task]
        if not rambo.register_tasks():
            raise RuntimeError("RAMBO task registration requires an active Isaac Sim Kit application")
        env_cfg = parse_env_cfg(task, device=args_cli.device or "cuda:0", num_envs=1)
        # Do not rely solely on the registry contract: this concrete smoke
        # entry point must itself request PhysX before Gym constructs the env.
        configure_physx(env_cfg)
        if args_cli.m6_zero_action_contract:
            assert m6_contract is not None
            m6_contract["deterministic_scene"] = _configure_m6_zero_action_scene(
                env_cfg, seed=args_cli.seed, steps=args_cli.steps
            )
            _seed_m6_runtime(args_cli.seed, torch)
            m6_contract["host_rng_seeded"] = True
            if getattr(env_cfg.sim, "use_newton_actuators", None) is not False:
                raise TaskSmokeError("M6 zero-action contract requires use_newton_actuators=False")
            m6_contract["configured_use_newton_actuators"] = False
        else:
            env_cfg.seed = args_cli.seed
            if args_cli.disable_events:
                env_cfg.events = None
        env = gym.make(task, cfg=env_cfg)
        if m6_contract is not None:
            base_env_for_m6 = env.unwrapped
            # This hook changes no task behavior; it makes a native safety
            # termination diagnosable should the fail-closed contract trip.
            base_env_for_m6._record_termination_diagnostics = True
            m6_contract["native_termination_diagnostics_enabled"] = True
        summary["backend_before"] = assert_physx_environment(env)
        if m6_contract is not None:
            m6_contract["actual_physx_manager_before"] = summary["backend_before"]["actual_manager"]
            m6_contract["actual_physx_manager_before_evidence"] = summary["backend_before"]
        summary["runtime"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        summary["config"] = {
            "dt_s": float(env.unwrapped.cfg.sim.dt),
            "decimation": int(env.unwrapped.cfg.decimation),
            "use_newton_actuators": bool(env.unwrapped.cfg.sim.use_newton_actuators),
        }
        if m6_contract is not None:
            if getattr(env.unwrapped.cfg, "events", object()) is not None:
                raise TaskSmokeError("M6 zero-action contract unexpectedly instantiated events")
            if getattr(env.unwrapped.cfg.sim, "use_newton_actuators", None) is not False:
                raise TaskSmokeError("M6 zero-action contract lost use_newton_actuators=False")

        observations, _ = env.reset(seed=args_cli.seed)
        _finite_tree(observations, "reset_observations")
        policy_observations = observations["policy"]
        if tuple(policy_observations.shape) != (1, expected_observation_dim):
            raise RuntimeError(
                f"Unexpected policy observation shape: {tuple(policy_observations.shape)}, "
                f"expected (1, {expected_observation_dim})"
            )
        if tuple(env.action_space.shape) != (1, 18):
            raise RuntimeError(f"Unexpected action-space shape: {tuple(env.action_space.shape)}")

        zero_action = None
        scene_contract = None
        if m6_contract is not None:
            scene_contract = _resolve_m6_scene_contract(
                env.unwrapped,
                torch=torch,
                body_order=GO2_BODY_ORDER,
                joint_order=GO2_JOINT_ORDER,
                resolve_go2_indices=resolve_go2_indices,
                ordered_jacobians=ordered_jacobians,
            )
            m6_contract["resolved_scene"] = scene_contract["summary"]
            m6_trace_writer = _StateTraceWriter(output_dir / M6_TRACE_FILENAME)
            m6_state_progress = {"records": 0, "trace_file": M6_TRACE_FILENAME}
            post_reset_state = _collect_m6_public_state(
                scene_contract, step=0, num_envs=1, torch=torch
            )
            post_reset_state["phase"] = "post_reset"
            post_reset_state["episode_length_buf"] = _m6_episode_counter(
                env.unwrapped, expected_step=0, num_envs=1, torch=torch
            )
            m6_trace_writer.write(post_reset_state)
            _update_m6_state_progress(m6_state_progress, post_reset_state)
            zero_action = torch.zeros(
                env.action_space.shape, dtype=torch.float32, device=env.unwrapped.device
            )
            if int(torch.count_nonzero(zero_action).detach().cpu()) != 0:
                raise TaskSmokeError("M6 zero action tensor is not exactly all zero")
            m6_contract["zero_action"] = {
                "shape": list(zero_action.shape),
                "dtype": str(zero_action.dtype),
                "sha256": _tensor_sha256(zero_action, "m6_zero_action", torch),
            }

        for step in range(args_cli.steps):
            if zero_action is None:
                action = torch.zeros(
                    env.action_space.shape, dtype=torch.float32, device=env.unwrapped.device
                )
            else:
                action = zero_action
                if int(torch.count_nonzero(action).detach().cpu()) != 0:
                    raise TaskSmokeError(f"M6 action at step {step + 1} is not exactly zero")
            transition = env.step(action)
            _finite_tree(transition, f"transition_{step}")
            if m6_contract is not None:
                assert scene_contract is not None
                assert m6_trace_writer is not None
                assert m6_state_progress is not None
                dones = _zero_action_dones(transition, num_envs=1, torch=torch)
                episode_length_values = _m6_episode_counter(
                    env.unwrapped, expected_step=step + 1, num_envs=1, torch=torch
                )
                state_record = _collect_m6_public_state(
                    scene_contract, step=step + 1, num_envs=1, torch=torch
                )
                state_record["phase"] = "post_step"
                state_record["dones"] = dones
                state_record["episode_length_buf"] = episode_length_values
                m6_trace_writer.write(state_record)
                _update_m6_state_progress(m6_state_progress, state_record)
                m6_contract["dones_false_steps"] += 1

        summary["backend_after"] = assert_physx_environment(env)
        if m6_contract is not None:
            assert m6_state_progress is not None
            m6_contract["actual_physx_manager_after"] = summary["backend_after"]["actual_manager"]
            m6_contract["actual_physx_manager_after_evidence"] = summary["backend_after"]
            m6_contract["public_state"] = m6_state_progress
            m6_contract["no_resets_after_initial"] = True
            m6_contract["resets_after_initial_verified"] = True
            m6_contract["no_explosion"] = True
            m6_contract["passed"] = True
        summary["observation_shape"] = list(policy_observations.shape)
        summary["action_shape"] = list(env.action_space.shape)
        summary["steps_completed"] = args_cli.steps
        summary["physics_ticks_completed"] = args_cli.steps * int(env.unwrapped.cfg.decimation)
        summary["passed"] = True
        print("RAMBO_PHYSX_TASK_SMOKE_SUCCESS", flush=True)
    except BaseException as error:
        exit_code = 1
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        if m6_contract is not None and base_env_for_m6 is not None:
            diagnostics = getattr(base_env_for_m6, "_last_termination_diagnostics", None)
            if isinstance(diagnostics, dict):
                m6_contract["native_termination_diagnostics"] = _jsonable(diagnostics)
        traceback.print_exception(type(error), error, error.__traceback__)
    finally:
        evidence_paths: list[Path] = []
        if m6_trace_writer is not None:
            try:
                m6_trace_writer.close()
                evidence_paths.append(m6_trace_writer.path)
                if m6_contract is not None:
                    public_state = m6_contract.setdefault("public_state", m6_state_progress or {"records": 0})
                    public_state["records"] = m6_trace_writer.records
                    public_state["trace_file"] = m6_trace_writer.path.name
                    public_state["trace_sha256"] = _sha256_path(m6_trace_writer.path)
            except BaseException as trace_error:
                exit_code = 1
                summary["m6_trace_close_error"] = f"{type(trace_error).__name__}: {trace_error}"
                if "traceback" not in summary:
                    summary["traceback"] = traceback.format_exc()
        if env is not None and "backend_after" not in summary:
            try:
                from rambo.utils import assert_physx_environment

                summary["backend_after"] = assert_physx_environment(env)
                if m6_contract is not None:
                    m6_contract["actual_physx_manager_after"] = summary["backend_after"]["actual_manager"]
                    m6_contract["actual_physx_manager_after_evidence"] = summary["backend_after"]
            except BaseException as backend_error:
                exit_code = 1
                summary["backend_after_error"] = f"{type(backend_error).__name__}: {backend_error}"
                if "traceback" not in summary:
                    summary["traceback"] = traceback.format_exc()
        if env is not None:
            try:
                env.close()
            except BaseException as close_error:
                exit_code = 1
                summary["close_error"] = f"{type(close_error).__name__}: {close_error}"
                if "traceback" not in summary:
                    summary["traceback"] = traceback.format_exc()
        if m6_contract is not None:
            m6_contract["passed"] = exit_code == 0
        summary["passed"] = exit_code == 0
        summary["shutdown_mode"] = "isaacsim_default_fast_shutdown"
        _write_summary_and_checksums(output_dir, summary, tuple(evidence_paths))
        if simulation_app is not None:
            simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
