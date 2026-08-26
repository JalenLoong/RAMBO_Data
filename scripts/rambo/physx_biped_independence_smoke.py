#!/usr/bin/env python3
"""Record a two-step, PhysX-only FL/FR biped independence gate.

The smoke intentionally has no policy, RGB, GUI, randomization, or checkpoint
dependency.  After a deterministic reset it writes two distinct safe front-foot
position commands, an all-zero FL force command, and a small nonzero FR force
command.  It then executes exactly two zero-action biped steps and fails closed
unless public PhysX and controller evidence proves those paths remained
independent.

Use an explicit ``--viz none`` or ``--viz kit``.  The latter only selects the
existing Kit visualizer; all task debug visualization remains disabled.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, MutableMapping
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import traceback
from typing import Any, Callable

from rambo.validation.biped_independence import (
    ARTIFACT_KIND,
    BIPED_BASELINE_TERMINATION_FLAGS,
    NUM_ENVS,
    NUM_FEET,
    NUM_LOGICAL_JOINTS,
    PERMANENT_WRENCH_OUTPUT_API,
    PHYSX_CFG_FQN,
    REQUIRED_STEPS,
    SCHEMA_VERSION,
    TASK_ID,
    validate_biped_independence_summary,
    validate_command_buffers,
    validate_permanent_wrench_output,
    validate_qp_contact_trace,
)


ARTIFACT_ROOT = Path("/workspace/runs/audit/isaac60/M9")
FL_POSITION_COMMAND = (0.22, 0.10, 0.48)
FR_POSITION_COMMAND = (0.24, -0.10, 0.50)
FL_FORCE_COMMAND = (0.0, 0.0, 0.0)
FR_FORCE_COMMAND = (0.0, 1.0, 0.0)
EXPECTED_ACTION_DIM = 18
EXPECTED_OBSERVATION_DIM = 435
FIXED_SEED = 42


class BipedIndependenceSmokeError(RuntimeError):
    """Raised when the focused M9 runtime independence contract is incomplete."""


def _git_output(cwd: Path, *arguments: str) -> str:
    """Read one source-provenance fact without starting Kit or mutating Git state."""

    process = subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip() or "unknown Git error"
        raise BipedIndependenceSmokeError(f"Cannot inspect clean source provenance in {cwd}: {message}")
    return process.stdout.rstrip("\n")


def _collect_clean_source_provenance() -> dict[str, Any]:
    """Prove the actual imported RAMBO module is this clean checkout before Kit starts."""

    launcher_root = Path(_git_output(Path(__file__).resolve().parents[2], "rev-parse", "--show-toplevel")).resolve()
    try:
        import rambo
    except ModuleNotFoundError as error:  # pragma: no cover - target-runtime installation guard.
        raise BipedIndependenceSmokeError("Cannot import RAMBO for M9 source provenance") from error
    module_path = Path(rambo.__file__).resolve()
    module_root = Path(_git_output(module_path.parent, "rev-parse", "--show-toplevel")).resolve()
    if module_root != launcher_root:
        raise BipedIndependenceSmokeError(
            "Imported RAMBO module is not from this M9 launcher checkout: "
            f"module_root={module_root}, launcher_root={launcher_root}"
        )
    status = _git_output(module_root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    if status:
        raise BipedIndependenceSmokeError("M9 final independence artifact requires a clean RAMBO worktree")
    return {
        "schema_version": 1,
        "collection_phase": "pre_app_launcher_source",
        "module_path": str(module_path),
        "git_root": str(module_root),
        "git_head": _git_output(module_root, "rev-parse", "HEAD"),
        "pre_run_worktree_clean": True,
        "pre_run_status_porcelain_v1": [],
    }


def _build_parser() -> tuple[argparse.ArgumentParser, type]:
    """Build the CLI without importing any simulator module at file import time."""

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as error:  # pragma: no cover - target-runtime guard.
        raise RuntimeError(
            "physx_biped_independence_smoke.py requires the RAMBO Isaac Lab target runtime"
        ) from error

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=f"New M9 evidence directory below {ARTIFACT_ROOT}.",
    )
    parser.add_argument("--seed", type=int, default=FIXED_SEED, help=f"Must remain the fixed seed {FIXED_SEED}.")
    parser.add_argument(
        "--disable-fabric",
        "--disable_fabric",
        dest="disable_fabric",
        action="store_true",
        help="Use USD I/O instead of Fabric.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser, AppLauncher


def _validate_cli_contract(
    parser: Any,
    args: Any,
    argv: list[str],
    *,
    validate_visualizer_args: Callable[[Any, Any, list[str]], list[str]] | None = None,
) -> list[str]:
    """Apply the shared fail-closed RAMBO visualizer gate to this focused smoke."""

    if validate_visualizer_args is None:
        from rambo.utils.physx import validate_rambo_visualizer_args

        validate_visualizer_args = validate_rambo_visualizer_args

    if int(args.seed) != FIXED_SEED:
        parser.error(f"--seed is fixed at {FIXED_SEED} for this deterministic independence gate")
    return validate_visualizer_args(parser, args, argv)


def _runtime_imports() -> dict[str, Any]:
    """Load the target-runtime dependencies only after ``AppLauncher`` starts Kit."""

    from rambo.torch_runtime import ensure_cuda_linalg_loaded

    ensure_cuda_linalg_loaded()
    import gymnasium as gym
    import rambo
    import torch
    from rambo.utils.physx import assert_physx_environment, configure_physx
    from rambo.utils.registry import parse_env_cfg

    if not rambo.register_tasks():
        raise BipedIndependenceSmokeError("RAMBO task registration requires an active Isaac Sim Kit application")
    return {
        "assert_physx_environment": assert_physx_environment,
        "configure_physx": configure_physx,
        "gym": gym,
        "parse_env_cfg": parse_env_cfg,
        "torch": torch,
    }


def _fqn(value: Any) -> str:
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BipedIndependenceSmokeError(message)


def _required_attr(value: Any, name: str, label: str) -> Any:
    try:
        result = getattr(value, name)
    except AttributeError as error:
        raise BipedIndependenceSmokeError(f"Required API {label}.{name} is unavailable") from error
    if result is None:
        raise BipedIndependenceSmokeError(f"Required API {label}.{name} is unavailable")
    return result


def _require_tensor(value: Any, label: str, torch: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        raise BipedIndependenceSmokeError(f"{label} must be a torch.Tensor, got {type(value).__name__}")
    if value.is_floating_point() or value.is_complex():
        if not bool(torch.isfinite(value).all()):
            raise BipedIndependenceSmokeError(f"{label} contains NaN or Inf")
    return value


def _tensor_record(value: Any, label: str, torch: Any) -> dict[str, Any]:
    tensor = _require_tensor(value, label, torch).detach()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "values": tensor.cpu().tolist(),
    }


def _tensor_stats(value: Any, label: str, torch: Any) -> dict[str, Any]:
    tensor = _require_tensor(value, label, torch).detach()
    if tensor.numel() == 0:
        raise BipedIndependenceSmokeError(f"{label} must not be empty")
    numeric = tensor.float()
    if not bool(torch.isfinite(numeric).all()):
        raise BipedIndependenceSmokeError(f"{label} contains non-finite values after float conversion")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": True,
        "min": float(numeric.min().cpu()),
        "max": float(numeric.max().cpu()),
        "abs_max": float(numeric.abs().max().cpu()),
    }


def _public_torch_view(value: Any, label: str, torch: Any) -> Any:
    """Read a documented Isaac Lab ``ProxyArray`` through its public Torch view."""

    tensor = getattr(value, "torch", None)
    if not isinstance(tensor, torch.Tensor):
        raise BipedIndependenceSmokeError(
            f"Required public API {label} does not expose a torch.Tensor .torch view"
        )
    return _require_tensor(tensor, label, torch)


def _set_mapping_flag(cfg: Any, mapping_name: str, key: str, value: bool) -> None:
    mapping = _required_attr(cfg, mapping_name, "env_cfg")
    if not isinstance(mapping, MutableMapping):
        raise BipedIndependenceSmokeError(f"env_cfg.{mapping_name} must be a mutable mapping")
    if key not in mapping:
        raise BipedIndependenceSmokeError(f"env_cfg.{mapping_name}.{key} is unavailable")
    mapping[key] = value


def _set_required_attr(cfg: Any, name: str, value: Any) -> None:
    try:
        getattr(cfg, name)
    except AttributeError as error:
        raise BipedIndependenceSmokeError(f"Required API env_cfg.{name} is unavailable") from error
    setattr(cfg, name, value)


def _configure_independence_env_cfg(
    env_cfg: Any,
    *,
    configure_physx: Callable[[Any], Any],
    seed: int,
) -> dict[str, Any]:
    """Strip every nonessential source of randomness and task debug visualization."""

    configure_physx(env_cfg)
    sim_cfg = _required_attr(env_cfg, "sim", "env_cfg")
    scene_cfg = _required_attr(env_cfg, "scene", "env_cfg")
    terrain_cfg = _required_attr(env_cfg, "terrain", "env_cfg")
    _set_required_attr(env_cfg, "seed", seed)
    _set_required_attr(scene_cfg, "num_envs", NUM_ENVS)
    _set_required_attr(sim_cfg, "use_newton_actuators", False)
    _set_required_attr(env_cfg, "events", None)
    _set_required_attr(env_cfg, "randomize_initial_state", False)
    _set_required_attr(env_cfg, "randomize_episode_progress", False)
    _set_required_attr(env_cfg, "contact_phase_offset_s", 0.0)
    _set_required_attr(env_cfg, "obs_noise", False)
    _set_required_attr(env_cfg, "action_noise_model", None)
    _set_required_attr(env_cfg, "observation_noise_model", None)
    _set_required_attr(env_cfg, "enable_rgb_camera", False)
    _set_required_attr(env_cfg, "rerender_on_reset", False)
    _set_required_attr(env_cfg, "num_rerenders_on_reset", 0)
    _set_required_attr(env_cfg, "wait_for_textures", False)
    _set_required_attr(terrain_cfg, "debug_vis", False)

    for name in (
        "enable_sampled_velocity_commands",
        "enable_sampled_pos_commands",
        "enable_sampled_force_commands",
        "velocity_debug_vis",
        "pos_debug_vis",
        "force_debug_vis",
        "use_actual_contact",
    ):
        _set_required_attr(env_cfg, name, False)
    _set_required_attr(env_cfg, "add_feedforward_torque", True)
    _set_mapping_flag(env_cfg, "contact_generator_config", "contact_generator_debug_vis", False)
    _set_mapping_flag(env_cfg, "joint_position_controller_config", "joint_position_controller_debug_vis", False)
    _set_mapping_flag(env_cfg, "qp_torque_optimizer_config", "qp_debug_vis", False)

    configured_physics = _fqn(_required_attr(sim_cfg, "physics", "env_cfg.sim"))
    if configured_physics != PHYSX_CFG_FQN:
        raise BipedIndependenceSmokeError(f"Biped independence config is not PhysxCfg: {configured_physics}")
    if getattr(sim_cfg, "use_newton_actuators", None) is not False:
        raise BipedIndependenceSmokeError("Biped independence smoke requires use_newton_actuators=False")
    for name, expected in BIPED_BASELINE_TERMINATION_FLAGS.items():
        actual = getattr(env_cfg, name, None)
        if actual is not expected:
            raise BipedIndependenceSmokeError(
                f"Biped independence smoke must preserve {name}={expected}, got {actual!r}"
            )
    return {
        "cfg": configured_physics,
        "use_newton_actuators": False,
        "use_fabric": bool(getattr(sim_cfg, "use_fabric", False)),
    }


def _determinism_evidence(env_cfg: Any) -> dict[str, Any]:
    """Capture every runtime override that keeps this gate repeatable and nonvisual."""

    evidence = {
        "seed": getattr(env_cfg, "seed", None),
        "events_disabled": getattr(env_cfg, "events", object()) is None,
        "randomize_initial_state": getattr(env_cfg, "randomize_initial_state", None),
        "randomize_episode_progress": getattr(env_cfg, "randomize_episode_progress", None),
        "contact_phase_offset_s": getattr(env_cfg, "contact_phase_offset_s", None),
        "observation_noise": getattr(env_cfg, "obs_noise", None),
        "action_noise_model_disabled": getattr(env_cfg, "action_noise_model", object()) is None,
        "observation_noise_model_disabled": getattr(env_cfg, "observation_noise_model", object()) is None,
        "enable_rgb_camera": getattr(env_cfg, "enable_rgb_camera", None),
        "rerender_on_reset": getattr(env_cfg, "rerender_on_reset", None),
        "num_rerenders_on_reset": getattr(env_cfg, "num_rerenders_on_reset", None),
        "wait_for_textures": getattr(env_cfg, "wait_for_textures", None),
        "baseline_termination_flags": {
            name: getattr(env_cfg, name, None) for name in BIPED_BASELINE_TERMINATION_FLAGS
        },
        "sampled_commands": {
            "velocity": getattr(env_cfg, "enable_sampled_velocity_commands", None),
            "position": getattr(env_cfg, "enable_sampled_pos_commands", None),
            "force": getattr(env_cfg, "enable_sampled_force_commands", None),
        },
        "debug_visualization": {
            "velocity": getattr(env_cfg, "velocity_debug_vis", None),
            "position": getattr(env_cfg, "pos_debug_vis", None),
            "force": getattr(env_cfg, "force_debug_vis", None),
            "terrain": getattr(getattr(env_cfg, "terrain", None), "debug_vis", None),
            "contact_generator": getattr(env_cfg, "contact_generator_config", {}).get(
                "contact_generator_debug_vis"
            ),
            "joint_position_controller": getattr(env_cfg, "joint_position_controller_config", {}).get(
                "joint_position_controller_debug_vis"
            ),
            "qp": getattr(env_cfg, "qp_torque_optimizer_config", {}).get("qp_debug_vis"),
        },
    }
    _require(evidence["events_disabled"] is True, "events must be disabled")
    for key in (
        "randomize_initial_state",
        "randomize_episode_progress",
        "observation_noise",
        "enable_rgb_camera",
        "rerender_on_reset",
        "wait_for_textures",
    ):
        _require(evidence[key] is False, f"determinism.{key} must be false")
    _require(evidence["contact_phase_offset_s"] == 0.0, "contact phase offset must be zero")
    _require(evidence["action_noise_model_disabled"] is True, "action noise must be disabled")
    _require(evidence["observation_noise_model_disabled"] is True, "observation noise model must be disabled")
    _require(evidence["num_rerenders_on_reset"] == 0, "reset rerenders must be disabled")
    for key, value in evidence["sampled_commands"].items():
        _require(value is False, f"sampled command {key} must be disabled")
    for key, value in evidence["debug_visualization"].items():
        _require(value is False, f"debug visualization {key} must be disabled")
    _require(
        evidence["baseline_termination_flags"] == BIPED_BASELINE_TERMINATION_FLAGS,
        "biped baseline termination flags must remain unchanged",
    )
    return evidence


def _command_range(cfg: Any, name: str) -> tuple[float, float]:
    value = _required_attr(cfg, name, "env_cfg")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise BipedIndependenceSmokeError(f"env_cfg.{name} must be a two-value numeric range")
    lower = float(value[0])
    upper = float(value[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise BipedIndependenceSmokeError(f"env_cfg.{name} is not a finite ordered range")
    return lower, upper


def _validate_safe_command_ranges(env_cfg: Any) -> dict[str, list[float]]:
    """Require every hard-coded command to remain inside the biped's declared safe range."""

    specifications = (
        ("fl_position", FL_POSITION_COMMAND, ("fl_pos_x", "fl_pos_y", "fl_pos_z")),
        ("fr_position", FR_POSITION_COMMAND, ("fr_pos_x", "fr_pos_y", "fr_pos_z")),
        ("fl_force", FL_FORCE_COMMAND, ("fl_force_x", "fl_force_y", "fl_force_z")),
        ("fr_force", FR_FORCE_COMMAND, ("fr_force_x", "fr_force_y", "fr_force_z")),
    )
    evidence: dict[str, list[float]] = {}
    for label, values, range_names in specifications:
        ranges = [_command_range(env_cfg, name) for name in range_names]
        for axis, (requested, bounds) in enumerate(zip(values, ranges, strict=True)):
            if not bounds[0] <= requested <= bounds[1]:
                raise BipedIndependenceSmokeError(
                    f"{label}[{axis}]={requested} is outside configured safe range {bounds}"
                )
        evidence[label] = [float(value) for pair in ranges for value in pair]
    return evidence


def _verify_runtime_debug_vis_disabled(base_env: Any) -> dict[str, bool]:
    """Fail if an instantiated controller re-enabled its own visualization hook."""

    values: dict[str, bool] = {}
    for attribute, label in (
        ("contact_generator", "contact_generator"),
        ("joint_position_controller", "joint_position_controller"),
        ("torque_optimizer", "qp_torque_optimizer"),
    ):
        controller = _required_attr(base_env, attribute, "base_env")
        debug_vis = getattr(controller, "debug_vis", None)
        if debug_vis is not False:
            raise BipedIndependenceSmokeError(f"Runtime {label}.debug_vis must be false")
        values[label] = False
    return values


def _seed_everything(seed: int, torch: Any) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _policy_observation(observations: Any, label: str, torch: Any) -> Any:
    if not isinstance(observations, Mapping):
        raise BipedIndependenceSmokeError(f"{label} must be an observation mapping")
    policy = _require_tensor(observations.get("policy"), f"{label}.policy", torch)
    expected_shape = (NUM_ENVS, EXPECTED_OBSERVATION_DIM)
    if tuple(policy.shape) != expected_shape:
        raise BipedIndependenceSmokeError(
            f"{label}.policy must have shape {expected_shape}, got {tuple(policy.shape)}"
        )
    return policy


def _command_tensor(base_env: Any, name: str, torch: Any) -> Any:
    tensor = _require_tensor(_required_attr(base_env, name, "base_env"), f"base_env.{name}", torch)
    expected_shape = (NUM_ENVS, 3)
    if tuple(tensor.shape) != expected_shape:
        raise BipedIndependenceSmokeError(
            f"base_env.{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    return tensor


def _write_front_commands(base_env: Any, torch: Any) -> None:
    """Write deliberately asymmetric, in-range FL/FR commands after reset."""

    commands = {
        "_ee_pos_fl_commands": FL_POSITION_COMMAND,
        "_ee_pos_fr_commands": FR_POSITION_COMMAND,
        "_ee_force_fl_commands": FL_FORCE_COMMAND,
        "_ee_force_fr_commands": FR_FORCE_COMMAND,
    }
    for name, requested in commands.items():
        buffer = _command_tensor(base_env, name, torch)
        target = torch.tensor([requested], device=buffer.device, dtype=buffer.dtype)
        buffer.copy_(target)
        if not bool(torch.equal(buffer, target)):
            raise BipedIndependenceSmokeError(f"Failed to write exact {name} command buffer")


def _command_buffer_record(base_env: Any, torch: Any) -> dict[str, Any]:
    record = {
        "fl_position": _tensor_record(_command_tensor(base_env, "_ee_pos_fl_commands", torch), "FL position", torch),
        "fr_position": _tensor_record(_command_tensor(base_env, "_ee_pos_fr_commands", torch), "FR position", torch),
        "fl_force": _tensor_record(_command_tensor(base_env, "_ee_force_fl_commands", torch), "FL force", torch),
        "fr_force": _tensor_record(_command_tensor(base_env, "_ee_force_fr_commands", torch), "FR force", torch),
    }
    validate_command_buffers(record)
    return record


def _force_raw_front_contacts_false(base_env: Any, torch: Any) -> None:
    """Make the schedule source explicitly false immediately before each QP step."""

    generator = _required_attr(base_env, "contact_generator", "base_env")
    source = _require_tensor(
        _required_attr(generator, "desired_contact_state", "base_env.contact_generator"),
        "contact_generator.desired_contact_state",
        torch,
    )
    if tuple(source.shape) != (NUM_ENVS, NUM_FEET):
        raise BipedIndependenceSmokeError(
            "contact_generator.desired_contact_state must have shape "
            f"({NUM_ENVS}, {NUM_FEET}), got {tuple(source.shape)}"
        )
    if source.dtype != torch.bool:
        raise BipedIndependenceSmokeError("contact_generator.desired_contact_state must be bool")
    source[:, :2] = False
    if bool(torch.any(source[:, :2])):
        raise BipedIndependenceSmokeError("Could not force raw FL/FR desired contacts false")


def _resolve_front_foot_body_ids(base_env: Any) -> dict[str, dict[str, Any]]:
    """Resolve physical body indices from the public articulation body-name list."""

    scene = _required_attr(base_env, "scene", "base_env")
    articulations = _required_attr(scene, "articulations", "base_env.scene")
    if not isinstance(articulations, Mapping):
        raise BipedIndependenceSmokeError("base_env.scene.articulations must be a mapping")
    robot = articulations.get("robot")
    if robot is None:
        raise BipedIndependenceSmokeError("base_env.scene.articulations['robot'] is unavailable")
    data = _required_attr(robot, "data", "robot")
    names = _required_attr(data, "body_names", "robot.data")
    if not isinstance(names, (list, tuple)):
        raise BipedIndependenceSmokeError("robot.data.body_names must be an ordered list or tuple")
    result: dict[str, dict[str, Any]] = {}
    for key, body_name in (("fl", "FL_foot"), ("fr", "FR_foot")):
        if list(names).count(body_name) != 1:
            raise BipedIndependenceSmokeError(f"robot.data.body_names must resolve exactly one {body_name}")
        result[key] = {"body_name": body_name, "body_id": int(list(names).index(body_name))}
    if result["fl"]["body_id"] == result["fr"]["body_id"]:
        raise BipedIndependenceSmokeError("FL_foot and FR_foot resolved to the same physical body id")
    result["robot"] = {"value": robot}
    return result


def _permanent_wrench_output_record(
    robot: Any,
    front_feet: Mapping[str, Mapping[str, Any]],
    torch: Any,
) -> dict[str, Any]:
    """Read the public composed output, never a private composer backing buffer."""

    composer = _required_attr(robot, "permanent_wrench_composer", "robot")
    out_force = _public_torch_view(
        _required_attr(composer, "out_force_b", "robot.permanent_wrench_composer"),
        PERMANENT_WRENCH_OUTPUT_API,
        torch,
    )
    if out_force.ndim != 3 or out_force.shape[0] != NUM_ENVS or out_force.shape[-1] != 3:
        raise BipedIndependenceSmokeError(
            "Public permanent wrench output must have shape "
            f"({NUM_ENVS}, body_count, 3), got {tuple(out_force.shape)}"
        )
    records: dict[str, Any] = {}
    for key in ("fl", "fr"):
        foot = front_feet[key]
        body_id = int(foot["body_id"])
        if body_id < 0 or body_id >= out_force.shape[1]:
            raise BipedIndependenceSmokeError(f"Resolved {foot['body_name']} id is outside wrench output")
        records[key] = {
            "body_name": foot["body_name"],
            "body_id": body_id,
            "force_body": _tensor_record(
                out_force[:, body_id, :],
                f"permanent wrench output {foot['body_name']}",
                torch,
            ),
        }
    output = {
        "api": PERMANENT_WRENCH_OUTPUT_API,
        "active": bool(_required_attr(composer, "active", "robot.permanent_wrench_composer")),
        "shape": list(out_force.shape),
        "front_feet": records,
    }
    validate_permanent_wrench_output(output)
    return output


def _qp_trace_record(payload: Any, *, step: int, torch: Any) -> dict[str, Any]:
    """Serialize every required clone from the opt-in biped QP diagnostic sink."""

    if not isinstance(payload, Mapping):
        raise BipedIndependenceSmokeError("Runtime QP diagnostic sink payload must be a mapping")
    required = (
        "contact_source",
        "source_contact_state",
        "contact_schedule_mode",
        "contact_schedule_phase",
        "front_leg_logical_joint_slots",
        "front_leg_override_applied",
        "use_actual_contact",
        "contact_state_expanded",
        "grf",
        "desired_joint_torque",
        "desired_tor",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise BipedIndependenceSmokeError(f"Runtime QP diagnostic payload is missing {missing}")
    record = {
        "step": step,
        "contact_source": payload["contact_source"],
        "source_contact_state": _tensor_record(payload["source_contact_state"], "trace source contact", torch),
        "contact_schedule_mode": _tensor_record(payload["contact_schedule_mode"], "trace contact mode", torch),
        "contact_schedule_phase": _tensor_record(payload["contact_schedule_phase"], "trace contact phase", torch),
        "front_leg_logical_joint_slots": list(payload["front_leg_logical_joint_slots"]),
        "front_leg_override_applied": payload["front_leg_override_applied"],
        "use_actual_contact": payload["use_actual_contact"],
        "contact_state_expanded": _tensor_record(
            payload["contact_state_expanded"], "trace expanded contact", torch
        ),
        "grf": _tensor_record(payload["grf"], "trace grf", torch),
        "desired_joint_torque": _tensor_record(
            payload["desired_joint_torque"], "trace desired joint torque", torch
        ),
        "desired_tor": _tensor_record(payload["desired_tor"], "trace desired tor", torch),
    }
    validate_qp_contact_trace(record)
    return record


def _zero_action(base_env: Any, torch: Any) -> Any:
    action_space = _required_attr(base_env, "action_space", "base_env")
    shape = tuple(getattr(action_space, "shape", ()))
    expected_space_shape = (NUM_ENVS, EXPECTED_ACTION_DIM)
    if shape != expected_space_shape:
        raise BipedIndependenceSmokeError(
            f"Biped action space must have shape {expected_space_shape}, got {shape}"
        )
    num_envs = int(_required_attr(base_env, "num_envs", "base_env"))
    if num_envs != NUM_ENVS:
        raise BipedIndependenceSmokeError(f"Biped independence smoke requires exactly {NUM_ENVS} env, got {num_envs}")
    return torch.zeros((NUM_ENVS, EXPECTED_ACTION_DIM), device=base_env.device, dtype=torch.float32)


def _ensure_artifact_directory(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    root = ARTIFACT_ROOT.resolve()
    try:
        output_dir.relative_to(root)
    except ValueError as error:
        raise BipedIndependenceSmokeError(
            f"--output-dir must be below the M9 artifact root {root}: {output_dir}"
        ) from error
    if output_dir.exists():
        raise BipedIndependenceSmokeError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    return output_dir


def _assert_json_finite(value: Any, label: str = "summary") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_json_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{label}[{index}]")
    elif isinstance(value, bool) or value is None or isinstance(value, str):
        return
    elif isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise BipedIndependenceSmokeError(f"{label} is non-finite")
    else:
        raise BipedIndependenceSmokeError(f"{label} is not JSON-safe: {type(value).__name__}")


def _write_summary(output_dir: Path, summary: Mapping[str, Any]) -> None:
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    (output_dir / "checksums.sha256").write_text(f"{digest}  summary.json\n", encoding="utf-8")
    print(f"SUMMARY_PATH={summary_path}", flush=True)


def main() -> int:
    parser, app_launcher_type = _build_parser()
    args_cli = parser.parse_args()
    visualizer_selection = _validate_cli_contract(parser, args_cli, sys.argv[1:])
    try:
        output_dir = _ensure_artifact_directory(args_cli.output_dir)
    except BipedIndependenceSmokeError as error:
        parser.error(str(error))

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "passed": False,
        "task": TASK_ID,
        "requested_steps": REQUIRED_STEPS,
        "steps_completed": 0,
        "seed": args_cli.seed,
        "viz": visualizer_selection,
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "command_values": {
            "fl_position": list(FL_POSITION_COMMAND),
            "fr_position": list(FR_POSITION_COMMAND),
            "fl_force": list(FL_FORCE_COMMAND),
            "fr_force": list(FR_FORCE_COMMAND),
        },
    }
    try:
        summary["provenance"] = _collect_clean_source_provenance()
    except BipedIndependenceSmokeError as error:
        parser.error(str(error))
    simulation_app = None
    env = None
    base_env = None
    diagnostic_setter: Callable[[Any], Any] | None = None
    exit_code = 0
    try:
        app_launcher = app_launcher_type(args_cli)
        simulation_app = app_launcher.app
        runtime = _runtime_imports()
        torch = runtime["torch"]
        _seed_everything(args_cli.seed, torch)

        env_cfg = runtime["parse_env_cfg"](
            TASK_ID,
            device=args_cli.device or "cuda:0",
            num_envs=NUM_ENVS,
            use_fabric=not args_cli.disable_fabric,
        )
        summary["configured_physics"] = _configure_independence_env_cfg(
            env_cfg,
            configure_physx=runtime["configure_physx"],
            seed=args_cli.seed,
        )
        summary["determinism"] = _determinism_evidence(env_cfg)
        summary["command_safety_ranges"] = _validate_safe_command_ranges(env_cfg)

        env = runtime["gym"].make(TASK_ID, cfg=env_cfg)
        base_env = getattr(env, "unwrapped", env)
        summary["backend_before"] = runtime["assert_physx_environment"](env)
        summary["runtime_debug_visualization"] = _verify_runtime_debug_vis_disabled(base_env)

        reset_observations, _ = base_env.reset(seed=args_cli.seed)
        reset_policy = _policy_observation(reset_observations, "reset_observations", torch)
        summary["reset_observation"] = _tensor_stats(reset_policy, "reset policy observation", torch)
        _write_front_commands(base_env, torch)
        summary["command_buffers_before_steps"] = _command_buffer_record(base_env, torch)

        resolved = _resolve_front_foot_body_ids(base_env)
        robot = resolved.pop("robot")["value"]
        summary["front_foot_body_ids"] = resolved
        zero_action = _zero_action(base_env, torch)
        summary["zero_action"] = _tensor_record(zero_action, "zero action", torch)
        summary["runtime_shapes"] = {
            "policy_observation": list(reset_policy.shape),
            "zero_action": list(zero_action.shape),
            "front_command_buffer": [NUM_ENVS, 3],
            "contact_schedule": [NUM_ENVS, NUM_FEET],
            "contact_state_expanded": [NUM_ENVS, NUM_LOGICAL_JOINTS],
            "grf": [NUM_ENVS, NUM_LOGICAL_JOINTS],
            "desired_tor": [NUM_ENVS, NUM_LOGICAL_JOINTS],
        }
        if not bool(torch.all(zero_action == 0.0)):
            raise BipedIndependenceSmokeError("zero action construction failed")

        diagnostic_setter = getattr(base_env, "set_runtime_qp_diagnostic_sink", None)
        if not callable(diagnostic_setter):
            raise BipedIndependenceSmokeError(
                "Biped environment does not expose the required opt-in runtime QP diagnostic sink"
            )
        trace_payloads: list[dict[str, Any]] = []

        def _sink(payload: dict[str, Any]) -> None:
            trace_payloads.append(payload)

        diagnostic_setter(_sink)
        step_records: list[dict[str, Any]] = []
        try:
            for step in range(1, REQUIRED_STEPS + 1):
                _force_raw_front_contacts_false(base_env, torch)
                trace_count_before = len(trace_payloads)
                observations, rewards, terminated, time_outs, _ = base_env.step(zero_action)
                if len(trace_payloads) != trace_count_before + 1:
                    raise BipedIndependenceSmokeError(
                        "Each biped step must emit exactly one runtime QP diagnostic trace"
                    )
                policy_observation = _policy_observation(observations, f"step_{step}_observations", torch)
                _require_tensor(rewards, f"step_{step}_rewards", torch)
                _require_tensor(terminated, f"step_{step}_terminated", torch)
                _require_tensor(time_outs, f"step_{step}_time_outs", torch)
                if bool(torch.any(terminated)) or bool(torch.any(time_outs)):
                    raise BipedIndependenceSmokeError(f"Biped reset or timeout occurred during required step {step}")
                step_records.append(
                    {
                        "step": step,
                        "observation": _tensor_stats(policy_observation, f"step_{step}_policy", torch),
                        "reward": _tensor_stats(rewards, f"step_{step}_reward", torch),
                        "qp_contact_trace": _qp_trace_record(trace_payloads[-1], step=step, torch=torch),
                        "permanent_wrench_output": _permanent_wrench_output_record(robot, resolved, torch),
                    }
                )
        finally:
            diagnostic_setter(None)
            diagnostic_setter = None

        summary["steps"] = step_records
        summary["steps_completed"] = len(step_records)
        summary["command_buffers_after_steps"] = _command_buffer_record(base_env, torch)
        summary["backend_after"] = runtime["assert_physx_environment"](env)
        summary["passed"] = True
        _assert_json_finite(summary)
        validate_biped_independence_summary(summary)
        print("RAMBO_PHYSX_BIPED_INDEPENDENCE_SMOKE_SUCCESS", flush=True)
    except BaseException as error:
        exit_code = 1
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        traceback.print_exception(type(error), error, error.__traceback__)
    finally:
        if diagnostic_setter is not None:
            try:
                diagnostic_setter(None)
            except BaseException as error:
                exit_code = 1
                summary["diagnostic_sink_close_error"] = f"{type(error).__name__}: {error}"
        if env is not None:
            try:
                env.close()
            except BaseException as error:
                exit_code = 1
                summary["close_error"] = f"{type(error).__name__}: {error}"
        summary["passed"] = exit_code == 0
        summary["shutdown_mode"] = "isaacsim_default_fast_shutdown"
        _write_summary(output_dir, summary)

    if simulation_app is not None:
        simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
