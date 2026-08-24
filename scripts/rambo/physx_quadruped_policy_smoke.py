#!/usr/bin/env python3
"""Replay the released quadruped CRL2 policy for a finite PhysX-only smoke.

This launcher is deliberately narrower than ``play.py``: it accepts one
released checkpoint contract, one environment, and a finite number of policy
steps.  It records the exact configured and active PhysX classes before and
after the rollout so a successful artifact is evidence of actual PhysX use,
not merely the presence of a PhysX package in the environment.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import traceback
from typing import Any


TASK = "Isaac-RAMBO-Quadruped-Go2-v0"
DEFAULT_STEPS = 16
MEMORY_MONITOR_INTERVAL_STEPS = 100
MEMORY_ANALYSIS_START_STEP = 1000
MEMORY_WINDOW_STEPS = 500
MIB = 1024 * 1024
GPU_SLOPE_LIMIT_BYTES_PER_100_STEPS = 1 * MIB
GPU_MEDIAN_DELTA_LIMIT_BYTES = 128 * MIB
RSS_SLOPE_LIMIT_BYTES_PER_100_STEPS = 4 * MIB
RSS_MEDIAN_DELTA_LIMIT_BYTES = 512 * MIB


class PolicySmokeError(RuntimeError):
    """Raised when the fixed quadruped policy-smoke contract is violated."""


def _build_parser() -> tuple[argparse.ArgumentParser, type]:
    """Build the parser without importing Isaac Sim at module-import time."""

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:  # pragma: no cover - target-runtime guard.
        raise RuntimeError(
            "physx_quadruped_policy_smoke.py requires the RAMBO Isaac Lab runtime"
        ) from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Released quadruped model_2000.pt; its SHA-256 is verified before loading.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of synchronized quadruped environments to replay.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Finite policy steps to execute.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed Python/NumPy/Torch/environment seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for the JSON summary and its SHA-256 sidecar.",
    )
    parser.add_argument(
        "--disable-fabric",
        "--disable_fabric",
        dest="disable_fabric",
        action="store_true",
        help="Use USD I/O instead of Fabric.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser, AppLauncher


def _runtime_imports() -> dict[str, Any]:
    """Import simulator-dependent modules only after Kit has been launched."""

    from rambo.torch_runtime import ensure_cuda_linalg_loaded

    ensure_cuda_linalg_loaded()
    import gymnasium as gym
    import rambo
    import torch
    from crl2.algorithms import PPO
    from rambo.rl import Crl2VecEnvWrapper
    from rambo.utils.physx import (
        PHYSX_CFG_FQN,
        assert_physx_environment,
        configure_physx,
    )
    from rambo.utils.registry import load_cfg_from_registry, parse_env_cfg
    from rambo.validation.checkpoints import (
        contract_for_task,
        load_verified_checkpoint,
        restore_runner,
        sha256_file,
    )

    if not rambo.register_tasks():
        raise PolicySmokeError("RAMBO task registration requires an active Isaac Sim Kit application")
    return {
        "Crl2VecEnvWrapper": Crl2VecEnvWrapper,
        "PHYSX_CFG_FQN": PHYSX_CFG_FQN,
        "PPO": PPO,
        "assert_physx_environment": assert_physx_environment,
        "configure_physx": configure_physx,
        "contract_for_task": contract_for_task,
        "gym": gym,
        "load_cfg_from_registry": load_cfg_from_registry,
        "load_verified_checkpoint": load_verified_checkpoint,
        "parse_env_cfg": parse_env_cfg,
        "restore_runner": restore_runner,
        "sha256_file": sha256_file,
        "torch": torch,
    }


def _fqn(value: Any) -> str:
    """Return a stable fully-qualified class identity for summary evidence."""

    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _jsonable(value: Any) -> Any:
    """Convert summary values to JSON without retaining device tensors."""

    if isinstance(value, Path):
        return str(value)
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


def _tensor_metrics(tensor: Any, label: str) -> dict[str, Any]:
    """Fail closed on non-finite policy data and return small replay evidence."""

    import torch

    if not isinstance(tensor, torch.Tensor):
        raise PolicySmokeError(f"{label} must be a torch.Tensor, got {type(tensor).__name__}")
    if not bool(torch.isfinite(tensor).all()):
        raise PolicySmokeError(f"{label} contains NaN or Inf")
    detached = tensor.detach()
    return {
        "shape": list(detached.shape),
        "min": float(detached.min().cpu()),
        "max": float(detached.max().cpu()),
        "mean": float(detached.mean().cpu()),
        "abs_max": float(detached.abs().max().cpu()),
    }


def _public_torch_view(value: Any, name: str, torch: Any) -> Any:
    """Read an Isaac Lab public ProxyArray through its documented Torch view.

    Long-run safety checks intentionally do not fall back to RAMBO's private
    ``_robot`` implementation details.  A target runtime that does not expose
    this public Isaac Lab data contract must stop rather than silently weaken
    the validation.
    """

    tensor = getattr(value, "torch", None)
    if not isinstance(tensor, torch.Tensor):
        raise PolicySmokeError(
            f"Required public Isaac Lab data field {name} does not expose a torch.Tensor .torch view"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise PolicySmokeError(f"Required public Isaac Lab data field {name} contains NaN or Inf")
    return tensor


def _public_robot_data(base_env: Any) -> Any:
    """Resolve the robot through the public RAMBO scene/articulation registry."""

    scene = getattr(base_env, "scene", None)
    articulations = getattr(scene, "articulations", None)
    if not isinstance(articulations, Mapping):
        raise PolicySmokeError(
            "Required public RAMBO API base_env.scene.articulations is unavailable for safety monitoring"
        )
    robot = articulations.get("robot")
    if robot is None:
        raise PolicySmokeError(
            "Required public RAMBO API base_env.scene.articulations['robot'] is unavailable for safety monitoring"
        )
    data = getattr(robot, "data", None)
    if data is None:
        raise PolicySmokeError("Required public Isaac Lab Articulation.data is unavailable for safety monitoring")
    return data


def _collect_public_robot_state(
    base_env: Any,
    *,
    num_envs: int,
    step: int,
    contract: Any,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read every requested safety signal from public RAMBO/Isaac Lab data.

    Returns a JSON-ready per-step record and compact extrema used by the
    fail-closed checks.  No private environment or backend data is consulted.
    """

    data = _public_robot_data(base_env)
    root_link_pos_w = _public_torch_view(
        getattr(data, "root_link_pos_w", None), "robot.data.root_link_pos_w", torch
    )
    projected_gravity_b = _public_torch_view(
        getattr(data, "projected_gravity_b", None), "robot.data.projected_gravity_b", torch
    )
    joint_vel = _public_torch_view(getattr(data, "joint_vel", None), "robot.data.joint_vel", torch)
    joint_vel_limits = _public_torch_view(
        getattr(data, "joint_vel_limits", None), "robot.data.joint_vel_limits", torch
    )

    if root_link_pos_w.ndim != 2 or tuple(root_link_pos_w.shape) != (num_envs, 3):
        raise PolicySmokeError(
            "Public robot.data.root_link_pos_w has unexpected shape: "
            f"expected ({num_envs}, 3), got {tuple(root_link_pos_w.shape)}"
        )
    if projected_gravity_b.ndim != 2 or tuple(projected_gravity_b.shape) != (num_envs, 3):
        raise PolicySmokeError(
            "Public robot.data.projected_gravity_b has unexpected shape: "
            f"expected ({num_envs}, 3), got {tuple(projected_gravity_b.shape)}"
        )
    if joint_vel.ndim != 2 or joint_vel.shape[0] != num_envs:
        raise PolicySmokeError(
            "Public robot.data.joint_vel has unexpected shape: "
            f"expected ({num_envs}, joint_count), got {tuple(joint_vel.shape)}"
        )
    if tuple(joint_vel_limits.shape) != tuple(joint_vel.shape):
        raise PolicySmokeError(
            "Public robot.data.joint_vel_limits shape does not match joint_vel: "
            f"limits={tuple(joint_vel_limits.shape)}, velocity={tuple(joint_vel.shape)}"
        )
    if not bool(torch.all(joint_vel_limits > 0.0)):
        raise PolicySmokeError("Public robot.data.joint_vel_limits must be finite positive simulation limits")

    gravity_target = torch.tensor(
        contract.gravity_target, device=projected_gravity_b.device, dtype=projected_gravity_b.dtype
    )
    base_height = root_link_pos_w[:, 2]
    orientation_error = torch.linalg.vector_norm(projected_gravity_b - gravity_target, dim=-1)
    joint_velocity_abs = joint_vel.abs()
    joint_velocity_ratio = joint_velocity_abs / joint_vel_limits

    min_base_height = float(base_height.min().detach().cpu())
    max_orientation_error = float(orientation_error.max().detach().cpu())
    max_joint_velocity_ratio = float(joint_velocity_ratio.max().detach().cpu())
    if min_base_height < contract.min_base_height:
        raise PolicySmokeError(
            f"base height {min_base_height:.6f} is below {contract.min_base_height:.6f} at step {step}"
        )
    if max_orientation_error > contract.max_orientation_error:
        raise PolicySmokeError(
            "orientation error "
            f"{max_orientation_error:.6f} exceeds {contract.max_orientation_error:.6f} at step {step}"
        )
    if max_joint_velocity_ratio > 1.01:
        raise PolicySmokeError(
            "joint velocity exceeds the public simulation velocity limit by more than 1.01x "
            f"at step {step}: ratio={max_joint_velocity_ratio:.6f}"
        )

    joint_names = getattr(data, "joint_names", None)
    if not isinstance(joint_names, (list, tuple)) or len(joint_names) != joint_vel.shape[1]:
        raise PolicySmokeError(
            "Required public Isaac Lab robot.data.joint_names is unavailable or mismatched for velocity evidence"
        )
    record = {
        "step": step,
        "base_height_m": base_height.detach().cpu().tolist(),
        "orientation_error": orientation_error.detach().cpu().tolist(),
        "joint_velocity_rad_s": joint_vel.detach().cpu().tolist(),
    }
    extrema = {
        "min_base_height": min_base_height,
        "max_orientation_error": max_orientation_error,
        "max_joint_velocity_ratio": max_joint_velocity_ratio,
        "max_abs_joint_velocity_per_joint": joint_velocity_abs.amax(dim=0).detach().cpu().tolist(),
        "joint_velocity_limit_per_joint": joint_vel_limits[0].detach().cpu().tolist(),
        "joint_names": list(joint_names),
    }
    return record, extrema


class _StateTraceWriter:
    """Stream per-step public-state evidence without retaining a long rollout in RAM."""

    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("x", encoding="utf-8")
        self.records = 0

    def write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        self.records += 1

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def _rss_bytes() -> int:
    """Read current Linux resident-set size exactly from the public procfs interface."""

    statm_path = Path("/proc/self/statm")
    try:
        resident_pages = int(statm_path.read_text(encoding="utf-8").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, OSError, ValueError) as exc:
        raise PolicySmokeError("RSS monitoring requires readable Linux /proc/self/statm") from exc


def _memory_sample(step: int, torch: Any, device: Any) -> dict[str, int]:
    """Capture the exact requested GPU-allocation and RSS quantities."""

    if not torch.cuda.is_available():
        raise PolicySmokeError("M6 memory monitoring requires a CUDA-capable PyTorch runtime")
    return {
        "step": step,
        "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device=device)),
        "rss_bytes": _rss_bytes(),
    }


def _linear_slope_bytes_per_100_steps(samples: list[dict[str, int]], field: str) -> float:
    """Return ordinary-least-squares growth over 100 policy steps."""

    if len(samples) < 2:
        raise PolicySmokeError(f"Need at least two memory samples to calculate {field} slope")
    xs = [float(sample["step"]) for sample in samples]
    ys = [float(sample[field]) for sample in samples]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0.0:
        raise PolicySmokeError(f"Memory sample steps are degenerate for {field} slope")
    return 100.0 * sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def _memory_analysis(samples: list[dict[str, int]], total_steps: int) -> dict[str, Any]:
    """Calculate and enforce the approved long-run memory leak thresholds."""

    if total_steps < MEMORY_ANALYSIS_START_STEP:
        return {
            "evaluated": False,
            "reason": f"requires steps >= {MEMORY_ANALYSIS_START_STEP}",
        }
    post_warmup = [sample for sample in samples if sample["step"] >= MEMORY_ANALYSIS_START_STEP]
    first_window = [sample for sample in samples if sample["step"] <= MEMORY_WINDOW_STEPS]
    last_window_start = total_steps - MEMORY_WINDOW_STEPS
    last_window = [sample for sample in samples if sample["step"] > last_window_start]
    if len(post_warmup) < 2 or not first_window or not last_window:
        raise PolicySmokeError("Insufficient periodic memory samples for fail-closed M6 analysis")

    result: dict[str, Any] = {
        "evaluated": True,
        "analysis_start_step": MEMORY_ANALYSIS_START_STEP,
        "first_window_steps": [sample["step"] for sample in first_window],
        "last_window_steps": [sample["step"] for sample in last_window],
    }
    for label, field, slope_limit, median_delta_limit in (
        (
            "gpu",
            "gpu_allocated_bytes",
            GPU_SLOPE_LIMIT_BYTES_PER_100_STEPS,
            GPU_MEDIAN_DELTA_LIMIT_BYTES,
        ),
        ("rss", "rss_bytes", RSS_SLOPE_LIMIT_BYTES_PER_100_STEPS, RSS_MEDIAN_DELTA_LIMIT_BYTES),
    ):
        slope = _linear_slope_bytes_per_100_steps(post_warmup, field)
        first_median = float(statistics.median(sample[field] for sample in first_window))
        last_median = float(statistics.median(sample[field] for sample in last_window))
        median_delta = last_median - first_median
        result[label] = {
            "slope_bytes_per_100_steps": slope,
            "slope_mib_per_100_steps": slope / MIB,
            "slope_limit_mib_per_100_steps": slope_limit / MIB,
            "first_500_step_median_bytes": first_median,
            "last_500_step_median_bytes": last_median,
            "median_delta_bytes": median_delta,
            "median_delta_mib": median_delta / MIB,
            "median_delta_limit_mib": median_delta_limit / MIB,
        }
        if slope > slope_limit:
            raise PolicySmokeError(
                f"{label} allocation slope {slope / MIB:.6f} MiB/100 steps exceeds "
                f"{slope_limit / MIB:.6f} MiB/100 steps"
            )
        if median_delta > median_delta_limit:
            raise PolicySmokeError(
                f"{label} first/last 500-step median growth {median_delta / MIB:.6f} MiB exceeds "
                f"{median_delta_limit / MIB:.6f} MiB"
            )
    return result


def _configure_deterministic_smoke(env_cfg: Any, seed: int, steps: int) -> Any:
    """Disable randomization/camera work and leave enough time for a finite replay."""

    if hasattr(env_cfg, "seed"):
        env_cfg.seed = seed
    if hasattr(env_cfg, "events"):
        env_cfg.events = None
    for attribute in (
        "randomize_initial_state",
        "obs_noise",
        "randomize_episode_progress",
        "enable_sampled_velocity_commands",
        "enable_sampled_pos_commands",
        "enable_sampled_force_commands",
        "velocity_debug_vis",
        "pos_debug_vis",
        "force_debug_vis",
        "enable_rgb_camera",
    ):
        if hasattr(env_cfg, attribute):
            setattr(env_cfg, attribute, False)
    required_disabled = (
        "events",
        "randomize_initial_state",
        "obs_noise",
        "randomize_episode_progress",
        "enable_sampled_velocity_commands",
        "enable_sampled_pos_commands",
        "enable_sampled_force_commands",
        "enable_rgb_camera",
    )
    for attribute in required_disabled:
        expected = None if attribute == "events" else False
        if not hasattr(env_cfg, attribute) or getattr(env_cfg, attribute) is not expected:
            raise PolicySmokeError(
                f"Deterministic M6 smoke requires env_cfg.{attribute}={expected!r}"
            )
    try:
        step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    except (AttributeError, TypeError, ValueError) as exc:
        raise PolicySmokeError("M6 smoke cannot determine public simulation step duration") from exc
    if step_dt <= 0.0:
        raise PolicySmokeError(f"M6 smoke requires a positive policy step duration, got {step_dt}")
    # A one-second margin avoids the task's normal finite-horizon timeout on
    # the final requested step without modifying robot, controller, or physics
    # semantics.  The RAMBO contact schedule has the same original 10-second
    # horizon, so preserve its existing modes and extend only each final
    # segment on a copied configuration.  Otherwise a 30-second acceptance
    # run crosses a schedule boundary that the controller cannot represent.
    required_duration_s = steps * step_dt + 1.0
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), required_duration_s)
    contact_generator_config = getattr(env_cfg, "contact_generator_config", None)
    if not isinstance(contact_generator_config, dict):
        raise PolicySmokeError("M6 smoke requires a dict env_cfg.contact_generator_config")
    copied_contact_generator_config = copy.deepcopy(contact_generator_config)
    contact_sequence = copied_contact_generator_config.get("contact_sequence")
    if not isinstance(contact_sequence, dict) or not contact_sequence:
        raise PolicySmokeError("M6 smoke requires a non-empty contact_generator_config.contact_sequence")
    for foot_name, sequence in contact_sequence.items():
        if not isinstance(sequence, list) or not sequence:
            raise PolicySmokeError(f"M6 contact sequence for {foot_name!r} must be a non-empty list")
        if any(not isinstance(segment, list) or len(segment) < 2 for segment in sequence):
            raise PolicySmokeError(f"M6 contact sequence for {foot_name!r} has an invalid segment")
        total_duration_s = sum(float(segment[1]) for segment in sequence)
        if total_duration_s < required_duration_s:
            sequence[-1][1] = float(sequence[-1][1]) + (required_duration_s - total_duration_s)
    env_cfg.contact_generator_config = copied_contact_generator_config
    return env_cfg


def _seed_everything(seed: int, env: Any, torch: Any) -> None:
    """Seed the host libraries and the instantiated RAMBO environment."""

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env.seed(seed)


def _validate_policy_environment(env: Any, contract: Any, torch: Any, num_envs: int) -> Any:
    """Check fixed released-policy dimensions before creating the CRL2 runner."""

    observations, _ = env.get_observations()
    if tuple(observations.shape) != (num_envs, contract.observation_dim):
        raise PolicySmokeError(
            "Unexpected initial policy-observation shape: "
            f"expected ({num_envs}, {contract.observation_dim}), got {tuple(observations.shape)}"
        )
    if int(env.num_actions) != contract.action_dim:
        raise PolicySmokeError(
            f"Unexpected action dimension: expected {contract.action_dim}, got {env.num_actions}"
        )
    _tensor_metrics(observations, "initial_policy_observations")
    return observations


def _sha256_path(path: Path) -> str:
    """Hash an evidence file without holding its contents in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_summary(
    output_dir: Path, summary: dict[str, Any], evidence_paths: tuple[Path, ...] = ()
) -> None:
    """Persist a machine-verifiable M6 evidence bundle and all streamed traces."""

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths = (summary_path, *[path for path in evidence_paths if path.is_file()])
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{_sha256_path(path)}  {path.name}\n" for path in paths), encoding="utf-8"
    )
    print(f"SUMMARY_PATH={summary_path}", flush=True)


def main() -> int:
    parser, app_launcher_type = _build_parser()
    args_cli = parser.parse_args()
    from rambo.utils.physx import validate_rambo_visualizer_args

    visualizer_selection = validate_rambo_visualizer_args(parser, args_cli, sys.argv[1:])
    if args_cli.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args_cli.steps <= 0:
        parser.error("--steps must be positive")
    output_dir = args_cli.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "passed": False,
        "task": TASK,
        "num_envs": args_cli.num_envs,
        "requested_steps": args_cli.steps,
        "seed": args_cli.seed,
        "viz": visualizer_selection,
        "eula_acceptance": "explicit_user_consent",
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
    }
    simulation_app = None
    env = None
    runtime: dict[str, Any] | None = None
    trace_writer: _StateTraceWriter | None = None
    memory_samples: list[dict[str, int]] = []
    rollout_progress: dict[str, Any] = {
        "steps_completed": 0,
        "reward_total": 0.0,
        "terminal_count_all_envs": 0,
        "terminal_count_by_step": [],
        "total_env_steps": args_cli.num_envs * args_cli.steps,
    }
    state_progress: dict[str, Any] = {
        "data_source": {
            "base_height": "base_env.scene.articulations['robot'].data.root_link_pos_w.torch[:, 2]",
            "orientation_error": "norm(robot.data.projected_gravity_b.torch - (0, 0, -1))",
            "joint_velocity": "robot.data.joint_vel.torch",
            "joint_velocity_limit": "robot.data.joint_vel_limits.torch",
        },
        "records": 0,
        "min_base_height_m": float("inf"),
        "max_orientation_error": 0.0,
        "max_joint_velocity_ratio_to_sim_limit": 0.0,
    }
    exit_code = 0
    try:
        app_launcher = app_launcher_type(args_cli)
        simulation_app = app_launcher.app
        runtime = _runtime_imports()
        contract = runtime["contract_for_task"](TASK)
        checkpoint_path = args_cli.checkpoint.expanduser().resolve()
        checkpoint = runtime["load_verified_checkpoint"](checkpoint_path, contract)
        summary["checkpoint"] = {
            "path": checkpoint_path,
            "sha256": runtime["sha256_file"](checkpoint_path),
            "iteration": int(checkpoint["iteration"]),
            "normalizer_count": int(checkpoint["obs_normalizer_count"]),
        }

        env_cfg = runtime["parse_env_cfg"](
            TASK,
            device=args_cli.device or "cuda:0",
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        # This direct assignment is intentionally repeated even though the RAMBO
        # registry already enforces it: every RAMBO instantiation is explicitly
        # configured for PhysX at its immediate call site.
        runtime["configure_physx"](env_cfg)
        _configure_deterministic_smoke(env_cfg, args_cli.seed, args_cli.steps)
        configured_physics = _fqn(env_cfg.sim.physics)
        if configured_physics != runtime["PHYSX_CFG_FQN"]:
            raise PolicySmokeError(
                f"RAMBO smoke configuration is not PhysxCfg: {configured_physics}"
            )
        if getattr(env_cfg.sim, "use_newton_actuators", None) is not False:
            raise PolicySmokeError("RAMBO smoke requires use_newton_actuators=False")
        summary["configured_physics"] = {
            "cfg": configured_physics,
            "use_newton_actuators": False,
            "use_fabric": bool(env_cfg.sim.use_fabric),
        }
        summary["deterministic_config"] = {
            "events": env_cfg.events,
            "randomize_initial_state": bool(env_cfg.randomize_initial_state),
            "obs_noise": bool(env_cfg.obs_noise),
            "randomize_episode_progress": bool(env_cfg.randomize_episode_progress),
            "enable_sampled_velocity_commands": bool(env_cfg.enable_sampled_velocity_commands),
            "enable_sampled_pos_commands": bool(env_cfg.enable_sampled_pos_commands),
            "enable_sampled_force_commands": bool(env_cfg.enable_sampled_force_commands),
            "enable_rgb_camera": bool(env_cfg.enable_rgb_camera),
            "episode_length_s": float(env_cfg.episode_length_s),
            "contact_sequence_total_duration_s": {
                str(foot_name): sum(float(segment[1]) for segment in sequence)
                for foot_name, sequence in env_cfg.contact_generator_config["contact_sequence"].items()
            },
        }

        agent_cfg = runtime["load_cfg_from_registry"](TASK, "crl2_cfg_entry_point")
        if not isinstance(agent_cfg, dict):
            raise PolicySmokeError("RAMBO CRL2 agent configuration must be a YAML dictionary")
        agent_cfg["seed"] = args_cli.seed
        agent_cfg["general"]["num_envs"] = args_cli.num_envs

        env = runtime["Crl2VecEnvWrapper"](runtime["gym"].make(TASK, cfg=env_cfg))
        summary["backend_before"] = runtime["assert_physx_environment"](env)
        base_env = env.unwrapped
        if getattr(base_env.cfg, "enable_rgb_camera", None) is not False:
            raise PolicySmokeError("M6 policy smoke requires the RAMBO RGB camera to remain disabled")
        sensors = getattr(base_env.scene, "sensors", None)
        if isinstance(sensors, Mapping) and "front_camera" in sensors:
            raise PolicySmokeError("M6 policy smoke unexpectedly instantiated a front camera")
        _seed_everything(args_cli.seed, env, runtime["torch"])
        observations, _ = env.reset()
        _validate_policy_environment(env, contract, runtime["torch"], args_cli.num_envs)

        runner = runtime["PPO"](
            task=TASK,
            env=env,
            agent_cfg=agent_cfg,
            train=False,
            device=env.device,
        )
        runtime["restore_runner"](runner, checkpoint, load_values=False, verify=True)
        if runner.current_iteration != contract.iteration:
            raise PolicySmokeError(
                f"CRL2 runner iteration mismatch: expected {contract.iteration}, got {runner.current_iteration}"
            )
        if runner.obs_normalizer.count != contract.normalizer_count:
            raise PolicySmokeError(
                "CRL2 empirical-normalizer count did not restore from the verified checkpoint"
            )
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        trace_writer = _StateTraceWriter(output_dir / "public_state_trace.jsonl")
        memory_samples.append(_memory_sample(0, runtime["torch"], env.device))
        summary["rollout"] = rollout_progress
        summary["public_state_monitor"] = state_progress
        action_min = float("inf")
        action_max = float("-inf")
        action_abs_max = 0.0
        action_mean_total = 0.0
        for step in range(args_cli.steps):
            with runtime["torch"].inference_mode():
                actions = policy(observations)
                if tuple(actions.shape) != (args_cli.num_envs, contract.action_dim):
                    raise PolicySmokeError(
                        f"Unexpected policy action shape at step {step}: {tuple(actions.shape)}"
                    )
                action_metrics = _tensor_metrics(actions, f"policy_actions_{step}")
                action_min = min(action_min, action_metrics["min"])
                action_max = max(action_max, action_metrics["max"])
                action_abs_max = max(action_abs_max, action_metrics["abs_max"])
                action_mean_total += action_metrics["mean"]
                observations, rewards, dones, _ = env.step(actions)
                _tensor_metrics(observations, f"policy_observations_{step}")
                _tensor_metrics(rewards, f"rewards_{step}")
                rollout_progress["reward_total"] += float(rewards.detach().sum().cpu())
                step_terminal_count = int(runtime["torch"].count_nonzero(dones).detach().cpu())
                rollout_progress["terminal_count_all_envs"] += step_terminal_count
                rollout_progress["terminal_count_by_step"].append(step_terminal_count)

                state_record, state_extrema = _collect_public_robot_state(
                    base_env,
                    num_envs=args_cli.num_envs,
                    step=step + 1,
                    contract=contract,
                    torch=runtime["torch"],
                )
                trace_writer.write(state_record)
                state_progress["records"] = trace_writer.records
                state_progress["min_base_height_m"] = min(
                    state_progress["min_base_height_m"], state_extrema["min_base_height"]
                )
                state_progress["max_orientation_error"] = max(
                    state_progress["max_orientation_error"], state_extrema["max_orientation_error"]
                )
                state_progress["max_joint_velocity_ratio_to_sim_limit"] = max(
                    state_progress["max_joint_velocity_ratio_to_sim_limit"],
                    state_extrema["max_joint_velocity_ratio"],
                )
                previous_joint_max = state_progress.get("max_abs_joint_velocity_per_joint")
                if previous_joint_max is None:
                    state_progress["max_abs_joint_velocity_per_joint"] = state_extrema[
                        "max_abs_joint_velocity_per_joint"
                    ]
                    state_progress["joint_velocity_limit_per_joint"] = state_extrema[
                        "joint_velocity_limit_per_joint"
                    ]
                    state_progress["joint_names"] = state_extrema["joint_names"]
                else:
                    state_progress["max_abs_joint_velocity_per_joint"] = [
                        max(float(previous), float(current))
                        for previous, current in zip(
                            previous_joint_max, state_extrema["max_abs_joint_velocity_per_joint"]
                        )
                    ]
                rollout_progress["steps_completed"] = step + 1

                if (step + 1) % MEMORY_MONITOR_INTERVAL_STEPS == 0:
                    memory_samples.append(_memory_sample(step + 1, runtime["torch"], env.device))
                if step_terminal_count:
                    raise PolicySmokeError(
                        "One or more environments terminated or timed out during smoke "
                        f"at step {step}: {step_terminal_count}/{args_cli.num_envs}"
                    )

        summary["backend_after"] = runtime["assert_physx_environment"](env)
        summary["policy"] = {
            "observation_shape": [args_cli.num_envs, contract.observation_dim],
            "action_shape": [args_cli.num_envs, contract.action_dim],
            "runner_iteration": int(runner.current_iteration),
            "normalizer_count": int(runner.obs_normalizer.count),
            "action_min": action_min,
            "action_max": action_max,
            "action_abs_max": action_abs_max,
            "mean_action_mean": action_mean_total / args_cli.steps,
        }
        summary["memory_monitor"] = {
            "interval_steps": MEMORY_MONITOR_INTERVAL_STEPS,
            "samples": memory_samples,
            "analysis": _memory_analysis(memory_samples, args_cli.steps),
        }
        summary["passed"] = True
        print("RAMBO_PHYSX_QUADRUPED_POLICY_SMOKE_SUCCESS", flush=True)
    except BaseException as error:
        exit_code = 1
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        traceback.print_exception(type(error), error, error.__traceback__)
    finally:
        evidence_paths: tuple[Path, ...] = ()
        if trace_writer is not None:
            try:
                trace_writer.close()
                evidence_paths = (trace_writer.path,)
                state_progress["trace_file"] = trace_writer.path.name
                state_progress["trace_sha256"] = _sha256_path(trace_writer.path)
                state_progress["records"] = trace_writer.records
            except BaseException as trace_error:
                exit_code = 1
                summary["state_trace_close_error"] = f"{type(trace_error).__name__}: {trace_error}"
                if "traceback" not in summary:
                    summary["traceback"] = traceback.format_exc()
        if state_progress["records"] == 0:
            state_progress["min_base_height_m"] = None
        summary.setdefault("rollout", rollout_progress)
        summary.setdefault("public_state_monitor", state_progress)
        summary.setdefault(
            "memory_monitor",
            {
                "interval_steps": MEMORY_MONITOR_INTERVAL_STEPS,
                "samples": memory_samples,
                "analysis": {"evaluated": False, "reason": "rollout did not complete"},
            },
        )
        if env is not None and runtime is not None and "backend_after" not in summary:
            try:
                summary["backend_after"] = runtime["assert_physx_environment"](env)
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
        summary["passed"] = exit_code == 0
        summary["shutdown_mode"] = "isaacsim_default_fast_shutdown"
        _write_summary(output_dir, summary, evidence_paths)

    if simulation_app is not None:
        simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
