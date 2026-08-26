#!/usr/bin/env python3
"""Replay the released RAMBO biped policy through an explicit PhysX-only gate.

The launcher intentionally has a small, fixed surface: it accepts only the
released biped checkpoint, a finite number of policy steps, and the audited
``--viz none`` / ``--viz kit`` choices.  Its artifact records both the
configured and the *actual* physics manager before and after replay so a
passing result cannot be confused with merely having a PhysX wheel installed.

The biped's 19.6-second validation gait phase is part of the checkpoint
contract.  It is configured independently of episode time and the contact
schedule is extended before the environment is instantiated.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import traceback
from typing import Any, Callable


TASK = "Isaac-RAMBO-Biped-Go2-v0"
DEFAULT_STEPS = 16
BIPED_VALIDATION_PHASE_OFFSET_S = 19.6
MIN_VALIDATION_DURATION_S = 31.0
ARTIFACT_ROOT = Path("/workspace/runs/audit/isaac60/M9")
MEMORY_MONITOR_INTERVAL_STEPS = 100
MEMORY_ANALYSIS_START_STEP = 1000
MEMORY_WINDOW_STEPS = 500
MIB = 1024 * 1024
GPU_SLOPE_LIMIT_BYTES_PER_100_STEPS = 1 * MIB
GPU_MEDIAN_DELTA_LIMIT_BYTES = 128 * MIB
RSS_SLOPE_LIMIT_BYTES_PER_100_STEPS = 4 * MIB
RSS_MEDIAN_DELTA_LIMIT_BYTES = 512 * MIB


class PolicySmokeError(RuntimeError):
    """Raised when a released biped-policy replay violates its contract."""


def _build_parser() -> tuple[argparse.ArgumentParser, type]:
    """Create the CLI parser without importing the simulator at module import time."""

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:  # pragma: no cover - target-runtime guard.
        raise RuntimeError(
            "physx_biped_policy_smoke.py requires the RAMBO Isaac Lab target runtime"
        ) from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Released biped model_4000.pt; its SHA-256 allow-list is checked before loading.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of synchronized biped environments to replay.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Finite policy steps to execute.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed Python/NumPy/Torch/environment seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=f"New M9 evidence directory below {ARTIFACT_ROOT}.",
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
    """Import target-runtime dependencies after AppLauncher started Kit."""

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
    from rambo.validation.rollout import configure_validation_cfg

    if not rambo.register_tasks():
        raise PolicySmokeError("RAMBO task registration requires an active Isaac Sim Kit application")
    return {
        "Crl2VecEnvWrapper": Crl2VecEnvWrapper,
        "PHYSX_CFG_FQN": PHYSX_CFG_FQN,
        "PPO": PPO,
        "assert_physx_environment": assert_physx_environment,
        "configure_physx": configure_physx,
        "configure_validation_cfg": configure_validation_cfg,
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
    """Return a stable fully-qualified class identity for artifact evidence."""

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
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _tensor_metrics(tensor: Any, label: str) -> dict[str, Any]:
    """Fail closed on non-finite tensor data and return compact evidence."""

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
    """Read an Isaac Lab 3 public ProxyArray via its documented Torch view."""

    tensor = getattr(value, "torch", None)
    if not isinstance(tensor, torch.Tensor):
        raise PolicySmokeError(
            f"Required public Isaac Lab data field {name} does not expose a torch.Tensor .torch view"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise PolicySmokeError(f"Required public Isaac Lab data field {name} contains NaN or Inf")
    return tensor


def _public_robot_data(base_env: Any) -> Any:
    """Resolve the biped robot through RAMBO's public scene registry."""

    scene = getattr(base_env, "scene", None)
    articulations = getattr(scene, "articulations", None)
    if not isinstance(articulations, Mapping):
        raise PolicySmokeError("Required public base_env.scene.articulations is unavailable")
    robot = articulations.get("robot")
    if robot is None:
        raise PolicySmokeError("Required public base_env.scene.articulations['robot'] is unavailable")
    data = getattr(robot, "data", None)
    if data is None:
        raise PolicySmokeError("Required public Isaac Lab Articulation.data is unavailable")
    return data


def _collect_public_robot_safety(
    base_env: Any,
    *,
    contract: Any,
    num_envs: int,
    step: int,
    torch: Any,
) -> dict[str, Any]:
    """Check biped posture and joints solely through public Isaac Lab data."""

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

    if tuple(root_link_pos_w.shape) != (num_envs, 3):
        raise PolicySmokeError(
            "Public robot.data.root_link_pos_w shape mismatch: "
            f"expected ({num_envs}, 3), got {tuple(root_link_pos_w.shape)}"
        )
    if tuple(projected_gravity_b.shape) != (num_envs, 3):
        raise PolicySmokeError(
            "Public robot.data.projected_gravity_b shape mismatch: "
            f"expected ({num_envs}, 3), got {tuple(projected_gravity_b.shape)}"
        )
    if joint_vel.ndim != 2 or joint_vel.shape[0] != num_envs:
        raise PolicySmokeError(
            "Public robot.data.joint_vel shape mismatch: "
            f"expected ({num_envs}, joint_count), got {tuple(joint_vel.shape)}"
        )
    if tuple(joint_vel_limits.shape) != tuple(joint_vel.shape):
        raise PolicySmokeError(
            "Public robot.data.joint_vel_limits must have the same shape as joint_vel: "
            f"limits={tuple(joint_vel_limits.shape)}, velocity={tuple(joint_vel.shape)}"
        )
    if not bool(torch.all(joint_vel_limits > 0.0)):
        raise PolicySmokeError("Public robot.data.joint_vel_limits must be strictly positive")

    gravity_target = torch.tensor(
        contract.gravity_target, device=projected_gravity_b.device, dtype=projected_gravity_b.dtype
    )
    base_height = root_link_pos_w[:, 2]
    orientation_error = torch.linalg.vector_norm(projected_gravity_b - gravity_target, dim=-1)
    joint_velocity_ratio = joint_vel.abs() / joint_vel_limits
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
        raise PolicySmokeError("Public robot.data.joint_names is unavailable or mismatched")
    return {
        "min_base_height_m": min_base_height,
        "max_orientation_error": max_orientation_error,
        "max_joint_velocity_ratio": max_joint_velocity_ratio,
        "max_abs_joint_velocity_per_joint_rad_s": joint_vel.abs().amax(dim=0).detach().cpu().tolist(),
        "joint_velocity_limit_per_joint_rad_s": joint_vel_limits[0].detach().cpu().tolist(),
        "joint_names": list(joint_names),
    }


def _duration_for_steps(steps: int) -> float:
    """Return a no-timeout duration for the 100 Hz finite biped rollout."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    return max(MIN_VALIDATION_DURATION_S, steps * 0.01 + 1.0)


def _validate_contact_sequence_duration(env_cfg: Any, required_duration_s: float) -> dict[str, float]:
    """Ensure all contact programs cover the fixed phase plus the rollout."""

    config = getattr(env_cfg, "contact_generator_config", None)
    sequence_by_foot = config.get("contact_sequence") if isinstance(config, Mapping) else None
    if not isinstance(sequence_by_foot, Mapping) or not sequence_by_foot:
        raise PolicySmokeError("Biped smoke requires a non-empty contact_generator_config.contact_sequence")
    durations: dict[str, float] = {}
    for foot, sequence in sequence_by_foot.items():
        if not isinstance(sequence, list) or not sequence:
            raise PolicySmokeError(f"Biped contact sequence for {foot!r} is empty")
        try:
            duration = sum(float(segment[1]) for segment in sequence)
        except (IndexError, TypeError, ValueError) as exc:
            raise PolicySmokeError(f"Biped contact sequence for {foot!r} is malformed") from exc
        if duration + 1.0e-9 < required_duration_s:
            raise PolicySmokeError(
                f"Biped contact sequence for {foot!r} covers {duration:.6f}s, "
                f"not required {required_duration_s:.6f}s"
            )
        durations[str(foot)] = duration
    return durations


def _configure_biped_smoke(
    env_cfg: Any,
    *,
    configure_validation_cfg: Callable[..., Any],
    seed: int,
    steps: int,
) -> dict[str, Any]:
    """Apply deterministic biped validation without treating 19.6 s as episode time."""

    duration_s = _duration_for_steps(steps)
    configure_validation_cfg(
        env_cfg,
        duration_s=duration_s,
        enable_rgb_camera=False,
        contact_phase_offset_s=BIPED_VALIDATION_PHASE_OFFSET_S,
    )
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = seed
    if hasattr(env_cfg, "enable_rgb_camera"):
        env_cfg.enable_rgb_camera = False

    phase_offset_s = float(getattr(env_cfg, "contact_phase_offset_s", float("nan")))
    if not math.isclose(phase_offset_s, BIPED_VALIDATION_PHASE_OFFSET_S, rel_tol=0.0, abs_tol=1.0e-9):
        raise PolicySmokeError(
            "Biped validation must use the checkpoint-approved 19.6-second gait phase; "
            f"got {phase_offset_s!r}"
        )
    episode_length_s = float(getattr(env_cfg, "episode_length_s", float("nan")))
    if not math.isclose(episode_length_s, duration_s, rel_tol=0.0, abs_tol=1.0e-9):
        raise PolicySmokeError(
            f"Biped validation episode length must be {duration_s:.6f}s, got {episode_length_s!r}"
        )
    if getattr(env_cfg, "events", None) is not None:
        raise PolicySmokeError("Biped validation must disable startup and interval randomization events")

    required_contact_duration_s = duration_s + phase_offset_s
    contact_durations_s = _validate_contact_sequence_duration(env_cfg, required_contact_duration_s)
    return {
        "episode_length_s": episode_length_s,
        "phase_offset_s": phase_offset_s,
        "required_contact_duration_s": required_contact_duration_s,
        "contact_durations_s": contact_durations_s,
        "randomize_initial_state": bool(getattr(env_cfg, "randomize_initial_state", False)),
        "randomize_episode_progress": bool(getattr(env_cfg, "randomize_episode_progress", False)),
        "observation_noise": bool(getattr(env_cfg, "obs_noise", False)),
        "enable_rgb_camera": bool(getattr(env_cfg, "enable_rgb_camera", False)),
    }


def _seed_everything(seed: int, env: Any, torch: Any) -> None:
    """Seed host libraries and the instantiated RAMBO vector environment."""

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env.seed(seed)


def _validate_policy_environment(env: Any, contract: Any, torch: Any, num_envs: int) -> Any:
    """Verify the released biped 435/18 input/output contract before replay."""

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


def _rss_bytes() -> int:
    """Read Linux resident-set size from procfs without a third-party monitor."""

    statm_path = Path("/proc/self/statm")
    try:
        resident_pages = int(statm_path.read_text(encoding="utf-8").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, OSError, ValueError) as exc:
        raise PolicySmokeError("RSS monitoring requires readable Linux /proc/self/statm") from exc


def _memory_sample(step: int, torch: Any, device: Any) -> dict[str, int]:
    """Capture the approved CUDA-allocation and RSS memory signals."""

    if not torch.cuda.is_available():
        raise PolicySmokeError("Biped memory monitoring requires a CUDA-capable PyTorch runtime")
    return {
        "step": step,
        "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device=device)),
        "rss_bytes": _rss_bytes(),
    }


def _linear_slope_bytes_per_100_steps(samples: list[dict[str, int]], field: str) -> float:
    """Return ordinary-least-squares allocation growth over 100 policy steps."""

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
    """Enforce the approved long-run GPU/RSS leak thresholds fail-closed."""

    result: dict[str, Any] = {
        "sample_interval_steps": MEMORY_MONITOR_INTERVAL_STEPS,
        "samples": samples,
    }
    if total_steps < MEMORY_ANALYSIS_START_STEP:
        result.update(
            {
                "evaluated": False,
                "reason": f"requires steps >= {MEMORY_ANALYSIS_START_STEP}",
            }
        )
        return result

    post_warmup = [sample for sample in samples if sample["step"] >= MEMORY_ANALYSIS_START_STEP]
    first_window = [sample for sample in samples if sample["step"] <= MEMORY_WINDOW_STEPS]
    last_window_start = total_steps - MEMORY_WINDOW_STEPS
    last_window = [sample for sample in samples if sample["step"] > last_window_start]
    if len(post_warmup) < 2 or not first_window or not last_window:
        raise PolicySmokeError("Insufficient periodic memory samples for fail-closed M9 analysis")

    result.update(
        {
            "evaluated": True,
            "analysis_start_step": MEMORY_ANALYSIS_START_STEP,
            "first_window_steps": [sample["step"] for sample in first_window],
            "last_window_steps": [sample["step"] for sample in last_window],
        }
    )
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


def _ensure_m9_artifact_directory(output_dir: Path) -> Path:
    """Keep migration evidence under the declared M9 artifact root."""

    output_dir = output_dir.expanduser().resolve()
    artifact_root = ARTIFACT_ROOT.resolve()
    try:
        output_dir.relative_to(artifact_root)
    except ValueError as exc:
        raise PolicySmokeError(
            f"--output-dir must be below the M9 artifact root {artifact_root}: {output_dir}"
        ) from exc
    if output_dir.exists():
        raise PolicySmokeError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    return output_dir


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    """Persist machine-verifiable M9 evidence and a SHA-256 sidecar."""

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    (output_dir / "checksums.sha256").write_text(f"{digest}  summary.json\n", encoding="utf-8")
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
    try:
        output_dir = _ensure_m9_artifact_directory(args_cli.output_dir)
    except PolicySmokeError as exc:
        parser.error(str(exc))

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
    exit_code = 0
    try:
        app_launcher = app_launcher_type(args_cli)
        simulation_app = app_launcher.app
        runtime = _runtime_imports()
        contract = runtime["contract_for_task"](TASK)
        if contract.observation_dim != 435 or contract.action_dim != 18:
            raise PolicySmokeError(
                "The biped checkpoint contract must remain exactly observation=435/action=18"
            )
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
        # Keep the backend choice local to every RAMBO runtime entry point:
        # a future registry/default change may never select a different backend.
        runtime["configure_physx"](env_cfg)
        summary["validation_config"] = _configure_biped_smoke(
            env_cfg,
            configure_validation_cfg=runtime["configure_validation_cfg"],
            seed=args_cli.seed,
            steps=args_cli.steps,
        )
        configured_physics = _fqn(env_cfg.sim.physics)
        if configured_physics != runtime["PHYSX_CFG_FQN"]:
            raise PolicySmokeError(f"RAMBO biped configuration is not PhysxCfg: {configured_physics}")
        if getattr(env_cfg.sim, "use_newton_actuators", None) is not False:
            raise PolicySmokeError("RAMBO biped smoke requires use_newton_actuators=False")
        summary["configured_physics"] = {
            "cfg": configured_physics,
            "use_newton_actuators": False,
            "use_fabric": bool(env_cfg.sim.use_fabric),
        }

        agent_cfg = runtime["load_cfg_from_registry"](TASK, "crl2_cfg_entry_point")
        if not isinstance(agent_cfg, dict):
            raise PolicySmokeError("RAMBO CRL2 agent configuration must load as a YAML dictionary")
        agent_cfg["seed"] = args_cli.seed
        agent_cfg["general"]["num_envs"] = args_cli.num_envs

        env = runtime["Crl2VecEnvWrapper"](runtime["gym"].make(TASK, cfg=env_cfg))
        summary["backend_before"] = runtime["assert_physx_environment"](env)
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
            raise PolicySmokeError("CRL2 normalizer count did not restore from the verified checkpoint")
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        action_min = float("inf")
        action_max = float("-inf")
        action_abs_max = 0.0
        action_mean_total = 0.0
        reward_total = 0.0
        terminal_count = 0
        safety_extrema: dict[str, Any] | None = None
        memory_samples = [_memory_sample(0, runtime["torch"], env.device)]
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
                reward_total += float(rewards.detach().sum().cpu())
                step_terminal_count = int(runtime["torch"].count_nonzero(dones).detach().cpu())
                terminal_count += step_terminal_count
                if step_terminal_count:
                    raise PolicySmokeError(
                        "One or more biped environments terminated or timed out at "
                        f"step {step}: {step_terminal_count}/{args_cli.num_envs}"
                    )
                safety = _collect_public_robot_safety(
                    env.unwrapped,
                    contract=contract,
                    num_envs=args_cli.num_envs,
                    step=step,
                    torch=runtime["torch"],
                )
                if safety_extrema is None:
                    safety_extrema = safety
                else:
                    safety_extrema["min_base_height_m"] = min(
                        safety_extrema["min_base_height_m"], safety["min_base_height_m"]
                    )
                    safety_extrema["max_orientation_error"] = max(
                        safety_extrema["max_orientation_error"], safety["max_orientation_error"]
                    )
                    safety_extrema["max_joint_velocity_ratio"] = max(
                        safety_extrema["max_joint_velocity_ratio"], safety["max_joint_velocity_ratio"]
                    )
                    safety_extrema["max_abs_joint_velocity_per_joint_rad_s"] = [
                        max(previous, current)
                        for previous, current in zip(
                            safety_extrema["max_abs_joint_velocity_per_joint_rad_s"],
                            safety["max_abs_joint_velocity_per_joint_rad_s"],
                            strict=True,
                        )
                    ]
                completed_steps = step + 1
                if (
                    completed_steps % MEMORY_MONITOR_INTERVAL_STEPS == 0
                    or completed_steps == args_cli.steps
                ):
                    memory_samples.append(_memory_sample(completed_steps, runtime["torch"], env.device))

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
        summary["rollout"] = {
            "steps_completed": args_cli.steps,
            "reward_total": reward_total,
            "terminal_count_all_envs": terminal_count,
            "total_env_steps": args_cli.num_envs * args_cli.steps,
            "public_safety_extrema": safety_extrema,
        }
        summary["memory"] = _memory_analysis(memory_samples, args_cli.steps)
        summary["passed"] = True
        print("RAMBO_PHYSX_BIPED_POLICY_SMOKE_SUCCESS", flush=True)
    except BaseException as error:
        exit_code = 1
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        traceback.print_exception(type(error), error, error.__traceback__)
    finally:
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
        _write_summary(output_dir, summary)

    if simulation_app is not None:
        simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
