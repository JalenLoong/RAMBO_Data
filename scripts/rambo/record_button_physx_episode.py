#!/usr/bin/env python3
"""Record a deterministic RAMBO Button episode under the actual PhysX manager.

The default is the 30-second acceptance episode: exactly 3000 policy actions,
observations, and post-step states at 100 Hz, plus 375 fresh RGB frames at the
RAMBO camera's exact eight-action-step cadence.  ``--short-smoke`` is an
explicitly labelled structural gate for a smaller multiple of eight; it never
claims to be the 30-second acceptance artifact.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import traceback
from typing import Any


def _build_parser() -> tuple[argparse.ArgumentParser, type[Any]]:
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as error:  # pragma: no cover - target-runtime guard.
        raise RuntimeError("Run this recorder through scripts/rambo/run.sh") from error

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted released quadruped model_2000.pt")
    parser.add_argument("--seed", type=int, default=42, help="Fixed Python/NumPy/Torch/environment seed")
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Policy steps; 3000 is the required 30-second acceptance episode",
    )
    parser.add_argument(
        "--short-smoke",
        action="store_true",
        help="Permit a non-3000 structural smoke run; the artifact is explicitly non-acceptance",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New evidence directory")
    parser.add_argument(
        "--disable-fabric",
        "--disable_fabric",
        dest="disable_fabric",
        action="store_true",
        help="Use USD I/O instead of Fabric",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser, AppLauncher


def _runtime_imports() -> dict[str, Any]:
    """Delay all simulator imports until AppLauncher has initialized Kit."""

    from rambo.torch_runtime import ensure_cuda_linalg_loaded

    ensure_cuda_linalg_loaded()
    import imageio.v3 as imageio
    import numpy as np
    import torch
    import gymnasium as gym
    import rambo
    from crl2.algorithms import PPO
    from rambo.rl import Crl2VecEnvWrapper
    from rambo.tasks.common.camera import camera_update_interval_steps
    from rambo.utils.physx import (
        PHYSX_CFG_FQN,
        assert_physx_environment,
        configure_physx,
    )
    from rambo.utils.registry import load_cfg_from_registry, parse_env_cfg
    from rambo.validation.button_episode_artifact import (
        ACCEPTANCE_STEPS,
        ACTION_DIMENSION,
        ACTION_SCHEMA_VERSION,
        BUTTON_HOLD_STEPS,
        BUTTON_PRESS_THRESHOLD_M,
        BUTTON_REBOUND_THRESHOLD_M,
        CAMERA_INTERVAL_ACTION_STEPS,
        DECIMATION,
        M10_PROMPT,
        MOTION_MINIMUM_DELTA_M,
        OBSERVATION_DIMENSION,
        OBSERVATION_SCHEMA_VERSION,
        PHYSICS_DT_NS,
        POLICY_DT_NS,
        QUADRUPED_CHECKPOINT_SHA256,
        SCHEMA_VERSION,
        TASK_ID,
        TARGET_ISAACLAB_COMMIT,
        TARGET_ISAACLAB_TAG,
        TARGET_ISAACSIM_VERSION,
        TRACE_FILENAMES,
        jsonable,
        sha256_file,
        write_checksums,
        write_json,
    )
    from rambo.validation.checkpoints import contract_for_task, load_verified_checkpoint, restore_runner

    if not rambo.register_tasks():
        raise ButtonEpisodeRecorderError("RAMBO task registration requires an active Isaac Sim Kit application")
    return {
        "ACCEPTANCE_STEPS": ACCEPTANCE_STEPS,
        "ACTION_DIMENSION": ACTION_DIMENSION,
        "ACTION_SCHEMA_VERSION": ACTION_SCHEMA_VERSION,
        "BUTTON_HOLD_STEPS": BUTTON_HOLD_STEPS,
        "BUTTON_PRESS_THRESHOLD_M": BUTTON_PRESS_THRESHOLD_M,
        "BUTTON_REBOUND_THRESHOLD_M": BUTTON_REBOUND_THRESHOLD_M,
        "CAMERA_INTERVAL_ACTION_STEPS": CAMERA_INTERVAL_ACTION_STEPS,
        "Crl2VecEnvWrapper": Crl2VecEnvWrapper,
        "DECIMATION": DECIMATION,
        "M10_PROMPT": M10_PROMPT,
        "MOTION_MINIMUM_DELTA_M": MOTION_MINIMUM_DELTA_M,
        "OBSERVATION_DIMENSION": OBSERVATION_DIMENSION,
        "OBSERVATION_SCHEMA_VERSION": OBSERVATION_SCHEMA_VERSION,
        "PHYSICS_DT_NS": PHYSICS_DT_NS,
        "PHYSX_CFG_FQN": PHYSX_CFG_FQN,
        "POLICY_DT_NS": POLICY_DT_NS,
        "PPO": PPO,
        "QUADRUPED_CHECKPOINT_SHA256": QUADRUPED_CHECKPOINT_SHA256,
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "TASK_ID": TASK_ID,
        "TARGET_ISAACLAB_COMMIT": TARGET_ISAACLAB_COMMIT,
        "TARGET_ISAACLAB_TAG": TARGET_ISAACLAB_TAG,
        "TARGET_ISAACSIM_VERSION": TARGET_ISAACSIM_VERSION,
        "TRACE_FILENAMES": TRACE_FILENAMES,
        "assert_physx_environment": assert_physx_environment,
        "camera_update_interval_steps": camera_update_interval_steps,
        "configure_physx": configure_physx,
        "contract_for_task": contract_for_task,
        "gym": gym,
        "imageio": imageio,
        "jsonable": jsonable,
        "load_cfg_from_registry": load_cfg_from_registry,
        "load_verified_checkpoint": load_verified_checkpoint,
        "np": np,
        "parse_env_cfg": parse_env_cfg,
        "restore_runner": restore_runner,
        "sha256_file": sha256_file,
        "torch": torch,
        "write_checksums": write_checksums,
        "write_json": write_json,
    }


class ButtonEpisodeRecorderError(RuntimeError):
    """Raised when an episode cannot meet the strict PhysX evidence contract."""


# Keep this gate byte-for-byte compatible with the approved long-run RAMBO
# policy-smoke methodology.  A full M10 artifact is only accepted when the
# post-warm-up allocation slope and first/last window medians stay below these
# thresholds; an explicitly short smoke retains samples but cannot evaluate it.
MEMORY_MONITOR_INTERVAL_STEPS = 100
MEMORY_ANALYSIS_START_STEP = 1000
MEMORY_WINDOW_STEPS = 500
MIB = 1024 * 1024
GPU_SLOPE_LIMIT_BYTES_PER_100_STEPS = 1 * MIB
GPU_MEDIAN_DELTA_LIMIT_BYTES = 128 * MIB
RSS_SLOPE_LIMIT_BYTES_PER_100_STEPS = 4 * MIB
RSS_MEDIAN_DELTA_LIMIT_BYTES = 512 * MIB


class _JsonlWriter:
    """Stream one evidence class without retaining the full episode in memory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("x", encoding="utf-8")
        self.records = 0

    def write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        self.records += 1

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


class _RgbCpuStaging:
    """One reusable host buffer for all GPU-to-PNG RGB copies in an episode."""

    def __init__(self, source: Any, torch: Any) -> None:
        if source.ndim != 3 or tuple(source.shape) != (480, 640, 3):
            raise ButtonEpisodeRecorderError(f"RGB staging source has invalid shape: {tuple(source.shape)}")
        if source.dtype != torch.uint8:
            raise ButtonEpisodeRecorderError(f"RGB staging source has invalid dtype: {source.dtype}")
        self._torch = torch
        try:
            self._buffer = torch.empty(
                tuple(source.shape), dtype=source.dtype, device="cpu", pin_memory=True
            )
        except RuntimeError as error:
            raise ButtonEpisodeRecorderError("M10 RGB staging requires one pinned CPU buffer") from error
        self._array = self._buffer.numpy()
        self._copies = 0

    def copy_from(self, source: Any) -> Any:
        if tuple(source.shape) != tuple(self._buffer.shape) or source.dtype != self._buffer.dtype:
            raise ButtonEpisodeRecorderError("RGB source changed shape or dtype after staging allocation")
        self._buffer.copy_(source, non_blocking=True)
        if source.is_cuda:
            self._torch.cuda.synchronize(device=source.device)
        self._copies += 1
        return self._array

    def evidence(self) -> dict[str, Any]:
        return {
            "reused": True,
            "allocations": 1,
            "copies": self._copies,
            "shape": list(self._buffer.shape),
            "dtype": str(self._buffer.dtype),
            "pinned_memory": bool(self._buffer.is_pinned()),
        }


class _ButtonAcceptanceAccumulator:
    """Keep only scalar acceptance extrema while evidence streams to JSONL."""

    def __init__(
        self,
        *,
        press_threshold_m: float,
        hold_steps: int,
        rebound_threshold_m: float,
        minimum_motion_m: float,
    ) -> None:
        self._press_threshold_m = press_threshold_m
        self._hold_steps = hold_steps
        self._rebound_threshold_m = rebound_threshold_m
        self._minimum_motion_m = minimum_motion_m
        self._initial_base: tuple[float, float, float] | None = None
        self._initial_fl: tuple[float, float, float] | None = None
        self._max_base_delta = 0.0
        self._max_fl_delta = 0.0
        self._max_displacement = 0.0
        self._current_pressed_steps = 0
        self._max_pressed_steps = 0
        self._success_seen = False
        self._first_success_step: int | None = None
        self._first_rebound_after_success_step: int | None = None
        self._min_displacement_after_success: float | None = None

    @staticmethod
    def _single_vector(value: Any, label: str) -> tuple[float, float, float]:
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list) or len(value[0]) != 3:
            raise ButtonEpisodeRecorderError(f"{label} must have one XYZ vector")
        vector = tuple(float(component) for component in value[0])
        if not all(math.isfinite(component) for component in vector):
            raise ButtonEpisodeRecorderError(f"{label} contains non-finite values")
        return vector  # type: ignore[return-value]

    @staticmethod
    def _single_scalar(value: Any, label: str) -> float:
        if not isinstance(value, list) or len(value) != 1:
            raise ButtonEpisodeRecorderError(f"{label} must have one scalar")
        result = float(value[0])
        if not math.isfinite(result):
            raise ButtonEpisodeRecorderError(f"{label} contains a non-finite value")
        return result

    @staticmethod
    def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
        return math.sqrt(sum((first - second) ** 2 for first, second in zip(left, right)))

    def update(self, state_record: dict[str, Any]) -> None:
        robot = state_record["robot"]
        button = state_record["button"]
        action_step = int(state_record["action_step"])
        base_position = self._single_vector(robot["root_link_pos_w_m"], "root_link_pos_w_m")
        fl_position = self._single_vector(robot["fl_foot_link_pos_w_m"], "fl_foot_link_pos_w_m")
        if self._initial_base is None:
            self._initial_base = base_position
            self._initial_fl = fl_position
        else:
            self._max_base_delta = max(self._max_base_delta, self._distance(base_position, self._initial_base))
            assert self._initial_fl is not None
            self._max_fl_delta = max(self._max_fl_delta, self._distance(fl_position, self._initial_fl))
        displacement = self._single_scalar(button["displacement_m"], "button_displacement_m")
        if displacement < 0.0:
            raise ButtonEpisodeRecorderError("Button displacement cannot be negative")
        self._max_displacement = max(self._max_displacement, displacement)
        if displacement >= self._press_threshold_m:
            self._current_pressed_steps += 1
        else:
            self._current_pressed_steps = 0
        self._max_pressed_steps = max(self._max_pressed_steps, self._current_pressed_steps)
        step_success = any(bool(value) for value in button["success"])
        if self._success_seen:
            self._min_displacement_after_success = (
                displacement
                if self._min_displacement_after_success is None
                else min(self._min_displacement_after_success, displacement)
            )
            if displacement <= self._rebound_threshold_m and self._first_rebound_after_success_step is None:
                self._first_rebound_after_success_step = action_step
        if step_success and self._first_success_step is None:
            self._first_success_step = action_step
        self._success_seen |= step_success

    def evidence(self) -> dict[str, Any]:
        return {
            "button": {
                "press_threshold_m": self._press_threshold_m,
                "hold_steps_required": self._hold_steps,
                "rebound_threshold_m": self._rebound_threshold_m,
                "max_displacement_m": self._max_displacement,
                "max_consecutive_pressed_action_steps": self._max_pressed_steps,
                "first_success_action_step": self._first_success_step,
                "first_rebound_after_success_action_step": self._first_rebound_after_success_step,
                "min_displacement_after_success_m": self._min_displacement_after_success,
            },
            "motion": {
                "minimum_required_delta_m": self._minimum_motion_m,
                "baseline_action_step": 1,
                "max_base_root_link_displacement_m": self._max_base_delta,
                "max_fl_foot_link_displacement_m": self._max_fl_delta,
            },
        }

    def assert_full_acceptance(self) -> None:
        evidence = self.evidence()
        button = evidence["button"]
        motion = evidence["motion"]
        if button["max_displacement_m"] < self._press_threshold_m:
            raise ButtonEpisodeRecorderError("Button episode never reached the required 12 mm press threshold")
        if button["max_consecutive_pressed_action_steps"] < self._hold_steps:
            raise ButtonEpisodeRecorderError("Button episode did not hold the press for five action steps")
        if (
            button["min_displacement_after_success_m"] is None
            or button["min_displacement_after_success_m"] > self._rebound_threshold_m
        ):
            raise ButtonEpisodeRecorderError("Button episode did not rebound to 2 mm or less after success")
        if motion["max_base_root_link_displacement_m"] < self._minimum_motion_m:
            raise ButtonEpisodeRecorderError("Button episode lacks the required 0.05 m base-motion evidence")
        if motion["max_fl_foot_link_displacement_m"] < self._minimum_motion_m:
            raise ButtonEpisodeRecorderError("Button episode lacks the required 0.05 m FL-foot-motion evidence")


def _rss_bytes() -> int:
    """Read Linux resident-set size through procfs for the M10 memory gate."""

    statm_path = Path("/proc/self/statm")
    try:
        resident_pages = int(statm_path.read_text(encoding="utf-8").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, OSError, ValueError) as error:
        raise ButtonEpisodeRecorderError(
            "M10 memory monitoring requires readable Linux /proc/self/statm"
        ) from error


def _memory_sample(step: int, torch: Any, device: Any, *, policy_dt_ns: int) -> dict[str, int]:
    """Capture a synchronized 100-step GPU/RSS allocation sample."""

    if not torch.cuda.is_available():
        raise ButtonEpisodeRecorderError("M10 memory monitoring requires a CUDA-capable PyTorch runtime")
    torch.cuda.synchronize(device=device)
    return {
        "action_step": step,
        "timestamp_ns": step * policy_dt_ns,
        "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device=device)),
        "rss_bytes": _rss_bytes(),
    }


def _linear_slope_bytes_per_100_steps(samples: list[dict[str, int]], field: str) -> float:
    """Calculate ordinary-least-squares allocation growth per 100 action steps."""

    if len(samples) < 2:
        raise ButtonEpisodeRecorderError(f"Need at least two memory samples to calculate {field} slope")
    xs = [float(sample["action_step"]) for sample in samples]
    ys = [float(sample[field]) for sample in samples]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0.0:
        raise ButtonEpisodeRecorderError(f"Memory sample steps are degenerate for {field} slope")
    return 100.0 * sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def _memory_analysis(samples: list[dict[str, int]], total_steps: int) -> dict[str, Any]:
    """Apply the approved M10 post-step-1000 GPU/RSS leak gate."""

    if total_steps < MEMORY_ANALYSIS_START_STEP:
        return {
            "evaluated": False,
            "reason": f"requires action_steps >= {MEMORY_ANALYSIS_START_STEP}",
        }
    post_warmup = [sample for sample in samples if sample["action_step"] >= MEMORY_ANALYSIS_START_STEP]
    first_window = [sample for sample in samples if sample["action_step"] <= MEMORY_WINDOW_STEPS]
    last_window_start = total_steps - MEMORY_WINDOW_STEPS
    last_window = [sample for sample in samples if sample["action_step"] > last_window_start]
    if len(post_warmup) < 2 or not first_window or not last_window:
        raise ButtonEpisodeRecorderError("Insufficient periodic memory samples for fail-closed M10 analysis")

    result: dict[str, Any] = {
        "evaluated": True,
        "analysis_start_action_step": MEMORY_ANALYSIS_START_STEP,
        "first_window_action_steps": [sample["action_step"] for sample in first_window],
        "last_window_action_steps": [sample["action_step"] for sample in last_window],
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
            "first_500_action_step_median_bytes": first_median,
            "last_500_action_step_median_bytes": last_median,
            "median_delta_bytes": median_delta,
            "median_delta_mib": median_delta / MIB,
            "median_delta_limit_mib": median_delta_limit / MIB,
        }
        if slope > slope_limit:
            raise ButtonEpisodeRecorderError(
                f"{label} allocation slope {slope / MIB:.6f} MiB/100 steps exceeds "
                f"{slope_limit / MIB:.6f} MiB/100 steps"
            )
        if median_delta > median_delta_limit:
            raise ButtonEpisodeRecorderError(
                f"{label} first/last 500-step median growth {median_delta / MIB:.6f} MiB exceeds "
                f"{median_delta_limit / MIB:.6f} MiB"
            )
    return result


def _fqn(value: Any) -> str:
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _proxy_tensor(value: Any, label: str, torch: Any) -> Any:
    tensor = getattr(value, "torch", None)
    if not isinstance(tensor, torch.Tensor):
        raise ButtonEpisodeRecorderError(f"{label} must expose Isaac Lab 3 public ProxyArray .torch data")
    if not bool(torch.isfinite(tensor).all()):
        raise ButtonEpisodeRecorderError(f"{label} contains NaN or Inf")
    return tensor


def _tensor_list(value: Any, label: str, torch: Any, *, finite: bool = True) -> list[Any]:
    if not isinstance(value, torch.Tensor):
        raise ButtonEpisodeRecorderError(f"{label} must be a torch.Tensor, got {type(value).__name__}")
    if finite and not bool(torch.isfinite(value).all()):
        raise ButtonEpisodeRecorderError(f"{label} contains NaN or Inf")
    return value.detach().cpu().tolist()


def _runtime_metadata(torch: Any) -> dict[str, Any]:
    """Capture Python, CUDA, GPU, and driver evidence without selecting a backend."""

    if not torch.cuda.is_available():
        raise ButtonEpisodeRecorderError("Button episode recording requires CUDA for Isaac Sim RTX rendering")
    device_index = torch.cuda.current_device()
    driver = "unavailable"
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_line = process.stdout.strip().splitlines()
        if process.returncode == 0 and first_line and first_line[0].strip():
            driver = first_line[0].strip()
        elif process.stderr.strip():
            driver = f"unavailable:{process.stderr.strip()}"
    except (OSError, subprocess.SubprocessError) as error:  # pragma: no cover - diagnostic fallback.
        driver = f"unavailable:{type(error).__name__}"
    try:
        isaacsim_version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError as error:
        raise ButtonEpisodeRecorderError("Pinned Isaac Sim package metadata is unavailable") from error
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda or "unknown"),
        "gpu": str(torch.cuda.get_device_name(device_index)),
        "gpu_index": int(device_index),
        "driver": driver,
        "isaacsim_version": isaacsim_version,
    }


def _git_output(cwd: Path, *arguments: str) -> str:
    try:
        process = subprocess.run(
            ["git", "-C", str(cwd), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ButtonEpisodeRecorderError(f"Cannot inspect Git provenance in {cwd}") from error
    if process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip() or "unknown Git error"
        raise ButtonEpisodeRecorderError(f"Cannot inspect Git provenance in {cwd}: {message}")
    return process.stdout.strip()


def _collect_provenance(
    runtime_metadata: dict[str, Any],
    *,
    expected_isaaclab_tag: str,
    expected_isaaclab_commit: str,
    expected_isaacsim_version: str,
) -> dict[str, Any]:
    """Collect and fail-close the exact RAMBO/Isaac Lab/Isaac Sim identities."""

    rambo_root = Path(__file__).resolve().parents[2]
    rambo_git_root = Path(_git_output(rambo_root, "rev-parse", "--show-toplevel"))
    rambo_commit = _git_output(rambo_git_root, "rev-parse", "HEAD")
    rambo_dirty = bool(_git_output(rambo_git_root, "status", "--porcelain"))
    try:
        import isaaclab
    except ModuleNotFoundError as error:  # pragma: no cover - only callable after Kit launch.
        raise ButtonEpisodeRecorderError("Pinned Isaac Lab package is unavailable after Kit launch") from error
    isaaclab_module_path = Path(isaaclab.__file__).resolve()
    isaaclab_root = Path(_git_output(isaaclab_module_path.parent, "rev-parse", "--show-toplevel"))
    isaaclab_tag = _git_output(isaaclab_root, "describe", "--tags", "--exact-match")
    isaaclab_commit = _git_output(isaaclab_root, "rev-parse", "HEAD")
    if isaaclab_tag != expected_isaaclab_tag:
        raise ButtonEpisodeRecorderError(
            f"Isaac Lab tag is {isaaclab_tag!r}, expected {expected_isaaclab_tag!r}"
        )
    if isaaclab_commit != expected_isaaclab_commit:
        raise ButtonEpisodeRecorderError(
            f"Isaac Lab commit is {isaaclab_commit!r}, expected {expected_isaaclab_commit!r}"
        )
    if runtime_metadata.get("isaacsim_version") != expected_isaacsim_version:
        raise ButtonEpisodeRecorderError(
            "Isaac Sim version is "
            f"{runtime_metadata.get('isaacsim_version')!r}, expected {expected_isaacsim_version!r}"
        )
    return {
        "rambo_git_commit": rambo_commit,
        "rambo_git_dirty": rambo_dirty,
        "isaaclab": {"tag": isaaclab_tag, "commit": isaaclab_commit},
        "isaacsim_version": expected_isaacsim_version,
        "gpu": {"model": runtime_metadata["gpu"], "driver": runtime_metadata["driver"]},
    }


def _seed_everything(seed: int, env: Any, np: Any, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)


def _configure_button_cfg(env_cfg: Any, *, seed: int, steps: int, configure_physx: Any) -> None:
    """Make the recorder deterministic and explicitly select PhysX at this call site."""

    configure_physx(env_cfg)
    env_cfg.seed = seed
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
    ):
        if hasattr(env_cfg, attribute):
            setattr(env_cfg, attribute, False)
    if not hasattr(env_cfg, "enable_rgb_camera") or not hasattr(env_cfg, "front_camera"):
        raise ButtonEpisodeRecorderError("Button task config does not expose the front RGB camera contract")
    env_cfg.enable_rgb_camera = True
    env_cfg.scene.lazy_sensor_update = True
    policy_dt_s = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), steps * policy_dt_s + 1.0)


def _resolve_public_robot(base_env: Any) -> Any:
    scene = getattr(base_env, "scene", None)
    articulations = getattr(scene, "articulations", None)
    if not hasattr(articulations, "get"):
        raise ButtonEpisodeRecorderError("Button scene has no public articulation registry")
    robot = articulations.get("robot")
    if robot is None:
        raise ButtonEpisodeRecorderError("Button scene is missing public 'robot' articulation")
    return robot


def _post_state_record(base_env: Any, *, action_step: int, terminal: Any, torch: Any) -> tuple[dict[str, Any], bool, bool]:
    """Capture post-step robot/contact/button state exclusively through public data views."""

    robot = _resolve_public_robot(base_env)
    data = robot.data
    root_pos = _proxy_tensor(data.root_link_pos_w, "robot.data.root_link_pos_w", torch)
    root_quat = _proxy_tensor(data.root_link_quat_w, "robot.data.root_link_quat_w", torch)
    joint_pos = _proxy_tensor(data.joint_pos, "robot.data.joint_pos", torch)
    joint_vel = _proxy_tensor(data.joint_vel, "robot.data.joint_vel", torch)
    body_link_pos = _proxy_tensor(data.body_link_pos_w, "robot.data.body_link_pos_w", torch)
    body_names = getattr(data, "body_names", None)
    if not isinstance(body_names, (list, tuple)) or "FL_foot" not in body_names:
        raise ButtonEpisodeRecorderError("robot.data.body_names does not expose FL_foot")
    fl_foot_index = body_names.index("FL_foot")
    if body_link_pos.ndim != 3 or body_link_pos.shape[1] <= fl_foot_index or body_link_pos.shape[2] != 3:
        raise ButtonEpisodeRecorderError("robot.data.body_link_pos_w has no valid FL_foot XYZ position")
    fl_foot_pos = body_link_pos[:, fl_foot_index, :]
    sensor = getattr(base_env.scene, "sensors", {}).get("contact_sensor")
    if sensor is None:
        raise ButtonEpisodeRecorderError("Button scene is missing public contact_sensor")
    contact_history = _proxy_tensor(
        sensor.data.net_forces_w_history,
        "contact_sensor.data.net_forces_w_history",
        torch,
    )
    contact_force_max = torch.linalg.vector_norm(contact_history, dim=-1).amax(dim=(1, 2)).detach().cpu().tolist()
    displacement = base_env.button_displacement
    success = base_env.button_success
    released = base_env.button_released
    ready = base_env.manipulator_ready
    success_list = _tensor_list(success, "button_success", torch, finite=False)
    released_list = _tensor_list(released, "button_released", torch, finite=False)
    return (
        {
            "action_step": action_step,
            "timestamp_ns": action_step * 10_000_000,
            "robot": {
                "root_link_pos_w_m": _tensor_list(root_pos, "root_link_pos_w", torch),
                "root_link_quat_w_xyzw": _tensor_list(root_quat, "root_link_quat_w", torch),
                "fl_foot_body_name": "FL_foot",
                "fl_foot_link_pos_w_m": _tensor_list(fl_foot_pos, "fl_foot_link_pos_w", torch),
                "joint_pos_rad": _tensor_list(joint_pos, "joint_pos", torch),
                "joint_vel_rad_s": _tensor_list(joint_vel, "joint_vel", torch),
            },
            "contact_sensor": {
                "net_force_history_shape": list(contact_history.shape),
                "max_force_norm_n": contact_force_max,
            },
            "button": {
                "displacement_m": _tensor_list(displacement, "button_displacement", torch),
                "success": success_list,
                "released": released_list,
                "manipulator_ready": _tensor_list(ready, "manipulator_ready", torch, finite=False),
            },
            "terminal": _tensor_list(terminal, "terminal", torch, finite=False),
        },
        any(bool(value) for value in success_list),
        any(bool(value) for value in released_list),
    )


def _camera_frame_id(camera: Any, torch: Any) -> int:
    frame = _proxy_tensor(camera.frame, "front_camera.frame", torch)
    if frame.numel() != 1:
        raise ButtonEpisodeRecorderError(f"front_camera.frame must have one element, got {tuple(frame.shape)}")
    return int(frame.detach().reshape(-1)[0].cpu().item())


def _capture_rgb_if_fresh(
    *,
    camera: Any,
    action_step: int,
    output_dir: Path,
    last_frame_id: int,
    rgb_writer: _JsonlWriter,
    rgb_staging: _RgbCpuStaging | None,
    imageio: Any,
    torch: Any,
    expected_physics_ticks: int,
) -> tuple[int, _RgbCpuStaging | None]:
    """Capture exactly one new frame when the public camera frame counter advances."""

    data = camera.data  # Camera data must be read before frame for lazy refresh.
    current_frame_id = _camera_frame_id(camera, torch)
    if current_frame_id < last_frame_id:
        raise ButtonEpisodeRecorderError(
            f"Front camera frame regressed from {last_frame_id} to {current_frame_id}"
        )
    if current_frame_id == last_frame_id:
        return last_frame_id, rgb_staging
    if current_frame_id != last_frame_id + 1:
        raise ButtonEpisodeRecorderError(
            f"Front camera frame skipped from {last_frame_id} to {current_frame_id}"
        )
    output = getattr(data, "output", None)
    if not isinstance(output, dict) or "rgb" not in output:
        raise ButtonEpisodeRecorderError("Front camera does not expose public RGB output")
    rgb = _proxy_tensor(output["rgb"], "front_camera.data.output['rgb']", torch)
    if tuple(rgb.shape) != (1, 480, 640, 3):
        raise ButtonEpisodeRecorderError(f"Unexpected RGB shape: {tuple(rgb.shape)}")
    rgb_float = rgb.float()
    mean = float(rgb_float.mean().detach().cpu())
    std = float(rgb_float.std().detach().cpu())
    if mean <= 2.0 or std <= 1.0:
        raise ButtonEpisodeRecorderError(f"RGB is black or lacks variation: mean={mean}, std={std}")
    ticks = _proxy_tensor(camera.rambo_physics_ticks, "front_camera.rambo_physics_ticks", torch)
    if ticks.numel() != 1:
        raise ButtonEpisodeRecorderError(f"front_camera.rambo_physics_ticks shape is invalid: {tuple(ticks.shape)}")
    physics_ticks = int(ticks.detach().reshape(-1)[0].cpu().item())
    if physics_ticks != expected_physics_ticks:
        raise ButtonEpisodeRecorderError(
            f"Camera tick mismatch at action {action_step}: got {physics_ticks}, expected {expected_physics_ticks}"
        )
    frame_index = rgb_writer.records + 1
    relative_path = Path("rgb") / f"{frame_index:06d}.png"
    if rgb_staging is None:
        rgb_staging = _RgbCpuStaging(rgb[0], torch)
    imageio.imwrite(output_dir / relative_path, rgb_staging.copy_from(rgb[0]))
    rgb_writer.write(
        {
            "frame_index": frame_index,
            "action_step": action_step,
            "timestamp_ns": action_step * 10_000_000,
            "physics_ticks": physics_ticks,
            "camera_frame_id": current_frame_id,
            "path": str(relative_path),
            "shape": list(rgb.shape),
            "dtype": str(rgb.dtype),
            "mean": mean,
            "std": std,
        }
    )
    return current_frame_id, rgb_staging


def _apply_loco_manip_schedule(
    base_env: Any,
    *,
    action_step: int,
    leg_target: Any,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    """Mirror the accepted non-GUI Button loco-manip command timing exactly."""

    dtype = leg_target.dtype
    device = leg_target.device
    base_velocity = torch.zeros((1, 3), dtype=dtype, device=device)
    leg_velocity = torch.zeros((1, 3), dtype=dtype, device=device)
    if 110 <= action_step < 270:
        base_velocity[:, 0] = 0.4
    elif 300 <= action_step < 550:
        leg_velocity[:, 2] = 0.10
    elif 550 <= action_step < 810:
        leg_velocity[:, 0] = 0.12
    elif 810 <= action_step < 910:
        leg_velocity[:, 0] = -0.12

    displacement = base_env.button_displacement
    success = base_env.button_success
    ready = base_env.manipulator_ready
    if bool(torch.any(displacement >= base_env.cfg.button_contact_guard_m).item()):
        base_velocity[:, 0].clamp_(max=0.0)
    if bool(torch.any(success).item()):
        base_velocity.zero_()
        leg_velocity[:, 0].clamp_(max=0.0)
    if bool(torch.all(ready).item()):
        lower = torch.tensor([0.1934, 0.0, 0.0], dtype=dtype, device=device).view(1, 3)
        upper = torch.tensor([0.50, 0.20, 0.40], dtype=dtype, device=device).view(1, 3)
        leg_target = torch.clamp(leg_target + leg_velocity * float(base_env.step_dt), lower, upper)
    force = torch.zeros((1, 3), dtype=dtype, device=device)
    base_env.set_loco_manip_commands(
        base_velocity=base_velocity,
        fl_position=leg_target,
        fl_force=force,
    )
    return leg_target, {
        "base_velocity": base_velocity.detach().cpu().tolist(),
        "fl_position": leg_target.detach().cpu().tolist(),
        "fl_force": force.detach().cpu().tolist(),
    }


def _configure_output_dir(path: Path) -> Path:
    output_dir = path.expanduser().resolve()
    if output_dir.exists():
        raise ButtonEpisodeRecorderError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "rgb").mkdir()
    return output_dir


def main() -> int:
    parser, app_launcher_type = _build_parser()
    args = parser.parse_args()
    from rambo.utils.physx import validate_rambo_visualizer_args
    from rambo.validation.button_episode_artifact import (
        ACCEPTANCE_STEPS,
        ACTION_DIMENSION,
        ACTION_SCHEMA_VERSION,
        BUTTON_HOLD_STEPS,
        BUTTON_PRESS_THRESHOLD_M,
        BUTTON_REBOUND_THRESHOLD_M,
        CAMERA_INTERVAL_ACTION_STEPS,
        DECIMATION,
        M10_PROMPT,
        MOTION_MINIMUM_DELTA_M,
        OBSERVATION_DIMENSION,
        OBSERVATION_SCHEMA_VERSION,
        PHYSICS_DT_NS,
        POLICY_DT_NS,
        QUADRUPED_CHECKPOINT_SHA256,
        SCHEMA_VERSION,
        TASK_ID,
        TARGET_ISAACLAB_COMMIT,
        TARGET_ISAACLAB_TAG,
        TARGET_ISAACSIM_VERSION,
        TRACE_FILENAMES,
        write_checksums,
        write_json,
    )

    visualizer = validate_rambo_visualizer_args(parser, args, sys.argv[1:])
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.steps <= 0 or args.steps % CAMERA_INTERVAL_ACTION_STEPS:
        parser.error("--steps must be a positive multiple of 8 for the exact RGB cadence")
    if args.steps != ACCEPTANCE_STEPS and not args.short_smoke:
        parser.error("non-3000 --steps requires --short-smoke and is never an acceptance artifact")
    if args.steps == ACCEPTANCE_STEPS and args.short_smoke:
        parser.error("--short-smoke cannot label the required 3000-step acceptance episode")
    args.enable_cameras = True
    output_dir = _configure_output_dir(args.output_dir)

    timing = {
        "physics_dt_s": PHYSICS_DT_NS / 1_000_000_000,
        "control_decimation": DECIMATION,
        "control_dt_s": POLICY_DT_NS / 1_000_000_000,
        "camera_update_period_s": 0.08,
        "camera_rate_hz": 12.5,
        "camera_interval_physics_ticks": 40,
        "camera_interval_action_steps": CAMERA_INTERVAL_ACTION_STEPS,
    }
    policy_schema = {
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

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_ID,
        "prompt": M10_PROMPT,
        "requested_action_steps": args.steps,
        "short_smoke": bool(args.short_smoke),
        "timebase": {
            "policy_dt_ns": POLICY_DT_NS,
            "physics_dt_ns": PHYSICS_DT_NS,
            "decimation": DECIMATION,
            "camera_interval_action_steps": CAMERA_INTERVAL_ACTION_STEPS,
        },
        "timing": timing,
        "policy_schema": policy_schema,
        "files": {**TRACE_FILENAMES, "rgb_directory": "rgb"},
    }
    write_json(output_dir / "manifest.json", manifest)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "passed": False,
        "task": TASK_ID,
        "prompt": M10_PROMPT,
        "seed": args.seed,
        "requested_action_steps": args.steps,
        "expected_rgb_frames": args.steps // CAMERA_INTERVAL_ACTION_STEPS,
        "short_smoke": bool(args.short_smoke),
        "viz": visualizer[0],
        "enable_cameras": True,
        "eula_acceptance": "explicit_user_consent",
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "timing": timing,
        "policy_schema": policy_schema,
    }
    memory_samples: list[dict[str, int]] = []
    summary["memory_monitor"] = {
        "interval_action_steps": MEMORY_MONITOR_INTERVAL_STEPS,
        "analysis_start_action_step": MEMORY_ANALYSIS_START_STEP,
        "first_last_window_action_steps": MEMORY_WINDOW_STEPS,
        "limits": {
            "gpu_slope_mib_per_100_steps": GPU_SLOPE_LIMIT_BYTES_PER_100_STEPS / MIB,
            "gpu_median_delta_mib": GPU_MEDIAN_DELTA_LIMIT_BYTES / MIB,
            "rss_slope_mib_per_100_steps": RSS_SLOPE_LIMIT_BYTES_PER_100_STEPS / MIB,
            "rss_median_delta_mib": RSS_MEDIAN_DELTA_LIMIT_BYTES / MIB,
        },
        "samples": memory_samples,
        "analysis": {"evaluated": False, "reason": "recording did not complete"},
    }
    app = None
    env = None
    runtime: dict[str, Any] | None = None
    writers: dict[str, _JsonlWriter] = {}
    exit_code = 0
    error: BaseException | None = None
    success_seen = False
    release_after_success_seen = False
    terminal_count = 0
    last_frame_id: int | None = None
    rgb_staging: _RgbCpuStaging | None = None
    acceptance = _ButtonAcceptanceAccumulator(
        press_threshold_m=BUTTON_PRESS_THRESHOLD_M,
        hold_steps=BUTTON_HOLD_STEPS,
        rebound_threshold_m=BUTTON_REBOUND_THRESHOLD_M,
        minimum_motion_m=MOTION_MINIMUM_DELTA_M,
    )

    try:
        app = app_launcher_type(args)
        simulation_app = app.app
        runtime = _runtime_imports()
        torch = runtime["torch"]
        summary["runtime"] = _runtime_metadata(torch)
        provenance = _collect_provenance(
            summary["runtime"],
            expected_isaaclab_tag=TARGET_ISAACLAB_TAG,
            expected_isaaclab_commit=TARGET_ISAACLAB_COMMIT,
            expected_isaacsim_version=TARGET_ISAACSIM_VERSION,
        )
        manifest["provenance"] = provenance
        summary["provenance"] = provenance
        if not args.short_smoke and provenance["rambo_git_dirty"] is not False:
            raise ButtonEpisodeRecorderError(
                "Full M10 acceptance recording requires a clean RAMBO Git HEAD; commit source changes first"
            )
        contract = runtime["contract_for_task"](runtime["TASK_ID"])
        if contract.sha256 != QUADRUPED_CHECKPOINT_SHA256:
            raise ButtonEpisodeRecorderError("Button task contract is not pinned to the released quadruped allowlist SHA")
        if contract.observation_dim != OBSERVATION_DIMENSION or contract.action_dim != ACTION_DIMENSION:
            raise ButtonEpisodeRecorderError("Button task contract dimensions are not the required 405/18 policy schema")
        checkpoint_path = args.checkpoint.expanduser().resolve()
        checkpoint = runtime["load_verified_checkpoint"](checkpoint_path, contract)
        checkpoint_sha256 = runtime["sha256_file"](checkpoint_path)
        if checkpoint_sha256 != QUADRUPED_CHECKPOINT_SHA256:
            raise ButtonEpisodeRecorderError("Checkpoint is not the released quadruped allowlist SHA")
        summary["checkpoint"] = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "allowlist_sha256": QUADRUPED_CHECKPOINT_SHA256,
            "iteration": int(checkpoint["iteration"]),
            "normalizer_count": int(checkpoint["obs_normalizer_count"]),
            "observation_dim": int(contract.observation_dim),
            "action_dim": int(contract.action_dim),
        }

        env_cfg = runtime["parse_env_cfg"](
            runtime["TASK_ID"],
            device=args.device or "cuda:0",
            num_envs=1,
            use_fabric=not args.disable_fabric,
        )
        _configure_button_cfg(
            env_cfg,
            seed=args.seed,
            steps=args.steps,
            configure_physx=runtime["configure_physx"],
        )
        configured_fqn = _fqn(env_cfg.sim.physics)
        if configured_fqn != runtime["PHYSX_CFG_FQN"]:
            raise ButtonEpisodeRecorderError(f"Recorder config is not PhysxCfg: {configured_fqn}")
        if getattr(env_cfg.sim, "use_newton_actuators", None) is not False:
            raise ButtonEpisodeRecorderError("Recorder requires use_newton_actuators=False")
        summary["configured_physics"] = {
            "cfg": configured_fqn,
            "use_newton_actuators": False,
            "episode_length_s": float(env_cfg.episode_length_s),
            "camera_enabled": bool(env_cfg.enable_rgb_camera),
        }

        agent_cfg = runtime["load_cfg_from_registry"](runtime["TASK_ID"], "crl2_cfg_entry_point")
        if not isinstance(agent_cfg, dict):
            raise ButtonEpisodeRecorderError("RAMBO CRL2 agent configuration must be a dictionary")
        agent_cfg["seed"] = args.seed
        agent_cfg["general"]["num_envs"] = 1

        env = runtime["Crl2VecEnvWrapper"](runtime["gym"].make(runtime["TASK_ID"], cfg=env_cfg))
        summary["backend_before"] = runtime["assert_physx_environment"](env)
        _seed_everything(args.seed, env, runtime["np"], torch)
        # Crl2VecEnvWrapper owns a zero-argument reset; its public seed method
        # above has already seeded the wrapped environment deterministically.
        observations, _ = env.reset()
        if tuple(observations.shape) != (1, contract.observation_dim):
            raise ButtonEpisodeRecorderError(
                f"Unexpected initial observation shape: {tuple(observations.shape)}"
            )
        if not bool(torch.isfinite(observations).all()):
            raise ButtonEpisodeRecorderError("Initial policy observation contains NaN or Inf")
        if observations.dtype != torch.float32:
            raise ButtonEpisodeRecorderError(f"Policy observation dtype is not torch.float32: {observations.dtype}")
        base_env = env.unwrapped
        camera = getattr(base_env, "front_camera", None)
        if camera is None or base_env.scene.sensors.get("front_camera") is not camera:
            raise ButtonEpisodeRecorderError("Button task did not expose the public front RGB camera")
        physics_dt_s = float(base_env.cfg.sim.dt)
        policy_dt_s = physics_dt_s * int(base_env.cfg.decimation)
        if not math.isclose(physics_dt_s, PHYSICS_DT_NS / 1_000_000_000, rel_tol=0.0, abs_tol=1.0e-12):
            raise ButtonEpisodeRecorderError(f"Unexpected physics timebase: {physics_dt_s}")
        if int(base_env.cfg.decimation) != DECIMATION:
            raise ButtonEpisodeRecorderError(f"Unexpected control decimation: {base_env.cfg.decimation}")
        if abs(policy_dt_s * 1_000_000_000 - POLICY_DT_NS) > 1.0e-6:
            raise ButtonEpisodeRecorderError(f"Unexpected policy timebase: {policy_dt_s}")
        camera_update_period_s = float(camera.cfg.update_period)
        if not math.isclose(camera_update_period_s, 0.08, rel_tol=0.0, abs_tol=1.0e-12):
            raise ButtonEpisodeRecorderError(f"Unexpected front camera period: {camera_update_period_s}")
        cadence_ticks = runtime["camera_update_interval_steps"](camera_update_period_s, physics_dt_s)
        if cadence_ticks != 40 or cadence_ticks // int(base_env.cfg.decimation) != CAMERA_INTERVAL_ACTION_STEPS:
            raise ButtonEpisodeRecorderError("Front camera does not expose the required exact 40-tick/eight-step cadence")
        actual_timing = {
            "physics_dt_s": physics_dt_s,
            "control_decimation": int(base_env.cfg.decimation),
            "control_dt_s": policy_dt_s,
            "camera_update_period_s": camera_update_period_s,
            "camera_rate_hz": 1.0 / camera_update_period_s,
            "camera_interval_physics_ticks": cadence_ticks,
            "camera_interval_action_steps": cadence_ticks // int(base_env.cfg.decimation),
        }
        if actual_timing != timing:
            raise ButtonEpisodeRecorderError(f"Actual timing differs from M10 contract: {actual_timing}")
        manifest["timing"] = actual_timing
        summary["timing"] = actual_timing
        if not math.isclose(
            float(base_env.cfg.button_press_threshold_m), BUTTON_PRESS_THRESHOLD_M, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ButtonEpisodeRecorderError("Button task does not expose the required 12 mm press threshold")
        if int(base_env.cfg.button_hold_steps) != BUTTON_HOLD_STEPS:
            raise ButtonEpisodeRecorderError("Button task does not expose the required five-step hold")
        if not math.isclose(
            float(base_env.cfg.button_release_threshold_m),
            BUTTON_REBOUND_THRESHOLD_M,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ButtonEpisodeRecorderError("Button task does not expose the required 2 mm rebound threshold")
        camera.data
        last_frame_id = _camera_frame_id(camera, torch)
        initial_ticks = _proxy_tensor(camera.rambo_physics_ticks, "front_camera.rambo_physics_ticks", torch)
        if int(initial_ticks.detach().reshape(-1)[0].cpu().item()) != 0:
            raise ButtonEpisodeRecorderError("Front camera exact tick counter was not reset before recording")

        runner = runtime["PPO"](
            task=runtime["TASK_ID"], env=env, agent_cfg=agent_cfg, train=False, device=env.device
        )
        runtime["restore_runner"](runner, checkpoint, load_values=False, verify=True)
        policy = runner.get_inference_policy(device=base_env.device)
        for key, filename in runtime["TRACE_FILENAMES"].items():
            writers[key] = _JsonlWriter(output_dir / filename)

        leg_target = torch.tensor([[0.1934, 0.142, 0.05]], dtype=observations.dtype, device=base_env.device)
        memory_samples.append(
            _memory_sample(0, torch, base_env.device, policy_dt_ns=runtime["POLICY_DT_NS"])
        )
        for action_step in range(1, args.steps + 1):
            with torch.inference_mode():
                leg_target, command = _apply_loco_manip_schedule(
                    base_env, action_step=action_step, leg_target=leg_target, torch=torch
                )
                action = policy(observations)
                if tuple(action.shape) != (1, contract.action_dim):
                    raise ButtonEpisodeRecorderError(
                        f"Unexpected policy action shape at step {action_step}: {tuple(action.shape)}"
                    )
                if not bool(torch.isfinite(action).all()):
                    raise ButtonEpisodeRecorderError(f"Policy action is non-finite at step {action_step}")
                if action.dtype != torch.float32:
                    raise ButtonEpisodeRecorderError(f"Policy action dtype is not torch.float32: {action.dtype}")
                writers["actions"].write(
                    {
                        "action_step": action_step,
                        "start_timestamp_ns": (action_step - 1) * POLICY_DT_NS,
                        "end_timestamp_ns": action_step * POLICY_DT_NS,
                        "policy_action": _tensor_list(action, "policy_action", torch),
                        "command": command,
                    }
                )
                observations, rewards, dones, _ = env.step(action)
                if not bool(torch.isfinite(observations).all()):
                    raise ButtonEpisodeRecorderError(f"Observation is non-finite at step {action_step}")
                if not bool(torch.isfinite(rewards).all()):
                    raise ButtonEpisodeRecorderError(f"Reward is non-finite at step {action_step}")
                writers["observations"].write(
                    {
                        "action_step": action_step,
                        "timestamp_ns": action_step * POLICY_DT_NS,
                        "policy_observation": _tensor_list(observations, "policy_observation", torch),
                        "reward": _tensor_list(rewards, "reward", torch),
                    }
                )
                state_record, step_success, step_released = _post_state_record(
                    base_env, action_step=action_step, terminal=dones, torch=torch
                )
                writers["post_states"].write(state_record)
                acceptance.update(state_record)
                release_after_success_seen |= success_seen and step_released
                success_seen |= step_success
                step_terminal_count = int(torch.count_nonzero(dones).detach().cpu())
                terminal_count += step_terminal_count
                if step_terminal_count:
                    raise ButtonEpisodeRecorderError(
                        f"Button episode terminated at action step {action_step}: {step_terminal_count}/1"
                    )
                assert last_frame_id is not None
                last_frame_id, rgb_staging = _capture_rgb_if_fresh(
                    camera=camera,
                    action_step=action_step,
                    output_dir=output_dir,
                    last_frame_id=last_frame_id,
                    rgb_writer=writers["rgb_frames"],
                    rgb_staging=rgb_staging,
                    imageio=runtime["imageio"],
                    torch=torch,
                    expected_physics_ticks=action_step * int(base_env.cfg.decimation),
                )
                if action_step % MEMORY_MONITOR_INTERVAL_STEPS == 0:
                    memory_samples.append(
                        _memory_sample(
                            action_step,
                            torch,
                            base_env.device,
                            policy_dt_ns=runtime["POLICY_DT_NS"],
                        )
                    )

        expected_rgb = args.steps // CAMERA_INTERVAL_ACTION_STEPS
        if writers["rgb_frames"].records != expected_rgb:
            raise ButtonEpisodeRecorderError(
                f"Expected {expected_rgb} RGB frames, captured {writers['rgb_frames'].records}"
            )
        if rgb_staging is None:
            raise ButtonEpisodeRecorderError("RGB recorder never allocated its required reusable CPU staging buffer")
        if rgb_staging.evidence()["copies"] != expected_rgb:
            raise ButtonEpisodeRecorderError("RGB staging copy count disagrees with captured RGB frames")
        summary["rgb_cpu_staging"] = rgb_staging.evidence()
        summary["acceptance"] = acceptance.evidence()
        if not args.short_smoke and not success_seen:
            raise ButtonEpisodeRecorderError("Full Button episode never reached the success state")
        if not args.short_smoke and not release_after_success_seen:
            raise ButtonEpisodeRecorderError("Full Button episode did not rebound after success")
        if not args.short_smoke:
            acceptance.assert_full_acceptance()
        summary["memory_monitor"]["analysis"] = _memory_analysis(memory_samples, args.steps)
        summary["backend_after"] = runtime["assert_physx_environment"](env)
        summary["passed"] = True
        print("RAMBO_BUTTON_PHYSX_EPISODE_SUCCESS", flush=True)
    except BaseException as captured_error:
        error = captured_error
        exit_code = 1
        summary["error"] = f"{type(captured_error).__name__}: {captured_error}"
        summary["traceback"] = traceback.format_exc()
        traceback.print_exception(type(captured_error), captured_error, captured_error.__traceback__)
    finally:
        for writer in writers.values():
            try:
                writer.close()
            except BaseException as close_error:
                exit_code = 1
                summary.setdefault("writer_close_errors", []).append(
                    f"{type(close_error).__name__}: {close_error}"
                )
        summary["trace_counts"] = {
            "actions": writers.get("actions").records if "actions" in writers else 0,
            "observations": writers.get("observations").records if "observations" in writers else 0,
            "post_states": writers.get("post_states").records if "post_states" in writers else 0,
            "rgb_frames": writers.get("rgb_frames").records if "rgb_frames" in writers else 0,
        }
        summary["terminal"] = {"terminal_count": terminal_count}
        summary["success"] = {
            "success_seen": success_seen,
            "release_after_success_seen": release_after_success_seen,
        }
        summary.setdefault("acceptance", acceptance.evidence())
        if rgb_staging is not None:
            summary.setdefault("rgb_cpu_staging", rgb_staging.evidence())
        if env is not None and runtime is not None and "backend_after" not in summary:
            try:
                summary["backend_after"] = runtime["assert_physx_environment"](env)
            except BaseException as backend_error:
                summary["backend_after_error"] = f"{type(backend_error).__name__}: {backend_error}"
                exit_code = 1
        if env is not None:
            try:
                env.close()
            except BaseException as close_error:
                summary["close_error"] = f"{type(close_error).__name__}: {close_error}"
                exit_code = 1
        summary["passed"] = exit_code == 0 and error is None
        manifest["completed_action_steps"] = summary["trace_counts"]["actions"]
        manifest["completed_rgb_frames"] = summary["trace_counts"]["rgb_frames"]
        if runtime is not None:
            runtime["write_json"](output_dir / "manifest.json", manifest)
            summary_path = runtime["write_json"](output_dir / "summary.json", summary)
            runtime["write_checksums"](output_dir)
        else:
            write_json(output_dir / "manifest.json", manifest)
            summary_path = write_json(output_dir / "summary.json", summary)
            write_checksums(output_dir)
        print(f"SUMMARY_PATH={summary_path}", flush=True)
        if app is not None:
            app.app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
