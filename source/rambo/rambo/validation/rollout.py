"""Deterministic RAMBO policy-rollout and RGB validation helpers.

This module deliberately contains no Isaac Sim imports at module import time.
The command-line entry points launch ``AppLauncher`` first and import these
helpers afterwards, which keeps source inspection and ordinary Python tooling
usable without an installed simulator runtime.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .checkpoints import CheckpointContract, CheckpointContractError


class RolloutValidationError(RuntimeError):
    """Raised when a policy rollout violates a RAMBO acceptance invariant."""


def _termination_diagnostic_text(base_env: Any) -> str:
    """Format a terminal-state snapshot captured by a RAMBO QP environment."""

    diagnostics = getattr(base_env, "_last_termination_diagnostics", None)
    if not isinstance(diagnostics, dict):
        return ""

    import torch

    def _scalar(value: Any) -> float | None:
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return None
        return float(value.detach().reshape(-1)[0].cpu())

    active_reasons: list[str] = []
    for name in (
        "low_base_height",
        "large_orientation_error",
        "undesired_foot_contact",
        "body_or_head_contact",
        "limb_contact",
        "time_out",
    ):
        value = diagnostics.get(name)
        if isinstance(value, torch.Tensor) and bool(torch.any(value)):
            active_reasons.append(name)

    details: list[str] = []
    base_height = _scalar(diagnostics.get("base_height"))
    if base_height is not None:
        details.append(f"base_height={base_height:.6f}")
    orientation_error = _scalar(diagnostics.get("orientation_error"))
    if orientation_error is not None:
        details.append(f"orientation_error={orientation_error:.6f}")

    if not active_reasons and not details:
        return ""
    reason_text = ", ".join(active_reasons) if active_reasons else "unknown"
    detail_text = f"; {', '.join(details)}" if details else ""
    return f" (reasons: {reason_text}{detail_text})"


def configure_validation_cfg(
    env_cfg: Any,
    *,
    duration_s: float = 31.0,
    enable_rgb_camera: bool = True,
    contact_phase_offset_s: float | None = None,
) -> Any:
    """Apply the deterministic 30-second acceptance-test configuration.

    The policy has a 100 Hz control interval.  Thirty-one seconds leaves a
    one-second margin before timeout while the validator executes exactly 3000
    policy steps.  The contact sequence is copied before extending its final
    segment so registry-owned class configuration is never mutated in place.
    Biped may select an explicit fixed contact phase without advancing its
    episode clock; that offset is also included in the required sequence span.
    """

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")

    env_cfg.episode_length_s = float(duration_s)
    configured_phase_offset_s = getattr(env_cfg, "validation_contact_phase_offset_s", None)
    if contact_phase_offset_s is None:
        contact_phase_offset_s = configured_phase_offset_s
    if contact_phase_offset_s is not None:
        contact_phase_offset_s = float(contact_phase_offset_s)
        if not math.isfinite(contact_phase_offset_s) or contact_phase_offset_s < 0.0:
            raise ValueError("contact_phase_offset_s must be a finite non-negative duration")
        if not hasattr(env_cfg, "contact_phase_offset_s"):
            raise RolloutValidationError(
                "This task does not support a deterministic contact phase offset"
            )
        env_cfg.contact_phase_offset_s = contact_phase_offset_s
    else:
        contact_phase_offset_s = 0.0
    # Startup and interval EventCfg terms are domain randomization as well.
    # Removing them is necessary because their state is sampled before the
    # first policy action, independently of the per-reset switches below.
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
    ):
        if hasattr(env_cfg, attribute):
            setattr(env_cfg, attribute, False)

    if enable_rgb_camera:
        if not hasattr(env_cfg, "enable_rgb_camera"):
            raise RolloutValidationError(
                "RAMBO validation requires env_cfg.enable_rgb_camera; task config is incomplete"
            )
        env_cfg.enable_rgb_camera = True
        # RAMBO's camera has an integer physics-tick cadence and the recorder
        # reads it on each policy step.  Preserve Isaac Lab's lazy sensor
        # update semantics: forcing an RTX render every 2 ms is unnecessary
        # for a 12.5 Hz camera and can perturb a marginally stable biped
        # rollout through the renderer/PhysX scheduling path.
        scene_cfg = getattr(env_cfg, "scene", None)
        if scene_cfg is not None and hasattr(scene_cfg, "lazy_sensor_update"):
            scene_cfg.lazy_sensor_update = True

    # These controllers create visual markers and Kit callbacks from nested
    # task dictionaries, independently of the top-level debug flags above.
    # Copy first so a validation run cannot mutate a registry-owned config.
    for attribute, debug_keys in (
        ("contact_generator_config", ("contact_generator_debug_vis",)),
        ("joint_position_controller_config", ("joint_position_controller_debug_vis",)),
        ("qp_torque_optimizer_config", ("qp_debug_vis",)),
    ):
        nested_config = getattr(env_cfg, attribute, None)
        if isinstance(nested_config, dict):
            nested_config = copy.deepcopy(nested_config)
            for debug_key in debug_keys:
                if debug_key in nested_config:
                    nested_config[debug_key] = False
            setattr(env_cfg, attribute, nested_config)

    contact_generator_config = getattr(env_cfg, "contact_generator_config", None)
    if contact_generator_config is None:
        raise RolloutValidationError("RAMBO validation requires contact_generator_config")
    copied_config = copy.deepcopy(contact_generator_config)
    contact_sequence = copied_config.get("contact_sequence")
    if not isinstance(contact_sequence, dict) or not contact_sequence:
        raise RolloutValidationError("contact_generator_config.contact_sequence must be a non-empty dict")
    for foot_name, sequence in contact_sequence.items():
        if not isinstance(sequence, list) or not sequence:
            raise RolloutValidationError(f"contact sequence for {foot_name!r} must be a non-empty list")
        if any(not isinstance(segment, list) or len(segment) < 2 for segment in sequence):
            raise RolloutValidationError(f"contact sequence for {foot_name!r} has an invalid segment")
        total_duration = sum(float(segment[1]) for segment in sequence)
        required_duration_s = duration_s + contact_phase_offset_s
        if total_duration < required_duration_s:
            sequence[-1][1] = float(sequence[-1][1]) + (required_duration_s - total_duration)
    env_cfg.contact_generator_config = copied_config
    return env_cfg


def seed_everything(seed: int, env: Any | None = None) -> None:
    """Seed Python, NumPy, Torch, and the Isaac Lab environment consistently."""

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.seed(seed)


def get_policy_timestep(base_env: Any) -> float:
    """Read and validate the native 100 Hz policy timestep from the environment."""

    step_dt = float(getattr(base_env, "step_dt"))
    if not math.isclose(step_dt, 0.01, abs_tol=1.0e-9, rel_tol=0.0):
        raise RolloutValidationError(f"Expected a 100 Hz policy timestep, got step_dt={step_dt}")
    return step_dt


def validate_environment_contract(env: Any, contract: CheckpointContract) -> Any:
    """Verify environment observation/action dimensions before runner creation."""

    import torch

    observations, _ = env.get_observations()
    if observations.ndim != 2:
        raise RolloutValidationError(
            f"Expected batched policy observations, got shape {tuple(observations.shape)}"
        )
    if observations.shape[1] != contract.observation_dim:
        raise RolloutValidationError(
            "Environment observation dimension mismatch: "
            f"expected {contract.observation_dim}, got {observations.shape[1]}"
        )
    if int(env.num_actions) != contract.action_dim:
        raise RolloutValidationError(
            "Environment action dimension mismatch: "
            f"expected {contract.action_dim}, got {env.num_actions}"
        )
    if not torch.isfinite(observations).all():
        raise RolloutValidationError("Initial policy observation contains NaN or Inf")
    return observations


def get_front_camera(base_env: Any) -> Any:
    """Return the stable public camera when present, with a legacy fallback."""

    camera = getattr(base_env, "front_camera", None)
    if camera is None:
        camera = getattr(base_env, "_front_camera", None)
    if camera is None:
        raise RolloutValidationError("RAMBO task did not expose a front RGB camera")
    return camera


def _camera_frame_id(camera: Any) -> int:
    frame = getattr(camera, "frame", None)
    if frame is None:
        raise RolloutValidationError("Front camera does not expose a frame counter")
    if hasattr(frame, "detach"):
        frame = frame.detach().reshape(-1)[0].item()
    elif hasattr(frame, "__getitem__") and not isinstance(frame, (str, bytes)):
        try:
            frame = frame[0]
        except (IndexError, KeyError, TypeError):
            pass
    if hasattr(frame, "item"):
        frame = frame.item()
    try:
        return int(frame)
    except (TypeError, ValueError) as exc:
        raise RolloutValidationError(f"Invalid camera frame counter: {frame!r}") from exc


def _camera_data(camera: Any) -> Any:
    """Read camera data before its frame counter.

    Isaac Lab scenes use lazy sensor updates by default.  A ``Camera`` marks
    itself outdated during ``scene.update()``, but increments ``frame`` only
    when its ``data`` property is read.  Sampling the frame counter first
    would therefore observe a permanent value and never consume a fresh RGB
    image.
    """

    data = getattr(camera, "data", None)
    if data is None:
        raise RolloutValidationError("Front camera does not expose sensor data")
    return data


def _flush_terminal_camera_render(base_env: Any, camera: Any) -> None:
    """Flush the final RTX render product without advancing camera cadence.

    The final policy step can land exactly on the 0.08 s sensor deadline.
    Rendering once more makes that terminal product visible to Replicator, and
    reading ``data`` consumes it only if the camera is already marked
    outdated.  It cannot create an extra frame because no sensor ``update``
    call is made here.
    """

    simulation = getattr(base_env, "sim", None)
    render = getattr(simulation, "render", None)
    if callable(render):
        render()
    _camera_data(camera)


def _to_rgb_uint8(value: Any) -> Any:
    """Convert a one-environment RGB/RGBA tensor into HxWx3 uint8 NumPy data."""

    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    frame = np.asarray(value)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise RolloutValidationError(f"Expected HxWx3/4 camera output, got shape {frame.shape}")
    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            scale = 255.0 if float(frame.max(initial=0.0)) <= 1.0 else 1.0
            frame = np.clip(frame * scale, 0.0, 255.0).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


@dataclass
class RgbFrameRecorder:
    """Write only fresh front-camera frames and report portable image checks."""

    output_dir: Path
    expected_count: int
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        if self.expected_count <= 0:
            raise ValueError("expected_count must be positive")
        self.output_dir = Path(self.output_dir)
        self.rgb_dir = self.output_dir / "rgb"
        self.rgb_dir.mkdir(parents=True, exist_ok=False)
        self._last_frame_id: int | None = None
        self._timestamps: list[float] = []
        self._frame_ids: list[int] = []
        self._frame_means: list[float] = []
        self._frame_stds: list[float] = []
        self._adjacent_mad: list[float] = []
        self._first_frame: Any | None = None
        self._previous_frame: Any | None = None
        self._resolution_ok = True
        self._contact_sheet_indices = {
            round(index * (self.expected_count - 1) / 5) for index in range(6)
        }
        self._contact_sheet_frames: dict[int, Any] = {}

    def prime(self, camera: Any) -> None:
        """Set the post-reset camera frame as the baseline without recording it."""

        _camera_data(camera)
        self._last_frame_id = _camera_frame_id(camera)

    def capture_if_new(self, camera: Any, timestamp_s: float) -> bool:
        """Persist a frame iff the camera reports a newly rendered frame."""

        import imageio.v2 as imageio
        import numpy as np

        data = _camera_data(camera)
        frame_id = _camera_frame_id(camera)
        if self._last_frame_id is None:
            self._last_frame_id = frame_id
            return False
        if frame_id < self._last_frame_id:
            raise RolloutValidationError(
                f"Camera frame counter regressed from {self._last_frame_id} to {frame_id}"
            )
        if frame_id == self._last_frame_id:
            return False
        if frame_id != self._last_frame_id + 1:
            raise RolloutValidationError(
                "Missed front-camera frames: "
                f"expected {self._last_frame_id + 1}, got {frame_id}"
            )

        output = getattr(data, "output", None)
        if not isinstance(output, dict) or "rgb" not in output:
            raise RolloutValidationError("Front camera has no 'rgb' output")
        frame = _to_rgb_uint8(output["rgb"])
        frame_index = len(self._timestamps)
        imageio.imwrite(self.rgb_dir / f"{frame_index:06d}.png", frame)

        self._timestamps.append(float(timestamp_s))
        self._frame_ids.append(frame_id)
        self._frame_means.append(float(frame.mean()))
        self._frame_stds.append(float(frame.std()))
        self._resolution_ok &= frame.shape == (self.height, self.width, 3)
        if self._first_frame is None:
            self._first_frame = frame.copy()
        if self._previous_frame is not None:
            self._adjacent_mad.append(
                float(np.mean(np.abs(frame.astype(np.int16) - self._previous_frame.astype(np.int16))))
            )
        self._previous_frame = frame.copy()
        if frame_index in self._contact_sheet_indices:
            self._contact_sheet_frames[frame_index] = frame.copy()
        self._last_frame_id = frame_id
        return True

    def _write_contact_sheet(self) -> Path | None:
        if not self._contact_sheet_frames:
            return None

        import imageio.v2 as imageio
        import numpy as np

        thumbnails = []
        for frame_index in sorted(self._contact_sheet_frames):
            frame = self._contact_sheet_frames[frame_index]
            thumbnails.append(frame[::4, ::4])
        contact_sheet = np.concatenate(thumbnails, axis=1)
        path = self.output_dir / "contact_sheet.png"
        imageio.imwrite(path, contact_sheet)
        return path

    def finalize(self) -> dict[str, Any]:
        """Persist timestamp arrays and return non-throwing RGB acceptance results."""

        import numpy as np

        np.save(self.output_dir / "rgb_timestamps.npy", np.asarray(self._timestamps, dtype=np.float64))
        np.save(self.output_dir / "rgb_frame_ids.npy", np.asarray(self._frame_ids, dtype=np.int64))
        contact_sheet = self._write_contact_sheet()

        failures: list[str] = []
        if len(self._timestamps) != self.expected_count:
            failures.append(
                f"expected {self.expected_count} fresh RGB frames, captured {len(self._timestamps)}"
            )
        if not self._resolution_ok:
            failures.append(f"RGB resolution is not consistently {self.width}x{self.height}")
        if not self._frame_means or min(self._frame_means) <= 2.0:
            failures.append("at least one RGB frame is black or near-black")
        if not self._frame_stds or min(self._frame_stds) <= 1.0:
            failures.append("at least one RGB frame has insufficient image variation")

        first_last_mad = 0.0
        changing_fraction = 0.0
        max_adjacent_mad = 0.0
        if self._first_frame is None or self._previous_frame is None or not self._adjacent_mad:
            failures.append("insufficient RGB frames to verify temporal variation")
        else:
            first_last_mad = float(
                np.mean(
                    np.abs(
                        self._first_frame.astype(np.int16) - self._previous_frame.astype(np.int16)
                    )
                )
            )
            max_adjacent_mad = max(self._adjacent_mad)
            changing_fraction = float(np.mean(np.asarray(self._adjacent_mad) > 0.1))
            if not (
                max_adjacent_mad > 1.0
                and first_last_mad > 1.0
                and changing_fraction > 0.1
            ):
                failures.append("RGB stream is fixed or lacks expected temporal variation")

        return {
            "passed": not failures,
            "failures": failures,
            "frame_count": len(self._timestamps),
            "expected_frame_count": self.expected_count,
            "resolution": [self.width, self.height],
            "mean_range": [min(self._frame_means, default=0.0), max(self._frame_means, default=0.0)],
            "std_range": [min(self._frame_stds, default=0.0), max(self._frame_stds, default=0.0)],
            "max_adjacent_mad": max_adjacent_mad,
            "first_last_mad": first_last_mad,
            "changing_fraction": changing_fraction,
            "timestamps_path": "rgb_timestamps.npy",
            "frame_ids_path": "rgb_frame_ids.npy",
            "frames_path": "rgb",
            "contact_sheet_path": None if contact_sheet is None else contact_sheet.name,
            "visual_review_required": True,
        }


def _finite_or_raise(name: str, value: Any, step: int) -> None:
    import torch

    if not torch.isfinite(value).all():
        raise RolloutValidationError(f"{name} contains NaN or Inf at policy step {step}")


def _max_abs(value: Any) -> float:
    return float(value.detach().abs().max().item())


def _require_attr(object_: Any, attribute: str, description: str) -> Any:
    value = getattr(object_, attribute, None)
    if value is None:
        raise RolloutValidationError(f"RAMBO task did not expose {description} ({attribute})")
    return value


def _get_qp_cost(extras: Any) -> Any:
    if not isinstance(extras, dict):
        raise RolloutValidationError("CRL2 step extras must be a dict")
    log = extras.get("log")
    if not isinstance(log, dict) or "Step Log/QP Cost" not in log:
        raise RolloutValidationError("RAMBO step extras did not expose 'Step Log/QP Cost'")
    return log["Step Log/QP Cost"]


def _get_torque_limit(base_env: Any, desired_torque: Any) -> Any:
    import torch

    torque_limit = getattr(base_env, "joint_torque_limit", None)
    if torque_limit is None:
        torque_limit = getattr(getattr(base_env, "cfg", None), "joint_torque_limit", None)
    if torque_limit is None:
        raise RolloutValidationError("RAMBO task did not expose joint_torque_limit")
    return torch.as_tensor(torque_limit, device=desired_torque.device, dtype=desired_torque.dtype)


def _camera_expected_count(camera: Any, policy_timestep: float, steps: int) -> int:
    camera_cfg = getattr(camera, "cfg", None)
    update_period = getattr(camera_cfg, "update_period", None)
    if update_period is None:
        raise RolloutValidationError("Front camera config did not expose update_period")
    update_period = float(update_period)
    interval_steps = round(update_period / policy_timestep)
    if interval_steps <= 0 or not math.isclose(
        interval_steps * policy_timestep, update_period, abs_tol=1.0e-9, rel_tol=0.0
    ):
        raise RolloutValidationError(
            "Front camera update_period must be an integer multiple of policy step_dt: "
            f"update_period={update_period}, step_dt={policy_timestep}"
        )
    if steps % interval_steps:
        raise RolloutValidationError(
            f"steps={steps} is not divisible by camera interval {interval_steps} policy steps"
        )
    return steps // interval_steps


def run_policy_rollout(
    env: Any,
    policy: Callable[[Any], Any],
    contract: CheckpointContract,
    *,
    steps: int,
    rgb_recorder: RgbFrameRecorder | None = None,
) -> dict[str, Any]:
    """Run a finite policy rollout and enforce all numerical/safety invariants."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    import torch

    base_env = env.unwrapped
    # QP environments retain a pre-reset terminal snapshot when this flag is
    # set.  Generic Gym environments simply ignore the attribute.
    base_env._record_termination_diagnostics = True
    base_env._last_termination_diagnostics = None
    policy_timestep = get_policy_timestep(base_env)
    physics_timestep = float(getattr(base_env, "physics_dt"))
    observations = validate_environment_contract(env, contract)
    camera = None
    expected_camera_count = None
    if rgb_recorder is not None:
        camera = get_front_camera(base_env)
        expected_camera_count = _camera_expected_count(camera, policy_timestep, steps)
        if rgb_recorder.expected_count != expected_camera_count:
            raise RolloutValidationError(
                "RGB recorder expected frame count does not match camera schedule: "
                f"recorder={rgb_recorder.expected_count}, camera={expected_camera_count}"
            )
        rgb_recorder.prime(camera)

    action_samples: list[Any] = []
    max_abs = {
        "grf": 0.0,
        "desired_joint_position": 0.0,
        "desired_joint_velocity": 0.0,
        "desired_joint_torque": 0.0,
        "applied_joint_torque": 0.0,
        "qp_cost": 0.0,
    }
    min_base_height = math.inf
    max_orientation_error = 0.0
    max_torque_limit_violation = 0.0

    for step in range(steps):
        with torch.inference_mode():
            actions = policy(observations)
            _finite_or_raise("policy action", actions, step)
            action_samples.append(actions.detach().cpu())

            next_observations, rewards, dones, extras = env.step(actions)
            _finite_or_raise("policy observation", next_observations, step)
            _finite_or_raise("reward", rewards, step)
            if bool(torch.any(dones)):
                raise RolloutValidationError(
                    f"environment terminated or timed out at policy step {step}"
                    f"{_termination_diagnostic_text(base_env)}"
                )

            torque_optimizer = _require_attr(base_env, "torque_optimizer", "torque optimizer")
            grf = _require_attr(torque_optimizer, "grf", "QP ground-reaction forces")
            desired_position = _require_attr(base_env, "desired_pos", "desired joint positions")
            desired_velocity = _require_attr(base_env, "desired_vel", "desired joint velocities")
            desired_torque = _require_attr(base_env, "desired_tor", "desired joint torques")
            robot = _require_attr(base_env, "_robot", "robot articulation")
            robot_data = _require_attr(robot, "data", "robot articulation data")
            applied_torque = _require_attr(robot_data, "applied_torque", "applied joint torques")
            root_pos_w = _require_attr(robot_data, "root_pos_w", "root position")
            projected_gravity_b = _require_attr(robot_data, "projected_gravity_b", "projected gravity")
            qp_cost = _get_qp_cost(extras)

            for name, value in {
                "QP ground-reaction force": grf,
                "desired joint position": desired_position,
                "desired joint velocity": desired_velocity,
                "desired joint torque": desired_torque,
                "applied joint torque": applied_torque,
                "QP cost": qp_cost,
                "root position": root_pos_w,
                "projected gravity": projected_gravity_b,
            }.items():
                _finite_or_raise(name, value, step)

            max_abs["grf"] = max(max_abs["grf"], _max_abs(grf))
            max_abs["desired_joint_position"] = max(
                max_abs["desired_joint_position"], _max_abs(desired_position)
            )
            max_abs["desired_joint_velocity"] = max(
                max_abs["desired_joint_velocity"], _max_abs(desired_velocity)
            )
            max_abs["desired_joint_torque"] = max(
                max_abs["desired_joint_torque"], _max_abs(desired_torque)
            )
            max_abs["applied_joint_torque"] = max(
                max_abs["applied_joint_torque"], _max_abs(applied_torque)
            )
            max_abs["qp_cost"] = max(max_abs["qp_cost"], _max_abs(qp_cost))

            base_height = root_pos_w[:, 2]
            current_min_height = float(base_height.min().item())
            min_base_height = min(min_base_height, current_min_height)
            if current_min_height < contract.min_base_height:
                raise RolloutValidationError(
                    f"base height {current_min_height:.6f} is below {contract.min_base_height:.6f} at step {step}"
                )

            gravity_target = torch.tensor(
                contract.gravity_target, device=projected_gravity_b.device, dtype=projected_gravity_b.dtype
            )
            orientation_error = torch.linalg.vector_norm(projected_gravity_b - gravity_target, dim=-1)
            current_max_orientation_error = float(orientation_error.max().item())
            max_orientation_error = max(max_orientation_error, current_max_orientation_error)
            if current_max_orientation_error > contract.max_orientation_error:
                raise RolloutValidationError(
                    "orientation error "
                    f"{current_max_orientation_error:.6f} exceeds {contract.max_orientation_error:.6f} "
                    f"at step {step}"
                )

            torque_limit = _get_torque_limit(base_env, desired_torque)
            torque_violation = (desired_torque.abs() - torque_limit).amax()
            max_torque_limit_violation = max(max_torque_limit_violation, float(torque_violation.item()))
            if float(torque_violation.item()) > 1.0e-4:
                raise RolloutValidationError(
                    "desired torque exceeds joint_torque_limit by "
                    f"{float(torque_violation.item()):.6f} at step {step}"
                )

            observations = next_observations
            if rgb_recorder is not None and camera is not None:
                rgb_recorder.capture_if_new(camera, (step + 1) * policy_timestep)

    if rgb_recorder is not None and camera is not None:
        # The normal in-loop capture handles each due frame.  This terminal
        # flush only recovers a final render product that arrived after the
        # last sensor read, and remains a no-op when the frame was already
        # consumed.
        _flush_terminal_camera_render(base_env, camera)
        rgb_recorder.capture_if_new(camera, steps * policy_timestep)

    actions = torch.cat(action_samples, dim=0)
    return {
        "steps": steps,
        "policy_timestep_s": policy_timestep,
        "physics_timestep_s": physics_timestep,
        "policy_frequency_hz": 1.0 / policy_timestep,
        "physics_frequency_hz": 1.0 / physics_timestep,
        "expected_camera_frame_count": expected_camera_count,
        "actions": {
            "shape": list(actions.shape),
            "min": float(actions.min().item()),
            "max": float(actions.max().item()),
            "mean": float(actions.mean().item()),
            "std": float(actions.std().item()),
            "nan_count": int(torch.isnan(actions).sum().item()),
            "inf_count": int(torch.isinf(actions).sum().item()),
        },
        "controller_max_abs": max_abs,
        "min_base_height": min_base_height,
        "base_height_threshold": contract.min_base_height,
        "max_orientation_error": max_orientation_error,
        "orientation_error_threshold": contract.max_orientation_error,
        "max_torque_limit_violation": max_torque_limit_violation,
    }


def prepare_output_dir(path: str | Path) -> Path:
    """Create an empty validation artifact directory without overwriting output."""

    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RolloutValidationError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_summary(output_dir: str | Path, summary: dict[str, Any]) -> Path:
    """Write the machine-readable validation result with stable UTF-8 formatting."""

    path = Path(output_dir) / "summary.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, sort_keys=True)
        file.write("\n")
    return path


def runtime_metadata() -> dict[str, Any]:
    """Collect lightweight, optional runtime version details for summary.json."""

    metadata: dict[str, Any] = {}
    try:
        import torch

        metadata["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            metadata["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
            metadata["cuda_runtime"] = torch.version.cuda
    except Exception as exc:  # pragma: no cover - diagnostic-only fallback
        metadata["torch_metadata_error"] = repr(exc)
    try:
        import isaaclab

        metadata["isaaclab_version"] = getattr(isaaclab, "__version__", "unknown")
    except Exception:
        pass
    try:
        import isaacsim

        metadata["isaacsim_version"] = getattr(isaacsim, "__version__", "unknown")
    except Exception:
        pass
    return metadata


__all__ = (
    "RgbFrameRecorder",
    "RolloutValidationError",
    "configure_validation_cfg",
    "get_front_camera",
    "get_policy_timestep",
    "prepare_output_dir",
    "run_policy_rollout",
    "runtime_metadata",
    "seed_everything",
    "validate_environment_contract",
    "write_summary",
)
