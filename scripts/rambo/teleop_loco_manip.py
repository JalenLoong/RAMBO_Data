#!/usr/bin/env python3
"""Keyboard loco-manip teleoperation for Go2 FL manipulation tasks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np


TASK_ID = "Isaac-RAMBO-Quadruped-Button-Go2-v0"
LIFT_BASKET_TASK_ID = "Isaac-RAMBO-Quadruped-Lift-Basket-Go2-v0"
VIEW_THIRD_PERSON = "third-person"
VIEW_EGO = "ego"
VIEW_TASK = "task"
EGO_VIEW_CAMERA_PRIM_PATH = "/World/envs/env_0/Robot/base/front_camera"
EE_MIN = np.array([0.1934, 0.0, 0.0], dtype=np.float32)
EE_MAX = np.array([0.50, 0.20, 0.40], dtype=np.float32)
EE_DEFAULT = np.array([0.1934, 0.142, 0.05], dtype=np.float32)
GUI_ARTIFACT_ROOT = Path("/workspace/runs/audit/isaac60/M8")
GUI_ARTIFACT_MAX_STEPS = 3000


def _utc_now() -> str:
    """Return a compact, timezone-explicit evidence timestamp."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fqn(value: Any) -> str:
    """Return the stable fully-qualified identity of a config or runtime object."""

    target = value if isinstance(value, type) else type(value)
    return f"{target.__module__}.{target.__qualname__}"


def _keyboard_event_name(event: Any) -> str:
    """Normalize Carb keyboard key values exposed as enums or strings."""

    raw_key = event.input
    return raw_key.name if hasattr(raw_key, "name") else str(raw_key)


def _keyboard_event_type_name(event: Any) -> str:
    """Normalize the two callback event types persisted by the GUI recorder."""

    raw_type = event.type
    name = raw_type.name if hasattr(raw_type, "name") else str(raw_type)
    if name.endswith(".KEY_PRESS"):
        return "KEY_PRESS"
    if name.endswith(".KEY_RELEASE"):
        return "KEY_RELEASE"
    return name


def _build_parser() -> tuple[argparse.ArgumentParser, type]:
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as error:  # pragma: no cover - target-runtime guard.
        raise RuntimeError("Run this script through scripts/rambo/run.sh") from error

    parser = argparse.ArgumentParser(
        description="Teleoperate RAMBO Go2 locomotion and its FL manipulation leg."
    )
    parser.add_argument("--task", choices=(TASK_ID, LIFT_BASKET_TASK_ID), default=TASK_ID)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted quadruped model_2000.pt")
    parser.add_argument(
        "--view",
        choices=(VIEW_THIRD_PERSON, VIEW_EGO, VIEW_TASK),
        default=VIEW_THIRD_PERSON,
        help=(
            "Kit viewport camera: 'ego' selects the physical base-mounted front RGB camera; "
            "'third-person' preserves the existing world view."
        ),
    )
    parser.add_argument("--camera-setup", choices=("robot-dual-v3",), default=None,
                        help="Versioned Lift-basket forward-ego + bracket-mounted task-camera setup")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-length-s", type=float, default=300.0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs until the GUI closes")
    parser.add_argument(
        "--gui-artifact-dir",
        type=Path,
        default=None,
        help=(
            "Explicitly opt in to a fresh, bounded M8 GUI keyboard artifact. "
            "This mode requires --viz kit and a named operator for the post-close TTY attestation."
        ),
    )
    parser.add_argument(
        "--operator-name",
        default=None,
        help="Named operator for the required post-close GUI attestation; this is not the declaration itself.",
    )
    parser.add_argument(
        "--gui-arm-timeout-s",
        type=float,
        default=120.0,
        help="Maximum wall-clock wait for the first native keyboard callback in GUI artifact mode.",
    )
    parser.add_argument(
        "--telemetry-every",
        type=int,
        default=0,
        help="Print interactive state every N control steps; 0 disables periodic output",
    )
    smoke = parser.add_mutually_exclusive_group()
    smoke.add_argument("--smoke-walk", action="store_true", help="Verify a scripted walking command")
    smoke.add_argument(
        "--smoke-press",
        action="store_true",
        help="Verify physical FL press, latch, retraction, and spring return",
    )
    smoke.add_argument(
        "--smoke-loco-manip",
        action="store_true",
        help="Verify walking to the wall followed by a physical FL button press",
    )
    parser.add_argument("--vx-sensitivity", type=float, default=0.4)
    parser.add_argument("--vy-sensitivity", type=float, default=0.2)
    parser.add_argument("--wz-sensitivity", type=float, default=0.4)
    parser.add_argument("--fl-x-speed", type=float, default=0.12)
    parser.add_argument("--fl-y-speed", type=float, default=0.08)
    parser.add_argument("--fl-z-speed", type=float, default=0.10)
    parser.add_argument(
        "--disable-fabric",
        "--disable_fabric",
        dest="disable_fabric",
        action="store_true",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser, AppLauncher


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.episode_length_s <= 1.0:
        parser.error("--episode-length-s must exceed the one-second FL stance phase")
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    if args.telemetry_every < 0:
        parser.error("--telemetry-every must be non-negative")
    for name in ("vx_sensitivity", "vy_sensitivity", "wz_sensitivity"):
        value = float(getattr(args, name))
        if value <= 0.0 or value > 0.5:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 0.5]")
    for name in ("fl_x_speed", "fl_y_speed", "fl_z_speed"):
        if float(getattr(args, name)) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.task != TASK_ID and any(
        bool(getattr(args, name, False))
        for name in ("smoke_walk", "smoke_press", "smoke_loco_manip")
    ):
        parser.error("Scripted teleop smoke modes currently require the Button task")


def _validate_view_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Keep the ego camera as an interactive Kit-only view selection."""

    if args.view in (VIEW_EGO, VIEW_TASK) and getattr(args, "rambo_visualizer", None) != ["kit"]:
        parser.error(f"--view {args.view} requires explicit --viz kit")
    if getattr(args, "task", TASK_ID) == LIFT_BASKET_TASK_ID and getattr(args, "camera_setup", None) is None:
        args.camera_setup = "robot-dual-v3"
    setup = getattr(args, "camera_setup", None)
    if setup in ("robot-dual-v3",) and args.task != LIFT_BASKET_TASK_ID:
        parser.error(f"--camera-setup {setup} requires the Lift-basket task")
    if args.view == VIEW_TASK and setup not in ("robot-dual-v3",):
        parser.error("--view task requires a robot-dual camera setup")


def _configure_ego_view_launcher(args: argparse.Namespace) -> None:
    """Request Kit camera support before AppLauncher creates the RTX sensor."""

    if args.view in (VIEW_EGO, VIEW_TASK) or getattr(args, "camera_setup", None) in ("robot-dual-v3",):
        # Isaac Sim creates the renderer while constructing the environment, so
        # this must be set before ``AppLauncher`` rather than in _run().
        args.enable_cameras = True


def _validate_gui_artifact_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Fail closed before AppLauncher when an auditable GUI run was requested."""

    artifact_dir = getattr(args, "gui_artifact_dir", None)
    operator_name = getattr(args, "operator_name", None)
    if artifact_dir is None:
        if operator_name is not None:
            parser.error("--operator-name requires --gui-artifact-dir")
        return
    if args.task != TASK_ID:
        parser.error("--gui-artifact-dir is currently defined only for the Button task")
    if getattr(args, "rambo_visualizer", None) != ["kit"]:
        parser.error("--gui-artifact-dir requires explicit --viz kit")
    if any(
        bool(getattr(args, name, False))
        for name in ("smoke_walk", "smoke_press", "smoke_loco_manip")
    ):
        parser.error("--gui-artifact-dir forbids all scripted smoke modes")
    if args.max_steps <= 0 or args.max_steps > GUI_ARTIFACT_MAX_STEPS:
        parser.error(
            f"--gui-artifact-dir requires finite --max-steps in [1, {GUI_ARTIFACT_MAX_STEPS}]"
        )
    if not isinstance(operator_name, str) or not operator_name.strip():
        parser.error("--gui-artifact-dir requires a non-empty --operator-name")
    if len(operator_name.strip()) > 160:
        parser.error("--operator-name must be at most 160 characters")
    gui_arm_timeout_s = float(getattr(args, "gui_arm_timeout_s", 120.0))
    if gui_arm_timeout_s <= 0.0 or gui_arm_timeout_s > 300.0:
        parser.error("--gui-artifact-dir requires --gui-arm-timeout-s in (0, 300]")
    output_dir = Path(artifact_dir).expanduser().resolve()
    artifact_root = GUI_ARTIFACT_ROOT.resolve()
    try:
        output_dir.relative_to(artifact_root)
    except ValueError:
        parser.error(f"--gui-artifact-dir must be below the M8 artifact root {artifact_root}")
    if output_dir == artifact_root or output_dir.exists():
        parser.error(f"Refusing to overwrite existing GUI artifact directory: {output_dir}")
    args.gui_artifact_dir = output_dir


def _extend_contact_schedule(env_cfg: Any, minimum_horizon_s: float) -> None:
    for sequence in env_cfg.contact_generator_config["contact_sequence"].values():
        horizon = sum(float(segment[1]) for segment in sequence)
        if horizon < minimum_horizon_s:
            sequence[-1][1] += minimum_horizon_s - horizon


def _configure_environment(env_cfg: Any, args: argparse.Namespace) -> None:
    env_cfg.seed = args.seed
    env_cfg.events = None
    env_cfg.obs_noise = False
    env_cfg.randomize_episode_progress = False
    env_cfg.randomize_initial_state = False
    env_cfg.enable_sampled_velocity_commands = False
    env_cfg.enable_sampled_pos_commands = False
    env_cfg.enable_sampled_force_commands = False
    env_cfg.episode_length_s = args.episode_length_s
    _extend_contact_schedule(env_cfg, args.episode_length_s + 1.0)
    env_cfg.scene.env_spacing = 10.0
    env_cfg.viewer.eye = [-1.6, -2.0, 1.10]
    env_cfg.viewer.lookat = [0.70, 0.10, 0.28]
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.asset_name = None
    # Button may expose a front debug camera. Lift uses the fixed mounted pair.
    if getattr(args, "view", VIEW_THIRD_PERSON) == VIEW_EGO:
        env_cfg.enable_rgb_camera = True
    if args.task == LIFT_BASKET_TASK_ID:
        # Calf/thigh contact is a useful fail-fast condition for training, but
        # it is too strict for interactive leg manipulation around the source
        # basket handle. Keep fall, orientation, and body/head safety gates.
        env_cfg.terminate_on_limb_contact = False
    if getattr(args, "camera_setup", None) in ("robot-dual-v3",):
        from rambo.tasks.common.lift_camera_rig import configure
        configure(env_cfg, args.camera_setup)


def _activate_ego_viewport(base_env: Any) -> str:
    """Select the live base-mounted RTX camera in the active Kit viewport.

    This changes only the operator's viewport camera. It neither writes a robot
    command nor changes the camera calibration or sensor output path.
    """

    camera = getattr(base_env, "front_camera", None)
    if camera is None or base_env.scene.sensors.get("front_camera") is not camera:
        raise RuntimeError("--view ego requires the public RAMBO front_camera sensor")

    import omni.usd
    from isaacsim.core.rendering_manager import ViewportManager
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(EGO_VIEW_CAMERA_PRIM_PATH)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        raise RuntimeError(
            "RAMBO front camera is not a live USD Camera at "
            f"{EGO_VIEW_CAMERA_PRIM_PATH}"
        )
    ready, waited_frames = ViewportManager.wait_for_viewport(max_frames=120)
    if not ready:
        raise RuntimeError("Isaac Sim Kit viewport did not become ready for ego view")
    ViewportManager.set_camera(EGO_VIEW_CAMERA_PRIM_PATH)
    selected_path = str(ViewportManager.get_camera().GetPath())
    if selected_path != EGO_VIEW_CAMERA_PRIM_PATH:
        raise RuntimeError(
            "Isaac Sim viewport selected a different camera after ego-view request: "
            f"{selected_path}"
        )
    print(
        f"[VIEW] ego camera={selected_path} viewport_ready_after_frames={waited_frames}",
        flush=True,
    )
    return selected_path


def _select_mounted_view(view):
    from isaacsim.core.rendering_manager import ViewportManager
    from rambo.tasks.common.lift_camera_rig import CAMERAS
    path = "/OmniverseKit_Persp" if view == VIEW_THIRD_PERSON else CAMERAS[view]["path"]
    ViewportManager.set_camera(path)
    if str(ViewportManager.get_camera().GetPath()) != path:
        raise RuntimeError(f"Viewport failed to select {path}")
    print(f"[VIEW] {view} camera={path}", flush=True)


def _validate_button_environment(base_env: Any) -> None:
    required = (
        "set_loco_manip_commands",
        "button_displacement",
        "button_success",
        "button_released",
        "manipulator_ready",
        "clear_button_success",
    )
    missing = [name for name in required if not hasattr(base_env, name)]
    if missing:
        raise RuntimeError("Button task is missing loco-manip interfaces: " + ", ".join(missing))
    if not callable(base_env.set_loco_manip_commands):
        raise RuntimeError("Button task set_loco_manip_commands interface is not callable")


def _validate_loco_manip_environment(base_env: Any) -> None:
    required = ("set_loco_manip_commands", "manipulator_ready", "task_metrics", "task_success")
    missing = [name for name in required if not hasattr(base_env, name)]
    if missing:
        raise RuntimeError("Object task is missing loco-manip interfaces: " + ", ".join(missing))
    if not callable(base_env.set_loco_manip_commands):
        raise RuntimeError("Object task set_loco_manip_commands interface is not callable")


def _validate_lift_basket_source_physics() -> dict[str, Any]:
    """Fail if the runtime stage reintroduces proxy collision or motion constraints."""

    import math
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    basket_path = "/World/envs/env_0/LingBotTask/basket"
    basket = stage.GetPrimAtPath(basket_path)
    if not basket.IsValid() or not basket.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("Lift basket source rigid body is missing")
    mass = float(UsdPhysics.MassAPI(basket).GetMassAttr().Get())
    if not math.isclose(mass, 0.53, abs_tol=1.0e-6):
        raise RuntimeError(f"Lift basket source mass changed: {mass}")
    collisions = [
        prim
        for prim in Usd.PrimRange(basket)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    enabled = [
        prim
        for prim in collisions
        if bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
    ]
    approximations = [
        UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        for prim in enabled
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
    ]
    if len(enabled) != 1 or approximations != ["convexDecomposition"]:
        raise RuntimeError(
            "Lift basket must use exactly its source convex-decomposition collision: "
            f"enabled={len(enabled)} approximations={approximations}"
        )
    if stage.GetPrimAtPath(f"{basket_path}/CollisionProxy").IsValid():
        raise RuntimeError("Lift basket runtime contains a forbidden collision proxy")
    task_root = stage.GetPrimAtPath("/World/envs/env_0/LingBotTask")
    joints = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(task_root)
        if prim.IsA(UsdPhysics.PrismaticJoint)
        or prim.IsA(UsdPhysics.FixedJoint)
        or prim.IsA(UsdPhysics.RevoluteJoint)
        or prim.IsA(UsdPhysics.SphericalJoint)
    ]
    if joints:
        raise RuntimeError(f"Lift basket runtime contains forbidden task joints: {joints}")
    evidence = {
        "mass_kg": mass,
        "collision_paths": [str(prim.GetPath()) for prim in enabled],
        "collision_approximations": approximations,
        "task_joints": joints,
    }
    print(f"[LIFT-PHYSICS] {evidence}", flush=True)
    return evidence


def _prepare_smoke_press_start(base_env: Any, torch: Any) -> None:
    """Place Go2 where the trained FL target can reach the physical cap."""

    env_ids = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long)
    writer_env_ids = env_ids.to(dtype=torch.int32)
    root_pose = base_env._robot.data.default_root_pose.torch.clone()
    root_velocity = base_env._robot.data.default_root_vel.torch.clone()
    root_pose[:, 0] = 0.58
    root_pose[:, 1] = 0.0
    root_pose[:, 2] += 0.10
    root_velocity.zero_()
    joint_pos = base_env._robot.data.default_joint_pos.torch.clone()
    joint_vel = torch.zeros_like(base_env._robot.data.default_joint_vel.torch)
    base_env._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=writer_env_ids)
    base_env._robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=writer_env_ids)
    base_env._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=writer_env_ids)
    base_env._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=writer_env_ids)
    zero_commands = torch.zeros((base_env.num_envs, 3), device=base_env.device, dtype=joint_pos.dtype)
    base_env.set_loco_manip_commands(
        base_velocity=zero_commands,
        fl_position=torch.as_tensor(EE_DEFAULT, device=base_env.device, dtype=joint_pos.dtype).view(1, 3),
        fl_force=zero_commands,
        env_ids=env_ids,
    )
    base_env.scene.write_data_to_sim()
    base_env.sim.forward()
    base_env._obs_history.zero_()
    base_env.contact_generator.reset_idx(env_ids)
    base_env._desired_joint_pos[env_ids] = base_env.joint_position_controller.reset_idx(env_ids)
    base_env._last_action[env_ids] = 0.0
    base_env.obs_buf = base_env._get_observations()


def _make_keyboard(base_env: Any, args: argparse.Namespace, *, event_observer: Any = None):
    """Create the existing Carb-backed keyboard device, optionally observing callbacks.

    ``event_observer`` receives a JSON-safe snapshot *after* the native callback
    has updated its command state.  It is observation-only: this function does
    not construct, replay, or inject keyboard events.
    """

    import carb
    from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

    class ManipulatorKeyboard(Se2Keyboard):
        _LEG_KEYS = {
            "W": (0, 1.0),
            "S": (0, -1.0),
            "A": (1, 1.0),
            "D": (1, -1.0),
            "R": (2, 1.0),
            "F": (2, -1.0),
        }

        def __init__(self) -> None:
            self._held_leg_keys: set[str] = set()
            self._clear_success_requested = False
            self._event_observer = event_observer
            self._fl_speeds = np.array(
                [args.fl_x_speed, args.fl_y_speed, args.fl_z_speed], dtype=np.float32
            )
            super().__init__(
                Se2KeyboardCfg(
                    v_x_sensitivity=args.vx_sensitivity,
                    v_y_sensitivity=args.vy_sensitivity,
                    omega_z_sensitivity=args.wz_sensitivity,
                    sim_device=str(base_env.device),
                )
            )

        def reset(self) -> None:
            super().reset()
            self._held_leg_keys.clear()
            self._clear_success_requested = False

        def leg_velocity(self) -> np.ndarray:
            velocity = np.zeros(3, dtype=np.float32)
            for key in self._held_leg_keys:
                axis, sign = self._LEG_KEYS[key]
                velocity[axis] += sign * self._fl_speeds[axis]
            return velocity

        def consume_clear_success_request(self) -> bool:
            requested = self._clear_success_requested
            self._clear_success_requested = False
            return requested

        def _observe_keyboard_callback(self, event: Any, key: str) -> None:
            if self._event_observer is None:
                return
            self._event_observer(
                {
                    "callback_monotonic_ns": time.monotonic_ns(),
                    "source": "carb_keyboard_callback",
                    "callback_observed_only": True,
                    "key": key,
                    "event_type": _keyboard_event_type_name(event),
                    "handled_by_loco_manip": bool(
                        key in self._LEG_KEYS
                        or key == "C"
                        or key == "L"
                        or key in self._INPUT_KEY_MAPPING
                        or key in self._additional_callbacks
                    ),
                    "held_leg_keys": sorted(self._held_leg_keys),
                    "base_command": np.asarray(self._base_command, dtype=np.float32).tolist(),
                    "clear_success_requested": bool(self._clear_success_requested),
                }
            )

        def _on_keyboard_event(self, event, *callback_args, **callback_kwargs):
            key = _keyboard_event_name(event)
            try:
                if key in self._LEG_KEYS:
                    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                        self._held_leg_keys.add(key)
                    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                        self._held_leg_keys.discard(key)
                    return True
                if event.type == carb.input.KeyboardEventType.KEY_PRESS and key == "C":
                    self._clear_success_requested = True
                    return True
                if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    if key == "L":
                        self.reset()
                    elif key in self._INPUT_KEY_MAPPING:
                        self._base_command += self._INPUT_KEY_MAPPING[key]
                    if key in self._additional_callbacks:
                        self._additional_callbacks[key]()
                elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                    if key in self._INPUT_KEY_MAPPING:
                        self._base_command -= self._INPUT_KEY_MAPPING[key]
                return True
            finally:
                self._observe_keyboard_callback(event, key)

        def __str__(self) -> str:
            description = (
                super().__str__()
                + "\n\tFL forward/back: W / S"
                + "\n\tFL lateral +/-: A / D"
                + "\n\tFL up/down: R / F"
                + "\n\tStop and reset all commands: L"
            )
            if args.task == TASK_ID:
                description += "\n\tClear released-button success: C"
            return description

    return ManipulatorKeyboard()


class _HeadlessInput:
    def reset(self) -> None:
        return None

    def advance(self):
        return np.zeros(3, dtype=np.float32)

    def leg_velocity(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def consume_clear_success_request(self) -> bool:
        return False


def _gui_state_record(
    base_env: Any,
    *,
    step_count: int,
    base_command: np.ndarray,
    leg_target: np.ndarray,
    leg_velocity: np.ndarray,
    callback_events_observed: int,
) -> dict[str, Any]:
    """Capture one post-step state for an explicitly requested GUI artifact."""

    root_position = base_env._robot.data.root_link_pos_w.torch[0].detach().cpu().tolist()
    return {
        "action_step": step_count,
        "monotonic_ns": time.monotonic_ns(),
        "simulation_time_s": float(step_count * base_env.step_dt),
        "base_command": np.asarray(base_command, dtype=np.float32).tolist(),
        "leg_velocity": np.asarray(leg_velocity, dtype=np.float32).tolist(),
        "leg_target": np.asarray(leg_target, dtype=np.float32).tolist(),
        "root_position_m": root_position,
        "button_displacement_m": float(base_env.button_displacement[0].item()),
        "button_success": bool(base_env.button_success[0].item()),
        "button_released": bool(base_env.button_released[0].item()),
        "manipulator_ready": bool(base_env.manipulator_ready[0].item()),
        "terminal": False,
        "callback_events_observed": callback_events_observed,
    }


def _gui_m8_evidence(
    *,
    base_nonzero_state_count: int,
    fl_nonzero_velocity_state_count: int,
    unhandled_space_press_count: int,
    max_fl_target_delta_m: float,
    max_root_position_delta_m: float,
    max_button_displacement_m: float,
    button_press_threshold_seen: bool,
    button_success_after_threshold_seen: bool,
    first_button_success_action_step: int | None,
    button_rebound_after_success_seen: bool,
    first_button_rebound_action_step: int | None,
) -> dict[str, Any]:
    """Return the M8 GUI behavior facts that the offline validator recomputes."""

    return {
        "base_nonzero_state_count": base_nonzero_state_count,
        "fl_nonzero_velocity_state_count": fl_nonzero_velocity_state_count,
        "unhandled_space_press_count": unhandled_space_press_count,
        "max_fl_target_delta_m": max_fl_target_delta_m,
        "max_root_position_delta_m": max_root_position_delta_m,
        "max_button_displacement_m": max_button_displacement_m,
        "button_press_threshold_seen": button_press_threshold_seen,
        "button_success_after_threshold_seen": button_success_after_threshold_seen,
        "first_button_success_action_step": first_button_success_action_step,
        "button_rebound_after_success_seen": button_rebound_after_success_seen,
        "first_button_rebound_action_step": first_button_rebound_action_step,
    }


def _assert_complete_gui_m8_evidence(schema: Any, evidence: dict[str, Any]) -> None:
    """Fail the runtime artifact instead of merely leaving an offline-rejected summary."""

    missing: list[str] = []
    if evidence["base_nonzero_state_count"] <= 0:
        missing.append("non-zero base command")
    if evidence["fl_nonzero_velocity_state_count"] <= 0:
        missing.append("non-zero FL velocity")
    if evidence["max_fl_target_delta_m"] < schema.FL_TARGET_MINIMUM_DELTA_M:
        missing.append("FL target movement")
    if evidence["unhandled_space_press_count"] <= 0:
        missing.append("unhandled SPACE press")
    if evidence["button_press_threshold_seen"] is not True:
        missing.append("12-mm button press")
    if evidence["button_success_after_threshold_seen"] is not True:
        missing.append("button success after press")
    if evidence["button_rebound_after_success_seen"] is not True:
        missing.append("button rebound/released after success")
    if missing:
        raise RuntimeError("GUI artifact is incomplete: " + ", ".join(missing))


def _gui_artifact_summary(
    schema: Any,
    *,
    args: argparse.Namespace,
    outcome: str,
    executed_action_steps: int,
    event_count: int,
    state_count: int,
    handled_key_press_count: int,
    active_command_state_count: int,
    configured_physics: dict[str, Any] | None,
    backend_before: dict[str, Any] | None,
    backend_after: dict[str, Any] | None,
    checkpoint_sha256: str,
    runtime: dict[str, Any],
    m8_evidence: dict[str, Any],
    failure: BaseException | None = None,
) -> dict[str, Any]:
    """Build a completion or failure summary without claiming physical proof."""

    summary: dict[str, Any] = {
        "schema_version": schema.SCHEMA_VERSION,
        "artifact_kind": schema.ARTIFACT_KIND,
        "task": schema.TASK_ID,
        "outcome": outcome,
        "visualizer": schema.VISUALIZER,
        "smoke_mode": False,
        "max_action_steps": args.max_steps,
        "arm_timeout_s": float(args.gui_arm_timeout_s),
        "executed_action_steps": executed_action_steps,
        "event_count": event_count,
        "state_count": state_count,
        "handled_key_press_count": handled_key_press_count,
        "active_command_state_count": active_command_state_count,
        "configured_physics": configured_physics,
        "backend_before": backend_before,
        "backend_after": backend_after,
        "checkpoint": {
            "path": str(args.checkpoint.expanduser().resolve()),
            "sha256": checkpoint_sha256,
        },
        "runtime": runtime,
        "telemetry": {
            "states_file": schema.STATES_FILENAME,
            "interval_action_steps": 1,
            "state_records": state_count,
        },
        "m8_evidence": m8_evidence,
        "operator_attestation_required": True,
        "physical_keyboard_independently_proven": False,
        "limitation": schema.PHYSICALITY_LIMITATION,
        "post_close_exit_required": True,
        "post_close_exit_file": schema.PROCESS_EXIT_FILENAME,
        "post_close_exit_runner": schema.POST_CLOSE_RUNNER,
        "post_close_attestation_required": True,
        "post_close_attestation_file": schema.ATTESTATION_FILENAME,
        "post_close_attestation_confirmation_method": schema.ATTESTATION_CONFIRMATION_METHOD,
        "finished_at_utc": _utc_now(),
        "finished_monotonic_ns": time.monotonic_ns(),
    }
    if failure is not None:
        summary["failure"] = f"{type(failure).__name__}: {failure}"
    return summary


def _run(args: argparse.Namespace, simulation_app: Any) -> int:
    from rambo.torch_runtime import ensure_cuda_linalg_loaded

    gui_artifact_enabled = getattr(args, "gui_artifact_dir", None) is not None
    button_task = args.task == TASK_ID
    if gui_artifact_enabled:
        if getattr(args, "rambo_visualizer", None) != ["kit"]:
            raise RuntimeError("GUI artifact mode requires the Kit visualizer")
        if args.smoke_walk or args.smoke_press or args.smoke_loco_manip:
            raise RuntimeError("GUI artifact mode forbids scripted smoke modes")
        if args.max_steps <= 0 or args.max_steps > GUI_ARTIFACT_MAX_STEPS:
            raise RuntimeError("GUI artifact mode requires a finite max-steps limit")

    ensure_cuda_linalg_loaded()
    import gymnasium as gym
    import rambo
    import torch
    from crl2.algorithms import PPO
    from rambo.rl import Crl2VecEnvWrapper
    from rambo.utils.registry import load_cfg_from_registry, parse_env_cfg
    from rambo.utils.physx import PHYSX_CFG_FQN, assert_physx_environment, configure_physx
    from rambo.validation.checkpoints import contract_for_task, load_verified_checkpoint, restore_runner
    from rambo.validation import gui_keyboard_artifact as gui_artifact_schema
    from rambo.validation.gui_keyboard_artifact import GuiKeyboardArtifactWriter
    from rambo.validation.rollout import seed_everything, validate_environment_contract

    rambo.register_tasks()
    contract = contract_for_task(args.task)
    checkpoint = load_verified_checkpoint(args.checkpoint, contract)
    env_cfg = parse_env_cfg(args.task, num_envs=1, use_fabric=not args.disable_fabric)
    _configure_environment(env_cfg, args)
    configure_physx(env_cfg)
    configured_physics = {
        "cfg": _fqn(env_cfg.sim.physics),
        "use_newton_actuators": getattr(env_cfg.sim, "use_newton_actuators", None),
    }
    if configured_physics["cfg"] != PHYSX_CFG_FQN:
        raise RuntimeError(f"Teleop config is not PhysxCfg: {configured_physics['cfg']}")
    if configured_physics["use_newton_actuators"] is not False:
        raise RuntimeError("Teleop requires use_newton_actuators=False")
    agent_cfg = load_cfg_from_registry(args.task, "crl2_cfg_entry_point")
    if not isinstance(agent_cfg, dict):
        raise RuntimeError("RAMBO CRL2 configuration must be a dictionary")
    agent_cfg["seed"] = args.seed
    agent_cfg["general"]["num_envs"] = 1

    gui_artifact: GuiKeyboardArtifactWriter | None = None
    gui_backend_before: dict[str, Any] | None = None
    gui_backend_after: dict[str, Any] | None = None
    gui_handled_key_presses = 0
    gui_unhandled_space_presses = 0
    gui_active_command_states = 0
    gui_base_nonzero_states = 0
    gui_fl_nonzero_velocity_states = 0
    gui_max_fl_target_delta_m = 0.0
    gui_max_root_position_delta_m = 0.0
    gui_max_button_displacement_m = 0.0
    gui_button_press_threshold_seen = False
    gui_button_success_after_threshold_seen = False
    gui_first_button_success_action_step: int | None = None
    gui_button_rebound_after_success_seen = False
    gui_first_button_rebound_action_step: int | None = None
    gui_initial_root_position: np.ndarray | None = None
    gui_runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
    }
    step_count = 0
    env = None
    try:
        env = Crl2VecEnvWrapper(gym.make(args.task, cfg=env_cfg))
        physics_evidence = assert_physx_environment(env)
        gui_backend_before = physics_evidence
        print(
            "[PHYSX] "
            f"manager={physics_evidence['actual_manager']} "
            f"use_newton_actuators={physics_evidence['use_newton_actuators']}",
            flush=True,
        )
        seed_everything(args.seed, env)
        observations, _ = env.reset()
        validate_environment_contract(env, contract)
        base_env = env.unwrapped
        if button_task:
            _validate_button_environment(base_env)
        else:
            _validate_loco_manip_environment(base_env)
            _validate_lift_basket_source_physics()
        if args.view == VIEW_EGO:
            _activate_ego_viewport(base_env)
        elif args.view == VIEW_TASK:
            from isaacsim.core.rendering_manager import ViewportManager
            if not ViewportManager.wait_for_viewport(max_frames=120)[0]:
                raise RuntimeError("Kit viewport not ready")
            _select_mounted_view(VIEW_TASK)
        base_env._record_termination_diagnostics = True

        runner = PPO(task=args.task, env=env, agent_cfg=agent_cfg, train=False, device=env.device)
        restore_runner(runner, checkpoint, load_values=False, verify=True)
        policy = runner.get_inference_policy(device=base_env.device)

        if args.smoke_press:
            _prepare_smoke_press_start(base_env, torch)
            env._last_observations = base_env.obs_buf
            observations, _ = env.get_observations()

        headless = getattr(args, "rambo_visualizer", None) == ["none"]
        if gui_artifact_enabled and headless:
            raise RuntimeError("GUI artifact mode cannot use the no-visualizer input path")
        if gui_artifact_enabled:
            gui_runtime["device"] = str(base_env.device)
            gui_started_at_utc = _utc_now()
            gui_started_monotonic_ns = time.monotonic_ns()
            gui_manifest = {
                "schema_version": gui_artifact_schema.SCHEMA_VERSION,
                "artifact_kind": gui_artifact_schema.ARTIFACT_KIND,
                "task": gui_artifact_schema.TASK_ID,
                "mode": gui_artifact_schema.MODE,
                "visualizer": gui_artifact_schema.VISUALIZER,
                "smoke_mode": False,
                "max_action_steps": args.max_steps,
                "arm_timeout_s": float(args.gui_arm_timeout_s),
                "initial_leg_target": EE_DEFAULT.tolist(),
                "files": {
                    "events": gui_artifact_schema.EVENTS_FILENAME,
                    "states": gui_artifact_schema.STATES_FILENAME,
                    "attestation": gui_artifact_schema.ATTESTATION_FILENAME,
                    "process_exit": gui_artifact_schema.PROCESS_EXIT_FILENAME,
                },
                "post_close_exit": {
                    "required": True,
                    "file": gui_artifact_schema.PROCESS_EXIT_FILENAME,
                    "captured_by": gui_artifact_schema.POST_CLOSE_RUNNER,
                },
                "post_close_attestation": {
                    "required": True,
                    "file": gui_artifact_schema.ATTESTATION_FILENAME,
                    "confirmation_method": gui_artifact_schema.ATTESTATION_CONFIRMATION_METHOD,
                },
                "input_capture": {
                    "source": "carb_keyboard_callback",
                    "observed_only": True,
                    "synthetic_input_injected": False,
                    "physical_keyboard_independently_proven": False,
                    "limitation": gui_artifact_schema.PHYSICALITY_LIMITATION,
                },
                "telemetry": {
                    "states_file": gui_artifact_schema.STATES_FILENAME,
                    "interval_action_steps": 1,
                },
                "configured_physics": configured_physics,
                "backend_before": gui_backend_before,
                "started_at_utc": gui_started_at_utc,
                "started_monotonic_ns": gui_started_monotonic_ns,
            }
            gui_artifact = GuiKeyboardArtifactWriter(
                args.gui_artifact_dir,
                manifest=gui_manifest,
            )

        def observe_gui_keyboard_event(record: dict[str, Any]) -> None:
            nonlocal gui_handled_key_presses, gui_unhandled_space_presses
            if gui_artifact is None:
                return
            gui_artifact.record_event(record)
            if record["event_type"] == "KEY_PRESS" and record["handled_by_loco_manip"]:
                gui_handled_key_presses += 1
            if (
                record["key"] == "SPACE"
                and record["event_type"] == "KEY_PRESS"
                and record["handled_by_loco_manip"] is False
            ):
                gui_unhandled_space_presses += 1

        teleop = (
            _HeadlessInput()
            if headless
            else _make_keyboard(
                base_env,
                args,
                event_observer=observe_gui_keyboard_event if gui_artifact is not None else None,
            )
        )
        teleop.reset()
        if not headless and getattr(args, "camera_setup", None) in ("robot-dual-v3",):
            teleop.add_callback("F6", lambda: _select_mounted_view(VIEW_EGO))
            teleop.add_callback("F7", lambda: _select_mounted_view(VIEW_TASK))
            teleop.add_callback("F8", lambda: _select_mounted_view(VIEW_THIRD_PERSON))
            print("[CAMERAS] F6: native ego | F7: task camera | F8: third person", flush=True)
        leg_target = EE_DEFAULT.copy()
        prior_success = bool(
            base_env.button_success[0].item()
            if button_task
            else base_env.task_success[0].item()
        )
        prior_released = bool(base_env.button_released[0].item()) if button_task else False
        release_announced = False
        was_ready = False
        press_succeeded = False
        press_released = False
        step_count = 0
        if args.max_steps:
            max_steps = args.max_steps
        elif args.smoke_loco_manip:
            max_steps = 950
        elif args.smoke_press:
            max_steps = 800
        elif args.smoke_walk:
            max_steps = 350
        else:
            max_steps = 0
        smoke_mode = args.smoke_walk or args.smoke_press or args.smoke_loco_manip
        press_mode = args.smoke_press or args.smoke_loco_manip
        retract_start = 810 if args.smoke_loco_manip else 610
        retract_end = 910 if args.smoke_loco_manip else 710
        walk_start_x = float(base_env._robot.data.root_link_pos_w.torch[0, 0].item())
        walk_start_joint_pos = base_env._robot.data.joint_pos.torch.clone()

        print(f"[INFO] Verified checkpoint: {args.checkpoint.resolve()}")
        if not headless:
            print(teleop)
            print(
                f"[TELEOP] view={args.view}. Click the Isaac Sim viewport before using the keyboard.",
                flush=True,
            )
            print("LOCO_MANIP_TELEOP_READY", flush=True)
        if gui_artifact is not None:
            print(
                "[GUI-ARTIFACT] Click the viewport, then press a supported keyboard key to arm "
                f"the finite {max_steps}-step recording (timeout {args.gui_arm_timeout_s:.1f}s).",
                flush=True,
            )
            arm_deadline = time.monotonic() + float(args.gui_arm_timeout_s)
            while simulation_app.is_running() and gui_handled_key_presses == 0:
                if time.monotonic() >= arm_deadline:
                    raise RuntimeError("GUI artifact did not observe a supported native keyboard press before timeout")
                simulation_app.update()
            if gui_handled_key_presses == 0:
                raise RuntimeError("GUI artifact closed before a supported native keyboard press armed the run")
            print("[GUI-ARTIFACT] Native keyboard callback observed; recording action states.", flush=True)

        while simulation_app.is_running() and (max_steps == 0 or step_count < max_steps):
            with torch.inference_mode():
                ready = bool(base_env.manipulator_ready[0].item())
                displacement = float(base_env.button_displacement[0].item()) if button_task else 0.0
                success = bool(
                    base_env.button_success[0].item()
                    if button_task
                    else base_env.task_success[0].item()
                )
                if ready and not was_ready:
                    print("[FL] ready: FL is in swing/manipulator mode.", flush=True)
                was_ready = ready

                if (
                    button_task
                    and teleop.consume_clear_success_request()
                    and bool(base_env.button_released[0].item())
                ):
                    cleared = base_env.clear_button_success()
                    if bool(cleared[0].item()):
                        release_announced = False
                        print("[BUTTON] CLEARED", flush=True)

                if args.smoke_press:
                    base_command = np.zeros(3, dtype=np.float32)
                    leg_velocity = np.zeros(3, dtype=np.float32)
                    if 100 <= step_count < 350:
                        leg_velocity[2] = args.fl_z_speed
                    elif 350 <= step_count < 610:
                        leg_velocity[0] = args.fl_x_speed
                    elif 610 <= step_count < 710:
                        leg_velocity[0] = -args.fl_x_speed
                elif args.smoke_loco_manip:
                    base_command = np.zeros(3, dtype=np.float32)
                    leg_velocity = np.zeros(3, dtype=np.float32)
                    if 110 <= step_count < 270:
                        base_command[0] = args.vx_sensitivity
                    elif 300 <= step_count < 550:
                        leg_velocity[2] = args.fl_z_speed
                    elif 550 <= step_count < 810:
                        leg_velocity[0] = args.fl_x_speed
                    elif 810 <= step_count < 910:
                        leg_velocity[0] = -args.fl_x_speed
                elif args.smoke_walk:
                    base_command = np.array([args.vx_sensitivity, 0.0, 0.0], dtype=np.float32)
                    leg_velocity = np.zeros(3, dtype=np.float32)
                    if step_count < 110 or step_count >= 250:
                        base_command.fill(0.0)
                else:
                    base_command_raw = teleop.advance()
                    if hasattr(base_command_raw, "detach"):
                        base_command_raw = base_command_raw.detach().cpu().numpy()
                    base_command = np.asarray(base_command_raw, dtype=np.float32).copy()
                    leg_velocity = teleop.leg_velocity()

                if button_task and displacement >= base_env.cfg.button_contact_guard_m:
                    base_command[0] = min(base_command[0], 0.0)
                if button_task and success:
                    base_command.fill(0.0)
                    leg_velocity[0] = min(leg_velocity[0], 0.0)
                if ready:
                    leg_target = np.clip(leg_target + leg_velocity * base_env.step_dt, EE_MIN, EE_MAX)

                base_env.set_loco_manip_commands(
                    base_velocity=torch.as_tensor(base_command, device=base_env.device).view(1, 3),
                    fl_position=torch.as_tensor(leg_target, device=base_env.device).view(1, 3),
                    fl_force=torch.zeros((1, 3), device=base_env.device, dtype=torch.float32),
                )

                actions = policy(observations)
                if not torch.isfinite(actions).all():
                    raise RuntimeError(f"Policy action is non-finite at step {step_count}")
                observations, _, dones, _ = env.step(actions)
                step_count += 1
                if bool(torch.any(dones)):
                    teleop.reset()
                    mode = "smoke test" if smoke_mode else "teleoperation"
                    diagnostics = {
                        name: value.detach().cpu().tolist()
                        for name, value in getattr(
                            base_env,
                            "_last_termination_diagnostics",
                            {},
                        ).items()
                    }
                    print(
                        f"[TERMINATION] mode={mode} step={step_count} diagnostics={diagnostics}",
                        flush=True,
                    )
                    if not smoke_mode and gui_artifact is None:
                        # DirectRLEnv has already reset the terminated single
                        # environment before returning the new observations.
                        # Reset only the operator-side command state and keep
                        # the interactive Kit window alive.
                        leg_target = EE_DEFAULT.copy()
                        prior_success = bool(
                            base_env.button_success[0].item()
                            if button_task
                            else base_env.task_success[0].item()
                        )
                        prior_released = (
                            bool(base_env.button_released[0].item())
                            if button_task
                            else False
                        )
                        release_announced = False
                        was_ready = False
                        print(
                            "[TELEOP] Safety termination auto-reset; GUI remains active.",
                            flush=True,
                        )
                        continue
                    raise RuntimeError(
                        f"Environment terminated during {mode} at step {step_count}"
                    )

                new_success = bool(
                    base_env.button_success[0].item()
                    if button_task
                    else base_env.task_success[0].item()
                )
                if button_task and new_success and not prior_success:
                    press_succeeded = press_succeeded or press_mode
                    release_announced = False
                    print(
                        "[BUTTON] PRESSED: "
                        f"travel={float(base_env.button_displacement[0].item()) * 1000.0:.1f} mm "
                        f"threshold={base_env.cfg.button_press_threshold_m * 1000.0:.1f} mm "
                        f"sustained_steps>={base_env.cfg.button_hold_steps}",
                        flush=True,
                    )
                elif not button_task and new_success and not prior_success:
                    metrics = base_env.task_metrics
                    print(
                        "[LIFT] SUCCESS: "
                        f"clearance={float(metrics['clearance_m'][0]):.3f}m "
                        f"tilt={float(metrics['tilt_rad'][0]):.3f}rad",
                        flush=True,
                    )
                prior_success = new_success
                new_released = bool(base_env.button_released[0].item()) if button_task else False
                if button_task and new_success and not release_announced and not prior_released and new_released:
                    print(
                        "[BUTTON] RELEASED: "
                        f"travel={float(base_env.button_displacement[0].item()) * 1000.0:.1f} mm "
                        f"threshold={base_env.cfg.button_release_threshold_m * 1000.0:.1f} mm",
                        flush=True,
                    )
                    release_announced = True
                prior_released = new_released
                if (
                    button_task
                    and press_mode
                    and press_succeeded
                    and bool(base_env.button_released[0].item())
                    and step_count >= retract_start
                ):
                    press_released = True

                if gui_artifact is not None:
                    state_record = _gui_state_record(
                        base_env,
                        step_count=step_count,
                        base_command=base_command,
                        leg_target=leg_target,
                        leg_velocity=leg_velocity,
                        callback_events_observed=gui_artifact.event_count,
                    )
                    gui_artifact.record_state(state_record)
                    base_nonzero = bool(np.any(np.abs(base_command) > 1.0e-9))
                    fl_nonzero_velocity = bool(np.any(np.abs(leg_velocity) > 1.0e-9))
                    if base_nonzero or fl_nonzero_velocity:
                        gui_active_command_states += 1
                    if base_nonzero:
                        gui_base_nonzero_states += 1
                    if fl_nonzero_velocity:
                        gui_fl_nonzero_velocity_states += 1
                    gui_max_fl_target_delta_m = max(
                        gui_max_fl_target_delta_m,
                        float(np.linalg.norm(np.asarray(leg_target, dtype=np.float64) - EE_DEFAULT)),
                    )
                    root_position = np.asarray(state_record["root_position_m"], dtype=np.float64)
                    if gui_initial_root_position is None:
                        gui_initial_root_position = root_position.copy()
                    gui_max_root_position_delta_m = max(
                        gui_max_root_position_delta_m,
                        float(np.linalg.norm(root_position - gui_initial_root_position)),
                    )
                    button_displacement_m = float(state_record["button_displacement_m"])
                    gui_max_button_displacement_m = max(
                        gui_max_button_displacement_m,
                        button_displacement_m,
                    )
                    if button_displacement_m >= gui_artifact_schema.BUTTON_PRESS_THRESHOLD_M:
                        gui_button_press_threshold_seen = True
                    if bool(state_record["button_success"]) and gui_button_press_threshold_seen:
                        gui_button_success_after_threshold_seen = True
                        if gui_first_button_success_action_step is None:
                            gui_first_button_success_action_step = step_count
                    if (
                        gui_first_button_success_action_step is not None
                        and step_count > gui_first_button_success_action_step
                        and bool(state_record["button_released"])
                        and button_displacement_m <= gui_artifact_schema.BUTTON_REBOUND_THRESHOLD_M
                    ):
                        gui_button_rebound_after_success_seen = True
                        if gui_first_button_rebound_action_step is None:
                            gui_first_button_rebound_action_step = step_count

                report_every = 50 if smoke_mode else args.telemetry_every
                if report_every and step_count % report_every == 0:
                    report_label = "SMOKE" if smoke_mode else "STATE"
                    prefix = (
                        f"[{report_label}] step={step_count} "
                        f"root_x={float(base_env._robot.data.root_link_pos_w.torch[0, 0]):.3f} "
                        f"fl=({leg_target[0]:.3f},{leg_target[1]:.3f},{leg_target[2]:.3f}) "
                    )
                    if button_task:
                        detail = (
                            f"button={float(base_env.button_displacement[0]) * 1000.0:.1f}mm "
                            f"success={int(new_success)} released={int(new_released)}"
                        )
                    else:
                        metrics = base_env.task_metrics
                        detail = (
                            f"clearance={float(metrics['clearance_m'][0]):.3f}m "
                            f"tilt={float(metrics['tilt_rad'][0]):.3f}rad "
                            f"success={int(new_success)}"
                        )
                    print(prefix + detail, flush=True)
                if press_mode and press_succeeded and press_released and step_count >= retract_end:
                    break

        if gui_artifact is not None and step_count != max_steps:
            raise RuntimeError(
                "GUI artifact run ended before its required finite --max-steps limit: "
                f"executed {step_count}, expected {max_steps}"
            )
        physics_evidence_after = assert_physx_environment(env)
        gui_backend_after = physics_evidence_after
        print(
            "[PHYSX] post "
            f"manager={physics_evidence_after['actual_manager']} "
            f"use_newton_actuators={physics_evidence_after['use_newton_actuators']}",
            flush=True,
        )
        gui_evidence: dict[str, Any] | None = None
        if gui_artifact is not None:
            gui_evidence = _gui_m8_evidence(
                base_nonzero_state_count=gui_base_nonzero_states,
                fl_nonzero_velocity_state_count=gui_fl_nonzero_velocity_states,
                unhandled_space_press_count=gui_unhandled_space_presses,
                max_fl_target_delta_m=gui_max_fl_target_delta_m,
                max_root_position_delta_m=gui_max_root_position_delta_m,
                max_button_displacement_m=gui_max_button_displacement_m,
                button_press_threshold_seen=gui_button_press_threshold_seen,
                button_success_after_threshold_seen=gui_button_success_after_threshold_seen,
                first_button_success_action_step=gui_first_button_success_action_step,
                button_rebound_after_success_seen=gui_button_rebound_after_success_seen,
                first_button_rebound_action_step=gui_first_button_rebound_action_step,
            )
            _assert_complete_gui_m8_evidence(gui_artifact_schema, gui_evidence)
        if press_mode:
            if not press_succeeded:
                raise RuntimeError("Physical button did not reach the 12-mm press threshold")
            if not press_released:
                raise RuntimeError("Button pressed but did not spring back after FL retraction")
            if args.smoke_loco_manip:
                walked = float(base_env._robot.data.root_link_pos_w.torch[0, 0].item()) - walk_start_x
                joint_delta = float(
                    torch.max(torch.abs(base_env._robot.data.joint_pos.torch - walk_start_joint_pos)).item()
                )
                print(f"[SMOKE] forward displacement={walked:.3f}m joint_delta={joint_delta:.3f}rad")
                if walked < 0.05 or joint_delta < 0.05:
                    raise RuntimeError("Combined smoke test did not produce measurable walking motion")
                print("LOCO_MANIP_SMOKE_LOCO_MANIP_SUCCESS", flush=True)
            else:
                print("LOCO_MANIP_SMOKE_PRESS_SUCCESS", flush=True)
        elif args.smoke_walk:
            walked = float(base_env._robot.data.root_link_pos_w.torch[0, 0].item()) - walk_start_x
            joint_delta = float(torch.max(torch.abs(base_env._robot.data.joint_pos.torch - walk_start_joint_pos)).item())
            print(f"[SMOKE] forward displacement={walked:.3f}m joint_delta={joint_delta:.3f}rad")
            if walked < 0.05 or joint_delta < 0.05:
                raise RuntimeError("Walking smoke test did not produce measurable base and joint motion")
            print("LOCO_MANIP_SMOKE_WALK_SUCCESS", flush=True)
        else:
            prefix = (
                f"[FINAL] root_x={float(base_env._robot.data.root_link_pos_w.torch[0, 0].item()):.3f} "
                f"fl=({leg_target[0]:.3f},{leg_target[1]:.3f},{leg_target[2]:.3f}) "
            )
            if button_task:
                detail = (
                    f"button={float(base_env.button_displacement[0].item()) * 1000.0:.1f}mm "
                    f"success={int(bool(base_env.button_success[0].item()))} "
                    f"released={int(bool(base_env.button_released[0].item()))}"
                )
            else:
                metrics = base_env.task_metrics
                detail = (
                    f"clearance={float(metrics['clearance_m'][0]):.3f}m "
                    f"tilt={float(metrics['tilt_rad'][0]):.3f}rad "
                    f"success={int(bool(base_env.task_success[0].item()))}"
                )
            print(prefix + detail, flush=True)
            print(f"LOCO_MANIP_STEPS={step_count}", flush=True)
        if gui_artifact is not None:
            gui_artifact.finalize(
                _gui_artifact_summary(
                    gui_artifact_schema,
                    args=args,
                    outcome="completed",
                    executed_action_steps=step_count,
                    event_count=gui_artifact.event_count,
                    state_count=gui_artifact.state_count,
                    handled_key_press_count=gui_handled_key_presses,
                    active_command_state_count=gui_active_command_states,
                    configured_physics=configured_physics,
                    backend_before=gui_backend_before,
                    backend_after=gui_backend_after,
                    checkpoint_sha256=contract.sha256,
                    runtime=gui_runtime,
                    m8_evidence=gui_evidence,
                )
            )
            print(
                "[GUI-ARTIFACT] Pre-close evidence written; acceptance requires the dedicated "
                f"post-close runner {gui_artifact_schema.POST_CLOSE_RUNNER}.",
                flush=True,
            )
            gui_artifact = None
        return 0
    except BaseException as error:
        if gui_artifact is not None:
            try:
                gui_artifact.finalize(
                    _gui_artifact_summary(
                        gui_artifact_schema,
                        args=args,
                        outcome="failed",
                        executed_action_steps=step_count,
                        event_count=gui_artifact.event_count,
                        state_count=gui_artifact.state_count,
                        handled_key_press_count=gui_handled_key_presses,
                        active_command_state_count=gui_active_command_states,
                        configured_physics=configured_physics,
                        backend_before=gui_backend_before,
                        backend_after=gui_backend_after,
                        checkpoint_sha256=contract.sha256,
                        runtime=gui_runtime,
                        m8_evidence=_gui_m8_evidence(
                            base_nonzero_state_count=gui_base_nonzero_states,
                            fl_nonzero_velocity_state_count=gui_fl_nonzero_velocity_states,
                            unhandled_space_press_count=gui_unhandled_space_presses,
                            max_fl_target_delta_m=gui_max_fl_target_delta_m,
                            max_root_position_delta_m=gui_max_root_position_delta_m,
                            max_button_displacement_m=gui_max_button_displacement_m,
                            button_press_threshold_seen=gui_button_press_threshold_seen,
                            button_success_after_threshold_seen=gui_button_success_after_threshold_seen,
                            first_button_success_action_step=gui_first_button_success_action_step,
                            button_rebound_after_success_seen=gui_button_rebound_after_success_seen,
                            first_button_rebound_action_step=gui_first_button_rebound_action_step,
                        ),
                        failure=error,
                    )
                )
            except BaseException as finalize_error:
                print(
                    f"[GUI-ARTIFACT] failed to finalize evidence: {type(finalize_error).__name__}: {finalize_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    finally:
        if env is not None:
            env.close()


def main() -> int:
    parser, app_launcher_type = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    try:
        from rambo.utils.physx import validate_rambo_visualizer_args
    except ModuleNotFoundError as error:  # pragma: no cover - target-runtime guard.
        raise RuntimeError("Run this script through scripts/rambo/run.sh") from error
    args.rambo_visualizer = validate_rambo_visualizer_args(parser, args, sys.argv[1:])
    _validate_view_args(parser, args)
    _configure_ego_view_launcher(args)
    _validate_gui_artifact_args(parser, args)
    app_launcher = app_launcher_type(args)
    simulation_app = app_launcher.app
    exit_code = 1
    try:
        exit_code = _run(args, simulation_app)
        return exit_code
    except BaseException as error:
        print("RAMBO loco-manip teleoperation failed:", file=sys.stderr, flush=True)
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
        return 1
    finally:
        # Kit fast shutdown exits the process here; retain failures as failures.
        simulation_app.close(exit_code=exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
