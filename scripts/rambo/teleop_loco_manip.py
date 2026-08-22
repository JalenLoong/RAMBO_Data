#!/usr/bin/env python3
"""Keyboard loco-manip teleoperation for Go2 walking and FL button pressing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np


TASK_ID = "Isaac-RAMBO-Quadruped-Button-Go2-v0"
EE_MIN = np.array([0.1934, 0.0, 0.0], dtype=np.float32)
EE_MAX = np.array([0.50, 0.20, 0.40], dtype=np.float32)
EE_DEFAULT = np.array([0.1934, 0.142, 0.05], dtype=np.float32)


def _keyboard_event_name(event: Any) -> str:
    """Normalize Carb keyboard inputs across native and synthetic 5.1 events."""

    raw_key = event.input
    return raw_key.name if hasattr(raw_key, "name") else str(raw_key)


def _build_parser() -> tuple[argparse.ArgumentParser, type]:
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as error:  # pragma: no cover - target-runtime guard.
        raise RuntimeError("Run this script through scripts/rambo/run.sh") from error

    parser = argparse.ArgumentParser(
        description="Teleoperate RAMBO Go2 locomotion and its FL button-pressing leg."
    )
    parser.add_argument("--task", choices=(TASK_ID,), default=TASK_ID)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted quadruped model_2000.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-length-s", type=float, default=300.0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs until the GUI closes")
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


def _validate_button_environment(base_env: Any) -> None:
    required = (
        "_velocity_commands",
        "_ee_pos_commands",
        "_ee_force_commands",
        "button_displacement",
        "button_success",
        "button_released",
        "manipulator_ready",
        "clear_button_success",
    )
    missing = [name for name in required if not hasattr(base_env, name)]
    if missing:
        raise RuntimeError("Button task is missing loco-manip interfaces: " + ", ".join(missing))
    if tuple(base_env._velocity_commands.shape) != (1, 3):
        raise RuntimeError(f"Unexpected velocity command shape: {base_env._velocity_commands.shape}")
    if tuple(base_env._ee_pos_commands.shape) != (1, 3):
        raise RuntimeError(f"Unexpected FL position command shape: {base_env._ee_pos_commands.shape}")
    if tuple(base_env._ee_force_commands.shape) != (1, 3):
        raise RuntimeError(f"Unexpected FL force command shape: {base_env._ee_force_commands.shape}")


def _prepare_smoke_press_start(base_env: Any, torch: Any) -> None:
    """Place Go2 where the trained FL target can reach the physical cap."""

    env_ids = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long)
    root_state = base_env._robot.data.default_root_state.clone()
    root_state[:, 0] = 0.58
    root_state[:, 1] = 0.0
    root_state[:, 2] += 0.10
    root_state[:, 7:] = 0.0
    joint_pos = base_env._robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(base_env._robot.data.default_joint_vel)
    base_env._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
    base_env._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
    base_env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    base_env._velocity_commands.zero_()
    base_env._ee_pos_commands.copy_(
        torch.as_tensor(EE_DEFAULT, device=base_env.device).view(1, 3)
    )
    base_env._ee_force_commands.zero_()
    base_env.scene.write_data_to_sim()
    base_env.sim.forward()
    base_env._obs_history.zero_()
    base_env.contact_generator.reset_idx(env_ids)
    base_env._desired_joint_pos[env_ids] = base_env.joint_position_controller.reset_idx(env_ids)
    base_env._last_action[env_ids] = 0.0
    base_env.obs_buf = base_env._get_observations()


def _make_keyboard(base_env: Any, args: argparse.Namespace):
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

        def _on_keyboard_event(self, event, *callback_args, **callback_kwargs):
            key = _keyboard_event_name(event)
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

        def __str__(self) -> str:
            return (
                super().__str__()
                + "\n\tFL forward/back: W / S"
                + "\n\tFL lateral +/-: A / D"
                + "\n\tFL up/down: R / F"
                + "\n\tStop and reset all commands: L"
                + "\n\tClear released-button success: C"
            )

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


def _run(args: argparse.Namespace, simulation_app: Any) -> int:
    from rambo.torch_runtime import ensure_cuda_linalg_loaded

    ensure_cuda_linalg_loaded()
    import gymnasium as gym
    import rambo
    import torch
    from crl2.algorithms import PPO
    from rambo.rl import Crl2VecEnvWrapper
    from rambo.utils.registry import load_cfg_from_registry, parse_env_cfg
    from rambo.validation.checkpoints import contract_for_task, load_verified_checkpoint, restore_runner
    from rambo.validation.rollout import seed_everything, validate_environment_contract

    rambo.register_tasks()
    contract = contract_for_task(args.task)
    checkpoint = load_verified_checkpoint(args.checkpoint, contract)
    env_cfg = parse_env_cfg(args.task, num_envs=1, use_fabric=not args.disable_fabric)
    _configure_environment(env_cfg, args)
    agent_cfg = load_cfg_from_registry(args.task, "crl2_cfg_entry_point")
    if not isinstance(agent_cfg, dict):
        raise RuntimeError("RAMBO CRL2 configuration must be a dictionary")
    agent_cfg["seed"] = args.seed
    agent_cfg["general"]["num_envs"] = 1

    env = None
    try:
        env = Crl2VecEnvWrapper(gym.make(args.task, cfg=env_cfg))
        seed_everything(args.seed, env)
        observations, _ = env.reset()
        validate_environment_contract(env, contract)
        base_env = env.unwrapped
        _validate_button_environment(base_env)

        runner = PPO(task=args.task, env=env, agent_cfg=agent_cfg, train=False, device=env.device)
        restore_runner(runner, checkpoint, load_values=False, verify=True)
        policy = runner.get_inference_policy(device=base_env.device)

        if args.smoke_press:
            _prepare_smoke_press_start(base_env, torch)
            env._last_observations = base_env.obs_buf
            observations, _ = env.get_observations()

        headless = bool(getattr(args, "headless", False))
        teleop = _HeadlessInput() if headless else _make_keyboard(base_env, args)
        teleop.reset()
        leg_target = EE_DEFAULT.copy()
        prior_success = False
        prior_released = bool(base_env.button_released[0].item())
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
        walk_start_x = float(base_env._robot.data.root_pos_w[0, 0].item())
        walk_start_joint_pos = base_env._robot.data.joint_pos.clone()

        print(f"[INFO] Verified checkpoint: {args.checkpoint.resolve()}")
        if not headless:
            print(teleop)
            print("[TELEOP] Click the Isaac Sim viewport before using the keyboard.", flush=True)
            print("LOCO_MANIP_TELEOP_READY", flush=True)

        while simulation_app.is_running() and (max_steps == 0 or step_count < max_steps):
            with torch.inference_mode():
                ready = bool(base_env.manipulator_ready[0].item())
                displacement = float(base_env.button_displacement[0].item())
                success = bool(base_env.button_success[0].item())
                if ready and not was_ready:
                    print("[FL] ready: FL is in swing/manipulator mode.", flush=True)
                was_ready = ready

                if teleop.consume_clear_success_request() and bool(base_env.button_released[0].item()):
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

                if displacement >= base_env.cfg.button_contact_guard_m:
                    base_command[0] = min(base_command[0], 0.0)
                if success:
                    base_command.fill(0.0)
                    leg_velocity[0] = min(leg_velocity[0], 0.0)
                if ready:
                    leg_target = np.clip(leg_target + leg_velocity * base_env.step_dt, EE_MIN, EE_MAX)

                base_env._velocity_commands.copy_(
                    torch.as_tensor(base_command, device=base_env.device).view(1, 3)
                )
                base_env._ee_pos_commands.copy_(
                    torch.as_tensor(leg_target, device=base_env.device).view(1, 3)
                )
                base_env._ee_force_commands.zero_()

                actions = policy(observations)
                if not torch.isfinite(actions).all():
                    raise RuntimeError(f"Policy action is non-finite at step {step_count}")
                observations, _, dones, _ = env.step(actions)
                step_count += 1
                if bool(torch.any(dones)):
                    teleop.reset()
                    mode = "smoke test" if smoke_mode else "teleoperation"
                    raise RuntimeError(f"Environment terminated during {mode} at step {step_count}")

                new_success = bool(base_env.button_success[0].item())
                if new_success and not prior_success:
                    press_succeeded = press_succeeded or press_mode
                    release_announced = False
                    print(
                        f"[BUTTON] PRESSED: travel={float(base_env.button_displacement[0].item()) * 1000.0:.1f} mm",
                        flush=True,
                    )
                prior_success = new_success
                new_released = bool(base_env.button_released[0].item())
                if new_success and not release_announced and not prior_released and new_released:
                    print("[BUTTON] RELEASED", flush=True)
                    release_announced = True
                prior_released = new_released
                if (
                    press_mode
                    and press_succeeded
                    and bool(base_env.button_released[0].item())
                    and step_count >= retract_start
                ):
                    press_released = True

                report_every = 50 if smoke_mode else args.telemetry_every
                if report_every and step_count % report_every == 0:
                    report_label = "SMOKE" if smoke_mode else "STATE"
                    print(
                        f"[{report_label}] step={step_count} "
                        f"root_x={float(base_env._robot.data.root_pos_w[0, 0]):.3f} "
                        f"fl=({leg_target[0]:.3f},{leg_target[1]:.3f},{leg_target[2]:.3f}) "
                        f"button={float(base_env.button_displacement[0]) * 1000.0:.1f}mm "
                        f"success={int(new_success)} released={int(new_released)}",
                        flush=True,
                    )
                if press_mode and press_succeeded and press_released and step_count >= retract_end:
                    break

        if press_mode:
            if not press_succeeded:
                raise RuntimeError("Physical button did not reach the 12-mm press threshold")
            if not press_released:
                raise RuntimeError("Button pressed but did not spring back after FL retraction")
            if args.smoke_loco_manip:
                walked = float(base_env._robot.data.root_pos_w[0, 0].item()) - walk_start_x
                joint_delta = float(
                    torch.max(torch.abs(base_env._robot.data.joint_pos - walk_start_joint_pos)).item()
                )
                print(f"[SMOKE] forward displacement={walked:.3f}m joint_delta={joint_delta:.3f}rad")
                if walked < 0.05 or joint_delta < 0.05:
                    raise RuntimeError("Combined smoke test did not produce measurable walking motion")
                print("LOCO_MANIP_SMOKE_LOCO_MANIP_SUCCESS", flush=True)
            else:
                print("LOCO_MANIP_SMOKE_PRESS_SUCCESS", flush=True)
        elif args.smoke_walk:
            walked = float(base_env._robot.data.root_pos_w[0, 0].item()) - walk_start_x
            joint_delta = float(torch.max(torch.abs(base_env._robot.data.joint_pos - walk_start_joint_pos)).item())
            print(f"[SMOKE] forward displacement={walked:.3f}m joint_delta={joint_delta:.3f}rad")
            if walked < 0.05 or joint_delta < 0.05:
                raise RuntimeError("Walking smoke test did not produce measurable base and joint motion")
            print("LOCO_MANIP_SMOKE_WALK_SUCCESS", flush=True)
        else:
            print(
                f"[FINAL] root_x={float(base_env._robot.data.root_pos_w[0, 0].item()):.3f} "
                f"fl=({leg_target[0]:.3f},{leg_target[1]:.3f},{leg_target[2]:.3f}) "
                f"button={float(base_env.button_displacement[0].item()) * 1000.0:.1f}mm "
                f"success={int(bool(base_env.button_success[0].item()))} "
                f"released={int(bool(base_env.button_released[0].item()))}",
                flush=True,
            )
            print(f"LOCO_MANIP_STEPS={step_count}", flush=True)
        return 0
    finally:
        if env is not None:
            env.close()


def main() -> int:
    parser, app_launcher_type = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    app_launcher = app_launcher_type(args)
    simulation_app = app_launcher.app
    try:
        return _run(args, simulation_app)
    except BaseException as error:
        print("RAMBO loco-manip teleoperation failed:", file=sys.stderr, flush=True)
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
