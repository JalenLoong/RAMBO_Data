#!/usr/bin/env python3
"""Replay a verified RAMBO CRL2 checkpoint in Isaac Sim 6 / Isaac Lab 3 PhysX."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback


def _build_parser() -> tuple[argparse.ArgumentParser, type]:
    """Create the CLI parser without importing Isaac Sim at module import time."""

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:  # pragma: no cover - target-runtime guard.
        raise RuntimeError(
            "scripts/rambo/play.py must be run inside the RAMBO Isaac Lab environment"
        ) from exc

    parser = argparse.ArgumentParser(description="Replay a verified RAMBO CRL2 checkpoint.")
    parser.add_argument("--task", choices=(
        "Isaac-RAMBO-Quadruped-Go2-v0",
    ), required=True, help="Registered RAMBO Gym task.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted local model_*.pt path.")
    parser.add_argument("--seed", type=int, default=42, help="Policy/environment seed.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of synchronized environments.")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Finite replay length; 0 keeps playing until the simulator is closed.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use the fixed-seed, no-randomization validation configuration (requires --steps > 0).",
    )
    parser.add_argument(
        "--enable-rgb-camera",
        action="store_true",
        help="Enable the RAMBO front RGB camera and Isaac Sim camera rendering.",
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


def _runtime_imports():
    """Import simulator-dependent modules after AppLauncher has started Isaac Sim."""

    from rambo.torch_runtime import ensure_cuda_linalg_loaded

    ensure_cuda_linalg_loaded()
    import gymnasium as gym
    import rambo
    import torch
    from crl2.algorithms import PPO
    from rambo.rl import Crl2VecEnvWrapper
    from rambo.utils.physx import assert_physx_environment, configure_physx
    from rambo.utils.registry import load_cfg_from_registry, parse_env_cfg
    from rambo.validation.checkpoints import (
        contract_for_task,
        load_verified_checkpoint,
        restore_runner,
    )
    from rambo.validation.rollout import (
        RolloutValidationError,
        configure_validation_cfg,
        seed_everything,
        validate_environment_contract,
    )

    rambo.register_tasks()
    return {
        "gym": gym,
        "torch": torch,
        "PPO": PPO,
        "Crl2VecEnvWrapper": Crl2VecEnvWrapper,
        "RolloutValidationError": RolloutValidationError,
        "assert_physx_environment": assert_physx_environment,
        "configure_physx": configure_physx,
        "configure_validation_cfg": configure_validation_cfg,
        "contract_for_task": contract_for_task,
        "load_cfg_from_registry": load_cfg_from_registry,
        "load_verified_checkpoint": load_verified_checkpoint,
        "parse_env_cfg": parse_env_cfg,
        "restore_runner": restore_runner,
        "seed_everything": seed_everything,
        "validate_environment_contract": validate_environment_contract,
    }


def _print_failure(error: BaseException) -> None:
    """Report a runtime failure before leaving Kit without cleanup."""

    print("RAMBO playback failed:", file=sys.stderr, flush=True)
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    sys.stderr.flush()


def main() -> int:
    parser, app_launcher_type = _build_parser()
    args_cli = parser.parse_args()
    from rambo.utils.physx import validate_rambo_visualizer_args

    visualizer_selection = validate_rambo_visualizer_args(parser, args_cli, sys.argv[1:])
    if args_cli.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args_cli.steps < 0:
        parser.error("--steps must be non-negative")
    if args_cli.deterministic and args_cli.steps == 0:
        parser.error("--deterministic requires a finite --steps value")
    if args_cli.enable_rgb_camera:
        # AppLauncher chooses a camera-capable experience before the app exists.
        args_cli.enable_cameras = True

    app_launcher = app_launcher_type(args_cli)
    simulation_app = app_launcher.app
    print(f"RAMBO_PHYSX_VIZ={visualizer_selection}", flush=True)
    env = None
    captured_error: BaseException | None = None
    try:
        runtime = _runtime_imports()
        contract = runtime["contract_for_task"](args_cli.task)
        checkpoint = runtime["load_verified_checkpoint"](args_cli.checkpoint, contract)

        parse_kwargs = {
            "num_envs": args_cli.num_envs,
            "use_fabric": not args_cli.disable_fabric,
        }
        device = getattr(args_cli, "device", None)
        if device is not None:
            parse_kwargs["device"] = device
        env_cfg = runtime["parse_env_cfg"](args_cli.task, **parse_kwargs)
        # Repeat the selection at this production call site.  Registry-level
        # defaults are also configured, but a future default may never turn a
        # RAMBO replay into an implicit backend choice.
        runtime["configure_physx"](env_cfg)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args_cli.seed
        if args_cli.enable_rgb_camera:
            if not hasattr(env_cfg, "enable_rgb_camera"):
                raise runtime["RolloutValidationError"](
                    "Task configuration does not support a RAMBO front RGB camera"
                )
            env_cfg.enable_rgb_camera = True
        if args_cli.deterministic:
            duration_s = max(31.0, args_cli.steps * 0.01 + 1.0)
            runtime["configure_validation_cfg"](
                env_cfg,
                duration_s=duration_s,
                enable_rgb_camera=args_cli.enable_rgb_camera,
            )

        agent_cfg = runtime["load_cfg_from_registry"](args_cli.task, "crl2_cfg_entry_point")
        if not isinstance(agent_cfg, dict):
            raise RuntimeError("RAMBO CRL2 agent config must load as a YAML dictionary")
        agent_cfg["seed"] = args_cli.seed
        agent_cfg["general"]["num_envs"] = args_cli.num_envs

        env = runtime["Crl2VecEnvWrapper"](runtime["gym"].make(args_cli.task, cfg=env_cfg))
        print(f"PHYSX_BACKEND_BEFORE={runtime['assert_physx_environment'](env)}", flush=True)
        runtime["seed_everything"](args_cli.seed, env)
        observations, _ = env.reset()
        runtime["validate_environment_contract"](env, contract)

        runner = runtime["PPO"](
            task=args_cli.task,
            env=env,
            agent_cfg=agent_cfg,
            train=False,
            device=env.device,
        )
        runtime["restore_runner"](runner, checkpoint, load_values=False, verify=True)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        print(f"[INFO] Verified {contract.mode} checkpoint: {args_cli.checkpoint.resolve()}")

        completed_steps = 0
        while simulation_app.is_running() and (
            args_cli.steps == 0 or completed_steps < args_cli.steps
        ):
            with runtime["torch"].inference_mode():
                actions = policy(observations)
                if not runtime["torch"].isfinite(actions).all():
                    raise runtime["RolloutValidationError"](
                        f"policy action contains NaN or Inf at step {completed_steps}"
                    )
                observations, rewards, dones, _ = env.step(actions)
                if not runtime["torch"].isfinite(observations).all():
                    raise runtime["RolloutValidationError"](
                        f"policy observation contains NaN or Inf at step {completed_steps}"
                    )
                if not runtime["torch"].isfinite(rewards).all():
                    raise runtime["RolloutValidationError"](
                        f"reward contains NaN or Inf at step {completed_steps}"
                    )
                if bool(runtime["torch"].any(dones)):
                    raise runtime["RolloutValidationError"](
                        f"environment terminated or timed out at step {completed_steps}"
                    )
            completed_steps += 1
        print(f"PLAYBACK_STEPS={completed_steps}", flush=True)
        print(f"PHYSX_BACKEND_AFTER={runtime['assert_physx_environment'](env)}", flush=True)
    except BaseException as exc:
        captured_error = exc
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as close_error:
                if captured_error is None:
                    captured_error = close_error
                else:
                    print(
                        "Additional RAMBO environment-close failure:",
                        file=sys.stderr,
                        flush=True,
                    )
                    traceback.print_exception(
                        type(close_error),
                        close_error,
                        close_error.__traceback__,
                        file=sys.stderr,
                    )
                    sys.stderr.flush()

    if captured_error is not None:
        _print_failure(captured_error)
        exit_code = 1
    else:
        exit_code = 0

    # Normal cleanup is required for every production replay path.
    simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
