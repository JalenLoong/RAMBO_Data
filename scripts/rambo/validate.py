#!/usr/bin/env python3
"""Validate a RAMBO checkpoint with a finite RGB-enabled Isaac Lab rollout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
from typing import Any


def _build_parser() -> tuple[argparse.ArgumentParser, type]:
    """Build the CLI parser while keeping source imports simulator-independent."""

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:  # pragma: no cover - target-runtime guard.
        raise RuntimeError(
            "scripts/rambo/validate.py must be run inside the RAMBO Isaac Lab environment"
        ) from exc

    parser = argparse.ArgumentParser(description="Validate a RAMBO CRL2 policy checkpoint.")
    parser.add_argument("--task", choices=(
        "Isaac-RAMBO-Quadruped-Go2-v0",
        "Isaac-RAMBO-Biped-Go2-v0",
    ), required=True, help="Registered RAMBO Gym task.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted local model_*.pt path.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed validation seed.")
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="100 Hz policy steps; 3000 is the 30-second acceptance run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New empty directory for summary.json and RGB artifacts.",
    )
    parser.add_argument(
        "--contact-phase-offset-s",
        type=float,
        default=None,
        help="Diagnostic biped gait-phase override; default uses the task's 19.6 s acceptance phase.",
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
    """Import simulator dependencies after AppLauncher has initialized Isaac Sim."""

    from rambo.torch_runtime import ensure_cuda_linalg_loaded

    ensure_cuda_linalg_loaded()
    import gymnasium as gym
    import rambo
    from crl2.algorithms import PPO
    from rambo.rl import Crl2VecEnvWrapper
    from rambo.utils.registry import load_cfg_from_registry, parse_env_cfg
    from rambo.validation.checkpoints import (
        contract_for_task,
        load_verified_checkpoint,
        restore_runner,
    )
    from rambo.validation.rollout import (
        RgbFrameRecorder,
        RolloutValidationError,
        configure_validation_cfg,
        prepare_output_dir,
        run_policy_rollout,
        runtime_metadata,
        seed_everything,
        validate_environment_contract,
        write_summary,
    )

    rambo.register_tasks()
    return {
        "gym": gym,
        "PPO": PPO,
        "Crl2VecEnvWrapper": Crl2VecEnvWrapper,
        "RgbFrameRecorder": RgbFrameRecorder,
        "RolloutValidationError": RolloutValidationError,
        "configure_validation_cfg": configure_validation_cfg,
        "contract_for_task": contract_for_task,
        "load_cfg_from_registry": load_cfg_from_registry,
        "load_verified_checkpoint": load_verified_checkpoint,
        "parse_env_cfg": parse_env_cfg,
        "prepare_output_dir": prepare_output_dir,
        "restore_runner": restore_runner,
        "run_policy_rollout": run_policy_rollout,
        "runtime_metadata": runtime_metadata,
        "seed_everything": seed_everything,
        "validate_environment_contract": validate_environment_contract,
        "write_summary": write_summary,
    }


def _print_failure(prefix: str, error: BaseException) -> None:
    """Print a failure while preserving the underlying traceback."""

    print(prefix, file=sys.stderr, flush=True)
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    sys.stderr.flush()


def main() -> int:
    parser, app_launcher_type = _build_parser()
    args_cli = parser.parse_args()
    if args_cli.steps <= 0:
        parser.error("--steps must be positive")
    # The RAMBO camera contract is 0.08 s, i.e. one fresh RGB frame per eight
    # 100 Hz policy steps.  This makes 3000 steps exactly 375 frames.
    if args_cli.steps % 8:
        parser.error("--steps must be divisible by 8 for the 12.5 Hz RGB contract")
    args_cli.enable_cameras = True

    app_launcher = app_launcher_type(args_cli)
    simulation_app = app_launcher.app
    runtime: dict[str, Any] | None = None
    env = None
    output_dir = None
    rgb_recorder = None
    rgb_metrics: dict[str, Any] | None = None
    rollout_metrics: dict[str, Any] | None = None
    summary: dict[str, Any] = {
        "schema_version": 1,
        "passed": False,
        "task": args_cli.task,
        "seed": args_cli.seed,
        "requested_steps": args_cli.steps,
    }
    captured_error: BaseException | None = None
    secondary_errors: list[tuple[str, BaseException]] = []

    try:
        runtime = _runtime_imports()
        output_dir = runtime["prepare_output_dir"](args_cli.output_dir)
        summary["runtime"] = runtime["runtime_metadata"]()

        contract = runtime["contract_for_task"](args_cli.task)
        checkpoint = runtime["load_verified_checkpoint"](args_cli.checkpoint, contract)
        summary["checkpoint"] = {
            "path": str(args_cli.checkpoint.expanduser().resolve()),
            "sha256": contract.sha256,
            "iteration": contract.iteration,
            "observation_dim": contract.observation_dim,
            "action_dim": contract.action_dim,
            "normalizer_count": contract.normalizer_count,
        }
        summary["mode"] = contract.mode

        parse_kwargs = {"num_envs": 1, "use_fabric": not args_cli.disable_fabric}
        device = getattr(args_cli, "device", None)
        if device is not None:
            parse_kwargs["device"] = device
        env_cfg = runtime["parse_env_cfg"](args_cli.task, **parse_kwargs)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args_cli.seed
        runtime["configure_validation_cfg"](
            env_cfg,
            duration_s=31.0,
            enable_rgb_camera=True,
            contact_phase_offset_s=args_cli.contact_phase_offset_s,
        )
        summary["validation_config"] = {
            "episode_length_s": float(env_cfg.episode_length_s),
            "randomize_initial_state": bool(getattr(env_cfg, "randomize_initial_state", False)),
            "randomize_episode_progress": bool(getattr(env_cfg, "randomize_episode_progress", False)),
            "contact_phase_offset_s": float(getattr(env_cfg, "contact_phase_offset_s", 0.0)),
            "observation_noise": bool(getattr(env_cfg, "obs_noise", False)),
        }

        agent_cfg = runtime["load_cfg_from_registry"](args_cli.task, "crl2_cfg_entry_point")
        if not isinstance(agent_cfg, dict):
            raise runtime["RolloutValidationError"](
                "RAMBO CRL2 agent config must load as a YAML dictionary"
            )
        agent_cfg["seed"] = args_cli.seed
        agent_cfg["general"]["num_envs"] = 1

        env = runtime["Crl2VecEnvWrapper"](runtime["gym"].make(args_cli.task, cfg=env_cfg))
        runtime["seed_everything"](args_cli.seed, env)
        env.reset()
        runtime["validate_environment_contract"](env, contract)

        runner = runtime["PPO"](
            task=args_cli.task,
            env=env,
            agent_cfg=agent_cfg,
            train=False,
            device=env.device,
        )
        runtime["restore_runner"](runner, checkpoint, load_values=True, verify=True)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # The rollout helper independently verifies the actual camera period.
        rgb_recorder = runtime["RgbFrameRecorder"](
            output_dir=output_dir,
            expected_count=args_cli.steps // 8,
            width=640,
            height=480,
        )
        rollout_metrics = runtime["run_policy_rollout"](
            env,
            policy,
            contract,
            steps=args_cli.steps,
            rgb_recorder=rgb_recorder,
        )
        rgb_metrics = rgb_recorder.finalize()
        if not rgb_metrics["passed"]:
            raise runtime["RolloutValidationError"](
                "RGB validation failed: " + "; ".join(rgb_metrics["failures"])
            )
    except BaseException as exc:
        captured_error = exc
    finally:
        if rgb_recorder is not None and rgb_metrics is None:
            try:
                rgb_metrics = rgb_recorder.finalize()
            except BaseException as rgb_exc:  # Preserve the primary rollout failure when present.
                if captured_error is None:
                    captured_error = rgb_exc
                rgb_metrics = {
                    "passed": False,
                    "failures": [f"RGB finalization exception: {type(rgb_exc).__name__}: {rgb_exc}"],
                }

        if env is not None:
            try:
                env.close()
            except BaseException as close_exc:
                if captured_error is None:
                    captured_error = close_exc
                else:
                    secondary_errors.append(("environment close", close_exc))

        if captured_error is not None:
            summary["error"] = f"{type(captured_error).__name__}: {captured_error}"
        if rollout_metrics is not None:
            summary["rollout"] = rollout_metrics
        if rgb_metrics is not None:
            summary["rgb"] = rgb_metrics
        summary["passed"] = bool(
            captured_error is None
            and rollout_metrics is not None
            and rgb_metrics is not None
            and rgb_metrics.get("passed", False)
        )

        if secondary_errors:
            summary["secondary_errors"] = [
                f"{stage}: {type(error).__name__}: {error}"
                for stage, error in secondary_errors
            ]

        if output_dir is not None and runtime is not None:
            try:
                summary_path = runtime["write_summary"](output_dir, summary)
                print(f"SUMMARY_PATH={summary_path}", flush=True)
            except BaseException as summary_exc:
                if captured_error is None:
                    captured_error = summary_exc
                else:
                    secondary_errors.append(("summary write", summary_exc))

    if captured_error is not None:
        # ``skip_cleanup=True`` terminates the Kit process immediately.  Do
        # not call it on failure, or it can swallow the validation error
        # before the traceback and non-zero exit status are visible.
        _print_failure("RAMBO validation failed:", captured_error)
        for stage, error in secondary_errors:
            _print_failure(f"Additional {stage} failure:", error)
        return 1

    # Keep the machine-readable success marker before immediate Kit shutdown.
    # RGB frames, summary.json, and the environment have all been finalized.
    print("VALIDATION_SUCCESS", flush=True)
    # In Isaac Sim 5.1, the no-wait close still invokes Replicator's
    # synchronous stop path and can hang on a live camera render product.
    # Use its documented immediate-exit path only on the fully successful
    # path, after all evidence has been flushed.
    simulation_app.close(skip_cleanup=True)
    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        # A live Kit application keeps native worker threads alive after a
        # normal Python ``SystemExit``.  summary.json and the traceback were
        # flushed before returning this status, so terminate the one-shot CLI
        # without entering the known-hanging graceful shutdown path.
        import os

        os._exit(_exit_code)
    raise SystemExit(_exit_code)
