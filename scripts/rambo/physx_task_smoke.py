#!/usr/bin/env python3
"""Run a finite RAMBO task smoke test with an explicit, audited PhysX backend.

The launcher deliberately accepts only ``--viz none`` and ``--viz kit``.  It
never offers a Newton visualizer or backend option, and it records the requested
and active physics classes before and after the rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any

from isaaclab.app import AppLauncher


TASKS = {
    "quadruped": ("Isaac-RAMBO-Quadruped-Go2-v0", 405),
    "biped": ("Isaac-RAMBO-Biped-Go2-v0", 435),
    "button": ("Isaac-RAMBO-Quadruped-Button-Go2-v0", 405),
}


def _has_option(name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


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


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=tuple(TASKS), required=True)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--disable-events", action="store_true", help="Use a deterministic no-domain-randomization smoke scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.steps <= 0:
    parser.error("--steps must be positive")
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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> int:
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
    env = None
    exit_code = 0
    try:
        from rambo.torch_runtime import ensure_cuda_linalg_loaded

        ensure_cuda_linalg_loaded()
        import gymnasium as gym
        import rambo
        import torch
        from rambo.utils import assert_physx_environment, parse_env_cfg

        task, expected_observation_dim = TASKS[args_cli.task]
        if not rambo.register_tasks():
            raise RuntimeError("RAMBO task registration requires an active Isaac Sim Kit application")
        env_cfg = parse_env_cfg(task, device=args_cli.device or "cuda:0", num_envs=1)
        env_cfg.seed = args_cli.seed
        if args_cli.disable_events:
            env_cfg.events = None
        env = gym.make(task, cfg=env_cfg)
        summary["backend_before"] = assert_physx_environment(env)
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
        for step in range(args_cli.steps):
            action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.unwrapped.device)
            transition = env.step(action)
            _finite_tree(transition, f"transition_{step}")
        summary["backend_after"] = assert_physx_environment(env)
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
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        (output_dir / "checksums.sha256").write_text(f"{digest}  summary.json\n", encoding="utf-8")
        print(f"SUMMARY_PATH={summary_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    code = main()
    simulation_app.close(exit_code=code)
    raise SystemExit(code)
