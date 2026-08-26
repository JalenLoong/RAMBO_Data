#!/usr/bin/env python3
"""Open one LingBot object task in Isaac Sim Kit for interactive scene inspection.

This entry point intentionally does not load a policy, advance an episode, or
write dataset artifacts.  It resets a single PhysX environment, pauses the
timeline, and leaves Kit running so an operator can inspect the generated USD
prims and tune asset geometry in the GUI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback


TASK_IDS = {
    "lift_basket": "Isaac-RAMBO-Quadruped-Lift-Basket-Go2-v0",
    "pull_object_into_basket": "Isaac-RAMBO-Quadruped-Pull-Object-Into-Basket-Go2-v0",
    "shoot_ball_into_goal": "Isaac-RAMBO-Quadruped-Shoot-Ball-Into-Goal-Go2-v0",
}


def _prefer_simready_bundled_python() -> Path:
    """Isolate the SimReady browser's bundled AWS dependencies for this GUI tool.

    The RAMBO runtime pins boto3/botocore for unrelated tooling.  Isaac Sim's
    SimReady extension ships a newer private bundle, and mixing the global
    ``botocore.exceptions`` with its bundled ``botocore.credentials`` makes the
    extension fail on the missing ``LoginError`` symbol.  Prepending the
    extension bundle here affects only this preview process and leaves the
    production venv untouched.
    """

    python_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / python_tag / "site-packages"
    candidates = sorted(
        (site_packages / "isaacsim" / "extscache").glob(
            "omni.simready.content.browser-*/pip_prebundle"
        )
    )
    if not candidates:
        raise RuntimeError("Isaac Sim's bundled SimReady Python dependencies were not found")
    bundle = candidates[-1]
    sys.path.insert(0, str(bundle))
    return bundle


def _build_parser() -> tuple[argparse.ArgumentParser, type]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-profile", choices=tuple(TASK_IDS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--primary-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Optional world position override for the manipulated object.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser, AppLauncher


def main() -> int:
    simready_bundle = _prefer_simready_bundled_python()
    parser, app_launcher_type = _build_parser()
    args = parser.parse_args()

    from rambo.utils.physx import validate_rambo_visualizer_args

    selection = validate_rambo_visualizer_args(parser, args, sys.argv[1:])
    if selection != ["kit"]:
        parser.error("interactive object-task preview requires --viz kit")

    app_launcher = app_launcher_type(args)
    simulation_app = app_launcher.app
    env = None
    try:
        from isaaclab_tasks.utils import parse_env_cfg
        import gymnasium as gym
        import omni.timeline
        import rambo

        from rambo.utils.physx import assert_physx_environment, configure_physx

        rambo.register_tasks()
        task_id = TASK_IDS[args.task_profile]
        parse_kwargs = {"num_envs": 1, "use_fabric": False}
        if getattr(args, "device", None) is not None:
            parse_kwargs["device"] = args.device
        env_cfg = parse_env_cfg(task_id, **parse_kwargs)
        configure_physx(env_cfg)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        if args.primary_position is not None:
            env_cfg.primary_position = tuple(args.primary_position)

        env = gym.make(task_id, cfg=env_cfg)
        env.reset(seed=args.seed)
        backend = assert_physx_environment(env)

        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        print(f"PREVIEW_TASK={task_id}", flush=True)
        print(f"SIMREADY_PYTHON_BUNDLE={simready_bundle}", flush=True)
        print("PREVIEW_STAGE_ROOT=/World/envs/env_0/LingBotTask", flush=True)
        print(f"PHYSX_BACKEND={backend}", flush=True)
        print("The PhysX timeline is paused. Close the Kit window or press Ctrl+C to exit.", flush=True)

        while simulation_app.is_running():
            simulation_app.update()
        return 0
    except KeyboardInterrupt:
        return 0
    except BaseException as error:
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
