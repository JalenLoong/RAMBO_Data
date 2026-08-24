#!/usr/bin/env python3
"""Finite Isaac Sim 6 / Isaac Lab 3 smoke gates with a mandatory PhysX backend.

This file intentionally does not import RAMBO.  It is the layer-separation gate
used before diagnosing any RAMBO migration failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

from isaaclab.app import AppLauncher


def _has_option(name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenario", choices=("cartpole", "go2", "camera"), required=True)
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.steps <= 0:
    parser.error("--steps must be positive")
if _has_option("--headless"):
    parser.error("--headless is forbidden; use --viz none")
if not _has_option("--viz") and not _has_option("--visualizer"):
    parser.error("--viz must be explicitly set to none or kit")
# Isaac Lab normalizes the explicit CLI spelling ``--viz none`` to ``None``.
# Keep a canonical list for evidence while accepting only the two safe modes.
if args_cli.visualizer is None:
    visualizer_selection = ["none"]
elif args_cli.visualizer == ["kit"]:
    visualizer_selection = ["kit"]
else:
    parser.error("only --viz none and --viz kit are permitted; Newton visualizers are forbidden")
if args_cli.experience:
    parser.error("--experience is forbidden for this fixed official PhysX gate")
if args_cli.kit_args:
    parser.error("--kit_args is forbidden for this fixed official PhysX gate")
if args_cli.livestream not in (-1, 0):
    parser.error("--livestream may only be omitted or set to 0 for this gate")
if args_cli.scenario == "camera":
    args_cli.enable_cameras = True

# Use the vendor default shutdown path.  In this pinned Kit build, disabling
# fast shutdown takes the known full-extension teardown path that can segfault
# after successful work; all evidence is therefore written before close(), and
# the parent process records the final exit status separately.
args_cli.fast_shutdown = True
args_cli.livestream = 0

output_dir = args_cli.output_dir.expanduser().resolve()
if output_dir.exists():
    raise RuntimeError(f"Refusing to overwrite existing output directory: {output_dir}")
output_dir.mkdir(parents=True)

# AppLauncher must be constructed before importing Kit-dependent Isaac Lab APIs.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


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


def _write_summary(payload: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _assert_physx(sim: Any) -> dict[str, Any]:
    """Fail closed unless the actual running manager is the exact PhysX manager."""

    cfg = sim.cfg.physics
    manager = sim.physics_manager
    cfg_fqn = f"{type(cfg).__module__}.{type(cfg).__qualname__}"
    expected_cfg_fqn = "isaaclab_physx.physics.physx_manager_cfg.PhysxCfg"
    expected_manager_fqn = "isaaclab_physx.physics.physx_manager.PhysxManager"
    if cfg_fqn != expected_cfg_fqn:
        raise RuntimeError(f"Physics config is not PhysxCfg: {cfg_fqn}")
    # AppLauncher temporarily unloads and restores Isaac Lab modules while Kit
    # starts.  That can yield an equivalent re-imported class object, so compare
    # the exact fully-qualified backend identity rather than Python object
    # identity.  Any non-PhysX manager (including Newton) fails closed here.
    manager_fqn = f"{manager.__module__}.{manager.__qualname__}"
    if manager_fqn != expected_manager_fqn:
        raise RuntimeError(f"Physics manager is not PhysxManager: {manager_fqn}")
    configured_manager = cfg.class_type
    configured_manager_fqn = (
        f"{configured_manager.__module__}.{configured_manager.__qualname__}"
        if isinstance(configured_manager, type)
        else str(configured_manager)
    )
    expected_configured_manager = "isaaclab_physx.physics.physx_manager:PhysxManager"
    if configured_manager_fqn not in {expected_manager_fqn, expected_configured_manager}:
        raise RuntimeError(f"PhysX config does not resolve to PhysxManager: {configured_manager_fqn}")
    use_newton_actuators = getattr(sim.cfg, "use_newton_actuators", None)
    if use_newton_actuators is not False:
        raise RuntimeError("Newton actuators are enabled or not explicitly disabled")
    return {
        "requested_cfg": cfg_fqn,
        "actual_manager": manager_fqn,
        "configured_manager": configured_manager_fqn,
        "use_newton_actuators": use_newton_actuators,
        "physics_dt_s": float(sim.get_physics_dt()),
        "device": str(sim.device),
        "viz": visualizer_selection,
    }


def _finite_tree(value: Any, label: str) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{label} contains NaN or Inf")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")


def _run_task(task: str) -> dict[str, Any]:
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401 - registers official tasks.
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_physx.physics import PhysxCfg

    env_cfg = parse_env_cfg(task, device=args_cli.device or "cuda:0", num_envs=1)
    env_cfg.sim.physics = PhysxCfg()
    env_cfg.sim.use_newton_actuators = False
    env = gym.make(task, cfg=env_cfg)
    try:
        _assert_physx(env.unwrapped.sim)
        observations, _ = env.reset(seed=42)
        _finite_tree(observations, "reset_observations")
        for step in range(args_cli.steps):
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device, dtype=torch.float32)
            transition = env.step(actions)
            _finite_tree(transition, f"transition_{step}")
        backend_after_steps = _assert_physx(env.unwrapped.sim)
        return {
            "task": task,
            "steps": args_cli.steps,
            "action_shape": list(env.action_space.shape),
            "backend": backend_after_steps,
        }
    finally:
        env.close()


def _run_camera() -> dict[str, Any]:
    import imageio.v3 as iio
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            device=args_cli.device or "cuda:0", dt=1.0 / 60.0, physics=PhysxCfg(), use_newton_actuators=False
        )
    )
    camera = None
    try:
        _assert_physx(sim)
        sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
        light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        cube_cfg = sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        )
        cube_cfg.func("/World/VisibleCube", cube_cfg, translation=(0.0, 0.0, 0.2))
        sim_utils.create_prim("/World/Origin_00", "Xform")
        camera = Camera(
            CameraCfg(
                prim_path="/World/Origin_00/CameraSensor",
                update_period=0.0,
                height=480,
                width=640,
                data_types=["rgb"],
                renderer_cfg=IsaacRtxRendererCfg(),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1.0e5),
                ),
            )
        )
        sim.reset()
        camera.set_world_poses_from_view(
            torch.tensor([[2.5, 2.5, 2.5]], device=sim.device),
            torch.tensor([[0.0, 0.0, 0.0]], device=sim.device),
        )
        rgb = None
        for _ in range(max(args_cli.steps, 8)):
            sim.step()
            camera.update(dt=sim.get_physics_dt())
            candidate = camera.data.output["rgb"]
            if not hasattr(candidate, "torch"):
                raise RuntimeError("Camera rgb output is not a ProxyArray with explicit .torch access")
            rgb = candidate.torch
        if rgb is None or tuple(rgb.shape[-3:]) not in {(480, 640, 3), (480, 640, 4)}:
            raise RuntimeError(f"Unexpected RGB shape: {None if rgb is None else tuple(rgb.shape)}")
        frame = rgb[0, ..., :3].detach().cpu()
        if not torch.isfinite(frame).all():
            raise RuntimeError("Camera frame contains NaN or Inf")
        mean = float(frame.float().mean().item())
        std = float(frame.float().std().item())
        if mean <= 2.0 or std <= 1.0:
            raise RuntimeError(f"Camera image is black or lacks variance: mean={mean}, std={std}")
        iio.imwrite(output_dir / "rgb.png", frame.numpy())
        return {
            "steps": max(args_cli.steps, 8),
            "backend": _assert_physx(sim),
            "rgb_shape": list(rgb.shape),
            "rgb_mean": mean,
            "rgb_std": std,
            "rgb_path": "rgb.png",
        }
    finally:
        if camera is not None:
            del camera
        try:
            sim.stop()
        finally:
            sim.clear_instance()


def main() -> int:
    summary: dict[str, Any] = {
        "passed": False,
        "scenario": args_cli.scenario,
        "viz": visualizer_selection,
        "eula_acceptance": "explicit_user_consent",
    }
    exit_code = 0
    try:
        if args_cli.scenario == "cartpole":
            summary.update(_run_task("Isaac-Cartpole-v0"))
        elif args_cli.scenario == "go2":
            summary.update(_run_task("Isaac-Velocity-Flat-Unitree-Go2-v0"))
        else:
            summary.update(_run_camera())
    except BaseException as error:
        exit_code = 1
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        traceback.print_exception(type(error), error, error.__traceback__)
    finally:
        summary["passed"] = exit_code == 0
        summary["shutdown_mode"] = "isaacsim_default_fast_shutdown"
        summary["external_exit_code_required"] = True
        _write_summary(summary)
    if exit_code == 0:
        print("OFFICIAL_PHYSX_SMOKE_SUCCESS", flush=True)
    else:
        print("OFFICIAL_PHYSX_SMOKE_FAILURE", flush=True)
    # With ``fast_shutdown=True``, close() normally does not return.  Its
    # ``exit_code`` parameter preserves workload failure status without this
    # script ever bypassing normal cleanup or directly terminating the process.
    simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
