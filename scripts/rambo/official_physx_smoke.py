#!/usr/bin/env python3
"""Finite Isaac Sim 6 / Isaac Lab 3 smoke gates with a mandatory PhysX backend.

This file intentionally does not import RAMBO.  It is the layer-separation gate
used before diagnosing any RAMBO migration failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

from isaaclab.app import AppLauncher

from m2_cartpole_gui_artifact import (
    GUI_OBSERVATION_MODE,
    MAX_OBSERVATION_SECONDS,
    MIN_OBSERVATION_SECONDS,
    RTX_RENDERER_EXTENSION_IDS,
    collect_gui_gpu_sample,
    is_explicit_rtx_renderer_extension,
)


GUI_OBSERVATION_FRAME_PACING_S = 1.0 / 30.0
GUI_OBSERVATION_MAX_EXECUTED_STEPS = 10_000


def _has_option(name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenario", choices=("cartpole", "cartpole-direct", "go2", "camera"), required=True)
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument(
    "--gui-observation-seconds",
    type=float,
    default=None,
    help=(
        "Required only for the bounded M2.3 Direct Cartpole --viz kit gate. "
        "Keeps the visible task live for a finite manual-observation interval."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.steps <= 0:
    parser.error("--steps must be positive")
if args_cli.num_envs <= 0:
    parser.error("--num-envs must be positive")
if args_cli.scenario in {"go2", "camera"} and args_cli.num_envs != 1:
    parser.error(f"--scenario {args_cli.scenario} requires exactly one environment")
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

if visualizer_selection == ["kit"]:
    if args_cli.scenario != "cartpole-direct" or args_cli.num_envs != 1:
        parser.error("--viz kit is reserved for the one-environment Direct Cartpole M2.3 GUI gate")
    if args_cli.gui_observation_seconds is None:
        parser.error("--viz kit requires --gui-observation-seconds for a bounded manual M2.3 observation")
    if not MIN_OBSERVATION_SECONDS <= args_cli.gui_observation_seconds <= MAX_OBSERVATION_SECONDS:
        parser.error(
            "--gui-observation-seconds must be within "
            f"[{MIN_OBSERVATION_SECONDS:g}, {MAX_OBSERVATION_SECONDS:g}]"
        )
elif args_cli.gui_observation_seconds is not None:
    parser.error("--gui-observation-seconds requires the dedicated --scenario cartpole-direct --viz kit gate")

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


def _active_gpu_index() -> int:
    """Resolve the Kit-selected CUDA device instead of trusting host defaults."""

    device = str(app_launcher.device)
    if not device.startswith("cuda:"):
        raise RuntimeError(f"M2.3 GUI requires a CUDA renderer device, got {device!r}")
    try:
        index = int(device.split(":", 1)[1])
    except ValueError as error:
        raise RuntimeError(f"Cannot parse Kit CUDA device: {device!r}") from error
    if index < 0:
        raise RuntimeError(f"Kit selected an invalid GPU index: {index}")
    return index


def _renderer_evidence() -> dict[str, Any]:
    """Read live Kit renderer state without changing any renderer setting."""

    import carb
    import omni.kit.app

    settings = carb.settings.get_settings()
    active_gpu = settings.get("/renderer/activeGpu")
    active_renderer = settings.get("/renderer/active")
    if isinstance(active_gpu, bool):
        raise RuntimeError("Kit renderer active GPU setting is not an integer")
    try:
        active_gpu_index = int(active_gpu)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Kit renderer did not expose /renderer/activeGpu: {active_gpu!r}") from error
    if not isinstance(active_renderer, str) or not active_renderer.strip():
        raise RuntimeError(f"Kit renderer did not expose /renderer/active: {active_renderer!r}")

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    # Only a direct RTX render-delegate extension is accepted as renderer
    # evidence.  Generic renderer-core plumbing can be enabled even when RTX
    # is not the active renderer and must never satisfy this gate.
    candidates = RTX_RENDERER_EXTENSION_IDS
    enabled_rtx_extensions = [
        extension for extension in candidates if extension_manager.is_extension_enabled(extension)
    ]
    if not any(is_explicit_rtx_renderer_extension(extension) for extension in enabled_rtx_extensions):
        raise RuntimeError("No explicitly RTX-labelled renderer extension is enabled during the Kit GUI gate")
    return {
        "active_gpu_setting": active_gpu_index,
        "active_renderer_setting": active_renderer,
        "enabled_rtx_extensions": enabled_rtx_extensions,
    }


def _collect_gui_gpu_sample() -> dict[str, Any]:
    expected_gpu = _active_gpu_index()
    renderer = _renderer_evidence()
    if renderer["active_gpu_setting"] != expected_gpu:
        raise RuntimeError(
            "Kit renderer active GPU differs from the AppLauncher CUDA device: "
            f"{renderer['active_gpu_setting']} != {expected_gpu}"
        )
    return collect_gui_gpu_sample(active_gpu_index=expected_gpu, renderer=renderer)


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
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    (output_dir / "checksums.sha256").write_text("".join(checksum_lines), encoding="utf-8")


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

    env_cfg = parse_env_cfg(task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
    if not hasattr(env_cfg, "seed"):
        raise RuntimeError(f"Official task {task} does not expose the required seed configuration")
    env_cfg.seed = 42
    env_cfg.sim.physics = PhysxCfg()
    env_cfg.sim.use_newton_actuators = False
    env = gym.make(task, cfg=env_cfg)
    try:
        if int(env.unwrapped.num_envs) != args_cli.num_envs:
            raise RuntimeError(
                f"Task constructed {env.unwrapped.num_envs} environments, expected {args_cli.num_envs}"
            )
        backend_before = _assert_physx(env.unwrapped.sim)
        observations, _ = env.reset(seed=42)
        _finite_tree(observations, "reset_observations")
        gui_observation_seconds = args_cli.gui_observation_seconds if visualizer_selection == ["kit"] else None
        gui_started = time.monotonic() if gui_observation_seconds is not None else None
        gui_deadline = (
            gui_started + gui_observation_seconds
            if gui_started is not None and gui_observation_seconds is not None
            else None
        )
        gpu_samples: list[dict[str, Any]] = []
        if gui_observation_seconds is not None:
            # The first sample is collected while the task/viewport are live,
            # never from a post-close host check.
            gpu_samples.append(_collect_gui_gpu_sample())
        executed_steps = 0
        while executed_steps < args_cli.steps or (
            gui_deadline is not None and time.monotonic() < gui_deadline
        ):
            if gui_observation_seconds is not None and executed_steps >= GUI_OBSERVATION_MAX_EXECUTED_STEPS:
                raise RuntimeError("M2.3 GUI observation exceeded its finite execution-step limit")
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device, dtype=torch.float32)
            transition = env.step(actions)
            _finite_tree(transition, f"transition_{executed_steps}")
            executed_steps += 1
            if gui_deadline is not None:
                remaining = gui_deadline - time.monotonic()
                if remaining > 0.0:
                    time.sleep(min(GUI_OBSERVATION_FRAME_PACING_S, remaining))
        backend_after_steps = _assert_physx(env.unwrapped.sim)
        result: dict[str, Any] = {
            "task": task,
            "steps": args_cli.steps,
            "executed_steps": executed_steps,
            "num_envs": args_cli.num_envs,
            "action_shape": list(env.action_space.shape),
            "backend_before": backend_before,
            "backend_after": backend_after_steps,
        }
        if gui_observation_seconds is not None and gui_started is not None:
            # The second sample makes the GPU-memory claim temporal: both
            # samples occur while Kit, the official task, and the viewport live.
            gpu_samples.append(_collect_gui_gpu_sample())
            result["gui_observation"] = {
                "mode": GUI_OBSERVATION_MODE,
                "requested_wall_time_s": gui_observation_seconds,
                "actual_wall_time_s": time.monotonic() - gui_started,
                "frame_pacing_s": GUI_OBSERVATION_FRAME_PACING_S,
                "maximum_executed_steps": GUI_OBSERVATION_MAX_EXECUTED_STEPS,
                "expected_active_gpu_index": _active_gpu_index(),
                "gpu_samples": gpu_samples,
                "operator_attestation_required_after_close": True,
                "automatic_visual_observation_claimed": False,
            }
        return result
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
        backend_before = _assert_physx(sim)
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
            "backend_before": backend_before,
            "backend_after": _assert_physx(sim),
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
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
    }
    exit_code = 0
    try:
        if args_cli.scenario == "cartpole":
            summary.update(_run_task("Isaac-Cartpole-v0"))
        elif args_cli.scenario == "cartpole-direct":
            summary.update(_run_task("Isaac-Cartpole-Direct-v0"))
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
