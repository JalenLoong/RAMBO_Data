#!/usr/bin/env python3
"""Exercise the official Isaac Lab Go2 stack before starting a RAMBO task.

This deliberately has no RAMBO imports.  It is the first runtime gate for the
Isaac Sim 5.1 / Isaac Lab 2.3.2 migration: validate the pinned Python/Torch
CUDA runtime, spawn ``UNITREE_GO2_CFG`` on an official ground plane, and run a
finite number of physics steps.  ``--enable-camera`` additionally attaches a
640x480 RGB camera to the Go2 base and checks one rendered frame.

The script never accepts an NVIDIA EULA itself.  Its first Kit launch will use
the normal NVIDIA prompt unless the operator has already accepted the EULA.

Examples:
    scripts/rambo/run.sh scripts/rambo/smoke.py --headless --steps 100
    scripts/rambo/run.sh scripts/rambo/smoke.py --headless --enable-camera \
        --camera-output /tmp/go2-smoke-rgb.npy
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> tuple[argparse.ArgumentParser, type]:
    """Build the CLI without importing RAMBO or scene modules prematurely."""

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as error:  # pragma: no cover - target-runtime guard.
        raise RuntimeError(
            "scripts/rambo/smoke.py must run in the RAMBO Isaac Lab 2.3.2 environment"
        ) from error

    parser = argparse.ArgumentParser(
        description="Finite official-Go2 Isaac Lab smoke test for the RAMBO migration."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Finite number of 0.002 s physics steps to execute (default: 100).",
    )
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=0.002,
        help="Physics time step in seconds (default: 0.002).",
    )
    parser.add_argument(
        "--enable-camera",
        "--enable_camera",
        dest="enable_camera",
        action="store_true",
        help="Attach and validate one base-mounted 640x480 RGB camera.",
    )
    parser.add_argument(
        "--camera-output",
        type=Path,
        help="Optional new .npy path for the captured RGB frame (requires --enable-camera).",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser, AppLauncher


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject ambiguous or unsafe CLI combinations before Kit starts."""

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.physics_dt <= 0.0:
        parser.error("--physics-dt must be positive")
    if not str(args.device).startswith("cuda"):
        parser.error("--device must identify a CUDA device for this RTX 5080 migration gate")
    if args.camera_output is not None:
        if not args.enable_camera:
            parser.error("--camera-output requires --enable-camera")
        if args.camera_output.suffix != ".npy":
            parser.error("--camera-output must end in .npy")
        if args.camera_output.expanduser().exists():
            parser.error(f"--camera-output already exists: {args.camera_output}")
    if args.enable_camera:
        # AppLauncher selects a camera-capable Kit experience before the app exists.
        args.enable_cameras = True


def _check_target_runtime() -> dict[str, Any]:
    """Validate the Python 3.11 / Torch 2.7 CUDA 12.8 Blackwell gate."""

    import torch

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Expected Python 3.11, got {sys.version}")
    if torch.__version__.split("+", maxsplit=1)[0] != "2.7.0":
        raise RuntimeError(f"Expected PyTorch 2.7.0, got {torch.__version__}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected the CUDA 12.8 PyTorch wheel, got CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")

    device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    architectures = tuple(torch.cuda.get_arch_list())
    if capability != (12, 0):
        raise RuntimeError(
            "This migration gate requires Blackwell compute capability (12, 0); "
            f"got {capability} on {torch.cuda.get_device_name(device_index)!r}"
        )
    if "sm_120" not in architectures:
        raise RuntimeError(f"PyTorch was not compiled with sm_120 support: {architectures}")

    # Submit and synchronize a tiny CUDA operation.  This verifies that a device
    # can execute work, not merely that it is listed by the driver.
    probe = torch.arange(16, device=f"cuda:{device_index}", dtype=torch.float32)
    probe_sum = float((probe.square().sum()).item())
    torch.cuda.synchronize(device_index)

    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device_index),
        "compute_capability": capability,
        "compiled_architectures": architectures,
        "cuda_probe_sum": probe_sum,
    }


def _runtime_imports() -> dict[str, Any]:
    """Load Isaac Lab entities only after AppLauncher has initialized Kit."""

    import isaaclab.sim as sim_utils
    import numpy as np
    import torch
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

    return {
        "AssetBaseCfg": AssetBaseCfg,
        "CameraCfg": CameraCfg,
        "InteractiveScene": InteractiveScene,
        "InteractiveSceneCfg": InteractiveSceneCfg,
        "UNITREE_GO2_CFG": UNITREE_GO2_CFG,
        "configclass": configclass,
        "np": np,
        "sim_utils": sim_utils,
        "torch": torch,
    }


def _make_scene_cfg(runtime: dict[str, Any], *, enable_camera: bool) -> Any:
    """Construct a one-Go2 official scene, optionally with an RGB camera."""

    asset_base_cfg = runtime["AssetBaseCfg"]
    camera_cfg_type = runtime["CameraCfg"]
    interactive_scene_cfg = runtime["InteractiveSceneCfg"]
    configclass = runtime["configclass"]
    sim_utils = runtime["sim_utils"]
    unitree_go2_cfg = runtime["UNITREE_GO2_CFG"]

    @configclass
    class Go2SmokeSceneCfg(interactive_scene_cfg):
        # Global ground, followed by the articulation, sensor, and a light.  This
        # order is the one required by Isaac Lab's InteractiveScene parser.
        ground = asset_base_cfg(
            prim_path="/World/GroundPlane",
            spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0), color=(0.2, 0.2, 0.2)),
        )
        robot = unitree_go2_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        camera = None
        light = asset_base_cfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
        )

    scene_cfg = Go2SmokeSceneCfg(num_envs=1, env_spacing=2.0, lazy_sensor_update=False)
    if enable_camera:
        scene_cfg.camera = camera_cfg_type(
            prim_path="{ENV_REGEX_NS}/Robot/base/smoke_camera",
            update_period=0.0,
            offset=camera_cfg_type.OffsetCfg(
                pos=(0.30, 0.0, 0.08),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 20.0),
            ),
            width=640,
            height=480,
        )
    return scene_cfg


def _reset_go2(scene: Any) -> None:
    """Place the Go2 in its official default state and initialize its targets."""

    robot = scene.articulations["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    joint_pos = robot.data.default_joint_pos
    joint_vel = robot.data.default_joint_vel
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    scene.write_data_to_sim()
    scene.reset()


def _check_finite_robot_state(runtime: dict[str, Any], scene: Any) -> dict[str, float]:
    """Make the short physics run fail loudly on invalid robot state."""

    torch = runtime["torch"]
    robot = scene.articulations["robot"]
    tensors = {
        "root_pos_w": robot.data.root_pos_w,
        "root_quat_w": robot.data.root_quat_w,
        "joint_pos": robot.data.joint_pos,
        "joint_vel": robot.data.joint_vel,
    }
    invalid = [name for name, value in tensors.items() if not bool(torch.isfinite(value).all())]
    if invalid:
        raise RuntimeError(f"Official Go2 produced NaN/Inf in: {', '.join(invalid)}")
    return {
        "root_height_m": float(robot.data.root_pos_w[0, 2].item()),
        "max_abs_joint_velocity_rad_s": float(robot.data.joint_vel.abs().max().item()),
    }


def _capture_rgb(runtime: dict[str, Any], scene: Any, output_path: Path | None) -> dict[str, Any]:
    """Validate the official Camera RGB tensor and optionally persist one frame."""

    np = runtime["np"]
    torch = runtime["torch"]
    camera = scene.sensors["camera"]
    rgb = camera.data.output.get("rgb")
    expected_shape = (1, 480, 640, 3)
    if rgb is None:
        raise RuntimeError("Camera did not produce an RGB frame")
    if tuple(rgb.shape) != expected_shape:
        raise RuntimeError(f"Unexpected RGB shape: expected {expected_shape}, got {tuple(rgb.shape)}")
    if rgb.dtype != torch.uint8:
        raise RuntimeError(f"Unexpected RGB dtype: expected torch.uint8, got {rgb.dtype}")
    if int(rgb.max().item()) == 0:
        raise RuntimeError("Captured RGB frame is completely black")

    frame = rgb[0].contiguous().cpu().numpy()
    frame_sha256 = hashlib.sha256(frame.tobytes()).hexdigest()
    saved_path: str | None = None
    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing camera output: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, frame)
        saved_path = str(output_path)

    return {
        "shape": tuple(int(value) for value in frame.shape),
        "dtype": str(frame.dtype),
        "min": int(frame.min()),
        "max": int(frame.max()),
        "mean": float(frame.mean()),
        "sha256": frame_sha256,
        "output": saved_path,
    }


def _run_smoke(simulation_app: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Spawn, reset, and simulate the official Go2 scene for a finite run."""

    runtime = _runtime_imports()
    sim_utils = runtime["sim_utils"]
    simulation_cfg = sim_utils.SimulationCfg(
        dt=args.physics_dt,
        render_interval=1,
        device=args.device,
    )
    sim = sim_utils.SimulationContext(simulation_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 1.5], target=[0.0, 0.0, 0.3])

    scene = runtime["InteractiveScene"](_make_scene_cfg(runtime, enable_camera=args.enable_camera))
    sim.reset()
    _reset_go2(scene)

    # Isaac Lab's official camera tests warm up a few render steps before
    # sampling a render product.  Retain that requirement while still making
    # the command strictly finite even when a caller requests fewer steps.
    executed_steps = max(args.steps, 5 if args.enable_camera else 1)
    robot = scene.articulations["robot"]
    target_joint_pos = robot.data.default_joint_pos
    sim_dt = sim.get_physics_dt()
    for _ in range(executed_steps):
        if not simulation_app.is_running():
            raise RuntimeError("Isaac Sim stopped before the finite smoke run completed")
        robot.set_joint_position_target(target_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    metrics: dict[str, Any] = {
        "physics_dt_s": sim_dt,
        "requested_steps": args.steps,
        "executed_steps": executed_steps,
        "robot": _check_finite_robot_state(runtime, scene),
    }
    if args.enable_camera:
        metrics["camera"] = _capture_rgb(runtime, scene, args.camera_output)
    return metrics


def main() -> None:
    parser, app_launcher_type = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    runtime = _check_target_runtime()
    print(f"CUDA_RUNTIME={runtime}")

    app_launcher = app_launcher_type(args)
    simulation_app = app_launcher.app
    try:
        smoke = _run_smoke(simulation_app, args)
        # Emit finite-run evidence before immediate Kit shutdown can release
        # stdout-related resources.
        print(f"GO2_SMOKE={smoke}", flush=True)
        print("GO2_SMOKE_SUCCESS", flush=True)
    finally:
        # Isaac Lab cameras create a live Replicator render product.  In Isaac
        # Sim 5.1, ``close(wait_for_replicator=False)`` still synchronously
        # calls ``rep.orchestrator.stop()``, which can stall after a completed
        # finite camera run.  This command has no writer work left to drain:
        # RGB was synchronously read (and optionally saved) above.  Use the
        # documented immediate-exit path only after that work is finalized.
        simulation_app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
