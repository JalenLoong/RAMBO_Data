#!/usr/bin/env python3
"""Run an isolated official Unitree Go2 PhysX and RTX RGB smoke gate.

This is an Isaac Sim / Isaac Lab stack gate, not an application-task test.  It
uses only the official Unitree Go2 asset configuration, creates exactly one
articulation on a ground plane, and fails closed unless the configured and live
physics backends are the exact PhysX implementations.

The command requires an explicit ``--viz none`` or ``--viz kit`` choice.  A
successful run writes a fresh M2 artifact directory containing ``summary.json``,
one 640x480 RGB image, and a SHA-256 manifest.  It never imports project task
packages or selects a non-PhysX backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any


ARTIFACT_ROOT = Path("/workspace/runs/audit/isaac60/M2")
DEFAULT_STEPS = 1000
PHYSICS_DT_S = 0.002
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
PHYSX_CFG_FQN = "isaaclab_physx.physics.physx_manager_cfg.PhysxCfg"
PHYSX_MANAGER_FQN = "isaaclab_physx.physics.physx_manager.PhysxManager"
PHYSX_MANAGER_FACTORY_FQN = "isaaclab_physx.physics.physx_manager:PhysxManager"

# These are the public articulation state tensors exercised on every physics
# step.  The packed root/body states cover all link pose/velocity state, and
# the joint fields cover the articulated degrees of freedom and actuator output.
STATE_TENSOR_FIELDS = (
    "root_state_w",
    "root_link_state_w",
    "root_com_state_w",
    "body_state_w",
    "body_link_state_w",
    "body_com_state_w",
    "joint_pos",
    "joint_vel",
    "joint_acc",
    "computed_torque",
    "applied_torque",
)


def _has_option(arguments: list[str], name: str) -> bool:
    """Return whether an option was explicitly present in the raw CLI."""

    return any(argument == name or argument.startswith(f"{name}=") for argument in arguments)


def _build_parser() -> tuple[argparse.ArgumentParser, type[Any]]:
    """Build the launcher parser without creating a Kit application."""

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Bounded physics steps to execute (default: {DEFAULT_STEPS}).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic Torch seed for this gate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=f"New evidence directory below {ARTIFACT_ROOT}.",
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


def _validate_launcher_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace, raw_arguments: list[str]
) -> str:
    """Apply the fixed, explicit visualizer contract before Kit starts."""

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if _has_option(raw_arguments, "--headless"):
        parser.error("--headless is forbidden; explicitly use --viz none")
    if not (_has_option(raw_arguments, "--viz") or _has_option(raw_arguments, "--visualizer")):
        parser.error("--viz must be explicitly set to none or kit")
    if args.visualizer is None:
        visualizer = "none"
    elif args.visualizer == ["kit"]:
        visualizer = "kit"
    else:
        parser.error("only --viz none and --viz kit are permitted")
    if args.experience:
        parser.error("--experience is forbidden for this fixed official gate")
    if args.kit_args:
        parser.error("--kit_args is forbidden for this fixed official gate")
    if args.livestream not in (-1, 0):
        parser.error("--livestream may only be omitted or set to 0")

    # A camera-capable renderer is mandatory even when no interactive viewport
    # is requested.  The fixed values keep every launch path auditable.
    args.enable_cameras = True
    args.fast_shutdown = True
    args.livestream = 0
    return visualizer


def _prepare_output_dir(path: Path) -> Path:
    """Create exactly one fresh M2 evidence directory and never overwrite it."""

    output_dir = path.expanduser().resolve()
    artifact_root = ARTIFACT_ROOT.resolve()
    if output_dir == artifact_root or not output_dir.is_relative_to(artifact_root):
        raise ValueError(f"--output-dir must be below {artifact_root}: {output_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact directory: {output_dir}")
    output_dir.mkdir(parents=True)
    return output_dir


def _jsonable(value: Any) -> Any:
    """Convert summary values to stable JSON-compatible data."""

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


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(output_dir: Path) -> dict[str, str]:
    """Write a deterministic manifest for every evidence file except itself."""

    manifest = output_dir / "checksums.sha256"
    checksums = {
        path.relative_to(output_dir).as_posix(): _sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != manifest
    }
    manifest.write_text(
        "".join(f"{digest}  {relative_path}\n" for relative_path, digest in checksums.items()),
        encoding="utf-8",
    )
    return checksums


def _proxy_tensor(value: Any, label: str, torch: Any) -> Any:
    """Read an Isaac Lab ProxyArray explicitly and require a finite Torch tensor."""

    if not hasattr(value, "torch"):
        raise RuntimeError(f"{label} is not a public ProxyArray with explicit .torch access")
    tensor = value.torch
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"{label}.torch is not a torch.Tensor: {type(tensor).__name__}")
    if tensor.ndim == 0 or tensor.shape[0] != 1:
        raise RuntimeError(f"{label} must have exactly one environment in its leading dimension: {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{label} contains NaN or Inf")
    return tensor


def _state_snapshot(robot: Any, torch: Any) -> dict[str, list[int]]:
    """Validate every contracted public state tensor and return its shape."""

    shapes: dict[str, list[int]] = {}
    for field in STATE_TENSOR_FIELDS:
        tensor = _proxy_tensor(getattr(robot.data, field), f"robot.data.{field}", torch)
        shapes[field] = list(tensor.shape)
    return shapes


def _assert_physx(sim: Any, *, visualizer: str) -> dict[str, Any]:
    """Fail closed unless both configured and live managers are the exact PhysX backend."""

    physics_cfg = sim.cfg.physics
    physics_manager = sim.physics_manager
    cfg_fqn = f"{type(physics_cfg).__module__}.{type(physics_cfg).__qualname__}"
    manager_fqn = f"{physics_manager.__module__}.{physics_manager.__qualname__}"
    configured_manager = physics_cfg.class_type
    configured_manager_fqn = (
        f"{configured_manager.__module__}.{configured_manager.__qualname__}"
        if isinstance(configured_manager, type)
        else str(configured_manager)
    )
    if cfg_fqn != PHYSX_CFG_FQN:
        raise RuntimeError(f"Physics config is not PhysxCfg: {cfg_fqn}")
    if manager_fqn != PHYSX_MANAGER_FQN:
        raise RuntimeError(f"Live physics manager is not PhysxManager: {manager_fqn}")
    if configured_manager_fqn not in {PHYSX_MANAGER_FQN, PHYSX_MANAGER_FACTORY_FQN}:
        raise RuntimeError(f"PhysX config does not resolve to PhysxManager: {configured_manager_fqn}")
    if getattr(sim.cfg, "use_newton_actuators", None) is not False:
        raise RuntimeError("Newton actuators are enabled or not explicitly disabled")
    return {
        "requested_cfg": cfg_fqn,
        "actual_manager": manager_fqn,
        "configured_manager": configured_manager_fqn,
        "use_newton_actuators": False,
        "physics_dt_s": float(sim.get_physics_dt()),
        "device": str(sim.device),
        "viz": visualizer,
    }


def _runtime_imports() -> dict[str, Any]:
    """Load only official simulator APIs after AppLauncher initialized Kit."""

    import imageio.v3 as imageio
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    return {
        "Articulation": Articulation,
        "Camera": Camera,
        "CameraCfg": CameraCfg,
        "IsaacRtxRendererCfg": IsaacRtxRendererCfg,
        "PhysxCfg": PhysxCfg,
        "UNITREE_GO2_CFG": UNITREE_GO2_CFG,
        "imageio": imageio,
        "sim_utils": sim_utils,
        "torch": torch,
    }


def _make_camera(runtime: dict[str, Any]) -> Any:
    """Create the single official RTX RGB camera used by this gate."""

    sim_utils = runtime["sim_utils"]
    sim_utils.create_prim("/World/Go2CameraRig", "Xform")
    return runtime["Camera"](
        runtime["CameraCfg"](
            prim_path="/World/Go2CameraRig/Camera",
            update_period=0.0,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            data_types=["rgb"],
            renderer_cfg=runtime["IsaacRtxRendererCfg"](),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
        )
    )


def _reset_robot(robot: Any, torch: Any) -> None:
    """Restore the official asset's deterministic default state for its one instance."""

    root_pose = _proxy_tensor(robot.data.default_root_pose, "robot.data.default_root_pose", torch).clone()
    root_velocity = _proxy_tensor(robot.data.default_root_vel, "robot.data.default_root_vel", torch).clone()
    joint_position = _proxy_tensor(robot.data.default_joint_pos, "robot.data.default_joint_pos", torch).clone()
    joint_velocity = _proxy_tensor(robot.data.default_joint_vel, "robot.data.default_joint_vel", torch).clone()
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=root_velocity)
    robot.write_joint_position_to_sim_index(position=joint_position)
    robot.write_joint_velocity_to_sim_index(velocity=joint_velocity)
    robot.set_joint_position_target_index(target=joint_position)
    robot.write_data_to_sim()
    robot.reset()


def _run_gate(args: argparse.Namespace, *, visualizer: str, output_dir: Path) -> dict[str, Any]:
    """Execute the bounded one-Go2 physics and camera gate."""

    runtime = _runtime_imports()
    torch = runtime["torch"]
    sim_utils = runtime["sim_utils"]
    sim = None
    camera = None
    try:
        torch.manual_seed(args.seed)
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=PHYSICS_DT_S,
                device=args.device or "cuda:0",
                physics=runtime["PhysxCfg"](),
                use_newton_actuators=False,
            )
        )
        backend_before = _assert_physx(sim, visualizer=visualizer)
        if abs(float(sim.get_physics_dt()) - PHYSICS_DT_S) > 1.0e-12:
            raise RuntimeError(
                f"Physics dt is {float(sim.get_physics_dt())}, expected the fixed {PHYSICS_DT_S} seconds"
            )

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        robot = runtime["Articulation"](runtime["UNITREE_GO2_CFG"].replace(prim_path="/World/Go2"))
        camera = _make_camera(runtime)

        sim.reset()
        _reset_robot(robot, torch)
        camera.set_world_poses_from_view(
            eyes=torch.tensor([[2.4, 2.4, 1.5]], dtype=torch.float32, device=sim.device),
            targets=torch.tensor([[0.0, 0.0, 0.35]], dtype=torch.float32, device=sim.device),
        )

        physics_dt = float(sim.get_physics_dt())
        first_state_shapes = _state_snapshot(robot, torch)
        rgb = None
        for step in range(1, args.steps + 1):
            joint_target = _proxy_tensor(robot.data.default_joint_pos, "robot.data.default_joint_pos", torch)
            robot.set_joint_position_target_index(target=joint_target)
            robot.write_data_to_sim()
            sim.step()
            robot.update(physics_dt)
            camera.update(dt=physics_dt)
            _state_snapshot(robot, torch)
            candidate = _proxy_tensor(camera.data.output["rgb"], "camera.data.output.rgb", torch)
            if tuple(candidate.shape) not in {
                (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                (1, IMAGE_HEIGHT, IMAGE_WIDTH, 4),
            }:
                raise RuntimeError(f"Unexpected RGB shape: {tuple(candidate.shape)}")
            rgb = candidate

        if rgb is None:
            raise RuntimeError("No RGB frame was produced")
        frame = rgb[0, ..., :3].detach().cpu()
        if not torch.isfinite(frame).all():
            raise RuntimeError("Saved RGB frame contains NaN or Inf")
        rgb_mean = float(frame.float().mean().item())
        rgb_std = float(frame.float().std().item())
        if rgb_mean <= 2.0 or rgb_std <= 1.0:
            raise RuntimeError(f"RGB frame is black or lacks variation: mean={rgb_mean}, std={rgb_std}")
        runtime["imageio"].imwrite(output_dir / "go2_rgb.png", frame.numpy())

        backend_after = _assert_physx(sim, visualizer=visualizer)
        return {
            "asset": "isaaclab_assets.robots.unitree.UNITREE_GO2_CFG",
            "num_envs": 1,
            "physics_steps": args.steps,
            "physics_dt_s": physics_dt,
            "physics_ticks": args.steps,
            "backend_before": backend_before,
            "backend_after": backend_after,
            "state_tensor_fields": list(STATE_TENSOR_FIELDS),
            "state_shapes": first_state_shapes,
            "state_finite_checks": args.steps * len(STATE_TENSOR_FIELDS),
            "camera": {
                "renderer": "isaaclab_physx.renderers.IsaacRtxRendererCfg",
                "resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
                "data_types": ["rgb"],
                "rgb_shape": list(rgb.shape),
                "rgb_mean": rgb_mean,
                "rgb_std": rgb_std,
                "frame_path": "go2_rgb.png",
            },
        }
    finally:
        if camera is not None:
            del camera
        if sim is not None:
            try:
                sim.stop()
            finally:
                sim.clear_instance()


def main() -> int:
    parser, app_launcher_type = _build_parser()
    args = parser.parse_args()
    visualizer = _validate_launcher_args(parser, args, sys.argv[1:])
    output_dir = _prepare_output_dir(args.output_dir)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "passed": False,
        "gate": "official_go2_physx_rgb",
        "seed": args.seed,
        "requested_physics_steps": args.steps,
        "viz": visualizer,
        "eula_acceptance": "explicit_user_consent",
        "artifact": {
            "summary_path": "summary.json",
            "checksum_manifest": "checksums.sha256",
        },
    }
    app_launcher = None
    exit_code = 0
    try:
        app_launcher = app_launcher_type(args)
        summary.update(_run_gate(args, visualizer=visualizer, output_dir=output_dir))
    except BaseException as error:
        exit_code = 1
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        traceback.print_exception(type(error), error, error.__traceback__)
    finally:
        summary["passed"] = exit_code == 0
        summary["shutdown_mode"] = "isaacsim_default_fast_shutdown"
        summary["external_exit_code_required"] = True
        _write_summary(output_dir, summary)
        _write_checksums(output_dir)
    if exit_code == 0:
        print("OFFICIAL_GO2_PHYSX_SMOKE_SUCCESS", flush=True)
    else:
        print("OFFICIAL_GO2_PHYSX_SMOKE_FAILURE", flush=True)
    if app_launcher is not None:
        # This normal Kit close path preserves the workload exit status.  The
        # evidence above is deliberately complete before the app is closed.
        app_launcher.app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
