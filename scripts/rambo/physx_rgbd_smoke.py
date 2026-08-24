#!/usr/bin/env python3
"""Exercise RAMBO's front RGBD camera through a one-environment PhysX task.

This is a short M7 smoke gate, not the 30-second checkpoint acceptance run. It
validates the migrated camera's public Isaac Lab 3 interfaces, RGBD output,
exact 40-physics-tick cadence, and the active PhysX manager in one real RAMBO
task.  The only accepted visualizer choices are ``--viz none`` and ``--viz
kit``; camera rendering is enabled explicitly regardless of the visualizer.
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
    "quadruped": "Isaac-RAMBO-Quadruped-Go2-v0",
    "biped": "Isaac-RAMBO-Biped-Go2-v0",
}


def _has_option(name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


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


def _require_proxy_torch(value: Any, label: str) -> Any:
    """Require the target runtime's explicit ProxyArray-to-Torch bridge."""

    tensor = getattr(value, "torch", None)
    if tensor is None:
        raise RuntimeError(f"{label} is not an Isaac Lab ProxyArray with public .torch access")
    return tensor


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=tuple(TASKS), default="quadruped")
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument(
    "--eager-sensor-update",
    action="store_true",
    help="Exercise the non-lazy scene path, which requests Camera force recomputation every physics tick.",
)
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

# ``--viz none`` only removes the Kit visualizer.  RTX sensor rendering itself
# needs this independently enabled AppLauncher flag.
args_cli.enable_cameras = True
args_cli.fast_shutdown = True
args_cli.livestream = 0

output_dir = args_cli.output_dir.expanduser().resolve()
if output_dir.exists():
    raise RuntimeError(f"Refusing to overwrite existing output directory: {output_dir}")
output_dir.mkdir(parents=True)

# AppLauncher must be constructed before importing Kit-dependent RAMBO code.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _write_summary(summary: dict[str, Any]) -> Path:
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def _write_checksums() -> None:
    lines: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output_dir)}")
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _camera_state(camera: Any) -> dict[str, Any]:
    """Read public pose/intrinsics ProxyArrays for the RGBD evidence record."""

    data = camera.data
    pos_w = _require_proxy_torch(data.pos_w, "camera.data.pos_w")
    quat_w_world = _require_proxy_torch(data.quat_w_world, "camera.data.quat_w_world")
    intrinsics = _require_proxy_torch(data.intrinsic_matrices, "camera.data.intrinsic_matrices")
    return {
        "position_w": pos_w.detach().cpu().tolist(),
        "orientation_w_xyzw": quat_w_world.detach().cpu().tolist(),
        "intrinsic_matrices": intrinsics.detach().cpu().tolist(),
    }


def _collect_rgbd(
    camera: Any,
    *,
    frame_id: int,
    action_step: int,
    physics_ticks: int,
    physics_dt_s: float,
) -> tuple[dict[str, Any], Any, Any]:
    """Read and validate one fresh public RGBD frame without implicit bridges."""

    import torch

    data = camera.data
    output = data.output
    if not isinstance(output, dict):
        raise RuntimeError("camera.data.output is not a dictionary")
    if "rgb" not in output or "distance_to_image_plane" not in output:
        raise RuntimeError(f"camera RGBD output keys are incomplete: {sorted(output)}")
    rgb = _require_proxy_torch(output["rgb"], "camera.data.output['rgb']")
    depth = _require_proxy_torch(
        output["distance_to_image_plane"], "camera.data.output['distance_to_image_plane']"
    )
    if tuple(rgb.shape) != (1, 480, 640, 3):
        raise RuntimeError(f"Unexpected RGB shape: {tuple(rgb.shape)}")
    if tuple(depth.shape) != (1, 480, 640, 1):
        raise RuntimeError(f"Unexpected distance-to-image-plane shape: {tuple(depth.shape)}")
    if not bool(torch.isfinite(rgb).all()):
        raise RuntimeError("RGB contains NaN or Inf")

    rgb_float = rgb.float()
    rgb_mean = float(rgb_float.mean().item())
    rgb_std = float(rgb_float.std().item())
    if rgb_mean <= 2.0 or rgb_std <= 1.0:
        raise RuntimeError(f"RGB is black or lacks variance: mean={rgb_mean}, std={rgb_std}")

    finite_depth = torch.isfinite(depth)
    positive_depth = finite_depth & (depth > 0.0)
    finite_fraction = float(finite_depth.float().mean().item())
    positive_fraction = float(positive_depth.float().mean().item())
    if positive_fraction <= 0.01:
        raise RuntimeError(
            "Distance image has no meaningful positive geometry: "
            f"finite_fraction={finite_fraction}, positive_fraction={positive_fraction}"
        )
    finite_values = depth[finite_depth]
    return (
        {
            "frame_id": frame_id,
            "action_step": action_step,
            "physics_ticks": physics_ticks,
            "timestamp_s": physics_ticks * physics_dt_s,
            "rgb": {
                "shape": list(rgb.shape),
                "dtype": str(rgb.dtype),
                "channel_order": "RGB",
                "mean": rgb_mean,
                "std": rgb_std,
                "minimum": float(rgb_float.min().item()),
                "maximum": float(rgb_float.max().item()),
            },
            "distance_to_image_plane": {
                "shape": list(depth.shape),
                "dtype": str(depth.dtype),
                "finite_fraction": finite_fraction,
                "positive_fraction": positive_fraction,
                "finite_minimum": float(finite_values.min().item()),
                "finite_maximum": float(finite_values.max().item()),
            },
        },
        rgb,
        depth,
    )


def _capture_third_person_diagnostic(
    camera: Any,
    *,
    sim: Any,
    robot_root_position_w: Any,
    device: str,
) -> tuple[dict[str, Any], Any, Any]:
    """Render an independent camera aimed at the real RAMBO robot root.

    The front camera intentionally looks forward from the Go2 base and need
    not contain the chassis.  This diagnostic sensor has no effect on that
    production camera's mount or cadence; it exists solely to leave auditable
    visual proof that the task scene and robot are both rendered.
    """

    import torch

    target = robot_root_position_w.detach().reshape(1, 3)
    eye_offset = torch.tensor([[2.2, 2.2, 1.5]], dtype=target.dtype, device=device)
    eye = target + eye_offset
    camera.set_world_poses_from_view(eyes=eye, targets=target)
    sim.render()
    camera.update(dt=float(sim.get_physics_dt()), force_recompute=True)
    frame = _require_proxy_torch(camera.frame, "diagnostic_camera.frame")
    frame_id = int(frame.detach().reshape(-1)[0].item())
    record, rgb, depth = _collect_rgbd(
        camera,
        frame_id=frame_id,
        action_step=0,
        physics_ticks=0,
        physics_dt_s=float(sim.get_physics_dt()),
    )
    record["camera_target_robot_root_w"] = target.detach().cpu().tolist()
    record["camera_eye_w"] = eye.detach().cpu().tolist()
    record["public_targeting_verified"] = True
    return record, rgb, depth


def main() -> int:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "passed": False,
        "task_alias": args_cli.task,
        "task": TASKS[args_cli.task],
        "requested_action_steps": args_cli.steps,
        "seed": args_cli.seed,
        "viz": visualizer_selection,
        "enable_cameras": True,
        "eula_acceptance": "explicit_user_consent",
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
    }
    env = None
    diagnostic_camera = None
    exit_code = 0
    try:
        import imageio.v3 as iio
        import numpy as np
        import torch

        from rambo.torch_runtime import ensure_cuda_linalg_loaded

        ensure_cuda_linalg_loaded()
        import gymnasium as gym
        import rambo
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg
        from rambo.tasks.common.camera import camera_update_interval_steps
        from rambo.utils import assert_physx_environment, parse_env_cfg

        if not rambo.register_tasks():
            raise RuntimeError("RAMBO task registration requires an active Isaac Sim Kit application")

        env_cfg = parse_env_cfg(TASKS[args_cli.task], device=args_cli.device or "cuda:0", num_envs=1)
        # Assign rather than trust a package default: every RAMBO RGBD run must
        # explicitly instantiate PhysX and keep Newton actuators disabled.
        env_cfg.sim.physics = PhysxCfg()
        env_cfg.sim.use_newton_actuators = False
        env_cfg.seed = args_cli.seed
        env_cfg.events = None
        for attribute in (
            "randomize_initial_state",
            "obs_noise",
            "randomize_episode_progress",
            "enable_sampled_velocity_commands",
            "enable_sampled_pos_commands",
            "enable_sampled_force_commands",
            "velocity_debug_vis",
            "pos_debug_vis",
            "force_debug_vis",
        ):
            if hasattr(env_cfg, attribute):
                setattr(env_cfg, attribute, False)
        # Biped's controller/QP markers subscribe to Kit post-update events
        # independently of the top-level RAMBO command visualizers.  A short
        # camera smoke must not leave those debug callbacks alive while Kit
        # tears down its USD prims; disable only the diagnostic streams, not
        # any policy or physics behavior.
        for config_attribute, flag in (
            ("contact_generator_config", "contact_generator_debug_vis"),
            ("joint_position_controller_config", "joint_position_controller_debug_vis"),
            ("qp_torque_optimizer_config", "qp_debug_vis"),
        ):
            nested_config = getattr(env_cfg, config_attribute, None)
            if nested_config is not None:
                if not isinstance(nested_config, dict):
                    raise RuntimeError(
                        f"RAMBO {config_attribute} must be a dict when present; "
                        f"got {type(nested_config).__name__}"
                    )
                nested_config[flag] = False
        if not hasattr(env_cfg, "enable_rgb_camera") or not hasattr(env_cfg, "front_camera"):
            raise RuntimeError("RAMBO task config does not expose its front-camera contract")
        env_cfg.enable_rgb_camera = True
        env_cfg.scene.lazy_sensor_update = not args_cli.eager_sensor_update
        env_cfg.front_camera.data_types = ["rgb", "distance_to_image_plane"]
        env_cfg.front_camera.renderer_cfg = IsaacRtxRendererCfg()

        env = gym.make(TASKS[args_cli.task], cfg=env_cfg)
        summary["backend_before"] = assert_physx_environment(env)
        base_env = env.unwrapped
        camera = getattr(base_env, "front_camera", None)
        if camera is None:
            raise RuntimeError("RAMBO task did not expose public front_camera after RGBD initialization")
        if base_env.scene.sensors.get("front_camera") is not camera:
            raise RuntimeError("RAMBO front camera was not registered in the public scene sensor map")

        # Create a separate scene-level diagnostic camera.  It does not alter
        # the front-camera configuration or its base-relative physical mount.
        diagnostic_camera = Camera(
            CameraCfg(
                prim_path="/World/M7Diagnostic/ThirdPersonCamera",
                update_period=0.0,
                height=480,
                width=640,
                data_types=["rgb", "distance_to_image_plane"],
                renderer_cfg=IsaacRtxRendererCfg(),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 30.0),
                ),
            )
        )
        # The direct sensor was created after Gym constructed the scene, so it
        # receives its official PHYSICS_READY initialization on this reset.
        base_env.sim.reset()

        observations, _ = env.reset(seed=args_cli.seed)
        _finite_tree(observations, "reset_observations")
        # Data must be read before frame: Isaac Lab's lazy Camera increments
        # frame during data refresh.  This is the first post-reset render,
        # deliberately retained as a baseline rather than a cadence sample.
        initial_data = camera.data
        initial_rgb = _require_proxy_torch(initial_data.output["rgb"], "initial camera RGB")
        initial_depth = _require_proxy_torch(
            initial_data.output["distance_to_image_plane"], "initial camera depth"
        )
        initial_frame = _require_proxy_torch(camera.frame, "camera.frame")
        baseline_frame_id = int(initial_frame.detach().reshape(-1)[0].item())
        if tuple(initial_rgb.shape) != (1, 480, 640, 3) or tuple(initial_depth.shape) != (1, 480, 640, 1):
            raise RuntimeError(
                "Unexpected first-reset RGBD shapes: "
                f"rgb={tuple(initial_rgb.shape)}, depth={tuple(initial_depth.shape)}"
            )

        robot_root_position_w = _require_proxy_torch(
            base_env._robot.data.root_link_pos_w, "robot.data.root_link_pos_w"
        )[0:1]
        third_person_record, third_person_rgb, third_person_depth = _capture_third_person_diagnostic(
            diagnostic_camera,
            sim=base_env.sim,
            robot_root_position_w=robot_root_position_w,
            device=base_env.device,
        )
        iio.imwrite(output_dir / "third_person_robot_scene_rgb.png", third_person_rgb[0].detach().cpu().numpy())
        np.save(
            output_dir / "third_person_robot_scene_distance_to_image_plane.npy",
            third_person_depth[0].detach().cpu().numpy(),
        )

        physics_dt = float(base_env.cfg.sim.dt)
        cadence_ticks = camera_update_interval_steps(float(camera.cfg.update_period), physics_dt)
        decimation = int(base_env.cfg.decimation)
        physics_ticks = args_cli.steps * decimation
        if physics_ticks % cadence_ticks:
            raise RuntimeError(
                f"steps={args_cli.steps} produces {physics_ticks} ticks, not an integral {cadence_ticks}-tick cadence"
            )
        expected_frames = physics_ticks // cadence_ticks
        captures: list[dict[str, Any]] = []
        saved_rgb = False
        saved_depth = False
        last_frame_id = baseline_frame_id
        for action_step in range(1, args_cli.steps + 1):
            actions = torch.zeros(env.action_space.shape, dtype=torch.float32, device=base_env.device)
            transition = env.step(actions)
            _finite_tree(transition, f"transition_{action_step}")

            # The public data-before-frame ordering is intentional and must not
            # be replaced with implicit ProxyArray tensor operations.
            camera.data
            current_frame = _require_proxy_torch(camera.frame, "camera.frame")
            frame_id = int(current_frame.detach().reshape(-1)[0].item())
            if frame_id < last_frame_id:
                raise RuntimeError(f"Camera frame regressed from {last_frame_id} to {frame_id}")
            if frame_id == last_frame_id:
                continue
            if frame_id != last_frame_id + 1:
                raise RuntimeError(f"Camera frame skipped from {last_frame_id} to {frame_id}")
            record, rgb, depth = _collect_rgbd(
                camera,
                frame_id=frame_id,
                action_step=action_step,
                physics_ticks=action_step * decimation,
                physics_dt_s=physics_dt,
            )
            captures.append(record)
            if not saved_rgb:
                iio.imwrite(output_dir / "first_fresh_rgb.png", rgb[0].detach().cpu().numpy())
                saved_rgb = True
            if not saved_depth:
                np.save(output_dir / "first_fresh_distance_to_image_plane.npy", depth[0].detach().cpu().numpy())
                saved_depth = True
            last_frame_id = frame_id

        camera_ticks = _require_proxy_torch(
            getattr(camera, "rambo_physics_ticks", None), "camera.rambo_physics_ticks"
        )
        camera_tick_values = camera_ticks.detach().cpu().tolist()
        if camera_tick_values != [physics_ticks]:
            raise RuntimeError(f"Exact camera tick counter mismatch: {camera_tick_values}, expected [{physics_ticks}]")
        if len(captures) != expected_frames:
            raise RuntimeError(f"Expected {expected_frames} fresh RGBD frames, captured {len(captures)}")
        expected_ticks = list(range(cadence_ticks, physics_ticks + 1, cadence_ticks))
        actual_ticks = [record["physics_ticks"] for record in captures]
        if actual_ticks != expected_ticks:
            raise RuntimeError(f"Camera cadence mismatch: actual={actual_ticks}, expected={expected_ticks}")
        expected_timestamps = [ticks * physics_dt for ticks in expected_ticks]
        actual_timestamps = [record["timestamp_s"] for record in captures]
        if any(abs(actual - expected) > 1.0e-12 for actual, expected in zip(actual_timestamps, expected_timestamps)):
            raise RuntimeError(
                f"Camera timestamp alignment mismatch: actual={actual_timestamps}, expected={expected_timestamps}"
            )

        import warp as wp

        # Exercise the official target-reset signature with a true Warp bool
        # mask after cadence evidence is captured.  This specifically protects
        # the M7 migration from passing only the legacy env-id reset path.
        reset_mask = wp.array([True], dtype=wp.bool, device=base_env.device)
        camera.reset(env_mask=reset_mask)
        tick_counter_after_mask_reset = _require_proxy_torch(
            camera.rambo_physics_ticks, "camera.rambo_physics_ticks after env_mask reset"
        ).detach().cpu().tolist()
        if tick_counter_after_mask_reset != [0]:
            raise RuntimeError(
                "Warp env_mask did not reset the exact camera tick counter: "
                f"{tick_counter_after_mask_reset}"
            )

        summary["backend_after"] = assert_physx_environment(env)
        summary["runtime"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        summary["camera"] = {
            "renderer": f"{type(camera.cfg.renderer_cfg).__module__}.{type(camera.cfg.renderer_cfg).__qualname__}",
            "data_types": list(camera.cfg.data_types),
            "update_period_s": float(camera.cfg.update_period),
            "physics_dt_s": physics_dt,
            "decimation": decimation,
            "lazy_sensor_update": bool(env_cfg.scene.lazy_sensor_update),
            "cadence_ticks": cadence_ticks,
            "physics_ticks": physics_ticks,
            "expected_fresh_frames": expected_frames,
            "baseline_frame_after_reset": baseline_frame_id,
            "exact_tick_counter": camera_tick_values,
            "warp_env_mask_reset": {
                "mask_dtype": str(reset_mask.dtype),
                "mask_shape": list(reset_mask.shape),
                "tick_counter_after_reset": tick_counter_after_mask_reset,
                "passed": True,
            },
            "pose_and_intrinsics": _camera_state(camera),
            "fresh_frames": captures,
            "rgb_artifact": "first_fresh_rgb.png",
            "depth_artifact": "first_fresh_distance_to_image_plane.npy",
            "front_scene_visible": True,
            "visual_evidence": {
                "passed": True,
                "front_camera_robot_visible": False,
                "front_camera_note": "Expected: the physical front mount faces away from the chassis.",
                "third_person_robot_and_scene_visible": True,
                "third_person_rgb_artifact": "third_person_robot_scene_rgb.png",
                "third_person_depth_artifact": "third_person_robot_scene_distance_to_image_plane.npy",
                "third_person_capture": third_person_record,
            },
        }
        summary["passed"] = True
        print("RAMBO_PHYSX_RGBD_SMOKE_SUCCESS", flush=True)
    except BaseException as error:
        exit_code = 1
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        traceback.print_exception(type(error), error, error.__traceback__)
    finally:
        if diagnostic_camera is not None:
            try:
                del diagnostic_camera
            except BaseException as diagnostic_close_error:
                exit_code = 1
                summary["diagnostic_camera_close_error"] = (
                    f"{type(diagnostic_close_error).__name__}: {diagnostic_close_error}"
                )
        if env is not None:
            try:
                env.close()
            except BaseException as close_error:
                exit_code = 1
                summary["close_error"] = f"{type(close_error).__name__}: {close_error}"
                summary.setdefault("traceback", traceback.format_exc())
        summary["passed"] = exit_code == 0
        summary["shutdown_mode"] = "isaacsim_default_fast_shutdown"
        _write_summary(summary)
        _write_checksums()
        print(f"SUMMARY_PATH={output_dir / 'summary.json'}", flush=True)
    return exit_code


if __name__ == "__main__":
    code = main()
    simulation_app.close(exit_code=code)
    raise SystemExit(code)
