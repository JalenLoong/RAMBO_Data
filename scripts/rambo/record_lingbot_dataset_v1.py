#!/usr/bin/env python3
"""Record and atomically publish the first LingBot-VA -> RAMBO Dataset V1 sample."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import record_button_physx_episode as legacy


class LingBotDatasetRecorderError(RuntimeError):
    """Raised when the golden sample cannot satisfy its strict contract."""


def _build_parser() -> tuple[argparse.ArgumentParser, type[Any]]:
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as error:  # pragma: no cover - target runtime only.
        raise RuntimeError("Run this recorder through scripts/rambo/run.sh") from error
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/datasets/lingbot_rambo_v1/press_button/episode_000001"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-id", type=str, default=None, help="episode_000001 through episode_000010")
    parser.add_argument(
        "--button-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.99, 0.142, 0.30),
        help="Button cap rest translation; panel follows by the same XYZ delta",
    )
    parser.add_argument("--randomization-config", type=Path, default=None)
    parser.add_argument("--randomization-seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--disable-fabric",
        "--disable_fabric",
        dest="disable_fabric",
        action="store_true",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser, AppLauncher


def _prepare_staging(target: Path) -> tuple[Path, Path]:
    target = target.expanduser().resolve()
    staging = target.with_name(f"{target.name}.inprogress")
    if target.exists():
        raise LingBotDatasetRecorderError(f"refusing to overwrite published episode: {target}")
    if staging.exists():
        raise LingBotDatasetRecorderError(f"refusing to overwrite prior staging directory: {staging}")
    staging.mkdir(parents=True)
    for relative in ("actions", "raw_debug", "video/camera", "video/ego", "video/left", "video/right"):
        (staging / relative).mkdir(parents=True)
    return target, staging


def _np_single(tensor: Any, torch: Any, label: str) -> Any:
    value = legacy._proxy_tensor(tensor, label, torch)
    if value.shape[0] != 1:
        raise LingBotDatasetRecorderError(f"{label} must have one environment, got {tuple(value.shape)}")
    return value.detach().cpu().numpy()[0]


def _camera_frame_id(camera: Any, torch: Any, label: str) -> int:
    frame = legacy._proxy_tensor(camera.frame, f"{label}.frame", torch)
    if frame.numel() != 1:
        raise LingBotDatasetRecorderError(f"{label}.frame must contain one value")
    return int(frame.detach().reshape(-1)[0].cpu().item())


def _camera_state(data: Any, torch: Any, label: str) -> tuple[Any, Any]:
    pos = _np_single(data.pos_w, torch, f"{label}.data.pos_w")
    quat = _np_single(data.quat_w_world, torch, f"{label}.data.quat_w_world")
    intrinsic = _np_single(data.intrinsic_matrices, torch, f"{label}.data.intrinsic_matrices")
    pose = torch.cat(
        (
            torch.as_tensor(pos, dtype=torch.float32),
            torch.as_tensor(quat, dtype=torch.float32),
        )
    ).numpy()
    return pose, torch.as_tensor(intrinsic, dtype=torch.float32).numpy()


def _rgb_tensor(data: Any, torch: Any, label: str) -> Any:
    output = getattr(data, "output", None)
    if not isinstance(output, dict) or "rgb" not in output:
        raise LingBotDatasetRecorderError(f"{label} has no public RGB output")
    rgb = legacy._proxy_tensor(output["rgb"], f"{label}.data.output['rgb']", torch)
    if tuple(rgb.shape) != (1, 480, 640, 3) or rgb.dtype != torch.uint8:
        raise LingBotDatasetRecorderError(
            f"{label} RGB must be uint8 (1,480,640,3), got {rgb.dtype} {tuple(rgb.shape)}"
        )
    mean = float(rgb.float().mean().detach().cpu())
    std = float(rgb.float().std().detach().cpu())
    if not math.isfinite(mean) or not math.isfinite(std) or mean <= 2.0 or std <= 1.0:
        raise LingBotDatasetRecorderError(f"{label} RGB is black/non-finite: mean={mean}, std={std}")
    return rgb[0]


def _write_video(episode_dir: Path, view: str) -> None:
    output = episode_dir / "video" / f"{view}.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        "12.5",
        "-start_number",
        "0",
        "-i",
        str(episode_dir / "video" / view / "frame_%06d.png"),
        "-frames:v",
        "375",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if process.returncode != 0 or not output.is_file() or output.stat().st_size <= 0:
        raise LingBotDatasetRecorderError(
            f"ffmpeg failed for {view}: {process.stderr.strip() or process.stdout.strip()}"
        )


def _create_contact_sheet(
    episode_dir: Path,
    *,
    guard_step: int,
    threshold_step: int,
    success_step: int,
    rebound_step: int,
    frame_for_step: Any,
) -> list[dict[str, int | str]]:
    from PIL import Image, ImageDraw

    events = [
        ("start", 1),
        ("approach", 110),
        ("near-button", guard_step),
        ("pre-press", max(1, threshold_step - 1)),
        ("press-threshold", threshold_step),
        ("success", success_step),
        ("rebound", rebound_step),
        ("end", 3000),
    ]
    thumb_width, thumb_height = 320, 240
    label_height = 40
    canvas = Image.new("RGB", (thumb_width * 3, (thumb_height + label_height) * len(events)), "white")
    draw = ImageDraw.Draw(canvas)
    records: list[dict[str, int | str]] = []
    for row, (name, event_step) in enumerate(events):
        frame_index = int(frame_for_step(event_step))
        actual_step = 8 * (frame_index + 1)
        timestamp_ns = actual_step * 10_000_000
        y = row * (thumb_height + label_height)
        draw.text(
            (5, y + 4),
            f"{name}: event step {event_step}; selected frame {frame_index}; actual step {actual_step}; t={timestamp_ns / 1e9:.2f}s",
            fill="black",
        )
        for column, view in enumerate(("ego", "left", "right")):
            path = episode_dir / "video" / view / f"frame_{frame_index:06d}.png"
            with Image.open(path) as image:
                image = image.convert("RGB").resize((thumb_width, thumb_height))
                canvas.paste(image, (column * thumb_width, y + label_height))
            draw.text((column * thumb_width + 5, y + label_height + 5), view, fill="white", stroke_width=2, stroke_fill="black")
        records.append(
            {
                "event": name,
                "event_policy_step": event_step,
                "selected_frame_index": frame_index,
                "selected_policy_step": actual_step,
                "selected_timestamp_ns": timestamp_ns,
            }
        )
    canvas.save(episode_dir / "contact_sheet.png")
    return records


def _git_status(root: Path) -> list[str]:
    process = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise LingBotDatasetRecorderError(f"cannot inspect RAMBO dirty state: {process.stderr.strip()}")
    return process.stdout.splitlines()


def _record(args: Any, app_launcher_type: type[Any], staging: Path) -> dict[str, Any]:
    from rambo.validation import lingbot_dataset_v1 as schema

    app = None
    env = None
    runtime: dict[str, Any] | None = None
    try:
        app = app_launcher_type(args)
        runtime = legacy._runtime_imports()
        np = runtime["np"]
        torch = runtime["torch"]
        imageio = runtime["imageio"]
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        runtime_metadata = legacy._runtime_metadata(torch)
        provenance = legacy._collect_provenance(
            runtime_metadata,
            expected_isaaclab_tag=runtime["TARGET_ISAACLAB_TAG"],
            expected_isaaclab_commit=runtime["TARGET_ISAACLAB_COMMIT"],
            expected_isaacsim_version=runtime["TARGET_ISAACSIM_VERSION"],
        )
        checkpoint_path = args.checkpoint.expanduser().resolve()
        contract = runtime["contract_for_task"](schema.TASK_ID)
        checkpoint = runtime["load_verified_checkpoint"](checkpoint_path, contract)
        checkpoint_sha = runtime["sha256_file"](checkpoint_path)
        if checkpoint_sha != runtime["QUADRUPED_CHECKPOINT_SHA256"]:
            raise LingBotDatasetRecorderError("checkpoint SHA differs from the released model_2000.pt contract")
        if contract.observation_dim != schema.OBSERVATION_DIM or contract.action_dim != schema.POLICY_ACTION_DIM:
            raise LingBotDatasetRecorderError("checkpoint contract is not the required 405D/18D schema")

        button_position = tuple(float(value) for value in args.button_position)
        if len(button_position) != 3 or not all(math.isfinite(value) for value in button_position):
            raise LingBotDatasetRecorderError("Button position must contain three finite values")
        reference_button_position = (0.99, 0.142, 0.30)
        button_delta = tuple(current - reference for current, reference in zip(button_position, reference_button_position))
        env_cfg = runtime["parse_env_cfg"](
            schema.TASK_ID,
            device=args.device or "cuda:0",
            num_envs=1,
            use_fabric=not args.disable_fabric,
        )
        # The episode seed is reserved for the deterministic Button-position
        # sampler.  Keep the simulator/controller RNG at the golden seed so
        # Button translation is the only scene variable.
        runtime_seed = schema.SEED
        legacy._configure_button_cfg(
            env_cfg,
            seed=runtime_seed,
            steps=schema.POLICY_STEPS,
            configure_physx=runtime["configure_physx"],
        )
        env_cfg.button_cap_rest_pos = button_position
        env_cfg.button_panel_center = tuple(
            reference + delta for reference, delta in zip((1.06, 0.0, 0.50), button_delta)
        )
        env_cfg.button_indicator_pos = tuple(
            reference + delta for reference, delta in zip((0.99, 0.142, 0.46), button_delta)
        )
        env_cfg.button_status_marker_enabled = False
        env_cfg.front_camera.renderer_cfg = IsaacRtxRendererCfg()
        env_cfg.front_camera.update_latest_camera_pose = True
        agent_cfg = runtime["load_cfg_from_registry"](schema.TASK_ID, "crl2_cfg_entry_point")
        if not isinstance(agent_cfg, dict):
            raise LingBotDatasetRecorderError("RAMBO agent configuration is not a dictionary")
        agent_cfg["seed"] = runtime_seed
        agent_cfg["general"]["num_envs"] = 1

        env = runtime["Crl2VecEnvWrapper"](runtime["gym"].make(schema.TASK_ID, cfg=env_cfg))
        backend_before = runtime["assert_physx_environment"](env)
        base_env = env.unwrapped
        external_cameras: dict[str, Any] = {}
        for view in ("left", "right"):
            external_cameras[view] = Camera(
                CameraCfg(
                    prim_path=f"/World/LingBotDataset/{view.capitalize()}Camera",
                    update_period=0.0,
                    height=schema.IMAGE_HEIGHT,
                    width=schema.IMAGE_WIDTH,
                    data_types=["rgb"],
                    renderer_cfg=IsaacRtxRendererCfg(),
                    update_latest_camera_pose=True,
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=18.0,
                        focus_distance=400.0,
                        horizontal_aperture=20.955,
                        clipping_range=(0.1, 20.0),
                    ),
                )
            )
        base_env.sim.reset()
        legacy._seed_everything(runtime_seed, env, np, torch)
        observations, _ = env.reset()
        if tuple(observations.shape) != (1, schema.OBSERVATION_DIM) or observations.dtype != torch.float32:
            raise LingBotDatasetRecorderError(f"invalid initial observation: {observations.dtype} {tuple(observations.shape)}")
        if not bool(torch.isfinite(observations).all()):
            raise LingBotDatasetRecorderError("initial observation contains NaN or Inf")

        camera = getattr(base_env, "front_camera", None)
        if camera is None or base_env.scene.sensors.get("front_camera") is not camera:
            raise LingBotDatasetRecorderError("public Go2 front camera is unavailable")
        physics_dt_s = float(base_env.cfg.sim.dt)
        policy_dt_s = physics_dt_s * int(base_env.cfg.decimation)
        if not math.isclose(physics_dt_s, schema.PHYSICS_DT_NS / 1e9, abs_tol=1e-12, rel_tol=0.0):
            raise LingBotDatasetRecorderError(f"unexpected physics dt: {physics_dt_s}")
        if int(base_env.cfg.decimation) != schema.CONTROL_DECIMATION:
            raise LingBotDatasetRecorderError(f"unexpected control decimation: {base_env.cfg.decimation}")
        if not math.isclose(policy_dt_s, schema.POLICY_DT_NS / 1e9, abs_tol=1e-12, rel_tol=0.0):
            raise LingBotDatasetRecorderError(f"unexpected policy dt: {policy_dt_s}")
        if not math.isclose(float(camera.cfg.update_period), 0.08, abs_tol=1e-12, rel_tol=0.0):
            raise LingBotDatasetRecorderError("ego camera cadence is not 80 ms")

        eyes = {
            "left": [0.45, 2.40, 1.20],
            "right": [0.45, -2.40, 1.20],
        }
        target = [0.60, 0.00, 0.35]
        for view, external in external_cameras.items():
            eye_tensor = torch.tensor([eyes[view]], dtype=torch.float32, device=base_env.device)
            target_tensor = torch.tensor([target], dtype=torch.float32, device=base_env.device)
            external.set_world_poses_from_view(eyes=eye_tensor, targets=target_tensor)
        base_env.sim.render()
        for external in external_cameras.values():
            external.update(dt=physics_dt_s, force_recompute=True)

        # Reset-baseline images and frame IDs are deliberately not stored.
        ego_initial_data = camera.data
        _rgb_tensor(ego_initial_data, torch, "ego")
        ego_last_frame_id = _camera_frame_id(camera, torch, "ego")
        initial_ticks = legacy._proxy_tensor(camera.rambo_physics_ticks, "ego.rambo_physics_ticks", torch)
        if int(initial_ticks.detach().reshape(-1)[0].cpu().item()) != 0:
            raise LingBotDatasetRecorderError("ego camera tick counter is not zero at reset baseline")
        external_last_frame_ids: dict[str, int] = {}
        for view, external in external_cameras.items():
            data = external.data
            _rgb_tensor(data, torch, view)
            external_last_frame_ids[view] = _camera_frame_id(external, torch, view)

        runner = runtime["PPO"](
            task=schema.TASK_ID, env=env, agent_cfg=agent_cfg, train=False, device=env.device
        )
        runtime["restore_runner"](runner, checkpoint, load_values=False, verify=True)
        policy = runner.get_inference_policy(device=base_env.device)

        robot = legacy._resolve_public_robot(base_env)
        robot_data = robot.data
        initial_root_pose = np.concatenate(
            (
                _np_single(robot_data.root_link_pos_w, torch, "robot.root_link_pos_w"),
                _np_single(robot_data.root_link_quat_w, torch, "robot.root_link_quat_w"),
            )
        ).astype(np.float32)
        button_cap = getattr(base_env.scene, "rigid_objects", {}).get("button_cap")
        if button_cap is None:
            raise LingBotDatasetRecorderError("public button_cap rigid object is unavailable")
        initial_button_pose = np.concatenate(
            (
                _np_single(button_cap.data.root_link_pos_w, torch, "button_cap.root_link_pos_w"),
                _np_single(button_cap.data.root_link_quat_w, torch, "button_cap.root_link_quat_w"),
            )
        ).astype(np.float32)
        initial_button_state = {
            "displacement_m": float(base_env.button_displacement.detach().cpu().reshape(-1)[0]),
            "success": bool(base_env.button_success.detach().cpu().reshape(-1)[0]),
            "released": bool(base_env.button_released.detach().cpu().reshape(-1)[0]),
        }
        joint_dimension = int(_np_single(robot_data.joint_pos, torch, "robot.joint_pos").shape[0])

        raw_command = np.empty((schema.POLICY_STEPS, schema.RAW_COMMAND_DIM), dtype=np.float32)
        policy_action = np.empty((schema.POLICY_STEPS, schema.POLICY_ACTION_DIM), dtype=np.float32)
        observation = np.empty((schema.POLICY_STEPS, schema.OBSERVATION_DIM), dtype=np.float32)
        reward = np.empty((schema.POLICY_STEPS,), dtype=np.float32)
        root_pose = np.empty((schema.POLICY_STEPS, 7), dtype=np.float32)
        joint_pos = np.empty((schema.POLICY_STEPS, joint_dimension), dtype=np.float32)
        joint_vel = np.empty((schema.POLICY_STEPS, joint_dimension), dtype=np.float32)
        fl_foot_position = np.empty((schema.POLICY_STEPS, 3), dtype=np.float32)
        button_displacement = np.empty((schema.POLICY_STEPS,), dtype=np.float32)
        button_success = np.empty((schema.POLICY_STEPS,), dtype=np.bool_)
        button_released = np.empty((schema.POLICY_STEPS,), dtype=np.bool_)
        terminal = np.empty((schema.POLICY_STEPS,), dtype=np.bool_)
        contact_max_force = np.empty((schema.POLICY_STEPS,), dtype=np.float32)
        poses = {view: np.empty((schema.RGB_FRAME_COUNT, 7), dtype=np.float32) for view in schema.VIEWS}
        intrinsics: dict[str, Any] = {}
        frame_records: list[dict[str, Any]] = []
        staging_buffers: dict[str, Any] = {}

        leg_target = torch.tensor([[0.1934, 0.142, 0.05]], dtype=torch.float32, device=base_env.device)
        for action_step in range(1, schema.POLICY_STEPS + 1):
            index = action_step - 1
            with torch.inference_mode():
                leg_target, command = legacy._apply_loco_manip_schedule(
                    base_env, action_step=action_step, leg_target=leg_target, torch=torch
                )
                raw_command[index] = np.asarray(
                    command["base_velocity"][0] + command["fl_position"][0] + command["fl_force"][0],
                    dtype=np.float32,
                )
                action = policy(observations)
                if tuple(action.shape) != (1, schema.POLICY_ACTION_DIM) or action.dtype != torch.float32:
                    raise LingBotDatasetRecorderError(
                        f"invalid policy action at step {action_step}: {action.dtype} {tuple(action.shape)}"
                    )
                if not bool(torch.isfinite(action).all()):
                    raise LingBotDatasetRecorderError(f"policy action contains NaN/Inf at step {action_step}")
                policy_action[index] = action.detach().cpu().numpy()[0]
                observations, rewards, dones, _ = env.step(action)
                if not bool(torch.isfinite(observations).all()) or not bool(torch.isfinite(rewards).all()):
                    raise LingBotDatasetRecorderError(f"non-finite observation/reward at step {action_step}")
                observation[index] = observations.detach().cpu().numpy()[0]
                reward[index] = float(rewards.detach().cpu().reshape(-1)[0])
                state, _step_success, _step_released = legacy._post_state_record(
                    base_env, action_step=action_step, terminal=dones, torch=torch
                )
                root_pose[index, :3] = np.asarray(state["robot"]["root_link_pos_w_m"][0], dtype=np.float32)
                root_pose[index, 3:] = np.asarray(state["robot"]["root_link_quat_w_xyzw"][0], dtype=np.float32)
                joint_pos[index] = np.asarray(state["robot"]["joint_pos_rad"][0], dtype=np.float32)
                joint_vel[index] = np.asarray(state["robot"]["joint_vel_rad_s"][0], dtype=np.float32)
                fl_foot_position[index] = np.asarray(state["robot"]["fl_foot_link_pos_w_m"][0], dtype=np.float32)
                button_displacement[index] = float(state["button"]["displacement_m"][0])
                button_success[index] = bool(state["button"]["success"][0])
                button_released[index] = bool(state["button"]["released"][0])
                terminal[index] = bool(state["terminal"][0])
                contact_max_force[index] = float(state["contact_sensor"]["max_force_norm_n"][0])
                if terminal[index]:
                    raise LingBotDatasetRecorderError(f"episode terminated at policy step {action_step}")

                if action_step % schema.RGB_INTERVAL_POLICY_STEPS == 0:
                    frame_index = action_step // schema.RGB_INTERVAL_POLICY_STEPS - 1
                    ego_data = camera.data
                    ego_frame_id = _camera_frame_id(camera, torch, "ego")
                    if ego_frame_id != ego_last_frame_id + 1:
                        raise LingBotDatasetRecorderError(
                            f"ego frame ID is {ego_frame_id} at step {action_step}, expected {ego_last_frame_id + 1}"
                        )
                    ticks = legacy._proxy_tensor(camera.rambo_physics_ticks, "ego.rambo_physics_ticks", torch)
                    actual_ticks = int(ticks.detach().reshape(-1)[0].cpu().item())
                    expected_ticks = action_step * schema.CONTROL_DECIMATION
                    if actual_ticks != expected_ticks:
                        raise LingBotDatasetRecorderError(
                            f"ego tick is {actual_ticks} at step {action_step}, expected {expected_ticks}"
                        )
                    ego_last_frame_id = ego_frame_id

                    # One render of the current post-step state, followed by two
                    # manual sensor updates; neither call advances physics.
                    base_env.sim.render()
                    for external in external_cameras.values():
                        external.update(dt=physics_dt_s, force_recompute=True)

                    triplet: dict[str, Any] = {}
                    camera_items = {"ego": camera, **external_cameras}
                    for view, sensor in camera_items.items():
                        data = ego_data if view == "ego" else sensor.data
                        frame_id = ego_frame_id if view == "ego" else _camera_frame_id(sensor, torch, view)
                        if view != "ego":
                            expected_frame_id = external_last_frame_ids[view] + 1
                            if frame_id != expected_frame_id:
                                raise LingBotDatasetRecorderError(
                                    f"{view} frame ID is {frame_id}, expected {expected_frame_id}"
                                )
                            external_last_frame_ids[view] = frame_id
                        rgb = _rgb_tensor(data, torch, view)
                        pose, intrinsic = _camera_state(data, torch, view)
                        if view == "ego":
                            pose = schema.ego_pose_from_root_pose(root_pose[index])
                        poses[view][frame_index] = pose
                        if view not in intrinsics:
                            intrinsics[view] = intrinsic.astype(np.float32)
                        elif not np.allclose(intrinsics[view], intrinsic, atol=1e-5, rtol=0.0):
                            raise LingBotDatasetRecorderError(f"{view} intrinsics changed during recording")
                        if view not in staging_buffers:
                            staging_buffers[view] = legacy._RgbCpuStaging(rgb, torch)
                        relative_path = f"video/{view}/frame_{frame_index:06d}.png"
                        imageio.imwrite(staging / relative_path, staging_buffers[view].copy_from(rgb))
                        triplet[view] = {"path": relative_path, "sensor_frame_id": frame_id}
                    first_train = 4 * frame_index
                    frame_records.append(
                        {
                            "frame_index": frame_index,
                            "policy_step": action_step,
                            "timestamp_ns": action_step * schema.POLICY_DT_NS,
                            "preceding_train_indices": list(range(first_train, first_train + 4)),
                            "views": triplet,
                        }
                    )

        if len(frame_records) != schema.RGB_FRAME_COUNT:
            raise LingBotDatasetRecorderError(f"captured {len(frame_records)} triplets, expected 375")
        if not bool(button_success.any()):
            raise LingBotDatasetRecorderError("real Button detector never reported success")
        events = schema.derive_button_events(button_displacement, button_success)
        if events["computed_hold_success_step"] != events["detector_success_step"]:
            raise LingBotDatasetRecorderError("Button success detector disagrees with displacement/hold trace")
        if events["first_rebound_step"] is None:
            raise LingBotDatasetRecorderError("Button did not rebound after success")
        guard_indices = np.flatnonzero(button_displacement >= float(base_env.cfg.button_contact_guard_m))
        if not guard_indices.size:
            raise LingBotDatasetRecorderError("Button never crossed the configured contact guard")
        guard_step = int(guard_indices[0] + 1)
        threshold_step = int(events["first_press_threshold_step"])
        success_step = int(events["detector_success_step"])
        rebound_step = int(events["first_rebound_step"])

        action_indices = schema.expected_action_indices()
        action_arrays = {
            "raw": raw_command,
            "raw_timestamps": action_indices["raw_timestamps"],
            "raw_indices": action_indices["raw_indices"],
            "train": raw_command[::2].copy(),
            "train_timestamps": action_indices["train_timestamps"],
            "train_indices": action_indices["train_indices"],
            "train_source_raw_indices": action_indices["train_source_raw_indices"],
        }
        schema.validate_action_arrays(action_arrays)
        for key, relative in schema.ACTION_FILES.items():
            np.save(staging / relative, action_arrays[key], allow_pickle=False)
        debug_arrays = {
            "policy_action": policy_action,
            "observation": observation,
            "reward": reward,
            "root_pose": root_pose,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "fl_foot_position": fl_foot_position,
            "button_displacement": button_displacement,
            "button_success": button_success,
            "button_released": button_released,
            "terminal": terminal,
            "contact_max_force": contact_max_force,
        }
        for key, relative in schema.RAW_DEBUG_FILES.items():
            np.save(staging / relative, debug_arrays[key], allow_pickle=False)
        for view in schema.VIEWS:
            np.save(staging / f"video/camera/{view}_pose_w.npy", poses[view], allow_pickle=False)
            np.save(staging / f"video/camera/{view}_intrinsics.npy", intrinsics[view], allow_pickle=False)
        schema.write_json(staging / "video/frame_index.json", {"schema_version": 1, "frames": frame_records})

        backend_after = runtime["assert_physx_environment"](env)
        repo_root = Path(__file__).resolve().parents[2]
        source_hashes = {
            "recorder_sha256": schema.sha256_file(Path(__file__).resolve()),
            "schema_sha256": schema.sha256_file(Path(schema.__file__).resolve()),
            "validator_sha256": schema.sha256_file(Path(__file__).with_name("validate_lingbot_dataset_v1.py")),
        }
        camera_common = {
            "resolution": [schema.IMAGE_WIDTH, schema.IMAGE_HEIGHT],
            "focal_length_mm": 18.0,
            "horizontal_aperture_mm": 20.955,
            "clipping_range_m": [0.1, 20.0],
            "renderer": "IsaacRtxRendererCfg",
        }
        cameras = {}
        for view in schema.VIEWS:
            cameras[view] = {
                **camera_common,
                "pose_trace_file": f"video/camera/{view}_pose_w.npy",
                "intrinsics_file": f"video/camera/{view}_intrinsics.npy",
                "pose_format": "[x,y,z,qx,qy,qz,qw] world frame",
                "world_fixed": view != "ego",
            }
            if view in eyes:
                cameras[view]["configured_eye_w_m"] = eyes[view]
                cameras[view]["configured_target_w_m"] = target
            else:
                cameras[view]["pose_source"] = (
                    "post-step robot root pose composed with fixed Go2 front-camera mount [0.30,0.00,0.08]"
                )
        metadata = {
            "dataset": schema.DATASET_NAME,
            "dataset_version": schema.DATASET_VERSION,
            "schema_version": schema.SCHEMA_VERSION,
            "episode_id": args.episode_id,
            "task_id": schema.TASK_ID,
            "task_instruction": schema.TASK_INSTRUCTION,
            "seed": args.seed,
            "runtime_seed": runtime_seed,
            "duration_s": 30.0,
            "rates_hz": {
                "physics": schema.PHYSICS_RATE_HZ,
                "policy": schema.POLICY_RATE_HZ,
                "raw_command": schema.POLICY_RATE_HZ,
                "train_command": schema.TRAIN_RATE_HZ,
                "rgb": schema.RGB_RATE_HZ,
            },
            "dimensions": {
                "high_level_command": schema.RAW_COMMAND_DIM,
                "policy_action": schema.POLICY_ACTION_DIM,
                "observation": schema.OBSERVATION_DIM,
                "joint_state": joint_dimension,
            },
            "counts": {
                "policy_steps": schema.POLICY_STEPS,
                "raw_commands": schema.RAW_COMMAND_COUNT,
                "train_commands": schema.TRAIN_COMMAND_COUNT,
                "rgb_frames_per_view": schema.RGB_FRAME_COUNT,
            },
            "command_columns": list(schema.COMMAND_COLUMNS),
            "cameras": cameras,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "iteration": int(checkpoint["iteration"]),
                "strict_load_verified": True,
            },
            "rambo": {
                "git_commit": provenance["rambo_git_commit"],
                "git_dirty": provenance["rambo_git_dirty"],
                "git_status_porcelain": _git_status(repo_root),
            },
            "source_hashes": source_hashes,
            "runtime": {
                **runtime_metadata,
                "isaac_lab": provenance["isaaclab"],
                "physx_before": backend_before,
                "physx_after": backend_after,
            },
            "success": {
                "value": True,
                "first_policy_step": success_step,
                "first_timestamp_ns": success_step * schema.POLICY_DT_NS,
            },
            "timestamp_conventions": {
                "raw_command": "interval start, i*10 ms",
                "train_command": "interval start, j*20 ms",
                "rgb": "post-step state, (k+1)*80 ms",
            },
        }
        if args.episode_id != schema.EPISODE_ID:
            if args.randomization_config is None or args.randomization_seed is None:
                raise LingBotDatasetRecorderError("randomized episodes require --randomization-config and --randomization-seed")
            config = json.loads(args.randomization_config.read_text(encoding="utf-8"))
            metadata["randomization"] = {
                "enabled": True,
                "parameter": "button_position",
                "seed": args.randomization_seed,
                "config_path": str(args.randomization_config.resolve()),
                "config_version": config.get("version"),
                "reference_position": list(reference_button_position),
                "sampled_position": list(button_position),
                "delta": list(button_delta),
                "orientation_fixed": [0.0, 0.0, 0.0, 1.0],
            }
        task = {
            "instruction": schema.TASK_INSTRUCTION,
            "task_id": schema.TASK_ID,
            "command_program_id": schema.COMMAND_PROGRAM_ID,
            "goal": {"type": "button_press"},
            "initial_state": {
                "robot_root_pose_w_xyzw": initial_root_pose.tolist(),
                "button_cap_pose_w_xyzw": initial_button_pose.tolist(),
                "button": initial_button_state,
            },
            "button": {
                "cap_rest_position_w_m": list(base_env.cfg.button_cap_rest_pos),
                "panel_center_w_m": list(base_env.cfg.button_panel_center),
                "stroke_m": float(base_env.cfg.button_stroke_m),
                "press_threshold_m": float(base_env.cfg.button_press_threshold_m),
                "release_threshold_m": float(base_env.cfg.button_release_threshold_m),
                "hold_policy_steps": int(base_env.cfg.button_hold_steps),
                "contact_guard_m": float(base_env.cfg.button_contact_guard_m),
                "status_marker_rendered": bool(base_env.cfg.button_status_marker_enabled),
            },
            "command_program": {
                "source": "record_button_physx_episode._apply_loco_manip_schedule",
                "target_integration_dt_s": policy_dt_s,
                "fl_position_lower_m": [0.1934, 0.0, 0.0],
                "fl_position_upper_m": [0.50, 0.20, 0.40],
            },
        }
        outcome = {
            "success": True,
            "first_press_threshold_policy_step": threshold_step,
            "first_press_threshold_timestamp_ns": threshold_step * schema.POLICY_DT_NS,
            "first_success_policy_step": success_step,
            "first_success_timestamp_ns": success_step * schema.POLICY_DT_NS,
            "first_rebound_policy_step": rebound_step,
            "first_rebound_timestamp_ns": rebound_step * schema.POLICY_DT_NS,
            "max_button_displacement_m": float(events["max_displacement_m"]),
            "terminal": False,
            "terminal_reason": "fixed_duration_complete",
            "duration_s": 30.0,
        }
        sync = {
            "policy_step_index_base": 1,
            "raw_command_index_base": 0,
            "train_command_index_base": 0,
            "rgb_frame_index_base": 0,
            "command_timestamp_semantics": "interval_start",
            "rgb_timestamp_semantics": "post_step",
            "raw_command_formula": "raw[i] applies during [i*10ms,(i+1)*10ms), policy_step=i+1",
            "train_command_formula": "train[j]=raw[2*j], timestamp=j*20ms",
            "rgb_formula": "frame[k] is post policy_step=8*(k+1), timestamp=(k+1)*80ms",
            "rgb_to_train_formula": "frame[k] follows train indices [4*k,4*k+3]",
            "physics_formula": "policy_step n completes after 5*n PhysX ticks",
        }
        schema.write_json(staging / "metadata.json", metadata)
        schema.write_json(staging / "task.json", task)
        schema.write_json(staging / "outcome.json", outcome)
        schema.write_json(staging / "sync.json", sync)

        contact_events = _create_contact_sheet(
            staging,
            guard_step=guard_step,
            threshold_step=threshold_step,
            success_step=success_step,
            rebound_step=rebound_step,
            frame_for_step=schema.rgb_frame_index_for_policy_step,
        )
        task["contact_sheet_events"] = contact_events
        schema.write_json(staging / "task.json", task)
        for view in schema.VIEWS:
            _write_video(staging, view)
        schema.write_checksums(staging)
        validation = schema.validate_episode(staging, verify_checksums=True, verify_images=True)
        print(json.dumps(validation, indent=2, sort_keys=True), flush=True)
        print("LINGBOT_RAMBO_DATASET_V1_INTERNAL_VALIDATION_PASS", flush=True)
        return validation
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.app.close()


def main() -> int:
    parser, app_launcher_type = _build_parser()
    args = parser.parse_args()
    from rambo.utils.physx import validate_rambo_visualizer_args
    from rambo.validation import lingbot_dataset_v1 as schema

    validate_rambo_visualizer_args(parser, args, sys.argv[1:])
    args.episode_id = args.episode_id or args.output_dir.expanduser().resolve().name
    if not __import__("re").fullmatch(r"episode_\d{6}", args.episode_id):
        parser.error("--episode-id must use episode_000001-style naming")
    episode_number = int(args.episode_id.rsplit("_", 1)[1])
    if episode_number == 1 and args.seed != schema.SEED:
        parser.error("episode_000001 requires --seed 42")
    if episode_number > 1 and args.seed != episode_number:
        parser.error("randomized episodes require --seed equal to their numeric episode ID")
    if args.steps != schema.POLICY_STEPS:
        parser.error("Dataset V1 golden sample requires exactly --steps 3000")
    if args.output_dir.expanduser().resolve().name != args.episode_id:
        parser.error("--output-dir basename must equal --episode-id")
    if episode_number > 1 and (args.randomization_config is None or args.randomization_seed != args.seed):
        parser.error("randomized episodes require --randomization-config and --randomization-seed equal to --seed")
    args.enable_cameras = True
    # Atomic publication happens after SimulationApp.close() returns.  Kit's
    # fast-shutdown path terminates the interpreter inside close(), which would
    # strand an otherwise validated artifact in its .inprogress directory.
    args.fast_shutdown = False
    args.livestream = 0
    target: Path
    staging: Path
    try:
        target, staging = _prepare_staging(args.output_dir)
        validation = _record(args, app_launcher_type, staging)
        if not validation.get("passed"):
            raise LingBotDatasetRecorderError("internal offline validation did not pass")
        os.replace(staging, target)
        print(f"PUBLISHED_EPISODE={target}", flush=True)
        print("LINGBOT_RAMBO_DATASET_V1_GOLDEN_SAMPLE_RECORDED", flush=True)
        return 0
    except BaseException as error:
        traceback.print_exception(type(error), error, error.__traceback__)
        print(f"No formal episode was published; staging may remain at {locals().get('staging', 'uncreated')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
