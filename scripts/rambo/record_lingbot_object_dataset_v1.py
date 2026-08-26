#!/usr/bin/env python3
"""Record one procedural LingBot-RAMBO object-task Dataset V1 episode."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

import record_button_physx_episode as legacy
import record_lingbot_dataset_v1 as base


class ObjectRecorderError(RuntimeError):
    pass


def _parser() -> tuple[argparse.ArgumentParser, type[Any]]:
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-profile", choices=("lift_basket", "pull_object_into_basket", "shoot_ball_into_goal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--randomization-config", type=Path)
    parser.add_argument("--randomization-seed", type=int)
    parser.add_argument("--primary-position", type=float, nargs=3, metavar=("X", "Y", "Z"), required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--disable-fabric", "--disable_fabric", dest="disable_fabric", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser, AppLauncher


def _schedule(name: str, step: int, target: Any, torch: Any) -> tuple[Any, dict[str, Any]]:
    """Task-local deterministic 9D commands; the policy residual remains native."""
    velocity = torch.zeros((1, 3), dtype=target.dtype, device=target.device)
    force = torch.zeros((1, 3), dtype=target.dtype, device=target.device)
    target_velocity = torch.zeros((1, 3), dtype=target.dtype, device=target.device)
    # Keep the torso stationary: the task objects are deliberately within the
    # published FL workspace.  Unlike an abrupt Cartesian target replacement,
    # every phase integrates a bounded 100-Hz command exactly as the accepted
    # Button program does, which keeps QP/PhysX stable.
    if 110 <= step < 190:
        velocity[:, 0] = 0.12
    if name == "lift_basket":
        if 300 <= step < 500:
            target_velocity[:, 2] = 0.10
        elif 500 <= step < 750:
            target_velocity[:, 0] = 0.12
        elif 750 <= step < 900:
            target_velocity[:, 2] = 0.12
            force[:, 2] = 20.0
        elif 900 <= step < 1100:
            target_velocity[:, 2] = -0.12
        elif 1100 <= step < 1350:
            target_velocity[:, 0] = -0.12
    elif name == "pull_object_into_basket":
        if 300 <= step < 550:
            target_velocity[:, 0] = 0.12
        elif 550 <= step < 800:
            target_velocity[:, 0] = -0.12
    else:
        if 300 <= step < 550:
            target_velocity[:, 0] = 0.12
        elif 550 <= step < 650:
            target_velocity[:, 0] = -0.12
            force[:, 0] = 18.0
    lower = torch.tensor([[0.1934, 0.0, 0.0]], dtype=target.dtype, device=target.device)
    upper = torch.tensor([[0.50, 0.20, 0.40]], dtype=target.dtype, device=target.device)
    target = torch.clamp(target + target_velocity * 0.01, lower, upper)
    return target, {"base_velocity": velocity, "fl_position": target, "fl_force": force}


def _contact_sheet(root: Path, profile: str, events: dict[str, Any]) -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw
    success = int(events["first_success_step"])
    contact = int(events["first_contact_step"])
    labels = [("start", 1), ("approach", 110), ("contact", contact), ("pre-success", max(1, success - 1)), ("success", success), ("hold", min(3000, success + 25)), ("end", 3000)]
    thumb = (320, 240); label_h = 36
    canvas = Image.new("RGB", (960, len(labels) * (thumb[1] + label_h)), "white")
    draw = ImageDraw.Draw(canvas); result = []
    for row, (label, step) in enumerate(labels):
        idx = min(374, math.ceil(step / 8) - 1); actual = 8 * (idx + 1); y = row * (thumb[1] + label_h)
        draw.text((4, y + 3), f"{label}: event step {step}; selected frame {idx}; actual step {actual}; t={actual / 100:.2f}s", fill="black")
        for column, view in enumerate(("ego", "left", "right")):
            with Image.open(root / "video" / view / f"frame_{idx:06d}.png") as image:
                canvas.paste(image.convert("RGB").resize(thumb), (column * thumb[0], y + label_h))
            draw.text((column * thumb[0] + 5, y + label_h + 5), view, fill="white", stroke_width=2, stroke_fill="black")
        result.append({"event": label, "event_policy_step": step, "selected_frame_index": idx, "selected_policy_step": actual, "selected_timestamp_ns": actual * 10_000_000})
    canvas.save(root / "contact_sheet.png")
    return result


def _record(args: Any, launcher: type[Any], staging: Path) -> dict[str, Any]:
    from rambo.validation import lingbot_dataset_v1 as common
    from rambo.validation import lingbot_object_dataset_v1 as schema
    app = env = None
    try:
        app = launcher(args); runtime = legacy._runtime_imports(); torch = runtime["torch"]; imageio = runtime["imageio"]
        from isaaclab.sensors import Camera, CameraCfg
        import isaaclab.sim as sim_utils
        from isaaclab_physx.renderers import IsaacRtxRendererCfg
        cfg = schema.profile(args.task_profile)
        task_id = cfg["task_id"]
        checkpoint_path = args.checkpoint.resolve(); contract = runtime["contract_for_task"](task_id)
        checkpoint = runtime["load_verified_checkpoint"](checkpoint_path, contract)
        if contract.observation_dim != common.OBSERVATION_DIM or contract.action_dim != common.POLICY_ACTION_DIM:
            raise ObjectRecorderError("released checkpoint contract mismatch")
        env_cfg = runtime["parse_env_cfg"](task_id, device=args.device or "cuda:0", num_envs=1, use_fabric=not args.disable_fabric)
        legacy._configure_button_cfg(env_cfg, seed=common.SEED, steps=common.POLICY_STEPS, configure_physx=runtime["configure_physx"])
        position = tuple(float(v) for v in args.primary_position)
        env_cfg.primary_position = position
        env_cfg.front_camera.renderer_cfg = IsaacRtxRendererCfg(); env_cfg.front_camera.update_latest_camera_pose = True
        agent_cfg = runtime["load_cfg_from_registry"](task_id, "crl2_cfg_entry_point"); agent_cfg["seed"] = common.SEED; agent_cfg["general"]["num_envs"] = 1
        env = runtime["Crl2VecEnvWrapper"](runtime["gym"].make(task_id, cfg=env_cfg)); backend_before = runtime["assert_physx_environment"](env)
        base_env = env.unwrapped
        camera_params = {
            "lift_basket": ({"left": [0.50, 2.40, 1.25], "right": [0.50, -2.40, 1.25]}, [0.65, 0.10, 0.25]),
            "pull_object_into_basket": ({"left": [0.70, 2.70, 1.40], "right": [0.70, -2.70, 1.40]}, [0.85, 0.00, 0.38]),
            "shoot_ball_into_goal": ({"left": [0.90, 3.00, 1.50], "right": [0.90, -3.00, 1.50]}, [1.00, 0.00, 0.25]),
        }[args.task_profile]
        eyes, target = camera_params; external = {}
        for view in ("left", "right"):
            sensor = Camera(CameraCfg(prim_path=f"/World/LingBotObjectDataset{args.task_profile.title().replace('_', '')}{view.capitalize()}Camera", update_period=0.0, height=480, width=640, data_types=["rgb"], renderer_cfg=IsaacRtxRendererCfg(), update_latest_camera_pose=True, spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0))))
            external[view] = sensor
        base_env.sim.reset(); legacy._seed_everything(common.SEED, env, runtime["np"], torch); observations, _ = env.reset()
        camera = base_env.front_camera; physics_dt = float(base_env.cfg.sim.dt)
        for view, sensor in external.items():
            sensor.set_world_poses_from_view(eyes=torch.tensor([eyes[view]], dtype=torch.float32, device=base_env.device), targets=torch.tensor([target], dtype=torch.float32, device=base_env.device))
        base_env.sim.render(); [sensor.update(dt=physics_dt, force_recompute=True) for sensor in external.values()]
        ego_last = base._camera_frame_id(camera, torch, "ego"); external_last = {view: base._camera_frame_id(sensor, torch, view) for view, sensor in external.items()}
        runner = runtime["PPO"](task=task_id, env=env, agent_cfg=agent_cfg, train=False, device=env.device); runtime["restore_runner"](runner, checkpoint, load_values=False, verify=True); policy = runner.get_inference_policy(device=base_env.device)
        robot = legacy._resolve_public_robot(base_env); joint_dim = int(base._np_single(robot.data.joint_pos, torch, "joint").shape[0])
        initial_robot = np.concatenate((base._np_single(robot.data.root_link_pos_w, torch, "initial root"), base._np_single(robot.data.root_link_quat_w, torch, "initial quat"))).astype(np.float32)
        initial_primary = base_env.primary_pose_w.detach().cpu().numpy()[0].astype(np.float32)
        raw = np.empty((3000, 9), np.float32); actions = np.empty((3000, 18), np.float32); obs = np.empty((3000, 405), np.float32); rewards = np.empty(3000, np.float32); root = np.empty((3000, 7), np.float32); jpos = np.empty((3000, joint_dim), np.float32); jvel = np.empty((3000, joint_dim), np.float32); foot = np.empty((3000, 3), np.float32); terminal = np.empty(3000, np.bool_); contact_max = np.empty(3000, np.float32)
        primary_pose = np.empty((3000, 7), np.float32); primary_velocity = np.empty((3000, 6), np.float32); success = np.empty(3000, np.bool_); fl_contact = np.empty(3000, np.bool_); task_metrics: dict[str, np.ndarray] = {}
        metric_names = {"lift_basket": ("clearance_m", "tilt_rad"), "pull_object_into_basket": ("inside_basket", "speed_m_s", "left_table"), "shoot_ball_into_goal": ("scored", "behind_goal_m")}[args.task_profile]
        for name in metric_names: task_metrics[name] = np.empty(3000, np.float32 if name not in ("inside_basket", "left_table", "scored") else np.bool_)
        poses = {view: np.empty((375, 7), np.float32) for view in common.VIEWS}; intrinsics: dict[str, Any] = {}; frames=[]; buffers={}; leg = torch.tensor([[0.1934, 0.142, 0.05]], dtype=torch.float32, device=base_env.device)
        for step in range(1, 3001):
            i = step - 1
            with torch.inference_mode():
                leg, command = _schedule(args.task_profile, step, leg, torch); base_env.set_loco_manip_commands(**command); raw[i] = np.asarray(command["base_velocity"][0].cpu().tolist() + command["fl_position"][0].cpu().tolist() + command["fl_force"][0].cpu().tolist(), np.float32)
                action = policy(observations); actions[i] = action.detach().cpu().numpy()[0]; observations, reward_t, dones, _ = env.step(action); obs[i] = observations.detach().cpu().numpy()[0]; rewards[i] = float(reward_t.detach().cpu().reshape(-1)[0]); terminal[i] = bool(dones.detach().cpu().reshape(-1)[0])
                if terminal[i] or not np.isfinite(obs[i]).all() or not np.isfinite(rewards[i]): raise ObjectRecorderError(f"invalid terminal/nonfinite state at {step}")
                root[i] = np.concatenate((base._np_single(robot.data.root_link_pos_w, torch, "root"), base._np_single(robot.data.root_link_quat_w, torch, "quat"))).astype(np.float32); jpos[i] = base._np_single(robot.data.joint_pos, torch, "jpos"); jvel[i] = base._np_single(robot.data.joint_vel, torch, "jvel"); foot[i] = base._np_single(robot.data.body_link_pos_w, torch, "foot")[int(base_env.feet_ids[0])]
                primary_pose[i] = base_env.primary_pose_w.detach().cpu().numpy()[0]; primary_velocity[i] = base_env.primary_velocity_w.detach().cpu().numpy()[0]; success[i] = bool(base_env.task_success.detach().cpu()[0]); fl_contact[i] = bool(base_env.fl_contact_seen.detach().cpu()[0]); metrics = base_env.task_metrics
                for name in metric_names: task_metrics[name][i] = metrics[name].detach().cpu().numpy()[0]
                contact_max[i] = float(torch.linalg.vector_norm(base_env._contact_sensor.data.net_forces_w_history.torch, dim=-1).amax().detach().cpu())
                if step % 8 == 0:
                    k = step // 8 - 1; ego_data = camera.data; ego_id = base._camera_frame_id(camera, torch, "ego")
                    if ego_id != ego_last + 1: raise ObjectRecorderError("ego RGB cadence failure")
                    ego_last = ego_id; base_env.sim.render(); [sensor.update(dt=physics_dt, force_recompute=True) for sensor in external.values()]; triplet={}
                    for view, sensor in {"ego": camera, **external}.items():
                        data = ego_data if view == "ego" else sensor.data; fid = ego_id if view == "ego" else base._camera_frame_id(sensor, torch, view)
                        if view != "ego":
                            if fid != external_last[view] + 1: raise ObjectRecorderError(f"{view} RGB cadence failure")
                            external_last[view] = fid
                        rgb = base._rgb_tensor(data, torch, view); pose, intrinsic = base._camera_state(data, torch, view); poses[view][k] = common.ego_pose_from_root_pose(root[i]) if view == "ego" else pose
                        intrinsics.setdefault(view, intrinsic.astype(np.float32)); buffers.setdefault(view, legacy._RgbCpuStaging(rgb, torch)); relative=f"video/{view}/frame_{k:06d}.png"; imageio.imwrite(staging / relative, buffers[view].copy_from(rgb)); triplet[view] = {"path": relative, "sensor_frame_id": fid}
                    frames.append({"frame_index": k, "policy_step": step, "timestamp_ns": step * 10_000_000, "preceding_train_indices": list(range(4*k, 4*k+4)), "views": triplet})
        if len(frames) != 375: raise ObjectRecorderError("RGB frame count failure")
        metrics_for_validation = {"fl_contact": fl_contact, **task_metrics}; events = schema.derive_events(args.task_profile, metrics_for_validation, success)
        if not events["success"]: raise ObjectRecorderError("real task detector never reported success")
        arrays = {"raw": raw, "raw_timestamps": np.arange(3000, dtype=np.int64)*10_000_000, "raw_indices": np.arange(3000, dtype=np.int64), "train": raw[::2].copy(), "train_timestamps": np.arange(1500, dtype=np.int64)*20_000_000, "train_indices": np.arange(1500, dtype=np.int64), "train_source_raw_indices": np.arange(1500, dtype=np.int64)*2}
        for key, relative in common.ACTION_FILES.items(): np.save(staging / relative, arrays[key], allow_pickle=False)
        debug = {"policy_action_18d": actions, "observation_405d": obs, "reward": rewards, "root_pose_w": root, "joint_pos": jpos, "joint_vel": jvel, "fl_foot_position_w": foot, "terminal": terminal, "contact_max_force_n": contact_max, "primary_pose_w": primary_pose, "primary_velocity_w": primary_velocity, "task_success": success, "fl_contact": fl_contact, **task_metrics}
        for name, value in debug.items(): np.save(staging / "raw_debug" / f"{name}.npy", value, allow_pickle=False)
        for view in common.VIEWS: np.save(staging / f"video/camera/{view}_pose_w.npy", poses[view], allow_pickle=False); np.save(staging / f"video/camera/{view}_intrinsics.npy", intrinsics[view], allow_pickle=False); base._write_video(staging, view)
        common.write_json(staging / "video/frame_index.json", {"schema_version": 1, "frames": frames})
        cameras={view: {"resolution": [640,480], "focal_length_mm":18.0, "horizontal_aperture_mm":20.955, "clipping_range_m":[0.1,20.0], "renderer":"IsaacRtxRendererCfg", "world_fixed":view!="ego", "pose_trace_file":f"video/camera/{view}_pose_w.npy", "intrinsics_file":f"video/camera/{view}_intrinsics.npy"} for view in common.VIEWS}
        for view in external: cameras[view].update({"configured_eye_w_m": eyes[view], "configured_target_w_m":target})
        metadata={"dataset":"lingbot_rambo","dataset_version":"1.0","schema_version":1,"task_profile":args.task_profile,"episode_id":args.episode_id,"task_id":task_id,"task_instruction":cfg["instruction"],"seed":args.seed,"runtime_seed":common.SEED,"duration_s":30.0,"rates_hz":{"physics":500,"policy":100,"raw_command":100,"train_command":50,"rgb":12.5},"dimensions":{"high_level_command":9,"policy_action":18,"observation":405,"joint_state":joint_dim},"counts":{"policy_steps":3000,"raw_commands":3000,"train_commands":1500,"rgb_frames_per_view":375},"command_columns":list(common.COMMAND_COLUMNS),"cameras":cameras,"checkpoint":{"path":str(checkpoint_path),"sha256":runtime["sha256_file"](checkpoint_path),"iteration":int(checkpoint["iteration"]),"strict_load_verified":True},"runtime":{"physx_before":backend_before,"physx_after":runtime["assert_physx_environment"](env)},"success":{"value":True,"first_policy_step":events["first_success_step"],"first_timestamp_ns":events["first_success_step"]*10_000_000}}
        if args.seed != 42:
            conf=json.loads(args.randomization_config.read_text()); metadata["randomization"]={"enabled":True,"parameter":cfg["primary_key"]+"_position","seed":args.randomization_seed,"config_path":str(args.randomization_config.resolve()),"config_version":conf["version"],"reference_position":conf["reference_primary_position"],"sampled_position":list(position),"delta":(np.asarray(position)-np.asarray(conf["reference_primary_position"])).tolist(),"orientation_fixed":[0,0,0,1]}
        task={"instruction":cfg["instruction"],"task_id":task_id,"command_program_id":cfg["command_program_id"],"goal":{"type":cfg["goal_type"]},"initial_state":{"robot_root_pose_w_xyzw":initial_robot.tolist(),"primary_pose_w_xyzw":initial_primary.tolist()},"scene":{"task_kind":args.task_profile,"primary_position_w_m":list(position),"geometry":cfg["geometry"],"detector":events},"command_program":{"source":"record_lingbot_object_dataset_v1._schedule","phases":"approach/engage/actuate/recover","target_integration_dt_s":0.01,"fl_position_lower_m":[0.1934,0,0],"fl_position_upper_m":[0.5,0.2,0.4]}}
        outcome={"success":True,"first_contact_policy_step":events["first_contact_step"],"first_success_policy_step":events["first_success_step"],"first_success_timestamp_ns":events["first_success_step"]*10_000_000,"evidence":{key:value for key,value in events.items() if key not in ("success","first_success_step","first_contact_step")},"terminal":False,"terminal_reason":"fixed_duration_complete","duration_s":30.0}
        common.write_json(staging / "metadata.json", metadata); common.write_json(staging / "task.json", task); common.write_json(staging / "outcome.json", outcome); common.write_json(staging / "sync.json", {"policy_step_index_base":1,"raw_command_index_base":0,"train_command_index_base":0,"rgb_frame_index_base":0,"raw_command_formula":"raw[i] applies during [i*10ms,(i+1)*10ms), policy_step=i+1","train_command_formula":"train[j]=raw[2*j], timestamp=j*20ms","rgb_formula":"frame[k] is post policy_step=8*(k+1), timestamp=(k+1)*80ms","rgb_to_train_formula":"frame[k] follows train indices [4*k,4*k+3]","physics_formula":"policy_step n completes after 5*n PhysX ticks"})
        task["contact_sheet_events"] = _contact_sheet(staging, args.task_profile, events); common.write_json(staging / "task.json", task); common.write_checksums(staging)
        result=schema.validate_episode(staging); print(json.dumps(result, indent=2)); return result
    finally:
        if env is not None: env.close()
        if app is not None: app.app.close()


def main() -> int:
    parser, launcher = _parser(); args = parser.parse_args()
    from rambo.utils.physx import validate_rambo_visualizer_args
    from rambo.validation import lingbot_dataset_v1 as common
    validate_rambo_visualizer_args(parser, args, sys.argv[1:]); number=int(args.episode_id.rsplit("_",1)[1])
    if args.steps != 3000 or (number == 1 and args.seed != 42) or (number > 1 and (args.seed != number or args.randomization_seed != number or args.randomization_config is None)): parser.error("invalid Dataset V1 episode seed/configuration")
    args.enable_cameras=True; args.fast_shutdown=False; args.livestream=0
    try:
        target, staging = base._prepare_staging(args.output_dir); result=_record(args, launcher, staging); os.replace(staging, target); print(f"PUBLISHED_EPISODE={target}"); return 0
    except BaseException as error:
        traceback.print_exception(type(error), error, error.__traceback__); return 1


if __name__ == "__main__": raise SystemExit(main())
