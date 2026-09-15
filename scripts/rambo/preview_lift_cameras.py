"""Bounded dual-camera render diagnostics, never dataset publication."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np


def main():
    from isaaclab.app import AppLauncher
    from rambo.utils.physx import validate_rambo_visualizer_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=240)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    mode = validate_rambo_visualizer_args(parser, args, sys.argv[1:])
    if not 8 <= args.steps <= 1600 or args.steps % 8:
        parser.error("--steps must be a multiple of eight in [8,1600]")
    args.enable_cameras = True
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    workspace = Path(os.environ["WORKSPACE_ROOT"])
    repo = Path(__file__).resolve().parents[2]
    report = {"kind": "camera_setup_preview", "passed": False, "dataset_episode": False,
              "physical_keyboard_tested": False, "visualizer": mode}
    app = env = None
    code = 1
    rows = []
    started = time.monotonic()
    try:
        app = AppLauncher(args)
        from rambo.tasks.common import lift_camera_rig as rig
        from PIL import Image
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Usd, UsdGeom, UsdPhysics

        spec = importlib.util.spec_from_file_location("lift_teleop", repo / "scripts/rambo/teleop_loco_manip.py")
        teleop = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(teleop)
        import torch
        import gymnasium as gym
        from crl2.algorithms import PPO
        from rambo.rl import Crl2VecEnvWrapper
        from rambo.utils.registry import parse_env_cfg, load_cfg_from_registry
        from rambo.utils.physx import configure_physx, assert_physx_environment
        from rambo.validation.checkpoints import contract_for_task, load_verified_checkpoint, restore_runner
        from rambo.validation.rollout import seed_everything
        rt = dict(torch=torch, gym=gym, PPO=PPO, Crl2VecEnvWrapper=Crl2VecEnvWrapper,
                  parse_env_cfg=parse_env_cfg, load_cfg_from_registry=load_cfg_from_registry,
                  configure_physx=configure_physx, assert_physx_environment=assert_physx_environment,
                  contract_for_task=contract_for_task, load_verified_checkpoint=load_verified_checkpoint,
                  restore_runner=restore_runner)
        torch = rt["torch"]
        task = teleop.LIFT_BASKET_TASK_ID
        cfg = rt["parse_env_cfg"](task, device="cuda:0", num_envs=1, use_fabric=True)
        teleop._configure_environment(cfg, argparse.Namespace(task=task, seed=42, episode_length_s=32.,
                                                             view="third-person", camera_setup=rig.SETUP_ID))
        rt["configure_physx"](cfg)
        cfg.sim.render_interval = 40
        cfg.viewer.eye = (1.65, 1.65, 1.15)
        cfg.viewer.lookat = (.25, .0, .40)
        checkpoint_path = workspace / "checkpoints/rambo/go2/quadruped/model_2000.pt"
        contract = rt["contract_for_task"](task)
        checkpoint = rt["load_verified_checkpoint"](checkpoint_path, contract)
        agent = rt["load_cfg_from_registry"](task, "crl2_cfg_entry_point")
        agent["general"]["num_envs"] = 1
        agent["seed"] = 42
        env = rt["Crl2VecEnvWrapper"](rt["gym"].make(task, cfg=cfg))
        be = env.unwrapped
        seed_everything(42, env)
        obs, _ = env.reset()
        report.update({"setup_id": rig.SETUP_ID, "cameras": rig.CAMERAS,
                       "dimensions": {v: [c.get("width", 640), c.get("height", 480)] for v, c in rig.CAMERAS.items()}, "sensor_rate_hz": 12.5,
                       "physics": rt["assert_physx_environment"](env),
                       "basket_physics": teleop._validate_lift_basket_source_physics(),
                       "checkpoint_sha256": contract.sha256,
                       "basket_initial_position": list(cfg.primary_position),
                       "bracket": {"rigidly_parented": True, "collision": False, "mass_modeled": False}})
        stage = omni.usd.get_context().get_stage()
        sensors = {"ego": be.front_camera, "task": be.task_camera}
        scene_cameras = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Camera)
                         and str(p.GetPath()).startswith("/World/")]
        if sorted(scene_cameras) != sorted(c["path"] for c in rig.CAMERAS.values()):
            raise ValueError(f"Unexpected scene cameras: {scene_cameras}")
        for p in Usd.PrimRange(stage.GetPrimAtPath(rig.RIG_PATH)):
            if p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.MassAPI):
                raise ValueError(f"Visual camera rig has unintended physics: {p.GetPath()}")
        report["scene_camera_paths"] = scene_cameras
        params = {}
        for view, sensor in sensors.items():
            params[view] = rep.AnnotatorRegistry.get_annotator("CameraParams", device="cpu")
            params[view].attach(sensor._render_data.render_product_paths)
        viewport_rgb = None
        if mode == ["kit"]:
            from isaacsim.core.rendering_manager import ViewportManager
            if not ViewportManager.wait_for_viewport()[0]:
                raise RuntimeError("No Kit viewport")
            for view in ("ego", "task", "third-person"):
                teleop._select_mounted_view(view)
            viewport = ViewportManager.get_viewport_api()
            viewport_rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            viewport_rgb.attach([viewport.render_product_path])
            report["viewport_switches_verified"] = ["ego", "task", "third-person"]
        runner = rt["PPO"](task=task, env=env, agent_cfg=agent, train=False, device=env.device)
        rt["restore_runner"](runner, checkpoint, load_values=False, verify=True)
        policy = runner.get_inference_policy(device=be.device)
        robot = be.scene.articulations["robot"]
        first_root = None
        for step in range(1, args.steps + 1):
            with torch.inference_mode():
                cmd = np.array([0., 0., 0., .1934, .142, .05, 0., 0., 0.], dtype=np.float32)
                be.set_loco_manip_commands(base_velocity=torch.tensor(cmd[None, :3], device=be.device),
                                          fl_position=torch.tensor(cmd[None, 3:6], device=be.device),
                                          fl_force=torch.tensor(cmd[None, 6:], device=be.device))
                action = policy(obs)
                if not torch.isfinite(action).all():
                    raise ValueError("Nonfinite action")
                obs, reward, done, _ = env.step(action)
                if done.any() or not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
                    raise ValueError(f"Invalid/terminal state at step {step}")
            root = np.concatenate((robot.data.root_link_pos_w.torch[0].cpu().numpy(),
                                   robot.data.root_link_quat_w.torch[0].cpu().numpy()))
            if first_root is None:
                first_root = root.copy()
            if step % 8 == 0:
                row = {"step": step, "root_xyzw": root.tolist(), "views": {},
                       "basket": be.primary_pose_w[0].cpu().tolist(),
                       "success": bool(be.task_success[0].cpu())}
                for view, sensor in sensors.items():
                    rgb = sensor.data.output["rgb"].torch[0].cpu().numpy()
                    mount = rig.CAMERAS[view]
                    if rgb.dtype != np.uint8 or rgb.shape != (mount.get("height", 480), mount.get("width", 640), 3) or rgb.std() < 1:
                        raise ValueError(f"Invalid RGB for {view}")
                    intrinsic = sensor.data.intrinsic_matrices.torch[0].cpu().numpy()
                    if "horizontal_aperture_mm" in mount:
                        expected_focal_px = [
                            mount.get("width", 640) * mount["focal_length_mm"] / mount["horizontal_aperture_mm"],
                            mount.get("height", 480) * mount["focal_length_mm"] / mount["vertical_aperture_mm"],
                        ]
                        if not np.allclose([intrinsic[0, 0], intrinsic[1, 1]], expected_focal_px, atol=1e-3):
                            raise ValueError(f"Unexpected {view} intrinsics: {intrinsic}")
                    native = params[view].get_data()
                    matrix = np.asarray(native["cameraViewTransform"]).reshape(4, 4)
                    error = rig.validate_renderer_transform(root, view, matrix)
                    Image.fromarray(rgb).save(out / "frames" / f"{view}_{step:06d}.png")
                    row["views"][view] = {"rtx_view_matrix": matrix.tolist(), "error": error,
                                         "intrinsics": intrinsic.tolist(),
                                         "rgb_std": float(rgb.std())}
                rows.append(row)
                if viewport_rgb is not None and step in (8, 120, args.steps):
                    be.sim.render()
                    rgb = viewport_rgb.get_data()
                    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.std() < 1:
                        raise ValueError("Invalid inspection viewport RGB")
                    Image.fromarray(rgb[:, :, :3]).save(out / f"rig_overview_{step:06d}.png")
            if step % 100 == 0:
                print(f"CAMERA_PREVIEW step={step} success={bool(be.task_success[0])}", flush=True)
        report.update({"passed": True, "steps": args.steps, "frames_per_view": len(rows),
                       "elapsed_s": time.monotonic()-started, "terminal_count": 0,
                       "root_displacement_m": float(np.linalg.norm(root[:3]-first_root[:3])),
                       "success_seen": any(r["success"] for r in rows),
                       "maximum_pose_error": max(max(v["error"].values()) for r in rows for v in r["views"].values())})
        code = 0
    except BaseException:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        (out / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
        (out / "camera_frames.json").write_text(json.dumps(rows)+"\n")
        for source in (Path(__file__), repo / "scripts/rambo/teleop_loco_manip.py",
                       repo / "source/rambo/rambo/tasks/common/lift_camera_rig.py",
                       repo / "source/rambo/rambo/tasks/direct/rambo_quadruped/object_tasks_env.py"):
            shutil.copy2(source, out / source.name)
        (out / "rambo-diff.patch").write_bytes(subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--binary"]))
        (out / "rambo-status.txt").write_bytes(subprocess.check_output(["git", "-C", str(repo), "status", "--short"]))
        (out / "checksums.sha256").write_text("".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}\n"
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "checksums.sha256"))
        print("CAMERA_RESULT " + json.dumps(report), flush=True)
        if app is not None:
            app.app.close(exit_code=code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
