#!/usr/bin/env python3
"""Launch passive Dolly + released biped Go2 with native keyboard teleoperation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from rambo.biped_teleop import BipedKeyboardState,HELP

TASK="Isaac-RAMBO-Biped-Push-Dolly-Go2-v0"


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    workspace=Path(os.environ.get("WORKSPACE_ROOT",Path(__file__).resolve().parents[4]))
    p.add_argument("--checkpoint",type=Path,default=workspace/"checkpoints/rambo/go2/biped/model_4000.pt")
    p.add_argument("--max-steps",type=int,default=0,help="0: interactive until window closes")
    p.add_argument("--output-dir",type=Path,help="Fresh optional smoke evidence directory")
    p.add_argument("--smoke-walk",action="store_true",help="Explicit synthetic command smoke, not keyboard evidence")
    AppLauncher.add_app_launcher_args(p)
    args=p.parse_args()
    from rambo.utils.physx import validate_rambo_visualizer_args
    mode=validate_rambo_visualizer_args(p,args,sys.argv[1:])
    if args.max_steps<0:p.error("--max-steps must be nonnegative")
    if mode==["none"] and not args.max_steps:p.error("--viz none requires finite --max-steps")
    args.enable_cameras=True;args.fast_shutdown=True;args.livestream=0
    output=args.output_dir
    if output:output.mkdir(parents=True,exist_ok=False)
    app=env=subscription=None;code=1;rows=[];summary={"task":TASK,"passed":False}
    started=time.monotonic()
    try:
        app=AppLauncher(args)
        from rambo.torch_runtime import ensure_cuda_linalg_loaded
        ensure_cuda_linalg_loaded()
        import torch,rambo,gymnasium as gym
        from crl2.algorithms import PPO
        from rambo.rl import Crl2VecEnvWrapper
        from rambo.utils.registry import parse_env_cfg,load_cfg_from_registry
        from rambo.utils.physx import configure_physx,assert_physx_environment
        from rambo.validation.rollout import configure_validation_cfg,seed_everything
        from rambo.validation.checkpoints import contract_for_task,load_verified_checkpoint,restore_runner
        from rambo.tasks.direct.rambo_biped.push_dolly_env import asset_path
        from isaaclab.sensors import Camera,CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg
        import isaaclab.sim as sim
        rambo.register_tasks()
        cfg=parse_env_cfg(TASK,device="cuda:0",num_envs=1,use_fabric=True)
        configure_validation_cfg(cfg,duration_s=max(300.,args.max_steps*.01+1),enable_rgb_camera=True,contact_phase_offset_s=19.6)
        configure_physx(cfg);cfg.seed=42;cfg.sim.render_interval=40
        cfg.viewer.eye=(2.4,2.8,1.65);cfg.viewer.lookat=(.65,0.,.38)
        contract=contract_for_task("Isaac-RAMBO-Biped-Go2-v0")
        checkpoint=load_verified_checkpoint(args.checkpoint,contract)
        agent=load_cfg_from_registry(TASK,"crl2_cfg_entry_point");agent["general"]["num_envs"]=1;agent["seed"]=42
        env=Crl2VecEnvWrapper(gym.make(TASK,cfg=cfg));be=env.unwrapped
        be._record_termination_diagnostics=True
        if output:
            overview=Camera(CameraCfg(prim_path="/World/DollyOverview",update_period=0.,height=480,width=640,
                data_types=["rgb"],renderer_cfg=IsaacRtxRendererCfg(),
                spawn=sim.PinholeCameraCfg(focal_length=18.,horizontal_aperture=20.955)))
            be.sim.reset()
        seed_everything(42,env);obs,_=env.reset()
        if output:
            overview.set_world_poses_from_view(eyes=torch.tensor([cfg.viewer.eye],device=be.device),
                targets=torch.tensor([cfg.viewer.lookat],device=be.device))
        runner=PPO(task=TASK,env=env,agent_cfg=agent,train=False,device=env.device)
        restore_runner(runner,checkpoint,load_values=False,verify=True)
        policy=runner.get_inference_policy(device=be.device)
        controls=BipedKeyboardState();keyboard_events=[]
        ui_label=None
        if mode==["kit"]:
            import carb.input,omni.appwindow,omni.ui as ui
            interface=carb.input.acquire_input_interface()
            keyboard=omni.appwindow.get_default_app_window().get_keyboard()
            def callback(event,*_):
                key=event.input.name
                if event.type in (carb.input.KeyboardEventType.KEY_PRESS,carb.input.KeyboardEventType.KEY_REPEAT):
                    controls.event(key,True)
                    keyboard_events.append({"key":key,"pressed":True})
                elif event.type==carb.input.KeyboardEventType.KEY_RELEASE:
                    controls.event(key,False)
                    keyboard_events.append({"key":key,"pressed":False})
                return True
            subscription=interface.subscribe_to_keyboard_events(keyboard,callback)
            panel=ui.Window("RAMBO Biped - Push Dolly",width=390,height=285)
            with panel.frame:
                with ui.VStack():
                    ui.Label(HELP,word_wrap=True)
                    ui_label=ui.Label("Both front feet selected",word_wrap=True)
        print(HELP,flush=True)
        summary.update({"physics":assert_physx_environment(env),"checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "dolly_asset":str(asset_path()),"dolly_sha256":hashlib.sha256(asset_path().read_bytes()).hexdigest(),
            "dolly_joints":list(be.dolly.data.joint_names),"dolly_bodies":list(be.dolly.data.body_names),
            "visualizer":mode,"synthetic_smoke":args.smoke_walk,"physical_keyboard_verified":False})
        if len(be.dolly.data.joint_names)!=8:raise ValueError("Expected eight passive dolly joints")
        step=0;resets=0
        while app.app.is_running() and not controls.quit_requested and (not args.max_steps or step<args.max_steps):
            tick=time.monotonic()
            if not be.sim.is_playing():
                app.app.update();time.sleep(.01);continue
            if controls.reset_requested:
                controls.reset()
                be.set_biped_commands(torch.tensor(controls.advance()[None],device=be.device))
                obs,_=env.reset();resets+=1
            if args.smoke_walk:
                controls.event("UP",step>=100 and step<350)
            cmd=controls.advance()
            with torch.inference_mode():
                be.set_biped_commands(torch.tensor(cmd[None],device=be.device))
                action=policy(obs)
                if not torch.isfinite(action).all():raise RuntimeError("Nonfinite action")
                obs,_,done,_=env.step(action)
            step+=1
            if not torch.isfinite(obs).all():raise RuntimeError("Nonfinite observation")
            root=be.scene.articulations["robot"].data.root_link_pos_w.torch[0].cpu().numpy().copy()
            dolly=be.dolly.data.root_link_pos_w.torch[0].cpu().numpy().copy()
            joints=be.dolly.data.joint_pos.torch[0].cpu().numpy().copy()
            if not np.isfinite(dolly).all() or not np.isfinite(joints).all():raise RuntimeError("Nonfinite dolly")
            if output:rows.append({"step":step,"command":cmd.tolist(),"robot":root.tolist(),"dolly":dolly.tolist(),"dolly_joints":joints.tolist(),"done":bool(done.any())})
            if done.any():
                controls.reset();resets+=1
                print("[TELEOP] Reset after robot termination or episode timeout",flush=True)
                if args.max_steps:raise RuntimeError("Robot terminated in bounded smoke")
                be.set_biped_commands(torch.tensor(controls.advance()[None],device=be.device))
                obs,_=env.reset()
            if output and (step==8 or step==args.max_steps):
                from PIL import Image
                be.sim.render();overview.update(dt=.002,force_recompute=True)
                rgb=overview.data.output["rgb"].torch[0].cpu().numpy()
                if rgb.shape!=(480,640,3) or float(rgb.std())<1:raise RuntimeError("Invalid overview RGB")
                Image.fromarray(rgb).save(output/f"overview_{step:06d}.png")
            if step%100==0:
                print(f"DOLLY step={step} robot={root.round(3)} dolly={dolly.round(3)}",flush=True)
            if ui_label is not None and step%10==0:
                selected={(0,):"Left front foot",(1,):"Right front foot",(0,1):"Both front feet"}[controls.selected]
                ui_label.text=f"Selected: {selected}\nFL {controls.feet[0].round(3)}\nFR {controls.feet[1].round(3)}\nBase {cmd[:3].round(3)}"
            if mode==["kit"]:time.sleep(max(0.,.01-(time.monotonic()-tick)))
        summary.update({"passed":True,"steps":step,"resets":resets,"keyboard_events_observed":len(keyboard_events),
            "elapsed_s":time.monotonic()-started,"task_success_evaluated":False})
        code=0
    except BaseException:
        summary["error"]=traceback.format_exc();print(summary["error"],flush=True)
    finally:
        if output:
            (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
            (output/"trajectory.json").write_text(json.dumps(rows)+"\n")
            (output/"process_exit.json").write_text(json.dumps({"exit_code":code})+"\n")
            repo=Path(__file__).resolve().parents[2]
            (output/"git-status.txt").write_bytes(subprocess.check_output(["git","-C",str(repo),"status","--short"]))
            (output/"git-diff.patch").write_bytes(subprocess.check_output(["git","-C",str(repo),"diff","HEAD","--binary"]))
            for f in [Path(__file__),repo/"source/rambo/rambo/biped_teleop.py",repo/"source/rambo/rambo/tasks/direct/rambo_biped/push_dolly_env.py"]:
                shutil.copy2(f,output/f.name)
            files=sorted(p for p in output.iterdir() if p.is_file())
            (output/"checksums.sha256").write_text("".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n" for f in files))
        print("DOLLY_RESULT "+json.dumps(summary),flush=True)
        if subscription is not None:interface.unsubscribe_to_keyboard_events(keyboard,subscription)
        if app is not None:app.app.close(exit_code=code)
    return code


if __name__=="__main__":raise SystemExit(main())
