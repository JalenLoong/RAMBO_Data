"""Bounded DATA-004 terminal gate or exactly one approved technical pilot."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,traceback
from pathlib import Path
import numpy as np


def main():
    from isaaclab.app import AppLauncher
    from rambo.utils.physx import validate_rambo_visualizer_args
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ffmpeg',required=True)
    p.add_argument('--mode',choices=['terminal-gate','pilot'],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--terminal-gate',type=Path)
    p.add_argument('--max-actions',type=int,default=500)
    AppLauncher.add_app_launcher_args(p);args=p.parse_args();validate_rambo_visualizer_args(p,args,sys.argv[1:]);args.enable_cameras=True
    root=Path(os.environ['WORKSPACE_ROOT']);repo=Path(__file__).resolve().parents[2]
    profile=json.loads((repo/'configs/push_box_v2.json').read_text())
    if profile['approval']['status']!='approved':raise ValueError('Asset/task approval required')
    watched=[repo/'source/rambo/rambo/recording_v2.py',repo/'source/rambo/rambo/tasks/direct/rambo_quadruped/qp_env.py',repo/'source/rambo/rambo/tasks/direct/rambo_quadruped/push_box_v2.py',repo/'configs/push_box_v2.json',Path(__file__),repo/'source/rambo/rambo/tasks/common/push_box_geometry.py',repo/'source/rambo/rambo/tasks/common/push_box_expert.py']
    implementation={str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in watched}
    if args.mode=='pilot':
        gate=json.loads(args.terminal_gate.read_text()) if args.terminal_gate else {}
        if not gate.get('passed') or gate.get('implementation')!=implementation:raise ValueError('Current implementation must pass actual terminal-before-reset gate')
    if not 1<=args.max_actions<=1000:raise ValueError('Bounded <=20s only')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    out=args.output_dir.resolve();report=dict(work_id='DATA-004',mode=args.mode,passed=False,implementation=implementation,profile=profile,dataset_episode=args.mode=='pilot')
    app=rec=None;code=1
    try:
        app=AppLauncher(args)
        import torch,omni.replicator.core as rep
        from PIL import Image
        from crl2.algorithms import PPO
        from rambo.rl import Crl2VecEnvWrapper
        from rambo.utils.registry import parse_env_cfg,load_cfg_from_registry
        from rambo.utils.physx import configure_physx,assert_physx_environment
        from rambo.validation.checkpoints import contract_for_task,load_verified_checkpoint,restore_runner
        from rambo.tasks.direct.rambo_quadruped.push_box_v2 import PushBoxV2Env,asset_geometry
        from rambo.recording_v2 import Recorder,json_write
        from rambo.tasks.common.push_box_geometry import source_geometry,world_geometry
        from rambo.tasks.common.push_box_expert import command as expert_command
        from rambo.contracts_v2.runtime import prepare_command
        spec=importlib.util.spec_from_file_location('pilot_teleop',repo/'scripts/rambo/teleop_loco_manip.py');teleop=importlib.util.module_from_spec(spec);spec.loader.exec_module(teleop)
        task='Isaac-RAMBO-Quadruped-Push-Box-V2-Go2-v0'
        cfg=parse_env_cfg(task,device='cuda:0',num_envs=1,use_fabric=True)
        teleop._configure_environment(cfg,argparse.Namespace(task=teleop.LIFT_BASKET_TASK_ID,seed=42,episode_length_s=24.,view='third-person',camera_setup='robot-dual-v3'))
        configure_physx(cfg);cfg.sim.render_interval=10
        cfg.front_camera.update_period=.02;cfg.task_camera.update_period=.02
        cfg.terminate_on_body_contact=False;cfg.terminate_on_limb_contact=False;cfg.terminate_on_undesired_foot_contact=False
        asset=root/profile['asset_path'];assert hashlib.sha256(asset.read_bytes()).hexdigest()==profile['asset_sha256']
        cfg.approved_asset_path=str(asset);cfg.approved_profile=profile
        low,high=asset_geometry(asset);center=(low+high)/2
        cfg.primary_orientation=(0.,0.,0.,1.)
        geometry=source_geometry(low,high)
        # Native RAMBO's projected-COM regulator assumes reset XY=(0,0).
        # Keep that controller reference and express the relative geometry by
        # placing the box, rather than translating the robot's spawn.
        face_width=geometry['dimensions'][1]
        center_xy=(profile['initial_face_distance_x']+geometry['dimensions'][0]/2,float(cfg.ee_default_command[1]))
        cfg.primary_position=(center_xy[0]-float(center[0]),center_xy[1]-float(center[1]),profile['initial_floor_clearance']-float(low[2]))
        initial_geometry=world_geometry(geometry,[*cfg.primary_position,*cfg.primary_orientation])
        report['initial_geometry']=dict(source=geometry,world=initial_geometry,box_pose=[*cfg.primary_position,*cfg.primary_orientation],robot_config_position=list(cfg.robot.init_state.pos))
        cfg.pilot_max_physics_steps=(32 if args.mode=='terminal-gate' else args.max_actions)*10
        env=Crl2VecEnvWrapper(PushBoxV2Env(cfg));be=env.unwrapped
        observer=rep.create.camera(position=profile['observer']['position'],look_at=profile['observer']['look_at'],focal_length=24)
        rp=rep.create.render_product(observer,(1280,720));rgb=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu');rgb.attach(rp)
        rec=Recorder(be,out,args.ffmpeg,rgb);be._v2_recorder=rec
        # Track reset entry/exit around the unmodified existing reset implementation.
        reset=be._reset_idx;events=[]
        def observed_reset(ids):
            if rec.active:raise RuntimeError('Reset attempted before terminal capture was sealed')
            if rec.terminal is not None:events.append('reset_enter_after_terminal_copy')
            value=reset(ids)
            if rec.terminal is not None:events.append('reset_exit')
            return value
        be._reset_idx=observed_reset
        obs,_=env.reset()
        for _ in range(4):be.sim.render()
        contract=contract_for_task(teleop.LIFT_BASKET_TASK_ID)
        checkpoint=load_verified_checkpoint(root/'checkpoints/rambo/go2/quadruped/model_2000.pt',contract)
        agent=load_cfg_from_registry(task,'crl2_cfg_entry_point');agent['general']['num_envs']=1;agent['seed']=42
        runner=PPO(task=task,env=env,agent_cfg=agent,train=False,device=be.device);restore_runner(runner,checkpoint,load_values=False,verify=True);policy=runner.get_inference_policy(device=be.device)
        rec.begin();initial=rec.boundaries[0]
        for k in range(cfg.pilot_max_physics_steps//10):
            time=rec.ns/1e9
            request,expert=expert_command(be,time)
            prepared=prepare_command(request,f'{args.mode}:{k}',rec.tick);prepared['extensions']['expert']=expert;rec.submit(prepared)
            cmd=torch.tensor([prepared['filtered']],device=be.device,dtype=torch.float32)
            be.set_loco_manip_commands(cmd[:,:3],cmd[:,3:6],cmd[:,6:])
            for _ in range(2):
                with torch.inference_mode():
                    action=policy(obs)
                    if not torch.isfinite(action).all():raise ValueError('Nonfinite residual')
                    obs,_,done,_=env.step(action)
                if done.any():break
            if rec.terminal is not None:break
            rec.capture_boundary()
            if k%50==0:print(f'PILOT k={k} t={time:.2f} center={be.geometric_center[0].cpu().tolist()}',flush=True)
        if rec.terminal is None:raise ValueError('Missing terminal hook')
        if rec.terminal['partial_interval']:raise ValueError('Partial final action preserved; invalid episode')
        for _ in range(4):be.sim.render()
        after=rec.state();changes={}
        for role,sensor in [('ego',be.front_camera),('task_centric',be.task_camera)]:
            v=rec.arr(sensor.data.output['rgb'].torch[0]);Image.fromarray(v).save(out/'review'/f'post_reset_{role}.png')
            changes[role]=dict(pre_reset_sha256=hashlib.sha256(rec.terminal_rgb[role].tobytes()).hexdigest(),post_reset_sha256=hashlib.sha256(v.tobytes()).hexdigest(),mean_absolute_pixel_difference=float(np.abs(v.astype(float)-rec.terminal_rgb[role].astype(float)).mean()))
            assert changes[role]['pre_reset_sha256']==rec.terminal['rgb_hashes'][role]
        state_delta=float(np.max(np.abs(np.asarray(after['observation.state.base.position'])-np.asarray(rec.terminal['state']['observation.state.base.position']))))
        assert events==['reset_enter_after_terminal_copy','reset_exit']
        assert state_delta>1e-5 and all(v['mean_absolute_pixel_difference']>0 for v in changes.values())
        assert len(rec.boundaries)==len(rec.commands)+1
        for role in ['ego','task_centric']:
            ids=[x['sensor_frame_ids'][role] for x in rec.boundaries]
            assert all(b>a for a,b in zip(ids,ids[1:])),(role,ids)
        status='success' if rec.terminal['state']['task.success'][0] else 'failure'
        capture=rec.close(dict(mode=args.mode,diagnostic_only=True,profile=profile,implementation=implementation,checkpoint_sha256=contract.sha256,source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),status=status,task_text='Push the box into the target area.',initial_geometry=report['initial_geometry'],post_reset_state=after))
        report.update(passed=True,terminal_before_reset=True,terminal_simulation_time_ns=rec.terminal['simulation_time_ns'],n_actions=len(rec.commands),n_boundaries=len(rec.boundaries),reset_events=events,terminal_vs_reset=changes,state_difference_m=state_delta,status=status,physics=assert_physx_environment(env),contact_provenance='unknown_diagnostic_only')
        code=0
    except BaseException:
        report['error']=traceback.format_exc();print(report['error'],flush=True)
        if rec is not None:
            try:rec.close(dict(invalid=True,mode=args.mode,profile=profile))
            except Exception:pass
    finally:
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('PILOT_RESULT '+json.dumps(report),flush=True)
        if app is not None:app.app.close(exit_code=code)
    return code
if __name__=='__main__':raise SystemExit(main())
