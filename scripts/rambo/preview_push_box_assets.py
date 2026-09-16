"""DATA-004 candidate review only: real PhysX settling and static coverage poses.

No Push Box task registration, success detector, recorder or dataset episode.
Retains the Lift harness only to reuse the verified robot/camera/controller setup.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


def main():
    from isaaclab.app import AppLauncher
    from rambo.utils.physx import validate_rambo_visualizer_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    validate_rambo_visualizer_args(parser, args, sys.argv[1:])
    args.enable_cameras = True
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    asset_path = args.asset.resolve()
    report = dict(work_id='DATA-004', passed=False, dataset_episode=False,
                  approval='pending', task_success='not_evaluated', contact='unknown',
                  terminal_before_reset='not_run', source=str(asset_path),
                  source_sha256=hashlib.sha256(asset_path.read_bytes()).hexdigest(),
                  static_poses_are_not_executed_trajectories=True)
    app = None
    code = 1
    try:
        app = AppLauncher(args)
        import torch
        import omni.usd
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        import isaaclab.sim as sim
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from crl2.algorithms import PPO
        from rambo.rl import Crl2VecEnvWrapper
        from rambo.tasks.common import lift_camera_rig as rig
        from rambo.tasks.direct.rambo_quadruped.object_tasks_env import ObjectTaskQPEnv
        from rambo.utils.registry import parse_env_cfg, load_cfg_from_registry
        from rambo.utils.physx import configure_physx, assert_physx_environment
        from rambo.validation.checkpoints import contract_for_task, load_verified_checkpoint, restore_runner

        repo = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location('preview_teleop', repo/'scripts/rambo/teleop_loco_manip.py')
        teleop = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(teleop)
        source = Usd.Stage.Open(str(asset_path))
        prim = source.GetDefaultPrim()
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy']).ComputeLocalBound(prim).ComputeAlignedBox()
        authored_bounds=[list(bbox.GetMin()),list(bbox.GetMax())]
        # Scanned assets can carry stale authored extents. Placement must use
        # actual vertices, including source child transforms, not those hints.
        transforms=UsdGeom.XformCache()
        vertices=[]
        for mesh_prim in source.Traverse():
            if mesh_prim.IsA(UsdGeom.Mesh):
                relative=transforms.ComputeRelativeTransform(mesh_prim,prim)[0]
                vertices.extend(list(relative.Transform(Gf.Vec3d(*point)))
                                for point in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get())
        points=np.asarray(vertices,dtype=np.float64)
        low, high = points.min(axis=0),points.max(axis=0)
        assert UsdGeom.GetStageUpAxis(source) == 'Z' and UsdGeom.GetStageMetersPerUnit(source) == 1
        assert np.isfinite(low).all() and np.isfinite(high).all()
        center = (low+high)/2
        position = (.78-center[0], .15-center[1], .02-low[2])

        class CandidatePreview(ObjectTaskQPEnv):
            def _spawn_lift_basket(self):
                path = self._root+'/candidate'
                spawn = sim.UsdFileCfg(usd_path=str(asset_path))
                spawn.func(path, spawn, translation=tuple(self.cfg.primary_position), orientation=(0.,0.,0.,1.))
                self._primary = RigidObject(RigidObjectCfg(prim_path=path,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(self.cfg.primary_position), rot=(0.,0.,0.,1.))))
                self.scene.rigid_objects['candidate'] = self._primary

            def _update_task_state(self):
                pass  # Asset review has no success/contact attribution.

        task = teleop.LIFT_BASKET_TASK_ID
        cfg = parse_env_cfg(task, device='cuda:0', num_envs=1, use_fabric=True)
        teleop._configure_environment(cfg, argparse.Namespace(task=task,seed=42,episode_length_s=32.,
            view='third-person',camera_setup=rig.SETUP_ID))
        configure_physx(cfg)
        cfg.primary_position = tuple(float(x) for x in position)
        cfg.sim.render_interval = 40
        env = Crl2VecEnvWrapper(CandidatePreview(cfg))
        be = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        # Review-only goal outline, no physical collision surface.
        for i,(pos,size) in enumerate([
            ((.95,.15,.002),(.008,.5,.003)),((1.35,.15,.002),(.008,.5,.003)),
            ((1.15,-.10,.002),(.4,.008,.003)),((1.15,.40,.002),(.4,.008,.003))]):
            marker=sim.CuboidCfg(size=size,visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.05,.8,.15)))
            marker.func('/World/ReviewGoal_'+str(i),marker,translation=pos)
        observer = rep.create.camera(position=(1.8,2.0,1.5),look_at=(.65,.10,.2),focal_length=24)
        product = rep.create.render_product(observer,(1280,720))
        annotator = rep.AnnotatorRegistry.get_annotator('rgb',device='cpu')
        annotator.attach(product)
        obs,_ = env.reset()
        contract = contract_for_task(task)
        checkpoint = load_verified_checkpoint(Path(os.environ['RAMBO_CHECKPOINT_ROOT'])/'quadruped/model_2000.pt',contract)
        agent=load_cfg_from_registry(task,'crl2_cfg_entry_point');agent['general']['num_envs']=1;agent['seed']=42
        runner=PPO(task=task,env=env,agent_cfg=agent,train=False,device=env.device)
        restore_runner(runner,checkpoint,load_values=False,verify=True)
        policy=runner.get_inference_policy(device=be.device)
        robot=be.scene.articulations['robot']
        trace=[]
        for step in range(320):
            with torch.inference_mode():
                reach=max(0.,min(1.,(step-210)/90))
                command=torch.tensor([[0,0,0,.1934+reach*.1866,.142+reach*.018,.05+reach*.07,0,0,0]],device=be.device,dtype=torch.float32)
                be.set_loco_manip_commands(command[:,:3],command[:,3:6],command[:,6:])
                obs,reward,done,_=env.step(policy(obs))
                if done.any() or not torch.isfinite(obs).all():
                    raise ValueError(f'Preview robot terminated/nonfinite at {step}')
                trace.append(dict(control_step=step+1,command=command[0].cpu().tolist(),
                                  manipulator_ready=bool(be.manipulator_ready[0]),
                                  fl_foot_world=robot.data.body_link_pos_w.torch[0,be.feet_ids[0]].cpu().tolist(),
                                  object_pose=be.primary_pose_w[0].cpu().tolist(),
                                  object_velocity=be.primary_velocity_w[0].cpu().tolist()))
        stable_pose=be.primary_pose_w.clone()
        robot_pose=torch.cat((robot.data.root_link_pos_w.torch,robot.data.root_link_quat_w.torch),dim=-1).clone()
        report.update(physics=assert_physx_environment(env),control_steps=320,physics_steps=1600,
            checkpoint_sha256=contract.sha256,source_extent_m=(high-low).tolist(),source_local_bounds=[low.tolist(),high.tolist()],
            source_extent_method='actual mesh vertices relative to rigid root, including child transforms',
            authored_cached_bounds=authored_bounds,
            source_mass_kg=UsdPhysics.MassAPI(prim).GetMassAttr().Get(),placement_root=list(position),
            camera_geometry=rig.CAMERAS,preview_policy_camera_hz=12.5,
            observer=dict(role='monitor_only',position=[1.8,2.,1.5],look_at=[.65,.1,.2],cadence='static_review_only'),
            goal_proposal=dict(x=[.95,1.35],y=[-.10,.40],membership='object_center_xy',hold_seconds=0,approval='pending'),
            settling_final_pose=stable_pose[0].cpu().tolist(),settling_final_velocity=trace[-1]['object_velocity'])
        ids=torch.tensor([0],device=be.device,dtype=torch.int32)
        camera_params={}
        for view,sensor in [('ego',be.front_camera),('task',be.task_camera)]:
            native=rep.AnnotatorRegistry.get_annotator('CameraParams',device='cpu')
            native.attach(sensor._render_data.render_product_paths)
            camera_params[view]=native
        report['coverage_poses']=[]
        for label,robot_dx,box_dx in [('approach',0.,0.),('near',.25,0.),('goal',.62,.37)]:
            rp=robot_pose.clone();rp[:,0]+=robot_dx
            bp=stable_pose.clone();bp[:,0]+=box_dx
            robot.write_root_pose_to_sim_index(root_pose=rp,env_ids=ids)
            be._primary.write_root_pose_to_sim_index(root_pose=bp,env_ids=ids)
            be.scene.update(dt=0.)
            for _ in range(12):be.sim.render()
            images={}
            camera_checks={}
            for view,sensor in [('ego',be.front_camera),('task',be.task_camera)]:
                sensor.update(dt=.08,force_recompute=True)
                value=sensor.data.output['rgb'].torch[0].cpu().numpy().copy()
                if value.shape!=(720,1280,3) or value.std()<1:raise ValueError(f'Invalid {view} render')
                images[view]=value
                native=camera_params[view].get_data()
                actual_root=torch.cat((robot.data.root_link_pos_w.torch,robot.data.root_link_quat_w.torch),dim=-1)[0].cpu().numpy()
                camera_checks[view]=rig.validate_renderer_transform(actual_root,view,native['cameraViewTransform'])
                intrinsic=sensor.data.intrinsic_matrices.torch[0].cpu().numpy()
                mount=rig.CAMERAS[view]
                expected=[1280*mount['focal_length_mm']/mount['horizontal_aperture_mm'],720*mount['focal_length_mm']/mount['vertical_aperture_mm']]
                if not np.allclose([intrinsic[0,0],intrinsic[1,1]],expected,atol=1e-3):raise ValueError('Camera intrinsics changed')
                camera_checks[view]['intrinsics']=intrinsic.tolist()
            value=annotator.get_data()
            if not isinstance(value,np.ndarray) or value.ndim!=3:raise ValueError('Missing observer RGB')
            images['observer']=value[:,:,:3].copy()
            for view,value in images.items():Image.fromarray(value).save(out/f'{label}_{view}.png')
            report['coverage_poses'].append(dict(label=label,robot_pose=rp[0].cpu().tolist(),object_pose=bp[0].cpu().tolist(),kind='static_repositioned_review',camera_checks=camera_checks))
        (out/'settling-trace.json').write_text(json.dumps(trace,indent=2)+'\n')
        report['passed']=True;code=0
    except BaseException:
        report['error']=traceback.format_exc();print(report['error'],flush=True)
    finally:
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PUSH_BOX_REVIEW_RESULT '+json.dumps(report),flush=True)
        if app is not None:app.app.close(exit_code=code)
    return code


if __name__=='__main__':
    raise SystemExit(main())
