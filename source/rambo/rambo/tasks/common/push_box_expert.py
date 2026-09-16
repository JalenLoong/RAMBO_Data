"""Geometry-aware scripted demonstration references through unchanged native9."""
import numpy as np
from .push_box_corner_recovery import corner_reference


def command(env,time):
    import torch
    from isaaclab.utils.math import quat_apply_inverse,quat_mul,quat_inv
    geom=env.geometry_world();face=np.asarray(geom['face_center']);normal=np.asarray(geom['face_normal']);center=np.asarray(geom['center'])
    vy=0.
    expert=env.cfg.approved_profile['expert'];base=env.base_pos_w[0].detach().cpu().numpy()
    projected=quat_mul(env.base_quat,quat_inv(env.base_quat_rp))
    # Intersection of the intended face plane with a +X ray through the box
    # center: near the face center, with the nominal push line through the box
    # center rather than a rotating face-normal offset that amplifies yaw.
    line_point=face.copy();line_point[1]=center[1]
    if 'episode_scenario' in env.cfg.approved_profile:
        from .push_box_geometry import forward_face_ray
        line_point,normal=forward_face_ray(env._box_geometry,env.primary_pose_w[0].detach().cpu().numpy())
    else:
        line_point[0]=face[0]-normal[1]*(line_point[1]-face[1])/normal[0]
    if time<expert['settle_seconds']:
        result=[0.,0.,0.,.1934,.142,.05,0.,0.,0.];target=None;phase='settle';vx=0.
    else:
        distance=line_point[0]-base[0]
        if not hasattr(env,'_expert_stage'):env._expert_stage='base_approach'
        if env._expert_stage=='base_approach' and distance<=expert['face_distance_for_foot_approach']:
            env._expert_stage='normal_approach';env._expert_contact_start=time
        if env._expert_stage=='normal_approach' and time-env._expert_contact_start>=expert['normal_approach_seconds']:
            env._expert_stage='straight_push';env._expert_push_start=time
        phase=env._expert_stage
        if phase=='base_approach':
            u=min(1.,(time-expert['settle_seconds'])/.8);u=u*u*(3-2*u)
            neutral=base+np.array([.1934,.142,.05-base[2]])
            target=neutral*(1-u)+np.array([base[0]+expert['approach_foot_x'],line_point[1],line_point[2]])*u
            vx=expert['align_base_speed']
        elif phase=='normal_approach':
            u=(time-env._expert_contact_start)/expert['normal_approach_seconds']
            target=line_point+normal*(expert['standoff_m']*(1-u)-expert['penetration_m']*u)
            vx=0.
        else:
            # The body advances the foot. Never chase the box beyond the
            # existing controller's demonstrated manipulation workspace.
            foot_x=expert['push_foot_x'];vx=expert['push_base_speed']
            if 'motion' in expert:
                duration=expert['transition_s'];u=np.clip((time-env._expert_push_start)/duration,0.,1.);u=u*u*(3-2*u)
                foot_x=expert['approach_foot_x']*(1-u)+foot_x*u;vx*=u
                remaining=env.cfg.approved_profile['goal_x'][0]-min(x[0] for x in geom['footprint_xy'])
                lateral=all(env.cfg.approved_profile['goal_y'][0]<=x[1]<=env.cfg.approved_profile['goal_y'][1] for x in geom['footprint_xy'])
                if expert['motion']=='leg_finish' and not hasattr(env,'_expert_finish_start') and u>=1 and lateral and 0<remaining<=.03 and expert['leg_finish_max_x']-foot_x>=remaining+.01:
                    env._expert_finish_start=time;env._expert_finish_base=float(base[0]);env._expert_finish_foot=foot_x
                if hasattr(env,'_expert_finish_start'):
                    v=np.clip((time-env._expert_finish_start)/duration,0.,1.);v=v*v*(3-2*v)
                    foot_x=env._expert_finish_foot+(expert['leg_finish_max_x']-env._expert_finish_foot)*v
                    vx=expert['push_base_speed']*(1-v);phase='leg_finish'
            target=np.array([base[0]+foot_x,line_point[1],line_point[2]])
            if 'lateral_tracking_gain' in expert:
                # Bounded high-level position compensation keeps the actual
                # foot near the center ray despite the retained controller's
                # tracking bias. This does not alter the controller or forces.
                error=line_point[1]-env.actual_fl_world()[1]
                limit=expert['lateral_tracking_limit_m']
                env._expert_lateral_comp=float(np.clip(getattr(env,'_expert_lateral_comp',0.)+.02*expert['lateral_tracking_gain']*error,-limit,limit))
                target[1]+=env._expert_lateral_comp
        if phase in ('straight_push','leg_finish') or hasattr(env,'_corner_recovery'):
            target,vx,vy,recovery_phase=corner_reference(env,time,geom,base,target,vx)
            if recovery_phase is not None:phase=recovery_phase
        origin=env.base_pos_w[0].clone();origin[2]=0
        point=torch.as_tensor(target,device=env.device,dtype=torch.float32)[None]-origin[None]
        local=quat_apply_inverse(projected,point)[0]
        velocity=quat_apply_inverse(projected,torch.tensor([[vx,vy,0.]],device=env.device,dtype=torch.float32))[0]
        result=[float(velocity[0]),float(velocity[1]),0.,*local.cpu().tolist(),0.,0.,0.]
    diag=dict(phase=phase,intended_face_center=face.tolist(),intended_face_normal=normal.tolist(),
              commanded_contact_point=None if target is None else target.tolist(),centerline_face_intersection=line_point.tolist(),
              commanded_world_base_velocity=[vx,vy,0.],actual_fl_eef_world=env.actual_fl_world().tolist(),
              box_pose=env.primary_pose_w[0].detach().cpu().tolist(),box_yaw_rad=geom['yaw_rad'],contact_evidence='unknown_diagnostic_only',lateral_tracking_compensation_m=getattr(env,'_expert_lateral_comp',0.))
    env._expert_diagnostics=diag
    return result,diag
