"""Geometry-aware scripted demonstration references through unchanged native9."""
import numpy as np


def command(env,time):
    import torch
    from isaaclab.utils.math import quat_apply_inverse,quat_mul,quat_inv
    geom=env.geometry_world();face=np.asarray(geom['face_center']);normal=np.asarray(geom['face_normal']);center=np.asarray(geom['center'])
    expert=env.cfg.approved_profile['expert'];base=env.base_pos_w[0].detach().cpu().numpy()
    projected=quat_mul(env.base_quat,quat_inv(env.base_quat_rp))
    # Intersection of the intended face plane with a +X ray through the box
    # center: near the face center, with the nominal push line through the box
    # center rather than a rotating face-normal offset that amplifies yaw.
    line_point=face.copy();line_point[1]=center[1]
    line_point[0]=face[0]-normal[1]*(line_point[1]-face[1])/normal[0]
    if time<expert['settle_seconds']:
        result=[0.,0.,0.,.1934,.142,.05,0.,0.,0.];target=None;phase='settle';vx=0.
    else:
        distance=line_point[0]-base[0]
        if not hasattr(env,'_expert_stage'):env._expert_stage='base_approach'
        if env._expert_stage=='base_approach' and distance<=expert['face_distance_for_foot_approach']:
            env._expert_stage='normal_approach';env._expert_contact_start=time
        if env._expert_stage=='normal_approach' and time-env._expert_contact_start>=expert['normal_approach_seconds']:
            env._expert_stage='straight_push'
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
            target=np.array([base[0]+expert['push_foot_x'],line_point[1],line_point[2]])
            vx=expert['push_base_speed']
        origin=env.base_pos_w[0].clone();origin[2]=0
        point=torch.as_tensor(target,device=env.device,dtype=torch.float32)[None]-origin[None]
        local=quat_apply_inverse(projected,point)[0]
        velocity=quat_apply_inverse(projected,torch.tensor([[vx,0.,0.]],device=env.device))[0]
        result=[float(velocity[0]),float(velocity[1]),0.,*local.cpu().tolist(),0.,0.,0.]
    diag=dict(phase=phase,intended_face_center=face.tolist(),intended_face_normal=normal.tolist(),
              commanded_contact_point=None if target is None else target.tolist(),centerline_face_intersection=line_point.tolist(),
              commanded_world_base_velocity=[vx,0.,0.],actual_fl_eef_world=env.actual_fl_world().tolist(),
              box_pose=env.primary_pose_w[0].detach().cpu().tolist(),box_yaw_rad=geom['yaw_rad'],contact_evidence='unknown_diagnostic_only')
    env._expert_diagnostics=diag
    return result,diag
