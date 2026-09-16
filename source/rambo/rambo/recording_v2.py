"""Single-environment real acquisition. Offline Parquet packaging is separate."""
from pathlib import Path
import copy
import hashlib
import json
import subprocess
import numpy as np
from PIL import Image
from .dataset_v2.validation import PROFILE, digest


def json_write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


class VideoSink:
    def __init__(self,path,ffmpeg,fps):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.partial=self.path.with_suffix('.partial.mp4');self.count=0
        if self.path.exists() or self.partial.exists():raise ValueError('Refuse overwrite')
        crf=18 if fps==50 else 23
        self.command=[ffmpeg,'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-framerate',str(fps),'-i','pipe:0','-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709','-c:v','libx264','-threads','2','-crf',str(crf),'-preset','medium']
        if fps==50:self.command+=['-g','50','-keyint_min','50','-sc_threshold','0']
        self.command+=['-pix_fmt','yuv420p','-color_range','tv','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-fps_mode','cfr',str(self.partial)]
        self.errors=self.partial.with_suffix('.encoder.log').open('x')
        self.proc=subprocess.Popen(self.command,stdin=subprocess.PIPE,stderr=self.errors)
        self.receipt=dict(profile_hash=digest(PROFILE['policy_video']),encoder='libx264',crf=18,preset='medium',gop=50,keyint_min=50,scenecut=0,color_range='tv',command=self.command,encoder_version=subprocess.check_output([ffmpeg,'-version'],text=True).splitlines()[0])
    def write(self,rgb):
        if rgb.dtype!=np.uint8 or rgb.shape!=(720,1280,3):raise ValueError('RGB8 shape')
        self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes());self.count+=1
    def close(self):
        self.proc.stdin.close();code=self.proc.wait();self.errors.close()
        if code:raise RuntimeError(f'Encoder failed {self.partial}')
        # Only the offline full-video validator may rename the partial.
        return dict(partial=str(self.partial),final=str(self.path),count=self.count,receipt=self.receipt)


class Recorder:
    def __init__(self,env,out,ffmpeg,observer):
        self.env=env;self.out=Path(out);self.observer=observer;self.active=False
        self.physics=[];self.contacts=[];self.controls=[];self.boundaries=[];self.camera_index=[];self.commands=[]
        self.terminal=None;self.current=None;self.last_observation_command_id=None;self.ik_command_id=None;self.ik_tick=None
        self.last_sensor_update_global=None
        self.sinks={k:VideoSink(self.out/'cameras'/f'{k}.mp4',ffmpeg,25 if k=='observer' else 50) for k in ['ego','task_centric','observer']}
        sensor=env._contact_sensor;original=sensor._update_buffers_impl
        def observed_update(*args,**kwargs):
            result=original(*args,**kwargs);self.last_sensor_update_global=int(env._sim_step_counter)
            if self.active:self.contact_updated()
            return result
        sensor._update_buffers_impl=observed_update
    @property
    def tick(self):return int(self.env._sim_step_counter)-self.origin_tick
    @property
    def ns(self):return self.tick*2_000_000  # Actual completed PhysX steps, never camera/frame indices.
    def arr(self,tensor):return tensor.detach().cpu().numpy().copy()
    def contact_updated(self):
        if self.contacts and self.contacts[-1]['measurement_timestamp_ns']==self.ns:return
        e=self.env;forces=self.arr(e._contact_sensor._data.net_forces_w.torch[0])
        self.contacts.append(dict(measurement_timestamp_ns=self.ns,sensor_sequence_index=len(self.contacts),force=forces[int(e._contact_feet_ids[0])].tolist(),torque=None,frame='world',force_kind='net_normal',valid=True,all_body_net_normal=forces.tolist(),body_names=list(e._contact_sensor.body_names)))
    def begin(self):
        self.origin_tick=int(self.env._sim_step_counter);self.active=True
        if self.last_sensor_update_global==self.origin_tick:self.contact_updated()
        self.physics_snapshot(np.zeros(3),np.zeros(3));self.capture_boundary()
    def state(self):
        import torch
        from rambo.utils import math as rm
        e=self.env;r=e._robot.data;o=e._primary.data
        g=e.projected_gravity_b;R=e.base_rot_mat_rp;foot=e.ee_pos_b[:,0]
        pos=(R@foot.unsqueeze(-1)).squeeze(-1);pos[:,2]+=e.base_height
        # Derivative of the existing analytical projected-frame foot point.
        body_v=(e.all_foot_jacobian[:,:3]@e.joint_vel.unsqueeze(-1)).squeeze(-1)
        dg=-torch.linalg.cross(e.base_ang_vel_b,g,dim=-1);eps=1e-4
        g1=g+eps*dg;g1=g1/torch.linalg.vector_norm(g1,dim=-1,keepdim=True)
        R1=rm.rp_rotation_from_gravity_b(g1).transpose(1,2)
        vel=(R@body_v.unsqueeze(-1)).squeeze(-1)+((R1-R)/eps@foot.unsqueeze(-1)).squeeze(-1);vel[:,2]+=e.base_lin_vel_w[:,2]
        forces=e._contact_sensor._data.net_forces_w.torch[0]
        feet=forces[e._contact_feet_ids]
        center=e.geometric_center[0]
        values={'observation.state.base.position':r.root_link_pos_w.torch[0], 'observation.state.base.orientation':r.root_link_quat_w.torch[0], 'observation.state.base.linear_velocity':r.root_link_lin_vel_w.torch[0], 'observation.state.base.angular_velocity':r.root_link_ang_vel_w.torch[0], 'observation.state.projected_gravity':g[0], 'observation.state.joint_position':e.joint_pos[0], 'observation.state.joint_velocity':e.joint_vel[0], 'observation.state.fl_ee_position':pos[0], 'observation.state.fl_ee_velocity':vel[0], 'observation.state.fl_contact_force':feet[0], 'task.object.position':o.root_link_pos_w.torch[0], 'task.object.orientation':o.root_link_quat_w.torch[0], 'task.object.linear_velocity':o.root_link_lin_vel_w.torch[0], 'task.object.angular_velocity':o.root_link_ang_vel_w.torch[0]}
        state={k:self.arr(v).astype(np.float32).tolist() for k,v in values.items()}
        state.update({'observation.state.foot_contact':self.arr(torch.linalg.vector_norm(feet,dim=-1)>0).tolist(),'task.goal.position':[1.15,.15,0.],'task.progress':[float(center[0])-e._initial_center_x],'task.success':[bool(e.task_success[0])],'task.contact.fl_object':[False],'task.contact.body_object':[False]})
        return state
    def capture_boundary(self):
        if self.boundaries and self.boundaries[-1]['simulation_time_ns']==self.ns:return self.boundaries[-1]
        row=dict(simulation_time_ns=self.ns,physics_step=self.tick,state=self.state(),contact={'fl_object':None,'body_object':None,'status':'unknown','valid':False,'reason':'No reliable target-pair sensor configured; bool columns are invalid placeholders'},rgb_hashes={},sensor_frame_ids={})
        row['task_review']=self.env.review_diagnostics()
        self.last_rgb={}
        for role,sensor in [('ego',self.env.front_camera),('task_centric',self.env.task_camera)]:
            rgb=self.arr(sensor.data.output['rgb'].torch[0]);self.last_rgb[role]=rgb.copy()
            row['rgb_hashes'][role]=hashlib.sha256(rgb.tobytes()).hexdigest()
            frame=sensor.frame
            row['sensor_frame_ids'][role]=int(frame.torch[0]) if hasattr(frame,'torch') else int(frame[0])
            self.sinks[role].write(rgb)
            self.camera_index.append(dict(camera_key='observation.images.'+role,frame_index=self.sinks[role].count-1,simulation_time_ns=self.ns,physics_step=self.tick,valid=True))
        if self.tick%20==0:
            rgb=np.asarray(self.observer.get_data())[:,:,:3].copy();self.sinks['observer'].write(rgb)
            self.camera_index.append(dict(camera_key='monitor.images.observer',frame_index=self.sinks['observer'].count-1,simulation_time_ns=self.ns,physics_step=self.tick,valid=True))
        self.boundaries.append(row);return row
    def submit(self,prepared):
        self.current=copy.deepcopy(prepared)
        self.current['command_id']=prepared['command_id']
    def before_control(self,action):
        if not self.active:return
        e=self.env
        held=np.concatenate([self.arr(e._velocity_commands[0]),self.arr(e._ee_pos_commands[0]),self.arr(e._ee_force_commands[0])])
        if not np.array_equal(held,np.asarray(self.current['filtered'],np.float32)):raise ValueError('Actual submitted command differs from ledger')
        self.pending_control=dict(timestamp_ns=self.ns,controller_tick=self.tick//5,high_level_command=held.tolist(),policy_residual=self.arr(action[0]).tolist(),reference_state=self.arr(e._desired_joint_pos[0]).tolist(),residual_observation_command_id=self.last_observation_command_id,fl_ik_command_id=self.ik_command_id,fl_ik_computed_tick=self.ik_tick,command_id=self.current['command_id'],controller_mode='retained_quadruped_rambo',safety_flags=0)
    def physics_snapshot(self,force,torque):
        e=self.env;r=e._robot.data;o=e._primary.data
        row=dict(timestamp_ns=self.ns,physics_step=self.tick,base_pose=self.arr(r.root_link_pose_w.torch[0]).tolist(),base_twist=self.arr(r.root_link_vel_w.torch[0]).tolist(),joint_position=self.arr(e.joint_pos[0]).tolist(),joint_velocity=self.arr(e.joint_vel[0]).tolist(),joint_torque=self.arr(r.applied_torque.torch[0].index_select(0,e._go2_indices.joint_ids)).tolist(),object_pose=self.arr(o.root_link_pose_w.torch[0]).tolist(),object_velocity=self.arr(o.root_link_vel_w.torch[0]).tolist(),external_force=np.asarray(force).reshape(3).tolist(),external_torque=np.asarray(torque).reshape(3).tolist(),external_wrench_frame='body')
        row['actual_fl_eef_world']=e.actual_fl_world().tolist()
        self.physics.append(row)
    def after_physics(self,force,torque):
        if not self.active:return
        self.physics_snapshot(self.arr(force[0,0]),self.arr(torque[0,0]))
        if self.tick%5==1:
            e=self.env;self.pending_control.update(desired_joint_target=self.arr(e.desired_pos[0]).tolist(),desired_joint_velocity=self.arr(e.desired_vel[0]).tolist(),desired_joint_torque=self.arr(e.desired_tor[0]).tolist())
    def after_control(self):
        if not self.active:return
        self.pending_control['completed_at_ns']=self.ns;self.controls.append(self.pending_control)
        self.ik_command_id=self.current['command_id'];self.ik_tick=self.tick;self.last_observation_command_id=self.current['command_id']
        if self.tick%10==0:
            completed=copy.deepcopy(self.current);completed.update(status='completed',executed=completed['filtered'],executed_until_tick=self.tick)
            self.commands.append(completed)
    def before_reset(self,ids):
        if not self.active:return
        row=self.capture_boundary()
        self.terminal=copy.deepcopy(row)
        self.terminal_rgb={k:v.copy() for k,v in self.last_rgb.items()}
        self.terminal['partial_interval']=self.tick%10!=0
        self.terminal['reason']='robot_fall' if bool(self.env.fallen[0]) else ('task_success' if row['state']['task.success'][0] else 'timeout')
        for role,rgb in self.terminal_rgb.items():
            path=self.out/'review'/f'terminal_pre_reset_{role}.png';path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(path)
        self.active=False  # reset-side buffer updates must never contaminate this episode
    def close(self,extra):
        videos={k:s.close() for k,s in self.sinks.items()}
        payload=dict(boundaries=self.boundaries,commands=self.commands,controller=self.controls,physics=self.physics,contact_sensor=self.contacts,camera_index=self.camera_index,terminal=self.terminal,videos=videos,**extra)
        json_write(self.out/'capture.json',payload);return payload
