"""Explicitly synthetic CPU fixtures; never demonstrations or runtime evidence."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .validation import BUNDLE, CAMERAS, VERSION, PROFILE_HASHES, digest, file_hash
from .runtime import prepare_command


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def write_jsonl(path,rows):
    Path(path).write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows))


def checksum_episode(root):
    root=Path(root)
    write_json(root/'checksums.json',{p.relative_to(root).as_posix():file_hash(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='checksums.json'})


def write_episode(destination, *, last_tick=320, episode_id='synthetic-640ms', epoch=0):
    root=Path(destination)
    if root.exists() and any(root.iterdir()):raise ValueError('Refuse to overwrite a fixture directory')
    if type(last_tick) is not int or last_tick<0 or last_tick%5:raise ValueError('Fixture uses whole control steps')
    root.mkdir(parents=True,exist_ok=True)
    camera_rows=[]
    for index,tick in enumerate(range(0,last_tick+1,40)):
        for view,k in enumerate(CAMERAS):
            profile=BUNDLE[k]; name=f'rgb/{k}/{index:06d}.png';path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
            rgb=np.zeros((720,1280,3),np.uint8)
            rgb[...,0]=(np.arange(1280)[None,:]//5+index*7+view*70)%256
            rgb[...,1]=(np.arange(720)[:,None]//3+index*11)%256
            rgb[...,2]=50+view*80
            Image.fromarray(rgb).save(path)
            camera_rows.append(dict(camera_id=k,profile_hash=BUNDLE['cameras']['cameras'][k],sensor_frame_id=index,
                                    capture_tick=tick,timestamp_ns=tick*2_000_000,path=name,sha256=file_hash(path),
                                    intrinsics=copy.deepcopy(profile['intrinsics']),world_position=profile['mount_position_m'],
                                    world_quaternion_xyzw=profile['mount_quaternion_xyzw'],reset_epoch=epoch,extensions={}))
    write_jsonl(root/'rgb.jsonl',camera_rows)
    commands=[]
    for i,start in enumerate(range(0,last_tick,10)):
        row=prepare_command([0.1+i*.001,0,0,.1934,.142,.05,1 if i==0 else 0,0,0],f'fixture:{i}',start)
        row['executed_until_tick']=min(start+10,last_tick)
        row['status']='completed' if row['executed_until_tick']==row['end_tick'] else 'partial'
        row['executed']=row['filtered'].copy() if row['status']=='completed' else None
        commands.append(row)
    write_jsonl(root/'commands.jsonl',commands)
    trace=[]
    for start in range(0,last_tick,5):
        row=commands[start//10];previous=None if start==0 else commands[(start-5)//10]['command_id']
        trace.append(dict(start_tick=start,end_tick=start+5,command_id=row['command_id'],held=row['filtered'].copy(),
                          residual_observation_command_id=previous,fl_ik_command_id=previous,fl_ik_computed_tick=start,
                          residual_action_18d=[0.]*18,extensions={}))
    write_jsonl(root/'control.jsonl',trace);write_jsonl(root/'interventions.jsonl',[])
    ticks=np.arange(0,last_tick+1,5,dtype=np.int64)
    for group,layout in BUNDLE['telemetry']['groups'].items():
        arrays={'ticks':ticks}
        for name,spec in layout.items():
            a=np.zeros((len(ticks),*spec['shape']),dtype=spec['dtype'])
            if 'pose' in name:a[:,6]=1
            if name=='projected_gravity':a[:,2]=-1
            if name in ('source_tick','qp_source_tick'):a[:]=ticks
            arrays[name]=a;arrays[name+'_valid']=np.ones(len(ticks),bool)
        np.savez(root/f'{group}.npz',**arrays)
    task_cfg=dict(task_id='contract-diagnostic',version='2.0.0',metrics={'progress':{'unit':'m','description':'synthetic zero progress, not a task measurement'}},success_definition='not evaluated: synthetic fixture',object_body='synthetic_object',contact_pairs={k:dict(sensor_body=b,partner_body='synthetic_object',method='filtered_body_pair') for k,b in [('fl','FL_foot'),('body','base')]},extensions={})
    write_json(root/'task_config.json',task_cfg)
    ep=dict(schema_version=VERSION,dataset_id='synthetic-contract-fixtures',episode_id=episode_id,reset_epoch=epoch,seed=0,diagnostic_only=True,
            origin_sim_ns=0,warmup_end_sim_ns=0,last_tick=last_tick,profiles=copy.deepcopy(PROFILE_HASHES),
            task=dict(id=task_cfg['task_id'],version=task_cfg['version'],instruction='Diagnostic input; do not use for training.',language='en'),
            provenance=dict(rambo_commit='9a8989b56989c8450341b6071d33560c1f104c74',checkpoint_sha256='1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706',
                            isaac_sim='not_run',isaac_lab_commit='ffff603eafc6b74264a5261cc0183d6a65390d78',
                            asset=dict(source='synthetic://contract-fixture',license='synthetic numeric fixture; no imported asset',sha256=digest({'synthetic':True}),approval=None),
                            task_config=dict(path='task_config.json',sha256=file_hash(root/'task_config.json')),runtime_evidence=[]),
            files=dict(rgb_index='rgb.jsonl',commands='commands.jsonl',control_trace='control.jsonl',interventions='interventions.jsonl',
                       **{g:dict(path=g+'.npz',missing_reasons={}) for g in ('robot','contact','task')}),
            outcome=dict(success=False,success_tick=None,terminated=False,truncated=True,reason='synthetic_fixture_end',terminal_snapshot_tick=last_tick,snapshot_epoch=epoch),extensions={})
    write_json(root/'episode.json',ep);checksum_episode(root)
    return root


def message(kind,body, *, message_id, episode_id='synthetic-session', epoch=0):
    return dict(protocol_version=VERSION,session_id='fixture-session',episode_id=episode_id,reset_epoch=epoch,
                message_id=message_id,profiles=copy.deepcopy(PROFILE_HASHES),extensions={},type=kind,body=body)


class FakeBackend:
    """One explicitly fake 100Hz control step; retains prior-observation/IK IDs."""
    def __init__(self):self.calls=[];self.previous=None
    def step(self,schedule):
        row=copy.deepcopy(schedule)
        row.update(residual_observation_command_id=self.previous,fl_ik_command_id=self.previous,
                   fl_ik_computed_tick=row['start_tick'],residual_action_18d=[0.]*18,extensions={})
        self.calls.append(copy.deepcopy(row));self.previous=row['command_id']
        return row


def golden_timeline(fixture_directory):
    """Create one new diagnostic episode and derive a reproducible wire transcript."""
    from .runtime import SynchronousSession
    from .validation import read_json,read_jsonl
    root=write_episode(fixture_directory)
    ep=read_json(root/'episode.json');frames=read_jsonl(root/'rgb.jsonl');commands=read_jsonl(root/'commands.jsonl')
    s=SynchronousSession()
    reset=message('ResetRequest',dict(timeout_s=5,task=ep['task'],origin_sim_ns=0),message_id='reset');ready=s.handle(reset,now=0)
    obs=message('Observation',dict(tick=0,rgb=frames[:2],executed_history=[],instruction=ep['task']['instruction']),message_id='obs0');s.handle(obs,now=0)
    chunk=message('ActionChunk',dict(observation_id='obs0',chunk_id='chunk0',start_tick=0,action_period_ticks=10,actions=[r['requested'] for r in commands]),message_id='action0');accepted=s.handle(chunk,now=1)
    backend=FakeBackend()
    for _ in range(64):ack=s.step(backend)
    obs1=message('Observation',dict(tick=320,rgb=frames[2:],executed_history=s.history,instruction=ep['task']['instruction']),message_id='obs1');s.handle(obs1,now=2)
    return {'diagnostic_only':True,'description':'Synthetic timeline/protocol fixture. No actual images/commands from a robot.',
            'frames':frames,'commands':commands,'last_tick':320,'epoch':0,'origin_ns':0,
            'messages':dict(reset=reset,ready=ready,observation=obs,chunk=chunk,accepted=accepted,ack=ack,next_observation=obs1),
            'expected':{'rgb_per_camera':9,'latent_frames':3,'valid_commands':32,'invalid_initial_slots':16,'control_steps':64,'physics_ticks':320}}
