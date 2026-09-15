"""Synthetic compatibility fixtures only. This is not a Raw-to-Canonical converter."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from .validation import PROFILE, COLUMNS, LOCK, VERSION, CAMERAS, STATE_KEYS, digest, file_hash
from .media import decode_frame
from .finalization import finalize_video


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def checksums(root):
    root=Path(root)
    write_json(root/'meta/checksums.json',{p.relative_to(root).as_posix():file_hash(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'meta/checksums.json'})


def pixels(index, view):
    image=np.empty((720,1280,3),np.uint8)
    image[...,0]=(index*3+view*80)%256
    image[...,1]=np.arange(1280,dtype=np.uint16)[None,:]%256
    image[...,2]=np.arange(720,dtype=np.uint16)[:,None]%256
    # Robust visual source-index marker, retained through lossy encoding.
    for bit in range(12):
        image[16:48,16+bit*32:40+bit*32]=240 if index & (1<<bit) else 16
    return image


def policy_row(index):
    row={k:np.zeros(v['shape'],dtype=v['dtype']) for k,v in COLUMNS.items()}
    for key in ('observation.state.base.orientation','task.object.orientation'):row[key][3]=1
    row['observation.state.projected_gravity'][2]=-1
    request=np.array([index*.001,0,0,.1934,.142,.05,1 if index==0 else 0,0,0],np.float32)
    row['action.requested']=request
    row['action']=request.copy();row['action'][6:]=0
    row['simulation_time_ns'][0]=index*20_000_000;row['physics_step'][0]=index*10
    row['execution.completed'][0]=True;row['execution.force_zero_modified'][0]=bool(request[6:].any())
    return row


def _encode(path, count, view, tools, *, source_dir=None, observer=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix('.partial.mp4')
    assert not path.exists() and not partial.exists()
    fps=25 if observer else 50
    inp=['-framerate',str(fps),'-i',str(Path(source_dir)/'frame_%06d.png')] if source_dir else ['-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-framerate',str(fps),'-i','pipe:0']
    args=[tools.ffmpeg,'-hide_banner','-loglevel','error',*inp,'-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709',
          '-c:v','libx264','-crf','23' if observer else '18']
    if not observer:args+=['-preset','medium','-g','50','-keyint_min','50','-sc_threshold','0']
    args+=['-pix_fmt','yuv420p','-color_range','tv','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-fps_mode','cfr',str(partial)]
    version=subprocess.check_output([tools.ffmpeg,'-version'],text=True).splitlines()[0]
    receipt=dict(profile_hash=digest(PROFILE['policy_video']),encoder='libx264',crf=18,preset='medium',gop=50,keyint_min=50,scenecut=0,color_range='tv',command=args,encoder_version=version)
    if observer:
        receipt=None  # No policy-only GOP/preset receipt is claimed for the monitor.
    if source_dir:
        subprocess.run(args,check=True)
    else:
        proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        try:
            for i in range(count):proc.stdin.write(pixels(i,view).tobytes())
            proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        except BaseException:
            proc.kill();proc.wait();raise
        if code:raise RuntimeError(error.decode())
    result=finalize_video(partial,path,count=count,fps=fps,tools=tools,
                          profile=None if observer else PROFILE['policy_video'],receipt=None if observer else receipt)
    if result['episode_status']!='video_validated':raise RuntimeError(result)
    return receipt


def _write_table(path, columns):
    import pyarrow as pa
    import pyarrow.parquet as pq
    arrays={}
    for name,value in columns.items():
        if isinstance(value,np.ndarray):
            if value.ndim==1:arrays[name]=pa.array(value)
            else:arrays[name]=pa.array(value.tolist(),type=pa.list_(pa.from_numpy_dtype(value.dtype),value.shape[1]))
        else:arrays[name]=pa.array(value)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.table(arrays),path)


def build_fixture(destination, n, tools, *, observer=False):
    """Create raw and canonical diagnostic products from a common synthetic source."""
    import lerobot.datasets.lerobot_dataset as module
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root=Path(destination)
    if root.exists():raise ValueError('Refuse overwrite')
    root.mkdir(parents=True)
    raw=root/'raw';raw.mkdir();canonical=root/'canonical'
    raw_videos={};canon_receipts={}
    for view,key in enumerate(CAMERAS):
        role='ego' if view==0 else 'task_centric'
        video=raw/f'cameras/{role}.mp4';receipt=_encode(video,n+1,view,tools)
        raw_videos[key]=dict(path=video.relative_to(raw).as_posix(),sha256=file_hash(video),encoding=receipt)
    original=module.encode_video_frames
    def fixture_encoder(imgs_dir,video_path,fps,**kwargs):
        # Test-scope override: upstream encoding defaults do not match our requested profile.
        key=next(k for k in CAMERAS if k in str(video_path))
        canon_receipts[key]=_encode(video_path,n,CAMERAS.index(key),tools,source_dir=imgs_dir)
    module.encode_video_frames=fixture_encoder
    try:
        features={k:{'dtype':v['dtype'],'shape':tuple(v['shape']),'names':v['names']} for k,v in COLUMNS.items()}
        features.update({k:dict(dtype='video',shape=(3,720,1280),names=['channels','height','width']) for k in CAMERAS})
        ds=LeRobotDataset.create(repo_id='diagnostic/dataset-contract',root=canonical,fps=50,features=features,robot_type='go2_quadruped_native9',video_backend='pyav')
        for i in range(n):ds.add_frame({**policy_row(i),**{k:pixels(i,v) for v,k in enumerate(CAMERAS)}},task='诊断样例：非真实 demonstration。',timestamp=i/50)
        ds.save_episode()
    finally:
        module.encode_video_frames=original
    meta=canonical/'meta/episodes.jsonl'
    episodes=[json.loads(line) for line in meta.read_text().splitlines()]
    for ep in episodes:ep['action_config']=[dict(start_frame=0,end_frame=ep['length'],action_text=ep['tasks'][0])]
    meta.write_text(''.join(json.dumps(ep,ensure_ascii=False)+'\n' for ep in episodes))
    policy={k:np.stack([policy_row(i)[k] for i in range(n)]) for k in COLUMNS}
    _write_table(raw/'streams/policy_50hz.parquet',{'action.executed' if k=='action' else k:v for k,v in policy.items()})
    _write_table(raw/'streams/controller_100hz.parquet',{'timestamp_ns':np.arange(2*n,dtype=np.int64)*10_000_000,'controller_tick':np.arange(2*n,dtype=np.int64),'high_level_command':np.repeat(policy['action'],2,axis=0),'policy_residual':np.zeros((2*n,18),np.float32),'reference_state':np.zeros((2*n,12),np.float32),'desired_joint_target':np.zeros((2*n,12),np.float32),'desired_joint_velocity':np.zeros((2*n,12),np.float32),'desired_joint_torque':np.zeros((2*n,12),np.float32),'safety_flags':np.zeros(2*n,np.int32),'controller_mode':['synthetic']*(2*n)})
    times=np.arange(10*n+1,dtype=np.int64)*2_000_000
    # Synthetic 6ms updates illustrate why nominal 5ms does not establish actual 200Hz.
    ct=np.arange(0,20_000_000*n+1,6_000_000,dtype=np.int64)
    latest=ct[np.searchsorted(ct,times,side='right')-1]
    _write_table(raw/'streams/contact_sensor.parquet',{'measurement_timestamp_ns':ct,'sensor_sequence_index':np.arange(len(ct),dtype=np.int64),'force':np.zeros((len(ct),3),np.float32),'torque':[None]*len(ct),'frame':['world']*len(ct),'force_kind':['net_normal']*len(ct),'valid':np.ones(len(ct),bool)})
    _write_table(raw/'streams/physics_500hz.parquet',{'timestamp_ns':times,'physics_step':np.arange(10*n+1,dtype=np.int64),'base_pose':np.tile(np.array([0,0,0,0,0,0,1],np.float32),(len(times),1)),'base_twist':np.zeros((len(times),6),np.float32),'joint_position':np.zeros((len(times),12),np.float32),'joint_velocity':np.zeros((len(times),12),np.float32),'joint_torque':np.zeros((len(times),12),np.float32),'object_pose':np.tile(np.array([0,0,0,0,0,0,1],np.float32),(len(times),1)),'object_velocity':np.zeros((len(times),6),np.float32),'external_force':np.zeros((len(times),3),np.float32),'external_torque':np.zeros((len(times),3),np.float32),'external_wrench_frame':['body']*len(times),'contact_measurement_timestamp_ns':latest,'contact_measurement_is_fresh':times==latest})
    raw_counts={k:n+1 for k in CAMERAS}
    monitor=dict(present=observer,source_key='synthetic_observer',canonical_key='monitor.images.observer',reason='synthetic monitor' if observer else 'explicitly omitted in fixture',path='cameras/observer.mp4',crf=23,sha256=None)
    if observer:
        count=n//2+1;_encode(raw/monitor['path'],count,2,tools,observer=True);raw_counts['monitor.images.observer']=count
        monitor['sha256']=file_hash(raw/monitor['path'])
    def index_table(path,counts):
        rows=[(k,i,i*(40_000_000 if k.startswith('monitor.') else 20_000_000)) for k,count in counts.items() for i in range(count)]
        _write_table(path,{'camera_key':[r[0] for r in rows],'frame_index':np.array([r[1] for r in rows],np.int32),'simulation_time_ns':np.array([r[2] for r in rows],np.int64),'physics_step':np.array([r[2]//2_000_000 for r in rows],np.int64),'valid':np.ones(len(rows),bool)})
    index_table(raw/'streams/camera_index.parquet',raw_counts)
    index_table(canonical/'meta/camera_index_000000.parquet',{k:n for k in CAMERAS})
    terminal_state={k:v.tolist() for k,v in policy_row(n).items() if k in STATE_KEYS}
    raw_manifest=dict(dataset_schema_version=VERSION,profile_hash=LOCK['profile_hash'],diagnostic_only=True,episode_uid=f'synthetic-{n}',n_actions=n,n_boundaries=n+1,reset_epoch=0,
                      streams=dict(policy='streams/policy_50hz.parquet',controller='streams/controller_100hz.parquet',physics='streams/physics_500hz.parquet',contact_sensor='streams/contact_sensor.parquet',camera_index='streams/camera_index.parquet'),
                      videos=raw_videos,observer=monitor,terminal=dict(simulation_time_ns=n*20_000_000,state=terminal_state),extensions={})
    raw_manifest['stream_sha256']={k:file_hash(raw/v) for k,v in raw_manifest['streams'].items()}
    write_json(raw/'manifest.json',raw_manifest)
    terminal_images={};videos={}
    for key in CAMERAS:
        raw_video=raw/raw_videos[key]['path'];path=canonical/f'terminal/episode_000000/{key}.png';path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(decode_frame(raw_video,n,tools)).save(path)
        terminal_images[key]=dict(path=path.relative_to(canonical).as_posix(),sha256=file_hash(path),source_video_sha256=file_hash(raw_video),source_frame_index=n,simulation_time_ns=n*20_000_000)
        vp=canonical/ds.meta.get_video_file_path(0,key)
        videos[key]=dict(path=vp.relative_to(canonical).as_posix(),sha256=file_hash(vp),encoding=canon_receipts[key])
    ep=dict(episode_index=0,episode_uid=f'synthetic-{n}',length=n,raw_boundary_count=n+1,raw_episode_uid=f'synthetic-{n}',raw_manifest_sha256=file_hash(raw/'manifest.json'),reset_epoch=0,task_text=episodes[0]['tasks'][0],task_config_hash=digest({'synthetic_task':True}),controller_config_hash=digest({'unchanged_controller':True}),asset_hashes=[],source_commit='9a8989b56989c8450341b6071d33560c1f104c74',checkpoint_sha256='1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706',camera_config_hash=digest(PROFILE['policy_camera_geometry']),camera_index=dict(path='meta/camera_index_000000.parquet',sha256=file_hash(canonical/'meta/camera_index_000000.parquet')),videos=videos,terminal=dict(simulation_time_ns=n*20_000_000,physics_step=n*10,state=terminal_state,images=terminal_images),status='failure')
    write_json(canonical/'meta/contract.json',dict(dataset_schema_version=VERSION,profile_hash=LOCK['profile_hash'],columns_hash=LOCK['columns_hash'],transform_chain_hash=PROFILE['transform_chain_hash'],diagnostic_only=True,episodes=[ep],extensions={}))
    checksums(canonical)
    config=dict(dataset_root=str(raw.resolve()),canonical_root=str(canonical.resolve()),cache_root=str((root/'model-cache').resolve()),ffmpeg=tools.ffmpeg,ffprobe=tools.ffprobe)
    write_json(root/'paths.json',config)
    return config


if __name__ == '__main__':
    import argparse
    from .media import MediaTools
    parser=argparse.ArgumentParser(description='Generate synthetic diagnostic fixtures only')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--counts',type=int,nargs='+',default=[3,16,64,65])
    parser.add_argument('--ffmpeg',required=True);parser.add_argument('--ffprobe',required=True)
    args=parser.parse_args();tools=MediaTools(args.ffmpeg,args.ffprobe)
    if args.output.exists():raise SystemExit('Refuse existing fixture root')
    args.output.mkdir(parents=True)
    for n in args.counts:
        build_fixture(args.output/f'n{n}',n,tools,observer=(n==16))
        print(f'synthetic fixture N={n} ready',flush=True)
