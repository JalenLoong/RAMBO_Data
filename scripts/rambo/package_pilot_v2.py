"""Package real recorder output as Raw Parquet and one LeRobot canonical episode."""
import argparse,copy,json,subprocess
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import av
from PIL import Image
from rambo.dataset_v2.validation import COLUMNS,PROFILE,LOCK,VERSION,CAMERAS,digest,file_hash,validate_raw,validate_canonical
from rambo.dataset_v2.media import MediaTools,decode_frame,validate_video
from rambo.dataset_v2.finalization import finalize_video


def write_json(path,v):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def table(path,rows,types=None):
    types=types or {};fields={}
    for key in rows[0]:
        values=[r[key] for r in rows];sample=next((v for v in values if v is not None),None)
        typ=types.get(key)
        if typ is None:
            if isinstance(sample,bool):typ=pa.bool_()
            elif isinstance(sample,int):typ=pa.int64()
            elif isinstance(sample,float):typ=pa.float32()
            elif isinstance(sample,list) and sample and not isinstance(sample[0],list):
                typ=pa.list_(pa.string() if isinstance(sample[0],str) else (pa.bool_() if isinstance(sample[0],bool) else pa.float32()),len(sample))
        fields[key]=pa.array(values,type=typ)
    path.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.table(fields),path)


def encode_canonical(source,dest,n,tools):
    dest.parent.mkdir(parents=True,exist_ok=True);partial=dest.with_suffix('.partial.mp4')
    command=[tools.ffmpeg,'-hide_banner','-loglevel','error','-i',str(source),'-frames:v',str(n),'-vf','scale=in_range=tv:out_range=tv:in_color_matrix=bt709:out_color_matrix=bt709','-c:v','libx264','-threads','2','-crf','18','-preset','medium','-g','50','-keyint_min','50','-sc_threshold','0','-pix_fmt','yuv420p','-color_range','tv','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-fps_mode','cfr',str(partial)]
    subprocess.run(command,check=True)
    receipt=dict(profile_hash=digest(PROFILE['policy_video']),encoder='libx264',crf=18,preset='medium',gop=50,keyint_min=50,scenecut=0,color_range='tv',command=command,encoder_version=subprocess.check_output([tools.ffmpeg,'-version'],text=True).splitlines()[0])
    result=finalize_video(partial,dest,count=n,fps=50,tools=tools,profile=PROFILE['policy_video'],receipt=receipt)
    if result['episode_status']!='video_validated':raise ValueError(result)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--canonical',type=Path,required=True);p.add_argument('--ffmpeg',required=True);p.add_argument('--ffprobe',required=True);args=p.parse_args()
    raw=args.raw.resolve();canonical=args.canonical.resolve();tools=MediaTools(args.ffmpeg,args.ffprobe)
    if canonical.exists() or (raw/'manifest.json').exists():raise ValueError('Refuse overwrite')
    cap=json.loads((raw/'capture.json').read_text());summary=json.loads((raw/'summary.json').read_text());assert summary['passed'] and summary['terminal_before_reset']
    n=len(cap['commands']);boundaries=cap['boundaries'];assert len(boundaries)==n+1 and n>0
    assert not cap['terminal']['partial_interval']
    uid=raw.name;videos={}
    for role,key in [('ego',CAMERAS[0]),('task_centric',CAMERAS[1]),('observer','monitor.images.observer')]:
        rec=cap['videos'][role];partial=raw/'cameras'/f'{role}.partial.mp4';final=raw/'cameras'/f'{role}.mp4'
        if final.exists():
            assert not partial.exists(), 'Ambiguous video state'
            validate_video(final,count=rec['count'],fps=25 if role=='observer' else 50,tools=tools,profile=None if role=='observer' else PROFILE['policy_video'],receipt=None if role=='observer' else rec['receipt'])
            result={'episode_status':'video_validated'}
        else:
            result=finalize_video(partial,final,count=rec['count'],fps=25 if role=='observer' else 50,tools=tools,profile=None if role=='observer' else PROFILE['policy_video'],receipt=None if role=='observer' else rec['receipt'])
        if result['episode_status']!='video_validated':raise ValueError(result)
        if role!='observer':videos[key]=dict(path=str(final.relative_to(raw)),sha256=file_hash(final),encoding=rec['receipt'])
    rows=[]
    for obs,cmd in zip(boundaries[:-1],cap['commands']):
        assert cmd['status']=='completed' and cmd['start_tick']==obs['physics_step'] and cmd['end_tick']==obs['physics_step']+10
        row=copy.deepcopy(obs['state']);row.update({'action':cmd['executed'],'action.requested':np.asarray(cmd['requested'],np.float32).tolist(),'simulation_time_ns':obs['simulation_time_ns'],'physics_step':obs['physics_step'],'execution.completed':True,'execution.force_zero_modified':cmd['force_overridden']})
        for key,spec in COLUMNS.items():
            if spec['shape']==[1] and isinstance(row[key],list):row[key]=row[key][0]
        rows.append(row)
    types={key:pa.from_numpy_dtype(spec['dtype']) if spec['shape']==[1] else pa.list_(pa.from_numpy_dtype(spec['dtype']),spec['shape'][0]) for key,spec in COLUMNS.items()}
    table(raw/'streams/policy_50hz.parquet',[{'action.executed' if k=='action' else k:v for k,v in row.items()} for row in rows],{'action.executed' if k=='action' else k:v for k,v in types.items()})
    table(raw/'streams/controller_100hz.parquet',cap['controller'],{'safety_flags':pa.int32()})
    ct=np.array([r['measurement_timestamp_ns'] for r in cap['contact_sensor']],dtype=np.int64);assert len(ct)>0
    previous=None
    for row in cap['physics']:
        i=int(np.searchsorted(ct,row['timestamp_ns'],side='right'))-1;latest=None if i<0 else int(ct[i])
        row['contact_measurement_timestamp_ns']=latest;row['contact_measurement_is_fresh']=latest is not None and latest!=previous;previous=latest
    table(raw/'streams/physics_500hz.parquet',cap['physics'],{'contact_measurement_timestamp_ns':pa.int64()})
    table(raw/'streams/contact_sensor.parquet',cap['contact_sensor'])
    table(raw/'streams/camera_index.parquet',cap['camera_index'],{'frame_index':pa.int32()})
    provenance=dict(task_profile=cap['profile'],implementation=cap['implementation'],source_commit=cap['source_commit'],checkpoint_sha256=cap['checkpoint_sha256'],capture_sha256=file_hash(raw/'capture.json'),terminal_gate=summary,collector='geometry_aware_native9_expert_v2; existing residual controller; no object teleport/wrench',fl_velocity_method='analytical joint Jacobian plus finite-difference gravity-frame derivative epsilon=1e-4')
    provenance['initial_geometry']=cap['initial_geometry']
    write_json(raw/'metadata/provenance.json',provenance)
    write_json(raw/'metadata/task_review.json',{'boundaries':[dict(simulation_time_ns=b['simulation_time_ns'],**b['task_review']) for b in boundaries],'commands':[dict(simulation_time_ns=c['start_tick']*2_000_000,**c['extensions']['expert']) for c in cap['commands']]})
    contacts=dict(version='push-box-contact-diagnostic-v1',role='diagnostic_only',unavailable='unknown',bool_columns_are_invalid_placeholders=True,rows=[dict(simulation_time_ns=b['simulation_time_ns'],**b['contact']) for b in boundaries])
    write_json(raw/'metadata/contact_provenance.json',contacts)
    extension=dict(task_profile_version=cap['profile']['task_profile_version'],contact_diagnostics=dict(path='metadata/contact_provenance.json',sha256=file_hash(raw/'metadata/contact_provenance.json')),provenance=dict(path='metadata/provenance.json',sha256=file_hash(raw/'metadata/provenance.json')),origin='real_isaac_acquisition',purpose=cap['mode'],task_review=dict(path='metadata/task_review.json',sha256=file_hash(raw/'metadata/task_review.json')))
    streams={k:f'streams/{v}.parquet' for k,v in dict(policy='policy_50hz',controller='controller_100hz',physics='physics_500hz',contact_sensor='contact_sensor',camera_index='camera_index').items()}
    terminal=cap['terminal'];rawmanifest=dict(dataset_schema_version=VERSION,profile_hash=LOCK['profile_hash'],diagnostic_only=True,episode_uid=uid,n_actions=n,n_boundaries=n+1,reset_epoch=0,streams=streams,stream_sha256={k:file_hash(raw/v) for k,v in streams.items()},videos=videos,observer=dict(present=True,source_key='approved_fixed_observer',canonical_key='monitor.images.observer',reason='monitor only',path='cameras/observer.mp4',crf=23,sha256=file_hash(raw/'cameras/observer.mp4')),terminal=dict(simulation_time_ns=terminal['simulation_time_ns'],state=terminal['state']),extensions=extension)
    write_json(raw/'manifest.json',rawmanifest)
    rawcheck=validate_raw(raw,tools=tools);write_json(raw/'raw-validation.json',rawcheck)
    canonical.mkdir(parents=True)
    features={k:dict(dtype=v['dtype'],shape=v['shape'],names=v['names']) for k,v in COLUMNS.items()}
    for k in CAMERAS:features[k]=dict(dtype='video',shape=[3,720,1280],names=['channels','height','width'],info={'video.height':720,'video.width':1280,'video.codec':'h264','video.pix_fmt':'yuv420p','video.is_depth_map':False,'video.fps':50,'video.channels':3,'has_audio':False})
    for k in ['timestamp','frame_index','episode_index','index','task_index']:features[k]=dict(dtype='float32' if k=='timestamp' else 'int64',shape=[1],names=None)
    info=dict(codebase_version='v2.1',robot_type='go2_quadruped_native9',total_episodes=1,total_frames=n,total_tasks=1,total_videos=2,total_chunks=1,chunks_size=1000,fps=50,splits={'train':'0:1'},data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',features=features)
    write_json(canonical/'meta/info.json',info)
    for i,row in enumerate(rows):row.update(timestamp=row['simulation_time_ns']/1e9,frame_index=i,episode_index=0,index=i,task_index=0)
    table(canonical/'data/chunk-000/episode_000000.parquet',rows,{**types,'timestamp':pa.float32()})
    stats={}
    for key in rows[0]:
        a=np.asarray([r[key] for r in rows],dtype=np.float64).reshape(n,-1)
        stats[key]={name:value.tolist() for name,value in [('min',a.min(0)),('max',a.max(0)),('mean',a.mean(0)),('std',a.std(0))]};stats[key]['count']=[n]
    canonvideos={};images={}
    for key in CAMERAS:
        source=raw/videos[key]['path'];dest=canonical/info['video_path'].format(episode_chunk=0,video_key=key,episode_index=0)
        receipt=encode_canonical(source,dest,n,tools);canonvideos[key]=dict(path=str(dest.relative_to(canonical)),sha256=file_hash(dest),encoding=receipt)
        path=canonical/f'terminal/episode_000000/{key}.png';path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(decode_frame(source,n,tools)).save(path)
        images[key]=dict(path=str(path.relative_to(canonical)),sha256=file_hash(path),source_video_sha256=file_hash(source),source_frame_index=n,simulation_time_ns=terminal['simulation_time_ns'])
        sums=np.zeros(3);squares=np.zeros(3);lo=np.ones(3);hi=np.zeros(3);count=0
        with av.open(str(dest)) as container:
            for frame in container.decode(video=0):
                pixels=frame.to_ndarray(format='rgb24').reshape(-1,3).astype(np.float64)/255
                sums+=pixels.sum(0);squares+=(pixels*pixels).sum(0);lo=np.minimum(lo,pixels.min(0));hi=np.maximum(hi,pixels.max(0));count+=len(pixels)
        mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean*mean,0))
        stats[key]={name:v.reshape(3,1,1).tolist() for name,v in [('min',lo),('max',hi),('mean',mean),('std',std)]};stats[key]['count']=[n]
    (canonical/'meta/tasks.jsonl').write_text(json.dumps(dict(task_index=0,task=cap['task_text']))+'\n')
    (canonical/'meta/episodes.jsonl').write_text(json.dumps(dict(episode_index=0,tasks=[cap['task_text']],length=n,action_config=[dict(start_frame=0,end_frame=n,action_text=cap['task_text'])]))+'\n')
    (canonical/'meta/episodes_stats.jsonl').write_text(json.dumps(dict(episode_index=0,stats=stats))+'\n')
    table(canonical/'meta/camera_index_000000.parquet',[r for r in cap['camera_index'] if r['camera_key'] in CAMERAS and r['frame_index']<n],{'frame_index':pa.int32()})
    for name,value in [('provenance.json',provenance),('contact_provenance.json',contacts),('task_review.json',json.loads((raw/'metadata/task_review.json').read_text()))]:write_json(canonical/'metadata'/name,value)
    ep=dict(episode_index=0,episode_uid=uid,length=n,raw_boundary_count=n+1,raw_episode_uid=uid,raw_manifest_sha256=file_hash(raw/'manifest.json'),reset_epoch=0,task_text=cap['task_text'],task_config_hash=digest(cap['profile']),controller_config_hash=digest({'checkpoint':cap['checkpoint_sha256'],'implementation':cap['implementation']}),asset_hashes=[cap['profile']['asset_sha256']],source_commit=cap['source_commit'],checkpoint_sha256=cap['checkpoint_sha256'],camera_config_hash=digest(PROFILE['policy_camera_geometry']),camera_index=dict(path='meta/camera_index_000000.parquet',sha256=file_hash(canonical/'meta/camera_index_000000.parquet')),videos=canonvideos,terminal=dict(simulation_time_ns=terminal['simulation_time_ns'],physics_step=terminal['physics_step'],state=terminal['state'],images=images),status=cap['status'])
    write_json(canonical/'meta/contract.json',dict(dataset_schema_version=VERSION,profile_hash=LOCK['profile_hash'],columns_hash=LOCK['columns_hash'],transform_chain_hash=PROFILE['transform_chain_hash'],diagnostic_only=True,episodes=[ep],extensions={**extension,'split_semantics':'technical pilot only, LeRobot train alias is not approved SFT split','conversion':'raw limited-range BT709 decode/re-encode first N frames, no resizing; terminal decoded directly from raw frame N'}))
    write_json(canonical/'meta/checksums.json',{str(f.relative_to(canonical)):file_hash(f) for f in sorted(canonical.rglob('*')) if f.is_file() and f.name!='checksums.json'})
    result=validate_canonical(canonical,tools=tools);write_json(canonical.parent/(canonical.name+'-validation.json'),result)
    print(json.dumps({'raw':rawcheck,'canonical':result}),flush=True)
if __name__=='__main__':main()
