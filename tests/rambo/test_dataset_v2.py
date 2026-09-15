"""Offline dataset tooling tests; run with the locked policy/file-tool Python (LeRobot/PyArrow)."""
from pathlib import Path
import copy
import json
import shutil

import numpy as np
import pytest
pytest.importorskip('pyarrow')
pytest.importorskip('lerobot')
import pyarrow as pa
import pyarrow.parquet as pq

from rambo.dataset_v2 import DataPaths,MediaTools,validate_canonical,validate_raw
from rambo.dataset_v2.validation import ContractError,PROFILE,read_json,file_hash
from rambo.dataset_v2.fixtures import build_fixture,checksums,write_json
from rambo.dataset_v2.finalization import finalize_video
from rambo.dataset_v2.media import decode_frame,read_terminal_png,validate_video

WORKSPACE=Path(__file__).resolve().parents[4]
TOOLS=MediaTools(str(WORKSPACE/'cache/tools/bin/ffmpeg'),str(WORKSPACE/'cache/tools/bin/ffprobe'))

@pytest.fixture(scope='module')
def cases(tmp_path_factory):
    base=tmp_path_factory.mktemp('dataset-v2')
    return {n:DataPaths.from_config(build_fixture(base/f'n{n}',n,TOOLS,observer=(n==16))) for n in (3,16,64,65)}

@pytest.mark.parametrize('n',[3,16,64,65])
def test_native_le_robot_and_raw_shapes(cases,n):
    p=cases[n]
    assert validate_raw(p.dataset_root,tools=TOOLS)['raw_boundaries']==n+1
    result=validate_canonical(p.canonical_root,tools=TOOLS)
    assert result['episodes'][0]['rows']==n and result['episodes'][0]['terminal_action_count']==0
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    d=LeRobotDataset(str(p.canonical_root),root=p.canonical_root,video_backend='pyav',download_videos=False)
    assert len(d)==n and tuple(d[n-1]['action'].shape)==(9,)
    assert tuple(d[0]['observation.images.ego'].shape)==(3,720,1280)
    assert d[0]['task']=='诊断样例：非真实 demonstration。'


def test_terminal_snapshot_source_and_frame_order(cases):
    p=cases[64];raw=read_json(p.dataset_root/'manifest.json');ep=read_json(p.canonical_root/'meta/contract.json')['episodes'][0]
    for camera in PROFILE['camera_keys']:
        source=decode_frame(p.dataset_root/raw['videos'][camera]['path'],64,TOOLS)
        snapshot=read_terminal_png(p.canonical_root/ep['terminal']['images'][camera]['path'])
        np.testing.assert_array_equal(source,snapshot)
        for i in (0,31,50,63):
            image=decode_frame(p.canonical_root/ep['videos'][camera]['path'],i,TOOLS)
            marker=sum((int(np.median(image[20:44,20+b*32:36+b*32]))>128)<<b for b in range(12))
            assert marker==i

@pytest.mark.parametrize('mutation,code',[
 ('version','schema'),('observer','observer'),('camera_names','columns'),('terminal_action','schema'),
 ('terminal_time','terminal'),('terminal_frame','terminal'),('missing_terminal','terminal'),
 ('fake_action_row','timing'),('action_config','action_config'),('timestamps','timing'),
 ('force','transform'),('unconfirmed','execution'),('dtype','dtype'),('null','dtype'),
 ('clamp','transform'),('chain','transform'),('bad_receipt','media'),('corrupt_mp4','media'),
 ('partial','media'),('checksum','checksum')])
def test_canonical_negative(cases,tmp_path,mutation,code):
    root=tmp_path/'canonical';shutil.copytree(cases[16].canonical_root,root)
    m=read_json(root/'meta/contract.json');ep=m['episodes'][0]
    info=read_json(root/'meta/info.json')
    data=root/info['data_path'].format(episode_chunk=0,episode_index=0)
    if mutation=='version':m['dataset_schema_version']='2.0.0'
    if mutation=='observer':info['features']['observation.images.observer']=info['features']['observation.images.ego']
    if mutation=='camera_names':info['features']['action']['names'][0]='other'
    if mutation=='terminal_action':ep['terminal']['action']=[0.]*9
    if mutation=='terminal_time':ep['terminal']['simulation_time_ns']+=1
    if mutation=='terminal_frame':ep['terminal']['images']['observation.images.ego']['source_frame_index']-=1
    if mutation=='missing_terminal':ep['terminal']['images']['observation.images.ego']['path']='terminal/missing.png'
    if mutation=='chain':m['transform_chain_hash']='0'*64
    if mutation=='action_config':
        es=[json.loads(x) for x in (root/'meta/episodes.jsonl').read_text().splitlines()];es[0]['action_config'][0]['end_frame']-=1
        (root/'meta/episodes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in es))
    if mutation in ('fake_action_row','timestamps','force','unconfirmed','dtype','null','clamp'):
        t=pq.read_table(data)
        if mutation=='fake_action_row':t=pa.concat_tables([t,t.slice(0,1)])
        else:
            key={'timestamps':'simulation_time_ns','force':'action','unconfirmed':'execution.completed','dtype':'action','null':'execution.completed','clamp':'action'}[mutation]
            typ=t.schema.field(key).type;v=t[key].to_pylist()
            if mutation=='timestamps':v[1]=[123] if isinstance(v[1],list) else 123
            if mutation=='force':v[0][6]=1
            if mutation=='clamp':v[0][0]=1
            if mutation=='unconfirmed':v[0]=[False] if isinstance(v[0],list) else False
            if mutation=='null':v[0]=None
            if mutation=='dtype':typ=pa.list_(pa.float64(),9)
            t=t.set_column(t.column_names.index(key),key,pa.array(v,type=typ))
        pq.write_table(t,data)
    video=ep['videos']['observation.images.ego']
    if mutation=='bad_receipt':video['encoding']['crf']=23
    if mutation=='corrupt_mp4':
        (root/video['path']).write_bytes(b'bad movie');video['sha256']=file_hash(root/video['path'])
    if mutation=='partial':
        old=root/video['path'];new=old.with_suffix('.partial.mp4');old.rename(new)
        video['path']=new.relative_to(root).as_posix();info['video_path']=info['video_path'].replace('.mp4','.partial.mp4')
        # Both camera paths follow the changed template, so the partial-file guard is exercised.
        other=ep['videos']['observation.images.task_centric'];o=root/other['path'];q=o.with_suffix('.partial.mp4');o.rename(q);other['path']=q.relative_to(root).as_posix()
    write_json(root/'meta/info.json',info);write_json(root/'meta/contract.json',m);checksums(root)
    if mutation=='checksum':data.write_bytes(b'changed')
    with pytest.raises(ContractError) as e:validate_canonical(root,tools=TOOLS,check_media=mutation in ('bad_receipt','corrupt_mp4','partial'))
    assert e.value.code==code

@pytest.mark.parametrize('mutation,code',[('sensor_duplicate','contact'),('freshness','contact'),('torque','contact'),('external','force'),('hold','execution'),('camera_count','camera_index')])
def test_raw_negative(cases,tmp_path,mutation,code):
    root=tmp_path/'raw';shutil.copytree(cases[16].dataset_root,root);m=read_json(root/'manifest.json')
    stream='contact_sensor' if mutation in ('sensor_duplicate','torque') else 'controller' if mutation=='hold' else 'camera_index' if mutation=='camera_count' else 'physics'
    p=root/m['streams'][stream];t=pq.read_table(p)
    if mutation=='camera_count':t=t.slice(0,len(t)-1)
    else:
        key={'sensor_duplicate':'measurement_timestamp_ns','freshness':'contact_measurement_is_fresh','torque':'torque','external':'external_torque','hold':'high_level_command'}[mutation]
        values=t[key].to_pylist();typ=t.schema.field(key).type
        if mutation=='sensor_duplicate':values[1]=values[0]
        if mutation=='freshness':values[1]=True
        if mutation=='torque':values=[[0.,0.,0.]]*len(t);typ=pa.list_(pa.float32(),3)
        if mutation=='external':values[1][0]=1
        if mutation=='hold':values[1][0]=1
        t=t.set_column(t.column_names.index(key),key,pa.array(values,type=typ))
    pq.write_table(t,p);m['stream_sha256'][stream]=file_hash(p);write_json(root/'manifest.json',m)
    with pytest.raises(ContractError) as e:validate_raw(root,tools=TOOLS,check_media=False)
    assert e.value.code==code


def test_partial_finalization_does_not_publish_corrupt_video(tmp_path):
    p=tmp_path/'camera.partial.mp4';p.write_bytes(b'not a complete video')
    result=finalize_video(p,tmp_path/'camera.mp4',count=16,fps=50,tools=TOOLS)
    assert result['episode_status']=='invalid' and p.exists() and not (tmp_path/'camera.mp4').exists()


def test_paths_require_configuration_and_separate_cache(cases):
    p=cases[16];config=json.loads((p.dataset_root.parent/'paths.json').read_text())
    del config['cache_root']
    with pytest.raises(ContractError):DataPaths.from_config(config)
    config['cache_root']=str(p.canonical_root/'latents')
    with pytest.raises(ContractError,match='independent'):DataPaths.from_config(config)


def test_policy_encoding_rejects_observer_fps(cases):
    p=cases[16]
    with pytest.raises(ContractError,match='CFR'):validate_video(p.dataset_root/'cameras/observer.mp4',count=9,fps=50,tools=TOOLS)


def test_contact_start_does_not_invent_a_time_zero_measurement(cases,tmp_path):
    root=tmp_path/'raw';shutil.copytree(cases[16].dataset_root,root)
    m=read_json(root/'manifest.json');cp=root/m['streams']['contact_sensor'];pp=root/m['streams']['physics']
    contact=pq.read_table(cp).slice(1)
    contact=contact.set_column(contact.column_names.index('sensor_sequence_index'),'sensor_sequence_index',pa.array(range(len(contact)),type=pa.int64()))
    pq.write_table(contact,cp)
    physics=pq.read_table(pp);times=np.asarray(physics['timestamp_ns'].to_pylist());ct=np.asarray(contact['measurement_timestamp_ns'].to_pylist())
    ids=np.searchsorted(ct,times,side='right')-1
    latest=[None if i<0 else int(ct[i]) for i in ids]
    fresh=(ids>=0)&np.concatenate(([True],ids[1:]!=ids[:-1]))
    physics=physics.set_column(physics.column_names.index('contact_measurement_timestamp_ns'),'contact_measurement_timestamp_ns',pa.array(latest,type=pa.int64()))
    physics=physics.set_column(physics.column_names.index('contact_measurement_is_fresh'),'contact_measurement_is_fresh',pa.array(fresh))
    pq.write_table(physics,pp)
    for key in ('contact_sensor','physics'):m['stream_sha256'][key]=file_hash(root/m['streams'][key])
    write_json(root/'manifest.json',m)
    assert validate_raw(root,tools=TOOLS,check_media=False)['contract_valid']
