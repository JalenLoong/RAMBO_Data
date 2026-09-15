"""CPU contract tests. Fixtures and backend confirmations are explicitly synthetic."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from rambo.contracts_v2 import ContractError, validate_episode
from rambo.contracts_v2.validation import BUNDLE, PROFILE_HASHES, read_json, read_jsonl, validate_record
from rambo.contracts_v2.fixtures import write_episode, write_json, write_jsonl, checksum_episode, message, FakeBackend
from rambo.contracts_v2.runtime import prepare_command, CommandLedger, SynchronousSession


@pytest.fixture(scope='module')
def golden(tmp_path_factory):return write_episode(tmp_path_factory.mktemp('v2')/'golden')

@pytest.fixture
def episode(golden,tmp_path):
    out=tmp_path/'episode';shutil.copytree(golden,out);return out


def test_golden_counts_and_release_separation(golden):
    report=validate_episode(golden)
    assert (report['rgb_per_camera'],report['completed_commands'],report['control_steps'],report['physics_ticks'])==(9,32,64,320)
    assert report['diagnostic_only'] and not report['release_eligible']
    with pytest.raises(ContractError,match='release'):validate_episode(golden,release=True)


@pytest.mark.parametrize('case,code',[
 ('missing_rgb','timing'),('duplicate_rgb','timing'),('wrong_timestamp','timing'),('swapped_camera','profile'),
 ('wrong_epoch','epoch'),('bad_intrinsic','camera'),('unknown_field','schema'),('missing_provenance','schema'),
 ('wrong_version','schema'),('wrong_profile','profile'),('cross_reset_terminal','epoch'),('force_label','execution'),
 ('changed_transform','action'),('held_mismatch','execution'),('future_consumer','execution'),('future_ik','timing'),
 ('command_gap','execution'),('trace_gap','timing'),('duplicate_frame_id','camera'),('missing_validity_reason','telemetry'),
 ('wrong_dtype','array'),('object_array','array'),('wrong_shape','array'),('corrupt_image','camera'),('checksum','checksum')])
def test_negative_episodes(episode,case,code):
    ep=read_json(episode/'episode.json');frames=read_jsonl(episode/'rgb.jsonl');cmd=read_jsonl(episode/'commands.jsonl');trace=read_jsonl(episode/'control.jsonl')
    if case=='missing_rgb':frames.pop(4)
    elif case=='duplicate_rgb':
        duplicate=copy.deepcopy(frames[2]);duplicate['sensor_frame_id']=99
        duplicate['path']='rgb/go2_ego/duplicate.png'
        shutil.copyfile(episode/frames[2]['path'],episode/duplicate['path'])
        frames.insert(4,duplicate)
    elif case=='wrong_timestamp':frames[2]['timestamp_ns']+=1
    elif case=='swapped_camera':frames[0]['camera_id']='d435i_rgb_task'
    elif case=='wrong_epoch':frames[0]['reset_epoch']+=1
    elif case=='bad_intrinsic':frames[0]['intrinsics'][0][0]+=1
    elif case=='unknown_field':ep['extra']=1
    elif case=='missing_provenance':del ep['provenance']['checkpoint_sha256']
    elif case=='wrong_version':ep['schema_version']='2.1.0'
    elif case=='wrong_profile':ep['profiles']['timing']='0'*64
    elif case=='cross_reset_terminal':ep['outcome']['snapshot_epoch']=1
    elif case=='force_label':cmd[0]['executed'][6]=1
    elif case=='changed_transform':cmd[0]['transformed'][0]+=1
    elif case=='held_mismatch':trace[0]['held'][0]+=1
    elif case=='future_consumer':trace[0]['fl_ik_command_id']=cmd[1]['command_id']
    elif case=='future_ik':trace[0]['fl_ik_computed_tick']=5
    elif case=='command_gap':cmd.pop(2)
    elif case=='trace_gap':trace.pop(2)
    elif case=='duplicate_frame_id':frames[2]['sensor_frame_id']=frames[0]['sensor_frame_id']
    elif case in ('missing_validity_reason','wrong_dtype','object_array','wrong_shape'):
        with np.load(episode/'robot.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
        if case=='missing_validity_reason':a['fl_position_valid'][0]=False
        if case=='wrong_dtype':a['joint_position']=a['joint_position'].astype(np.float64)
        if case=='object_array':a['joint_position']=a['joint_position'].astype(object)
        if case=='wrong_shape':a['joint_position']=a['joint_position'][:,:11]
        np.savez(episode/'robot.npz',**a)
    elif case=='corrupt_image':(episode/frames[0]['path']).write_bytes(b'not png');frames[0]['sha256']=__import__('hashlib').sha256(b'not png').hexdigest()
    write_json(episode/'episode.json',ep);write_jsonl(episode/'rgb.jsonl',frames);write_jsonl(episode/'commands.jsonl',cmd);write_jsonl(episode/'control.jsonl',trace)
    checksum_episode(episode)
    if case=='checksum':(episode/'commands.jsonl').write_text('changed')
    with pytest.raises(ContractError) as e:validate_episode(episode)
    assert e.value.code==code


def test_unknown_measurement_requires_reason_and_is_not_zero_evidence(episode):
    ep=read_json(episode/'episode.json')
    with np.load(episode/'contact.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
    a['friction_force_valid'][:]=False
    np.savez(episode/'contact.npz',**a)
    ep['files']['contact']['missing_reasons']['friction_force']='sensor channel not enabled'
    write_json(episode/'episode.json',ep);checksum_episode(episode)
    assert validate_episode(episode)['contract_valid']


def test_force_preparation_is_atomic_and_does_not_clip():
    raw=np.array([12.,-9,4,.1934,.142,.05,1,-2,3]);original=raw.copy()
    row=prepare_command(raw,'one',0)
    np.testing.assert_array_equal(raw,original)
    assert row['requested']==raw.tolist()
    assert row['transformed'][:6]==raw[:6].astype(np.float32).astype(float).tolist()
    assert row['transformed'][6:]==[0,0,0] and row['executed'] is None
    for bad in ([0]*8,[0]*8+[float('nan')],[0]*8+[float('inf')],[0]*8+[True]):
        with pytest.raises(ContractError):prepare_command(bad,'one',0)
    assert row['executed'] is None


def test_partial_and_full_confirmation():
    ledger=CommandLedger([[0,0,0,.1934,.142,.05,1,0,0]]*16,'chunk',0);backend=FakeBackend()
    assert ledger.acknowledgement('accepted')['completed_count']==0
    ledger.confirm(backend.step(ledger.next_step()))
    assert ledger.acknowledgement('stopped')['commands'][0]['status']=='partial'
    assert ledger.commands[0]['executed'] is None
    ledger.confirm(backend.step(ledger.next_step()))
    assert ledger.commands[0]['executed'][6:]==[0,0,0]
    assert ledger.acknowledgement('in_progress')['completed_count']==1
    assert len(backend.calls)==2


def setup_session(golden,resetters=()):
    s=SynchronousSession(cache_resetters=resetters)
    task=read_json(golden/'episode.json')['task']
    reset=message('ResetRequest',dict(timeout_s=5,task=task,origin_sim_ns=0),message_id='reset')
    ready=s.handle(reset,now=0)
    frames=read_jsonl(golden/'rgb.jsonl')
    obs=message('Observation',dict(tick=0,rgb=frames[:2],executed_history=[],instruction=task['instruction']),message_id='obs0')
    s.handle(obs,now=0)
    chunk=message('ActionChunk',dict(observation_id='obs0',chunk_id='chunk0',start_tick=0,action_period_ticks=10,actions=[[0,0,0,.1934,.142,.05,2,0,0]]*32),message_id='action0')
    return s,reset,ready,obs,chunk,frames


def test_protocol_complete_idempotence_and_history(golden):
    s,reset,ready,obs,chunk,frames=setup_session(golden);backend=FakeBackend()
    accepted=s.handle(chunk,now=1)
    assert s.history==[] and accepted['type']=='ActionAccepted'
    assert s.handle(chunk,now=1)==accepted
    for _ in range(64):result=s.step(backend)
    assert result['type']=='ExecutionAck' and result['body']['completed_count']==32
    assert s.tick==320 and len(backend.calls)==64 and len(s.history)==32
    assert backend.calls[2]['residual_observation_command_id']=='chunk0:0'
    assert s.handle(chunk,now=2)==accepted and len(backend.calls)==64
    obs2=copy.deepcopy(obs);obs2['message_id']='obs1';obs2['body'].update(tick=320,rgb=frames[2:],executed_history=copy.deepcopy(s.history))
    s.handle(obs2,now=2)
    assert s.state=='WAITING' and s.tick==320


@pytest.mark.parametrize('case,code',[('conflicting_id','duplicate'),('stale_observation','observation'),('wrong_epoch','epoch'),('overlap','timing'),('bad_length','action'),('nonfinite','json'),('unknown_field','schema')])
def test_bad_protocol_messages(golden,case,code):
    s,_,_,_,chunk,_=setup_session(golden)
    if case=='conflicting_id':s.handle(chunk,now=0);chunk['body']['actions'][0]=[1]*9
    if case=='stale_observation':chunk['body']['observation_id']='old'
    if case=='wrong_epoch':chunk['reset_epoch']=3
    if case=='overlap':chunk['body']['start_tick']=10
    if case=='bad_length':chunk['body']['actions']=chunk['body']['actions'][:17]
    if case=='nonfinite':chunk['body']['actions'][0]=[float('nan')]*9
    if case=='unknown_field':chunk['body']['predicted_history']=[]
    with pytest.raises(ContractError) as e:s.handle(chunk,now=1)
    assert e.value.code==code and s.tick==0 and s.history==[]


def test_timeout_freezes_time(golden):
    s,*_=setup_session(golden)
    assert s.poll(now=4) is None and s.tick==0
    assert s.poll(now=5)['body']['code']=='timeout' and s.tick==0 and s.state=='ERROR'
    with pytest.raises(ContractError):s.step(FakeBackend())


def test_partial_stop_and_reset_cache_isolation(golden):
    caches=[['old'],['old'],['old'],['old']]
    s,reset,_,obs,chunk,_=setup_session(golden,[c.clear for c in caches])
    assert not any(caches)
    s.handle(chunk,now=1);s.step(FakeBackend());ack=s.finish(reason='terminal')
    assert ack['body']['end_tick']==5 and ack['body']['completed_count']==0 and not s.history
    for c in caches:c.append('stale')
    reset['message_id']='reset1';reset['reset_epoch']=1;reset['episode_id']='next'
    s.handle(reset,now=2)
    assert s.tick==0 and s.pending is None and not s.history and not any(caches)
    with pytest.raises(ContractError):s.handle(chunk,now=3)


def test_predicted_history_is_rejected(golden):
    s,reset,_,obs,chunk,frames=setup_session(golden)
    s.handle(chunk,now=1);backend=FakeBackend()
    for _ in range(64):s.step(backend)
    obs['message_id']='future';obs['body'].update(tick=320,rgb=frames[2:],executed_history=[])
    with pytest.raises(ContractError,match='history'):s.handle(obs,now=2)


def test_camera_profile_matches_retained_geometry():
    from rambo.tasks.common.lift_camera_rig import CAMERAS as source
    from rambo.collection import RAMBO_QUADRUPED_COMMAND_NAMES,RAMBO_QUADRUPED_COMMAND_UNITS,RAMBO_QUADRUPED_COMMAND_FRAMES
    from rambo.utils.articulation import GO2_JOINT_ORDER,GO2_FOOT_BODY_NAMES
    assert tuple(BUNDLE['action']['names'])==RAMBO_QUADRUPED_COMMAND_NAMES
    assert tuple(BUNDLE['action']['units'])==RAMBO_QUADRUPED_COMMAND_UNITS
    assert tuple(BUNDLE['action']['frames'])==RAMBO_QUADRUPED_COMMAND_FRAMES
    assert tuple(BUNDLE['telemetry']['joint_names'])==GO2_JOINT_ORDER
    assert tuple(BUNDLE['telemetry']['foot_names'])==GO2_FOOT_BODY_NAMES
    for old,k in [('ego','go2_ego'),('task','d435i_rgb_task')]:
        c=source[old];p=BUNDLE[k]
        np.testing.assert_allclose(p['mount_position_m'],c['position'])
        np.testing.assert_allclose(p['mount_quaternion_xyzw'],c['rotation_xyzw'])
        assert p['width']==c['width'] and p['height']==c['height']
        assert p['intrinsics'][0][0]==pytest.approx(c['width']*c['focal_length_mm']/c['horizontal_aperture_mm'])


def test_backend_failure_never_fabricates_execution(golden):
    s,_,_,_,chunk,_=setup_session(golden);s.handle(chunk,now=1)
    class Broken:
        def step(self,schedule):raise RuntimeError('synthetic backend failed before confirmation')
    with pytest.raises(ContractError,match='backend'):s.step(Broken())
    assert s.state=='ERROR' and s.tick==0 and not s.history
    ack=s.finish(reason='backend_error');assert ack['body']['completed_count']==0


def test_reset_retry_does_not_reset_twice(golden):
    calls=[];s,reset,_,_,chunk,_=setup_session(golden,[lambda:calls.append('reset')])
    s.handle(chunk,now=1);s.step(FakeBackend())
    assert len(calls)==1
    assert s.handle(reset,now=2)['type']=='Ready'
    assert len(calls)==1 and s.tick==5


def test_confirmation_must_match_exact_step():
    ledger=CommandLedger([[0]*9]*16,'x',0);backend=FakeBackend()
    row=backend.step(ledger.next_step());row['end_tick']=10
    with pytest.raises(ContractError):ledger.confirm(row)
    assert ledger.tick==0 and not ledger.trace and ledger.commands[0]['executed'] is None


def test_object_sensor_proxy_is_not_confirmed_contact(episode):
    ep=read_json(episode/'episode.json');cfg=read_json(episode/'task_config.json')
    cfg['contact_pairs']['fl']['method']='proximity_inference'
    write_json(episode/'task_config.json',cfg)
    from rambo.contracts_v2.validation import file_hash
    ep['provenance']['task_config']['sha256']=file_hash(episode/'task_config.json')
    write_json(episode/'episode.json',ep);checksum_episode(episode)
    with pytest.raises(ContractError,match='proximity'):validate_episode(episode)


def test_release_digest_binds_checksum_inventory(episode,tmp_path):
    from rambo.contracts_v2.validation import validate_release,file_hash
    m=dict(schema_version='2.0.0',dataset_id='synthetic-contract-fixtures',episodes=[dict(path='episode/episode.json',sha256=file_hash(episode/'episode.json'),checksums_sha256='0'*64)],extensions={})
    write_json(tmp_path/'release.json',m)
    with pytest.raises(ContractError,match='outer checksum'):validate_release(tmp_path/'release.json')
    m['episodes'][0]['checksums_sha256']=file_hash(episode/'checksums.json');write_json(tmp_path/'release.json',m)
    with pytest.raises(ContractError,match='synthetic/diagnostic'):validate_release(tmp_path/'release.json')


def test_corrupt_npy_header_rejected_before_allocation(episode):
    import io,zipfile
    buffer=io.BytesIO()
    np.lib.format.write_array_header_1_0(buffer,dict(descr='<f4',fortran_order=False,shape=(2**40,)))
    with zipfile.ZipFile(episode/'robot.npz','w') as archive:archive.writestr('joint_position.npy',buffer.getvalue())
    checksum_episode(episode)
    with pytest.raises(ContractError,match='array header/payload'):validate_episode(episode)
