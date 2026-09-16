import copy
from types import SimpleNamespace
import numpy as np
import pytest
from rambo.recording_v2 import Recorder


def recorder(tmp_path):
    r=Recorder.__new__(Recorder);r.active=True;r.origin_tick=0;r.out=tmp_path
    r.env=SimpleNamespace(_sim_step_counter=20,fallen=[False])
    r.last_rgb={k:np.full((4,4,3),12,np.uint8) for k in ['ego','task_centric']}
    boundary={'simulation_time_ns':40_000_000,'state':{'task.success':[False],'position':[1.,2.,3.]}}
    r.capture_boundary=lambda:boundary
    return r,boundary


def test_terminal_is_owned_copy_before_reset(tmp_path):
    r,b=recorder(tmp_path);r.before_reset([0]);b['state']['position'][0]=999
    r.last_rgb['ego'][:]=0;r.env._sim_step_counter=0
    assert r.terminal['simulation_time_ns']==40_000_000
    assert r.terminal['state']['position'][0]==1
    assert (r.terminal_rgb['ego']==12).all()
    assert not r.active and not r.terminal['partial_interval']


def test_capture_failure_does_not_seal_reset(tmp_path):
    r,_=recorder(tmp_path)
    def fail():raise OSError('camera unavailable')
    r.capture_boundary=fail
    with pytest.raises(OSError):r.before_reset([0])
    assert r.active


def test_partial_terminal_is_explicitly_invalid(tmp_path):
    r,_=recorder(tmp_path);r.env._sim_step_counter=15;r.before_reset([0])
    assert r.terminal['partial_interval']


def test_unknown_pair_evidence_cannot_be_false(tmp_path):
    import json
    from rambo.dataset_v2.validation import _task_contact_diagnostics,file_hash,ContractError
    side=tmp_path/'contact.json'
    rows=[dict(simulation_time_ns=i*20_000_000,valid=False,status='unknown',fl_object=None,body_object=None,reason='pair unavailable') for i in range(2)]
    data=dict(role='diagnostic_only',unavailable='unknown',bool_columns_are_invalid_placeholders=True,rows=rows)
    side.write_text(json.dumps(data))
    m={'extensions':{'task_profile_version':'push-box-v2-2','contact_diagnostics':{'path':'contact.json','sha256':file_hash(side)}}}
    _task_contact_diagnostics(tmp_path,m,1)
    rows[0]['fl_object']=False;side.write_text(json.dumps(data));m['extensions']['contact_diagnostics']['sha256']=file_hash(side)
    with pytest.raises(ContractError,match='unknown'):_task_contact_diagnostics(tmp_path,m,1)
