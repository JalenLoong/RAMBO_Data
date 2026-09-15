#!/usr/bin/env python3
"""Run RAMBO and isolated WAM validators against explicit synthetic positive/negative cases."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/rambo'))
from rambo.contracts_v2.validation import ContractError,validate_episode,read_json,read_jsonl,file_hash,SPEC
from rambo.contracts_v2.fixtures import write_episode,write_json,write_jsonl,checksum_episode
import numpy as np

CASES=[('golden',None),('synthetic_release','release'),('missing_rgb','timing'),('timestamp','timing'),
       ('camera_identity','profile'),('camera_pose','camera'),('epoch','epoch'),('units_profile','profile'),
       ('unknown_core','schema'),('missing_provenance','schema'),('dtype','array'),('object_array','array'),
       ('force_execution','execution'),('command_gap','execution'),('held_value','execution'),
       ('missing_reason','telemetry'),('inferred_contact','telemetry'),('outcome_mismatch','telemetry'),
       ('terminal_epoch','epoch'),('checksum','checksum'),('unsafe_path','path'),
       ('external_assistance','release'),('missing_runtime_evidence','release')]


def build_case(base,out,name):
    shutil.copytree(base,out);ep=read_json(out/'episode.json');frames=read_jsonl(out/'rgb.jsonl')
    commands=read_jsonl(out/'commands.jsonl');trace=read_jsonl(out/'control.jsonl')
    if name=='missing_rgb':frames.pop(4)
    if name=='timestamp':frames[2]['timestamp_ns']+=1
    if name=='camera_identity':frames[0]['profile_hash']=frames[1]['profile_hash']
    if name=='camera_pose':frames[0]['world_position'][0]+=.1
    if name=='epoch':frames[0]['reset_epoch']+=1
    if name=='units_profile':ep['profiles']['action']='0'*64
    if name=='unknown_core':ep['unknown_core']=True
    if name=='missing_provenance':del ep['provenance']['checkpoint_sha256']
    if name in ('dtype','object_array','missing_reason'):
        with np.load(out/'robot.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
        if name=='dtype':a['joint_position']=a['joint_position'].astype(np.float64)
        if name=='object_array':a['joint_position']=a['joint_position'].astype(object)
        if name=='missing_reason':a['fl_position_valid'][0]=False
        np.savez(out/'robot.npz',**a)
    if name=='force_execution':commands[0]['executed'][6]=1
    if name=='command_gap':commands.pop(2)
    if name=='held_value':trace[0]['held'][0]+=.1
    if name=='inferred_contact':
        cfg=read_json(out/'task_config.json');cfg['contact_pairs']['fl']['method']='proximity_inference';write_json(out/'task_config.json',cfg);ep['provenance']['task_config']['sha256']=file_hash(out/'task_config.json')
    if name=='outcome_mismatch':ep['outcome']['success']=True;ep['outcome']['success_tick']=320
    if name=='terminal_epoch':ep['outcome']['snapshot_epoch']+=1
    if name=='unsafe_path':ep['files']['commands']='../commands.jsonl'
    if name in ('external_assistance','missing_runtime_evidence'):
        # Negative gate probes, not claimed runtime results or published datasets.
        ep['diagnostic_only']=False;ep['outcome'].update(success=True,success_tick=320,truncated=False)
        with np.load(out/'task.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
        a['success'][-1]=True;np.savez(out/'task.npz',**a)
        if name=='external_assistance':
            write_jsonl(out/'interventions.jsonl',[dict(kind='external_wrench',source='synthetic_negative_test',target_body='FL_foot',frame='body',application_point='body_com',point_m=None,force_n=[1,0,0],torque_nm=[0,0,0],start_tick=0,end_tick=10,state_change=None,extensions={})])
    write_json(out/'episode.json',ep);write_jsonl(out/'rgb.jsonl',frames);write_jsonl(out/'commands.jsonl',commands);write_jsonl(out/'control.jsonl',trace);checksum_episode(out)
    if name=='checksum':(out/'commands.jsonl').write_text('corrupted')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--consumer-python',required=True);p.add_argument('--consumer-root',required=True,type=Path);args=p.parse_args()
    if args.output.exists():raise SystemExit('Refuse existing output directory')
    args.output.mkdir(parents=True);base=write_episode(args.output/'source_fixture');results=[]
    producer_files={f.name:file_hash(f) for f in SPEC.glob('*.json')}
    consumer_spec=args.consumer_root/'src/wam_policy/contracts_v2/spec'
    consumer_files={f.name:file_hash(f) for f in consumer_spec.glob('*.json')}
    if producer_files!=consumer_files:raise SystemExit('Specification copies differ')
    for name,expected in CASES:
        directory=args.output/name;build_case(base,directory,name)
        release=name in ('synthetic_release','external_assistance','missing_runtime_evidence')
        try:validate_episode(directory,release=release);producer=None
        except ContractError as error:producer=error.code
        command=[args.consumer_python,'-B','-m','wam_policy.contracts_v2',str(directory)]
        if release:command.append('--release-eligible')
        env=dict(os.environ);env['PYTHONPATH']=str(args.consumer_root/'src')
        proc=subprocess.run(command,cwd=args.consumer_root,env=env,text=True,capture_output=True)
        try:
            response=json.loads(proc.stdout);consumer=None if response['passed'] else response['code']
        except (json.JSONDecodeError,KeyError):consumer='unexpected_process_error'
        passed=producer==consumer==expected and proc.returncode==(0 if expected is None else 1)
        results.append(dict(case=name,expected_code=expected,producer_code=producer,consumer_code=consumer,passed=passed,consumer_exit=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
        print(f'{name}: {"passed" if passed else "FAILED"}',flush=True)
    report=dict(diagnostic_only=True,dataset_release=False,spec_files=producer_files,cases=results,passed=all(x['passed'] for x in results),runtime_conformance='not_run')
    write_json(args.output/'report.json',report)
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
