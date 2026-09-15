#!/usr/bin/env python3
"""Two isolated namespace processes validate the same synthetic LeRobot artifacts."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/rambo'))
from rambo.dataset_v2.fixtures import write_json,checksums
from rambo.dataset_v2.validation import read_json,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--python',required=True);p.add_argument('--consumer-root',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Refuse existing output')
    a.output.mkdir(parents=True)
    producer=ROOT/'source/rambo/rambo/dataset_v2/spec';consumer=a.consumer_root/'src/wam_policy/dataset_v2/spec'
    hashes=lambda path:{f.name:file_hash(f) for f in path.glob('*.json')}
    if hashes(producer)!=hashes(consumer):raise SystemExit('spec copies differ')
    cases=[('valid16',None),('valid64',None),('valid65',None),('legacy_version','schema'),('observer','observer'),('terminal_time','terminal'),('transform','transform'),('action_config','action_config'),('bad_encoding','media'),('bad_checksum','checksum')]
    results=[]
    for name,expected in cases:
        n=int(name[5:]) if name.startswith('valid') else 16
        source=a.fixtures/f'n{n}'
        root=a.output/name;shutil.copytree(source/'canonical',root)
        m=read_json(root/'meta/contract.json');info=read_json(root/'meta/info.json')
        if name=='legacy_version':m['dataset_schema_version']='2.0.0'
        if name=='observer':info['features']['observation.images.observer']=info['features']['observation.images.ego']
        if name=='terminal_time':m['episodes'][0]['terminal']['simulation_time_ns']+=1
        if name=='transform':m['transform_chain_hash']='0'*64
        if name=='bad_encoding':m['episodes'][0]['videos']['observation.images.ego']['encoding']['crf']=23
        if name=='action_config':
            rows=[json.loads(x) for x in (root/'meta/episodes.jsonl').read_text().splitlines()]
            rows[0]['action_config'][0]['start_frame']=1
            (root/'meta/episodes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        write_json(root/'meta/contract.json',m);write_json(root/'meta/info.json',info);checksums(root)
        if name=='bad_checksum':(root/'meta/tasks.jsonl').write_text('corrupt')
        cfg=read_json(source/'paths.json');cfg['canonical_root']=str(root.resolve());config=a.output/f'{name}-paths.json';write_json(config,cfg)
        observations={}
        for label,module,path in [('RAMBO_Data','rambo',ROOT/'source/rambo'),('WAM-Policy','wam_policy',a.consumer_root/'src')]:
            env=dict(os.environ);env['PYTHONPATH']=str(path)
            command=[a.python,'-B','-m',module+'.dataset_v2','--config',str(config)]
            run=subprocess.run(command,env=env,capture_output=True,text=True)
            try:body=json.loads(run.stdout);code=None if body['passed'] else body['code']
            except (json.JSONDecodeError,KeyError):code='unexpected_process_error'
            observations[label]={'code':code,'exit':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'command':command}
        passed=all(v['code']==expected and v['exit']==(0 if expected is None else 1) for v in observations.values())
        results.append({'case':name,'expected':expected,'passed':passed,'processes':observations})
        print(f'{name}: {"passed" if passed else "FAILED"}',flush=True)
    report={'diagnostic_only':True,'passed':all(x['passed'] for x in results),'cases':results,'spec_hashes':hashes(producer),'real_runtime':'not_run','published':False}
    write_json(a.output/'report.json',report)
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
