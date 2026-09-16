"""Merge validated episodes without re-encoding; preserve episode boundaries/splits."""
import argparse,copy,json,shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from rambo.dataset_v2.validation import validate_canonical,PROFILE,LOCK,VERSION,file_hash
from rambo.dataset_v2.media import MediaTools


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def merge(entries,destination,tools,work_id):
    destination=Path(destination)
    if destination.exists():raise ValueError('Refuse existing collection')
    infos=[];contracts=[]
    for entry in entries:
        root=Path(entry['canonical']);validate_canonical(root,tools=tools)
        c=json.loads((root/'meta/contract.json').read_text());assert len(c['episodes'])==1 and c['episodes'][0]['status']=='success'
        contracts.append(c);infos.append(json.loads((root/'meta/info.json').read_text()))
    assert entries and all(info['features']==infos[0]['features'] for info in infos)
    assert len({c['episodes'][0]['episode_uid'] for c in contracts})==len(entries)
    destination.mkdir(parents=True);info=copy.deepcopy(infos[0]);eps=[];episode_rows=[];stats_rows=[];metadata={};lineage={};offset=0;split_uids={}
    def copy_ref(root,ref,path):
        dst=destination/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/ref['path'],dst)
        assert file_hash(dst)==ref['sha256'];ref['path']=path
    for i,(entry,contract,source_info) in enumerate(zip(entries,contracts,infos)):
        root=Path(entry['canonical']);ep=copy.deepcopy(contract['episodes'][0]);uid=ep['episode_uid'];n=ep['length'];ep['episode_index']=i
        split_uids.setdefault(entry['split'],[]).append(uid)
        source_table=root/source_info['data_path'].format(episode_chunk=0,episode_index=0)
        table=pq.read_table(source_table)
        for key,values in [('episode_index',[i]*n),('index',range(offset,offset+n))]:table=table.set_column(table.column_names.index(key),key,pa.array(values,type=pa.int64()))
        dest=destination/info['data_path'].format(episode_chunk=0,episode_index=i);dest.parent.mkdir(parents=True,exist_ok=True);pq.write_table(table,dest)
        for key,ref in ep['videos'].items():copy_ref(root,ref,info['video_path'].format(episode_chunk=0,episode_index=i,video_key=key))
        for key,ref in ep['terminal']['images'].items():copy_ref(root,ref,f'terminal/episode_{i:06d}/{key}.png')
        copy_ref(root,ep['camera_index'],f'meta/camera_index_{i:06d}.parquet')
        ext=copy.deepcopy(contract['extensions'])
        # Single-episode exports have only a transport train alias. The
        # collection's explicit episode assignment is authoritative here.
        if 'split_semantics' in ext:
            ext['source_split_semantics']=ext.pop('split_semantics')
        ext['split_semantics']='whole-episode assignment in meta/split.json'
        ext['collection_split']=entry['split']
        for key in ['contact_diagnostics','provenance','task_review']:
            if key in ext:copy_ref(root,ext[key],f'metadata/{uid}/{Path(ext[key]["path"]).name}')
        metadata[uid]=ext;lineage[uid]={'canonical_root':str(root.resolve()),'inventory_sha256':file_hash(root/'meta/checksums.json')}
        row=json.loads((root/'meta/episodes.jsonl').read_text().strip());row['episode_index']=i;episode_rows.append(row)
        stat=json.loads((root/'meta/episodes_stats.jsonl').read_text().strip());stat['episode_index']=i
        for key in ['min','max','mean']:stat['stats']['index'][key]=[v+offset for v in stat['stats']['index'][key]];stat['stats']['episode_index'][key]=[i]
        stats_rows.append(stat);eps.append(ep);offset+=n
    assert all(ep['task_text']==eps[0]['task_text'] for ep in eps),'Single task collection only'
    # Contiguous, whole-episode splits, explicit even when diagnostics use HF train alias.
    splits={};start=0
    for split,uids in split_uids.items():
        selected=[j for j,entry in enumerate(entries) if entry['split']==split]
        assert selected==list(range(start,start+len(uids))),'Split groups must be contiguous'
        splits[split]=f'{start}:{start+len(uids)}';start+=len(uids)
    info.update(total_episodes=len(eps),total_frames=offset,total_videos=2*len(eps),total_chunks=1,splits=splits)
    write(destination/'meta/info.json',info)
    for name,rows in [('episodes.jsonl',episode_rows),('episodes_stats.jsonl',stats_rows),('tasks.jsonl',[dict(task_index=0,task=eps[0]['task_text'])])]:
        (destination/'meta'/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    write(destination/'meta/split.json',{'unit':'episode','episodes':split_uids,'work_id':work_id,'source':'predeclared collection scenarios; never frame/window split'})
    write(destination/'meta/contract.json',dict(dataset_schema_version=VERSION,profile_hash=LOCK['profile_hash'],columns_hash=LOCK['columns_hash'],transform_chain_hash=PROFILE['transform_chain_hash'],diagnostic_only=any(c['diagnostic_only'] for c in contracts),episodes=eps,extensions={'task_profile_version':'push-box-v2-2','episode_metadata':metadata,'source_datasets':lineage,'work_id':work_id,'split_manifest':'meta/split.json','video_copy':'byte-for-byte; no extra encoding generation'}))
    write(destination/'meta/checksums.json',{str(f.relative_to(destination)):file_hash(f) for f in sorted(destination.rglob('*')) if f.is_file() and f.name!='checksums.json'})
    result=validate_canonical(destination,tools=tools);write(destination.parent/(destination.name+'-validation.json'),result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entries',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--ffmpeg',required=True);p.add_argument('--ffprobe',required=True);p.add_argument('--work-id',default='DATA-005');args=p.parse_args()
    print(json.dumps(merge(json.loads(args.entries.read_text()),args.output,MediaTools(args.ffmpeg,args.ffprobe),args.work_id)),flush=True)
