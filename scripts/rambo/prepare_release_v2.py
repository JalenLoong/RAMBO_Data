"""Build a portable publication tree from an explicit usable-episode allowlist."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from publish_dataset_v2 import describe, safe_path, save


def read(path):
    return json.loads(Path(path).read_text())


def copy_tree(source, destination):
    source = Path(source)
    # Source symlinks are not an implicit license to publish another workspace tree.
    if source.is_symlink() or any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError(f'Symlinks in source tree: {source}')
    shutil.copytree(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    args = parser.parse_args()
    c = read(args.config)
    policy = read(c['publication_policy'])
    if policy['scope'] != 'usable_demonstrations_only':
        raise ValueError('Only explicit usable demonstration release is supported')
    stage = args.stage
    if stage.exists():
        raise ValueError('Refuse to overwrite staged release')
    release = safe_path(c['release'])
    if '/' in release:
        raise ValueError('Release ID must be one path component')
    canonical = Path(c['canonical_root'])
    contract = read(canonical / 'meta/contract.json')
    split = read(canonical / 'meta/split.json')['episodes']
    entries = c['episodes']
    uids = [e['episode_uid'] for e in entries]
    if len(set(uids)) != c['expected_count'] or len(entries) != c['expected_count']:
        raise ValueError('Duplicate or incomplete release allowlist')
    if set(uids) != {e['episode_uid'] for e in contract['episodes']}:
        raise ValueError('Raw allowlist and canonical episode membership differ')
    if {k: len(v) for k, v in split.items()} != c['expected_split']:
        raise ValueError('Wrong split')
    raw_destinations = []
    for e in entries:
        raw = Path(e['raw'])
        summary = read(raw / 'summary.json')
        if (not summary.get('passed') or summary.get('status') != 'success'
                or summary.get('mode') != 'demonstration'
                or not summary.get('terminal_before_reset') or not e['behavior'].get('passed')):
            raise ValueError(f'Ineligible episode: {e["episode_uid"]}')
        if e['episode_uid'] in c['excluded_episode_uids']:
            raise ValueError('Quarantined episode selected')
        dest = safe_path(e['raw_path'])
        if not dest.startswith('raw/v2/' + c['task'] + '/'):
            raise ValueError('Wrong Raw namespace')
        raw_destinations.append(dest)
    if len(set(raw_destinations)) != len(raw_destinations):
        raise ValueError('Raw destination collision')
    acceptance = read(c['acceptance'])
    if not acceptance.get('valid_data_path') or acceptance.get('actual_episodes') != c['expected_count']:
        raise ValueError('Collection acceptance not complete')
    stage.mkdir(parents=True)
    can_rel = f'canonical/lerobot_v2_1/{release}'
    copy_tree(canonical, stage / can_rel)
    rel = stage / 'releases' / release
    rel.mkdir(parents=True)
    records = []
    for e in entries:
        copy_tree(e['raw'], stage / e['raw_path'])
        record = {k: v for k, v in e.items() if k != 'raw'}
        record['source_local_raw_root'] = e['raw']
        record['split'] = next(k for k, v in split.items() if e['episode_uid'] in v)
        records.append(record)
        save(rel / 'evidence' / f'{e["episode_uid"]}-behavior.json', e['behavior'])
    save(rel / 'evidence/collection-acceptance.json', acceptance)
    for item in c['evidence_files']:
        dest = rel / 'evidence' / safe_path(item['name'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item['source'], dest)
    distributions = {key: dict(Counter(e['scenario'][key] for e in entries))
                     for key in ['asset', 'yaw_deg', 'motion', 'color']}
    save(rel / 'distribution.json', distributions)
    manifest = dict(work_id=c['work_id'], release=release, repo_id=policy['repo_id'],
        dataset_schema_version=contract['dataset_schema_version'], canonical_root=can_rel,
        scope='usable_demonstrations_only', episodes=records, split=split,
        source_local_canonical_root=str(canonical), license='unknown',
        demonstration_source='scripted high-level expert with the original RAMBO controller',
        exclusion_policy='Pilots, failures and quarantined attempts remain local; no such payloads are published.',
        model_cache='excluded', observer='Raw/review only; not a model input',
        timestamp_policy='Inherit authoritative simulation_time_ns and pre-reset terminal timestamps',
        source_implementation=c['source_implementation'])
    save(rel / 'manifest.json', manifest)
    review = rel / 'review'
    review.mkdir()
    # No absolute local paths in replay URLs. The manifest preserves historical provenance.
    replay = [dict(uid=e['episode_uid'], raw='../../../' + e['raw_path'],
                   split=e['split'], scenario=e['scenario'], behavior=e['behavior']) for e in records]
    save(review / 'episodes.json', replay)
    (review / 'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>QLM-Bench · Push Box</title>
<style>body{font:16px/1.5 system-ui;background:#eef3f6;color:#17313e;margin:24px}main{max-width:1440px;margin:auto}section{background:white;padding:22px;border-radius:12px;margin-bottom:20px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%}select,button{padding:10px;font:inherit}a{color:#086d84}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<main><section><h1>QLM-Bench · Push Box · 50 usable demonstrations</h1>
<p>Scripted demonstrations, Go2 FL manipulation, native9. Full box footprint in goal and robot upright for 3 policy ticks. Contact is diagnostic / unknown. License pending.</p>
<p>Ego/task: 50Hz. Observer: 25Hz, review only. Select an episode and play the three views.</p>
<select id="pick"></select> <button id="play">Play all from start</button> <button id="pause">Pause</button><p id="details"></p>
<div class="grid"><div>Ego<video controls preload="metadata"></video></div><div>Task<video controls preload="metadata"></video></div><div>Observer<video controls preload="metadata"></video></div></div>
<p><a href="../manifest.json">Manifest / source mapping</a> · <a href="../distribution.json">Distribution</a> · <a href="../evidence/collection-acceptance.json">Acceptance</a></p>
</section></main><script>
const pick=document.querySelector('#pick'),videos=[...document.querySelectorAll('video')];let rows=[];
function show(){const e=rows.find(e=>e.uid===pick.value);videos.forEach((v,i)=>{v.src=e.raw+'/cameras/'+['ego','task_centric','observer'][i]+'.mp4';v.load()});document.querySelector('#details').textContent=e.uid+' · '+e.scenario.asset+' · '+e.scenario.yaw_deg+'° · '+e.scenario.motion+' · '+e.split+' · '+e.behavior.duration_s+'s';}
fetch('episodes.json').then(r=>r.json()).then(r=>{rows=r;for(const e of rows){let o=document.createElement('option');o.value=e.uid;o.textContent=e.uid;pick.append(o)}show()});pick.onchange=show;
document.querySelector('#play').onclick=()=>videos.forEach(v=>{v.currentTime=0;v.play()});document.querySelector('#pause').onclick=()=>videos.forEach(v=>v.pause());
</script></html>''')
    files = {str(p.relative_to(stage)): describe(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    # Inventory cannot hash itself; checksums covers payload + manifest, and external inventory covers checksums.
    save(rel / 'checksums.json', {p: d['sha256'] for p, d in files.items()})
    files[str((rel / 'checksums.json').relative_to(stage))] = describe(rel / 'checksums.json')
    save(args.inventory, dict(work_id=c['work_id'], release=release, repo_id=policy['repo_id'], files=files))
    print(json.dumps(dict(episodes=len(records), files=len(files), bytes=sum(f['size'] for f in files.values()))))


if __name__ == '__main__':
    main()
