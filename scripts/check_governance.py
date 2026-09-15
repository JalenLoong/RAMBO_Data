#!/usr/bin/env python3
"""Validate the shared v2 registry, document lifecycle and v2-only commit titles."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import yaml

WORK = re.compile(r'([A-Z]+)-(\d{3})$')
ADR = re.compile(r'ADR-\d{4}$')
LEGACY = re.compile(r'V2-(?:BOOTSTRAP|CLEANUP|ADAPTATION|CONTRACTS|DATA-CONTRACT)\b')

def metadata(path):
    text = path.read_text(encoding='utf-8')
    if not text.startswith('---\n'):
        raise ValueError(f'{path.name}: missing frontmatter')
    return yaml.safe_load(text.split('---', 2)[1])

def registry_errors(registry):
    errors = []
    if registry.get('namespace') != 'adaptation-v2' or registry.get('authority_repository') != 'WAM-Policy':
        errors.append('wrong registry namespace/authority')
    ids, aliases, counts = set(), set(), {}
    for work in registry['works']:
        uid = work['id']; match = WORK.fullmatch(uid)
        if not match or work['category'] != match[1] or match[1] not in registry['categories']:
            errors.append(f'invalid Work ID/category: {uid}'); continue
        if uid in ids: errors.append(f'duplicate Work ID: {uid}')
        ids.add(uid); counts.setdefault(match[1], []).append(int(match[2]))
        if work['status'] not in ('active', 'completed', 'abandoned'):
            errors.append(f'invalid work status: {uid}')
        if not work['repositories'] or not set(work['repositories']) <= {'WAM-Policy','RAMBO_Data'}:
            errors.append(f'invalid participants: {uid}')
        for alias in work.get('aliases', []):
            if alias in aliases or alias in ids: errors.append(f'duplicate alias: {alias}')
            aliases.add(alias)
        place = 'active' if work['status'] == 'active' else 'archive'
        for field,tree,file in [('change','changes','change-spec.md'),('plan','work','plan.md')]:
            if work[field] != f'docs/{tree}/{place}/{uid}/{file}':
                errors.append(f'invalid {field} path: {uid}')
    for category,values in counts.items():
        if sorted(values) != list(range(1, max(values)+1)):
            errors.append(f'noncontiguous category numbering: {category}')
    return errors

def render_index(registry, repository):
    lines = ['---','id: WORK-INDEX','type: index','status: accepted','source_map: []','---',
             '# v2 work index','', 'Generated from the shared registry; v1 numbering is independent.','',
             '| Work ID | Status | ChangeSpec | ExecPlan |','|---|---|---|---|']
    for work in registry['works']:
        if repository not in work['repositories']: continue
        links = [f'[{label}](../{work[field].removeprefix("docs/")})' for field,label in [('change','spec'),('plan','plan')]]
        lines.append(f'| {work["id"]} | {work["status"]} | {links[0]} | {links[1]} |')
    return '\n'.join(lines)+'\n'

def legacy_current_errors(text):
    # Immutable historical paths and explicitly labelled historical/alias lines are allowed.
    errors=[]
    for line in text.splitlines():
        cleaned=re.sub(r'runs/[^\s`\)\]"\'<>]+','',line)
        if LEGACY.search(cleaned) and not re.search(r'\b(?:historical|alias|original)\b|历史|别名', cleaned, re.I):
            errors.append('legacy Work ID used as current identity: '+cleaned.strip())
    return errors

def check(root, peer=None, history=True):
    root=Path(root); errors=[]
    try:
        directory=root/'docs/governance'
        registry=json.loads((directory/'work-registry.json').read_text())
        config=json.loads((directory/'repository.json').read_text())
        errors.extend(registry_errors(registry)); repo=config['repository']
        source=json.loads((directory/'source.json').read_text())
        if hashlib.sha256((directory/'documentation.md').read_bytes()).hexdigest()!=source['sha256']:
            errors.append('original governance source hash mismatch')
        if peer is not None:
            for file in ['documentation.md','work-registry.json','source.json','migration.json']:
                if (directory/file).read_bytes()!=(Path(peer)/'docs/governance'/file).read_bytes():
                    errors.append('peer declaration mismatch: '+file)
        works={w['id']:w for w in registry['works']}
        expected={w['id'] for w in registry['works'] if repo in w['repositories']}
        for field,tree,typ in [('change','changes','change'),('plan','work','exec-plan')]:
            observed=[]
            for path in (root/'docs'/tree).glob('*/*/*.md'):
                meta=metadata(path)
                if meta.get('type')!=typ: continue
                uid=meta['id'];observed.append(uid)
                if uid not in expected:errors.append(f'unregistered {typ}: {uid}');continue
                work=works[uid]
                if path!=root/work[field] or meta['status']!=work['status']:
                    errors.append(f'{typ} path/status mismatch: {uid}')
            if set(observed)!=expected or len(observed)!=len(expected):
                errors.append(f'{typ} pairing/registry mismatch')
        for path in (root/'docs/decisions').glob('*.md'):
            meta=metadata(path)
            if not ADR.fullmatch(meta['id']) or path.stem!=meta['id']:
                errors.append('invalid ADR identity: '+path.name)
            if not meta.get('related_work') or not set(meta['related_work'])<=set(works):
                errors.append('invalid ADR related_work: '+path.name)
        index=root/'docs/work/INDEX.md'
        if not index.exists() or index.read_text()!=render_index(registry,repo):errors.append('stale generated work index')
        for path in [root/'AGENTS.md',root/'README.md',*list((root/'docs/current').rglob('*.md'))]:
            errors.extend(f'{path.name}: {e}' for e in legacy_current_errors(path.read_text()))
        for file in ['change-spec.md','exec-plan.md','adr.md','pull-request.md','run-manifest.yaml']:
            if '<WORK-ID>' not in (root/'docs/templates'/file).read_text():errors.append('template missing Work ID: '+file)
        if history:
            log=subprocess.check_output(['git','-C',str(root),'log',config['history_base']+'..HEAD','--format=%s'],text=True)
            for title in log.splitlines():
                prefix=re.match(r'^(?:\[[A-Z]+-\d{3}\])+ ',title)
                ids=re.findall(r'\[([A-Z]+-\d{3})\]',prefix[0] if prefix else '')
                if not ids or not set(ids)<=set(works):errors.append('invalid v2 commit title: '+title)
    except (ValueError,KeyError,OSError,subprocess.CalledProcessError,yaml.YAMLError) as exc:
        errors.append(str(exc))
    return errors

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--peer-root',type=Path)
    parser.add_argument('--write-index',action='store_true')
    args=parser.parse_args()
    if args.write_index:
        directory=args.root/'docs/governance';registry=json.loads((directory/'work-registry.json').read_text());repo=json.loads((directory/'repository.json').read_text())['repository']
        (args.root/'docs/work/INDEX.md').write_text(render_index(registry,repo))
    errors=check(args.root,args.peer_root)
    if errors:print('\n'.join(errors));return 1
    print('v2 governance: OK');return 0

if __name__=='__main__':raise SystemExit(main())
