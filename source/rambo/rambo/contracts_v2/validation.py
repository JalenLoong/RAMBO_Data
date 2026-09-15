"""Versioned offline contracts. No simulator, model, or publication side effects.

The WAM consumer maintains its own installed copy of this portable validator;
only declarative specifications/fixtures are exchanged between repositories.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile
import numpy as np
from PIL import Image

SPEC = Path(__file__).with_name('spec')
VERSION = '2.0.0'
CAMERAS = ('go2_ego', 'd435i_rgb_task')


class ContractError(ValueError):
    def __init__(self, code, detail):
        self.code, self.detail = code, str(detail)
        super().__init__(f'{code}: {detail}')


def require(condition, code, detail):
    if not condition:
        raise ContractError(code, detail)


def canonical_bytes(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                          allow_nan=False).encode('utf-8')
    except (ValueError, TypeError, UnicodeError) as error:
        raise ContractError('json', error) from error


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'json', f'duplicate key {key}')
        result[key] = value
    return result


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_pairs,
                           parse_constant=lambda x: (_ for _ in ()).throw(ContractError('json', x)))
        canonical_bytes(value)
        return value
    except (OSError, UnicodeError, ValueError) as error:
        raise ContractError('json', f'{path}: {error}') from error


def read_jsonl(path):
    try:
        lines = Path(path).read_text(encoding='utf-8').splitlines()
        require(all(line.strip() for line in lines), 'json', 'blank JSONL row')
        result = [json.loads(line, object_pairs_hook=_pairs) for line in lines]
        canonical_bytes(result)
        return result
    except (OSError, UnicodeError, ValueError) as error:
        raise ContractError('json', error) from error


def safe_path(root, name):
    require(isinstance(name, str) and name and '\\' not in name, 'path', name)
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts and name == path.as_posix(), 'path', name)
    root = Path(root).resolve()
    target = root / path
    require(all(not (root / Path(*path.parts[:i])).is_symlink() for i in range(1, len(path.parts)+1)), 'path', 'symlink')
    require(target.resolve().is_relative_to(root) and target.is_file(), 'path', name)
    return target


def load_bundle():
    lock = read_json(SPEC / 'lock.json')
    require(lock['version'] == VERSION and lock['authority'] == 'RAMBO_Data', 'profile', 'authority/version')
    for name, sha in lock['files'].items():
        require(file_hash(safe_path(SPEC, name)) == sha, 'profile', f'bundle file {name}')
    data = {name[:-5]: read_json(SPEC / name) for name in lock['files']}
    for name, sha in lock['profile_hashes'].items():
        require(digest(data[name]) == sha, 'profile', name)
    for camera in CAMERAS:
        require(digest(data[camera]) == data['cameras']['cameras'][camera], 'profile', camera)
    return lock, data


LOCK, BUNDLE = load_bundle()
PROFILE_HASHES = LOCK['profile_hashes']


def _check(value, schema, path, definitions):
    if '$ref' in schema:
        return _check(value, definitions[schema['$ref'].split('/')[-1]], path, definitions)
    if 'anyOf' in schema:
        for choice in schema['anyOf']:
            try:
                _check(value, choice, path, definitions)
                return
            except ContractError:
                pass
        raise ContractError('schema', f'{path}: no matching type')
    if 'const' in schema:
        require(type(value) is type(schema['const']) and value == schema['const'], 'schema', path)
    if 'enum' in schema:
        require(any(type(value) is type(x) and value == x for x in schema['enum']), 'schema', path)
    t = schema.get('type')
    types = {'object':dict, 'array':list, 'string':str, 'integer':int, 'boolean':bool, 'null':type(None)}
    if t == 'number':
        require(type(value) in (int, float) and math.isfinite(value), 'schema', path)
    elif t:
        require(type(value) is types[t], 'schema', f'{path}: expected {t}')
    if t in ('number', 'integer'):
        for key, passed in [('minimum', value >= schema.get('minimum', -math.inf)),
                            ('maximum', value <= schema.get('maximum', math.inf)),
                            ('exclusiveMinimum', value > schema.get('exclusiveMinimum', -math.inf))]:
            require(passed, 'schema', f'{path}: {key}')
    if t == 'string':
        require(len(value) >= schema.get('minLength', 0), 'schema', path)
        if 'pattern' in schema:
            require(re.fullmatch(schema['pattern'], value) is not None, 'schema', path)
    if t == 'array':
        require(schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', math.inf), 'schema', path)
        for i, item in enumerate(value):
            _check(item, schema['items'], f'{path}[{i}]', definitions)
    if t == 'object':
        props = schema.get('properties', {})
        require(set(schema.get('required', [])) <= value.keys(), 'schema', f'{path}: missing fields')
        for key, item in value.items():
            sub = props.get(key, schema.get('additionalProperties', True))
            require(sub is not False, 'schema', f'{path}: unknown {key}')
            if isinstance(sub, dict):
                _check(item, sub, f'{path}.{key}', definitions)


def validate_record(kind, value):
    canonical_bytes(value)
    defs = BUNDLE['schemas']['$defs']
    require(kind in defs, 'schema', f'unknown record kind {kind}')
    _check(value, defs[kind], kind, defs)
    return copy.deepcopy(value)


def validate_profiles(profiles):
    require(profiles == PROFILE_HASHES, 'profile', 'unrecognized profile hashes')


def validate_command(row):
    validate_record('command', row)
    start, end, until = row['start_tick'], row['end_tick'], row['executed_until_tick']
    require(start % 10 == 0 and end == start + 10 and start <= until <= end, 'timing', 'command interval')
    requested = np.asarray(row['requested'], dtype=np.float64)
    require(np.max(np.abs(requested))<=np.finfo(np.float32).max, 'action', 'not representable as float32')
    expected = requested.astype(np.float32).astype(np.float64); expected[6:] = 0
    require(np.array_equal(row['transformed'], expected) and np.array_equal(row['filtered'], expected), 'action', 'unrecorded transform/filter')
    require(row['force_overridden'] == bool(np.any(requested[6:] != 0)), 'action', 'force override flag')
    status = row['status']
    if status == 'completed':
        require(until == end and row['executed'] is not None and np.array_equal(row['executed'], expected), 'execution', 'unconfirmed or changed execution')
    elif status == 'partial':
        require(start < until < end and row['executed'] is None, 'execution', 'partial must not become a full label')
    else:
        require(until == start and row['executed'] is None, 'execution', 'not executed')
    return row


def _rotation(q):
    x,y,z,w = q
    return np.asarray([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                       [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                       [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def validate_frame(row, *, epoch, origin_ns, last_tick):
    validate_record('camera_frame', row)
    camera = row['camera_id']; profile = BUNDLE[camera]
    require(row['profile_hash'] == BUNDLE['cameras']['cameras'][camera], 'profile', 'camera identity')
    path=Path(row['path'])
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in row['path'] and row['path']==path.as_posix() and row['path'].startswith(f'rgb/{camera}/'), 'path', 'camera relative path')
    tick = row['capture_tick']
    require(row['reset_epoch'] == epoch, 'epoch', 'camera epoch')
    require(tick % 40 == 0 and tick <= last_tick, 'timing', 'camera tick')
    require(row['timestamp_ns'] == origin_ns + tick * 2_000_000 and row['timestamp_ns'] <= 2**63-1, 'timing', 'capture timestamp')
    require(np.allclose(row['intrinsics'], profile['intrinsics'], rtol=0, atol=1e-3), 'camera', 'intrinsics')
    require(abs(np.linalg.norm(row['world_quaternion_xyzw']) - 1) < 1e-5, 'camera', 'quaternion')


def _numeric_group(root, descriptor, group, last_tick):
    path = safe_path(root, descriptor['path'])
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            require(len(names) == len(set(names)) and sum(x.file_size for x in z.infolist()) <= 256*1024*1024,
                    'array', 'duplicate entries or excessive numeric payload')
            for entry in z.infolist():
                require(entry.filename.endswith('.npy') and '/' not in entry.filename, 'array', 'NPZ members must be named NPY arrays')
                with z.open(entry) as stream:
                    version=np.lib.format.read_magic(stream)
                    require(version in ((1,0),(2,0)), 'array', 'unsupported NPY format')
                    reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                    shape,fortran,dtype=reader(stream)
                    require(not dtype.hasobject and dtype.fields is None and dtype.kind in 'biuf', 'array', 'numeric/bool arrays only; no pickle')
                    size=math.prod(shape)*dtype.itemsize
                    require(0<=size<=256*1024*1024 and stream.tell()+size==entry.file_size, 'array', 'array header/payload size mismatch')
        with np.load(path, allow_pickle=False) as f:
            arrays = {key: f[key] for key in f.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise ContractError('array', error) from error
    layouts = BUNDLE['telemetry']['groups'][group]
    expected = {'ticks'} | set(layouts) | {name+'_valid' for name in layouts}
    require(set(arrays) == expected, 'array', f'{group}: array names')
    ticks = arrays['ticks']
    require(ticks.dtype == np.dtype('int64') and ticks.ndim == 1 and len(ticks)>0, 'array', 'ticks dtype/shape')
    require(len(ticks)==last_tick//5+1+int(last_tick%5!=0), 'timing', f'{group} snapshot count')
    require(ticks[-1]==last_tick and np.array_equal(ticks[:-1], np.arange(len(ticks)-1,dtype=np.int64)*5), 'timing', f'{group} control/terminal snapshots')
    reasons = descriptor['missing_reasons']
    require(set(reasons) <= set(layouts), 'telemetry', 'unknown missing reason field')
    for name, spec in layouts.items():
        a, valid = arrays[name], arrays[name+'_valid']
        require(a.dtype == np.dtype(spec['dtype']) and a.shape == (len(ticks), *spec['shape']), 'array', f'{group}.{name} dtype/shape')
        require(valid.dtype == np.dtype('bool') and valid.shape == ticks.shape, 'array', f'{name} validity')
        require(np.isfinite(a).all(), 'array', 'nonfinite telemetry')
        require(bool(valid.all()) or bool(reasons.get(name)), 'telemetry', f'{name} missing reason')
        if 'pose' in name:
            require(np.allclose(np.linalg.norm(a[valid,3:7],axis=1),1,atol=1e-4), 'telemetry', f'{name} quaternion')
    if group == 'contact':
        require(np.all(arrays['desired_force'] == 0), 'force', 'desired telemetry force')
        s = arrays['source_tick']; v = arrays['source_tick_valid']
        require(np.all((s[v]>=0)&(s[v]<=ticks[v])) and np.all(np.diff(s[v])>=0), 'timing', 'contact source ticks')
        require(np.all(~arrays['normal_force_valid'] | v) and np.all(~arrays['friction_force_valid'] | v), 'telemetry', 'force needs source timestamp')
        q=arrays['qp_source_tick'];qv=arrays['qp_source_tick_valid']
        require(np.all((q[qv]>=0)&(q[qv]<=ticks[qv])) and np.all(~arrays['qp_force_valid']|qv), 'timing', 'QP force source tick')
    if group == 'robot':
        v=arrays['projected_gravity_valid']
        require(np.allclose(np.linalg.norm(arrays['projected_gravity'][v],axis=1),1,atol=1e-4), 'telemetry', 'gravity must be a unit vector')
    return arrays


def runtime_subject(episode):
    return digest({'profiles':episode['profiles'], 'task_config':episode['provenance']['task_config']['sha256'],
                   'checkpoint_sha256':episode['provenance']['checkpoint_sha256']})


def validate_episode(directory, *, release=False):
    root = Path(directory)
    require(type(release) is bool, 'schema', 'release must be boolean')
    checks = read_json(root/'checksums.json')
    require(isinstance(checks,dict) and checks and 'checksums.json' not in checks, 'checksum', 'manifest')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root/'checksums.json'}
    require(set(checks) == actual and 'episode.json' in checks, 'checksum', 'inventory')
    for name, sha in checks.items():
        require(isinstance(sha,str) and re.fullmatch('[0-9a-f]{64}',sha) and file_hash(safe_path(root,name))==sha, 'checksum', name)
    ep = read_json(root/'episode.json'); validate_record('episode',ep); validate_profiles(ep['profiles'])
    require(ep['warmup_end_sim_ns'] <= ep['origin_sim_ns'], 'timing', 'warmup origin')
    last = ep['last_tick']; outcome = ep['outcome']
    require(outcome['terminal_snapshot_tick']==last and outcome['snapshot_epoch']==ep['reset_epoch'], 'epoch', 'terminal snapshot')
    require(not(outcome['terminated'] and outcome['truncated']), 'execution', 'ambiguous ending')
    require((outcome['success_tick'] is not None)==outcome['success'] and (not outcome['success'] or outcome['success_tick']<=last), 'telemetry', 'success time')
    files=ep['files']
    frames=read_jsonl(safe_path(root,files['rgb_index']))
    seen={k:[] for k in CAMERAS}; ids={k:set() for k in CAMERAS}; image_paths=set()
    for row in frames:
        validate_frame(row,epoch=ep['reset_epoch'],origin_ns=ep['origin_sim_ns'],last_tick=last)
        k=row['camera_id']; seen[k].append(row['capture_tick'])
        require(row['path'].startswith(f'rgb/{k}/'), 'camera', 'camera path identity')
        require(row['sensor_frame_id'] not in ids[k], 'camera', 'duplicate sensor frame')
        ids[k].add(row['sensor_frame_id'])
        require(row['path'] not in image_paths and row['sha256']==checks.get(row['path']), 'camera', 'image reference/hash')
        image_paths.add(row['path'])
        try:
            image_path=safe_path(root,row['path'])
            with image_path.open('rb') as image_file:
                header=image_file.read(26)
            require(header[:8]==b'\x89PNG\r\n\x1a\n' and header[12:16]==b'IHDR' and header[24:26]==bytes([8,2]), 'camera', 'PNG must contain RGB8 samples')
            with Image.open(image_path) as im:
                require(im.format=='PNG' and im.mode=='RGB' and im.size==(1280,720), 'camera', 'RGB format/size')
                im.load()
        except (OSError, ValueError) as error:
            if isinstance(error,ContractError):raise
            raise ContractError('camera',error) from error
    for k in CAMERAS:
        require(len(seen[k])==last//40+1 and all(tick==i*40 for i,tick in enumerate(seen[k])), 'timing', f'{k} missing/duplicate/out-of-order captures')
    rows=read_jsonl(safe_path(root,files['commands'])); commands={}
    for i,row in enumerate(rows):
        validate_command(row)
        require(row['command_id'] not in commands and row['start_tick']==i*10, 'execution', 'command sequence')
        expected_until=max(row['start_tick'],min(last,row['end_tick']))
        require(row['executed_until_tick']==expected_until, 'execution', 'command coverage')
        commands[row['command_id']]=row
    require(len(rows)*10>=last, 'execution', 'missing commands')
    trace=read_jsonl(safe_path(root,files['control_trace'])); cursor=0
    for row in trace:
        validate_record('control_step',row)
        require(row['start_tick']==cursor and row['end_tick']==min(cursor+5,last), 'timing', 'control coverage')
        require(row['command_id'] in commands, 'execution', 'unknown held command')
        cmd=commands[row['command_id']]
        require(cmd['start_tick']<=cursor<row['end_tick']<=cmd['executed_until_tick'] and np.array_equal(row['held'],cmd['filtered']), 'execution', 'held command mismatch')
        for field in ('residual_observation_command_id','fl_ik_command_id'):
            other=row[field]
            require(other is None or (other in commands and commands[other]['start_tick']<=cursor), 'execution', 'future/unknown consumption')
        require(row['fl_ik_computed_tick']<=cursor, 'timing', 'future IK computation')
        if row['fl_ik_command_id'] is not None:
            require(commands[row['fl_ik_command_id']]['start_tick']<=row['fl_ik_computed_tick'], 'timing', 'IK precedes its source command')
        cursor=row['end_tick']
    require(cursor==last, 'execution', 'control trace incomplete')
    interventions=read_jsonl(safe_path(root,files['interventions']))
    for x in interventions:
        validate_record('intervention',x)
        require(x['start_tick']<x['end_tick']<=last, 'timing', 'intervention interval')
        require((x['application_point']=='explicit')==(x['point_m'] is not None), 'force', 'wrench point')
        require((x['kind']=='state_intervention')==(x['state_change'] is not None), 'force', 'intervention kind')
    groups={g:_numeric_group(root,files[g],g,last) for g in ('robot','contact','task')}
    task_ref=ep['provenance']['task_config']; task_cfg=read_json(safe_path(root,task_ref['path']))
    require(checks.get(task_ref['path'])==task_ref['sha256'], 'provenance', 'task hash')
    validate_record('task_config',task_cfg)
    require(task_cfg['task_id']==ep['task']['id'] and task_cfg['version']==ep['task']['version'] and 'progress' in task_cfg['metrics'], 'provenance', 'task identity/metric')
    for name in ('fl','body'):
        pair=task_cfg['contact_pairs'][name]
        require(pair['partner_body']==task_cfg['object_body'], 'telemetry', 'contact partner identity')
        if groups['contact'][name+'_object_contact_valid'].any():
            require(pair['method']=='filtered_body_pair', 'telemetry', 'proximity is not confirmed pair contact')
    robot=groups['robot']
    for frame in frames:
        index=frame['capture_tick']//5
        if robot['base_pose_valid'][index]:
            pose=robot['base_pose'][index]; r=_rotation(pose[3:]); p=BUNDLE[frame['camera_id']]
            require(np.allclose(frame['world_position'],pose[:3]+r@np.asarray(p['mount_position_m']),rtol=0,atol=1e-5), 'camera', 'moving mount position')
            require(np.allclose(_rotation(frame['world_quaternion_xyzw']),r@_rotation(p['mount_quaternion_xyzw']),rtol=0,atol=1e-5), 'camera', 'moving mount orientation')
    task=groups['task'];valid=task['success_valid'];success=task['success']
    require(np.all(np.diff(success[valid].astype(np.int8))>=0), 'telemetry', 'success must be latched')
    if valid[-1]:
        require(bool(success[-1])==outcome['success'], 'telemetry', 'outcome/task success disagreement')
    hits=task['ticks'][valid & success]
    if len(hits):
        require(outcome['success_tick']==int(hits[0]), 'telemetry', 'first success timestamp')
    for ref in ep['provenance']['runtime_evidence'] + ([ep['provenance']['asset']['approval']] if ep['provenance']['asset']['approval'] else []):
        require(checks.get(ref['path'])==ref['sha256'], 'provenance', 'evidence hash')
        validate_record('evidence',read_json(safe_path(root,ref['path'])))
    if release:
        require(not ep['diagnostic_only'], 'release', 'synthetic/diagnostic episode')
        require(last>0 and not outcome['truncated'] and outcome['success'], 'release', 'baseline successful complete demonstration required')
        for g,arrays in groups.items():
            for field in BUNDLE['telemetry']['groups'][g]:
                if field not in ('friction_force','qp_force','qp_source_tick'):
                    require(arrays[field+'_valid'].all(), 'release', f'missing {g}.{field}')
        require(not any(x['kind']=='state_intervention' or np.any(x['force_n']) or np.any(x['torque_nm']) for x in interventions), 'release', 'external assistance')
        required_checks={'control_consumption','force_boundary','contact_pairs','camera_capture_sync','terminal_pre_reset','asset_dual_view_coverage'}
        evidence=[read_json(safe_path(root,r['path'])) for r in ep['provenance']['runtime_evidence']]
        require(any(e['kind']=='runtime_conformance' and e['status']=='passed' and not e['diagnostic_only'] and e['subject_sha256']==runtime_subject(ep) and all(e['checks'].get(k)=='passed' for k in required_checks) for e in evidence), 'release', 'runtime evidence required')
        approval=ep['provenance']['asset']['approval'];require(approval is not None,'release','asset approval required')
        a=read_json(safe_path(root,approval['path']))
        require(a['kind']=='asset_approval' and a['status']=='passed' and not a['diagnostic_only'] and a['subject_sha256']==ep['provenance']['asset']['sha256'], 'release', 'asset approval subject/status')
    return {'schema_version':VERSION,'episode_id':ep['episode_id'],'contract_valid':True,'release_eligible':True if release else None,
            'release_validation':'passed' if release else 'not_run',
            'rgb_per_camera':len(seen[CAMERAS[0]]),'completed_commands':sum(x['status']=='completed' for x in rows),
            'control_steps':len(trace),'physics_ticks':last,'diagnostic_only':ep['diagnostic_only']}


def validate_release(path):
    path=Path(path); manifest=read_json(path);validate_record('release',manifest)
    require(manifest['episodes'], 'release', 'empty release')
    results=[]; ids=set()
    for item in manifest['episodes']:
        ep_path=safe_path(path.parent,item['path'])
        require(ep_path.name=='episode.json' and file_hash(ep_path)==item['sha256'], 'checksum', 'episode manifest digest')
        require(file_hash(safe_path(ep_path.parent,'checksums.json'))==item['checksums_sha256'], 'checksum', 'outer checksum-manifest digest')
        ep=read_json(ep_path)
        require(ep['dataset_id']==manifest['dataset_id'] and ep['episode_id'] not in ids, 'release', 'dataset/episode identity')
        ids.add(ep['episode_id']);results.append(validate_episode(ep_path.parent,release=True))
    return results
