"""LeRobot-format contract validator. All operations are read-only; no publication claim."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from ..contracts_v2.validation import (ContractError, require, digest, file_hash, read_json,
                                     read_jsonl, safe_path, canonical_bytes, _check)
from .media import MediaTools, validate_video, read_terminal_png

SPEC = Path(__file__).with_name('spec')
LOCK = read_json(SPEC / 'lock.json')
for name, sha in LOCK['files'].items():
    require(file_hash(SPEC / name) == sha, 'profile', name)
PROFILE = read_json(SPEC / 'profile.json')
COLUMNS = read_json(SPEC / 'columns.json')
SCHEMAS = read_json(SPEC / 'schemas.json')['$defs']
VERSION = 'wam-quadruped-v2.1.0'
CAMERAS = tuple(PROFILE['camera_keys'])
STATE_KEYS = tuple(k for k in COLUMNS if k.startswith(('observation.state.', 'task.')))
require(digest(PROFILE) == LOCK['profile_hash'] and digest(COLUMNS) == LOCK['columns_hash'], 'profile', 'bundle digests')


def record(kind, value):
    canonical_bytes(value)
    _check(value, SCHEMAS[kind], kind, SCHEMAS)
    return value


@dataclass(frozen=True)
class DataPaths:
    dataset_root: Path
    canonical_root: Path
    cache_root: Path
    tools: MediaTools

    @classmethod
    def from_config(cls, config):
        record('paths', config)
        values = {key: os.path.expandvars(value) for key, value in config.items()}
        require(all('$' not in v for v in values.values()), 'config', 'unresolved environment variable')
        paths = {key: Path(values[key]) for key in ('dataset_root', 'canonical_root', 'cache_root')}
        require(all(p.is_absolute() for p in paths.values()), 'config', 'explicit absolute data roots required')
        canonical, cache = paths['canonical_root'].resolve(), paths['cache_root'].resolve()
        require(not cache.is_relative_to(canonical) and not canonical.is_relative_to(cache), 'config', 'cache and canonical roots must be independent')
        raw = paths['dataset_root'].resolve()
        require(not cache.is_relative_to(raw) and not raw.is_relative_to(cache), 'config', 'cache and raw roots must be independent')
        return cls(**paths, tools=MediaTools(values['ffmpeg'], values['ffprobe']))


def _pa():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ContractError('dependency', 'PyArrow is required by offline dataset validation') from error
    return pa, pq


def parquet(path):
    _, pq = _pa()
    try:
        return pq.read_table(path)
    except Exception as error:
        raise ContractError('parquet', str(error)) from error


def inventory(root):
    root = Path(root)
    index = read_json(root / 'meta/checksums.json')
    require(isinstance(index, dict) and index, 'checksum', 'inventory required')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and p != root / 'meta/checksums.json'}
    require(set(index) == actual, 'checksum', 'unlisted or missing payload')
    for name, sha in index.items():
        require(file_hash(safe_path(root, name)) == sha, 'checksum', name)
    return index


def _array(table, key, dtype, width=1):
    pa, _ = _pa()
    require(key in table.column_names, 'columns', key)
    require(table[key].null_count == 0, 'dtype', f'null values in {key}')
    t = table.schema.field(key).type
    scalar = pa.from_numpy_dtype(np.dtype(dtype))
    if pa.types.is_fixed_size_list(t) or pa.types.is_list(t):
        require(t.value_type == scalar, 'dtype', key)
        require(table[key].combine_chunks().flatten().null_count == 0, 'dtype', f'null components in {key}')
        if pa.types.is_fixed_size_list(t):
            require(t.list_size == width, 'shape', key)
    else:
        require(width == 1 and t == scalar, 'dtype', key)
    try:
        a = np.asarray(table[key].to_pylist(), dtype=dtype).reshape(len(table), -1)
    except (ValueError, TypeError) as error:
        raise ContractError('shape', key) from error
    require(a.shape == (len(table), width) and np.isfinite(a).all(), 'shape', key)
    return a


def _state_valid(values):
    for key in ('observation.state.base.orientation', 'task.object.orientation'):
        require(np.allclose(np.linalg.norm(values[key], axis=-1), 1, rtol=0, atol=1e-4), 'state', key)
    require(np.allclose(np.linalg.norm(values['observation.state.projected_gravity'], axis=-1), 1, atol=1e-4), 'state', 'gravity')


def _rows(table, n):
    require(len(table) == n and n > 0, 'timing', 'canonical row count')
    values = {k: _array(table, k, v['dtype'], v['shape'][0]) for k, v in COLUMNS.items()}
    require(np.array_equal(values['simulation_time_ns'][:, 0], np.arange(n, dtype=np.int64) * 20_000_000), 'timing', 'row timestamps')
    require(np.array_equal(values['physics_step'][:, 0], np.arange(n, dtype=np.int64) * 10), 'timing', 'physics step mapping')
    require(values['execution.completed'].all(), 'execution', 'partial/unconfirmed action in paired rows')
    expected = values['action.requested'].copy()
    expected[:, 6:] = 0
    require(np.array_equal(values['action'], expected), 'transform', 'unexplained command modification')
    require(np.array_equal(values['execution.force_zero_modified'][:, 0],
                           np.any(values['action.requested'][:, 6:] != 0, axis=1)), 'transform', 'force modification flag')
    _state_valid(values)
    return values


def _camera_index(table, counts, periods):
    pa, _ = _pa()
    fields = {'camera_key': pa.string(), 'frame_index': pa.int32(), 'simulation_time_ns': pa.int64(),
              'physics_step': pa.int64(), 'valid': pa.bool_()}
    require(set(table.column_names) == set(fields), 'camera_index', 'fields')
    require(all(table.schema.field(k).type == v for k, v in fields.items()), 'camera_index', 'dtypes')
    rows = table.to_pylist()
    require(set(r['camera_key'] for r in rows) == set(counts), 'camera_index', 'camera identities')
    for camera, n in counts.items():
        subset = [r for r in rows if r['camera_key'] == camera]
        require(len(subset) == n, 'camera_index', 'count')
        for i, row in enumerate(subset):
            require(row['frame_index'] == i and row['valid'] is True, 'camera_index', 'order/validity')
            ns = i * periods[camera]
            require(row['simulation_time_ns'] == ns and row['physics_step'] * 2_000_000 == ns,
                    'timing', 'camera simulation schedule')



def _task_contact_diagnostics(root, manifest, n):
    """The approved Push Box profile must preserve unknown, not false evidence."""
    ext=manifest.get('extensions', {})
    if ext.get('task_profile_version') != 'push-box-v2-2':
        return
    ref=ext.get('contact_diagnostics')
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'},'contact','explicit diagnostic availability sidecar required')
    path=safe_path(root,ref['path'])
    require(file_hash(path)==ref['sha256'],'contact','diagnostic sidecar checksum')
    data=read_json(path)
    require(data.get('role')=='diagnostic_only' and data.get('unavailable')=='unknown' and data.get('bool_columns_are_invalid_placeholders') is True,'contact','unknown semantics')
    rows=data.get('rows',[])
    require(len(rows)==n+1,'contact','diagnostics include terminal')
    for i,row in enumerate(rows):
        require(row['simulation_time_ns']==i*20_000_000,'contact','diagnostic time alignment')
        require(row.get('valid') is False and row.get('status')=='unknown' and row.get('fl_object') is None and row.get('body_object') is None and bool(row.get('reason')),'contact','unreliable pairs must be explicitly unknown')

def validate_canonical(root, *, tools, check_media=True):
    root = Path(root)
    sums = inventory(root)
    manifest = record('canonical', read_json(root / 'meta/contract.json'))
    require(manifest['profile_hash'] == LOCK['profile_hash'] and manifest['columns_hash'] == LOCK['columns_hash'], 'profile', 'dataset profile')
    require(manifest['transform_chain_hash'] == PROFILE['transform_chain_hash'], 'transform', 'chain hash')
    info = read_json(root / 'meta/info.json')
    require(info['codebase_version'] == 'v2.1' and info['fps'] == 50, 'version', 'LeRobot version/rate')
    features = info['features']
    require({k for k, v in features.items() if v['dtype'] in ('video', 'image')} == set(CAMERAS), 'observer', 'exactly two policy video features')
    require(not any(k.startswith('monitor.') or k == 'observation.images.observer' for k in features), 'observer', 'monitor cannot be a canonical feature')
    for key, spec in COLUMNS.items():
        require(key in features and features[key]['dtype'] == spec['dtype'] and features[key]['shape'] == spec['shape'], 'columns', key)
        require(features[key].get('names') == spec['names'], 'columns', f'{key} component names')
    for key in CAMERAS:
        require(features[key]['dtype'] == 'video' and features[key]['shape'] == [3, 720, 1280], 'camera', key)
    tasks = read_jsonl(root / 'meta/tasks.jsonl')
    task_lookup = {t['task_index']: t['task'] for t in tasks}
    require(len(task_lookup) == len(tasks) and all(type(t) is str and t.strip() for t in task_lookup.values()), 'text', 'task mapping')
    episodes = read_jsonl(root / 'meta/episodes.jsonl')
    require(len(episodes) == len(manifest['episodes']) == info['total_episodes'] and len(episodes) > 0, 'episode', 'episode counts')
    total = 0; reports = []; uids = set()
    for expected_index, (meta, ep) in enumerate(zip(episodes, manifest['episodes'])):
        n = ep['length']; index = ep['episode_index']
        require(ep['camera_config_hash'] == digest(PROFILE['policy_camera_geometry']), 'profile', 'camera geometry')
        require(index == meta['episode_index'] == expected_index and ep['episode_uid'] not in uids, 'episode', 'identity/order')
        uids.add(ep['episode_uid'])
        require(n == meta['length'] and ep['raw_boundary_count'] == n + 1, 'terminal', 'N/N+1')
        require(meta['tasks'] == [ep['task_text']] and ep['task_text'].strip(), 'text', 'episode instruction')
        require(meta.get('action_config') == [{'start_frame': 0, 'end_frame': n, 'action_text': ep['task_text']}], 'action_config', 'one full paired segment')
        file = info['data_path'].format(episode_chunk=index // info['chunks_size'], episode_index=index)
        table = parquet(safe_path(root, file)); values = _rows(table, n)
        require(set(table.column_names) == set(COLUMNS) | {'timestamp','frame_index','episode_index','index','task_index'}, 'columns', 'unexpected canonical columns')
        for key, expected in [('frame_index', np.arange(n)), ('episode_index', np.full(n, index)), ('index', np.arange(total, total+n))]:
            require(np.array_equal(_array(table, key, 'int64')[:, 0], expected), 'timing', key)
        require(np.allclose(_array(table, 'timestamp', 'float32')[:, 0], np.arange(n)/50, rtol=0, atol=1e-4), 'timing', 'LeRobot timestamp')
        require(all(task_lookup.get(int(t)) == ep['task_text'] for t in _array(table, 'task_index', 'int64')[:, 0]), 'text', 'row task')
        camera_index = ep['camera_index']
        require(sums.get(camera_index['path']) == camera_index['sha256'], 'checksum', 'camera index reference')
        _camera_index(parquet(safe_path(root, camera_index['path'])), {k:n for k in CAMERAS}, {k:20_000_000 for k in CAMERAS})
        for camera in CAMERAS:
            video = ep['videos'][camera]
            expected_path = info['video_path'].format(episode_chunk=index // info['chunks_size'], episode_index=index, video_key=camera)
            require(video['path'] == expected_path and sums.get(video['path']) == video['sha256'], 'camera', 'video path/hash')
            if check_media:
                validate_video(safe_path(root, video['path']), count=n, fps=50, tools=tools,
                               profile=PROFILE['policy_video'], receipt=video['encoding'])
        terminal = ep['terminal']
        require(terminal['simulation_time_ns'] == n * 20_000_000 and terminal['physics_step'] == n * 10, 'terminal', 'boundary time')
        _state_valid({k:np.asarray(v) for k,v in terminal['state'].items()})
        for camera in CAMERAS:
            image = terminal['images'][camera]
            require(image['source_frame_index'] == n and image['simulation_time_ns'] == n*20_000_000, 'terminal', 'source boundary')
            require(sums.get(image['path']) == image['sha256'], 'terminal', 'snapshot hash')
            read_terminal_png(safe_path(root, image['path']))
        require(ep['status'] != 'invalid', 'episode', 'invalid episode is not canonical')
        require(ep['status'] != 'success' or terminal['state']['task.success'] == [True], 'state', 'success must follow terminal task state')
        total += n
        reports.append({'episode_index':index,'rows':n,'terminal_action_count':0,'boundary_observations':n+1})
    if len(manifest['episodes'])==1:
        _task_contact_diagnostics(root, manifest, total)
    require(total == info['total_frames'] and info['total_videos'] == 2 * len(episodes), 'episode', 'totals')
    require(not any(p.parts[0] in ('latents','text_embeddings','normalization','cache') for p in map(Path,sums)), 'cache', 'model cache inside canonical')
    return {'contract_valid':True,'dataset_schema_version':VERSION,'diagnostic_only':manifest['diagnostic_only'],
            'episodes':reports,'video_validation':'passed' if check_media else 'not_run','release_validation':'not_run','published':False}


def validate_raw(root, *, tools, check_media=True):
    root = Path(root)
    raw = record('raw', read_json(root / 'manifest.json'))
    require(raw['profile_hash'] == LOCK['profile_hash'], 'profile', 'raw profile')
    n = raw['n_actions']
    require(n > 0 and raw['n_boundaries'] == n+1, 'terminal', 'raw boundary count')
    for key, path in raw['streams'].items():
        require(file_hash(safe_path(root,path)) == raw['stream_sha256'][key], 'checksum', f'raw {key}')
    policy_table = parquet(safe_path(root, raw['streams']['policy']))
    require('action.executed' in policy_table.column_names and 'action' not in policy_table.column_names, 'columns', 'raw executed action name')
    policy_table = policy_table.rename_columns(['action' if k == 'action.executed' else k for k in policy_table.column_names])
    _rows(policy_table, n)
    controller = parquet(safe_path(root, raw['streams']['controller']))
    require(len(controller) == 2*n, 'timing', 'controller step count')
    require(np.array_equal(_array(controller,'timestamp_ns','int64')[:,0],np.arange(2*n)*10_000_000), 'timing', 'controller timestamps')
    require(np.array_equal(_array(controller,'controller_tick','int64')[:,0],np.arange(2*n)), 'timing', 'controller ticks')
    held = _array(controller,'high_level_command','float32',9)
    _array(controller,'policy_residual','float32',18)
    for key in ('desired_joint_target','desired_joint_velocity','desired_joint_torque'):
        _array(controller,key,'float32',12)
    _array(controller,'safety_flags','int32')
    require('reference_state' in controller.column_names and 'controller_mode' in controller.column_names, 'columns', 'controller reference/mode')
    refs=controller['reference_state'].to_pylist()
    require(all(isinstance(x,list) and len(x)>0 and len(x)==len(refs[0]) and np.isfinite(x).all() for x in refs), 'state', 'controller reference vector')
    require(all(isinstance(x,str) and x for x in controller['controller_mode'].to_pylist()), 'state', 'controller mode')
    policy = _array(policy_table,'action','float32',9)
    require(np.array_equal(held,np.repeat(policy,2,axis=0)), 'execution', 'two control holds')
    physics = parquet(safe_path(root, raw['streams']['physics']))
    require(len(physics) == 10*n+1, 'timing', 'physics boundaries including initial')
    times = _array(physics,'timestamp_ns','int64')[:,0]
    require(np.array_equal(times,np.arange(10*n+1)*2_000_000), 'timing', 'physics timestamps')
    require(np.array_equal(_array(physics,'physics_step','int64')[:,0],np.arange(10*n+1)), 'timing', 'physics indices')
    for key in ('base_pose','object_pose'):
        value=_array(physics,key,'float32',7)
        require(np.allclose(np.linalg.norm(value[:,3:],axis=1),1,atol=1e-4),'state',key)
    for key,width in [('base_twist',6),('joint_position',12),('joint_velocity',12),('joint_torque',12),('object_velocity',6)]:
        _array(physics,key,'float32',width)
    for key in ('external_force','external_torque'):
        require(np.all(_array(physics,key,'float32',3) == 0), 'force', 'external assistance')
    require(set(physics['external_wrench_frame'].to_pylist()) <= {'world','body','projected_com'}, 'force', 'wrench frame')
    contact = parquet(safe_path(root, raw['streams']['contact_sensor']))
    ct = _array(contact,'measurement_timestamp_ns','int64')[:,0]
    seq = _array(contact,'sensor_sequence_index','int64')[:,0]
    require(len(ct)>0 and ct[0]>=0 and ct[-1]<=times[-1] and np.all(np.diff(ct)>0), 'contact', 'fresh measurement timestamps')
    require(np.array_equal(seq,np.arange(len(ct))), 'contact', 'measurement sequence')
    _array(contact,'force','float32',3)
    require(set(contact['frame'].to_pylist()) == {'world'} and set(contact['force_kind'].to_pylist()) == {'net_normal'}, 'contact', 'actual force meaning')
    require(_array(contact,'valid','bool').all(), 'contact', 'invalid measurements')
    require(all(t is None for t in contact['torque'].to_pylist()), 'contact', 'direct sensor torque unavailable; do not fabricate')
    pa, _ = _pa()
    require(physics.schema.field('contact_measurement_timestamp_ns').type == pa.int64(), 'contact', 'latest timestamp dtype')
    latest = physics['contact_measurement_timestamp_ns'].to_pylist()
    fresh = _array(physics,'contact_measurement_is_fresh','bool')[:,0]
    ids = np.searchsorted(ct,times,side='right')-1
    for i, index in enumerate(ids):
        require(latest[i] == (None if index < 0 else int(ct[index])), 'contact', 'latest measurement mapping')
    expected_fresh = (ids >= 0) & np.concatenate(([True], ids[1:] != ids[:-1]))
    require(np.array_equal(fresh,expected_fresh), 'contact', 'freshness flag')
    require(raw['terminal']['simulation_time_ns']==n*20_000_000,'terminal','raw terminal')
    _state_valid({k:np.asarray(v) for k,v in raw['terminal']['state'].items()})
    counts={k:n+1 for k in CAMERAS};periods={k:20_000_000 for k in CAMERAS}
    obs=raw['observer']
    if obs['present']:
        require(file_hash(safe_path(root,obs['path'])) == obs['sha256'], 'checksum', 'observer')
        counts['monitor.images.observer']=(n*20_000_000)//40_000_000+1
        periods['monitor.images.observer']=40_000_000
        if check_media:
            validate_video(safe_path(root,obs['path']),count=counts['monitor.images.observer'],fps=25,tools=tools)
    else:
        require(obs['reason'].strip(),'observer','missing observer requires explicit reason')
    _camera_index(parquet(safe_path(root,raw['streams']['camera_index'])),counts,periods)
    if check_media:
        for camera,video in raw['videos'].items():
            validate_video(safe_path(root,video['path']),count=n+1,fps=50,tools=tools,
                           profile=PROFILE['policy_video'],receipt=video['encoding'])
    _task_contact_diagnostics(root, raw, n)
    return {'contract_valid':True,'diagnostic_only':raw['diagnostic_only'],'raw_actions':n,'raw_boundaries':n+1,
            'physics_steps':10*n,'controller_steps':2*n,'sensor_updates':len(ct),'sensor_rate':'observed timestamps, not assumed 200Hz',
            'video_validation':'passed' if check_media else 'not_run','published':False}
