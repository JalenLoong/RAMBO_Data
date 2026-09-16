"""Publication must resume identical content and reject immutable collisions."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import json
import subprocess
import sys

spec = importlib.util.spec_from_file_location('publisher', Path(__file__).parents[2] / 'scripts/rambo/publish_dataset_v2.py')
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)


def test_resume_and_collision(tmp_path):
    f = tmp_path / 'file.json'
    f.write_text('{}')
    meta = pub.describe(f)
    inv = {'files': {'file.json': meta, 'next.json': meta}}
    remote = {'file.json': SimpleNamespace(size=2, lfs=None, blob_id=meta['git_blob'])}
    assert pub.pending_files(inv, remote) == ['next.json']
    remote['file.json'].blob_id = 'changed'
    with pytest.raises(ValueError, match='collision'):
        pub.pending_files(inv, remote)


def test_lfs_hash_and_local_mutation(tmp_path):
    f = tmp_path / 'video.mp4'
    f.write_bytes(b'video')
    meta = pub.describe(f)
    assert pub.matches(SimpleNamespace(size=5, lfs=SimpleNamespace(sha256=meta['sha256'])), meta)
    inv = {'files': {'video.mp4': meta}}
    pub.check_local(tmp_path, inv)
    f.write_bytes(b'other')
    with pytest.raises(ValueError, match='changed'):
        pub.check_local(tmp_path, inv)


def test_path_escape(tmp_path):
    with pytest.raises(ValueError, match='Unsafe'):
        pub.safe_path('../secret')
    external = tmp_path.parent / (tmp_path.name + '-external')
    external.write_text('outside')
    (tmp_path / 'link').symlink_to(external)
    with pytest.raises(ValueError, match='escaping'):
        pub.check_local(tmp_path, {'files': {'link': pub.describe(external)}})


@pytest.mark.parametrize('mode,status,excluded', [
    ('pilot', 'success', []), ('demonstration', 'failure', []),
    ('demonstration', 'success', ['episode']),
])
def test_ineligible_raw_never_staged(tmp_path, mode, status, excluded):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    raw, canonical = tmp_path / 'raw', tmp_path / 'canonical'
    write(raw / 'summary.json', dict(passed=True, status=status, mode=mode, terminal_before_reset=True))
    write(canonical / 'meta/contract.json', {'episodes': [{'episode_uid': 'episode'}]})
    write(canonical / 'meta/split.json', {'episodes': {'train': ['episode']}})
    write(tmp_path / 'policy.json', {'scope': 'usable_demonstrations_only'})
    write(tmp_path / 'config.json', dict(publication_policy=str(tmp_path / 'policy.json'),
        release='release', canonical_root=str(canonical), expected_count=1,
        expected_split={'train': 1}, task='push_box', excluded_episode_uids=excluded,
        episodes=[dict(episode_uid='episode', raw=str(raw),
                       raw_path='raw/v2/push_box/collection/episode', behavior={'passed': True})]))
    result = subprocess.run([sys.executable, str(Path(__file__).parents[2] / 'scripts/rambo/prepare_release_v2.py'),
        '--config', str(tmp_path / 'config.json'), '--stage', str(tmp_path / 'stage'),
        '--inventory', str(tmp_path / 'inventory.json')], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Ineligible episode' in result.stderr or 'Quarantined episode' in result.stderr
    assert not (tmp_path / 'stage').exists()
