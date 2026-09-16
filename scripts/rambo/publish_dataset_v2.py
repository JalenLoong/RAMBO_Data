"""Explicit, immutable release inventory and resumable Hugging Face publication.

No simulator/model imports. Authentication uses the existing Hugging Face login.
Root index publication is deliberately separate from immutable payload upload.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def describe(path):
    path = Path(path)
    git = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            git.update(chunk)
    return dict(size=path.stat().st_size, sha256=digest(path), git_blob=git.hexdigest())


def safe_path(value):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or str(p) != value or not p.parts:
        raise ValueError(f'Unsafe release path: {value}')
    return value


def check_local(stage, inventory):
    stage = Path(stage).resolve()
    for name, expected in inventory['files'].items():
        safe_path(name)
        p = stage / name
        if not p.is_file() or not p.resolve().is_relative_to(stage):
            raise ValueError(f'Missing or escaping file: {name}')
        if describe(p) != expected:
            raise ValueError(f'Local content changed: {name}')


def matches(remote, expected):
    if remote.size != expected['size']:
        return False
    if remote.lfs is not None:
        return remote.lfs.sha256 == expected['sha256']
    return remote.blob_id == expected['git_blob']


def pending_files(inventory, remote):
    pending = []
    for name, expected in inventory['files'].items():
        if name not in remote:
            pending.append(name)
        elif not matches(remote[name], expected):
            raise ValueError(f'Immutable remote path collision: {name}')
    return sorted(pending)


def remote_files(api, repo, revision):
    from huggingface_hub.hf_api import RepoFile
    return {p.path: p for p in api.list_repo_tree(repo, repo_type='dataset',
            revision=revision, recursive=True) if isinstance(p, RepoFile)}


def save(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['check', 'upload', 'verify', 'index'])
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--revision')
    p.add_argument('--index-dir', type=Path)
    p.add_argument('--verification', type=Path)
    p.add_argument('--batch-size', type=int, default=100)
    args = p.parse_args()
    inv = json.loads(args.inventory.read_text())
    check_local(args.stage, inv)
    if args.operation == 'check':
        save(args.report, dict(passed=True, files=len(inv['files']),
                              bytes=sum(f['size'] for f in inv['files'].values())))
        return
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    api = HfApi()
    repo = inv['repo_id']
    revision = args.revision or api.dataset_info(repo).sha
    remote = remote_files(api, repo, revision)
    pending = pending_files(inv, remote)
    if args.operation == 'upload':
        if args.batch_size < 1:
            raise ValueError('Positive batch size required')
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset + args.batch_size]
            result = api.create_commit(repo, repo_type='dataset', parent_commit=revision,
                commit_message=f"[{inv['work_id']}] {inv['release']} payload {offset + len(batch)}/{len(pending)}",
                operations=[CommitOperationAdd(path_in_repo=name,
                    path_or_fileobj=str(args.stage / name)) for name in batch])
            revision = result.oid
            save(args.report, dict(status='uploading', revision=revision,
                 uploaded_this_run=offset + len(batch), missing_at_start=len(pending)))
            print(f'Uploaded {offset + len(batch)}/{len(pending)} at {revision}', flush=True)
        remote = remote_files(api, repo, revision)
        pending = pending_files(inv, remote)
    if pending:
        raise ValueError(f'{len(pending)} files missing remotely')
    if args.operation == 'index':
        if args.index_dir is None or args.verification is None:
            raise ValueError('Index requires prepared index files and verification report')
        verified = json.loads(args.verification.read_text())
        if not verified.get('passed') or verified.get('repo_id') != repo or verified.get('release') != inv['release']:
            raise ValueError('Wrong or failed canonical readback verification')
        if verified.get('inventory_sha256') != digest(args.inventory):
            raise ValueError('Verification refers to another inventory')
        index = json.loads((args.index_dir / 'releases.json').read_text())
        entries = index['releases']
        if len([e for e in entries if e['id'] == inv['release']]) != 1:
            raise ValueError('Release must appear exactly once')
        if 'releases.json' in remote:
            old = json.loads(Path(hf_hub_download(repo, 'releases.json', repo_type='dataset', revision=revision)).read_text())
            by_id = {e['id']: e for e in entries}
            if any(by_id.get(e['id']) != e for e in old['releases']):
                raise ValueError('Cannot delete or rewrite an indexed historical release')
        result = api.create_commit(repo, repo_type='dataset', parent_commit=revision,
            commit_message=f"[{inv['work_id']}] publish verified {inv['release']}",
            operations=[CommitOperationAdd(path_in_repo=name,
                path_or_fileobj=str(args.index_dir / name)) for name in ['README.md', 'releases.json']])
        revision = result.oid
    save(args.report, dict(passed=True, repo_id=repo, release=inv['release'],
         revision=revision, inventory_sha256=digest(args.inventory), files=len(inv['files']),
         bytes=sum(f['size'] for f in inv['files'].values()), operation=args.operation))


if __name__ == '__main__':
    main()
