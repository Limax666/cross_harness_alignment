"""CPU/filesystem-only publication; original release gates, bounded parallel I/O.

The candidate must remain private and immutable throughout publication. This
operational helper is deliberately outside the frozen release runtime.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import threading

from hsrl_v5 import release as original
from hsrl_v5.dataset import Dataset


def parallel_each(function, values, workers):
    """Bound queued work; executor shutdown joins all workers after any error."""
    if not 1 <= workers <= 8:
        raise ValueError('workers_must_be_1_to_8')
    iterator, lock, stopped = iter(values), threading.Lock(), threading.Event()
    def consume():
        results = []
        while not stopped.is_set():
            with lock:
                value = next(iterator, None)
            if value is None:
                break
            try:
                results.append(function(value))
            except BaseException:
                stopped.set()
                raise
        return results
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(consume) for _ in range(workers)]
        return [value for future in futures for value in future.result()]


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _scan(source, target, workers):
    """Partition at depth three; anchor every open to no-follow directory fds."""
    source = Path(source)
    root = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    jobs, parents = [], []
    def open_directory(parts):
        descriptor = os.dup(root)
        try:
            for part in parts:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise
    def unchanged(descriptor, before):
        if _identity(os.fstat(descriptor)) != _identity(before):
            raise ValueError('source_changed_during_copy_or_inventory')
    def metadata(descriptor, parts):
        if target is not None:
            destination = os.open(target.joinpath(*parts), os.O_RDONLY | os.O_DIRECTORY)
            try:
                shutil.copystat(descriptor, destination)
            finally:
                os.close(destination)
    def visit(parts):
        parent = open_directory(parts[:-1])
        try:
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=parent)
        finally:
            os.close(parent)
        try:
            before = os.fstat(descriptor)
            if stat.S_ISDIR(before.st_mode):
                if target is not None:
                    target.joinpath(*parts).mkdir()
                files = set()
                with os.scandir(descriptor) as entries:
                    for entry in entries:
                        files.update(visit(parts + (entry.name,)))
                unchanged(descriptor, before)
                metadata(descriptor, parts)
                return files
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('source_special_file_rejected')
            if target is not None:
                with os.fdopen(os.dup(descriptor), 'rb') as reader:
                    with target.joinpath(*parts).open('xb') as writer:
                        shutil.copyfileobj(reader, writer)
                        writer.flush()
                        shutil.copystat(descriptor, writer.fileno())
            unchanged(descriptor, before)
            return {'/'.join(parts)}
        finally:
            os.close(descriptor)
    def partition(parts):
        descriptor = open_directory(parts)
        try:
            before = os.fstat(descriptor)
            parents.append((parts, before))
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    child = parts + (entry.name,)
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode) and len(child) < 3:
                        if target is not None:
                            target.joinpath(*child).mkdir()
                        partition(child)
                    elif stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode):
                        jobs.append(child)
                    else:
                        raise ValueError('source_symlink_or_special_file_rejected')
            unchanged(descriptor, before)
        finally:
            os.close(descriptor)
    try:
        if target is not None:
            target.mkdir(mode=0o700)
        partition(())
        files = set().union(*parallel_each(visit, jobs, workers))
        for parts, before in reversed(parents):
            descriptor = open_directory(parts)
            try:
                unchanged(descriptor, before)
                metadata(descriptor, parts)
            finally:
                os.close(descriptor)
        return files
    finally:
        os.close(root)


def copy_parallel(source, target, workers):
    return _scan(Path(source), Path(target), workers)


def check_closure(root, expected, workers):
    if _scan(Path(root), None, workers) != set(expected):
        raise ValueError('release_file_inventory_differs_from_manifest')


def verify_files(dataset, workers):
    # No verification cache: invoke the original method for EVERY declared file.
    return len(parallel_each(dataset.verify_file, dataset.files, workers))


def read_source_manifest(candidate):
    directory = os.open(candidate, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        descriptor = os.open('manifest.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=directory)
        with os.fdopen(descriptor) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('source_manifest_requires_regular_file')
            return json.load(stream)
    finally:
        os.close(directory)


def audit_candidate(root, workers):
    dataset = Dataset.for_verification(root)
    verify_files(dataset, workers)
    visited = []
    for block in range(875):
        first, second = dataset.sampler.step(block * 2), dataset.sampler.step(block * 2 + 1)
        if first['groups'] != second['groups'] or first['configuration'] == second['configuration']:
            raise ValueError('K2_schedule_identity_failed')
        visited.extend((g['source'], g['id']) for g in first['groups'])
    if len(visited) != len(set(visited)) or set(visited) != set(dataset.index):
        raise ValueError('full_875_block_sampler_coverage_failed')
    def public(key):
        if set(dataset.public(key)) != {'messages', 'tools'}:
            raise ValueError('private_fields_in_public_render')
    parallel_each(public, dataset.index, workers)


def publish(candidate, out, context_audit, runtime_audit, workers=8):
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError('publish_destination_exists:' + str(out))
    if not 1 <= workers <= 8:
        raise ValueError('workers_must_be_1_to_8')
    candidate = Path(candidate).absolute()
    if candidate.is_symlink():
        raise ValueError('source_symlink_rejected')
    manifest = read_source_manifest(candidate)
    if manifest.get('status') != 'candidate' or (candidate / 'READY.json').exists():
        raise ValueError('publish_requires_unpublished_candidate')
    original.require_admission(manifest['admission_report'])
    # Reject FIFOs/devices/links before the original loader opens any asset.
    check_closure(candidate, set(manifest['file_sha256']) | {'manifest.json'}, workers)
    source = Dataset.for_verification(candidate)
    verify_files(source, workers)
    out.parent.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix='.' + out.name + '.publication-', dir=out.parent))
    os.chmod(private, 0o700)
    stage = private / 'artifact'
    # Failures retain this private directory; executor closure precedes propagation.
    copied = copy_parallel(candidate, stage, workers)
    if copied != set(source.files) | {'manifest.json'}:
        raise ValueError('release_file_inventory_differs_from_manifest')
    if original.file_hash(candidate / 'manifest.json') != original.file_hash(stage / 'manifest.json'):
        raise ValueError('source_manifest_changed_during_copy')
    audit_candidate(stage, workers)
    manifest = original.read_json(stage / 'manifest.json')
    for kind, path in (('context', context_audit), ('runtime', runtime_audit)):
        target = stage / 'audits' / (kind + '.json')
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        manifest['file_sha256']['audits/' + kind + '.json'] = original.file_hash(target)
    original.validate_certificates(stage, manifest)
    manifest['status'] = 'ready'
    original.atomic_json(stage / 'manifest.json', manifest)
    original.atomic_json(stage / 'READY.json', {'manifest_sha256': original.file_hash(stage / 'manifest.json')})
    check_closure(stage, set(manifest['file_sha256']) | {'manifest.json', 'READY.json'}, workers)
    env = dict(os.environ, PYTHONPATH=str(stage / 'code'), PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, '-B', '-c',
                    'from hsrl_v5.dataset import Dataset; import sys; d=Dataset(sys.argv[1]); assert len(d.index)==14000', str(stage)],
                   cwd=stage, env=env, check=True, capture_output=True, text=True)
    original.install_no_replace(stage, out)
    private.rmdir()
    return {'status': 'ready', 'root': str(out), 'manifest_sha256': original.file_hash(out / 'manifest.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'out', 'context-audit', 'runtime-audit'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    args = parser.parse_args(argv)
    print(original.canonical(publish(args.candidate, args.out, args.context_audit, args.runtime_audit, args.workers)))


if __name__ == '__main__':
    main()
