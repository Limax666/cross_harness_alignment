"""Serial private data-grader reuse; actors and checker children are never reused.

Only this host controller mutates projection contents. Bind root inodes persist
until the exact owned container's removal is confirmed. No checker runs here.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import threading
import uuid

from .io import atomic_json, digest
from .isolation import CleanupUncertain, IsolationError
from .workspace import WorkspaceWorker, checker_boundary_contract, snapshot_entries
from .workspace_worker import MAX_FILE

MAX_CASES = 64
MAX_REQUESTS = 128


def replace_projection(source, target, expected=None):
    """Copy bounded regular bytes without sharing inodes or replacing bind roots."""
    source, target = Path(source), Path(target)
    if source.is_symlink() or target.is_symlink() or not source.is_dir() or not target.is_dir():
        raise ValueError('unsupported_projection_root')
    source, target = source.resolve(strict=True), target.resolve(strict=True)
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError('overlapping_projection_roots')
    original = snapshot_entries(source)
    if expected is not None and original != expected:
        raise IsolationError('projection_source_manifest_mismatch')
    for path in target.iterdir():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    # Preserve empty source directories as well as all file bytes.
    for directory in sorted(source.rglob('*')):
        if directory.is_dir():
            (target / directory.relative_to(source)).mkdir(mode=0o755, parents=True, exist_ok=True)
    for name, metadata in original.items():
        path = source / name
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != metadata['bytes']:
                raise IsolationError('projection_source_file_changed')
            data = stream.read(MAX_FILE + 1)
            after = os.fstat(stream.fileno())
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if identity(before) != identity(after) or len(data) != metadata['bytes']:
            raise IsolationError('projection_source_file_changed')
        if hashlib.sha256(data).hexdigest() != metadata['sha256']:
            raise IsolationError('projection_source_bytes_changed')
        output = target / name
        output.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        with output.open('xb') as stream:
            stream.write(data)
        output.chmod(0o644)
    if snapshot_entries(source) != original or snapshot_entries(target) != original:
        raise IsolationError('projection_copy_manifest_mismatch')
    return original


def validate_response_identity(response, nonce):
    result = response.get('result', {}) if isinstance(response, dict) else {}
    if (not isinstance(response, dict) or response.get('ok') is not True
            or response.get('request_nonce') != nonce or not isinstance(result, dict)
            or type(result.get('child_pid')) is not int or result['child_pid'] <= 1
            or type(result.get('returncode')) is not int
            or not isinstance(result.get('stdout'), str) or not isinstance(result.get('stderr'), str)):
        raise IsolationError('private_grader_response_identity_or_contract_failed')


class PrivateGraderPool:
    """One private container, serialized projections, at most 64 cases/128 children."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.lock = threading.RLock()
        self.worker = None
        self.closed = self.poisoned = False
        self.generation = 0
        self.cases, self.requests, self.evidence = set(), 0, []
        self.seen_pids = set()

    def _retire(self):
        worker = self.worker
        # WorkspaceWorker can fail before Popen; it has then created no container.
        if worker is not None and not hasattr(worker, 'process') and hasattr(worker, 'stderr'):
            worker.stderr.close()
        if worker is not None and hasattr(worker, 'process'):
            worker.close()
            cleanup = json.loads((worker.evidence / 'cleanup.json').read_text())
            if cleanup.get('removal_confirmed') is not True or cleanup['container'] != worker.name:
                raise CleanupUncertain('private_grader_pool_cleanup_identity_failed')
            for directory in self.evidence:
                atomic_json(directory / 'cleanup.json', cleanup)
        self.worker = None
        # Projection scratch is deleted only after confirmed container removal.
        if hasattr(self, 'projection') and self.projection.exists():
            shutil.rmtree(self.projection)
        self.evidence = []

    def poison(self, reason):
        with self.lock:
            self.poisoned = True
            atomic_json(self.directory / 'poison.json', {'reason': str(reason), 'generation': self.generation})
            self._retire()

    def grade(self, *, snapshot, checker, entries, evidence_dir, case_id):
        with self.lock:
            if self.closed or self.poisoned:
                raise IsolationError('private_grader_pool_closed_or_poisoned')
            evidence = Path(evidence_dir)
            evidence.mkdir(parents=True, exist_ok=False)
            try:
                boundary = checker_boundary_contract(Path(checker) / 'test.py')
                if boundary['status'] != 'readonly_data_checker_static_boundary_pass':
                    raise IsolationError('private_grader_boundary_pending')
                if self.worker is not None and ((case_id not in self.cases and len(self.cases) >= MAX_CASES)
                                                or self.requests >= MAX_REQUESTS):
                    self._retire()
                if self.worker is None:
                    self.generation += 1
                    generation = self.directory / f'generation_{self.generation:04d}'
                    generation.mkdir(mode=0o700)
                    self.projection = generation / 'projection'
                    self.projection.mkdir(mode=0o700)
                    self.snapshot = self.projection / 'workspace'
                    self.checker = self.projection / 'checker'
                    self.snapshot.mkdir(mode=0o755); self.checker.mkdir(mode=0o755)
                    self.cases, self.requests, self.seen_pids = set(), 0, set()
                snapshot_manifest = replace_projection(snapshot, self.snapshot, entries)
                checker_manifest = replace_projection(checker, self.checker)
                if checker_manifest['test.py']['sha256'] != boundary['checker_sha256']:
                    raise IsolationError('private_grader_boundary_bytes_changed')
                # Immutable, case-local source bytes survive later projection reuse.
                source_copy = evidence / 'source_snapshot'
                source_copy.mkdir(mode=0o755)
                replace_projection(self.checker, source_copy, checker_manifest)
                if self.worker is None:
                    self.worker = WorkspaceWorker.__new__(WorkspaceWorker)
                    self.worker.__init__(self.projection.parent / 'worker', snapshot=self.snapshot, checker=self.checker)
                self.evidence.append(evidence)
                isolation = json.loads((self.worker.evidence / 'isolation.json').read_text())
                atomic_json(evidence / 'isolation.json', isolation)
                nonce = uuid.uuid4().hex
                request = {'op': 'grade', 'request_nonce': nonce}
                proof = {'schema': 'private-data-grader-pool/v1', 'container': self.worker.name,
                         'request_nonce': nonce, 'generation': self.generation, 'case_id': case_id,
                         'pool_directory': str(self.directory.resolve()),
                         'pool_worker_evidence': str(self.worker.evidence.resolve()),
                         'source_snapshot': str(source_copy.resolve()),
                         'snapshot_manifest': snapshot_manifest, 'source_manifest': checker_manifest,
                         'snapshot_manifest_sha256': digest(snapshot_manifest),
                         'source_manifest_sha256': digest(checker_manifest),
                         'bind_roots': {'workspace': str(self.snapshot.resolve()), 'checker': str(self.checker.resolve())},
                         'bind_root_inodes': {'workspace': self.snapshot.stat().st_ino, 'checker': self.checker.stat().st_ino},
                         'all_projection_mounts_kernel_readonly': True,
                         'fresh_checker_child_each_request': True, 'tmp_cleared_before_grade': True,
                         'max_cases': MAX_CASES, 'max_grade_children': MAX_REQUESTS,
                         'boundary': boundary}
                atomic_json(evidence / 'request.json', request)
                atomic_json(evidence / 'projection.json', proof)
                response = self.worker.request(request, timeout=65)
                atomic_json(evidence / 'response.json', response)
                validate_response_identity(response, nonce)
                if response['result']['child_pid'] in self.seen_pids:
                    raise IsolationError('private_grader_child_pid_reused')
                self.seen_pids.add(response['result']['child_pid'])
                for original, projection, manifest in ((snapshot, self.snapshot, snapshot_manifest),
                                                        (checker, self.checker, checker_manifest)):
                    if snapshot_entries(original) != manifest or snapshot_entries(projection) != manifest:
                        raise IsolationError('private_grader_original_or_projection_changed')
                if proof['bind_root_inodes'] != {'workspace': self.snapshot.stat().st_ino, 'checker': self.checker.stat().st_ino}:
                    raise IsolationError('private_grader_bind_root_identity_changed')
                proof.update(child_pid=response['result']['child_pid'],
                             original_and_projection_manifests_verified_after_grade=True)
                atomic_json(evidence / 'projection.json', proof)
                self.cases.add(case_id)
                self.requests += 1
                return response, proof
            except BaseException as exc:
                try:
                    self.poison(repr(exc))
                except BaseException as cleanup:
                    atomic_json(self.directory / 'cleanup_error.json', {'error': repr(cleanup), 'original': repr(exc)})
                    raise CleanupUncertain('private_grader_pool_cleanup_uncertain:' + repr(cleanup)) from exc
                if isinstance(exc, (KeyboardInterrupt, SystemExit, IsolationError)):
                    raise
                raise IsolationError('private_grader_pool_contract_failed:' + repr(exc)) from exc

    def close(self):
        with self.lock:
            if self.closed:
                return
            self._retire()
            self.closed = True
            atomic_json(self.directory / 'cleanup.json', {'all_owned_containers_removal_confirmed': True,
                                                         'generations': self.generation, 'poisoned': self.poisoned})


class LaneGraderPools:
    """Lazy one pool per executor thread, with all ownership tracked centrally."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.local, self.lock, self.pools = threading.local(), threading.Lock(), []

    def get(self):
        if not hasattr(self.local, 'pool'):
            with self.lock:
                pool = PrivateGraderPool(self.directory / f'lane_{len(self.pools):02d}')
                self.pools.append(pool)
            self.local.pool = pool
        return self.local.pool

    def close(self):
        errors = []
        for pool in self.pools:
            try:
                pool.close()
            except BaseException as exc:
                errors.append({'pool': str(pool.directory), 'error': repr(exc)})
        atomic_json(self.directory / 'cleanup.json', {'all_owned_containers_removal_confirmed': not errors,
                                                    'pools': len(self.pools), 'errors': errors})
        if errors:
            raise CleanupUncertain('private_grader_lane_cleanup_unconfirmed:' + repr(errors))
