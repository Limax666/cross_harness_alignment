"""Bounded tmpfs-backed public workspace and separate read-only grader."""
from __future__ import annotations
import base64
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import uuid

from .isolation import JSONWorker, IMAGE, IsolationError
from .io import atomic_json, file_hash
from .workspace_worker import safe_path, snapshot_entries, PUBLIC


def numeric_reward(value):
    if isinstance(value, dict) and value:
        values = list(value.values())
    elif isinstance(value, (int, float)):
        values = [value]
    else:
        raise ValueError('checker_result_not_finite_numeric_scores')
    if any(not isinstance(score, (int, float)) or not math.isfinite(score) for score in values):
        raise ValueError('checker_result_not_finite_numeric_scores')
    if any(not 0 <= score <= 1 for score in values):
        raise ValueError('official_reward_outside_unit_interval')
    score = sum(values) / len(values)
    if not 0 <= score <= 1:
        raise ValueError('official_reward_outside_unit_interval')
    return float(score)


def _tool(name, description, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}}}


SCHEMAS = [
    _tool('read_file', 'Read UTF-8 from the task workspace.', {'path': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0}, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 65536}}, ['path']),
    _tool('write_file', 'Write UTF-8 to the task workspace.', {'path': {'type': 'string'}, 'content': {'type': 'string'}}, ['path', 'content']),
    _tool('run_python', 'Run Python in the isolated task workspace.', {'code': {'type': 'string'}}, ['code']),
    _tool('run_shell', 'Run a shell command in the isolated task workspace.', {'command': {'type': 'string'}}, ['command']),
]


class WorkspaceWorker(JSONWorker):
    def __init__(self, evidence_dir, *, assets=None, snapshot=None, checker=None):
        self.name = f'hsrl-v5-files-{os.getuid()}-{uuid.uuid4().hex}'
        self.closed, self.buffer = False, bytearray()
        self.evidence = Path(evidence_dir)
        self.evidence.mkdir(parents=True, exist_ok=False)
        socket = Path(f'/run/user/{os.getuid()}/docker.sock')
        if not socket.is_socket() or socket.stat().st_uid != os.getuid():
            raise IsolationError('owned_rootless_docker_socket_missing')
        self.env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(Path.home()),
                    'XDG_RUNTIME_DIR': str(socket.parent), 'DOCKER_HOST': 'unix://' + str(socket)}
        info = self._docker(['info', '--format', '{{json .SecurityOptions}}'])
        if info.returncode or 'name=rootless' not in json.loads(info.stdout):
            raise IsolationError('rootless_engine_not_verified')
        identity = self._docker(['image', 'inspect', IMAGE, '--format', '{{.Id}}'])
        if identity.returncode or identity.stdout.strip() != IMAGE:
            raise IsolationError('pinned_tool_image_missing')
        grader = checker is not None
        if grader != (snapshot is not None) or grader == (assets is not None):
            raise ValueError('workspace_worker_mount_role_conflict')
        mounts = [(Path(__file__).with_name('workspace_worker.py'), '/runner/worker.py')]
        if grader:
            mounts.extend([(Path(snapshot), str(PUBLIC)), (Path(checker), '/grader')])
        else:
            mounts.append((Path(assets), '/inputs'))
        command = ['docker', 'run', '--rm', '-i', '--name', self.name,
                   '--label', f'hsrl.v5.owner={os.getuid()}', '--log-driver=none',
                   '--network=none', '--read-only', '--user=65534:65534',
                   '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=64',
                   '--memory=1g', '--memory-swap=1g', '--ulimit=cpu=60:60', '--ulimit=nofile=512:512',
                   '--tmpfs=/root:rw,noexec,nosuid,nodev,size=256m,mode=1777',
                   '--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=128m,mode=1777',
                   '--env=HOME=/tmp', '--env=PYTHONDONTWRITEBYTECODE=1', '--env=PYTHONHASHSEED=0']
        for source, target in mounts:
            source = source.resolve(strict=True)
            if ',' in str(source):
                raise ValueError('unsupported_mount_path')
            command += ['--mount', f'type=bind,src={source},dst={target},readonly']
        command += [IMAGE, 'python3', '-u', '/runner/worker.py']
        if grader:
            command.append('--grader')
        self.command = command
        self.stderr = (self.evidence / 'worker.stderr.log').open('ab')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, env=self.env, bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        os.set_blocking(self.process.stdin.fileno(), False)
        try:
            ready = self.read(timeout=30)
            if ready != {'ready': True, 'mode': 'file_grader' if grader else 'file_tools'}:
                raise IsolationError('file_worker_start_contract')
            atomic_json(self.evidence / 'isolation.json', {
                'image': IMAGE, 'container': self.name, 'rootless': True, 'network': 'none',
                'rootfs': 'readonly', 'all_host_mounts': 'readonly', 'workspace': '256MiB_tmpfs',
                'memory_bytes': 1073741824, 'pids': 64, 'private_grader': grader,
                'mount_targets': [target for _, target in mounts]})
        except BaseException:
            self.close()
            raise

    def export_snapshot(self, destination):
        result = self.request({'op': 'snapshot'})
        if not result.get('ok'):
            raise IsolationError('workspace_snapshot_contract:' + str(result))
        entries = result['result']['entries']
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        for name, expected in entries.items():
            path = safe_path(destination, name)
            if expected['bytes'] > 32 * 1024 * 1024:
                raise IsolationError('snapshot_file_size_limit')
            path.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            with path.open('wb') as stream:
                while count < expected['bytes']:
                    response = self.request({'op': 'read_chunk', 'path': name, 'offset': count})
                    if not response.get('ok'):
                        raise IsolationError('snapshot_chunk_failed')
                    data = base64.b64decode(response['result']['base64'], validate=True)
                    if not data or count + len(data) > expected['bytes']:
                        raise IsolationError('snapshot_chunk_size_mismatch')
                    stream.write(data)
                    count += len(data)
            if file_hash(path) != expected['sha256']:
                raise IsolationError('snapshot_file_hash_mismatch')
        if snapshot_entries(destination) != entries:
            raise IsolationError('snapshot_manifest_mismatch')
        return entries
