"""Project-owned rootless Docker tools; no host shell, service socket or network in tasks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import uuid

from .backends import sha256_file, write_json

PUBLIC_WORKSPACE = '/root/.openclaw/workspace'
MAX_CAPTURE = 1_048_576
MAX_FILE = 32 * 1024 * 1024


class IsolationError(RuntimeError):
    pass


def scoped_path(root: Path, name: str) -> Path:
    root = root.resolve(strict=True)
    if name == PUBLIC_WORKSPACE:
        name = '.'
    elif name.startswith(PUBLIC_WORKSPACE + '/'):
        name = name[len(PUBLIC_WORKSPACE) + 1:]
    candidate = Path(name)
    if candidate.is_absolute() or '..' in candidate.parts:
        raise ValueError('workspace_path_escape')
    current = root
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('workspace_symlink_rejected')
    if not current.resolve().is_relative_to(root):
        raise ValueError('workspace_path_escape')
    return current


def _schema(name, description, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties,
                       'required': required, 'additionalProperties': False}}}


def safe_snapshot(source: Path, destination: Path):
    entries, total = {}, 0
    for path in sorted(source.rglob('*')):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise IsolationError('unsupported_workspace_entry:' + str(path.relative_to(source)))
        if path.is_file():
            total += path.stat().st_size
            if path.stat().st_size > MAX_FILE or total > 128 * 1024 * 1024 or len(entries) >= 4096:
                raise IsolationError('workspace_snapshot_limit')
            entries[path.relative_to(source).as_posix()] = sha256_file(path)
    shutil.copytree(source, destination, symlinks=True)
    return {'files': entries, 'bytes': total,
            'sha256': hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()}


class WorkspaceTools:
    schemas = [
        _schema('read_file', 'Read a UTF-8 workspace file; offset and limit are characters.',
                {'path': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0},
                 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 65536}}, ['path']),
        _schema('write_file', 'Write a UTF-8 file inside the task workspace.',
                {'path': {'type': 'string'}, 'content': {'type': 'string'}}, ['path', 'content']),
        _schema('run_python', 'Run Python code in the isolated task workspace without network access.',
                {'code': {'type': 'string'}}, ['code']),
        _schema('run_shell', 'Run a shell command in the isolated task workspace without network access.',
                {'command': {'type': 'string'}}, ['command']),
    ]

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve(strict=True)
        self.image = os.environ.get('HSRL_TOOL_IMAGE', '')
        if not self.image.startswith('sha256:') or len(self.image) != 71:
            raise IsolationError('frozen_tool_image_id_required')
        socket = Path(f'/run/user/{os.getuid()}/docker.sock')
        if not socket.is_socket() or socket.stat().st_uid != os.getuid():
            raise IsolationError('project_owned_rootless_docker_socket_required')
        self.host = 'unix://' + str(socket)
        self.env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'DOCKER_HOST': self.host,
                    'HOME': str(Path.home()), 'XDG_RUNTIME_DIR': str(socket.parent)}
        status = self._docker(['info', '--format', '{{json .SecurityOptions}}'], timeout=15)
        if status.returncode or 'name=rootless' not in json.loads(status.stdout):
            raise IsolationError('rootless_docker_not_verified')
        image = self._docker(['image', 'inspect', self.image, '--format', '{{.Id}}'], timeout=15)
        if image.returncode or image.stdout.strip() != self.image:
            raise IsolationError('frozen_tool_image_missing')
        self.containers = set()

    def _docker(self, args, timeout):
        return subprocess.run(['docker', *args], env=self.env, capture_output=True,
                              text=True, timeout=timeout)

    def _execute(self, argv, mounts, *, grader=False):
        name = f'hsrl-v4-{os.getuid()}-{uuid.uuid4().hex}'
        # This daemon has no cfs quota delegation. A real per-process CPU rlimit,
        # wall timeout, memory and PID bounds are explicit, rather than ignored --cpus.
        command = ['run', '--rm', '--name', name, '--label', 'hsrl.owner=' + str(os.getuid()),
                   '--network=none', '--read-only', '--cap-drop=ALL',
                   '--security-opt=no-new-privileges', '--pids-limit=64', '--memory=2g',
                   '--memory-swap=2g', '--ulimit=cpu=30:30', '--ulimit=nofile=1024:1024',
                   '--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=128m',
                   '--workdir=' + ('/grader' if grader else PUBLIC_WORKSPACE),
                   '--env=HOME=/tmp', '--env=PYTHONDONTWRITEBYTECODE=1',
                   '--env=PYTHONUNBUFFERED=1']
        for source, target, readonly in mounts:
            if ',' in str(source):
                raise IsolationError('unsupported_mount_path')
            command += ['--mount', f'type=bind,src={source},dst={target}' + (',readonly' if readonly else '')]
        command += [self.image, *argv]
        self.containers.add(name)
        try:
            # Output goes to temporary files so an untrusted command cannot exhaust
            # the trainer's RAM via communicate(). Only bounded bytes are returned.
            with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
                process = subprocess.Popen(['docker', *command], env=self.env, stdout=out, stderr=err)
                try:
                    code = process.wait(timeout=90 if grader else 45)
                except subprocess.TimeoutExpired:
                    self._docker(['rm', '-f', name], timeout=15)
                    process.kill()
                    process.wait(timeout=15)
                    raise IsolationError('private_grader_wall_timeout' if grader else 'task_tool_wall_timeout')
                out.seek(0); err.seek(0)
                stdout, stderr = out.read(MAX_CAPTURE + 1), err.read(MAX_CAPTURE + 1)
                if len(stdout) > MAX_CAPTURE or len(stderr) > MAX_CAPTURE:
                    raise IsolationError('tool_output_capture_limit')
                # Docker failures 125/126/127 are infra, never model utility zero.
                if code in (125, 126, 127, 137):
                    raise IsolationError(f'container_execution_failed:{code}:' + stderr.decode(errors='replace')[:1500])
                return {'valid': True, 'returncode': code,
                        'stdout': stdout.decode(errors='replace'), 'stderr': stderr.decode(errors='replace'),
                        'isolation': {'engine': 'project_owned_rootless_docker', 'image_id': self.image,
                                      'network': 'none', 'rootfs': 'readonly', 'grader': grader,
                                      'cpu_limit': '30sec_per_process_rlimit', 'wall_seconds': 90 if grader else 45,
                                      'memory_bytes': 2 * 1024 ** 3, 'pids': 64,
                                      'bind_targets': [target for _, target, _ in mounts]}}
        finally:
            # An errored Docker client does not prove its container has exited.
            # Cleanup addresses only the unique container this invocation created.
            removed = self._docker(['rm', '-f', name], timeout=15)
            if removed.returncode and 'No such container' not in removed.stderr:
                raise IsolationError('owned_container_cleanup_failed:' + name)
            self.containers.discard(name)

    def call(self, name, arguments):
        if name == 'read_file':
            path = scoped_path(self.workspace, arguments['path'])
            if not path.is_file():
                return json.dumps({'error': 'file_not_found'})
            if path.stat().st_size > MAX_FILE:
                return json.dumps({'error': 'file_size_limit'})
            try:
                text = path.read_text()
            except UnicodeError:
                return json.dumps({'error': 'not_utf8'})
            offset, limit = arguments.get('offset', 0), arguments.get('limit', 65536)
            return json.dumps({'content': text[offset:offset+limit], 'offset': offset,
                               'truncated': len(text) > offset + limit}, ensure_ascii=False)
        if name == 'write_file':
            path = scoped_path(self.workspace, arguments['path'])
            if len(arguments['content'].encode()) > MAX_FILE:
                return json.dumps({'error': 'write_size_limit'})
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(arguments['content'])
            return json.dumps({'written': len(arguments['content'].encode()), 'path': arguments['path']})
        if name == 'run_python':
            argv = ['python3', '-c', arguments['code']]
        elif name == 'run_shell':
            argv = ['/bin/sh', '-c', arguments['command']]
        else:
            raise ValueError('unknown_workspace_tool')
        return json.dumps(self._execute(argv, [(self.workspace, PUBLIC_WORKSPACE, False)]), ensure_ascii=False)

    def grade_private(self, checker_dir: Path, entry: Path):
        if entry.is_absolute() or '..' in entry.parts:
            raise IsolationError('grader_entry_escape')
        private = self.workspace.parent / 'private_grader'
        private.mkdir(exist_ok=False)
        snapshot = private / 'workspace_snapshot'
        identity = safe_snapshot(self.workspace, snapshot)
        # Source checker is never mounted during target execution. A frozen final
        # snapshot, not the mutable task workspace, is exposed to the checker.
        result = self._execute(['python3', str(Path('/grader') / entry), PUBLIC_WORKSPACE],
            [(snapshot, PUBLIC_WORKSPACE, True), (checker_dir.resolve(strict=True), '/grader', True)], grader=True)
        if identity != self._snapshot_identity(snapshot):
            raise IsolationError('grader_modified_snapshot')
        result['snapshot'] = identity
        result['snapshot_path'] = str(snapshot)
        write_json(private / 'grader_run.json', result)
        return result

    @staticmethod
    def _snapshot_identity(path):
        files = {p.relative_to(path).as_posix(): sha256_file(p) for p in sorted(path.rglob('*')) if p.is_file()}
        return {'files': files, 'bytes': sum(p.stat().st_size for p in path.rglob('*') if p.is_file()),
                'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}

    def close(self):
        for name in list(self.containers):
            self._docker(['rm', '-f', name], timeout=15)
            self.containers.discard(name)
