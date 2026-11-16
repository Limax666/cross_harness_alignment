"""Bounded, project-owned rootless containers for dataset tools and private graders."""
from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import select
import subprocess
import time
import uuid

from .io import canonical, atomic_json

IMAGE = 'sha256:a5babb29ecb8d6cc68066e19adfae7ac5569a44162e0e97162edd90f46bf676c'


class IsolationError(RuntimeError):
    pass


class CleanupUncertain(IsolationError):
    """Do not create a replacement until this owned container is removed."""


class JSONWorker:
    def __init__(self, evidence_dir, *, grader_source=None, image=IMAGE, supervisor=False):
        self.name = f'hsrl-v5-{os.getuid()}-{uuid.uuid4().hex}'
        self.closed, self.buffer = False, bytearray()
        self.evidence = Path(evidence_dir)
        self.evidence.mkdir(parents=True, exist_ok=True)
        socket = Path(f'/run/user/{os.getuid()}/docker.sock')
        if not socket.is_socket() or socket.stat().st_uid != os.getuid():
            raise IsolationError('owned_rootless_docker_socket_missing')
        if image != IMAGE:
            raise IsolationError('unfrozen_tool_image')
        self.env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(Path.home()),
                    'XDG_RUNTIME_DIR': str(socket.parent), 'DOCKER_HOST': 'unix://' + str(socket)}
        info = self._docker(['info', '--format', '{{json .SecurityOptions}}'])
        if info.returncode or 'name=rootless' not in json.loads(info.stdout):
            raise IsolationError('rootless_engine_not_verified')
        identity = self._docker(['image', 'inspect', IMAGE, '--format', '{{.Id}}'])
        if identity.returncode or identity.stdout.strip() != IMAGE:
            raise IsolationError('pinned_tool_image_missing')
        if supervisor and grader_source is not None:
            raise IsolationError('source_supervisor_cannot_mount_grader')
        self.supervisor = supervisor
        argv = ['docker', 'run', '--rm', '-i', '--name', self.name, '--log-driver=none',
                '--label', f'hsrl.v5.owner={os.getuid()}', '--network=none', '--read-only',
                '--user=65534:65534', '--cap-drop=ALL', '--security-opt=no-new-privileges',
                '--pids-limit=32', '--memory=512m', '--memory-swap=512m',
                '--ulimit=cpu=30:30', '--ulimit=nofile=256:256',
                '--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=128m,mode=1777',
                '--env=HOME=/tmp', '--env=PYTHONDONTWRITEBYTECODE=1',
                '--env=PYTHONHASHSEED=0', '--env=PYTHONUNBUFFERED=1',
                '--mount', f'type=bind,src={Path(__file__).with_name("agentdog_worker.py").resolve()},dst=/runner/worker.py,readonly']
        if grader_source is not None:
            source = Path(grader_source).resolve(strict=True)
            if ',' in str(source):
                raise IsolationError('unsupported_mount_path')
            argv += ['--mount', f'type=bind,src={source},dst=/runner/source_evaluator.py,readonly']
        argv += [IMAGE, 'python3', '-u', '/runner/worker.py']
        if grader_source is not None:
            argv += ['--grader']
        elif supervisor:
            argv += ['--supervisor']
        self.stderr = (self.evidence / 'worker.stderr.log').open('ab')
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, env=self.env, bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        os.set_blocking(self.process.stdin.fileno(), False)
        self.command = argv
        try:
            ready = self.read(timeout=30)
            expected_mode = 'grader' if grader_source else 'tool_supervisor' if supervisor else 'tools'
            if not ready.get('ready') or ready['mode'] != expected_mode:
                raise IsolationError('worker_start_contract_invalid')
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup:
                atomic_json(self.evidence / 'startup_and_cleanup_error.json', {'startup': repr(exc), 'cleanup': repr(cleanup)})
                raise CleanupUncertain('worker_start_failed:' + repr(exc) + ';cleanup:' + repr(cleanup)) from exc
            raise

    def _docker(self, args):
        try:
            return subprocess.run(['docker', *args], env=self.env, capture_output=True, text=True, timeout=20)
        except (subprocess.SubprocessError, OSError) as exc:
            raise IsolationError('docker_transport_failed:' + repr(exc)) from exc

    def read(self, timeout=5):
        deadline = time.monotonic() + timeout
        while True:
            if b'\n' in self.buffer:
                raw, _, rest = self.buffer.partition(b'\n')
                self.buffer = bytearray(rest)
                try:
                    value = json.loads(raw)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise IsolationError('worker_response_not_json') from exc
                if not isinstance(value, dict):
                    raise IsolationError('worker_response_not_object')
                return value
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise IsolationError('source_worker_deadline')
            events = self.selector.select(min(remaining, .5))
            for key, _ in events:
                block = os.read(key.fileobj.fileno(), 65536)
                if not block:
                    raise IsolationError('source_worker_closed:' + str(self.process.poll()))
                self.buffer.extend(block)
                if len(self.buffer) > 8 * 1024 * 1024:
                    raise IsolationError('worker_response_size_limit')

    def request(self, value, *, timeout=5):
        data = (canonical(value) + '\n').encode()
        if len(data) > 8 * 1024 * 1024:
            raise IsolationError('worker_request_size_limit')
        try:
            if self.process.poll() is not None:
                raise IsolationError('source_worker_not_alive')
            deadline = time.monotonic() + timeout
            remaining_data = memoryview(data)
            while remaining_data:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise IsolationError('source_worker_write_deadline')
                if not select.select([], [self.process.stdin], [], min(remaining, .5))[1]:
                    continue
                try:
                    count = os.write(self.process.stdin.fileno(), remaining_data)
                except BlockingIOError:
                    continue
                if not count:
                    raise IsolationError('source_worker_write_closed')
                remaining_data = remaining_data[count:]
            return self.read(max(0, deadline - time.monotonic()))
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup:
                atomic_json(self.evidence / 'request_and_cleanup_error.json', {'request': repr(exc), 'cleanup': repr(cleanup)})
                raise CleanupUncertain('worker_request_failed:' + repr(exc) + ';cleanup:' + repr(cleanup)) from exc
            raise

    def close(self):
        if self.closed:
            return
        removed_ok = False
        try:
            removal_error = None
            try:
                removed = self._docker(['rm', '-f', self.name])
                if removed.returncode and 'No such container' not in removed.stderr:
                    removal_error = removed.stderr
            except IsolationError as exc:
                removal_error = repr(exc)
            # Timed-out removal may still be running in Docker. Confirm this
            # exact owned container disappeared instead of declaring success.
            deadline = time.monotonic() + 30
            while True:
                inspected = self._docker(['container', 'inspect', self.name])
                if inspected.returncode and 'No such' in inspected.stderr:
                    removed_ok = True
                    break
                if inspected.returncode or time.monotonic() >= deadline:
                    raise CleanupUncertain('owned_worker_cleanup_unconfirmed:' + self.name + ':' + str(removal_error))
                time.sleep(.25)
            atomic_json(self.evidence / 'cleanup.json', {'container': self.name, 'removal_confirmed': True, 'prior_removal_error': removal_error})
        finally:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=20)
            self.selector.close()
            for stream in (self.process.stdin, self.process.stdout, self.stderr):
                stream.close()
            self.closed = removed_ok

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
