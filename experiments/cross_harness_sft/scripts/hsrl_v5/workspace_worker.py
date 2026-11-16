"""Container-only file tools. Original checkers exist only in separate graders."""
from __future__ import annotations
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time

PUBLIC = Path('/root/.openclaw/workspace')
MAX_FILE = 32 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024


def safe_path(root, name):
    root = Path(root).resolve(strict=True)
    if name == str(PUBLIC):
        name = '.'
    elif name.startswith(str(PUBLIC) + '/'):
        name = name[len(str(PUBLIC)) + 1:]
    part = Path(name)
    if part.is_absolute() or '..' in part.parts:
        raise ValueError('workspace_path_escape')
    value = root
    for component in part.parts:
        value = value / component
        if value.is_symlink():
            raise ValueError('workspace_symlink_rejected')
    if not value.resolve().is_relative_to(root):
        raise ValueError('workspace_path_escape')
    return value


def snapshot_entries(root):
    root = Path(root)
    entries, total = {}, 0
    for path in sorted(root.rglob('*')):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError('unsupported_workspace_entry')
        if path.is_file():
            size = path.stat().st_size
            total += size
            if size > MAX_FILE or total > MAX_TOTAL or len(entries) >= 4096:
                raise ValueError('workspace_snapshot_limit')
            entries[path.relative_to(root).as_posix()] = {'bytes': size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return entries


def bounded_command(argv, *, grader=False):
    """Bound child output while reading, rather than after an unbounded file."""
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/tmp',
           'LANG': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1'}
    child = subprocess.Popen(argv, cwd=PUBLIC, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    output = {'stdout': bytearray(), 'stderr': bytearray()}
    selector = selectors.DefaultSelector()
    selector.register(child.stdout, selectors.EVENT_READ, 'stdout')
    selector.register(child.stderr, selectors.EVENT_READ, 'stderr')
    deadline = time.monotonic() + (60 if grader else 30)
    try:
        while selector.get_map():
            if time.monotonic() >= deadline:
                raise RuntimeError('private_grader_wall_timeout' if grader else 'task_command_wall_timeout')
            for key, _ in selector.select(.1):
                block = os.read(key.fileobj.fileno(), 65536)
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                output[key.data].extend(block)
                if sum(map(len, output.values())) > 1024 * 1024:
                    raise RuntimeError('command_output_limit')
        code = child.wait(timeout=max(.1, deadline - time.monotonic()))
        return {'returncode': code, **{key: bytes(value).decode('utf-8', errors='replace') for key, value in output.items()}}
    finally:
        # Also terminate descendants that closed their inherited output pipes.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if child.poll() is None:
            child.wait(timeout=5)
        selector.close()
        child.stdout.close()
        child.stderr.close()


def main():
    grader = '--grader' in sys.argv
    if not grader:
        PUBLIC.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree('/inputs', PUBLIC)
    def respond(value):
        print(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')), flush=True)
    respond({'ready': True, 'mode': 'file_grader' if grader else 'file_tools'})
    for line in sys.stdin:
        try:
            if len(line.encode()) > 8 * 1024 * 1024:
                raise ValueError('request_size_limit')
            request = json.loads(line)
            op = request['op']
            if grader:
                if op != 'grade':
                    raise ValueError('private_grader_operation')
                result = bounded_command(['python3', '/grader/test.py', str(PUBLIC)], grader=True)
            elif op == 'snapshot':
                result = {'entries': snapshot_entries(PUBLIC)}
            elif op == 'read_chunk':
                path = safe_path(PUBLIC, request['path'])
                if not path.is_file() or path.stat().st_size > MAX_FILE:
                    raise ValueError('snapshot_file_contract')
                with path.open('rb') as stream:
                    stream.seek(request.get('offset', 0))
                    result = {'base64': base64.b64encode(stream.read(131072)).decode()}
            elif op == 'call':
                name, args = request['name'], request['arguments']
                if name == 'read_file':
                    path = safe_path(PUBLIC, args['path'])
                    if not path.is_file() or path.stat().st_size > MAX_FILE:
                        result = {'error': 'file_not_found_or_size_limit'}
                    else:
                        text = path.read_text()
                        offset, limit = args.get('offset', 0), args.get('limit', 65536)
                        result = {'content': text[offset:offset+limit], 'offset': offset, 'truncated': len(text) > offset+limit}
                elif name == 'write_file':
                    path = safe_path(PUBLIC, args['path'])
                    data = args['content'].encode()
                    if len(data) > MAX_FILE:
                        raise ValueError('write_size_limit')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    result = {'written': len(data), 'path': args['path']}
                elif name in ('run_python', 'run_shell'):
                    argv = ['python3', '-c', args['code']] if name == 'run_python' else ['/bin/sh', '-c', args['command']]
                    result = bounded_command(argv)
                else:
                    raise ValueError('unknown_file_tool')
            else:
                raise ValueError('unknown_file_worker_operation')
            respond({'ok': True, 'result': result})
        except Exception as exc:
            respond({'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)[:1000]})


if __name__ == '__main__':
    main()
