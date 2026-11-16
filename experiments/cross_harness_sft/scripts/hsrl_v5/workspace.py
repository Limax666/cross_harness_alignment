"""Bounded tmpfs-backed public workspace and separate read-only grader."""
from __future__ import annotations
import ast
import base64
import hashlib
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


CHECKER_BOUNDARY_VERSION = 'readonly_private_grader_no_actor_code_v2'
# Only data-oriented imports used by the pinned source are eligible. Unknown
# imports remain pending: importing an actor-written module is execution too.
DATA_CHECKER_IMPORTS = frozenset({
    '__future__', 'argparse', 'ast', 'base64', 'binascii', 'bisect', 'bz2',
    'calendar', 'codecs', 'collections', 'configparser', 'contextlib', 'copy', 'csv', 'dataclasses', 'datetime',
    'decimal', 'difflib', 'enum', 'fnmatch', 'fractions', 'functools', 'glob',
    'email', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'io', 'ipaddress', 'itertools', 'json', 'keyword',
    'locale', 'math', 'mimetypes', 'numbers', 'operator', 'os', 'pathlib',
    'ntpath', 'platform', 'posixpath', 'random', 're', 'shlex', 'shutil', 'stat', 'statistics', 'string', 'struct', 'sys',
    'tempfile', 'textwrap', 'time', 'tokenize', 'typing', 'unicodedata',
    'tarfile', 'traceback', 'urllib', 'uuid', 'warnings', 'xml', 'zipfile', 'zlib', 'zoneinfo',
    'bs4', 'cryptography', 'jinja2', 'lxml', 'numpy', 'openpyxl', 'pandas', 'yaml',
})
EXECUTION_MODULES = frozenset({
    'subprocess', 'importlib', 'runpy', 'code', 'codeop', 'multiprocessing',
    'ctypes', 'pickle', '_pickle', 'dill', 'cloudpickle', 'marshal', 'builtins',
    'shelve', 'joblib', 'pytest', 'doctest', 'posix', 'nt', 'pty', '_posixsubprocess', 'jinja2',
})
DYNAMIC_BUILTINS = frozenset({'exec', 'eval', 'compile', '__import__',
    '__builtins__', 'getattr', 'setattr', 'delattr', 'globals', 'locals', 'vars'})


def checker_boundary_contract(checker):
    """Read-only AST/import screen; never import or execute source checker code.

    This deliberately rejects capabilities even in dormant branches. It is a
    conservative compatibility gate, not a claim to sandbox arbitrary Python.
    """
    data = Path(checker).read_bytes()
    proof = {'version': CHECKER_BOUNDARY_VERSION,
             'checker_sha256': hashlib.sha256(data).hexdigest(),
             'source_execution_performed': False, 'evidence': [], 'reasons': [],
             'compatibility': {'private_original_checker': True,
                'readonly_output_snapshot': True, 'actor_code_in_grader': False,
                'writable_output_execution_supported': False}}
    def flag(node, reason, resolved):
        proof['evidence'].append({'line': getattr(node, 'lineno', None),
            'kind': type(node).__name__, 'expression': ast.unparse(node),
            'resolved': resolved, 'reason': reason})
    try:
        tree = ast.parse(data)
    except (SyntaxError, UnicodeError) as exc:
        proof['evidence'].append({'line': getattr(exc, 'lineno', None),
            'kind': type(exc).__name__, 'expression': str(exc),
            'resolved': '<parse_failed>', 'reason': 'checker_AST_parse_pending'})
        tree = None
    if tree is not None:
        nodes, aliases = list(ast.walk(tree)), {}
        for node in nodes:
            if isinstance(node, ast.Import):
                for item in node.names:
                    aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
                    root = item.name.split('.')[0]
                    if root in EXECUTION_MODULES or root not in DATA_CHECKER_IMPORTS:
                        flag(node, 'execution_or_unreviewed_import_pending', item.name)
                    if any('pickle' in part for part in item.name.split('.')):
                        flag(node, 'execution_or_deserialization_capability_pending', item.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                root = module.split('.')[0]
                if node.level or root in EXECUTION_MODULES or root not in DATA_CHECKER_IMPORTS or any(x.name == '*' for x in node.names):
                    flag(node, 'execution_or_unreviewed_import_pending', module)
                if any('pickle' in part for part in module.split('.')):
                    flag(node, 'execution_or_deserialization_capability_pending', module)
                for item in node.names:
                    aliases[item.asname or item.name] = module + '.' + item.name
        import_names = set(aliases)
        imported_module_roots = {name.split('.')[0] for name in aliases.values()}
        function_arguments = {node.arg for node in nodes if isinstance(node, ast.arg)}
        parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
        def resolve(node, seen=()):
            if isinstance(node, ast.Name):
                name = node.id
                if name in seen:
                    return name
                return aliases.get(name, name)
            if isinstance(node, ast.Attribute):
                return resolve(node.value) + '.' + node.attr
            return '<dynamic>'
        def literal_scratch(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                path = Path(node.value)
                return path.is_absolute() and path.parts[:2] == ('/', 'tmp') and '..' not in path.parts
            if isinstance(node, ast.Call) and resolve(node.func) == 'pathlib.Path' and node.args:
                return literal_scratch(node.args[0])
            return False
        # Resolve renamed functions/namespaces as well as import aliases.
        for _ in range(4):
            changed = False
            for node in nodes:
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                    value = node.value
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if isinstance(value, ast.Call) and resolve(value.func) in {'urllib.parse.urlparse', 'urllib.parse.urlsplit'}:
                        for target in targets:
                            if isinstance(target, ast.Name) and target.id not in import_names | function_arguments:
                                aliases[target.id] = 'urllib.parse._data_result'
                    elif isinstance(value, ast.Call):
                        for target in targets:
                            if isinstance(target, ast.Name) and aliases.get(target.id) == 'urllib.parse._data_result':
                                flag(node, 'ambiguous_data_namespace_rebinding_pending', target.id)
                    if isinstance(value, (ast.Name, ast.Attribute)):
                        resolved = resolve(value)
                        for target in targets:
                            if isinstance(target, ast.Name) and target.id in import_names:
                                flag(node, 'import_namespace_rebinding_pending', target.id)
                                continue
                            if isinstance(target, ast.Name) and target.id != resolved and aliases.get(target.id) != resolved:
                                aliases[target.id] = resolved
                                changed = True
            if not changed:
                break
        for node in nodes:
            if not isinstance(node, (ast.Name, ast.Attribute)) or not isinstance(node.ctx, ast.Load):
                continue
            resolved = resolve(node)
            root, _, tail = resolved.partition('.')
            name = resolved.rsplit('.', 1)[-1]
            if root in DYNAMIC_BUILTINS or (isinstance(node, ast.Name) and node.id in DYNAMIC_BUILTINS) or (isinstance(node, ast.Attribute) and node.attr.startswith('__') and node.attr not in {'__init__', '__name__', '__file__'}):
                flag(node, 'dynamic_code_or_reflection_pending', resolved)
            if root in EXECUTION_MODULES and root in imported_module_roots:
                flag(node, 'execution_or_deserialization_capability_pending', resolved)
            if root == 'os' and (name in {'system', 'popen', 'fork', 'forkpty', 'startfile'} or name.lstrip('_').startswith(('spawn', 'exec', 'posix_spawn'))):
                flag(node, 'process_execution_pending', resolved)
            if root == 'sys' and tail.split('.')[0] in {'path', 'modules', 'meta_path', 'path_hooks', 'path_importer_cache', 'settrace', 'setprofile', '_getframe', 'call_tracing'}:
                flag(node, 'dynamic_import_or_process_introspection_pending', resolved)
            if root == 'yaml' and name in {'load', 'load_all', 'unsafe_load', 'unsafe_load_all', 'full_load', 'full_load_all', 'Loader', 'UnsafeLoader', 'FullLoader', 'CLoader', 'CUnsafeLoader', 'CFullLoader'}:
                flag(node, 'unsafe_yaml_deserialization_pending', resolved)
            if (isinstance(node, ast.Attribute) and name in {'eval', 'query'} and not resolved.startswith('urllib.parse._data_result.')) or (root == 'pandas' and name in {'read_pickle', 'read_hdf', 'eval', 'query'}):
                flag(node, 'pandas_code_or_pickle_pending', resolved)
            if root == 'numpy' and any(part in {'ctypeslib', 'f2py', 'distutils'} for part in tail.split('.')):
                flag(node, 'numpy_code_loading_pending', resolved)
            if root == 'operator' and name in {'attrgetter', 'methodcaller'}:
                flag(node, 'dynamic_code_or_reflection_pending', resolved)
            if root == 'numpy' and name in {'load', 'loads', 'read_array'}:
                parent = parents.get(node)
                if not isinstance(parent, ast.Call) or parent.func is not node:
                    flag(node, 'escaped_numpy_deserializer_pending', resolved)
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            resolved = resolve(node.func)
            if resolved.startswith('numpy.') and resolved.rsplit('.', 1)[-1] in {'load', 'loads', 'read_array'}:
                values = [keyword.value for keyword in node.keywords if keyword.arg == 'allow_pickle']
                # Positional allow_pickle (numpy.load third argument), dynamic
                # kwargs or any nonliteral false cannot establish data-only IO.
                positional = node.args[2:3] if resolved.rsplit('.', 1)[-1] == 'load' else node.args[1:2]
                if any(keyword.arg is None for keyword in node.keywords) or any(not isinstance(value, ast.Constant) or value.value is not False for value in values + positional) or resolved.endswith('.loads'):
                    flag(node, 'numpy_pickle_loading_pending', resolved)
            name = resolved.rsplit('.', 1)[-1]
            mutator = name in {'write_text', 'write_bytes', 'mkdir', 'makedirs', 'unlink', 'rmdir', 'rmtree', 'touch'} or resolved.startswith(('os.', 'shutil.')) and name in {'rename', 'replace', 'truncate', 'chmod', 'chown', 'copy', 'copyfile', 'copy2', 'copytree', 'move'}
            if mutator:
                target = node.func.value if isinstance(node.func, ast.Attribute) and not resolved.startswith(('os.', 'shutil.')) else (node.args[0] if node.args else None)
                if not literal_scratch(target):
                    flag(node, 'readonly_snapshot_write_capability_pending', resolved)
            if name == 'open':
                if isinstance(node.func, ast.Name) or resolved in {'io.open', 'os.open'}:
                    modes = node.args[1:2]
                else:
                    modes = node.args[:1]
                modes += [keyword.value for keyword in node.keywords if keyword.arg == 'mode']
                target = (node.args[0] if node.args else None) if isinstance(node.func, ast.Name) or resolved in {'io.open', 'os.open'} else node.func.value
                if not literal_scratch(target) and (resolved == 'os.open' or any(keyword.arg is None for keyword in node.keywords) or any(not isinstance(mode, ast.Constant) or not isinstance(mode.value, str) or any(c in mode.value for c in 'wax+') for mode in modes)):
                    flag(node, 'readonly_snapshot_write_capability_pending', resolved)
    proof['evidence'] = sorted(proof['evidence'], key=lambda x: (x['line'] or 0, x['reason'], x['resolved']))
    proof['reasons'] = sorted({x['reason'] for x in proof['evidence']})
    proof['status'] = ('pending_unsupported_private_grader_boundary' if proof['reasons']
                       else 'readonly_data_checker_static_boundary_pass')
    return proof


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
        if checker is not None:
            boundary = checker_boundary_contract(Path(checker) / 'test.py')
            if boundary['status'] != 'readonly_data_checker_static_boundary_pass':
                raise ValueError('private_grader_boundary_pending:' + ','.join(boundary['reasons']))
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
