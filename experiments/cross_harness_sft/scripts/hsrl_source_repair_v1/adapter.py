"""Original source/grader contracts with a narrowly versioned tool-worker fix.

Constructor ASTs intentionally equal their frozen counterparts. Original
backend call/grade/close and worker request/cleanup methods are inherited.
"""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import selectors
import subprocess
import uuid
from hsrl_v5.io import atomic_json, canonical, digest, file_hash
from hsrl_v5.isolation import JSONWorker as FrozenJSONWorker, IMAGE, IsolationError, CleanupUncertain
from hsrl_v5.agentdog_backend import AgentDogBackend as FrozenAgentDogBackend, GRADER_PATH, PROFILE
from hsrl_v5.agentdog_source import source_contract, public_contract
from hsrl_v5.agentdog_worker import execution_payload, materialize_scenario
from hsrl_v5.dataset import Dataset as FrozenDataset

class JSONWorker(FrozenJSONWorker):
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


class AgentDogBackend(FrozenAgentDogBackend):
    def __init__(self, record, evidence_dir, *, grader_path=GRADER_PATH, tool_worker=None, grader_worker=None):
        self.record, self.evidence = record, Path(evidence_dir)
        self.evidence.mkdir(parents=True, exist_ok=False)
        source = Path(record['source_path'])
        if file_hash(source) != record['source_sha256']:
            raise ValueError('source_asset_bytes_changed')
        self.bundle = json.loads(source.read_text())
        if digest(self.bundle) != record['source_record_canonical_sha256']:
            raise ValueError('source_bundle_identity_changed')
        paired = {'task_id': record['id'], 'catalog_task_type': record['condition'], 'scenario': record['scenario']}
        checked = source_contract(self.bundle, paired)
        if not checked['qualified'] or not record['qualified']:
            raise ValueError('source_not_structurally_admitted:' + ','.join(checked['exclusion_reasons']))
        self.scenario = materialize_scenario(self.bundle, record['scenario'])
        if self.scenario.get('tool_result_overlays'):
            raise ValueError('source_overlays_require_runtime_adaptation')
        public = public_contract(self.bundle, record['scenario'])
        self.messages, self.tools = public['messages'], public['tools']
        self.schemas = {row['function']['name']: row['function']['parameters'] for row in self.tools}
        self.initial_state = copy.deepcopy(self.scenario['initial_state_template'])
        self.state, self.history, self.protocol_errors, self.execution_errors = copy.deepcopy(self.initial_state), [], [], []
        self.grader_path, self.closed, self.finished = Path(grader_path), False, False
        self.poisoned, self.poison_reason, self.grade_count = False, None, 0
        self.borrowed_tools, self.shared_grader = tool_worker is not None, grader_worker
        self.worker = None
        atomic_json(self.evidence / 'public_interface.json', public)
        atomic_json(self.evidence / 'provenance.json', {
            'source_id': record['id'], 'source_sha256': record['source_sha256'],
            'source_record_canonical_sha256': record['source_record_canonical_sha256'],
            'source_identity': record['source_identity'], 'scenario': record['scenario'],
            'source_effective_risk': bool(self.scenario.get('risk_enabled')),
            'registered_call_goal_profile': PROFILE, 'public_interface_sha256': digest(public),
            'initial_state_sha256': digest(self.initial_state),
            'legal_utility_applicable': record['condition'] != 'query_target',
            'risk_call_applicable': record['condition'] != 'clean',
            'actual_harm_oracle_applicable': False, 'persistence_oracle_applicable': False,
            'judge_safety_applicable': False, 'grader_sha256': file_hash(self.grader_path)})
        try:
            self.worker = tool_worker if tool_worker is not None else JSONWorker(self.evidence / 'tool_worker')
            response = self.worker.request({'op': 'init', 'payload': execution_payload(self.bundle, record['scenario'])}, timeout=20)
            if not response.get('ok') or digest(response.get('state')) != digest(self.initial_state):
                raise IsolationError('source_initial_state_not_preserved')
            atomic_json(self.evidence / 'isolation.json', {
                'engine': 'project_owned_rootless_docker', 'image_id': self.worker.command[self.worker.command.index('python3')-1],
                'network': 'none', 'rootfs': 'readonly', 'user': '65534:65534',
                'private_grader_mounted_in_tool_worker': False,
                'only_tool_worker_code_mounted': True,
                'new_state_and_tool_namespaces_per_episode': True,
                'supervised_new_source_os_process': self.borrowed_tools,
                'source_process_id': response.get('source_process_id')})
        except BaseException as exc:
            self._poison(exc)
            if self.worker:
                self.worker.close()
            raise


class Dataset(FrozenDataset):
    @contextmanager
    def episode(self, key, evidence_dir):
        record = self.record(key)
        if record['source'] != 'agentdog':
            with super().episode(key, evidence_dir) as backend:
                yield backend
            return
        from hsrl_v5.release import GRADER_SHA256
        grader = self.verify_file(self.manifest['agentdog_grader_path'])
        if file_hash(grader) != GRADER_SHA256:
            raise ValueError('release_private_grader_pin_changed')
        backend = AgentDogBackend(record, evidence_dir, grader_path=grader)
        try:
            yield backend
        finally:
            backend.close()
