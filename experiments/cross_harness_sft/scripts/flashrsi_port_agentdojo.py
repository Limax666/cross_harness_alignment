"""Run suite-specific FlashRSI Python-port candidates in isolated AgentDojo episodes."""
import asyncio
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

from flashrsi.core import Trajectory
from flashrsi_gepa_security import AgentDojoSecurityBenchmark, ROOT, _safe_name
from flashrsi_snapshot_guard import validate_snapshot_candidate

PORTS = {
    'opencode': ('flashrsi.agents.opencode', 'OpenCode', 'flashrsi_opencode_agentdojo_snapshot', 'make_opencode_snapshot', 'flashrsi_opencode_full_host_v2.py', 'AGENTS.md'),
    'openclaw': ('flashrsi.agents.openclaw', 'OpenClaw', 'flashrsi_openclaw_agentdojo_snapshot', 'make_openclaw_snapshot', 'flashrsi_openclaw_full_host.py', 'SOUL.md'),
    'hermes': ('flashrsi.agents.hermes', 'Hermes', 'flashrsi_hermes_agentdojo_snapshot', 'make_hermes_snapshot', 'flashrsi_hermes_full_host.py', 'SOUL.md'),
    'pi': ('flashrsi.agents.pi_agent', 'Pi', 'flashrsi_pi_agentdojo_snapshot', 'make_pi_snapshot', 'flashrsi_pi_full_host_v2.py', 'AGENTS.md'),
}


class PortBenchmark(AgentDojoSecurityBenchmark):
    def __init__(self, config, *, run_dir, host):
        super().__init__(config, run_dir=run_dir)
        self.host = host

    async def env_runner(self, target, task):
        with self._lock:
            self._case_sequence += 1
            sequence = self._case_sequence
        evidence = self.run_dir / 'evidence'; evidence.mkdir(parents=True, exist_ok=True)
        name = f'{sequence:08d}-{_safe_name(task.task_id)}'
        case_path = evidence / (name + '.json')
        snapshot_path = evidence / (name + '.snapshot.json')
        snapshot = target.snapshot
        snapshot.save(snapshot_path)
        env = {**os.environ, 'B0_TASK_ID':task.task_id,'B0_SEED':str(task.metadata['seed']),
               'B0_CASE_OUT':str(case_path),'B1_SNAPSHOT_JSON':str(snapshot_path),
               'B1_EXPECTED_SNAPSHOT_ID':snapshot.id}
        proc = await asyncio.create_subprocess_exec(sys.executable, str(ROOT/'scripts'/self.host), cwd=ROOT, env=env,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=390)
        except TimeoutError:
            proc.kill(); await proc.wait()
            return Trajectory(status='error', error_message='port episode host timeout')
        if proc.returncode or not case_path.is_file():
            return Trajectory(status='error', error_message=stderr.decode(errors='replace')[-2000:] or 'port host failed')
        record = json.loads(case_path.read_text())
        if record.get('task_id') != task.task_id or record.get('effective_snapshot_id') != snapshot.id:
            return Trajectory(status='error', error_message='port candidate evidence identity mismatch')
        (evidence / (name + '.binding.json')).write_text(json.dumps({
            'task_id': task.task_id, 'canonical_candidate_id': target._canonical_candidate_id,
            'effective_snapshot_id': snapshot.id,
        }, indent=2)+'\n')
        self._latest_records[task.task_id] = record
        return Trajectory(status=record.get('status','error'),output=record.get('output',''),
                          messages=tuple(record.get('trajectory_messages',())),
                          error_message=record.get('error_message') or record.get('unscorable_reason'),
                          usage=record.get('usage',{}))


def build_port(name, benchmark, config, policy):
    module, class_name, adapter, make_name, host, prompt = PORTS[name]
    base_class = getattr(importlib.import_module(module), class_name)
    make_snapshot = getattr(importlib.import_module(adapter), make_name)
    config = {**config, 'source_dir':str(Path(importlib.import_module(module).__file__).parent)}
    prototype = base_class(config)
    suites = {}
    for suite in ('workspace','travel','banking','slack'):
        case = next(t for t in benchmark.driver._case_map if t.startswith(suite+':') and t.endswith(':clean'))
        tools = benchmark.driver.start(case, 0)['tools']
        suites[suite] = make_snapshot(prototype, tools, policy)

    class SuiteAwarePort(base_class):
        def __init__(self):
            super().__init__(config)
            self._suite_snapshots = suites
            self._canonical_base = suites[sorted(suites)[0]]
            self._authorized_effective = {}
            self._canonical_candidate_id = None
            self.update(self._canonical_base)
            self._env_runner = benchmark.env_runner

        def _validate_snapshot(self, snapshot, *, baseline=None):
            authorized = self._authorized_effective.get(snapshot.id)
            if authorized is None:
                return super()._validate_snapshot(snapshot, baseline=baseline)
            suite, digest = authorized
            validate_snapshot_candidate(suites[suite], snapshot, {prompt}, prompt)
            if hashlib.sha256(snapshot.files[prompt].content).hexdigest() != digest:
                raise ValueError('effective prompt identity mismatch')

        async def rollout(self, tasks, *, snapshot=None):
            candidate = self.snapshot if snapshot is None else snapshot
            validate_snapshot_candidate(self._canonical_base, candidate, {prompt}, prompt)
            content = candidate.files[prompt].content
            digest = hashlib.sha256(content).hexdigest()
            self._canonical_candidate_id = candidate.id
            rows = []
            for task in tuple(tasks):
                suite = task.metadata.get('suite')
                if suite not in suites:
                    raise ValueError('unknown AgentDojo suite')
                base = suites[suite]
                with base.check_out() as directory:
                    (directory/prompt).write_bytes(content)
                    effective = base.commit(directory)._with_scope(candidate._scope)
                self._authorized_effective[effective.id] = (suite, digest)
                rows.extend(await super().rollout((task,), snapshot=effective))
            return tuple(rows)

    benchmark.host = host
    return SuiteAwarePort(), suites
