"""No source/model execution: only the host subprocess boundary is replaced."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import flashrsi_harnessrisk_benchmark_v17 as bridge
from flashrsi.core import Task


class Snapshot:
    id = 'actual-snapshot-identity'
    def save(self, path):
        path.write_text(json.dumps({'id': self.id}))


def test_failed_host_stderr_survives_strict_ranking_failure(tmp_path, monkeypatch):
    monkeypatch.setenv('HR_HARNESS', 'opencode')
    monkeypatch.setenv('HARNESSRISK_MODEL_PORT', '18764')
    monkeypatch.setenv('SHENGSUANYUN_API_KEY', 'private-never-record')
    class Process:
        pid, returncode = 123, 2
        async def wait(self): return self.returncode
    async def create(*args, **kwargs):
        kwargs['stdout'].write(b'actual host stdout'); kwargs['stdout'].flush()
        kwargs['stderr'].write(b'concrete host exception'); kwargs['stderr'].flush()
        return Process()
    monkeypatch.setattr(bridge.asyncio, 'create_subprocess_exec', create)
    benchmark = bridge.HarnessRiskBenchmark(tmp_path)
    task = Task(task_id='daily_025', question='actual owner', env='harnessrisk')
    result = asyncio.run(benchmark._run_once(SimpleNamespace(snapshot=Snapshot()), task))
    assert result.status == 'error' and result.error_message == 'concrete host exception'
    from dataclasses import replace
    benchmark.reject_unscorable = True
    with pytest.raises(RuntimeError, match='unscorable'):
        benchmark.grade([replace(result, task=task)])
    directory = tmp_path / 'evidence/00000001'
    assert (directory / 'host.stderr.log').read_bytes() == b'concrete host exception'
    proof = json.loads((directory / 'host_execution.json').read_text())
    assert proof['returncode'] == 2 and proof['status'] == 'error'
    assert proof['safe_environment'] == {'HR_HARNESS': 'opencode', 'HARNESSRISK_MODEL_PORT': '18764'}
    assert 'private-never-record' not in json.dumps(proof)


def test_success_preserves_official_metrics_and_snapshot_identity(tmp_path, monkeypatch):
    class Process:
        pid, returncode = 124, 0
        async def wait(self): return self.returncode
    async def create(*args, **kwargs):
        case = Path(args[-1]) / args[-4]
        case.mkdir()
        (case / 'harness_result.json').write_text(json.dumps({'snapshot_id': Snapshot.id, 'status': 'completed', 'turns': []}))
        (case / 'official_rule_evaluation.json').write_text(json.dumps({'runs': [{'case_id': 'daily_025',
            'metrics': {'utility_score': .5, 'attack_success_rate': 0, 'persistence_score': 0, 'detection_score': 1},
            'details': {'attack_success': {'passed': False}}}]}))
        return Process()
    monkeypatch.setattr(bridge.asyncio, 'create_subprocess_exec', create)
    benchmark = bridge.HarnessRiskBenchmark(tmp_path)
    task = Task(task_id='daily_025', question='owner', env='harnessrisk')
    result = asyncio.run(benchmark._run_once(SimpleNamespace(snapshot=Snapshot()), task))
    from dataclasses import replace
    graded = benchmark.grade([replace(result, task=task)])
    assert graded[0].metrics['utility_score'] == .5 and graded[0].metrics['verifier_coverage'] == 1
    assert json.loads((tmp_path / 'evidence/00000001/host_execution.json').read_text())['status'] == 'finished'


def test_timeout_keeps_already_written_host_bytes(tmp_path, monkeypatch):
    class Process:
        pid, returncode = 125, None
        killed = False
        async def wait(self):
            if not self.killed:
                raise TimeoutError()
            self.returncode = -9
            return -9
        def kill(self): self.killed = True
    async def create(*args, **kwargs):
        kwargs['stderr'].write(b'last message before timeout'); kwargs['stderr'].flush()
        return Process()
    monkeypatch.setattr(bridge.asyncio, 'create_subprocess_exec', create)
    benchmark = bridge.HarnessRiskBenchmark(tmp_path)
    task = Task(task_id='daily_025', question='owner', env='harnessrisk')
    result = asyncio.run(benchmark._run_once(SimpleNamespace(snapshot=Snapshot()), task))
    assert result.status == 'error' and result.error_message == 'HarnessRisk host timeout'
    folder = tmp_path / 'evidence/00000001'
    assert (folder / 'host.stderr.log').read_bytes() == b'last message before timeout'
    proof = json.loads((folder / 'host_execution.json').read_text())
    assert proof['timed_out'] is True and proof['returncode'] == -9
