"""Versioned HarnessRisk host evidence; original scoring and rollout preserved."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import time
from collections.abc import Mapping

from flashrsi.core import Trajectory
from flashrsi_harnessrisk_benchmark import HarnessRiskBenchmark as OriginalBenchmark, ROOT


class HarnessRiskBenchmark(OriginalBenchmark):
    async def _run_once(self, target, task):
        self._sequence += 1
        folder = self.run_dir / 'evidence' / f'{self._sequence:08d}'
        folder.mkdir(parents=True, mode=0o700)
        snap = target.snapshot
        snapfile = folder / 'snapshot.json'
        snap.save(snapfile)
        cases = folder / 'cases'
        cases.mkdir()
        command = [sys.executable, str(ROOT / 'scripts' / os.environ.get('HR_HOST_SCRIPT', 'flashrsi_harnessrisk_host.py')),
                   task.task_id, str(snapfile), snap.id, str(cases)]
        proof = {'protocol': 'harnessrisk-host-observation/v17', 'command': command, 'cwd': str(ROOT),
                 'case_id': task.task_id, 'snapshot_id': snap.id, 'started_time_ns': time.time_ns(),
                 'safe_environment': {key: os.environ.get(key) for key in ('HR_HARNESS', 'HARNESSRISK_MODEL_PORT')},
                 'timeout_seconds': 2460}
        started = time.monotonic()
        out, err, proc = b'', b'', None
        stdout_file = (folder / 'host.stdout.log').open('wb')
        stderr_file = (folder / 'host.stderr.log').open('wb')
        try:
            proc = await asyncio.create_subprocess_exec(*command, cwd=str(ROOT),
                stdout=stdout_file, stderr=stderr_file)
            proof['pid'] = proc.pid
            try:
                await asyncio.wait_for(proc.wait(), timeout=2460)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                proof.update(timed_out=True, returncode=proc.returncode, status='error',
                             error_message='HarnessRisk host timeout')
                return Trajectory(status='error', error_message='HarnessRisk host timeout')
            proof.update(timed_out=False, returncode=proc.returncode)
            err = (folder / 'host.stderr.log').read_bytes()
            path = cases / task.task_id
            evaluation, resultfile = path / 'official_rule_evaluation.json', path / 'harness_result.json'
            if proc.returncode or not evaluation.exists() or not resultfile.exists():
                detail = err.decode(errors='replace')[-2000:] or 'HarnessRisk episode failed'
                proof.update(status='error', error_message=detail,
                             evaluation_present=evaluation.exists(), harness_result_present=resultfile.exists())
                return Trajectory(status='error', error_message=detail)
            result = json.loads(resultfile.read_text())
            grade = json.loads(evaluation.read_text())['runs'][0]
            if result.get('snapshot_id') != snap.id or grade.get('case_id') != task.task_id or result.get('status') != 'completed':
                proof.update(status='error', error_message='snapshot identity or episode completion mismatch')
                return Trajectory(status='error', error_message=proof['error_message'])
            self._records[task.task_id] = grade
            turns = result.get('turns', [])
            proof['status'] = 'finished'
            return Trajectory(status='finished', output=turns[-1]['response_text'] if turns else '',
                messages=tuple(dict(m) if isinstance(m, Mapping) else {'role': 'event', 'content': str(m)}
                               for turn in turns for m in turn.get('messages', [])))
        except BaseException as exc:
            proof.update(status='error', exception_type=type(exc).__name__, error_message=str(exc))
            raise
        finally:
            proof['elapsed_seconds'] = time.monotonic() - started
            stdout_file.close()
            stderr_file.close()
            for filename in ('host.stdout.log', 'host.stderr.log'):
                (folder / filename).chmod(0o600)
            proof['stdout_file'], proof['stderr_file'] = 'host.stdout.log', 'host.stderr.log'
            destination = folder / 'host_execution.json'
            destination.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
            destination.chmod(0o600)
