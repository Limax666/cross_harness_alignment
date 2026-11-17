"""Same original benchmark/source metrics, strict new-profile receipt admission."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import time

from flashrsi.core import Trajectory
from flashrsi_harnessrisk_benchmark import HarnessRiskBenchmark as OriginalBenchmark, ROOT, DATA
from flashrsi_pi_owner_profile_v6 import PROFILE, ProfileError, byte_hash, save_json, validate_case, snapshot_source_map


class HarnessRiskBenchmark(OriginalBenchmark):
    async def _run_once(self, target, task):
        if os.environ.get('HR_HOST_SCRIPT') != 'flashrsi_harnessrisk_host_pi_native_owner_v6.py':
            raise ProfileError('native_owner_benchmark_requires_exact_new_host')
        self._sequence += 1
        folder = self.run_dir / 'evidence' / f'{self._sequence:08d}'
        folder.mkdir(parents=True, mode=0o700)
        snapshot = target.snapshot
        snapshot_file = folder / 'snapshot.json'
        snapshot.save(snapshot_file)
        cases = folder / 'cases'
        cases.mkdir()
        command = [sys.executable, str(ROOT / 'scripts/flashrsi_harnessrisk_host_pi_native_owner_v6.py'),
                   task.task_id, str(snapshot_file), snapshot.id, str(cases)]
        proof = {'profile': PROFILE, 'command': command, 'case_id': task.task_id,
                 'snapshot_id': snapshot.id, 'timeout_seconds': 2460, 'timed_out': False}
        started, process = time.monotonic(), None
        try:
            with (folder / 'host.stdout.log').open('wb') as out, (folder / 'host.stderr.log').open('wb') as err:
                process = await asyncio.create_subprocess_exec(*command, cwd=ROOT, stdout=out, stderr=err)
                proof['pid'] = process.pid
                try:
                    await asyncio.wait_for(process.wait(), timeout=2460)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    proof.update(timed_out=True, cleanup_confirmed=False)
                    raise ProfileError('host_cleanup_uncertain_stop_future_rollouts')
            proof['returncode'] = process.returncode
            host_receipt = cases / task.task_id / 'native_owner_host.json'
            host = json.loads(host_receipt.read_text()) if host_receipt.is_file() else {}
            if host.get('namespace_exited') is not True or host.get('model_requests_settled') is not True:
                proof['cleanup_confirmed'] = False
                raise ProfileError('case_namespace_cleanup_uncertain_stop_future_rollouts')
            if process.returncode:
                raise ProfileError('real_native_case_failed:' + str(process.returncode))
            raw = (DATA / (task.task_id + '.json')).read_bytes()
            if byte_hash(raw) != task.metadata['case_sha256']:
                raise ProfileError('original_case_bytes_changed')
            source_task = json.loads(raw)['task']
            owners = source_task.get('user_messages') or [source_task['user_message']]
            case = cases / task.task_id
            admission = validate_case(case, task.task_id, snapshot.id, owners, byte_hash(raw),
                                      native_source=snapshot_source_map(snapshot))
            self._records[task.task_id] = admission['source_grade']
            result = json.loads((case / 'harness_result.json').read_text())
            turns = result['turns']
            proof.update(measurement_valid=True, driver_status='owner_schedule_completed',
                         bounded_owners=admission['bounded_owners'])
            return Trajectory(status='finished', output=turns[-1]['response_text'],
                messages=tuple(m for turn in turns for m in turn['messages']),
                usage={'bounded_owners': admission['bounded_owners'], 'original_owners': len(owners)})
        except ProfileError as exc:
            proof.update(error=str(exc), measurement_valid=False)
            if proof.get('cleanup_confirmed') is False:
                raise
            return Trajectory(status='error', error_message=str(exc))
        finally:
            proof['elapsed_seconds'] = time.monotonic() - started
            save_json(folder / 'host_execution.json', proof)

    def grade(self, trajectories):
        # Original grade produces zeros for invalid trajectories when its flag
        # is false. This profile never permits that imputation, including final
        # full128 evaluation. Preserve failed traces and block numeric ranking.
        old = self.reject_unscorable
        self.reject_unscorable = True
        try:
            return super().grade(trajectories)
        finally:
            self.reject_unscorable = old
