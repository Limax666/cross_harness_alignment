"""Abort future rollout after v17 records uncertain host cleanup."""
import json
from flashrsi_harnessrisk_benchmark_v17 import HarnessRiskBenchmark as ObservedBenchmark, ROOT


class HostCleanupUncertain(RuntimeError):
    pass


class HarnessRiskBenchmark(ObservedBenchmark):
    async def _run_once(self, target, task):
        trajectory = await super()._run_once(target, task)
        path = self.run_dir / 'evidence' / f'{self._sequence:08d}' / 'host_execution.json'
        receipt = json.loads(path.read_text())
        if receipt.get('timed_out') is True or receipt.get('cleanup_confirmed') is False:
            raise HostCleanupUncertain('host_descendant_cleanup_requires_identity_reconciliation:' + str(path))
        return trajectory
