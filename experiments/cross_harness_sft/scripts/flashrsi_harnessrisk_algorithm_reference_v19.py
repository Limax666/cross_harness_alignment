"""v17 real-reference semantics with immediate uncertain-cleanup rollout stop."""
from pathlib import Path


def build_source():
    source = Path(__file__).with_name('flashrsi_harnessrisk_algorithm_reference_v17.py').read_text()
    old = 'from flashrsi_harnessrisk_benchmark_v17 import HarnessRiskBenchmark,ROOT'
    if source.count(old) != 1:
        raise RuntimeError('v19_cleanup_guard_import_transform_not_exact')
    return source.replace(old, 'from flashrsi_harnessrisk_benchmark_v18 import HarnessRiskBenchmark,ROOT')


if __name__ == '__main__':
    exec(compile(build_source(), __file__, 'exec'))
