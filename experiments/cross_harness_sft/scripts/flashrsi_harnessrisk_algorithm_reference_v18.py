"""v17 observations composed with the existing OpenCode GEPA v6 Worker."""
from pathlib import Path


def build_source():
    source = Path(__file__).with_name('flashrsi_harnessrisk_algorithm_opencode_gepa_v6.py').read_text()
    old = "source = Path(__file__).with_name('flashrsi_harnessrisk_algorithm.py').read_text()"
    original = 'from flashrsi_harnessrisk_benchmark import HarnessRiskBenchmark,ROOT'
    if source.count(old) != 1:
        raise RuntimeError('v18_algorithm_source_transform_not_exact')
    inject = old + '\nbenchmark_import = ' + repr(original) + "\nif source.count(benchmark_import) != 1:\n    raise RuntimeError('v18_benchmark_import_transform_not_exact')\nsource = source.replace(benchmark_import, 'from flashrsi_harnessrisk_benchmark_v17 import HarnessRiskBenchmark,ROOT')"
    return source.replace(old, inject)


if __name__ == '__main__':
    exec(compile(build_source(), __file__, 'exec'))
