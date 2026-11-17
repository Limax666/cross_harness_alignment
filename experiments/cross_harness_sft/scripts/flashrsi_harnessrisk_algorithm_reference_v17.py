"""Real reference entry with versioned, durable host subprocess evidence."""
from pathlib import Path

source = Path(__file__).with_name('flashrsi_harnessrisk_algorithm_reference_v7.py').read_text()
# v7 reads the immutable algorithm body before applying its real-reference
# transforms. Change only that body's benchmark import in this entry's globals.
old = "source = Path(__file__).with_name('flashrsi_harnessrisk_algorithm.py').read_text()"
if source.count(old) != 1:
    raise RuntimeError('v17 reference source transform did not match exactly once')
original_import = 'from flashrsi_harnessrisk_benchmark import HarnessRiskBenchmark,ROOT'
replacement = old + "\nbenchmark_import = " + repr(original_import) + "\nif source.count(benchmark_import) != 1:\n    raise RuntimeError('v17 benchmark import transform did not match exactly once')\nsource = source.replace(benchmark_import, 'from flashrsi_harnessrisk_benchmark_v17 import HarnessRiskBenchmark,ROOT')"
source = source.replace(old, replacement)
exec(compile(source, __file__, 'exec'))
