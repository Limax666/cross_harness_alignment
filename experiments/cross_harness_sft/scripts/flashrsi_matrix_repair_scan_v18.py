"""Read-only original strict scanner including the new v18 finite queue."""
from pathlib import Path


def build_source():
    source = Path(__file__).with_name('flashrsi_matrix_repair_scan_v17.py').read_text()
    # The expression occurs twice (root collection and queue reporting).
    old = ".get('protocol') == 'gpu1-finite-strict-matrix/v17'"
    if source.count(old) != 2:
        raise RuntimeError('strict_scanner_v18_transform_drift')
    return source.replace(old, ".get('protocol') in ('gpu1-finite-strict-matrix/v17', 'gpu1-finite-strict-matrix/v18')")


if __name__ == '__main__':
    exec(compile(build_source(), __file__, 'exec'))
