"""Read-only strict scanner including all three finite queue protocols."""
from flashrsi_matrix_repair_scan_v18 import build_source as prior_source


def build_source():
    source = prior_source()
    old = "('gpu1-finite-strict-matrix/v17', 'gpu1-finite-strict-matrix/v18')"
    if source.count(old) != 2:
        raise RuntimeError('scanner_v19_transform_drift')
    return source.replace(old, "('gpu1-finite-strict-matrix/v17', 'gpu1-finite-strict-matrix/v18', 'gpu1-finite-strict-matrix/v19')")


if __name__ == '__main__':
    exec(compile(build_source(), __file__, 'exec'))
