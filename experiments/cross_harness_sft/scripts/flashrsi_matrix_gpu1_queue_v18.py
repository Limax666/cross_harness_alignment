"""Finite GPU1 v17 queue, corrected fresh Codex32K model metadata profile."""
from pathlib import Path
from flashrsi_harnessrisk_codex_context_v18 import CONTEXT, replace_once


def build_source():
    source = Path(__file__).with_name('flashrsi_matrix_gpu1_queue_v17.py').read_text()
    for old, new in (
        ("'codex': 'flashrsi_harnessrisk_host_native_v3.py'", "'codex': 'flashrsi_harnessrisk_host_codex_context_v18.py'"),
        ('scripts/flashrsi_matrix_gpu1_queue_v17.py', 'scripts/flashrsi_matrix_gpu1_queue_v18.py'),
        ("['scripts/flashrsi_harnessrisk_initial_v17.py']", "['scripts/flashrsi_harnessrisk_initial_v18.py']"),
        (".flashrsi_gpu1_finite_queue_v17.lock", ".flashrsi_gpu1_finite_queue_v18.lock"),
        ('gpu1-finite-strict-matrix/v17', 'gpu1-finite-strict-matrix/v18'),
        ('    return contract\n', "    if harness == 'codex':\n        contract['native_context_metadata'] = CONTEXT_METADATA\n    return contract\n"),
    ):
        source = replace_once(source, old, new)
    return source


if __name__ == '__main__':
    CONTEXT_METADATA = CONTEXT
    exec(compile(build_source(), __file__, 'exec'))
