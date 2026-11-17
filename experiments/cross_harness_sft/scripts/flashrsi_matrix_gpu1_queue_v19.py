"""Finite GPU1 queue with both new native actual32K metadata profiles."""
from flashrsi_matrix_gpu1_queue_v18 import build_source as prior_source
from flashrsi_harnessrisk_codex_context_v18 import CONTEXT as CODEX_CONTEXT, replace_once
from flashrsi_harnessrisk_openclaw_context_v4 import CONTEXT as OPENCLAW_CONTEXT


def build_source():
    source = prior_source()
    for old, new in (
        ("'openclaw': 'flashrsi_harnessrisk_host_openclaw_caseenv_v3.py'", "'openclaw': 'flashrsi_harnessrisk_host_openclaw_context_v4.py'"),
        ('scripts/flashrsi_matrix_gpu1_queue_v18.py', 'scripts/flashrsi_matrix_gpu1_queue_v19.py'),
        ("['scripts/flashrsi_harnessrisk_initial_v18.py']", "['scripts/flashrsi_harnessrisk_initial_v19.py']"),
        ('.flashrsi_gpu1_finite_queue_v18.lock', '.flashrsi_gpu1_finite_queue_v19.lock'),
        ('gpu1-finite-strict-matrix/v18', 'gpu1-finite-strict-matrix/v19'),
        ('    return contract\n', "    if harness == 'openclaw':\n        contract['native_context_metadata'] = OPENCLAW_CONTEXT_METADATA\n    return contract\n"),
    ):
        source = replace_once(source, old, new)
    return source


if __name__ == '__main__':
    CONTEXT_METADATA, OPENCLAW_CONTEXT_METADATA = CODEX_CONTEXT, OPENCLAW_CONTEXT
    exec(compile(build_source(), __file__, 'exec'))
