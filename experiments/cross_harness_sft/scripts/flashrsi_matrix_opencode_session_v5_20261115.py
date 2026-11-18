"""Frozen v5 OpenCode launcher; v4 orchestration with only the audited host changed."""
from pathlib import Path

source = Path(__file__).with_name("flashrsi_matrix_opencode_session_v4_20261114.py").read_text()
source = source.replace(
    'flashrsi_harnessrisk_host_opencode_session_v4.py',
    'flashrsi_harnessrisk_host_opencode_session_v5.py',
)
source = source.replace('native session v4 with builtin MIME', 'native session v5 deterministic compaction')
if 'flashrsi_harnessrisk_host_opencode_session_v5.py' not in source:
    raise RuntimeError('v4 launcher shape changed; refusing unreviewed transform')
exec(compile(source, 'flashrsi_matrix_opencode_session_v4_20261114.py', 'exec'))
