"""Fresh full-128 initial/four-algorithm queue with a corrected tool environment."""
from pathlib import Path

source = Path(__file__).with_name('flashrsi_matrix_opencode_gpu1_v2_20261115.py').read_text()
changes = {
    "OUT = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v2_20261115'":
        "OUT = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v3_20261115'",
    "OLD = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v1_20261115'":
        "OLD = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v2_20261115'",
    'flashrsi_harnessrisk_host_opencode_session_v8.py': 'flashrsi_harnessrisk_host_opencode_session_v9.py',
    'flashrsi_harnessrisk_opencode_session_supervisor_v8.py': 'flashrsi_harnessrisk_opencode_session_supervisor_v9.py',
    "'scripts/flashrsi_harnessrisk_opencode_owner_anchor_v8.py',":
        "'scripts/flashrsi_harnessrisk_opencode_owner_anchor_v8.py',\n        'scripts/flashrsi_harnessrisk_opencode_environment_v9.py',",
    'private-v1-opencode-native-session-v8-user-anchor-gpu1-32k':
        'private-v1-opencode-native-session-v9-user-anchor-pythonpath-gpu1-32k',
    'v7 deterministic compaction can remove all user roles; Qwen chat template rejects those requests before inference':
        'v8 agent tool PATH omitted the already allowed runtime interpreter directory; repeated python: not found exhausted real tool budgets',
    'Separate audited provider projection restores exact current public owner as user only when native request has no user; raw SQLite/native request and actual forwarded request both retained':
        'Retain v8 audited owner projection; prepend existing runtime interpreter directory after workspace/bin, matching official PATH inheritance intent and existing OpenClaw adapter fix; no interpreter installation, permission or budget changes',
}
for old, new in changes.items():
    if old not in source:
        raise RuntimeError('v3 queue transform did not match: ' + old[:70])
    source = source.replace(old, new)
exec(compile(source, __file__, 'exec'))
