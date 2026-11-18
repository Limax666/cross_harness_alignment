"""Full independent baseline/RSI queue after public fixture environment repair."""
from flashrsi_matrix_opencode_gpu1_v3_20261115 import source

changes = {
    "OUT = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v3_20261115'":
        "OUT = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v4_20261115'",
    "OLD = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v2_20261115'":
        "OLD = ROOT / 'outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v3_20261115'",
    'flashrsi_harnessrisk_host_opencode_session_v9.py': 'flashrsi_harnessrisk_host_opencode_session_v10.py',
    'flashrsi_harnessrisk_opencode_session_supervisor_v9.py': 'flashrsi_harnessrisk_opencode_session_supervisor_v10.py',
    "'scripts/flashrsi_harnessrisk_opencode_environment_v9.py',":
        "'scripts/flashrsi_harnessrisk_opencode_environment_v9.py',\n        'scripts/flashrsi_harnessrisk_opencode_mock_identity_v10.py',",
    'private-v1-opencode-native-session-v9-user-anchor-pythonpath-gpu1-32k':
        'private-v1-opencode-native-session-v10-user-anchor-pythonpath-caseenv-gpu1-32k',
    'v8 agent tool PATH omitted the already allowed runtime interpreter directory; repeated python: not found exhausted real tool budgets':
        'v8 lost runtime interpreter PATH; v9 real CLI execution exposed omitted public mock case identity causing default case_023 and strict route rejection; both are explicit environment omissions',
    'Retain v8 audited owner projection; prepend existing runtime interpreter directory after workspace/bin, matching official PATH inheritance intent and existing OpenClaw adapter fix; no interpreter installation, permission or budget changes':
        'Retain v8 audited owner projection and v9 runtime PATH; forward exact current public case IDs and adapter-generated fake fixture credentials only; preserve routing gate, no real credentials or new filesystem/network permissions',
}
for old, new in changes.items():
    if old not in source:
        raise RuntimeError('v4 queue transform did not match: ' + old[:70])
    source = source.replace(old, new)
exec(compile(source, __file__, 'exec'))
