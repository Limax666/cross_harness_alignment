"""Independent v11 protocol: fix current-owner placement after compacted history."""
from flashrsi_harnessrisk_opencode_session_supervisor_v10 import source

old = 'from flashrsi_harnessrisk_opencode_owner_anchor_v8 import'
if old not in source:
    raise RuntimeError('v11 owner projection import transform did not match')
source = source.replace(old, 'from flashrsi_harnessrisk_opencode_owner_anchor_v11 import')
source = source.replace('private-v1-opencode-native-session-v10-user-anchor-pythonpath-caseenv',
                        'private-v1-opencode-native-session-v11-owner-suffix-pythonpath-caseenv')
exec(compile(source, __file__, 'exec'))
