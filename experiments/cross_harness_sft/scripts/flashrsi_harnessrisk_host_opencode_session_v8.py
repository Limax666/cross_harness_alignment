"""Independent v8 host; never mutate the active v7 implementation."""
from flashrsi_harnessrisk_host_opencode_session_v7 import source

source = source.replace('flashrsi_harnessrisk_opencode_session_supervisor_v7.py',
                        'flashrsi_harnessrisk_opencode_session_supervisor_v8.py')
source = source.replace('private-v1-opencode-native-session-v7-system-aware-compaction',
                        'private-v1-opencode-native-session-v8-user-anchor')
if 'supervisor_v8.py' not in source:
    raise RuntimeError('v8 host supervisor transform did not match')
exec(compile(source, __file__, 'exec'))
