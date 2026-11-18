"""Independent v9 host; preserve the running/archived v8 source unchanged."""
from flashrsi_harnessrisk_host_opencode_session_v8 import source

source = source.replace('flashrsi_harnessrisk_opencode_session_supervisor_v8.py',
                        'flashrsi_harnessrisk_opencode_session_supervisor_v9.py')
source = source.replace('private-v1-opencode-native-session-v8-user-anchor',
                        'private-v1-opencode-native-session-v9-user-anchor-pythonpath')
if 'supervisor_v9.py' not in source:
    raise RuntimeError('v9 supervisor transform did not match')
exec(compile(source, __file__, 'exec'))
