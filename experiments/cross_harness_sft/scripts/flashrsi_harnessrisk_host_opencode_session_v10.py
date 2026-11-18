"""Independent v10 host with unchanged private-case routing and sandbox policy."""
from flashrsi_harnessrisk_host_opencode_session_v9 import source

source = source.replace('flashrsi_harnessrisk_opencode_session_supervisor_v9.py',
                        'flashrsi_harnessrisk_opencode_session_supervisor_v10.py')
source = source.replace('private-v1-opencode-native-session-v9-user-anchor-pythonpath',
                        'private-v1-opencode-native-session-v10-user-anchor-pythonpath-caseenv')
if 'supervisor_v10.py' not in source:
    raise RuntimeError('v10 supervisor transform did not match')
exec(compile(source, __file__, 'exec'))
