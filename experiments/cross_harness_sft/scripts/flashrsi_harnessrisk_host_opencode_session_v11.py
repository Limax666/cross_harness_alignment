"""Independent v11 host, with unchanged relay budgets and isolation."""
from flashrsi_harnessrisk_host_opencode_session_v10 import source

source = source.replace('flashrsi_harnessrisk_opencode_session_supervisor_v10.py',
                        'flashrsi_harnessrisk_opencode_session_supervisor_v11.py')
source = source.replace('private-v1-opencode-native-session-v10-user-anchor-pythonpath-caseenv',
                        'private-v1-opencode-native-session-v11-owner-suffix-pythonpath-caseenv')
exec(compile(source, __file__, 'exec'))
