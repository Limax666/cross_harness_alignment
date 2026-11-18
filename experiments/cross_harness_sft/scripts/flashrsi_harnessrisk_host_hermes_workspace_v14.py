"""v14 host: identical context policy, corrected supervisor namespace."""
from flashrsi_harnessrisk_host_hermes_workspace_v13 import source

source = source.replace('flashrsi_harnessrisk_supervisor_hermes_workspace_v13.py',
                        'flashrsi_harnessrisk_supervisor_hermes_workspace_v14.py')
source = source.replace('private-v1-hermes-contextfit-v13-native-prune',
                        'private-v1-hermes-contextfit-v14-native-prune')
exec(compile(source, __file__, 'exec'))
