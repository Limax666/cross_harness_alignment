"""Independent v13 host; tokenizer refuses overflow exactly as before."""
from flashrsi_harnessrisk_host_hermes_workspace_v12 import source

source = source.replace('flashrsi_harnessrisk_supervisor_hermes_workspace_v12.py',
                        'flashrsi_harnessrisk_supervisor_hermes_workspace_v13.py')
source = source.replace('private-v1-hermes-contextfit-v12-caseenv',
                        'private-v1-hermes-contextfit-v13-native-prune')
if 'supervisor_hermes_workspace_v13.py' not in source:
    raise RuntimeError('Hermes v13 supervisor transform did not match')
exec(compile(source, __file__, 'exec'))
