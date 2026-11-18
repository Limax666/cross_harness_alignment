"""v13: use existing native proactive reclamation, retain v12 environment/isolation."""
from flashrsi_harnessrisk_supervisor_hermes_workspace_v12 import source, case_tool_environment
from flashrsi_harnessrisk_hermes_context_profile_v13 import native_context_profile

old = "'config':{'context_length':16384,'llm':"
if old not in source:
    raise RuntimeError('Hermes v13 native compression transform did not match')
source = source.replace(old, "'config':{'context_length':16384,'compression':native_context_profile(),'llm':")
source = source.replace('private-v1-hermes-contextfit-v12-caseenv',
                        'private-v1-hermes-contextfit-v13-native-prune')
exec(compile(source, __file__, 'exec'))
