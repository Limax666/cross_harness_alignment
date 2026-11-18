"""v9: retain v8 owner projection and restore the already allowed runtime on PATH."""
from flashrsi_harnessrisk_opencode_session_supervisor_v8 import source

old = "'PATH':str(work/'bin')+':/usr/bin:/bin'"
if old not in source:
    raise RuntimeError('v9 tool PATH transform did not match')
source = source.replace(old, "'PATH':runtime_tool_path(work,sys.executable)")
source = source.replace('class Proxy:',
    'from flashrsi_harnessrisk_opencode_environment_v9 import runtime_tool_path\n\nclass Proxy:')
source = source.replace('private-v1-opencode-native-session-v8-user-anchor',
                        'private-v1-opencode-native-session-v9-user-anchor-pythonpath')
exec(compile(source, __file__, 'exec'))
