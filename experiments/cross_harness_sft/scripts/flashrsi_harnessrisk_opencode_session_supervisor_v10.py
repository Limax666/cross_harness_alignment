"""v10: retain v9 isolation/PATH and forward the official mock client's case identity."""
from flashrsi_harnessrisk_opencode_session_supervisor_v9 import source

source = source.replace('class Proxy:',
    'from flashrsi_harnessrisk_opencode_mock_identity_v10 import mock_client_environment\n\nclass Proxy:')
old = "'TMPDIR':str(scratch),'MOCK_GATEWAY_BASE_URL':info['gateway']}"
if old not in source:
    raise RuntimeError('v10 mock environment transform did not match')
source = source.replace(old, old + "\n                    case_client_env=mock_client_environment(path/'state'/'.env',case)\n                    env.update(case_client_env)\n                    (path/'mock_client_environment_audit.json').write_text(json.dumps({'protocol':'opencode-mock-client-caseenv-v10','case_id':case,'forwarded_keys':sorted(case_client_env),'only_adapter_generated_fake_credentials':True})+'\\n')")
source = source.replace('private-v1-opencode-native-session-v9-user-anchor-pythonpath',
                        'private-v1-opencode-native-session-v10-user-anchor-pythonpath-caseenv')
exec(compile(source, __file__, 'exec'))
