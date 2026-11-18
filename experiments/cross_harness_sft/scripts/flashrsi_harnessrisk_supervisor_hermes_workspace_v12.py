"""Hermes v12: keep private workspace/16K context, repair public tool environment."""
from pathlib import Path
from flashrsi_harnessrisk_opencode_environment_v9 import runtime_tool_path
from flashrsi_harnessrisk_opencode_mock_identity_v10 import mock_client_environment


def case_tool_environment(visible, scratch, executable, gateway, env_file, case_id):
    env = {'PATH': runtime_tool_path(visible, executable), 'LANG': 'C.UTF-8',
           'HOME': str(scratch), 'TMPDIR': str(scratch), 'MOCK_GATEWAY_BASE_URL': gateway}
    env.update(mock_client_environment(env_file, case_id))
    return env


source = Path(__file__).with_name('flashrsi_harnessrisk_supervisor_hermes_workspace_v11.py').read_text()
old = "'environment':{'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','HOME':str(scratch),'TMPDIR':str(scratch),'MOCK_GATEWAY_BASE_URL':info['gateway']}"
if old not in source:
    raise RuntimeError('Hermes v12 environment transform did not match')
source = source.replace(old, "'environment':case_tool_environment(visible,scratch,sys.executable,info['gateway'],path/'state'/'.env',case)")
old_request = "                    request={'harness':'hermes'"
if old_request not in source:
    raise RuntimeError('Hermes v12 environment audit transform did not match')
source = source.replace(old_request,
    "                    (path/'mock_client_environment_audit.json').write_text(json.dumps({'protocol':'hermes-mock-client-caseenv-v12','case_id':case,'forwarded_keys':sorted(mock_client_environment(path/'state'/'.env',case)),'only_adapter_generated_fake_credentials':True})+'\\n')\n" + old_request)
source = source.replace("'privacy_protocol':PROTOCOL,", "'privacy_protocol':PROTOCOL,'execution_protocol':'private-v1-hermes-contextfit-v12-caseenv',")
exec(compile(source, __file__, 'exec'))
