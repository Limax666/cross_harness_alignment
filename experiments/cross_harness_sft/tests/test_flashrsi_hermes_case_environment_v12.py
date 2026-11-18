"""Exercise the runtime/case boundary with the actual Python tool entrypoint."""
import json
from pathlib import Path
import subprocess
import sys


def test_case_tool_environment_keeps_current_case_and_runtime_without_host_secrets(tmp_path):
    from flashrsi_harnessrisk_supervisor_hermes_workspace_v12 import case_tool_environment
    path = tmp_path / 'fixture.env'
    path.write_text('OPENCLAW_ADAPTER_CASE_ID=recovery_126\nBENCHMARK_HARNESS_CASE_ID=recovery_126\nSTRIPE_KEY=fake_stripe_key_126\nOPENAI_API_KEY=do-not-forward\n')
    env = case_tool_environment(tmp_path / 'work', tmp_path / 'scratch', sys.executable,
                                'http://127.0.0.1:12345', path, 'recovery_126')
    result = subprocess.run(['python', '-c',
        'import os,json;print(json.dumps({k:os.environ.get(k) for k in ["OPENCLAW_ADAPTER_CASE_ID","STRIPE_KEY","OPENAI_API_KEY"]}))'],
        env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {'OPENCLAW_ADAPTER_CASE_ID': 'recovery_126',
        'STRIPE_KEY': 'fake_stripe_key_126', 'OPENAI_API_KEY': None}
    assert env['MOCK_GATEWAY_BASE_URL'] == 'http://127.0.0.1:12345'
    assert env['HOME'] == str(tmp_path / 'scratch')
    assert env['TMPDIR'] == str(tmp_path / 'scratch')
