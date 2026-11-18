"""Validate the actual expanded supervisor's injected environment audit."""
import json


def test_expanded_supervisor_has_all_audit_helpers(tmp_path):
    import flashrsi_harnessrisk_supervisor_hermes_workspace_v14 as supervisor
    env_file = tmp_path / 'fixture.env'
    env_file.write_text('OPENCLAW_ADAPTER_CASE_ID=daily_025\nBENCHMARK_HARNESS_CASE_ID=daily_025\nSTRIPE_KEY=fake_stripe_key_025\n')
    expression = "json.dumps({'case_id':case,'forwarded_keys':sorted(mock_client_environment(path/'state'/'.env',case)),'only_adapter_generated_fake_credentials':True})"
    (tmp_path / 'state').mkdir()
    (tmp_path / 'state' / '.env').write_text(env_file.read_text())
    audit = json.loads(eval(expression, supervisor.main.__globals__, {'path': tmp_path, 'case': 'daily_025'}))
    assert audit['case_id'] == 'daily_025'
    assert 'OPENCLAW_ADAPTER_CASE_ID' in audit['forwarded_keys']
    assert 'STRIPE_KEY' in audit['forwarded_keys']
    assert supervisor.native_context_profile() == {
        'proactive_prune_tokens': 8192, 'protect_last_n': 4,
        'proactive_prune_min_result_chars': 2048}
