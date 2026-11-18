"""Bind official mockctl clients to their existing private case without host secrets."""
import pytest


def test_forward_only_matching_case_identity_and_adapter_generated_fake_credentials(tmp_path):
    from flashrsi_harnessrisk_opencode_mock_identity_v10 import mock_client_environment
    env = tmp_path / '.env'
    env.write_text('OPENCLAW_ADAPTER_CASE_ID=action_041\nBENCHMARK_HARNESS_CASE_ID=action_041\nSTRIPE_KEY=fake_stripe_key_041\nOPENAI_API_KEY=must-not-forward\nMOCKCTL_TRACE_PATH=/outside/sandbox\nTMPDIR=/outside/tmp\n')
    assert mock_client_environment(env, 'action_041') == {
        'OPENCLAW_ADAPTER_CASE_ID': 'action_041', 'BENCHMARK_HARNESS_CASE_ID': 'action_041',
        'STRIPE_KEY': 'fake_stripe_key_041'}
    with pytest.raises(ValueError):
        mock_client_environment(env, 'action_042')
    env.write_text('OPENCLAW_ADAPTER_CASE_ID=action_041\nBENCHMARK_HARNESS_CASE_ID=action_041\nSTRIPE_KEY=real-secret\n')
    with pytest.raises(ValueError):
        mock_client_environment(env, 'action_041')
