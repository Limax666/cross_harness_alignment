"""Regression coverage for OpenClaw's per-case mock service identity."""

from __future__ import annotations

from flashrsi_harnessrisk_openclaw_environment_supervisor_v2 import sandbox_case_environment


def test_sandbox_case_environment_keeps_only_current_case_service_variables():
    result = sandbox_case_environment(
        {
            "OPENCLAW_ADAPTER_CASE_ID": "daily_025",
            "BENCHMARK_HARNESS_CASE_ID": "daily_025",
            "MOCK_GATEWAY_BASE_URL": "http://127.0.0.1:53109",
            "STRIPE_KEY": "fake_stripe_key_025",
            "UNRELATED_HOST_VALUE": "must-not-cross-boundary",
        }
    )

    assert result["OPENCLAW_ADAPTER_CASE_ID"] == "daily_025"
    assert result["BENCHMARK_HARNESS_CASE_ID"] == "daily_025"
    assert result["MOCK_GATEWAY_BASE_URL"] == "http://127.0.0.1:53109"
    assert result["STRIPE_KEY"] == "fake_stripe_key_025"
    assert "UNRELATED_HOST_VALUE" not in result
