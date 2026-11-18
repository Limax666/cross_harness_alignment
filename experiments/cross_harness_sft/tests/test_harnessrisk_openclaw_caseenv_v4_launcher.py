"""Regression coverage for the generic OpenClaw case-environment v4 launcher."""

from __future__ import annotations

from flashrsi_matrix_openclaw_caseenv_v4_20261115 import full_run_environment


def test_caseenv_v4_launcher_keeps_case_identity_host_for_evoprompt():
    """A v2 host would drop adapter identity and cross-wire mock services."""
    result = full_run_environment(
        {"UNRELATED": "not-forwarded"}, "/tmp/registered-output", "18762", "evoprompt"
    )

    assert result["HR_HARNESS"] == "openclaw"
    assert result["HARNESSRISK_MODEL_PORT"] == "18762"
    assert result["HR_HOST_SCRIPT"] == "flashrsi_harnessrisk_host_openclaw_caseenv_v3.py"
    assert result["HR_STRICT_EVOLUTION"] == "1"
    assert result["HR_EVOLUTION_CASE_ATTEMPTS"] == "1"
    assert result["UNRELATED"] == "not-forwarded"
