"""Regression coverage for the OpenClaw case-environment v3 full-run launcher."""

from __future__ import annotations

from flashrsi_matrix_openclaw_caseenv_v3_20261114 import full_run_environment


def test_caseenv_v3_launcher_selects_the_identity_preserving_host():
    """Changing the host back to v2 would reintroduce cross-case mock identity loss."""
    result = full_run_environment({"UNRELATED": "not-forwarded"}, "/tmp/registered-output", "18762")

    assert result["HR_HARNESS"] == "openclaw"
    assert result["HARNESSRISK_MODEL_PORT"] == "18762"
    assert result["HR_HOST_SCRIPT"] == "flashrsi_harnessrisk_host_openclaw_caseenv_v3.py"
    assert result["HR_STRICT_EVOLUTION"] == "1"
    assert result["HR_EVOLUTION_CASE_ATTEMPTS"] == "1"
    assert result["UNRELATED"] == "not-forwarded"
