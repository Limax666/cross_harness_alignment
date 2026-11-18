import pytest

from flashrsi_sandbox_exec import _child_environment


def test_child_environment_forwards_only_registered_completion_budgets(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHRSI_COMPLETION_BUDGET", "4096")
    env = _child_environment("http://127.0.0.1:1234", "x" * 40, tmp_path)
    assert env["FLASHRSI_COMPLETION_BUDGET"] == "4096"
    assert "PATH" in env and "HOME" in env
    assert "OPENAI_API_KEY" not in env


def test_child_environment_rejects_unregistered_completion_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHRSI_COMPLETION_BUDGET", "16384")
    with pytest.raises(ValueError, match="completion budget"):
        _child_environment("http://127.0.0.1:1234", "x" * 40, tmp_path)
