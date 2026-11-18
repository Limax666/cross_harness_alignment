from pathlib import Path

import pytest


def test_opencode_runner_uses_checkout_relative_paths():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("flashrsi_opencode_full_host_v2.py", "flashrsi_opencode_full_supervisor_v2.py", "flashrsi_opencode_entry_v2.py"):
        source = (scripts / name).read_text()
        assert "/output/cross_harness_alignment" not in source
        assert "/opt/uv/python" not in source
        assert ".venv-agentdojo" not in source


def test_other_adapters_use_checkout_relative_paths():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for harness, suffix in (("openclaw", ""), ("hermes", ""), ("pi", "_v2")):
        for part in ("full_host", "full_supervisor", "entry"):
            name = f"flashrsi_{harness}_{part}{suffix}.py"
            source = (scripts / name).read_text()
            assert "/output/cross_harness_alignment" not in source
            assert "/opt/uv/python" not in source
            assert ".venv-agentdojo" not in source


def test_opencode_completion_budget_is_propagated_to_isolated_entry():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("flashrsi_opencode_full_host_v2.py", "flashrsi_opencode_full_supervisor_v2.py", "flashrsi_opencode_entry_v2.py"):
        assert "FLASHRSI_COMPLETION_BUDGET" in (scripts / name).read_text()


def test_openclaw_completion_budget_is_propagated_to_isolated_entry():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("flashrsi_openclaw_full_host.py", "flashrsi_openclaw_full_supervisor.py", "flashrsi_openclaw_entry.py"):
        assert "FLASHRSI_COMPLETION_BUDGET" in (scripts / name).read_text()


def test_hermes_completion_budget_is_propagated_to_isolated_entry():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("flashrsi_hermes_full_host.py", "flashrsi_hermes_full_supervisor.py", "flashrsi_hermes_entry.py"):
        assert "FLASHRSI_COMPLETION_BUDGET" in (scripts / name).read_text()


def test_pi_completion_budget_is_propagated_to_isolated_entry():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("flashrsi_pi_full_host_v2.py", "flashrsi_pi_full_supervisor_v2.py", "flashrsi_pi_entry_v2.py"):
        assert "FLASHRSI_COMPLETION_BUDGET" in (scripts / name).read_text()


@pytest.mark.parametrize("name", [
    "flashrsi_opencode_full_host_v2.py",
    "flashrsi_openclaw_full_host.py",
    "flashrsi_hermes_full_host.py",
    "flashrsi_pi_full_host_v2.py",
])
def test_completion_budget_host_parses(name):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    compile((scripts / name).read_text(), name, "exec")
