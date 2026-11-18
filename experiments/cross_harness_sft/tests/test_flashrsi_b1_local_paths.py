from pathlib import Path

import flashrsi_b1_agentdojo_host as host
import flashrsi_b1_agentdojo_supervisor as supervisor
import flashrsi_gepa_security as security


def test_b1_components_use_current_checkout():
    root = Path(__file__).resolve().parents[1]
    assert host.ROOT == supervisor.ROOT == security.ROOT == root


def test_sandbox_python_roots_cover_real_interpreter():
    python = Path(supervisor.sys.executable).resolve()
    roots = supervisor._python_readonly_roots(supervisor.sys.executable)
    assert any(python.is_relative_to(Path(root).resolve()) for root in roots)
    assert "/" not in roots
