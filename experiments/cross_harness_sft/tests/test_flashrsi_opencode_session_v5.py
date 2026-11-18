import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor" / "FlashRSI"))


def test_v5_launcher_uses_only_the_audited_v5_host():
    import flashrsi_matrix_opencode_session_v5_20261115 as launcher

    assert 'host_opencode_session_v5.py' in launcher.source
    assert 'host_opencode_session_v4.py' not in launcher.source


def test_v7_audit_replays_compaction_with_the_actual_system_prompt():
    """The ephemeral runtime system prompt must participate in the threshold."""
    import flashrsi_harnessrisk_opencode_session_audit_v7 as audit

    before = [{"role": "user", "content": "x" * 18_000}]
    from flashrsi.agents.opencode.runtime.opencode_native.compaction import compact

    complete = [{"role": "system", "content": "runtime policy"}, *before, {"role": "user", "content": "next"}]
    request = {"messages": audit._chat_messages(compact(
        complete, max_tokens=6000, preserve_recent_tokens=2048, tail_turns=1, prune=True
    ))}
    result = {"messages": [{"type": "agent_start", "session_file": "/tmp/session.sqlite"}]}
    record = audit.verify_turn(
        before,
        before + [{"role": "user", "content": "next"}],
        ["x" * 18_000, "next"],
        [request],
        result,
        __import__("pathlib").Path("/tmp/session.sqlite"),
    )
    assert record["runtime_system_prompt_replayed"] is True
