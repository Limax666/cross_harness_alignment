"""v7 audit: replay compaction with the ephemeral runtime system prompt."""
from flashrsi_harnessrisk_opencode_session_audit_v4 import _chat_messages, digest, read_session
from flashrsi.agents.opencode.runtime.opencode_native.compaction import compact


def verify_turn(before, after, owners, requests, result, session_path):
    if after[:len(before)] != before:
        raise ValueError("native session prefix changed; refusing history loss")
    if [m["content"] for m in after if m["role"] == "user"] != owners:
        raise ValueError("native owner turns are missing, duplicated, or rewritten")
    starts = [e for e in result.get("messages", []) if isinstance(e, dict) and e.get("type") == "agent_start"]
    if len(starts) != 1 or starts[0].get("session_file") != str(session_path):
        raise ValueError("unexpected native session path")
    if not requests:
        raise ValueError("no actual native model request")
    first = requests[0].get("messages", [])
    system = [m for m in first if m.get("role") == "system"]
    if len(system) != 1:
        raise ValueError("expected exactly one regenerated runtime system prompt")
    complete = [*system, *before, {"role": "user", "content": owners[-1]}]
    projected = compact(complete, max_tokens=6000, preserve_recent_tokens=2048, tail_turns=1, prune=True)
    expected = [m for m in _chat_messages(projected) if m.get("role") != "system"]
    actual = [m for m in first if m.get("role") != "system"]
    if actual != expected:
        raise ValueError("actual request differs from deterministic compacted native session with runtime system prompt")
    return {"owner_turn": len(owners), "session_prefix_preserved": True,
            "current_owner_only": True, "complete_native_history_retained": True,
            "deterministic_compacted_wire_projection": True,
            "runtime_system_prompt_replayed": True,
            "prior_native_messages": len(before), "after_native_messages": len(after),
            "prior_tool_results": sum(m["role"] == "tool" for m in before),
            "complete_before_sha256": digest(before), "after_sha256": digest(after),
            "compacted_wire_messages_sha256": digest(actual), "requests": len(requests)}
