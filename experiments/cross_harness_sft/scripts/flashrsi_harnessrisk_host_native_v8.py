"""Claude native v8: preserve raw messages but remove CLI-injected duplicate reminders."""
import copy
import json
import os
import sys
from pathlib import Path

import flashrsi_harnessrisk_host_native_v7 as base
from flashrsi_harnessrisk_privacy import PROTOCOL, assert_no_oracle

ROOT = Path(__file__).resolve().parents[1]
_AUTO_PREFIXES = (
    "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n# gitStatus\n",
    "<system-reminder>\nAttribution for git commits and pull requests you create from here on ",
)


def _is_cli_reminder(part):
    return (isinstance(part, dict) and part.get("type") == "text"
            and isinstance(part.get("text"), str)
            and part["text"].startswith(_AUTO_PREFIXES))


def _is_repeated_owner_text(part, initial_owner):
    return (isinstance(part, dict) and part.get("type") == "text"
            and isinstance(part.get("text"), str) and part["text"] == initial_owner)


def project_cli_history(body):
    """Return a lossless-for-tools model projection and an auditable removal list.

    Claude Code 2.1.319 injects its git/attribution reminder and repeats the
    owner prompt after every tool result.  These are runtime-generated blocks,
    never HarnessRisk case data.  Keep the initial owner turn and every tool
    result exactly; strip only later exact injected copies.
    """
    current = copy.deepcopy(body)
    messages = current.get("messages")
    if not isinstance(messages, list) or not messages:
        return current, []
    initial_owner = None
    first = messages[0]
    if isinstance(first, dict) and first.get("role") == "user" and isinstance(first.get("content"), list):
        for part in first["content"]:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                initial_owner = part["text"]
                break
    removed = []
    for message_index, message in enumerate(messages[1:], start=1):
        if not isinstance(message, dict) or message.get("role") != "user" or not isinstance(message.get("content"), list):
            continue
        kept = []
        for part_index, part in enumerate(message["content"]):
            if _is_cli_reminder(part) or (initial_owner is not None and _is_repeated_owner_text(part, initial_owner)):
                removed.append({"message_index": message_index, "part_index": part_index,
                                "sha256": __import__("hashlib").sha256(part["text"].encode()).hexdigest(),
                                "kind": "cli_system_reminder" if _is_cli_reminder(part) else "repeated_owner_prompt"})
            else:
                kept.append(part)
        message["content"] = kept
    return current, removed


def main():
    # Reuse v7 confinement/model transport, replacing only the native relay
    # factory.  The factory receives the raw request, writes an immutable raw
    # copy beside the projected request, then invokes the unchanged v7 relay.
    original_factory = base.create_native_relay
    raw_records = []
    def factory(path, token, model, upstream):
        def audited_upstream(route, body):
            assert_no_oracle(body)
            projected, removed = project_cli_history(body) if route == "/v1/messages" else (copy.deepcopy(body), [])
            raw_records.append({"path": route, "raw_request": copy.deepcopy(body), "projection_removed": removed})
            return upstream(route, projected)
        return original_factory(path, token, model, audited_upstream)
    base.create_native_relay = factory
    try:
        return base.main()
    finally:
        # base writes the forward request evidence.  Persist raw requests in a
        # separate evidence file without mutating base's already-frozen format.
        if len(sys.argv) == 5:
            evidence = Path(sys.argv[4]) / sys.argv[1]
            if evidence.is_dir():
                (evidence / "native_cli_projection_evidence_v8.json").write_text(json.dumps({"protocol": "private-v1-claude-cli-projection-v8", "requests": raw_records}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
