"""Audit complete OpenClaw native-session continuity on real owner turns.

Wire conversion functions copied unchanged from the registered native provider;
all audit inputs are actual SQLite rows and actual forwarded native requests.
"""
import hashlib
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

def _json_args(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(
            value if value is not None else {},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return "{}"

def _attachments(original: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return validated data URL attachments from a local tool receipt."""

    value = original.get("attachments")
    if not isinstance(value, list):
        receipt = original.get("receipt")
        value = receipt.get("attachments") if isinstance(receipt, Mapping) else None
    if not isinstance(value, list):
        return []
    return [
        item
        for item in value
        if isinstance(item, Mapping)
        and str(item.get("url", "")).startswith("data:")
        and "," in str(item.get("url", ""))
    ]

def _data_attachment(attachment: Mapping[str, Any]) -> tuple[str, str, str] | None:
    url = str(attachment.get("url", ""))
    if not url.startswith("data:") or "," not in url:
        return None
    head, data = url.split(",", 1)
    mime = str(
        attachment.get("mime")
        or head[5:].split(";", 1)[0]
        or "application/octet-stream"
    )
    return mime, url, data

def _chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate provider-neutral tool calls to Chat Completions messages."""

    out: list[dict[str, Any]] = []
    pending_media: list[Mapping[str, Any]] = []

    def flush_media() -> None:
        if not pending_media:
            return
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": "Attached media from tool result:"}
        ]
        for attachment in pending_media:
            parsed = _data_attachment(attachment)
            if parsed is None:
                continue
            mime, url, _data = parsed
            if mime.startswith("image/"):
                blocks.append({"type": "image_url", "image_url": {"url": url}})
            else:
                blocks.append(
                    {
                        "type": "file",
                        "file": {
                            "file_data": url,
                            "filename": str(attachment.get("filename") or "attachment"),
                        },
                    }
                )
        if len(blocks) > 1:
            out.append({"role": "user", "content": blocks})
        pending_media.clear()

    for original in messages:
        role = str(original.get("role", "user"))
        if role != "tool":
            flush_media()
        content = original.get("content", "")
        calls = original.get("tool_calls") or []
        if calls:
            item = {
                "role": role,
                "content": content or None,
                "tool_calls": [
                    {
                        "id": str(call.get("id") or "call"),
                        "type": "function",
                        "function": {
                            "name": str(call.get("name", "")),
                            "arguments": _json_args(call.get("arguments", {})),
                        },
                    }
                    for call in calls
                ],
            }
        elif role == "tool":
            value = (
                content
                if isinstance(content, str)
                else json.dumps(content, ensure_ascii=False)
            )
            item = {
                "role": "tool",
                "content": value,
                "tool_call_id": str(original.get("tool_call_id") or "call"),
            }
        else:
            item = {"role": role, "content": content}
        out.append(item)
        if role == "tool":
            pending_media.extend(_attachments(original))
    flush_media()
    return out

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_session(path, *, allow_missing=False):
    path = Path(path)
    if not path.is_file():
        if allow_missing:
            return []
        raise ValueError("native session is missing; refusing incomplete history")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("select role,content,meta from messages order by id").fetchall()
    return [{"role": role, "content": json.loads(content), **json.loads(meta)}
            for role, content, meta in rows]


def verify_turn(before, after, owners, requests, result, session_path):
    if after[:len(before)] != before:
        raise ValueError("native session prefix changed; refusing history loss")
    if [m["content"] for m in after if m["role"] == "user"] != owners:
        raise ValueError("native owner turns are missing, duplicated, or rewritten")
    starts = [e for e in result.get("messages", [])
              if isinstance(e, dict) and e.get("type") == "agent_start"]
    if len(starts) != 1 or Path(starts[0].get("session_file", "")).resolve() != session_path.resolve():
        raise ValueError("unexpected native session path")
    if not requests:
        raise ValueError("no actual native model request")
    expected = _chat_messages(before + [{"role": "user", "content": owners[-1]}])
    actual = requests[0].get("messages", [])
    # The native runtime regenerates the selected system prompt on each run;
    # it does not persist it. Every persisted user/assistant/tool field must
    # pass through its registered provider conversion exactly, in order.
    expected = [m for m in expected if m.get("role") != "system"]
    actual = [m for m in actual if m.get("role") != "system"]
    if actual != expected:
        raise ValueError("first actual request differs from complete native session")
    return {"owner_turn": len(owners), "session_prefix_preserved": True,
            "current_owner_only": True, "all_prior_native_wire_messages_equal": True,
            "prior_native_messages": len(before), "after_native_messages": len(after),
            "prior_tool_results": sum(m["role"] == "tool" for m in before),
            "owner_message_lengths": [len(m) for m in owners],
            "before_sha256": digest(before), "after_sha256": digest(after),
            "first_actual_wire_messages_sha256": digest(actual),
            "requests": len(requests)}
