"""Per-episode stdio MCP server backed by an official benchmark worker.

stdout belongs exclusively to MCP. Every executed call is durably appended to
the audit JSONL so the collector never trusts harness-rendered tool text.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams, TextContent, Tool


def append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--deny-tools", default="")
    parser.add_argument("--deny-values-json", default="[]")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    args = parser.parse_args()

    client = httpx.Client(base_url=args.bridge.rstrip("/"), timeout=args.timeout)
    response = client.post("/episodes/start", json={"task_id": args.task_id, "seed": args.seed})
    response.raise_for_status()
    started = response.json()
    episode_id = str(started["episode_id"])
    denied = {value.strip() for value in args.deny_tools.split(",") if value.strip()}
    denied_values = {str(value).lower() for value in json.loads(args.deny_values_json)}
    tools = started.get("tools") or []
    call_count = 0
    if not tools:
        raise RuntimeError("benchmark worker returned no tools; refusing non-agentic collection")
    append(args.audit, {"event": "episode_started", "episode_id": episode_id,
                        "task_id": args.task_id, "seed": args.seed,
                        "timestamp": datetime.now(timezone.utc).isoformat()})

    async def list_tools(_ctx: Any, _params: PaginatedRequestParams | None) -> ListToolsResult:
        result: list[Tool] = []
        for item in tools:
            function = item.get("function") or item
            result.append(Tool(name=str(function["name"]), description=str(function.get("description") or ""),
                               inputSchema=function.get("parameters") or {"type": "object", "properties": {}}))
        return ListToolsResult(tools=result)

    async def call_tool(_ctx: Any, params: CallToolRequestParams) -> CallToolResult:
        nonlocal call_count
        call_count += 1
        name = params.name
        arguments = params.arguments or {}
        serialized_args = json.dumps(arguments, ensure_ascii=False).lower()
        blocked_value = any(value and value in serialized_args for value in denied_values)
        over_limit = call_count > args.max_tool_calls
        if name in denied or blocked_value or over_limit:
            reason = ("configured_deny_tool" if name in denied else
                      "forbidden_argument_value" if blocked_value else "max_tool_calls")
            append(args.audit, {"event": "tool_call", "episode_id": episode_id, "tool_name": name,
                                "arguments": arguments, "observation": "BLOCKED_BY_PRETOOL_GUARD", "tool_result": None,
                                "invalid_call": False, "side_effect": False, "risk_event": False,
                                "guard_decision": "block", "guard_reason": reason})
            return CallToolResult(content=[TextContent(type="text", text="BLOCKED_BY_PRETOOL_GUARD: " + reason)])
        try:
            reply = client.post(f"/episodes/{episode_id}/tool", json={"name": name, "arguments": arguments})
            reply.raise_for_status()
            value = reply.json()
        except Exception as exc:
            append(args.audit, {"event": "tool_call", "episode_id": episode_id, "tool_name": name,
                                "arguments": arguments, "invalid_call": True, "parser_problem": True,
                                "error": f"{type(exc).__name__}: {exc}"})
            raise
        append(args.audit, {"event": "tool_call", "episode_id": episode_id, "tool_name": name,
                            "arguments": arguments, "observation": str(value.get("observation") or ""),
                            "tool_result": value.get("result"), "invalid_call": bool(value.get("invalid_call")),
                            "side_effect": bool(value.get("side_effect")), "risk_event": bool(value.get("risk_event")),
                            "guard_decision": "allow", "guard_reason": "policy_pass",
                            "timestamp": datetime.now(timezone.utc).isoformat()})
        return CallToolResult(content=[TextContent(type="text", text=str(value.get("observation") or ""))])

    server = Server("cross-harness-benchmark", on_list_tools=list_tools, on_call_tool=call_tool)

    import anyio

    async def serve() -> None:
        async with stdio_server() as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    try:
        anyio.run(serve)
    finally:
        append(args.audit, {"event": "mcp_stopped", "episode_id": episode_id,
                            "timestamp": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
