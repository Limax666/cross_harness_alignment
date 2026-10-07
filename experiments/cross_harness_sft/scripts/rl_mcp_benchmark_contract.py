"""Exact benchmark MCP wire mapping for the Codex/Claude Code RL adapters."""
from __future__ import annotations

import copy
import re


PREFIX = "mcp__benchmark__"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")


def mcp_benchmark_tools(native_tools: list[dict]) -> tuple[list[dict], dict[str, str]]:
    if not native_tools:
        raise ValueError("benchmark MCP needs at least one native tool")
    tools: list[dict] = []
    wire_to_native: dict[str, str] = {}
    for item in native_tools:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            raise ValueError("invalid native tool definition")
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"MCP tool requires lossy name normalization: {name!r}")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError(f"invalid native tool parameter schema: {name}")
        wire = PREFIX + name
        if len(wire) > 64 or wire in wire_to_native:
            raise ValueError(f"MCP name exceeds 64 characters or collides: {name}")
        converted = copy.deepcopy(function)
        converted["name"] = wire
        tools.append({"type": "function", "function": converted})
        wire_to_native[wire] = name
    return tools, wire_to_native
