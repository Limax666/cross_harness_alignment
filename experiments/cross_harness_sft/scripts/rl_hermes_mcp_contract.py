"""Hermes MCP wire names for the restricted AgentDojo pilot tool surface.

The real Hermes registry uses ``mcp__<server>__<tool>`` (64-character maximum).
This pilot uses only the ``benchmark`` MCP server and rejects names requiring
lossy normalization or truncation, so the mapping is exact and reversible.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
from pathlib import Path

_NATIVE = re.compile(r"[A-Za-z0-9_-]+\Z")
PREFIX = "mcp__benchmark__"
HERMES_ROOT = Path("/data/home/liumingxiao/.hermes/hermes-agent")
HERMES_PYTHON = HERMES_ROOT / "venv/bin/python"
_CONVERT = (
    "import json,sys; from types import SimpleNamespace; "
    "from tools.mcp_tool_schema import _convert_mcp_schema; "
    "items=json.load(sys.stdin); "
    "print(json.dumps([_convert_mcp_schema('benchmark', "
    "SimpleNamespace(name=f['name'], description=f.get('description'), "
    "input_schema=f['parameters'])) for f in items]))"
)


def hermes_mcp_tools(native_tools: list[dict]) -> tuple[list[dict], dict[str, str]]:
    if not native_tools:
        raise ValueError("Hermes MCP pilot requires native tools")
    native_functions = []
    wire_to_native = {}
    for item in native_tools:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            raise ValueError("invalid native tool schema")
        name = function.get("name")
        if not isinstance(name, str) or not _NATIVE.fullmatch(name):
            raise ValueError(f"Hermes MCP tool name requires normalization: {name!r}")
        wire = PREFIX + name
        if len(wire) > 64 or wire in wire_to_native:
            raise ValueError(f"Hermes MCP tool name is truncated or colliding: {name!r}")
        params = function.get("parameters")
        if not isinstance(params, dict) or params.get("type") != "object":
            raise ValueError(f"invalid MCP parameter schema for {name}")
        native_functions.append(copy.deepcopy(function))
        wire_to_native[wire] = name
    if not HERMES_PYTHON.is_file():
        raise RuntimeError("installed Hermes runtime is absent; cannot attest its MCP schema")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERMES_ROOT)
    result = subprocess.run([str(HERMES_PYTHON), "-c", _CONVERT],
                            input=json.dumps(native_functions), text=True,
                            capture_output=True, timeout=15, env=env, check=True)
    actual = json.loads(result.stdout)
    if len(actual) != len(native_functions):
        raise ValueError("Hermes MCP schema conversion dropped tools")
    exposed = []
    for function, converted in zip(native_functions, actual):
        wire = PREFIX + function["name"]
        if converted.get("name") != wire or not isinstance(converted.get("parameters"), dict):
            raise ValueError(f"installed Hermes disagrees with expected MCP wire contract: {wire}")
        exposed.append({"type": "function", "function": converted})
    return exposed, wire_to_native
