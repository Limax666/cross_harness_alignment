"""Adapt FlashRSI's OpenCode Python port to fixed benchmark tools.

The agent loop remains OpenCode's; only the tool registry is replaced. This
module supplies no isolation: the runtime must execute in the restricted child.
"""
import json
import re

_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,63}\Z')
_BRIDGE = '''import asyncio
import json
import os
import urllib.request
from pathlib import Path


def schemas():
    return json.loads((Path(__file__).parent / "benchmark_schemas.json").read_text())


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirect refused")


class ToolRunner:
    def __init__(self, workspace, snapshot_root, policy):
        self.workspace = workspace
        self._allowed = {tool["name"] for tool in schemas()}

    async def execute(self, name, args):
        if name not in self._allowed or not isinstance(args, dict):
            raise ValueError("unregistered tool")
        def call():
            url = os.environ["FLASHRSI_GATEWAY_URL"] + "/tool"
            token = os.environ["FLASHRSI_GATEWAY_TOKEN"]
            body = json.dumps({"name": name, "arguments": args}, allow_nan=False).encode()
            req = urllib.request.Request(url, data=body, headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
            with opener.open(req, timeout=30) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError("observation too large")
            value = json.loads(raw)
            if set(value) != {"observation"} or not isinstance(value["observation"], str):
                raise ValueError("invalid observation")
            return {"ok": True, "output": value["observation"]}
        return await asyncio.to_thread(call)
'''


def make_opencode_snapshot(agent, tools, safety_policy):
    if not isinstance(safety_policy, str) or not safety_policy.strip():
        raise ValueError('safety policy is required')
    if not isinstance(tools, list) or not tools:
        raise ValueError('tools are required')
    functions = []
    for item in tools:
        if not isinstance(item, dict) or item.get('type') != 'function' or not isinstance(item.get('function'), dict):
            raise ValueError('invalid tool')
        fn = item['function']
        if not isinstance(fn.get('name'), str) or not _NAME.fullmatch(fn['name']) or not isinstance(fn.get('parameters'), dict):
            raise ValueError('invalid tool schema')
        functions.append(fn)
    if len({fn['name'] for fn in functions}) != len(functions):
        raise ValueError('duplicate tool')
    root = agent.snapshot
    with root.check_out() as directory:
        runtime = directory / 'runtime/opencode_native'
        loop = runtime / 'loop.py'
        source = loop.read_text()
        old = 'from .tools import ToolRunner, schemas'
        if source.count(old) != 1:
            raise ValueError('unexpected OpenCode tool registry')
        loop.write_text(source.replace(old, 'from .benchmark_bridge import ToolRunner, schemas'))
        (runtime / 'benchmark_bridge.py').write_text(_BRIDGE)
        (runtime / 'benchmark_schemas.json').write_text(json.dumps(functions, ensure_ascii=False, allow_nan=False))
        prompt = directory / 'AGENTS.md'
        prompt.write_text(prompt.read_text().rstrip() + '\n\n## Trusted safety policy\n' + safety_policy.strip() + '\n')
        return root.commit(directory)
