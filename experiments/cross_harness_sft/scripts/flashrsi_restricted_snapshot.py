"""Prepare a tool-limited MiniAgent snapshot for AgentDojo development.

This snapshot does not itself provide OS isolation or validate the endpoint.
It must not be used for adversarial evaluation until both are independently
verified and the model/proxy/dispatcher evidence chain is wired end to end.
"""

import json
import re
import shutil

import yaml

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_DENIED = frozenset({"shell", "read_skill", "web_search", "read_url"})

# Generated source is constant except for a validated Python string literal.
# It only sends a fixed tool name and model arguments to the local gateway.
_TOOL_SOURCE = '''import asyncio
import json
import os
import urllib.error
import urllib.parse
import urllib.request

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("gateway redirect refused")

def _call(arguments):
    address = os.environ.get("FLASHRSI_GATEWAY_URL", "")
    token = os.environ.get("FLASHRSI_GATEWAY_TOKEN", "")
    url = urllib.parse.urlsplit(address)
    if (url.scheme != "http" or url.hostname != "127.0.0.1" or url.port is None
            or url.username is not None or url.password is not None
            or url.path or url.query or url.fragment):
        raise ValueError("invalid gateway address")
    if len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("invalid gateway token")
    body = json.dumps({"name": TOOL_NAME, "arguments": arguments}, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(address + "/tool", data=body, method="POST",
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=30) as response:
        data = response.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("gateway response too large")
    value = json.loads(data)
    if not isinstance(value, dict) or set(value) != {"observation"} or not isinstance(value["observation"], str):
        raise ValueError("invalid gateway observation")
    return value["observation"]

async def call(arguments, ctx):
    return await asyncio.to_thread(_call, arguments)
'''


def make_restricted_snapshot(agent, native_tools, *, safety_policy=None):
    if safety_policy is not None and (not isinstance(safety_policy, str) or not safety_policy.strip() or "\x00" in safety_policy):
        raise ValueError("safety policy must be nonempty text")
    if not isinstance(native_tools, list) or not native_tools:
        raise ValueError("native tool schemas are required")
    functions = []
    seen = set()
    for item in native_tools:
        if not isinstance(item, dict) or item.get("type") != "function" or not isinstance(item.get("function"), dict):
            raise ValueError("invalid native tool")
        function = item["function"]
        name = function.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in _DENIED or name in seen:
            raise ValueError("unsafe or duplicate tool name")
        if not isinstance(function.get("parameters"), dict) or function["parameters"].get("type") != "object":
            raise ValueError("native parameters must be a JSON object schema")
        # Reject non-JSON schema values before any snapshot modification.
        json.dumps(function, allow_nan=False)
        functions.append(function)
        seen.add(name)
    root = agent.snapshot
    with root.check_out() as directory:
        config_path = directory / "agent.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("invalid MiniAgent configuration")
        if safety_policy is not None:
            prompts = config.get("prompts")
            if not isinstance(prompts, dict) or prompts.get("system") != "prompts/system.md":
                raise ValueError("unrecognized MiniAgent system prompt")
            prompt_path = directory / "prompts/system.md"
            original = prompt_path.read_text(encoding="utf-8")
            prompt_path.write_text(original.rstrip() + "\n\n## Trusted safety policy\n" + safety_policy.strip() + "\n", encoding="utf-8")
        config.update(tools=[f["name"] for f in functions], skills=[], middlewares=[], sub_agents=[])
        config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        for name in ("tools", "tool_descriptions", "skills"):
            path = directory / name
            if path.exists():
                shutil.rmtree(path)
        (directory / "tools").mkdir()
        (directory / "tool_descriptions").mkdir()
        for function in functions:
            name = function["name"]
            (directory / "tools" / (name + ".py")).write_text(
                "TOOL_NAME = " + repr(name) + "\n" + _TOOL_SOURCE, encoding="utf-8")
            (directory / "tool_descriptions" / (name + ".json")).write_text(
                json.dumps(function, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return root.commit(directory)
