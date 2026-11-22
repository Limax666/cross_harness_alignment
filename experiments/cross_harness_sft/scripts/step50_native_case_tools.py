"""Bind two pinned FlashRSI kernels to a local model and case-only document tools."""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from unittest.mock import patch

import httpx

from flashrsi_harnessrisk_privacy import assert_no_oracle
from step50_pair_evidence import MAX_CALLS

ROUTE = {"provider": "custom", "model_name": "local-qwen", "api_key": "in-process",
         "base_url": "http://in-process.invalid/v1", "model_api": "chat_completions"}


def isolate_native_tools(harness):
    """Resolve the native loop's tool import to a fail-closed boundary before loading the kernel."""
    module_name = harness + "_native.tools"
    if module_name in sys.modules:
        raise RuntimeError("Native tool module already loaded before isolation")

    def unavailable(*args, **kwargs):
        raise RuntimeError("Product tools are unavailable in the case-document protocol")

    module = ModuleType(module_name)
    if harness == "hermes":
        module.HermesTools = unavailable
    else:
        module.ToolRunner, module.schemas = unavailable, unavailable
    sys.modules[module_name] = module


def hermes_config():
    return {"llm": dict(ROUTE), "max_iterations": MAX_CALLS, "api_max_retries": 0,
            "auto_recovery_cycles": 0, "enabled_tools": ["read_file", "write_file", "skill_view"],
            "skip_context_files": True, "load_soul_identity": True, "include_runtime_context": False,
            "compression": {"enabled": False}, "context_length": 32768}


def resources(harness, root, empty):
    """Use the native resource assembler on verified data files with no ambient instructions."""
    if harness == "hermes":
        from hermes_native.resources import load_resources
        result = load_resources(root, empty, hermes_config())
    else:
        from opencode_native.resources import load
        for parent in (empty, *empty.parents):
            if (parent / "AGENTS.md").exists():
                raise ValueError("OpenCode instruction search would import a host file")
        result = load(root, empty, {"skills": False, "prompt": {"instruction_files": ["AGENTS.md", "SOUL.md"]}})
        normalized = result.system_prompt.replace("Instructions from: " + str(root) + "/", "Instructions from: snapshot/")
        result = replace(result, system_prompt=normalized)
    if not result.system_prompt.strip():
        raise ValueError("Empty native system prompt")
    assert_no_oracle(result.system_prompt)
    return result


class CaseTools:
    def __init__(self, memory, turn, cwd, skill):
        self.memory, self.turn, self.cwd, self.skill = memory, turn, cwd, skill
        self.schemas = list(memory.schemas)
        if skill is not None:
            self.schemas.append({"type": "function", "function": {"name": "skill_view",
                "description": "Read the bundled hermes-agent skill as text.",
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                               "required": ["name"], "additionalProperties": False}}})

    def close(self):
        pass

    def execute(self, name, args):
        if name == "skill_view" and self.skill is not None:
            value = ({"status": "ok", "result": self.skill} if args == {"name": "hermes-agent"}
                     else {"status": "policy_block", "result": "Unknown bundled skill."})
        else:
            value = self.memory.dispatch(name, args)
        assert_no_oracle(value)
        return value

    def record(self, name, args, value, call_id):
        self.turn.events.append({"owner": self.turn.index, "name": name, "arguments": args,
                                 "result": value, "call_id": call_id})
        self.turn.transcript.append({"role": "tool", "name": name, "tool_call_id": call_id,
                                     "content": json.dumps(value, ensure_ascii=False)})


async def run_owner(harness, bundle, root, cwd, user, turn, skill):
    tools = CaseTools(turn.memory, turn, cwd, skill)
    if harness == "opencode":
        from opencode_native import loop, providers

        class DocumentRunner:
            async def execute(self, name, args):
                receipt = tools.execute(name, args)
                return {"ok": True, "output": json.dumps(receipt, ensure_ascii=False), "case_receipt": receipt}

        async def complete(config, messages, schemas, **kwargs):
            reply = await turn.chat(providers._chat_messages(messages), model="local-qwen",
                                    tools=schemas, tool_choice=kwargs.get("tool_choice"))
            return providers._parse_chat(reply.model_dump())

        # Capture native denials (last step or repeated call) as well as real document reads/writes.
        pending = {}
        def on_event(ev):
            if ev["type"] == "tool_start":
                pending[ev["id"]] = (ev["name"], ev["arguments"])
            elif ev["type"] == "tool_end":
                name, args = pending.pop(ev["id"])
                native = ev["result"]
                if "case_receipt" not in native and native.get("error_type") not in {"max_steps", "doom_loop"}:
                    raise RuntimeError("Unexpected OpenCode tool execution failure")
                value = native.get("case_receipt") or {"status": "policy_block", "result": native["error"]}
                tools.record(name, args, value, ev["id"])

        with ExitStack() as stack:
            for obj, field, bound in [(loop, "load", lambda *a: bundle),
                                      (loop, "ToolRunner", lambda *a: DocumentRunner()),
                                      (loop, "schemas", lambda: tools.schemas), (providers, "complete", complete)]:
                stack.enter_context(patch.object(obj, field, bound))
            result = await loop.run({"max_iterations": MAX_CALLS, "session": {"persist": False},
                                     "compaction": {"enabled": True, "threshold": 0.8},
                                     "max_context_tokens": 32768}, user, cwd, root, on_event=on_event)
        if pending:
            raise RuntimeError("Unmatched OpenCode tool event")
        return result["status"], result["output"], result.get("error_message")

    from hermes_native import loop
    import copy

    class DocumentRunner:
        def __init__(self):
            self.cwd, self.schemas = cwd, tools.schemas

        def execute(self, name, args):
            return json.dumps(tools.execute(name, args), ensure_ascii=False)

        def close(self):
            pass

    # This private transport goes straight to the local tokenizer/model. No sockets are opened.
    async def in_process(request):
        if str(request.url) != ROUTE["base_url"] + "/chat/completions":
            raise RuntimeError("Unexpected provider route")
        body = json.loads(request.content)
        reply = await turn.chat(body["messages"], model="local-qwen", tools=body.get("tools") or [],
                                tool_choice=body.get("tool_choice"))
        return httpx.Response(200, json=reply.model_dump())

    async def batch(calls, _runner, _cwd, events, append, recovery_host=None):
        for call in calls:
            name, args = call.function.name, json.loads(call.function.arguments)
            value = tools.execute(name, args)
            tools.record(name, args, value, call.id)
            append({"role": "tool", "name": name, "tool_call_id": call.id,
                    "content": json.dumps(value, ensure_ascii=False)})

    class LocalInferenceTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return await in_process(request)

    def client(**kwargs):
        return httpx.AsyncClient(transport=LocalInferenceTransport(), trust_env=False, **kwargs)

    with ExitStack() as stack:
        for field, bound in [("HermesTools", lambda *a, **kw: DocumentRunner()),
                             ("load_resources", lambda *a, **kw: copy.deepcopy(bundle)),
                             ("_execute_batch", batch), ("httpx", SimpleNamespace(AsyncClient=client))]:
            stack.enter_context(patch.object(loop, field, bound))
        result = await loop.run(hermes_config(), user, cwd, root)
    return result.status, result.output, result.error_message
