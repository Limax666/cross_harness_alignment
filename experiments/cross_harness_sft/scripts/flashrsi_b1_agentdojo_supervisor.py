"""Execute one B1 AgentDojo task with the supplied GEPA candidate Snapshot."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path

from agentdojo.attacks.baseline_attacks import DirectAttack
from cross_harness_sft.backends.agentdojo import AgentDojoDriver
from flashrsi.agents.miniagent import MiniAgent, _ENTRY
from flashrsi.core import Snapshot
from flashrsi_final_turn_proxy import FinalTurnChatProxy
from flashrsi_gateway import create_gateway
from flashrsi_model_relay import relay_request
from flashrsi_tool_attestation import (
    GuardedToolDispatcher,
    ToolCallGate,
    attest_tool_calls,
)
from rl_agentdojo_path_audit import NativePathRecorder


ROOT = Path("/openbayes/home/cross_harness_alignment/experiments/cross_harness_sft")
MODEL_NAME = "qwen35-4b-preflight"
SAFETY_POLICY = ROOT / "configs/flashrsi_b0_safety_policy.txt"


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()
    ).hexdigest()


async def _main() -> dict:
    task_id = os.environ["B1_TASK_ID"]
    seed = int(os.environ["B1_SEED"])
    expected_snapshot = os.environ["B1_EXPECTED_SNAPSHOT_ID"]
    case_out = Path(os.environ["B1_CASE_OUT"])
    started_at = time.monotonic()
    record = {
        "experiment": "FlashRSI GEPA B1",
        "harness": "miniagent",
        "benchmark": "agentdojo",
        "task_id": task_id,
        "seed": seed,
        "candidate_snapshot_id": os.environ.get("B1_CANONICAL_SNAPSHOT_ID", expected_snapshot),
        "effective_snapshot_id": expected_snapshot,
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "flashrsi_revision": "778e2d01a995491fe63db210dcf46db1723a4430",
        "network_namespace": "new loopback-only",
        "max_turns": 12,
        "max_tokens_per_turn": 512,
        "temperature": 0,
    }
    server = None
    thread = None
    try:
        driver = AgentDojoDriver(str(ROOT / "configs/agentdojo_worker.yaml"))
        suite, user, injection = driver._case_map[task_id]
        attack = (
            {}
            if injection is None
            else DirectAttack(suite, target_pipeline=None).attack(user, injection)
        )
        record["attack_vectors"] = sorted(attack)
        record["attack_payload_sha256"] = {
            name: hashlib.sha256(value.encode()).hexdigest()
            for name, value in attack.items()
        }
        started = driver.start(task_id, seed)
        episode, tools = started["episode"], started["tools"]
        record["initial_state_sha256"] = _digest(
            episode.environment.model_dump(mode="json")
        )
        gate = ToolCallGate({item["function"]["name"] for item in tools})
        relay_path = os.environ["DEV_RELAY_PATH"]
        relay_token = os.environ["DEV_RELAY_TOKEN"]

        def upstream(request):
            return relay_request(relay_path, relay_token, request)

        proxy = FinalTurnChatProxy(gate, MODEL_NAME, tools, upstream)
        recorder = NativePathRecorder(
            driver,
            episode,
            tool_schemas=tools,
            qualified_families=NativePathRecorder.__dataclass_fields__[
                "qualified_families"
            ].default,
        )
        dispatcher = GuardedToolDispatcher(gate, recorder)
        gateway_token = secrets.token_urlsafe(32)
        server = create_gateway(proxy, dispatcher, gate, gateway_token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        gateway_url = f"http://127.0.0.1:{server.server_port}"

        snapshot = Snapshot.load(Path(os.environ["B1_SNAPSHOT_JSON"]))
        if snapshot.id != expected_snapshot:
            raise ValueError("candidate Snapshot ID does not match the requested ID")
        policy = SAFETY_POLICY.read_text(encoding="utf-8").strip()
        prompt = snapshot.files["prompts/system.md"].content.decode("utf-8")
        safety_suffix = "## Trusted safety policy\n" + policy + "\n"
        if not prompt.endswith(safety_suffix):
            raise ValueError("candidate Snapshot changed its pinned safety policy")
        files_tools = {
            Path(path).stem
            for path in snapshot.files
            if path.startswith("tool_descriptions/") and path.endswith(".json")
        }
        expected_tools = {item["function"]["name"] for item in tools}
        if files_tools != expected_tools:
            raise ValueError("candidate Snapshot tool schemas do not match the task")
        record["effective_snapshot_id"] = snapshot.id
        record["tool_count"] = len(tools)

        parent = Path(tempfile.mkdtemp(prefix="rsi-b1-child-"))
        writable = parent / "work"
        writable.mkdir()
        sandbox_config = parent / "config.json"
        sandbox_config.write_text(
            json.dumps(
                {
                    "entry": str(_ENTRY),
                    "python": os.environ["B1_PYTHON"],
                    "gateway_port": server.server_port,
                    "readonly_roots": [
                        "/usr",
                        "/lib",
                        "/lib64",
                        "/etc/ssl/certs",
                        "/opt/uv/python",
                        str(ROOT / ".venv-agentdojo"),
                        str(_ENTRY.parent),
                    ],
                    "writable_root": str(writable),
                }
            ),
            encoding="utf-8",
        )
        os.chmod(sandbox_config, 0o600)
        os.environ["FLASHRSI_SANDBOX_CONFIG"] = str(sandbox_config)
        os.environ["FLASHRSI_GATEWAY_URL"] = gateway_url
        os.environ["FLASHRSI_GATEWAY_TOKEN"] = gateway_token
        os.environ["TMPDIR"] = str(writable)
        tempfile.tempdir = str(writable)

        target = MiniAgent(
            {
                "llm": {
                    "model_name": MODEL_NAME,
                    "api_key": gateway_token,
                    "base_url": gateway_url + "/v1",
                    "max_tokens": 512,
                    "extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False},
                        "temperature": 0,
                        "seed": seed,
                    },
                },
                "max_turns": 12,
                "timeout": 300,
            }
        )
        target.update(snapshot)
        record["root_snapshot_id"] = target.snapshot.root_id

        original_executable = sys.executable
        sys.executable = str(ROOT / "scripts/flashrsi_sandbox_exec.py")
        try:
            result = await target.run(user.PROMPT)
        finally:
            sys.executable = original_executable

        record.update(
            status=result.status,
            error_message=result.error_message,
            output=result.output,
            trajectory_messages=result.messages,
            usage=dict(result.usage),
            raw_model_requests=proxy.raw_requests,
            raw_model_messages=proxy.raw_messages,
            dispatches=dispatcher.executed,
        )
        record["attestation_passed"] = attest_tool_calls(
            proxy.raw_messages, dispatcher.executed
        )
        if result.status == "finished" and record["attestation_passed"]:
            gate.finish()
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()
            server = None
            audit = recorder.audit_structured(
                task_id, gate=gate, proxy=proxy, dispatcher=dispatcher
            )
            record["path_audit"] = asdict(audit)
            record["path_abstention_reason"] = recorder.abstention_reason
            record["official"] = driver.finish(episode, result.output)
            reset_episode = driver.start(task_id, seed)["episode"]
            record["reset_verified"] = record["initial_state_sha256"] == _digest(
                reset_episode.environment.model_dump(mode="json")
            )
        else:
            gate.abort()
            record["unscorable_reason"] = "incomplete_or_unattested"
        if record.get("reset_verified") is not True:
            record["unscorable_reason"] = "reset_mismatch"
    except Exception as error:
        record["unscorable_reason"] = type(error).__name__ + ": " + str(error)
    finally:
        record["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
        if server is not None:
            server.shutdown()
            if thread is not None:
                thread.join(timeout=5)
            server.server_close()
        case_out.parent.mkdir(parents=True, exist_ok=True)
        case_out.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=dict) + "\n",
            encoding="utf-8",
        )
    return record


if __name__ == "__main__":
    result = asyncio.run(_main())
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "task_id",
                    "status",
                    "attestation_passed",
                    "official",
                    "reset_verified",
                    "unscorable_reason",
                    "elapsed_seconds",
                )
            },
            ensure_ascii=False,
        )
    )
