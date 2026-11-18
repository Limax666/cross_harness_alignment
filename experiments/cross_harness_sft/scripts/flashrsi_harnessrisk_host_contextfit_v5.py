"""Hermes private cases with lossless, tokenizer-measured output budgets.

Independent protocol: no message/schema truncation and no generation retries.
The original 4096 ceiling is reduced only when it cannot fit the real input.
"""
import copy
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

from flashrsi_harnessrisk_privacy import assert_no_oracle
from flashrsi_model_relay import create_model_relay

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen35-4b-preflight"
PROTOCOL_ID = "private-v1-hermes-contextfit-v5"
MIN_OUTPUT = 512


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: host CASE_ID SNAPSHOT_JSON EXPECTED_ID RUN_ROOT")
    case_id, snapshot, expected, run_root = sys.argv[1:]
    if os.environ.get("HR_HARNESS") != "hermes":
        raise ValueError("contextfit v5 is restricted to Hermes")
    port = int(os.environ.get("HARNESSRISK_MODEL_PORT", "18761"))
    if not 1 <= port <= 65535:
        raise ValueError("invalid model port")
    token = (ROOT / "outputs/preflight_20261111/vllm_api_token").read_text().strip()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    decisions = []

    def call(route, body):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}" + route, data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        try:
            with opener.open(request, timeout=300) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raw = error.read(8192)
            decisions[-1]["failure"] = {"type": "upstream_http_error", "route": route,
                                        "status": error.code, "body_sha256": hashlib.sha256(raw).hexdigest(),
                                        "context_limit": b"maximum context length" in raw or b"maximum model length" in raw}
            raise RuntimeError("contextfit upstream request rejected; see trusted protocol evidence") from None

    def upstream(request):
        assert_no_oracle(request)
        record = {"sequence": len(decisions) + 1, "original_request_sha256": digest(request)}
        decisions.append(record)
        # Match the actual serving renderer. These are supported text-chat
        # fields, including tool schemas and the frozen enable_thinking flag.
        fields = ("model", "messages", "tools", "chat_template_kwargs", "add_generation_prompt",
                  "continue_final_message", "add_special_tokens")
        counted = call("/tokenize", {key: request[key] for key in fields if key in request})
        count, capacity = counted["count"], counted["max_model_len"]
        if type(count) is not int or type(capacity) is not int or not 0 < count < capacity:
            record["failure"] = {"type": "input_context_exhausted", "count": count, "capacity": capacity}
            raise ValueError("complete input does not fit context; refusing truncation")
        original = request.get("max_tokens", 4096)
        if type(original) is not int or not 0 < original <= 4096 or "max_completion_tokens" in request:
            raise ValueError("unexpected Hermes output budget contract")
        applied = min(original, capacity - count)
        record.update(prompt_tokens=count, max_model_len=capacity, original_max_tokens=original,
                      applied_max_tokens=applied, adjustment="context_capacity" if applied != original else None)
        if applied < MIN_OUTPUT:
            record["failure"] = {"type": "insufficient_output_context", "minimum": MIN_OUTPUT}
            raise ValueError("less than 512 output tokens remain; episode invalid")
        forwarded = copy.deepcopy(request)
        forwarded["max_tokens"] = applied
        if {k: v for k, v in forwarded.items() if k != "max_tokens"} != {k: v for k, v in request.items() if k != "max_tokens"}:
            raise AssertionError("message/schema/decoding mutation outside output limit")
        record["forwarded_request_sha256"] = digest(forwarded)
        response = call("/v1/chat/completions", forwarded)
        record["response_usage"] = copy.deepcopy(response.get("usage", {}))
        record["response_sha256"] = digest(response)
        record["finish_reason"] = response.get("choices", [{}])[0].get("finish_reason")
        if response.get("usage", {}).get("prompt_tokens") != count:
            record["failure"] = {"type": "tokenizer_render_mismatch"}
            raise ValueError("tokenizer count differs from actual inference prompt; episode invalid")
        return response

    with tempfile.TemporaryDirectory(prefix="hr-relay-") as temporary:
        home = Path(temporary) / "home"
        home.mkdir()
        path, relay_token = str(Path(temporary) / "model.sock"), secrets.token_urlsafe(32)
        relay = create_model_relay(path, relay_token, MODEL, upstream)
        thread = threading.Thread(target=relay.serve_forever, daemon=True)
        thread.start()
        env = {"PATH": "/usr/sbin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(home),
               "PYTHONPATH": ":".join(str(ROOT / x) for x in ("src", "scripts", "vendor/FlashRSI")),
               "DEV_RELAY_PATH": path, "DEV_RELAY_TOKEN": relay_token, "HR_CASE_ID": case_id,
               "HR_SNAPSHOT": str(Path(snapshot).resolve()), "HR_EXPECTED_ID": expected,
               "HR_RUN_ROOT": str(Path(run_root).resolve()), "HR_HARNESS": "hermes"}
        try:
            child = subprocess.Popen(
                ["unshare", "-Urnpf", "--mount-proc", "sh", "-c", 'ip link set lo up && exec "$1" "$2"',
                 "hr", sys.executable, str(ROOT / "scripts/flashrsi_harnessrisk_supervisor.py")],
                env=env, start_new_session=True)
            try:
                return child.wait(timeout=2400)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                return 124
        finally:
            evidence = Path(run_root) / case_id
            if evidence.is_dir():
                (evidence / "contextfit_protocol_evidence.json").write_text(
                    json.dumps({"protocol": PROTOCOL_ID, "requests": decisions}, ensure_ascii=False) + "\n")
            relay.shutdown()
            thread.join(timeout=5)
            relay.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
