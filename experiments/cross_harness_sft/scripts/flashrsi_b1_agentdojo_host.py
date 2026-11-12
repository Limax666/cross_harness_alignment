"""Trusted parent for one GEPA B1 AgentDojo episode."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import tempfile
import threading
import urllib.request
from pathlib import Path

from flashrsi_model_relay import create_model_relay


ROOT = Path("/openbayes/home/cross_harness_alignment/experiments/cross_harness_sft")


def _upstream(model_token):
    def call(request):
        req = urllib.request.Request(
            "http://127.0.0.1:18761/v1/chat/completions",
            data=json.dumps(request).encode(),
            headers={
                "Authorization": "Bearer " + model_token,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            req, timeout=120
        ) as response:
            return json.load(response)

    return call


def main():
    model_token = (ROOT / "outputs/preflight_20261111/vllm_api_token").read_text(
        encoding="utf-8"
    ).strip()
    with tempfile.TemporaryDirectory(prefix="rsi-b1-relay-") as directory:
        socket_path = str(Path(directory) / "model.sock")
        relay_token = secrets.token_urlsafe(32)
        relay = create_model_relay(
            socket_path, relay_token, "qwen35-4b-preflight", _upstream(model_token)
        )
        thread = threading.Thread(target=relay.serve_forever, daemon=True)
        thread.start()
        env = {
            "PATH": "/usr/sbin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "PYTHONPATH": ":".join(
                str(ROOT / item) for item in ("src", "scripts", "vendor/FlashRSI")
            ),
            "DEV_RELAY_PATH": socket_path,
            "DEV_RELAY_TOKEN": relay_token,
            "B1_TASK_ID": os.environ["B1_TASK_ID"],
            "B1_SEED": os.environ["B1_SEED"],
            "B1_EXPECTED_SNAPSHOT_ID": os.environ["B1_EXPECTED_SNAPSHOT_ID"],
            "B1_CANONICAL_SNAPSHOT_ID": os.environ["B1_CANONICAL_SNAPSHOT_ID"],
            "B1_SNAPSHOT_JSON": os.environ["B1_SNAPSHOT_JSON"],
            "B1_CASE_OUT": os.environ["B1_CASE_OUT"],
            "B1_PYTHON": str(ROOT / ".venv-agentdojo/bin/python"),
        }
        try:
            result = subprocess.run(
                [
                    "unshare",
                    "-Urn",
                    "sh",
                    "-c",
                    'ip link set lo up && exec "$1" "$2"',
                    "rsi",
                    str(ROOT / ".venv-agentdojo/bin/python"),
                    str(ROOT / "scripts/flashrsi_b1_agentdojo_supervisor.py"),
                ],
                env=env,
                timeout=360,
                text=True,
                capture_output=True,
            )
            summary = {
                "supervisor_exit": result.returncode,
                "stdout": result.stdout[-3000:],
                "stderr": result.stderr[-3000:],
            }
            print(json.dumps(summary, ensure_ascii=False))
            return result.returncode
        finally:
            relay.shutdown()
            thread.join(timeout=5)
            relay.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
