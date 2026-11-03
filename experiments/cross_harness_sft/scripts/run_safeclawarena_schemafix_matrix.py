#!/usr/bin/env python3
"""Evaluate SFT and CHS-PO on held-out SafeClawArena with isolated tool schemas.

Run both model arms concurrently on separate GPUs, and keep the pre-fix
evaluation untouched in its original output directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent.parent
OUT = EXP / "outputs/safeclawarena_chspo150_eval_schemafix_v3_20261016"
SERVER = EXP / "scripts/local_qwen_openai_server.py"
RUNNER = EXP / "scripts/run_safeclawarena_heldout_eval.py"
MODELS = {
    "sft": EXP / "checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267",
    "chspo150": EXP / "outputs/rl/multiharness_chspo_r20g_v4_from130_20261012/hf_merged_step_150",
}
RESOURCES = {
    "sft": {"gpu": 2, "port": 9321, "gateway": 19101},
    "chspo150": {"gpu": 3, "port": 9320, "gateway": 19102},
}


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"Interrupted by signal {signum}")


def say(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def start_server(role: str) -> subprocess.Popen:
    r = RESOURCES[role]
    model_id = f"qwen35-2b-{role}-safeclaw-schemafix"
    log = (OUT / f"{role}-server.log").open("a")
    cmd = [
        sys.executable, str(SERVER), "--role", "rl" if role == "chspo150" else "sft",
        "--base-model", str(MODELS[role]), "--model-id", model_id,
        "--host", "127.0.0.1", "--port", str(r["port"]),
        "--max-new-tokens", "1024", "--max-input-tokens", "65536",
        "--prompt-format", "segment", "--temperature", "0.3", "--seed", "42",
        "--max-requests-per-episode", "24", "--full-precision",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(r["gpu"]),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    log.close()
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError(f"{role} model server exited during startup")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{r['port']}/health", timeout=2) as response:
                if json.load(response).get("model") == model_id:
                    say(f"{role} model ready on GPU {r['gpu']}")
                    return proc
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError(f"{role} model server health check timed out")


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def run_stage(platform: str, servers: dict[str, subprocess.Popen]) -> None:
    workers: dict[str, subprocess.Popen] = {}
    try:
        for role in MODELS:
            if servers[role].poll() is not None:
                raise RuntimeError(f"{role} model server exited before {platform}")
            r = RESOURCES[role]
            model_id = f"qwen35-2b-{role}-safeclaw-schemafix"
            config = {"model": ("openai/" if platform == "seclaw" else "") + model_id,
                      "api_key": "local-eval", "api_base_url": f"http://10.0.2.2:{r['port']}"}
            config_path = OUT / f"{platform}-{role}-model.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            log = (OUT / f"{platform}-{role}-run.log").open("a")
            cmd = [
                sys.executable, str(RUNNER), "--platform", platform, "--role", role,
                "--model-config", str(config_path), "--output", str(OUT / platform / role),
                "--repetitions", "3", "--seed-endpoint", f"http://127.0.0.1:{r['port']}/admin/seed",
                "--container-name", f"{platform}-schemafix-{role}",
                "--gateway-port", str(r["gateway"] + (2 if platform == "seclaw" else 0)),
            ]
            workers[role] = subprocess.Popen(cmd, cwd=ROOT, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
            log.close()
            say(f"Started {platform}/{role} on GPU {r['gpu']}")
        while any(proc.poll() is None for proc in workers.values()):
            failed = [role for role, proc in workers.items() if proc.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"{platform} worker failed: {failed}")
            for role, proc in servers.items():
                if proc.poll() is not None:
                    raise RuntimeError(f"{role} model server exited during {platform}")
            time.sleep(15)
        failed = [role for role, proc in workers.items() if proc.returncode != 0]
        if failed:
            raise RuntimeError(f"{platform} worker failed: {failed}")
        say(f"{platform}: both arms finished")
    finally:
        for proc in workers.values():
            stop(proc)


def main() -> int:
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGHUP, _interrupt)
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = {
        "adapter_sha256": hashlib.sha256(SERVER.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        "prompt_format": "segment", "max_new_tokens": 1024,
        "max_input_tokens": 65536,
        "temperature": 0.3, "seed": 42,
        "models": {role: str(path) for role, path in MODELS.items()},
        "output": str(OUT),
    }
    record = OUT / "provenance.json"
    if record.exists() and json.loads(record.read_text()) != provenance:
        raise RuntimeError("Existing output has different provenance; choose a fresh run directory")
    record.write_text(json.dumps(provenance, indent=2) + "\n")
    servers: dict[str, subprocess.Popen] = {}
    try:
        for role in MODELS:
            servers[role] = start_server(role)
        for platform in ("nemoclaw", "seclaw"):
            run_stage(platform, servers)
        say("SafeClawArena schemafix matrix complete")
        return 0
    finally:
        for proc in servers.values():
            stop(proc)


if __name__ == "__main__":
    raise SystemExit(main())
