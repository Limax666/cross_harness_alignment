#!/usr/bin/env python3
"""Continue the held-out SafeClawArena matrix after the active NemoClaw RL run.

The official judge uses one fixed gateway port, so stages are serialized. Each
stage must have all 30 cases x 3 repetitions scored before the next begins.
"""

from __future__ import annotations

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
OUT = EXP / "outputs/safeclawarena_chspo150_eval"
SFT = EXP / "checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267"
RL = EXP / "outputs/rl/multiharness_chspo_r20g_v4_from130_20261012/hf_merged_step_150"
sys.path.insert(0, str(EXP / "scripts"))
from run_safeclawarena_heldout_eval import heldout_tasks  # noqa: E402


def say(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def complete(platform: str, role: str) -> bool:
    base = OUT / platform / role
    cases = [json.loads(p.read_text())["metadata"]["task_id"] for p in heldout_tasks()]
    for rep in (1, 2, 3):
        for case in cases:
            path = base / f"rep{rep}" / f"{case}.json"
            if not path.is_file():
                return False
            try:
                result = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                return False
            if not isinstance(result.get("score"), (int, float)) or not isinstance(result.get("attack_succeeded"), bool):
                return False
    return True


def tmux_alive(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def wait_for_existing() -> None:
    name = "safeclaw-nemo-chspo150"
    while tmux_alive(name):
        if not tmux_alive("safeclaw-chspo150-server"):
            raise RuntimeError("Active NemoClaw RL model server exited")
        say("Waiting for active NemoClaw CHSPO run to finish")
        time.sleep(60)
    if not complete("nemoclaw", "chspo150"):
        raise RuntimeError("NemoClaw CHSPO run exited without 90 valid verdicts")
    say("NemoClaw CHSPO: 90/90 valid verdicts")


def wait_for_seclaw_image() -> None:
    while subprocess.run(["docker", "image", "inspect", "seclaw-env:0.1.0"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        if not tmux_alive("safeclaw-seclaw-build"):
            raise RuntimeError("SecLaw image build exited without producing seclaw-env:0.1.0")
        say("Waiting for SecLaw image build")
        time.sleep(60)


def server(role: str) -> subprocess.Popen:
    model_id = f"qwen35-2b-{role}-safeclaw"
    checkpoint = RL if role == "chspo150" else SFT
    server_role = "rl" if role == "chspo150" else "sft"
    log = (OUT / f"{role}-matrix-server.log").open("w")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="3", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    cmd = [sys.executable, str(EXP / "scripts/local_qwen_openai_server.py"),
           "--role", server_role, "--base-model", str(checkpoint), "--model-id", model_id,
           "--host", "127.0.0.1", "--port", "9320", "--max-new-tokens", "1024",
           "--max-input-tokens", "32768", "--prompt-format", "segment",
           "--temperature", "0.3", "--seed", "42", "--max-requests-per-episode", "24",
           "--full-precision"]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError(f"{role} model server exited during startup; see {log.name}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:9320/health", timeout=2) as response:
                health = json.load(response)
            if health.get("model") == model_id:
                say(f"{role} model server ready")
                return proc
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError(f"{role} model server failed health check")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def run_stage(platform: str, role: str) -> None:
    if complete(platform, role):
        say(f"SKIP complete {platform}/{role}")
        return
    config = {"model": f"qwen35-2b-{role}-safeclaw", "api_key": "local-eval",
              "api_base_url": "http://10.0.2.2:9320"}
    if platform == "seclaw":
        config["model"] = "openai/" + config["model"]
    config_path = OUT / f"{platform}-{role}-model.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    log_path = OUT / f"{platform}-{role}-run.log"
    cmd = [sys.executable, str(EXP / "scripts/run_safeclawarena_heldout_eval.py"),
           "--platform", platform, "--role", role, "--model-config", str(config_path),
           "--output", str(OUT / platform / role), "--repetitions", "3",
           "--seed-endpoint", "http://127.0.0.1:9320/admin/seed"]
    say(f"RUN {platform}/{role}")
    with log_path.open("w") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not complete(platform, role):
        raise RuntimeError(f"{platform}/{role} finished without 90 valid verdicts")
    say(f"DONE {platform}/{role}: 90/90 valid verdicts")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    wait_for_existing()
    subprocess.run([sys.executable, str(EXP / "scripts/summarize_safeclawarena_heldout_eval.py")], check=True)
    subprocess.run(["tmux", "kill-session", "-t", "safeclaw-chspo150-server"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    wait_for_seclaw_image()
    for platform, role in (("nemoclaw", "sft"), ("seclaw", "chspo150"), ("seclaw", "sft")):
        proc = server(role)
        try:
            run_stage(platform, role)
        finally:
            stop_server(proc)
        subprocess.run([sys.executable, str(EXP / "scripts/summarize_safeclawarena_heldout_eval.py")], check=True)
    say("ALL STAGES COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
