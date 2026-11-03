#!/usr/bin/env python3
"""Run held-out SecLaw SFT and CHS-PO evaluations concurrently on GPUs 2 and 3."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent.parent
OUT = EXP / "outputs/safeclawarena_chspo150_eval"
RUNNER = EXP / "scripts/run_safeclawarena_heldout_eval.py"
sys.path.insert(0, str(EXP / "scripts"))
from run_safeclawarena_heldout_eval import heldout_tasks  # noqa: E402


def complete(role: str) -> bool:
    base = OUT / "seclaw" / role
    manifest = base / "run_manifest.json"
    if not manifest.is_file():
        return False
    cases = json.loads(manifest.read_text())["case_ids"]
    for rep in (1, 2, 3):
        for case_id in cases:
            path = base / f"rep{rep}" / f"{case_id}.json"
            try:
                row = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                return False
            if not isinstance(row.get("score"), (int, float)) or not isinstance(row.get("attack_succeeded"), bool):
                return False
    return True


def wait_nemoclaw() -> None:
    case_count = len(heldout_tasks()) * 3
    while True:
        ready = True
        for role in ("sft", "chspo150"):
            base = OUT / "nemoclaw" / role
            manifest = base / "run_manifest.json"
            if not manifest.is_file():
                ready = False
                break
            cases = json.loads(manifest.read_text())["case_ids"]
            valid = 0
            for rep in (1, 2, 3):
                for case_id in cases:
                    try:
                        row = json.loads((base / f"rep{rep}" / f"{case_id}.json").read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    valid += isinstance(row.get("score"), (int, float)) and isinstance(row.get("attack_succeeded"), bool)
            if valid != case_count:
                ready = False
                break
        if ready:
            print("NemoClaw SFT and CHS-PO: 90/90 each; starting SecLaw workers", flush=True)
            return
        print("Waiting for both NemoClaw arms to reach 90/90 valid verdicts", flush=True)
        time.sleep(60)


def worker(role: str, gpu: int, port: int, host_port: int, container: str, model_id: str) -> subprocess.Popen:
    config = {
        "model": "openai/" + model_id,
        "api_key": "local-eval",
        "api_base_url": f"http://10.0.2.2:{port}",
    }
    config_path = OUT / f"seclaw-{role}-gpu{gpu}-model.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if complete(role):
        print(f"SecLaw/{role} already complete", flush=True)
        return subprocess.Popen([sys.executable, "-c", "pass"])
    cmd = [sys.executable, str(RUNNER), "--platform", "seclaw", "--role", role,
           "--model-config", str(config_path), "--output", str(OUT / "seclaw" / role),
           "--repetitions", "3", "--seed-endpoint", f"http://127.0.0.1:{port}/admin/seed",
           "--container-name", container, "--gateway-port", str(host_port)]
    log = (OUT / f"seclaw-{role}-gpu{gpu}-run.log").open("w")
    print(f"Starting SecLaw/{role} on GPU {gpu}, isolated container {container}", flush=True)
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu)})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    wait_nemoclaw()
    # Reuse the already-loaded SFT and CHS-PO servers on ports 9321 / 9320.
    workers = [
        worker("sft", 2, 9321, 19092, "seclaw-env-gpu2", "qwen35-2b-sft-safeclaw-gpu2"),
        worker("chspo150", 3, 9320, 19093, "seclaw-env-gpu3", "qwen35-2b-chspo150-safeclaw"),
    ]
    codes = [p.wait() for p in workers]
    subprocess.run([sys.executable, str(EXP / "scripts/summarize_safeclawarena_heldout_eval.py")],
                   cwd=ROOT, check=False)
    if any(code != 0 for code in codes):
        return 1
    if not all(complete(role) for role in ("sft", "chspo150")):
        return 1
    print("SecLaw SFT and CHS-PO: 90/90 each", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
