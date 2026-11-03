#!/usr/bin/env python3
"""Run the native SafeClawArena judge on v7 test families, with resumable repeats.

The upstream judge uses a fixed container name per platform and one output file
per task, so this wrapper deliberately runs serially and isolates each repeat.
It does not replace or reinterpret the upstream scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
BENCH = EXP / "vendor/SafeClawArena"
SPLIT = EXP / "data/sft/teacher_release_v7/family_split_manifest.jsonl"


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"Interrupted by signal {signum}")


def heldout_tasks() -> list[Path]:
    families = set()
    exposed = set()
    for line in SPLIT.read_text().splitlines():
        item = json.loads(line)
        if item["benchmark_namespace"] != "safeclawarena":
            continue
        family = item["source_family_id"].lower()
        (families if item["split"] == "test" else exposed).add(family)
    if not families or families & exposed:
        raise RuntimeError("SafeClawArena test families are missing or overlap training/validation")
    cases = []
    for path in sorted((BENCH / "tasks").glob("*/*.json")):
        task = json.loads(path.read_text())
        task_id = task["metadata"]["task_id"]
        parts = task_id.split("-")
        family = f"safeclawarena:{parts[0].upper()}:{parts[1]}".lower()
        if family in families:
            cases.append(path)
    if len(cases) != 30:
        raise RuntimeError(f"Expected 30 frozen v7 test cases, found {len(cases)}")
    return cases


def main() -> int:
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGHUP, _interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=("nemoclaw", "seclaw"))
    parser.add_argument("--role", required=True, choices=("sft", "chspo150"))
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--seed-endpoint", help="Host-local model /admin/seed endpoint; resets the request cap and RNG per case")
    parser.add_argument("--container-name", help="Isolated Docker container name for concurrent workers")
    parser.add_argument("--gateway-port", type=int, help="Isolated host port for concurrent gateway workers")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    args.model_config = args.model_config.resolve()
    args.output = args.output.resolve()
    tasks = heldout_tasks()
    if args.list_only:
        for path in tasks:
            print(path.relative_to(BENCH))
        return 0
    if not args.model_config.is_file():
        parser.error("--model-config must exist")
    model_config = json.loads(args.model_config.read_text())
    if not all(model_config.get(k) for k in ("model", "api_key", "api_base_url")):
        parser.error("model config needs model, api_key, and api_base_url")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "platform": args.platform,
        "role": args.role,
        "benchmark_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BENCH, text=True).strip(),
        "split_manifest": str(SPLIT),
        "test_families": ["safeclawarena:PSE:2.4", "safeclawarena:SSI:1.2"],
        "case_ids": [json.loads(p.read_text())["metadata"]["task_id"] for p in tasks],
        "repetitions": args.repetitions,
        "model": model_config["model"],
        "api_base_url": model_config["api_base_url"],
        "seed_endpoint": args.seed_endpoint,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    ledger = args.output / "attempts.jsonl"
    error_count = 0
    for rep in range(1, args.repetitions + 1):
        rep_dir = args.output / f"rep{rep}"
        rep_dir.mkdir(exist_ok=True)
        for idx, case in enumerate(tasks, 1):
            case_id = json.loads(case.read_text())["metadata"]["task_id"]
            result_path = rep_dir / f"{case_id}.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if "error" not in result and isinstance(result.get("score"), (int, float)) and isinstance(result.get("attack_succeeded"), bool):
                    print(f"SKIP {args.platform} {args.role} rep{rep} {idx}/30 {case_id}", flush=True)
                    continue
            seed = int.from_bytes(hashlib.sha256(f"safeclaw-v7:{case_id}:rep{rep}".encode()).digest()[:4], "big")
            if args.seed_endpoint:
                request = urllib.request.Request(
                    args.seed_endpoint,
                    data=json.dumps({"seed": seed}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    if json.load(response).get("seed") != seed:
                        raise RuntimeError(f"Model seed reset failed for {case_id} rep{rep}")
            cmd = [sys.executable, str(BENCH / "scripts/judge.py"), str(case),
                   "--platform", args.platform, "--model-config", str(args.model_config),
                   "--output", str(rep_dir)]
            log_path = rep_dir / f"{case_id}.log"
            print(f"RUN {args.platform} {args.role} rep{rep} {idx}/30 {case_id}", flush=True)
            judge_env = os.environ.copy()
            if args.container_name:
                judge_env["SAFECLAW_CONTAINER"] = args.container_name
                judge_env["SAFECLAW_LOCKFILE"] = f"/tmp/safeclaw_reset_{args.container_name}.lock"
            if args.gateway_port:
                judge_env["SAFECLAW_GATEWAY_PORT"] = str(args.gateway_port)
            with log_path.open("w") as log:
                proc = subprocess.Popen(cmd, cwd=BENCH, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True, env=judge_env)
                try:
                    code = proc.wait(timeout=args.timeout_seconds)
                    failure = None if code == 0 else f"judge_exit_{code}"
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    if isinstance(exc, KeyboardInterrupt):
                        raise
                    failure = "wrapper_timeout"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            if "error" in result:
                failure = str(result["error"])
            elif not isinstance(result.get("score"), (int, float)) or not isinstance(result.get("attack_succeeded"), bool):
                failure = failure or "missing_official_verdict"
            row = {"platform": args.platform, "role": args.role, "rep": rep,
                   "case_id": case_id, "seed": seed, "status": "error" if failure else "scored",
                   "error": failure, "score": result.get("score"),
                   "attack_succeeded": result.get("attack_succeeded"),
                   "result_path": str(result_path), "log_path": str(log_path)}
            with ledger.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if failure:
                error_count += 1
                # A transport/setup failure on the first case invalidates the
                # adapter; do not spend an entire suite producing false zeros.
                if idx == 1 and rep == 1:
                    return 1
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
