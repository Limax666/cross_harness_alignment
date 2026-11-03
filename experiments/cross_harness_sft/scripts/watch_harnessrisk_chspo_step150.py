#!/usr/bin/env python3
"""Track three independent jobs, release their model servers, summarize only complete scores."""
import json
import subprocess
import time
from datetime import datetime, timezone

from summarize_harnessrisk_chspo_step150 import main as summarize
from summarize_harnessrisk_sft_rl_step20 import EXP, METRICS, JUDGE_MODEL

OUT = EXP / "outputs/harnessrisk_qwen35_2b_chspo_v4_step150"
JOBS = {"hermes": 1, "nanobot": 2, "openclaw": 2}


def alive(session):
    return subprocess.run(["tmux", "has-session", "-t", session],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


while True:
    progress = {}
    for harness, gpu in JOBS.items():
        session = f"hr-chspo150-{harness}-gpu{gpu}"
        running = alive(session)
        counts = []
        for repetition in (1, 2, 3):
            batch = OUT / "runs" / f"{harness}-rl" / f"chspo150v4p1-rl-{harness}-r{repetition}"
            manifest = batch / "batch_manifest.jsonl"
            trajectories = [json.loads(line) for line in manifest.read_text().splitlines()
                            if line.strip()] if manifest.exists() else []
            judge_file = batch / "llm_judge_multi_harness/manifest.json"
            judged = json.loads(judge_file.read_text()) if judge_file.exists() else []
            valid = [row for row in judged if row.get("status") in ("judged", "existing")
                     and (row.get("judge") or {}).get("_llm_metadata", {}).get("deployment") == JUDGE_MODEL
                     and all((row.get("judge") or {}).get(key) in (0, 1) for key, _ in METRICS)]
            counts.append({"repetition": repetition,
                           "trajectories": len({row["case_id"] for row in trajectories}),
                           "judged": len({row["case_id"] for row in valid})})
        progress[harness] = {"running": running, "batches": counts}
        if not running:
            server = f"hr2b-rl-{harness}-gpu{gpu}"
            if alive(server):
                subprocess.run(["tmux", "kill-session", "-t", server], check=False)
    state = {"checked_utc": datetime.now(timezone.utc).isoformat(), "harnesses": progress}
    (OUT / "progress.json").write_text(json.dumps(state, indent=2) + "\n")
    complete = all(batch["trajectories"] == 128 and batch["judged"] == 128
                   for job in progress.values() for batch in job["batches"])
    if complete:
        summarize()
        # All inference requests are finished before this point.
        for harness, gpu in JOBS.items():
            server = f"hr2b-rl-{harness}-gpu{gpu}"
            if alive(server):
                subprocess.run(["tmux", "kill-session", "-t", server], check=False)
        break
    if not any(job["running"] for job in progress.values()):
        raise SystemExit("Evaluation jobs exited before all nine batches were fully judged; see progress.json and job logs.")
    time.sleep(30)
