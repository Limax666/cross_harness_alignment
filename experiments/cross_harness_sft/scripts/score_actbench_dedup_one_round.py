#!/usr/bin/env python3
"""Wait for a role's ActBench collection, then apply official combined AGS scoring."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cross_harness_sft"
OUT = EXP / "outputs/actbench/dedup_qwen35_2b_v7"
BENCH = EXP / "vendor/actbench"
PYTHON = "/data/home/liumingxiao/miniforge3/envs/cross-harness-sft/bin/python"
JUDGE = "judge/openai/gpt-5.4-nano"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", choices=("base", "sft"), required=True)
    args = ap.parse_args()
    role_dir = OUT / "one_round_t480_r1" / args.role
    progress = role_dir / "progress.json"
    session = f"actbench-{args.role}-eval"
    while True:
        if progress.is_file():
            state = json.loads(progress.read_text(encoding="utf-8"))
            if state.get("status") == "collection_complete":
                break
        active = subprocess.run(["tmux", "has-session", "-t", session],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not active:
            raise SystemExit(f"Collection session {session} stopped before completion; inspect its log")
        time.sleep(30)

    credentials = yaml.safe_load(Path("/data/home/liumingxiao/.dsh/.credentials.yaml").read_text(encoding="utf-8"))
    key = credentials.get("SHENGSUANYUN_API_KEY") if isinstance(credentials, dict) else None
    if not key:
        raise SystemExit("SHENGSUANYUN_API_KEY missing from local credentials")
    env = os.environ.copy()
    env["SHENGSUANYUN_API_KEY"] = key
    env["ACTBENCH_LLM_BACKENDS_CONFIG"] = str(OUT / "judge_provider.yaml")
    summary = {"role": args.role, "judge_model": JUDGE, "harnesses": {}}
    for harness in ("hermes", "claudecode"):
        batch_dirs = sorted((role_dir / harness).glob("batch_[0-9][0-9][0-9]"))
        if len(batch_dirs) != 33:
            raise SystemExit(f"Expected 33 completed {harness} batches; found {len(batch_dirs)}")
        results = []
        for batch_dir in batch_dirs:
            score_path = batch_dir / "combined_ags_score.json"
            valid = False
            if score_path.is_file():
                try:
                    saved = json.loads(score_path.read_text(encoding="utf-8"))
                    valid = saved.get("scoring_mode") == "combined_ags" and saved.get("trajectory_count") == 8
                except (OSError, ValueError):
                    pass
            if not valid:
                command = [PYTHON, str(BENCH / "scripts/actbench_score.py"),
                           "--trajectory", str(batch_dir / "trajectories"),
                           "--mode", "combined-ags", "--judge-model", JUDGE,
                           "--output", str(score_path)]
                with (batch_dir / "scorer.log").open("a", encoding="utf-8") as log:
                    proc = subprocess.run(command, cwd=batch_dir, env=env,
                                          stdout=log, stderr=subprocess.STDOUT)
                if proc.returncode:
                    raise SystemExit(f"Official scorer failed: {batch_dir}; inspect scorer.log")
            saved = json.loads(score_path.read_text(encoding="utf-8"))
            if saved.get("trajectory_count") != 8:
                raise SystemExit(f"Unexpected scored trajectory count in {score_path}")
            results.extend(saved.get("results") or [])
            print(f"[{args.role}/{harness}] scored {len(results)}/264", flush=True)
            time.sleep(1)
        if len(results) != 264:
            raise SystemExit(f"Expected 264 {harness} scores, found {len(results)}")
        valid = [row for row in results if not row.get("evaluation_error")]
        summary["harnesses"][harness] = {
            "trajectories": len(results), "valid_scores": len(valid),
            "evaluation_errors": len(results) - len(valid),
            "asr": (sum(bool(row.get("attack_pass")) for row in valid) / len(valid)) if valid else None,
            "mean_ags": (sum(float(row.get("ags", 0)) for row in valid) / len(valid)) if valid else None,
        }
    target = role_dir / "official_combined_ags_summary.json"
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
