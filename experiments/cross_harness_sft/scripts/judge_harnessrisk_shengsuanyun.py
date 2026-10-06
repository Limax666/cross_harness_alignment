#!/usr/bin/env python3
"""Run the HarnessRisk canonical LLM judge through the configured private router."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
JUDGE = ROOT / "HarnessRisk/harness_adapter/scripts/eval_via_llm/llm_judge_multi_harness_batch_aiwave.py"
DATA = ROOT / "HarnessRisk/data/HarnessRisk"
DEFAULT_CREDENTIALS = Path.home() / ".dsh/.credentials.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--deployment", default="openai/gpt-5.4-nano")
    parser.add_argument("--base-url", default="https://router.shengsuanyun.com/api/v1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    env = os.environ.copy()
    if not args.dry_run and not env.get("SHENGSUANYUN_API_KEY"):
        if not args.credentials.is_file():
            parser.error("SHENGSUANYUN_API_KEY and credentials file are both missing")
        credentials = yaml.safe_load(args.credentials.read_text()) or {}
        key = credentials.get("SHENGSUANYUN_API_KEY")
        if not isinstance(key, str) or not key.strip():
            parser.error("SHENGSUANYUN_API_KEY is missing from credentials file")
        env["SHENGSUANYUN_API_KEY"] = key
    command = [
        os.sys.executable, str(JUDGE), str(args.batch_dir.resolve()),
        "--data-dir", str(DATA), "--base-url", args.base_url,
        "--deployment", args.deployment, "--model", args.deployment,
        "--api-key-env", "SHENGSUANYUN_API_KEY",
        "--max-completion-tokens", "2048",
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.overwrite:
        command.append("--overwrite")
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode:
        return completed.returncode
    if args.dry_run:
        return 0
    manifest = args.batch_dir / "llm_judge_multi_harness/manifest.json"
    records = json.loads(manifest.read_text())
    errors = [row for row in records if row.get("status") not in ("judged", "existing")]
    if errors:
        print(f"HarnessRisk LLM judge incomplete: {len(errors)}/{len(records)} cases", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
