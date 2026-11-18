"""Run a frozen, development-only 16K AgentDojo/MiniAgent validation sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RECORDS = REPO / "experiment_records/flashrsi_b1_20261112"
SNAPSHOT = "487f11214c730975b9ce68b903026e68321f5e7c7ff779c98844c293d6a65ade"
RUN_ROOT = ROOT / "outputs/flashrsi_local_20261112"
HOSTS = {'miniagent': 'flashrsi_b1_agentdojo_host.py',
         'opencode': 'flashrsi_opencode_full_host_v2.py',
         'openclaw': 'flashrsi_openclaw_full_host.py',
         'hermes': 'flashrsi_hermes_full_host.py', 'pi': 'flashrsi_pi_full_host_v2.py'}
SUITES = {"workspace", "travel", "banking", "slack"}
TASK = re.compile(r"(workspace|travel|banking|slack):user_task_[0-9]+:(clean|injection_task_[0-9]+)\Z")


def validation_cases(manifest):
    rows = []
    seen = set()
    for row in manifest["cases"]:
        if row.get("split") != "validation":
            continue
        task = row.get("task_id", "")
        if (not TASK.fullmatch(task) or row.get("family") != task.rsplit(":", 1)[0]
                or row.get("seed") != 0 or task in seen):
            raise ValueError("invalid or duplicate validation task")
        seen.add(task)
        rows.append(row)
    if not rows:
        raise ValueError("no validation tasks")
    return sorted(rows, key=lambda row: row["task_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--harness', choices=HOSTS, default='miniagent')
    args = parser.parse_args()
    run = RUN_ROOT / ('minia_validation_16k' if args.harness == 'miniagent' else f'{args.harness}_validation_16k')
    with urllib.request.urlopen("http://127.0.0.1:18761/v1/models", timeout=5) as response:
        models = json.load(response)["data"]
    if not any(x.get("id") == "qwen35-4b-preflight" and x.get("max_model_len") == 16384 for x in models):
        raise RuntimeError("expected pinned 16K development model endpoint")
    manifest_path = RECORDS / "source_run_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    rows = validation_cases(manifest)
    run.mkdir(parents=True, exist_ok=False)
    (run / "snapshots").mkdir()
    (run / "cases").mkdir()
    (run / "run_manifest.json").write_text(json.dumps({
        "status": "development_only_not_formally_qualified",
        "harness": args.harness,
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "model_revision": manifest["model_revision"], "flashrsi_revision": manifest["flashrsi_revision"],
        "max_model_len": 16384, "sampling_backend": "vllm_no_flashinfer_sampler",
        "cases": rows,
    }, ensure_ascii=False, indent=2) + "\n")
    archive = RECORDS / "gepa_miniagent_agentdojo_complete.tar.gz"
    with tarfile.open(archive) as tar:
        for suite in SUITES:
            path = f"gepa-miniagent-agentdojo_security-20261112-063859-287281/snapshots/initial-{suite}.json"
            data = tar.extractfile(path).read()
            if json.loads(data)["id"] != manifest["snapshot_by_suite"][suite]["id"]:
                raise ValueError("suite snapshot mismatch")
            (run / "snapshots" / f"{suite}.json").write_bytes(data)
    for row in rows:
        task = row["task_id"]
        suite = task.split(":", 1)[0]
        name = task.replace(":", "__")
        env = {**os.environ, "PYTHONPATH": ":".join(str(ROOT / x) for x in ("src", "scripts", "vendor/FlashRSI"))}
        case_path = run / "cases" / f"{name}.json"
        if args.harness == 'miniagent':
            env.update(B1_TASK_ID=task, B1_SEED="0", B1_CANONICAL_SNAPSHOT_ID=SNAPSHOT,
                       B1_EXPECTED_SNAPSHOT_ID=manifest["snapshot_by_suite"][suite]["id"],
                       B1_SNAPSHOT_JSON=str(run / "snapshots" / f"{suite}.json"), B1_CASE_OUT=str(case_path))
        else:
            env.update(B0_TASK_ID=task, B0_SEED="0", B0_CASE_OUT=str(case_path))
        with (run / "cases" / f"{name}.log").open("x") as log:
            try:
                result = subprocess.run([sys.executable, str(ROOT / 'scripts' / HOSTS[args.harness])],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=390)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = "host_timeout"
        with (run / "attempts.jsonl").open("a") as out:
            out.write(json.dumps({"task_id": task, "host_exit": code,
                                  "case_exists": case_path.is_file()}) + "\n")


if __name__ == "__main__":
    main()
