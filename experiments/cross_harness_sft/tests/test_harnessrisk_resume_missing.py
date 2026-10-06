"""A missing-trajectory case is retried once without erasing its old evidence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BATCH = ROOT / "HarnessRisk/harness_adapter/scripts/run_case_scripts/run_batch.sh"


def test_retry_only_missing_trajectory_and_preserve_audit():
    with tempfile.TemporaryDirectory(prefix="hr-resume-test-") as temporary:
        root = Path(temporary)
        scripts = root / "harness_adapter/scripts/run_case_scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(BATCH, scripts / "run_batch.sh")
        run_case = scripts / "run_case.sh"
        run_case.write_text('''#!/bin/bash
set -e
while [[ $# -gt 0 ]]; do
 case "$1" in
  --run-root) run_root="$2"; shift 2;;
  --run-id) run_id="$2"; shift 2;;
  --case-id) case_id="$2"; shift 2;;
  *) shift;;
 esac
done
mkdir -p "$run_root/$run_id/trajectory"
printf '{"case_id":"%s","status":"completed"}\\n' "$case_id" > "$run_root/$run_id/trajectory/trajectory.json"
''', encoding="utf-8")
        run_case.chmod(0o755)
        evaluator = root / "harness_adapter/scripts/evaluate_run.py"
        evaluator.write_text('''import json, sys
from pathlib import Path
p = Path(sys.argv[sys.argv.index("--output") + 1]); p.write_text(json.dumps({"runs": [], "summary": {}}))
''', encoding="utf-8")
        data = root / "data"
        data.mkdir()
        (data / "action_001.json").write_text('{"case_id":"action_001"}', encoding="utf-8")
        batch_root = root / "runs" / "resume-test"
        old_run = batch_root / "resume-test-action_001"
        old_run.mkdir(parents=True)
        (old_run / "old-error.txt").write_text("prior failure", encoding="utf-8")
        manifest = batch_root / "batch_manifest.jsonl"
        manifest.write_text(json.dumps({"case_id": "action_001", "run_id": old_run.name,
                                        "run_dir": str(old_run), "status": "failed", "exit_code": 1,
                                        "repetition": 1}) + "\n", encoding="utf-8")
        result = subprocess.run(
            ["bash", str(scripts / "run_batch.sh"), "--harness", "openclaw", "--data-dir", str(data),
             "--run-root", str(root / "runs"), "--batch-id", "resume-test", "--", "--timeout-seconds", "300"],
            env={**os.environ, "RESUME_BATCH": "1", "RETRY_MISSING_TRAJECTORY": "1"},
            text=True, capture_output=True, timeout=20,
        )
        assert result.returncode == 0, result.stderr
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["status"] == "completed"
        archive = batch_root / "archived_missing_trajectory"
        assert (archive / "replaced_manifest_rows.jsonl").exists()
        assert len(list(archive.glob("resume-test-action_001-*"))) == 1
        assert (old_run / "trajectory/trajectory.json").exists()
