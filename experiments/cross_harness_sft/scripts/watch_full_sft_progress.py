#!/usr/bin/env python3
"""Read-only, once-per-run progress snapshot for the active Qwen3.5-2B full SFT.

Intended for a ten-minute user crontab. It never signals or restarts training.
"""
from __future__ import annotations

import fcntl
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data/home/liumingxiao/cross_harness_alignment")
EXP = ROOT / "experiments/cross_harness_sft"
LOG = EXP / "outputs/sft_training/qwen35-2b-base-v7-all1421-manual.log"
REPORT_DIR = EXP / "outputs/sft_training/full_sft_monitor"
CHECKPOINT_DIR = EXP / "checkpoints/qwen35-2b-base-teacher-v7-all1421-manual"
SESSION = "sft-qwen35-2b-full"
TOTAL_STEPS = 267
STEP_RE = re.compile(r"step:(\d+)\s+-\s+[^\r\n]*?train/loss:([0-9.eE+-]+)")
LOSS_MA_RE = re.compile(r"train/loss_ma32:([0-9.eE+-]+)")
VAL_RE = re.compile(r"val/loss[': ]+([0-9.eE+-]+)")
ERROR_RE = re.compile(r"(torch\.OutOfMemoryError|ChildFailedError|Error executing job|Traceback \(most recent call last\)|Full-SFT guard failed)")


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, timeout=15, check=False)


def tail_text(path: Path, limit: int = 2_000_000) -> str:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - limit))
        return stream.read().decode("utf-8", "replace").replace("\r", "\n")


def latest_value(pattern: re.Pattern[str], content: str) -> float | None:
    values = pattern.findall(content)
    return float(values[-1]) if values else None


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "monitor.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        terminal_marker = REPORT_DIR / "monitor.finished"
        if terminal_marker.exists():
            return

        now = datetime.now(timezone.utc)
        state_path = REPORT_DIR / "monitor_state.json"
        prior = json.loads(state_path.read_text()) if state_path.exists() else {}
        session_alive = run("/usr/bin/tmux", "has-session", "-t", SESSION).returncode == 0
        processes = run("/usr/bin/ps", "-eo", "args=").stdout.splitlines()
        trainer_alive = any("-m verl.trainer.sft_trainer" in line and
                            "qwen35-2b-base-teacher-v7-all1421-manual" in line
                            for line in processes)
        gpu = run("/usr/bin/nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                  "--format=csv,noheader,nounits")
        gpu_rows = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
        gpu_1_2 = [line for line in gpu_rows if line.split(",", 1)[0].strip() in {"1", "2"}]

        content = tail_text(LOG) if LOG.exists() else ""
        steps = STEP_RE.findall(content)
        step = int(steps[-1][0]) if steps else 0
        train_loss = float(steps[-1][1]) if steps else None
        moving_loss = latest_value(LOSS_MA_RE, content)
        val_loss = latest_value(VAL_RE, content)
        errors = ERROR_RE.findall(content)
        last_step_seen = now.isoformat() if step != prior.get("step") else prior.get("last_step_seen", now.isoformat())
        minutes_without_step = (now - datetime.fromisoformat(last_step_seen)).total_seconds() / 60
        checkpoint_ready = (CHECKPOINT_DIR / f"global_step_{TOTAL_STEPS}").exists()
        final_metrics = "Final validation metrics:" in content
        if trainer_alive:
            status = "stalled" if minutes_without_step >= 20 else "running"
        elif step >= TOTAL_STEPS and (checkpoint_ready or final_metrics):
            status = "completed"
        elif session_alive:
            status = "no_training_process_in_tmux"
        else:
            status = "stopped_before_completion"
        if errors and not trainer_alive and status != "completed":
            status = "failed"
        if train_loss is not None and not math.isfinite(train_loss):
            status = "nonfinite_loss"

        report = {
            "checked_at": now.astimezone().isoformat(), "status": status,
            "step": step, "total_steps": TOTAL_STEPS,
            "epoch": min(3, (max(step, 1) - 1) // 89 + 1) if step else 0,
            "train_loss": train_loss, "train_loss_ma32": moving_loss,
            "validation_loss_latest": val_loss,
            "steps_since_last_check": step - prior.get("step", step),
            "minutes_without_step": round(minutes_without_step, 1),
            "tmux_session_alive": session_alive, "training_process_alive": trainer_alive,
            "gpu_1_2": gpu_1_2, "recent_errors": errors[-3:],
            "log_size_bytes": LOG.stat().st_size if LOG.exists() else 0,
        }
        line = (f"{report['checked_at']} status={status} step={step}/{TOTAL_STEPS} "
                f"epoch={report['epoch']}/3 loss={train_loss} ma32={moving_loss} "
                f"val={val_loss} delta_steps={report['steps_since_last_check']} "
                f"GPU={gpu_1_2} errors={errors[-3:]}")
        (REPORT_DIR / "latest.txt").write_text(line + "\n")
        with (REPORT_DIR / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(report, ensure_ascii=False) + "\n")
        state_path.write_text(json.dumps({"step": step, "last_step_seen": last_step_seen}))
        if status in {"completed", "failed", "stopped_before_completion", "no_training_process_in_tmux", "nonfinite_loss"}:
            terminal_marker.write_text(report["checked_at"] + " " + status + "\n")
        print(line, flush=True)


if __name__ == "__main__":
    main()
