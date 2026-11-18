"""Freeze and launch the required full-128 Claude v8 initial control once."""
import datetime, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

from flashrsi_matrix_repair_loop import ROOT, STATE, active_runs, save, scan

PRIOR = ROOT / "outputs/flashrsi_harnessrisk_4b_private_claude_events_tools_v7_gpu1_20261115"
BASE = ROOT / "outputs/flashrsi_harnessrisk_4b_private_claude_cli_projection_v8_gpu1_20261115"
HOST = "flashrsi_harnessrisk_host_native_v8.py"
PORT = "18764"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def stamp(): return datetime.datetime.now().astimezone().isoformat()
def journal(message):
    with (STATE / "repair_journal.md").open("a") as out: out.write("\n" + stamp() + ": " + message + "\n")

def main():
    if BASE.exists(): raise RuntimeError("v8 root already exists; duplicate refused")
    clients = active_runs()
    if any(str(path).startswith(str(BASE)) for path in clients): raise RuntimeError("v8 already active")
    BASE.mkdir()
    shutil.copy2(PRIOR / "case_manifest.json", BASE / "case_manifest.json")
    sources = {
        "scripts/flashrsi_harnessrisk_host_native_v8.py": sha(ROOT / "scripts" / HOST),
        "scripts/flashrsi_harnessrisk_host_native_v7.py": sha(ROOT / "scripts/flashrsi_harnessrisk_host_native_v7.py"),
        "scripts/flashrsi_harnessrisk_supervisor_events_tools_v7.py": sha(ROOT / "scripts/flashrsi_harnessrisk_supervisor_events_tools_v7.py"),
        "scripts/flashrsi_harnessrisk_claude_entry_events_tools_v4.py": sha(ROOT / "scripts/flashrsi_harnessrisk_claude_entry_events_tools_v4.py"),
        "scripts/flashrsi_harnessrisk_sandbox.py": sha(ROOT / "scripts/flashrsi_harnessrisk_sandbox.py"),
    }
    protocol = {"at": stamp(), "job_id": "claude_code/opro", "protocol": "private-v1-claude-cli-projection-v8-gpu1",
        "model_port": int(PORT), "host_script": HOST, "sources_sha256": sources,
        "predecessor": str(PRIOR), "old_error": "v7 retained 43/72 development episodes with 16-turn exhaustion after CLI repeated git/attribution reminders and the owner prompt on every tool round",
        "root_cause": "Claude Code 2.1.319 runtime-generated reminder/prompt duplication in Messages history, while tool_result blocks remain valid",
        "changes": ["lossless raw request evidence", "model projection removes only later exact CLI git/attribution reminder blocks and exact duplicate owner prompt", "all tool_result blocks and initial owner message retained"],
        "invariants": ["harnessrisk-private-case-v1", "same 128 manifest", "strict unscorable rejection", "16 target turns", "4096 target tokens", "one evolution attempt", "no retries", "no judge before exact accepted 128"],
        "pairing_limitation": "new message projection requires this same-v8 initial full-128 control before any changed-selection claim"}
    save(BASE / "repair_protocol.json", protocol)
    env = {**os.environ, "PYTHONPATH": "src:scripts:vendor/FlashRSI", "HR_HARNESS": "claude_code", "HARNESSRISK_MODEL_PORT": PORT,
           "HR_OUTPUT_ROOT": str(BASE), "HR_RUN_NAME": "claude_code_initial_full", "HR_HOST_SCRIPT": HOST}
    log = (BASE / "claude_code_initial_full.log").open("x")
    child = subprocess.Popen([sys.executable, "-u", "scripts/flashrsi_harnessrisk_miniagent_initial.py"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    save(BASE / "launch_state.json", {"at": stamp(), "status": "running_initial_control", "pid": child.pid, "protocol": protocol["protocol"], "endpoint": PORT})
    journal(f"Claude Code v8 CLI-projection same-protocol initial full128 control launched PID {child.pid} in {BASE}; first actual evidence pending; no evolution or judge starts before control acceptance.")
    return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    finally: scan()
