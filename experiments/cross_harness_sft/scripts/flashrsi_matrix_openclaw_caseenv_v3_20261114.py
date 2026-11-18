"""Serialized real OpenClaw full-128 execution with case-environment v3."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from flashrsi_matrix_repair_loop import ROOT, STATE, active_runs, inspect_run, save, scan


HOST_SCRIPT = "flashrsi_harnessrisk_host_openclaw_caseenv_v3.py"


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def journal(message):
    with (STATE / "repair_journal.md").open("a") as stream:
        stream.write("\n" + stamp() + ": " + message + "\n")


def full_run_environment(parent, output_root, port):
    return {**parent, "PYTHONPATH": "src:scripts:vendor/FlashRSI", "HR_HARNESS": "openclaw",
            "HARNESSRISK_MODEL_PORT": str(port), "HR_STRICT_EVOLUTION": "1",
            "HR_EVOLUTION_CASE_ATTEMPTS": "1", "HR_WORKER_MAX_TOKENS": "1024",
            "HR_OUTPUT_ROOT": str(output_root), "HR_HOST_SCRIPT": HOST_SCRIPT}


def clients_and_queue_reservations():
    clients = []
    for path, pid in active_runs().items():
        try:
            environment = dict(item.split(b"=", 1) for item in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0") if b"=" in item)
            clients.append({"pid": pid, "path": path, "port": environment.get(b"HARNESSRISK_MODEL_PORT", b"18761").decode()})
        except OSError:
            continue
    for process in Path("/proc").glob("[0-9]*"):
        try:
            arguments = (process / "cmdline").read_bytes().split(b"\0")
            names = [Path(item.decode(errors="replace")).name for item in arguments if item]
            if not any(name in ("flashrsi_harnessrisk_private_queue.sh", "flashrsi_harnessrisk_miniagent_initial.py") for name in names):
                continue
            environment = dict(item.split(b"=", 1) for item in (process / "environ").read_bytes().split(b"\0") if b"=" in item)
            base = environment.get(b"HR_OUTPUT_ROOT", b"").decode()
            if base and not any(Path(client["path"]).parent == Path(base) for client in clients):
                clients.append({"pid": int(process.name), "path": str(Path(base) / "queue_slot_reservation"),
                                "port": environment.get(b"HARNESSRISK_MODEL_PORT", b"18761").decode()})
        except OSError:
            continue
    return clients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    base = parser.parse_args().base.resolve()
    protocol = json.loads((base / "repair_protocol.json").read_text())
    algorithm = protocol["job_id"].split("/")[1]
    if algorithm != "gepa" or protocol["job_id"] != "openclaw/gepa":
        raise ValueError("caseenv-v3 launcher is restricted to openclaw/gepa")
    host = ROOT / "scripts" / HOST_SCRIPT
    relative_host = str(host.relative_to(ROOT))
    if not host.is_file() or hashlib.sha256(host.read_bytes()).hexdigest() != protocol["sources_sha256"][relative_host]:
        raise RuntimeError("registered v3 host missing or source changed")
    port = str(protocol["model_port"])
    if port not in ("18761", "18762"):
        raise ValueError("unregistered endpoint")
    run = base / "openclaw_gepa_full"
    with (base / "launch.lock").open("a") as job_lock:
        try:
            fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        if run.exists():
            raise RuntimeError("existing run preserved; duplicate refused")
        try:
            child = None
            while child is None:
                with (STATE / ("endpoint_" + port + ".lock")).open("a") as endpoint_lock:
                    fcntl.flock(endpoint_lock, fcntl.LOCK_EX)
                    clients = clients_and_queue_reservations()
                    if any(Path(client["path"]).name == run.name for client in clients):
                        raise RuntimeError("same harness/algorithm already active")
                    if sum(client["port"] == port for client in clients) < 2:
                        for relative, expected in protocol["sources_sha256"].items():
                            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                                raise RuntimeError("frozen source drift: " + relative)
                        initial = base / "openclaw_initial_full" / "initial_snapshot.json"
                        if hashlib.sha256(initial.read_bytes()).hexdigest() != protocol["initial_snapshot_file_sha256"]:
                            raise RuntimeError("initial Snapshot source changed")
                        log = (base / "openclaw_gepa_full.log").open("w")
                        child = subprocess.Popen([sys.executable, "-u", "scripts/flashrsi_harnessrisk_algorithm.py", "--algorithm", "gepa"], cwd=ROOT,
                                                 env=full_run_environment(os.environ, base, port), stdout=log, stderr=subprocess.STDOUT)
                        save(base / "launch_state.json", {"at": stamp(), "status": "running", "pid": child.pid,
                                                          "launcher_pid": os.getpid(), "active_before_launch": clients})
                        journal(f"OpenClaw gepa case-environment v3 full evolution launched PID {child.pid} in {run}; first real evidence pending; original B0 referenced only.")
                    else:
                        save(base / "launch_state.json", {"at": stamp(), "status": "waiting_endpoint_capacity", "pid": None,
                                                          "launcher_pid": os.getpid(), "clients": clients})
                if child is None:
                    time.sleep(30)
            code = child.wait()
            log.close()
            expected_ids = {item["case_id"] for item in json.loads((base / "case_manifest.json").read_text())["cases"]}
            before = inspect_run(run, expected_ids, None)
            save(base / "acceptance_before_judge.json", before)
            if code or before["status"] not in ("needs_judge", "complete"):
                save(base / "launch_state.json", {"at": stamp(), "status": "failed_full_gate", "pid": child.pid, "exit_code": code, "acceptance": before})
                journal(f"OpenClaw gepa case-environment v3 full gate failed, exit {code}; preserve raw evidence and diagnose; no invalid-case judge.")
                return code or 2
            if before["status"] == "needs_judge":
                with (base / "openclaw_gepa_judge.log").open("w") as judge_log:
                    code = subprocess.call([sys.executable, "scripts/judge_harnessrisk_shengsuanyun.py", str(run / "selected_cases"), "--deployment", "openai/gpt-5.4-nano"], cwd=ROOT, stdout=judge_log, stderr=subprocess.STDOUT)
            final = inspect_run(run, expected_ids, None)
            result = json.loads((run / "result.json").read_text())
            comparison = "same-caseenv-v3 full128 initial control required before gain claim" if result["selected_changed"] else "selected identity equals initial; no gain claim; old-v2 scores remain separate"
            save(base / "launch_state.json", {"at": stamp(), "status": final["status"], "pid": child.pid, "exit_code": code, "acceptance": final, "comparison": comparison})
            journal(f"OpenClaw gepa case-environment v3 scanner acceptance {final['status']}; {comparison}; judge reuses existing artifacts.")
            return 0 if final["status"] == "complete" else code or 3
        except Exception as error:
            save(base / "launch_state.json", {"at": stamp(), "status": "launcher_failed", "launcher_pid": os.getpid(), "error": type(error).__name__ + ": " + str(error)})
            raise
        finally:
            scan()


if __name__ == "__main__":
    raise SystemExit(main())
