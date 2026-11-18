"""Serialized real OpenClaw evolution/full128/judge under environment-session v2."""
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


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def journal(message):
    with (STATE / "repair_journal.md").open("a") as stream:
        stream.write("\n" + stamp() + ": " + message + "\n")


def clients_and_queue_reservations():
    clients = []
    for path, pid in active_runs().items():
        try:
            env = dict(x.split(b"=", 1) for x in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0") if b"=" in x)
            clients.append({"pid": pid, "path": path, "port": env.get(b"HARNESSRISK_MODEL_PORT", b"18761").decode()})
        except OSError:
            continue
    # The old private queue advances without the endpoint lock. Reserve its
    # slot even between algorithm children, rather than race that transition.
    for process in Path("/proc").glob("[0-9]*"):
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
            if not any(Path(x.decode(errors="replace")).name in ("flashrsi_harnessrisk_private_queue.sh", "flashrsi_harnessrisk_miniagent_initial.py") for x in args if x):
                continue
            env = dict(x.split(b"=", 1) for x in (process / "environ").read_bytes().split(b"\0") if b"=" in x)
            base = env.get(b"HR_OUTPUT_ROOT", b"").decode()
            if base and not any(Path(x["path"]).parent == Path(base) for x in clients):
                clients.append({"pid": int(process.name), "path": str(Path(base) / "queue_slot_reservation"),
                                "port": env.get(b"HARNESSRISK_MODEL_PORT", b"18761").decode()})
        except OSError:
            continue
    return clients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    protocol = json.loads((base / "repair_protocol.json").read_text())
    algorithm = protocol["job_id"].split("/")[1]
    if algorithm not in ("gepa", "evoprompt", "opro", "ace") or protocol["job_id"] != "openclaw/" + algorithm:
        raise ValueError("unexpected registered job")
    host = ROOT / "scripts/flashrsi_harnessrisk_host_openclaw_environment_v2.py"
    if not host.is_file() or hashlib.sha256(host.read_bytes()).hexdigest() != protocol["sources_sha256"][str(host.relative_to(ROOT))]:
        raise RuntimeError("registered host missing or source changed")
    port = str(protocol["model_port"])
    if port not in ("18761", "18762"):
        raise ValueError("unregistered endpoint")
    run = base / f"openclaw_{algorithm}_full"
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
                    if any(Path(x["path"]).name == run.name for x in clients):
                        raise RuntimeError("same harness/algorithm already active")
                    if sum(x["port"] == port for x in clients) < 2:
                        for relative, expected in protocol["sources_sha256"].items():
                            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                                raise RuntimeError("frozen source drift: " + relative)
                        initial = base / "openclaw_initial_full/initial_snapshot.json"
                        if hashlib.sha256(initial.read_bytes()).hexdigest() != protocol["initial_snapshot_file_sha256"]:
                            raise RuntimeError("initial Snapshot source changed")
                        env = {**os.environ, "PYTHONPATH": "src:scripts:vendor/FlashRSI", "HR_HARNESS": "openclaw",
                               "HARNESSRISK_MODEL_PORT": port, "HR_STRICT_EVOLUTION": "1", "HR_EVOLUTION_CASE_ATTEMPTS": "1",
                               "HR_WORKER_MAX_TOKENS": "1024", "HR_OUTPUT_ROOT": str(base),
                               "HR_HOST_SCRIPT": "flashrsi_harnessrisk_host_openclaw_environment_v2.py"}
                        log = (base / f"openclaw_{algorithm}_full.log").open("w")
                        child = subprocess.Popen([sys.executable, "-u", "scripts/flashrsi_harnessrisk_algorithm.py", "--algorithm", algorithm],
                                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                        save(base / "launch_state.json", {"at": stamp(), "status": "running", "pid": child.pid,
                                                         "launcher_pid": os.getpid(), "active_before_launch": clients})
                        journal(f"OpenClaw {algorithm} environment-session v2 full evolution launched PID {child.pid} in {run}; first real evidence pending; original B0 referenced only.")
                    else:
                        save(base / "launch_state.json", {"at": stamp(), "status": "waiting_endpoint_capacity", "pid": None,
                                                         "launcher_pid": os.getpid(), "clients": clients})
                if child is None:
                    time.sleep(30)
            code = child.wait()
            log.close()
            expected_ids = {x["case_id"] for x in json.loads((base / "case_manifest.json").read_text())["cases"]}
            before = inspect_run(run, expected_ids, None)
            save(base / "acceptance_before_judge.json", before)
            if code or before["status"] not in ("needs_judge", "complete"):
                save(base / "launch_state.json", {"at": stamp(), "status": "failed_full_gate", "pid": child.pid,
                                                 "exit_code": code, "acceptance": before})
                journal(f"OpenClaw {algorithm} environment-session v2 full gate failed, exit {code}; preserve raw evidence and diagnose; no invalid-case judge.")
                return code or 2
            if before["status"] == "needs_judge":
                with (base / f"openclaw_{algorithm}_judge.log").open("w") as judge_log:
                    code = subprocess.call([sys.executable, "scripts/judge_harnessrisk_shengsuanyun.py", str(run / "selected_cases"),
                                            "--deployment", "openai/gpt-5.4-nano"], cwd=ROOT, stdout=judge_log, stderr=subprocess.STDOUT)
            final = inspect_run(run, expected_ids, None)
            result = json.loads((run / "result.json").read_text())
            comparison = "same-environment-session-v2 full128 initial control required before gain claim" if result["selected_changed"] else "selected identity equals initial; no gain claim; old-v1 scores remain separate"
            save(base / "launch_state.json", {"at": stamp(), "status": final["status"], "pid": child.pid,
                                             "exit_code": code, "acceptance": final, "comparison": comparison})
            journal(f"OpenClaw {algorithm} environment-session v2 scanner acceptance {final['status']}; {comparison}; judge reuses existing artifacts.")
            return 0 if final["status"] == "complete" else code or 3
        except Exception as error:
            save(base / "launch_state.json", {"at": stamp(), "status": "launcher_failed", "launcher_pid": os.getpid(),
                                             "error": type(error).__name__ + ": " + str(error)})
            raise
        finally:
            scan()


if __name__ == "__main__":
    raise SystemExit(main())
