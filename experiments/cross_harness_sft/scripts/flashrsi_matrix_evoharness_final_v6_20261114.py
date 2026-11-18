"""Run real v6 Evo-Harness after one of GPU3's two client slots is free."""
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from flashrsi_matrix_repair_loop import STATE, ROOT, active_runs, inspect_run, save, scan

BASE = ROOT / "outputs/flashrsi_harnessrisk_4b_private_final_transcript_v6_20261114"
RUN = BASE / "miniagent_evo_harness_full"
PORT = "18762"


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def journal(message):
    with (STATE / "repair_journal.md").open("a") as stream:
        stream.write("\n" + stamp() + ": " + message + "\n")


def active_clients():
    clients = []
    for path, pid in active_runs().items():
        try:
            env = dict(x.split(b"=", 1) for x in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0") if b"=" in x)
            clients.append({"pid": pid, "path": path,
                            "port": env.get(b"HARNESSRISK_MODEL_PORT", b"18761").decode()})
        except FileNotFoundError:
            continue
    return clients


def main():
    with (BASE / "launch.lock").open("a") as job_lock:
        try:
            fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        if RUN.exists():
            raise RuntimeError("existing run preserved; refusing duplicate execution")
        protocol = json.loads((BASE / "repair_protocol.json").read_text())
        started = False
        try:
            while not started:
                # Atomic check/launch for repair launchers using this endpoint.
                with (STATE / "endpoint_18762.lock").open("a") as endpoint_lock:
                    fcntl.flock(endpoint_lock, fcntl.LOCK_EX)
                    clients = active_clients()
                    if any(Path(x["path"]).name == "miniagent_evo_harness_full" for x in clients):
                        raise RuntimeError("another Evo-Harness run is active; re-diagnose instead of duplicating")
                    if sum(x["port"] == PORT for x in clients) < 2:
                        for relative, expected in protocol["sources_sha256"].items():
                            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                                raise RuntimeError("frozen source drift: " + relative)
                        env = {**os.environ, "PYTHONPATH": "src:scripts:vendor/FlashRSI", "HR_HARNESS": "miniagent",
                               "HARNESSRISK_MODEL_PORT": PORT, "HR_STRICT_EVOLUTION": "1", "HR_EVOLUTION_CASE_ATTEMPTS": "1",
                               "HR_WORKER_MAX_TOKENS": "4096", "HR_OUTPUT_ROOT": str(BASE),
                               "HR_HOST_SCRIPT": "flashrsi_harnessrisk_host_final_transcript_v6.py"}
                        log = (BASE / "miniagent_evo_harness_full.log").open("w")
                        child = subprocess.Popen([sys.executable, "-u", "scripts/flashrsi_harnessrisk_algorithm.py",
                                                  "--algorithm", "evo_harness"], cwd=ROOT, env=env,
                                                 stdout=log, stderr=subprocess.STDOUT)
                        save(BASE / "launch_state.json", {"at": stamp(), "status": "running", "pid": child.pid,
                                                         "launcher_pid": os.getpid(), "active_before_launch": clients})
                        journal(f"v6 Evo-Harness real full evolution launched PID {child.pid} in {RUN}; final transcript framing is new protocol; first evidence pending.")
                        started = True
                    else:
                        save(BASE / "launch_state.json", {"at": stamp(), "status": "waiting_endpoint_capacity",
                                                         "pid": None, "launcher_pid": os.getpid(), "clients": clients})
                if not started:
                    time.sleep(30)
            code = child.wait()
            log.close()
            expected = {x["case_id"] for x in json.loads((BASE / "case_manifest.json").read_text())["cases"]}
            row = inspect_run(RUN, expected, None)
            save(BASE / "acceptance_before_judge.json", row)
            if code or row["status"] not in ("needs_judge", "complete"):
                save(BASE / "launch_state.json", {"at": stamp(), "status": "failed_full_gate", "pid": child.pid,
                                                 "exit_code": code, "acceptance": row})
                journal(f"v6 Evo-Harness full gate failed (exit {code}); preserved {RUN}; no invalid-case judge invoked. Return to new evidence diagnosis.")
                return code or 2
            if row["status"] == "needs_judge":
                with (BASE / "miniagent_evo_harness_judge.log").open("w") as judge_log:
                    code = subprocess.call([sys.executable, "scripts/judge_harnessrisk_shengsuanyun.py",
                                            str(RUN / "selected_cases"), "--deployment", "openai/gpt-5.4-nano"],
                                           cwd=ROOT, stdout=judge_log, stderr=subprocess.STDOUT)
            final = inspect_run(RUN, expected, None)
            save(BASE / "launch_state.json", {"at": stamp(), "status": final["status"], "pid": child.pid,
                                             "exit_code": code, "acceptance": final})
            journal(f"v6 Evo-Harness final scanner acceptance {final['status']} for {RUN}; judge reused existing files and did not overwrite.")
            return 0 if final["status"] == "complete" else code or 3
        except Exception as error:
            save(BASE / "launch_state.json", {"at": stamp(), "status": "launcher_failed", "launcher_pid": os.getpid(),
                                             "error": type(error).__name__ + ": " + str(error)})
            raise
        finally:
            scan()


if __name__ == "__main__":
    raise SystemExit(main())
