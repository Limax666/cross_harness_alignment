"""Run one frozen OpenClaw algorithm under the v3 case-identity host."""
import argparse, datetime, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path

from flashrsi_matrix_repair_loop import ROOT, STATE, active_runs, inspect_run, save, scan

HOST_SCRIPT = "flashrsi_harnessrisk_host_openclaw_caseenv_v3.py"
ALGORITHMS = {"opro", "gepa", "evoprompt", "ace", "skillopt", "evo_harness"}


def stamp(): return datetime.datetime.now().astimezone().isoformat()


def journal(message):
    with (STATE / "repair_journal.md").open("a") as stream:
        stream.write("\n" + stamp() + ": " + message + "\n")


def full_run_environment(parent, output_root, port, algorithm):
    if algorithm not in ALGORITHMS: raise ValueError("unsupported OpenClaw algorithm")
    return {**parent, "PYTHONPATH": "src:scripts:vendor/FlashRSI", "HR_HARNESS": "openclaw",
            "HARNESSRISK_MODEL_PORT": str(port), "HR_STRICT_EVOLUTION": "1",
            "HR_EVOLUTION_CASE_ATTEMPTS": "1", "HR_WORKER_MAX_TOKENS": "1024",
            "HR_OUTPUT_ROOT": str(output_root), "HR_HOST_SCRIPT": HOST_SCRIPT}


def live_clients(port):
    clients = []
    for path, pid in active_runs().items():
        try:
            env = dict(item.split(b"=", 1) for item in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0") if b"=" in item)
            clients.append({"pid": pid, "path": path, "port": env.get(b"HARNESSRISK_MODEL_PORT", b"18761").decode()})
        except OSError:
            continue
    return [item for item in clients if item["port"] == port]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    base = parser.parse_args().base.resolve()
    protocol = json.loads((base / "repair_protocol.json").read_text())
    job_id = protocol.get("job_id", "")
    if not job_id.startswith("openclaw/"): raise ValueError("v4 launcher is OpenClaw-only")
    algorithm = job_id.split("/", 1)[1]
    if algorithm not in ALGORITHMS: raise ValueError("unregistered OpenClaw algorithm")
    host = ROOT / "scripts" / HOST_SCRIPT
    relative_host = str(host.relative_to(ROOT))
    if not host.is_file() or hashlib.sha256(host.read_bytes()).hexdigest() != protocol["sources_sha256"][relative_host]:
        raise RuntimeError("registered case-identity host missing or source changed")
    port = str(protocol["model_port"])
    if port not in ("18761", "18762"): raise ValueError("unregistered endpoint")
    run = base / f"openclaw_{algorithm}_full"
    with (base / "launch.lock").open("a") as job_lock:
        try: fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return 75
        if run.exists(): raise RuntimeError("existing run preserved; duplicate refused")
        try:
            child = None
            while child is None:
                with (STATE / ("endpoint_" + port + ".lock")).open("a") as endpoint_lock:
                    fcntl.flock(endpoint_lock, fcntl.LOCK_EX)
                    clients = live_clients(port)
                    if sum(1 for _ in clients) < 2:
                        for relative, expected in protocol["sources_sha256"].items():
                            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                                raise RuntimeError("frozen source drift: " + relative)
                        initial = base / "openclaw_initial_full" / "initial_snapshot.json"
                        if hashlib.sha256(initial.read_bytes()).hexdigest() != protocol["initial_snapshot_file_sha256"]:
                            raise RuntimeError("initial Snapshot source changed")
                        log = (base / f"openclaw_{algorithm}_full.log").open("w")
                        child = subprocess.Popen([sys.executable, "-u", "scripts/flashrsi_harnessrisk_algorithm.py", "--algorithm", algorithm], cwd=ROOT,
                            env=full_run_environment(os.environ, base, port, algorithm), stdout=log, stderr=subprocess.STDOUT)
                        save(base / "launch_state.json", {"at": stamp(), "status": "running", "pid": child.pid,
                            "launcher_pid": os.getpid(), "active_before_launch": clients})
                        journal(f"OpenClaw {algorithm} case-environment v4 full evolution launched PID {child.pid} in {run}; first real evidence pending.")
                    else:
                        save(base / "launch_state.json", {"at": stamp(), "status": "waiting_endpoint_capacity", "pid": None,
                            "launcher_pid": os.getpid(), "clients": clients})
                if child is None: time.sleep(30)
            code = child.wait(); log.close()
            expected_ids = {item["case_id"] for item in json.loads((base / "case_manifest.json").read_text())["cases"]}
            before = inspect_run(run, expected_ids, None); save(base / "acceptance_before_judge.json", before)
            if code or before["status"] not in ("needs_judge", "complete"):
                save(base / "launch_state.json", {"at": stamp(), "status": "failed_full_gate", "pid": child.pid, "exit_code": code, "acceptance": before})
                journal(f"OpenClaw {algorithm} case-environment v4 full gate failed, exit {code}; preserve evidence; no invalid-case judge.")
                return code or 2
            if before["status"] == "needs_judge":
                with (base / f"openclaw_{algorithm}_judge.log").open("w") as judge_log:
                    code = subprocess.call([sys.executable, "scripts/judge_harnessrisk_shengsuanyun.py", str(run / "selected_cases"), "--deployment", "openai/gpt-5.4-nano"], cwd=ROOT, stdout=judge_log, stderr=subprocess.STDOUT)
            final = inspect_run(run, expected_ids, None)
            comparison = "same-caseenv-v4 full128 initial control required before changed-selection claim"
            save(base / "launch_state.json", {"at": stamp(), "status": final["status"], "pid": child.pid, "exit_code": code, "acceptance": final, "comparison": comparison})
            journal(f"OpenClaw {algorithm} case-environment v4 scanner acceptance {final['status']}; {comparison}.")
            return 0 if final["status"] == "complete" else code or 3
        finally:
            scan()


if __name__ == "__main__": raise SystemExit(main())
