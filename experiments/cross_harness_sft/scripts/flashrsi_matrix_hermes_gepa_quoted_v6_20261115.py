"""Freeze and run Hermes GEPA v6 with quoted-evidence reflection framing."""
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from flashrsi_matrix_repair_loop import ROOT, STATE, active_runs, inspect_run, save, scan


def stamp(): return datetime.datetime.now().astimezone().isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def journal(text):
    with (STATE / "repair_journal.md").open("a") as stream: stream.write("\n" + stamp() + ": " + text + "\n")


def clients(port):
    found=[]
    for path,pid in active_runs().items():
        try:
            env=dict(item.split(b"=",1) for item in (Path("/proc")/str(pid)/"environ").read_bytes().split(b"\0") if b"=" in item)
            if env.get(b"HARNESSRISK_MODEL_PORT",b"18761").decode()==port: found.append({"pid":pid,"path":path})
        except OSError: pass
    return found


def main():
    prior=ROOT/"outputs/flashrsi_harnessrisk_4b_private_hermes_gepa_contextfit_v5_20261114"
    base=ROOT/"outputs/flashrsi_harnessrisk_4b_private_hermes_gepa_quoted_v7_20261115"
    run=base/"hermes_gepa_full"
    if base.exists(): raise FileExistsError(base)
    sources={name:sha(ROOT/name) for name in (
        "scripts/flashrsi_matrix_hermes_gepa_quoted_v6_20261115.py",
        "scripts/flashrsi_harnessrisk_algorithm.py",
        "scripts/flashrsi_hermes_gepa_reflection_v6.py",
        "scripts/flashrsi_harnessrisk_host_contextfit_v5.py",
        "scripts/flashrsi_harnessrisk_supervisor.py",
        "scripts/flashrsi_harnessrisk_benchmark.py",
    )}
    base.mkdir(); shutil.copy2(prior/"case_manifest.json",base/"case_manifest.json")
    # The v5 paired_cases tree contains historical evidence symlinks; it is not
    # baseline input and dereferencing it made the v6 prelaunch copy fail after
    # partially creating its output directory. Freeze only the three baseline
    # files the algorithm reads.
    initial_dir=base/"hermes_initial_full"; initial_dir.mkdir()
    for name in ("initial_snapshot.json", "completed.json", "run_manifest.json"):
        shutil.copy2(prior/"hermes_initial_full"/name, initial_dir/name)
    initial=base/"hermes_initial_full/initial_snapshot.json"
    protocol={"at":stamp(),"job_id":"hermes/gepa","protocol":"private-v1-hermes-gepa-v7-quoted-evidence","model_port":18761,
      "sources_sha256":sources,"initial_snapshot_file_sha256":sha(initial),"predecessor":str(prior),
      "old_error":"v5 worker trajectory 00024 ended finish_reason=length after it replayed quoted Hermes system text at the fixed 1024-worker-token limit; v6 prelaunch copy then failed while dereferencing historical paired_cases evidence",
      "root_cause":"GEPA reflection evidence embeds system/user/tool transcript as unmarked prompt content, allowing the worker to treat quoted data as instructions and exhaust its reply budget",
      "changes":["Hermes-only GEPA subclass marks all reflection evidence quoted data","requires a single fenced replacement instruction and forbids quoted-data replay"],
      "invariants":["harnessrisk-private-case-v1","strict evolution rejects unscorable trajectories","one evolution attempt","12 target turns","1024 worker tokens","no evolution or final-case retries","no judge before accepted exact 128"]}
    save(base/"repair_protocol.json",protocol)
    for relative,digest in sources.items():
        target=base/"source_snapshot"/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/relative,target)
        if sha(target)!=digest: raise RuntimeError("source snapshot mismatch: "+relative)
    with (base/"launch.lock").open("x") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        child=None
        while child is None:
            with (STATE/"endpoint_18761.lock").open("a") as endpoint:
                fcntl.flock(endpoint,fcntl.LOCK_EX)
                active=clients("18761")
                if len(active)<2:
                    for relative,digest in sources.items():
                        if sha(ROOT/relative)!=digest: raise RuntimeError("frozen source drift: "+relative)
                    env={**os.environ,"PYTHONPATH":"src:scripts:vendor/FlashRSI","HR_HARNESS":"hermes","HARNESSRISK_MODEL_PORT":"18761","HR_STRICT_EVOLUTION":"1","HR_EVOLUTION_CASE_ATTEMPTS":"1","HR_WORKER_MAX_TOKENS":"1024","HR_OUTPUT_ROOT":str(base),"HR_HOST_SCRIPT":"flashrsi_harnessrisk_host_contextfit_v5.py","HR_GEPA_REFLECTION_PROTOCOL":"hermes-v6-quoted-evidence"}
                    log=(base/"hermes_gepa_full.log").open("x")
                    child=subprocess.Popen([sys.executable,"-u","scripts/flashrsi_harnessrisk_algorithm.py","--algorithm","gepa"],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    save(base/"launch_state.json",{"at":stamp(),"status":"running","launcher_pid":os.getpid(),"pid":child.pid,"active_before_launch":active})
                else: save(base/"launch_state.json",{"at":stamp(),"status":"waiting_endpoint_capacity","launcher_pid":os.getpid(),"clients":active})
            if child is None: time.sleep(30)
        code=child.wait(); log.close(); expected={x["case_id"] for x in json.loads((base/"case_manifest.json").read_text())["cases"]}; accepted=inspect_run(run,expected,None); save(base/"acceptance_before_judge.json",accepted)
        if code or accepted["status"] not in ("needs_judge","complete"):
            save(base/"launch_state.json",{"at":stamp(),"status":"failed_full_gate","pid":child.pid,"exit_code":code,"acceptance":accepted}); journal("Hermes GEPA quoted-evidence v6 full gate failed; raw evidence preserved and no invalid-case judge started."); return code or 2
        if accepted["status"]=="needs_judge":
            with (base/"hermes_gepa_judge.log").open("x") as out: code=subprocess.call([sys.executable,"scripts/judge_harnessrisk_shengsuanyun.py",str(run/"selected_cases"),"--deployment","openai/gpt-5.4-nano"],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
        final=inspect_run(run,expected,None); save(base/"launch_state.json",{"at":stamp(),"status":final["status"],"pid":child.pid,"exit_code":code,"acceptance":final}); journal("Hermes GEPA quoted-evidence v6 scanner status "+final["status"]+"; same-v6 B0 is required before any gain claim.")
    scan()


if __name__=="__main__": main()
