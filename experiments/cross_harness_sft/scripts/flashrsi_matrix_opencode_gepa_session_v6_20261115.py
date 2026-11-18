"""Freeze and start the OpenCode GEPA v6 full protocol without mutating v5."""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from flashrsi_matrix_repair_loop import ROOT, STATE, save


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def main():
    prior = ROOT / "outputs/flashrsi_harnessrisk_4b_private_opencode_gepa_session_v5_20261115"
    base = ROOT / "outputs/flashrsi_harnessrisk_4b_private_opencode_gepa_session_v6_20261115"
    if base.exists():
        raise FileExistsError(base)
    old = json.loads((prior / "repair_protocol.json").read_text())
    sources = dict(old["sources_sha256"])
    for name in (
        "scripts/flashrsi_matrix_opencode_session_v5_20261115.py",
        "scripts/flashrsi_harnessrisk_host_opencode_session_v5.py",
        "scripts/flashrsi_harnessrisk_opencode_session_supervisor_v5.py",
    ):
        sources.pop(name, None)
    for name in (
        "scripts/flashrsi_matrix_opencode_session_v6_20261115.py",
        "scripts/flashrsi_harnessrisk_host_opencode_session_v6.py",
        "scripts/flashrsi_harnessrisk_opencode_session_supervisor_v6.py",
        "scripts/flashrsi_matrix_opencode_gepa_session_v6_20261115.py",
    ):
        sources[name] = sha(ROOT / name)
    for relative, expected in sources.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("source drift before v6 freeze: " + relative)
    base.mkdir()
    for name in ("case_manifest.json",):
        shutil.copy2(prior / name, base / name)
    shutil.copytree(prior / "opencode_initial_full", base / "opencode_initial_full")
    initial = base / "opencode_initial_full" / "initial_snapshot.json"
    protocol = {
        "at": stamp(), "job_id": "opencode/gepa",
        "protocol": "private-v1-opencode-native-session-v6-compaction",
        "model_port": 18762,
        "queue_launcher": "scripts/flashrsi_matrix_opencode_session_v6_20261115.py",
        "sources_sha256": sources,
        "initial_snapshot_file_sha256": sha(initial),
        "predecessor": str(prior),
        "old_error": "v5 actual compaction evidence retained stale v4 protocol labels; stopped before scoring or judge",
        "scope": "real full GEPA evolution then selected full 128; strict scanner gate then canonical missing-only judge",
        "comparison": "v6 changes protocol metadata consistency only; v6 needs a same-v6 full initial control before a changed-snapshot gain claim. Prior baselines stay separate.",
        "invariants": ["harnessrisk-private-case-v1", "strict evolution rejects unscorable trajectories", "one evolution attempt", "12 turns", "no final-case retry", "no judge before accepted exact 128"],
    }
    save(base / "repair_protocol.json", protocol)
    save(base / "diagnosis_reference.json", {
        "predecessor": str(prior), "old_error_source": str(prior / "operator_stop.json"),
        "old_first_case": str(prior / "opencode_gepa_full/evidence/00000001/cases/daily_025/session_protocol_evidence.json"),
        "root_cause": "v5 host/supervisor retained hard-coded v4 evidence labels despite v5 compaction; audit label and execution protocol diverged.",
        "classification": "engineering protocol-metadata defect; not a score, attack-success, judge, privacy, or sandbox failure",
    })
    for relative, expected in sources.items():
        target = base / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        if sha(target) != expected:
            raise RuntimeError("source snapshot mismatch: " + relative)
    argv = [sys.executable, "-u", protocol["queue_launcher"], "--base", str(base)]
    with (base / "launcher.log").open("x") as log:
        child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    save(base / "launch_state.json", {"at": stamp(), "status": "launcher_started_capacity_gate_pending",
                                      "launcher_pid": child.pid, "pid": None, "argv": argv})
    action = {"at": stamp(), "job_id": "opencode/gepa",
              "root_cause": "v5 deterministic compaction was real but two hard-coded v4 metadata labels made the first evidence internally inconsistent.",
              "changes": ["new v6 immutable host, supervisor and launcher consistently label all protocol evidence v6",
                          "new v6 source snapshot and protocol freeze; v5 is preserved unchanged"],
              "run_dir": str(base / "opencode_gepa_full"), "pid": child.pid,
              "outcome": "real full v6 launcher started; endpoint capacity gate decides inference start",
              "next_action": "verify first real v6 evidence labels/privacy/official rule result, then require full exact-128 scanner acceptance before exactly one matching judge; no v6 gain claim without same-v6 B0."}
    save(STATE / "action_last.json", action)
    with (STATE / "repair_journal.md").open("a") as stream:
        stream.write("\n## " + stamp() + " — OpenCode GEPA v6 metadata-consistency full protocol\n\n")
        stream.write("Preserved v5 first evidence and operator stop: deterministic compaction occurred, but hard-coded v4 labels in host/supervisor made audit metadata inconsistent. This is an engineering protocol defect, not zero score, normal model budget result, attack success, judge, privacy, or sandbox failure. v6 replaces only those labels while retaining v5 audited compaction, strict evolution, case set, budgets, retries and gates. Frozen source hashes, predecessor and diagnosis are in " + str(base / "repair_protocol.json") + ".\n\n")
        stream.write("Started one real full v6 GEPA launcher PID " + str(child.pid) + " on port 18762 under its endpoint lock. It may wait for capacity; no model inference, score or judge is claimed until actual evidence exists. The v5 run remains invalid/preserved and is not reused. v6 requires same-v6 full B0 before a changed-snapshot gain claim.\n")
    print(json.dumps({"base": str(base), "launcher_pid": child.pid, "sources": len(sources)}))


if __name__ == "__main__":
    main()
