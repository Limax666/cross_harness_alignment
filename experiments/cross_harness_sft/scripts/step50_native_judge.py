"""Score completed local paired trajectories, independently from the model process."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import flashrsi_harnessrisk_canonical_judge_v2 as judge

MODEL = "openai/gpt-5.4-nano"


def load(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def score(root, done, args, data):
    case = done.parent
    group,arm,case_id=case.relative_to(root).parts
    output=root/"judge"/group/arm
    receipt=output/"results"/(case_id+".json")
    request=output/"requests"/(case_id+".json")
    identity=load(done)
    if identity["case_id"]!=case_id or identity["group"]!=group or identity["arm"]!=arm:
        raise ValueError("Case identity mismatch")
    if identity["run_manifest_sha256"]!=hashlib.sha256((root/"run_manifest.json").read_bytes()).hexdigest():
        raise ValueError("Protocol mismatch")
    if not load(case/"privacy_audit.json")["passed"] or load(case/"official_rule_evaluation.json")["case_id"]!=case_id:
        raise ValueError("Privacy or rule evidence missing")
    manifest = load(root / "run_manifest.json")
    harness = manifest["harness"]
    if harness not in {"hermes", "opencode"} or identity["harness"] != harness or manifest["protocol"] != "flashrsi-native-case-documents-v1":
        raise ValueError("Unknown harness or protocol")
    source = data / (case_id + ".json")
    if hashlib.sha256(source.read_bytes()).hexdigest() != identity["case_sha256"]:
        raise ValueError("Judge case changed")
    if receipt.exists():
        verify_receipt(receipt, identity, request)
        return
    if request.exists():
        return  # An interrupted request needs reconciliation with the provider before another charge.
    evidence=judge.build_evidence_bundle(case,data,args)
    # The judge only needs case and execution evidence. Local path and pairing labels stay in the receipt.
    evidence["case_source"]="frozen HarnessRisk case"
    evidence["harness"]={"name":harness}
    for field in ("tool_events","conversation"):
        for row in evidence.get(field,[]):
            row.pop("source_file",None)
    if evidence["run_status"]!="completed" or not evidence["conversation"] or evidence["case_id"]!=case_id:
        raise ValueError("Missing completed trajectory")
    sig=hashlib.sha256(json.dumps(evidence,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    save(output/"evidence"/(case_id+".json"),evidence)
    if judge.billing_block_state():
        raise RuntimeError("Existing judge provider billing hold is active")
    save(request,{"case_id":case_id,"evidence_sha256":sig,"time":timestamp(),"model":MODEL})
    try:
        result=judge.call_llm_judge(evidence,args)
        if result["case_id"]!=case_id or result["_llm_metadata"]["evidence_sha256"]!=sig:
            raise ValueError("Judge returned a different case or evidence")
        save(receipt,{"identity":identity,"judge":result,"time":timestamp()})
    except Exception as exc:
        if judge.is_billing_error(exc):
            judge.mark_billing_block(case_id)
        save(output/"errors"/(case_id+".json"),{"time":timestamp(),"error_type":type(exc).__name__,
            "http_status":getattr(exc,"status_code",None),"evidence_sha256":sig})
        raise
    rows=[{"case_id":p.stem,"status":"judged","file":p.name} for p in sorted(receipt.parent.glob("*.json"))]
    save(output/"manifest.json",rows)


def verify_receipt(path, identity, request):
    row = load(path)
    result = row["judge"]
    metadata = result["_llm_metadata"]
    sent = load(request)
    if (row["identity"] != identity or result["case_id"] != identity["case_id"]
            or metadata["model"] != MODEL or sent["case_id"] != identity["case_id"]
            or metadata["evidence_sha256"] != sent["evidence_sha256"]
            or any(result[k] not in (0, 1) for k in judge.METRIC_KEYS)):
        raise ValueError(f"Judge receipt mismatch: {path}")
    return result


def summary(root, data):
    run = load(root / "run_manifest.json")
    signature = hashlib.sha256((root / "run_manifest.json").read_bytes()).hexdigest()
    records = {}
    for group, spec in run["snapshots"].items():
        if spec["canonical_group"] != group:
            records[group] = {"same_snapshot_as": spec["canonical_group"]}
            continue
        rule, judged = {}, {}
        for arm in ("base", "step50"):
            rule[arm], judged[arm] = {}, {}
            for done in (root / group / arm).glob("*/complete.json"):
                cid, identity = done.parent.name, load(done)
                src = data / (cid + ".json")
                if (identity["group"], identity["arm"], identity["case_id"], identity["snapshot_id"],
                    identity["harness"], identity["run_manifest_sha256"], identity["case_sha256"]) != (
                        group, arm, cid, spec["id"], run["harness"], signature, hashlib.sha256(src.read_bytes()).hexdigest()):
                    raise ValueError("Completion identity mismatch")
                if not load(done.parent / "privacy_audit.json")["passed"]:
                    raise ValueError("Missing privacy pass")
                result = load(done.parent / "official_rule_evaluation.json")
                if result["case_id"] != cid or result["status"] != "completed":
                    raise ValueError("Wrong rule receipt")
                rule[arm][cid] = result["metrics"]
                base = root / "judge" / group / arm
                path = base / "results" / (cid + ".json")
                if path.is_file():
                    judged[arm][cid] = verify_receipt(path, identity, base / "requests" / path.name)
        matched = sorted(judged["base"].keys() & judged["step50"].keys())
        records[group] = {"valid_rule_arms": {arm: len(v) for arm, v in rule.items()},
            "judge_arms": {arm: len(v) for arm, v in judged.items()}, "paired_judge_cases": len(matched),
            "expected_pairs": 128, "paired_means": {
                arm: {k: sum(judged[arm][i][k] for i in matched) / len(matched) for k in judge.METRIC_KEYS}
                for arm in ("base", "step50")} if matched else None}
    return records


def main(root, data):
    if not (root / "run_manifest.json").is_file():
        raise ValueError("Await the producer run manifest")
    lock=(root/"judge.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    key=os.environ.get("SHENGSUANYUN_API_KEY")
    if not key:
        raise RuntimeError("Configured judge credential missing")
    args=SimpleNamespace(api_key=key,api_key_env="SHENGSUANYUN_API_KEY",model=MODEL,deployment=MODEL,
        base_url="https://router.shengsuanyun.com/api/v1",max_completion_tokens=2048,max_retries=0,
        response_format_json=False,request_timeout=120,max_message_chars=4000,max_tool_chars=3000,
        max_diff_content_chars=6000,max_final_answer_chars=8000,max_mock_service_chars=20000,include_reasoning=False)
    while True:
        for done in sorted(root.glob("*/*/*/complete.json")):
            score(root,done,args,data)
        if (root/"rollouts_finished.json").exists():
            # Rescan after observing the producer's final atomic marker.
            for done in sorted(root.glob("*/*/*/complete.json")):
                score(root,done,args,data)
            break
        time.sleep(10)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--summary", action="store_true", help="Summarize existing pairs offline")
    options=parser.parse_args()
    root = options.output.resolve()
    data = options.bench.resolve() / "data/HarnessRisk"
    if options.summary:
        print(json.dumps(summary(root, data), indent=2))
    else:
        main(root, data)
