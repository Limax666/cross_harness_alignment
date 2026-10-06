#!/usr/bin/env python3
"""Reuse audited judge verdicts only for bitwise-equivalent judge evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


EVIDENCE_FIELDS = (
    "benchmark", "harness_name", "task_id", "source_family_id", "source_task_type",
    "source_task_labels", "primary_behavior_type", "risk_tags", "source_score_status",
    "official_score", "source_objective", "source_evaluation", "task_metadata",
    "harness_context", "tools", "messages", "metrics",
)


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def evidence(row: dict) -> str:
    return json.dumps({key: row.get(key) for key in EVIDENCE_FIELDS},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    target_results = args.target / "judge_results.jsonl"
    if target_results.exists():
        parser.error("Target judge cache already exists")
    source_manifest = json.loads((args.source / "preprocessing_manifest.json").read_text())
    target_manifest = json.loads((args.target / "preprocessing_manifest.json").read_text())
    if source_manifest.get("status") != "judged_complete" or target_manifest.get("status") != "inventory_only_not_trainable":
        parser.error("Source must be complete and target must be inventory-only")
    spec = importlib.util.spec_from_file_location("teacher_release", Path(__file__).with_name("build_teacher_sft_release.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    prompt_sha = hashlib.sha256(module.JUDGE_POLICY.encode()).hexdigest()
    if source_manifest.get("judge_prompt_sha256") != prompt_sha or source_manifest.get("judge_prompt_version") != module.JUDGE_PROMPT_VERSION:
        parser.error("Judge policy version or prompt changed; cache reuse prohibited")
    old = {r["candidate_sha256"]: r for r in rows(args.source / "candidates.jsonl")}
    latest = {}
    for row in rows(args.source / "judge_results.jsonl"):
        if row.get("model") == source_manifest["judge_model"] and row.get("prompt_version") == source_manifest["judge_prompt_version"]:
            if row.get("verdict", {}).get("call_status") != "failed":
                latest[row["candidate_sha256"]] = row
    reuse = []
    for candidate in rows(args.target / "candidates.jsonl"):
        fingerprint = candidate["candidate_sha256"]
        prior = old.get(fingerprint)
        if prior is None or fingerprint not in latest:
            continue
        if prior["split"] != candidate["split"] or evidence(prior) != evidence(candidate):
            raise ValueError(f"Candidate fingerprint matched but judge evidence/split changed: {fingerprint}")
        result = dict(latest[fingerprint])
        result["reused_from_release"] = str(args.source)
        result["cache_imported_at_utc"] = datetime.now(timezone.utc).isoformat()
        reuse.append(result)
    with target_results.open("x", encoding="utf-8") as stream:
        for result in reuse:
            stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {"source": str(args.source), "source_judge_prompt_sha256": prompt_sha,
              "target": str(args.target), "reused": len(reuse),
              "target_candidates": target_manifest["candidate_count"],
              "remaining_for_judge": target_manifest["candidate_count"] - len(reuse)}
    (args.target / "judge_cache_import.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
