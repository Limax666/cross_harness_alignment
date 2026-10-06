#!/usr/bin/env python3
"""Fail closed on split, provenance, and verdict inconsistencies in an SFT release."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def rows(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(release: Path, require_complete: bool) -> dict:
    manifest = json.loads((release / "preprocessing_manifest.json").read_text())
    errors = []
    family_file = release / "family_split_manifest.jsonl"
    taxonomy_file = release / "taxonomy_mapping.jsonl"
    if sha(family_file) != manifest["family_manifest_sha256"]:
        errors.append("family_manifest_hash_mismatch")
    if sha(taxonomy_file) != manifest["taxonomy_mapping_sha256"]:
        errors.append("taxonomy_mapping_hash_mismatch")
    families = {(r["benchmark_namespace"], r["source_family_id"]): r["split"] for r in rows(family_file)}
    candidates = list(rows(release / "candidates.jsonl"))
    ids = [r["candidate_sha256"] for r in candidates]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_candidate_fingerprint")
    if len(candidates) != manifest["candidate_count"]:
        errors.append("candidate_count_mismatch")
    for r in candidates:
        if families.get((r["benchmark"], r["source_family_id"])) != r["split"]:
            errors.append("candidate_family_split_mismatch")
        if r["split"] not in {"train", "validation"}:
            errors.append("test_or_review_family_is_candidate")
    verdicts = {}
    judge_model = manifest.get("judge_model")
    judge_prompt = manifest.get("judge_prompt_version")
    for r in rows(release / "judge_results.jsonl"):
        if (judge_model is None or r.get("model") == judge_model) and (judge_prompt is None or r.get("prompt_version") == judge_prompt):
            if r["verdict"].get("call_status") != "failed":
                verdicts[r["candidate_sha256"]] = r["verdict"]
    accepted = Counter()
    for split in ("train", "validation"):
        for r in rows(release / f"accepted_{split}.jsonl"):
            fingerprint = r["candidate_sha256"]
            verdict = verdicts.get(fingerprint)
            if fingerprint not in ids or verdict is None or verdict.get("decision") != "accept":
                errors.append("accepted_without_valid_judge_verdict")
            if r["split"] != split or families.get((r["benchmark"], r["source_family_id"])) != split:
                errors.append("accepted_split_mismatch")
            if verdict is None or any(verdict.get(score, -1) < 3 for score in ("safety", "utility", "tool_correctness", "harness_fidelity")):
                errors.append("accepted_score_below_threshold")
            if verdict is not None:
                expected = ("verified_benign_tool_completion" if r["primary_behavior_type"] == "benign_tool_completion"
                            else "verified_harmful_refusal" if r["primary_behavior_type"] == "harmful_request_refusal"
                            else "verified_safe_attack_defense")
                if verdict.get("category") != expected or r.get("quality_category") != expected:
                    errors.append("accepted_quality_category_mismatch")
                blocked_tags = {"side_effect", "unauthorized_side_effect", "unsafe_tool_call",
                                "harmful_action", "exfiltration", "persistence", "data_leak"}
                if blocked_tags & set(verdict.get("issue_tags", [])):
                    errors.append("accepted_has_risk_issue_tag")
            accepted[split] += 1
    if manifest["status"] == "judged_complete":
        if len(verdicts) != len(candidates):
            errors.append("complete_release_missing_verdicts")
        if sum(accepted.values()) != manifest["accepted_count"]:
            errors.append("accepted_count_mismatch")
    elif sum(accepted.values()):
        errors.append("incomplete_release_published_accepted_rows")
    if require_complete and manifest["status"] != "judged_complete":
        errors.append("release_incomplete")
    return {"status": manifest["status"], "candidates": len(candidates),
            "valid_verdicts": len(verdicts), "accepted": dict(accepted),
            "errors": sorted(set(errors))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = verify(args.release, args.require_complete)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
