#!/usr/bin/env python3
"""Fail closed on a documented provider content-inspection block.

This never converts a failed call into an accept. It records why a candidate
cannot be judged so the release can finish with that row excluded from SFT.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("record_id")
    parser.add_argument("--http-status", type=int, required=True)
    parser.add_argument("--error-code", required=True)
    args = parser.parse_args()
    if (args.http_status, args.error_code) != (400, "data_inspection_failed"):
        parser.error("Only an observed HTTP 400 data_inspection_failed block may be recorded")
    candidate = next((json.loads(line) for line in (args.release / "candidates.jsonl").open()
                      if json.loads(line)["record_id"] == args.record_id), None)
    if candidate is None:
        parser.error("record ID is not a candidate")
    result_path = args.release / "judge_results.jsonl"
    attempts = [json.loads(line) for line in result_path.open()
                if json.loads(line)["candidate_sha256"] == candidate["candidate_sha256"]]
    if not attempts or attempts[-1]["verdict"].get("call_status") != "failed":
        parser.error("latest attempt must be a failed judge call")
    latest = attempts[-1]
    result = {"candidate_sha256": candidate["candidate_sha256"],
              "record_id": args.record_id, "model": latest["model"],
              "endpoint": latest["endpoint"], "prompt_version": latest["prompt_version"],
              "judged_at_utc": datetime.now(timezone.utc).isoformat(),
              "verdict": {"decision": "review", "category": "ambiguous",
                          "safety": 0, "utility": 0, "tool_correctness": 0,
                          "harness_fidelity": 0,
                          "issue_tags": ["provider_content_inspection_block", "unjudged_fail_closed"],
                          "rationale": "Independent judge provider returned HTTP 400 data_inspection_failed; no quality verdict was obtained.",
                          "call_status": "provider_blocked_review",
                          "provider_http_status": args.http_status,
                          "provider_error_code": args.error_code}}
    with result_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"record_id": args.record_id, "decision": "review",
                      "call_status": "provider_blocked_review"}))


if __name__ == "__main__":
    main()
