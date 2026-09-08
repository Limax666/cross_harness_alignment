from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .utils import read_jsonl, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "core" / "skillrl"))
from build_skill_use_sft import rejection_reason  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly verify and filter collected trajectories.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--rejected", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-response-length", type=int, default=8192)
    return parser.parse_args()


def provenance_reason(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    required = ["teacher_id", "benchmark_id", "benchmark_version", "task_id", "semantic_task_family_id", "harness_id", "harness_version", "harness_sha256", "safety_module"]
    missing = [key for key in required if not meta.get(key)]
    if missing:
        return "missing_provenance:" + ",".join(missing)
    if meta.get("native_harness") is not True or meta.get("controlled_teacher") is not False:
        return "not_native_harness"
    if not str(meta.get("visible_reasoning") or "").strip():
        return "missing_visible_reasoning"
    return ""


def main() -> int:
    args = parse_args()
    checker_args = SimpleNamespace(
        max_turns=args.max_turns,
        max_response_length=args.max_response_length,
        require_runtime_skill=False,
        allow_legacy_context_reconstruction=False,
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for row in read_jsonl(args.input):
        reason = provenance_reason(row) or rejection_reason(row, checker_args)
        if reason:
            copy = dict(row)
            copy["rejection_reason"] = reason
            rejected.append(copy)
            reasons[reason] += 1
        else:
            accepted.append(row)
    write_jsonl(args.accepted, accepted)
    write_jsonl(args.rejected, rejected)
    report = {"source": len(accepted) + len(rejected), "accepted": len(accepted), "rejected": len(rejected), "acceptance_rate": len(accepted) / max(1, len(accepted) + len(rejected)), "rejection_reasons": dict(sorted(reasons.items()))}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
