"""Strict rejection sampling and family-disjoint multi-turn SFT export."""
from __future__ import annotations

import argparse
import hashlib
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


def split_bucket(family: str, seed: str) -> float:
    return int(hashlib.sha256((seed + "\0" + family).encode()).hexdigest()[:16], 16) / float(16 ** 16)


def clean_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list): raise ValueError("messages missing")
    result: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("invalid message")
        result.append({key: value for key, value in message.items() if key in {"role", "content", "tool_calls", "tool_call_id", "name"}})
    if not result or result[-1].get("role") != "assistant": raise ValueError("trajectory has no final assistant turn")
    final = str(result[-1].get("content") or "")
    if "<think>" not in final or "</think>" not in final: raise ValueError("final target lacks think tags")
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--validation-fraction", type=float, default=.1)
    p.add_argument("--split-seed", default="cross-harness-rft-v2")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--max-response-length", type=int, default=16384)
    p.add_argument("--max-per-family", type=int, default=0)
    args = p.parse_args()
    strict = SimpleNamespace(max_turns=args.max_turns, max_response_length=args.max_response_length,
                             require_runtime_skill=False, allow_legacy_context_reconstruction=False)
    accepted: list[dict[str, Any]] = []; rejected: Counter[str] = Counter(); per_family: Counter[str] = Counter(); seen: set[str] = set()
    for row in read_jsonl(args.input):
        reason = rejection_reason(row, strict)
        meta = row.get("metadata") or {}
        if meta.get("native_harness") is not True or meta.get("controlled_teacher") is not False:
            reason = reason or "not_native_harness"
        if not str(meta.get("visible_reasoning") or "").strip():
            reason = reason or "missing_visible_reasoning"
        if reason: rejected[reason] += 1; continue
        try: messages = clean_messages(row)
        except ValueError as exc: rejected[str(exc)] += 1; continue
        digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if digest in seen: rejected["exact_duplicate"] += 1; continue
        family = str(meta.get("semantic_task_family_id") or meta.get("family_key") or meta.get("task_id"))
        if args.max_per_family and per_family[family] >= args.max_per_family: rejected["family_quota"] += 1; continue
        seen.add(digest); per_family[family] += 1
        tools = meta.get("available_tools")
        if not isinstance(tools, list) or not tools:
            rejected["missing_tool_registry"] += 1; continue
        accepted.append({"messages": messages, "tools": tools, "meta": {"source_episode_id": row.get("episode_id"), "family_key": family,
            "benchmark_id": meta.get("benchmark_id"), "task_id": meta.get("task_id"), "task_type": meta.get("task_type"),
            "harness_id": meta.get("harness_id"), "harness_version": meta.get("harness_version"),
            "teacher_id": meta.get("teacher_id"), "safety_module": meta.get("safety_module")}})
    train, validation = [], []
    for row in accepted:
        target = validation if split_bucket(row["meta"]["family_key"], args.split_seed) < args.validation_fraction else train
        target.append(row)
    train_families = {x["meta"]["family_key"] for x in train}; validation_families = {x["meta"]["family_key"] for x in validation}
    if train_families & validation_families: raise RuntimeError("family leakage")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train); write_jsonl(args.output_dir / "validation.jsonl", validation)
    report = {"schema": "cross_harness.rft_dataset.v2", "source": str(args.input), "accepted": len(accepted),
              "train": len(train), "validation": len(validation), "family_overlap": 0,
              "rejections": dict(sorted(rejected.items())), "distribution": {
                  "harness": dict(Counter(x["meta"]["harness_id"] for x in accepted)),
                  "benchmark": dict(Counter(x["meta"]["benchmark_id"] for x in accepted)),
                  "task_type": dict(Counter(x["meta"]["task_type"] for x in accepted))}}
    (args.output_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if train and validation else 2


if __name__ == "__main__": raise SystemExit(main())
