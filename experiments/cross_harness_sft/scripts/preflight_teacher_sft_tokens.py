#!/usr/bin/env python3
"""Audit complete accepted trajectories with the exact 2B tokenizer/template.

Never truncates a conversation. Overlength and render failures remain visible
in the length audit and are excluded only from this tokenizer-specific view.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Token preflight output already exists; use a new version")
    manifest = json.loads((args.release / "preprocessing_manifest.json").read_text())
    if manifest.get("status") != "judged_complete":
        parser.error("Only a complete judged release can be token-preflighted")
    from transformers import AutoTokenizer
    script = Path(__file__).with_name("build_multiharness_verl_sft.py")
    spec = importlib.util.spec_from_file_location("verl_renderer", script)
    renderer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(renderer)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    tokenizer.chat_template = renderer.VERL_SEGMENT_CHAT_TEMPLATE

    def count(messages: list[dict[str, str]]) -> int:
        result = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                return_tensors="pt", add_generation_prompt=False,
                                                enable_thinking=False)
        return int(result["input_ids"].shape[-1])

    args.output.mkdir(parents=True)
    counts = Counter()
    cells = defaultdict(lambda: {"rows": 0, "families": set(), "rendered_tokens": 0,
                                 "assistant_tokens": 0, "tool_calls": 0})
    with (args.output / "length_audit.jsonl").open("w", encoding="utf-8") as audit:
        for split in ("train", "validation"):
            source = args.release / f"accepted_{split}.jsonl"
            with (args.output / f"qualified_{split}.jsonl").open("w", encoding="utf-8") as qualified:
                for row in rows(source):
                    event = {"record_id": row["record_id"], "candidate_sha256": row["candidate_sha256"],
                             "split": split, "benchmark": row["benchmark"],
                             "harness_name": row["harness_name"],
                             "primary_behavior_type": row["primary_behavior_type"],
                             "quality_category": row["quality_category"],
                             "source_family_id": row["source_family_id"]}
                    try:
                        rendered = renderer.verl_messages(row["messages"])
                        total = count(rendered)
                        separate = [count([message]) for message in rendered]
                        if total != sum(separate):
                            raise ValueError("per-message tokenization differs from full trajectory")
                        assistant = sum(length for message, length in zip(rendered, separate)
                                        if message["role"] == "assistant")
                        if assistant <= 0:
                            raise ValueError("no supervised assistant tokens")
                        event.update(rendered_tokens=total, assistant_tokens=assistant,
                                     tool_calls=row["metrics"]["tool_calls"])
                        if total > args.max_length:
                            event.update(qualified=False, reason="over_length_unsegmentable_pending_review")
                        else:
                            event.update(qualified=True, reason="complete_trajectory_within_limit")
                            row["token_preflight"] = {"model": args.model, "max_length": args.max_length,
                                                      "rendered_tokens": total, "assistant_tokens": assistant,
                                                      "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
                            qualified.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                            cell = (split, row["harness_name"], row["benchmark"],
                                    row["primary_behavior_type"], row["quality_category"])
                            cells[cell]["rows"] += 1
                            cells[cell]["families"].add(row["source_family_id"])
                            cells[cell]["rendered_tokens"] += total
                            cells[cell]["assistant_tokens"] += assistant
                            cells[cell]["tool_calls"] += row["metrics"]["tool_calls"]
                    except Exception as exc:
                        event.update(qualified=False, reason="render_or_contract_error",
                                     error_type=type(exc).__name__, error=str(exc)[:240])
                    audit.write(json.dumps(event, ensure_ascii=False) + "\n")
                    counts[(split, event["reason"])] += 1
    report = {"format": "qwen35_2b_complete_trajectory_preflight_v1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_release_manifest_sha256": sha(args.release / "preprocessing_manifest.json"),
              "accepted_train_sha256": sha(args.release / "accepted_train.jsonl"),
              "accepted_validation_sha256": sha(args.release / "accepted_validation.jsonl"),
              "renderer_code_sha256": sha(script), "tokenizer": args.model,
              "tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
              "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
              "max_length": args.max_length, "truncation": "error",
              "counts": {"/".join(key): value for key, value in sorted(counts.items())},
              "cells": {"/".join(key): {**value, "families": len(value["families"])}
                        for key, value in sorted(cells.items())}}
    (args.output / "preflight_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "counts": report["counts"],
                      "qualified_cells": len(report["cells"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
