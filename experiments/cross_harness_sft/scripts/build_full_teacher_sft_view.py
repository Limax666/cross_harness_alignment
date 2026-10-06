#!/usr/bin/env python3
"""Freeze every tokenizer-qualified teacher trace in mixed, complete VeRL batches.

VeRL's SFT loader drops incomplete global batches. Explicitly recorded repeats
pad the train and validation views without losing any unique trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def is_benign(row: dict) -> bool:
    return row["primary_behavior_type"] == "benign_tool_completion"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("preflight", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--global-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output exists; freeze a new version rather than overwriting it")
    if args.global_batch < 4 or args.global_batch % 2:
        ap.error("Global batch must be even and at least four")
    train_path = args.preflight / "qualified_train.jsonl"
    val_path = args.preflight / "qualified_validation.jsonl"
    train, validation = read_jsonl(train_path), read_jsonl(val_path)
    preflight = json.loads((args.preflight / "preflight_manifest.json").read_text())
    if len(train) != preflight["counts"]["train/complete_trajectory_within_limit"]:
        ap.error("Qualified train count differs from tokenizer preflight")
    if len(validation) != preflight["counts"]["validation/complete_trajectory_within_limit"]:
        ap.error("Qualified validation count differs from tokenizer preflight")
    ids = [row["record_id"] for row in train]
    if len(ids) != len(set(ids)):
        ap.error("Qualified train contains duplicate record IDs")
    if {row["source_family_id"] for row in train} & {row["source_family_id"] for row in validation}:
        ap.error("Task-family leakage between train and validation")
    if any(row["split"] != "train" for row in train) or any(row["split"] != "validation" for row in validation):
        ap.error("Preflight split mismatch")

    rng = random.Random(args.seed)
    pad_count = (-len(train)) % args.global_batch
    benign = [row for row in train if is_benign(row)]
    risk = [row for row in train if not is_benign(row)]
    # Pad the smallest benign harnesses first, never repeating a row twice.
    by_harness = Counter(row["harness_name"] for row in benign)
    padding = []
    for harness in sorted(by_harness, key=lambda name: (by_harness[name], name)):
        if len(padding) == pad_count:
            break
        padding.append(min((row for row in benign if row["harness_name"] == harness),
                           key=lambda row: row["record_id"]))
    if len(padding) != pad_count:
        ap.error("Insufficient benign traces for explicit batch padding")
    padding_ids = [row["record_id"] for row in padding]
    benign.extend(padding)
    val_pad_count = (-len(validation)) % args.global_batch
    val_by_harness = Counter(row["harness_name"] for row in validation)
    val_quotas = {harness: val_pad_count * count // len(validation)
                  for harness, count in val_by_harness.items()}
    remaining = val_pad_count - sum(val_quotas.values())
    for harness in sorted(val_by_harness, key=lambda name: (-(val_pad_count * val_by_harness[name] % len(validation)), name))[:remaining]:
        val_quotas[harness] += 1
    val_padding = []
    used_val_families = set()
    for harness, quota in sorted(val_quotas.items()):
        candidates = sorted((row for row in validation if row["harness_name"] == harness),
                            key=lambda row: (row["source_family_id"], row["record_id"]))
        for _ in range(quota):
            chosen = next((row for row in candidates if row["source_family_id"] not in used_val_families),
                          candidates[0])
            val_padding.append(chosen)
            used_val_families.add(chosen["source_family_id"])
            candidates.remove(chosen)
    val_padding_ids = [row["record_id"] for row in val_padding]
    validation_padded = validation + val_padding
    if len(val_padding_ids) != len(set(val_padding_ids)) or len(validation_padded) % args.global_batch:
        ap.error("Validation padding must use distinct records and complete batches")
    batch_count = (len(train) + pad_count) // args.global_batch
    if len(benign) < batch_count or len(risk) < batch_count:
        ap.error("Cannot put benign and risk traces in every optimizer batch")
    targets = [len(benign) // batch_count] * batch_count
    indices = list(range(batch_count))
    rng.shuffle(indices)
    for index in indices[:len(benign) % batch_count]:
        targets[index] += 1
    if min(targets) < 1 or max(targets) >= args.global_batch:
        ap.error("Invalid benign/risk batch targets")
    batches: list[list[dict]] = [[] for _ in range(batch_count)]
    supervised_tokens = [0] * batch_count
    tie_order = {index: order for order, index in enumerate(indices)}

    def place(rows: list[dict], quota: list[int]) -> None:
        decorated = [(rng.random(), row) for row in rows]
        decorated.sort(key=lambda pair: (-pair[1]["token_preflight"]["assistant_tokens"], pair[0]))
        placed = [0] * batch_count
        for _, row in decorated:
            candidates = [index for index in range(batch_count) if placed[index] < quota[index]]
            if not candidates:
                ap.error("Batch quotas exhausted before all traces were placed")

            def score(index: int) -> tuple:
                batch = batches[index]
                return (
                    sum(x["source_family_id"] == row["source_family_id"] for x in batch),
                    sum(x["harness_name"] == row["harness_name"] for x in batch),
                    sum(x["primary_behavior_type"] == row["primary_behavior_type"] for x in batch),
                    sum(x["benchmark"] == row["benchmark"] for x in batch),
                    supervised_tokens[index], len(batch), tie_order[index],
                )

            chosen = min(candidates, key=score)
            batches[chosen].append(row)
            placed[chosen] += 1
            supervised_tokens[chosen] += row["token_preflight"]["assistant_tokens"]

    place(benign, targets)
    place(risk, [args.global_batch - target for target in targets])
    for batch in batches:
        if len(batch) != args.global_batch or not any(is_benign(row) for row in batch) or not any(not is_benign(row) for row in batch):
            ap.error("A planned optimizer batch is incomplete or lacks a task class")
        rng.shuffle(batch)
    ordered = [row for batch in batches for row in batch]
    observed = Counter(row["record_id"] for row in ordered)
    if set(observed) != set(ids) or sorted(key for key, count in observed.items() if count == 2) != sorted(padding_ids) or any(count > 2 for count in observed.values()):
        ap.error("Full-view coverage or explicit padding audit failed")

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "train_balanced.jsonl", ordered)
    write_jsonl(args.output / "validation_natural.jsonl", validation_padded)
    audit = [{"update": number + 1, "rows": len(batch),
              "harnesses": dict(Counter(row["harness_name"] for row in batch)),
              "sources": dict(Counter(row["benchmark"] for row in batch)),
              "benign": sum(is_benign(row) for row in batch),
              "risk": sum(not is_benign(row) for row in batch),
              "assistant_tokens": sum(row["token_preflight"]["assistant_tokens"] for row in batch)}
             for number, batch in enumerate(batches)]
    write_jsonl(args.output / "batch_audit.jsonl", audit)
    manifest = {
        "format": "complete_teacher_full_preflight_view_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_preflight_manifest_sha256": digest(args.preflight / "preflight_manifest.json"),
        "qualified_train_sha256": digest(train_path),
        "qualified_validation_sha256": digest(val_path),
        "seed": args.seed, "global_batch": args.global_batch,
        "unique_train_rows": len(train), "train_rows": len(ordered),
        "unique_validation_rows": len(validation), "validation_rows": len(validation_padded),
        "optimizer_batches_per_epoch": batch_count,
        "padding_duplicates": pad_count, "padding_record_ids": padding_ids,
        "validation_padding_duplicates": val_pad_count,
        "validation_padding_record_ids": val_padding_ids,
        "no_replacement_except_explicit_padding": True,
        "train_unique_families": len({row["source_family_id"] for row in train}),
        "validation_unique_families": len({row["source_family_id"] for row in validation}),
        "train_benign_unique": sum(is_benign(row) for row in train),
        "train_risk_unique": sum(not is_benign(row) for row in train),
        "by_harness_unique": dict(sorted(Counter(row["harness_name"] for row in train).items())),
        "by_source_unique": dict(sorted(Counter(row["benchmark"] for row in train).items())),
        "outputs": {name: digest(args.output / name) for name in
                    ("train_balanced.jsonl", "validation_natural.jsonl", "batch_audit.jsonl")},
    }
    (args.output / "sampling_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "unique_train_rows": len(train),
                      "train_rows_with_padding": len(ordered), "unique_validation_rows": len(validation),
                      "validation_rows_with_padding": len(validation_padded),
                      "batches_per_epoch": batch_count, "padding_duplicates": pad_count,
                      "validation_padding_duplicates": val_pad_count}, indent=2))


if __name__ == "__main__":
    main()
