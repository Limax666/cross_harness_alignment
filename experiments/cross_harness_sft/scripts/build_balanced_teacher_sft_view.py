#!/usr/bin/env python3
"""Freeze an auditable, without-replacement cell-stratified SFT order.

This selects whole qualified trajectories. Sparse harnesses stay in the audit
pool until they reach the registered 32-row minimum; no row is duplicated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell(row: dict) -> tuple[str, str, str, str]:
    return (row["harness_name"], row["benchmark"], row["primary_behavior_type"], row["quality_category"])


def risk(row: dict) -> bool:
    return row["primary_behavior_type"] != "benign_tool_completion"


def choose(pool: dict, drawn: Counter, allowed: set[str], preferred: bool | None) -> dict | None:
    """Hierarchical weighted round robin over represented, nonempty cells."""
    choices = [key for key, records in pool.items() if records and key[0] in allowed]
    if preferred is not None:
        preferred_keys = [key for key in choices if risk(pool[key][0]) == preferred]
        if preferred_keys:
            choices = preferred_keys
    if not choices:
        return None
    harnesses = {key[0] for key in choices}
    h = min(harnesses, key=lambda name: (sum(v for key, v in drawn.items() if key[0] == name), name))
    sources = {key[1] for key in choices if key[0] == h}
    s = min(sources, key=lambda name: (sum(v for key, v in drawn.items() if key[:2] == (h, name)), name))
    keys = [key for key in choices if key[:2] == (h, s)]
    k = min(keys, key=lambda key: (drawn[key], key[2], key[3]))
    drawn[k] += 1
    return pool[k].popleft()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("preflight", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--global-batch", type=int, default=8)
    ap.add_argument("--min-harness-rows", type=int, default=32)
    ap.add_argument("--max-rows-per-harness", type=int, default=0,
                    help="0 means shortest eligible harness, rounded down to global-batch multiple")
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output exists; freeze a new version rather than overwrite it")
    train_path = args.preflight / "qualified_train.jsonl"
    val_path = args.preflight / "qualified_validation.jsonl"
    train, validation = load(train_path), load(val_path)
    if len(validation) < 8:
        ap.error("Fewer than eight tokenizer-qualified validation rows")
    if any(row["split"] != "train" for row in train) or any(row["split"] != "validation" for row in validation):
        ap.error("Qualified input has a split mismatch")
    available = Counter(row["harness_name"] for row in train)
    included = sorted(h for h, n in available.items() if n >= args.min_harness_rows)
    if len(included) < 3:
        ap.error("Fewer than three harnesses satisfy the train-row gate")
    per_harness = args.max_rows_per_harness or min(available[h] for h in included)
    per_harness = (per_harness // args.global_batch) * args.global_batch
    if per_harness < args.min_harness_rows or any(available[h] < per_harness for h in included):
        ap.error("Per-harness budget violates the qualified-row gate")
    total = per_harness * len(included)
    total -= total % args.global_batch
    if total < args.global_batch * 16:
        ap.error("Fewer than 16 complete optimizer updates per epoch; collect more qualifying rows")

    grouped = defaultdict(list)
    for row in train:
        if row["harness_name"] in included:
            grouped[cell(row)].append(row)
    rng = random.Random(args.seed)
    pool = {}
    for key, records in sorted(grouped.items()):
        # Shuffle families first so a large same-family set cannot fill the pool head.
        by_family = defaultdict(list)
        for row in records:
            by_family[row["source_family_id"]].append(row)
        families = sorted(by_family)
        rng.shuffle(families)
        for family in families:
            rng.shuffle(by_family[family])
        order = []
        while any(by_family.values()):
            for family in families:
                if by_family[family]:
                    order.append(by_family[family].pop())
        pool[key] = deque(order)
    drawn = Counter()
    harness_drawn = Counter()
    selected = []
    batch_audit = []
    for update in range(total // args.global_batch):
        batch = []
        for position in range(args.global_batch):
            allowed = {h for h in included if harness_drawn[h] < per_harness}
            preferred = True if position == 0 else False if position == 1 else None
            row = choose(pool, drawn, allowed, preferred)
            if row is None:
                ap.error("A registered harness pool exhausted before its quota")
            batch.append(row)
            harness_drawn[row["harness_name"]] += 1
        if not any(not risk(row) for row in batch) or not any(risk(row) for row in batch):
            ap.error(f"Update {update} lacks a benign or risk trajectory")
        selected.extend(batch)
        batch_audit.append({"update": update + 1, "rows": len(batch),
                            "harnesses": dict(Counter(row["harness_name"] for row in batch)),
                            "benign": sum(not risk(row) for row in batch),
                            "risk": sum(risk(row) for row in batch),
                            "assistant_tokens": sum(row["token_preflight"]["assistant_tokens"] for row in batch)})
    ids = [row["record_id"] for row in selected]
    if len(ids) != len(set(ids)):
        ap.error("Repeated record_id in a without-replacement view")
    train_families = {row["source_family_id"] for row in selected}
    val_families = {row["source_family_id"] for row in validation}
    if train_families & val_families:
        ap.error("Family leakage between selected train and natural validation")
    args.output.mkdir(parents=True)
    def write(path: Path, records: list[dict]) -> None:
        with path.open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    write(args.output / "train_balanced.jsonl", selected)
    write(args.output / "validation_natural.jsonl", validation)
    write(args.output / "batch_audit.jsonl", batch_audit)
    def summarize(records: list[dict]) -> dict:
        groups = defaultdict(list)
        for row in records:
            groups[cell(row)].append(row)
        return {"/".join(key): {"rows": len(items),
                "families": len({item["source_family_id"] for item in items}),
                "assistant_tokens": sum(item["token_preflight"]["assistant_tokens"] for item in items),
                "mean_assistant_tokens": round(sum(item["token_preflight"]["assistant_tokens"] for item in items) / len(items), 2),
                "trajectory_mass": round(len(items) / len(records), 6)}
                for key, items in sorted(groups.items())}
    windows = []
    for first in range(0, len(selected), args.global_batch * 32):
        rows = selected[first:first + args.global_batch * 32]
        windows.append({"first_update": first // args.global_batch + 1,
                        "updates": len(rows) // args.global_batch,
                        "harnesses": dict(Counter(row["harness_name"] for row in rows)),
                        "sources": dict(Counter(row["benchmark"] for row in rows)),
                        "behaviors": dict(Counter(row["primary_behavior_type"] for row in rows)),
                        "assistant_tokens": sum(row["token_preflight"]["assistant_tokens"] for row in rows)})
    target_weights = {}
    for key in sorted(grouped):
        h, s, _, _ = key
        sources = {k[1] for k in grouped if k[0] == h}
        sibling_cells = {k for k in grouped if k[:2] == (h, s)}
        target_weights["/".join(key)] = 1 / len(included) / len(sources) / len(sibling_cells)
    manifest = {
        "format": "complete_teacher_cell_stratified_view_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_preflight_manifest_sha256": digest(args.preflight / "preflight_manifest.json"),
        "qualified_train_sha256": digest(train_path),
        "qualified_validation_sha256": digest(val_path),
        "seed": args.seed, "global_batch": args.global_batch, "gradient_accumulation": "set in trainer run manifest",
        "min_harness_rows": args.min_harness_rows,
        "included_harnesses": included, "excluded_sparse_harnesses": {h: n for h, n in sorted(available.items()) if h not in included},
        "available_by_harness": dict(sorted(available.items())),
        "per_harness_budget": per_harness, "selected_by_harness": dict(sorted(harness_drawn.items())),
        "train_rows": len(selected), "validation_rows": len(validation),
        "train_benign": sum(not risk(row) for row in selected),
        "train_risk": sum(risk(row) for row in selected),
        "validation_benign": sum(not risk(row) for row in validation),
        "validation_risk": sum(risk(row) for row in validation),
        "train_unique_families": len(train_families), "validation_unique_families": len(val_families),
        "repeat_factor": 1, "no_replacement_within_epoch": True,
        "target_cell_trajectory_weight": {k: round(v, 8) for k, v in target_weights.items()},
        "realized_train_cells": summarize(selected), "natural_validation_cells": summarize(validation),
        "logging_windows_32_updates": windows,
        "loss_limitation": "This file sets trajectory order only. The stock VeRL trainer uses token-weighted loss; cell-mean loss needs an explicit trainer implementation and verification.",
        "outputs": {name: digest(args.output / name) for name in
                    ("train_balanced.jsonl", "validation_natural.jsonl", "batch_audit.jsonl")},
    }
    (args.output / "sampling_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "train_rows": len(selected),
                      "validation_rows": len(validation), "included_harnesses": included,
                      "train_benign": manifest["train_benign"], "train_risk": manifest["train_risk"]}, indent=2))


if __name__ == "__main__":
    main()
