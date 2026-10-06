#!/usr/bin/env python3
"""Build a compact, deterministic working view without changing raw sources.

The view retains the latest SafeClawArena attempt per task, every other
non-ActBench row, and a family-matched ActBench sample across six harnesses.
The full append-only corpus remains canonical and keeps every attempt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
ROOT = EXP / "data/raw/multi_harness_teacher_v1"
RAW = ROOT / "trajectories.jsonl"
VIEWS = ROOT / "views"
HARNESS = ("claudecode", "hermes", "openagent", "openclaw", "opencode", "qwenpaw")
OPENCLAW_TEACHER = "deepseek/deepseek-v4-pro"
TASK_TYPES = ("benign", "attack")


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safeclaw_attempt_key(row: dict[str, Any]) -> tuple[str, str]:
    """Select by collection time, independent of task outcome or judge score."""
    return (str((row.get("raw_payload") or {}).get("time") or ""), str(row.get("record_id") or ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-harness", type=int, default=200,
                        help="ActBench rows per harness; must be even")
    parser.add_argument("--output", type=Path, default=VIEWS / "compact_actbench_1200_v1.jsonl")
    args = parser.parse_args()
    if args.per_harness < 2 or args.per_harness % 2:
        parser.error("--per-harness must be an even number >= 2")
    if not RAW.is_file():
        parser.error(f"raw corpus not found: {RAW}")

    half = args.per_harness // 2
    snapshot_size = RAW.stat().st_size
    # family -> task type -> harness -> candidate record IDs
    candidates: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    latest_safeclaw: dict[tuple[str, str], tuple[tuple[str, str], str]] = {}
    raw_hash = hashlib.sha256()
    raw_rows = 0
    with RAW.open("rb") as src:
        consumed = 0
        for line in src:
            consumed += len(line)
            if consumed > snapshot_size or not line.endswith(b"\n"):
                break
            raw_hash.update(line)
            raw_rows += 1
            row = json.loads(line)
            if row.get("benchmark") == "safeclawarena":
                key = (str(row.get("harness_name") or ""), str(row.get("task_id") or ""))
                choice = (safeclaw_attempt_key(row), str(row.get("record_id") or ""))
                if key not in latest_safeclaw or choice[0] > latest_safeclaw[key][0]:
                    latest_safeclaw[key] = choice
            if row.get("benchmark") != "actbench":
                continue
            harness = str(row.get("harness_name") or "")
            task_type = str(row.get("task_type") or "")
            if harness not in HARNESS or task_type not in TASK_TYPES:
                continue
            if harness == "openclaw" and row.get("teacher_model") != OPENCLAW_TEACHER:
                continue
            family = str(row.get("family") or "")
            rid = str(row.get("record_id") or "")
            if family and rid:
                candidates[family][task_type][harness].append(rid)

    selected_ids: set[str] = set()
    selected_families: dict[str, list[str]] = {}
    for task_type in TASK_TYPES:
        eligible = [
            family for family, labels in candidates.items()
            if task_type in labels and all(labels[task_type].get(h) for h in HARNESS)
        ]
        eligible.sort(key=lambda family: stable_key(f"actbench-v1|{task_type}|{family}"))
        if len(eligible) < half:
            parser.error(f"Only {len(eligible)} common {task_type} families; need {half}")
        chosen = eligible[:half]
        selected_families[task_type] = chosen
        for family in chosen:
            for harness in HARNESS:
                ids = candidates[family][task_type][harness]
                # One deterministic trajectory per matched family/harness.
                rid = min(ids, key=lambda value: stable_key(f"actbench-row-v1|{value}"))
                selected_ids.add(rid)

    VIEWS.mkdir(parents=True, exist_ok=True)
    output = args.output if args.output.is_absolute() else EXP / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_suffix(".manifest.json")
    if output.exists() or manifest.exists():
        parser.error(f"versioned output already exists; choose a new --output path: {output}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    counts = Counter()
    rows_out = 0
    selected_actbench_ids: set[str] = set()
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as dst, RAW.open(encoding="utf-8") as src:
            consumed = 0
            for line in src:
                consumed += len(line.encode("utf-8"))
                if consumed > snapshot_size or not line.endswith("\n"):
                    break
                row: dict[str, Any] = json.loads(line)
                if row.get("benchmark") == "actbench":
                    if row.get("record_id") not in selected_ids:
                        continue
                    selected_actbench_ids.add(row["record_id"])
                elif row.get("benchmark") == "safeclawarena":
                    key = (str(row.get("harness_name") or ""), str(row.get("task_id") or ""))
                    if row.get("record_id") != latest_safeclaw[key][1]:
                        continue
                dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows_out += 1
                counts[(str(row.get("benchmark") or "unknown"), str(row.get("harness_name") or "unknown"))] += 1
            dst.flush()
            os.fsync(dst.fileno())
        if selected_actbench_ids != selected_ids:
            raise RuntimeError(f"raw corpus changed during build; missing selected rows={len(selected_ids-selected_actbench_ids)}")
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, output)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise

    out_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "view": "compact_actbench_family_matched_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "compact working view; unfiltered candidates, not a train-ready release",
        "source_corpus": str(RAW.relative_to(EXP)),
        "source_rows_observed": raw_rows,
        "source_snapshot_bytes": snapshot_size,
        "source_sha256": raw_hash.hexdigest(),
        "output": str(output.relative_to(EXP)),
        "output_rows": rows_out,
        "output_sha256": out_hash,
        "actbench_rows": len(selected_ids),
        "actbench_target_per_harness": args.per_harness,
        "actbench_harnesses": list(HARNESS),
        "openclaw_teacher_model": OPENCLAW_TEACHER,
        "actbench_behavior_counts_per_harness": {t: half for t in TASK_TYPES},
        "actbench_selected_families": selected_families,
        "actbench_unique_selected_family_count": len(set().union(*(set(v) for v in selected_families.values()))),
        "actbench_row_counts_by_benchmark_harness": {
            f"{benchmark}/{harness}": count
            for (benchmark, harness), count in sorted(counts.items()) if benchmark == "actbench"
        },
        "all_non_actbench_rows_retained": False,
        "safeclawarena_latest_attempt_per_task": True,
        "safeclawarena_unique_tasks": len(latest_safeclaw),
        "safeclawarena_attempt_selection": "maximum native collection time per (harness, task_id), regardless of status or judged quality; all attempts remain in raw ledger",
        "raw_sources_modified": False,
        "selection": "same deterministic ActBench family sample across six harnesses plus latest SafeClawArena attempt per task; no outcome-based selection",
        "quality_filter_applied": False,
        "family_split_assigned": False,
    }
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "manifest": str(manifest),
        "rows": rows_out,
        "actbench_rows": len(selected_ids),
        "actbench_per_harness": args.per_harness,
        "other_rows": rows_out - len(selected_ids),
        "sha256": out_hash,
        "family_counts": {k: len(v) for k, v in selected_families.items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
