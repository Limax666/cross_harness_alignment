#!/usr/bin/env python3
"""Download the complete public ActBench release and append it to the raw corpus.

This is an archival/import step only: it does not apply quality or score filters.
Run with the project's ``cross-harness-sft`` conda environment, which provides
``huggingface_hub`` and ``pyarrow``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


PROJECT = Path(__file__).resolve().parents[3]
CORPUS = PROJECT / "experiments/cross_harness_sft/data/raw/multi_harness_teacher_v1"
CORPUS_JSONL = CORPUS / "trajectories.jsonl"
ARCHIVE = PROJECT / "experiments/cross_harness_sft/data/raw/actbench_public_v1"
REPO_ID = "ZJUICSR/ActBench"
REPO_URL = "https://huggingface.co/datasets/ZJUICSR/ActBench"

ARCHIVE_METADATA = (
    "README.md",
    "manifest.json",
    "metadata/combo_manifest.json",
    "metadata/data_files.json",
    "metadata/task_pairs_manifest.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(relative_path: str) -> Path:
    cached = hf_hub_download(
        repo_id=REPO_ID,
        filename=relative_path,
        repo_type="dataset",
        local_dir=str(ARCHIVE),
    )
    return Path(cached)


def parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def text_content(parts: Any) -> str:
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        return "" if parts is None else json.dumps(parts, ensure_ascii=False)
    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            texts.append(str(part.get("text") or ""))
        elif isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("type") != "toolCall":
            texts.append(json.dumps(part, ensure_ascii=False, sort_keys=True))
    return "\n".join(s for s in texts if s)


def as_openai_messages(trace: dict[str, Any], harness: str, task_id: str) -> list[dict[str, Any]]:
    """Make a readable interchange view while retaining the untouched trace in raw_payload."""
    context = "\n".join((
        "<harness_context>",
        f"name: {harness}",
        "interface: ActBench published native harness run (task-scoped MCP)",
        "tool_contract: ActBench task-scoped tools; see recorded tool calls and results",
        "</harness_context>",
    ))
    messages: list[dict[str, Any]] = [{"role": "system", "content": context}]
    transcript = trace.get("transcript") or {}
    entries = transcript.get("entries") or []
    if not isinstance(entries, list):
        return messages
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        kind = entry.get("type")
        message = entry.get("message") or {}
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        parts = message.get("content") or []
        if kind == "message" and role == "toolResult":
            for part in parts if isinstance(parts, list) else []:
                if not isinstance(part, dict):
                    continue
                messages.append({
                    "role": "tool",
                    "tool_call_id": str(part.get("tool_call_id") or part.get("id") or ""),
                    "name": str(part.get("name") or "unknown_tool"),
                    "content": str(part.get("text") or ""),
                })
        elif kind == "message":
            if role not in {"system", "developer", "user", "assistant"}:
                continue
            converted: dict[str, Any] = {"role": role, "content": text_content(parts)}
            calls = []
            if isinstance(parts, list):
                for part in parts:
                    if not isinstance(part, dict) or part.get("type") != "toolCall":
                        continue
                    calls.append({
                        "id": str(part.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": str(part.get("name") or "unknown_tool"),
                            "arguments": json.dumps(part.get("arguments") or {}, ensure_ascii=False),
                        },
                    })
            if calls:
                converted["tool_calls"] = calls
            messages.append(converted)
        elif kind == "toolResult" or role == "toolResult":
            if isinstance(parts, list):
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    messages.append({
                        "role": "tool",
                        "tool_call_id": str(part.get("tool_call_id") or part.get("id") or ""),
                        "name": str(part.get("name") or "unknown_tool"),
                        "content": str(part.get("text") or ""),
                    })
        elif role in {"user", "assistant", "system", "developer"}:
            messages.append({"role": role, "content": text_content(parts)})
    return messages


def observed_tool_schemas(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record tool names and observed argument keys; source transcript remains authoritative."""
    names: dict[str, set[str]] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            args = parse_json_string(function.get("arguments"))
            if name:
                names.setdefault(name, set()).update(args.keys() if isinstance(args, dict) else ())
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Tool observed in this published ActBench trajectory; consult raw_payload for the exact native interaction.",
                "parameters": {
                    "type": "object",
                    "properties": {key: {} for key in sorted(keys)},
                    "additionalProperties": True,
                },
            },
        }
        for name, keys in sorted(names.items())
    ]


def family_for_task(task_id: str) -> str:
    # Each task_pairs row contains the benign and adversarial form under this task_id.
    return f"actbench:{task_id}"


def main() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    CORPUS.mkdir(parents=True, exist_ok=True)

    for item in ARCHIVE_METADATA:
        fetch(item)
    manifest = json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((ARCHIVE / "metadata/data_files.json").read_text(encoding="utf-8"))
    task_manifest = json.loads((ARCHIVE / "metadata/task_pairs_manifest.json").read_text(encoding="utf-8"))

    expected_files = {item["path"]: item for item in data_manifest["files"]}
    if len(expected_files) != 21:
        raise ValueError(f"Expected 21 published data files, got {len(expected_files)}")
    for relative, expected in expected_files.items():
        path = fetch(relative)
        actual_hash = sha256_file(path)
        if path.stat().st_size != expected["size_bytes"] or actual_hash != expected["sha256"]:
            raise ValueError(f"Published file integrity mismatch for {relative}")

    task_path = ARCHIVE / task_manifest["data"]
    pairs: dict[str, dict[str, Any]] = {}
    pair_file = pq.ParquetFile(task_path)
    if pair_file.metadata.num_rows != 300:
        raise ValueError(f"Expected 300 task pairs, found {pair_file.metadata.num_rows}")
    for batch in pair_file.iter_batches(batch_size=32):
        for pair in batch.to_pylist():
            pairs[pair["task_id"]] = pair
    if len(pairs) != 300 or sum(1 for _ in pairs) * 2 != 600:
        raise ValueError(f"Task-pair uniqueness/count check failed: {len(pairs)}")

    # The top-level release manifest carries the authoritative path (under
    # data/trajectories/); metadata/combo_manifest.json is only a summary and
    # uses a shorter legacy path.
    combo_manifest = manifest.get("combos") or []
    if len(combo_manifest) != 20 or manifest.get("row_count") != 24000:
        raise ValueError("Release manifest does not match expected 20 combinations / 24,000 trajectories")

    existing_ids: set[str] = set()
    if CORPUS_JSONL.exists():
        with CORPUS_JSONL.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if line.strip():
                    existing_ids.add(json.loads(line).get("record_id", ""))

    staging = CORPUS / ".actbench_import_staging.jsonl"
    counts = Counter()
    added_by_harness = Counter()
    pair_roles = Counter()
    combo_counts = Counter()
    tool_call_count = 0
    observable_count = 0
    score_available = Counter()
    seen_source_ids: set[str] = set()
    with staging.open("w", encoding="utf-8") as output:
        for combo in combo_manifest:
            rel = combo["data"]
            parquet_path = ARCHIVE / rel
            parquet = pq.ParquetFile(parquet_path)
            if parquet.metadata.num_rows != combo["row_count"]:
                raise ValueError(f"Row count mismatch: {rel}: {parquet.metadata.num_rows}")
            local_combo = Counter()
            for batch in parquet.iter_batches(batch_size=32):
                for source_row in batch.to_pylist():
                    task_id = str(source_row["task_id"])
                    if task_id not in pairs:
                        raise ValueError(f"Trajectory references unknown task pair: {task_id}")
                    trace = parse_json_string(source_row.get("trajectory_json"))
                    score = parse_json_string(source_row.get("score_json"))
                    if not isinstance(trace, dict):
                        raise ValueError(f"Invalid trajectory_json in {rel} task {task_id}")
                    backend = str(source_row["backend"]).lower()
                    role = str(source_row["role"]).lower()
                    task_type = "benign" if role in {"clean", "benign", "benign_baseline"} else "attack"
                    messages = as_openai_messages(trace, backend, task_id)
                    tools = observed_tool_schemas(messages)
                    calls = sum(len(m.get("tool_calls") or []) for m in messages if m.get("role") == "assistant")
                    tool_call_count += calls
                    has_turns = len(messages) >= 2 and any(m.get("role") == "assistant" for m in messages)
                    observable_count += int(has_turns)
                    score_available["available" if isinstance(score, dict) and score.get("available") else "unavailable"] += 1

                    trajectory_id = str(source_row["trajectory_id"])
                    record_id = f"actbench:{source_row['combo_id']}:{trajectory_id}"
                    if record_id in seen_source_ids:
                        raise ValueError(f"Duplicate trajectory_id in release: {record_id}")
                    seen_source_ids.add(record_id)
                    local_combo[(task_type, "rows")] += 1
                    pair_roles[(task_id, task_type)] += 1
                    pair_meta = pairs[task_id]
                    model = str(source_row.get("model") or "unknown")
                    provider = model.split("/", 1)[0] if "/" in model else "unknown"
                    execution = trace.get("execution") or {}
                    row = {
                        "record_id": record_id,
                        "benchmark": "actbench",
                        "harness_name": backend,
                        "harness_context": "\n".join((
                            "<harness_context>", f"name: {backend}",
                            "interface: ActBench published native harness run (task-scoped MCP)",
                            "tool_contract: ActBench task-scoped tools; see recorded tool calls and results",
                            "</harness_context>",
                        )),
                        "teacher_model": model,
                        "teacher_provider": provider,
                        "source_kind": "actbench_published_trajectory",
                        "synthetic": False,
                        "collection_status": execution.get("status") or "completed",
                        "has_observable_turn_content": has_turns,
                        "source_path": f"experiments/cross_harness_sft/data/raw/actbench_public_v1/{rel}",
                        "source_line": None,
                        "task_id": task_id,
                        "family": family_for_task(task_id),
                        "task_type": task_type,
                        "official_score": score if isinstance(score, dict) else {},
                        "messages": messages,
                        "tools": tools,
                        "task_metadata": {
                            "scenario_id": pair_meta.get("scenario_id"),
                            "behavior_id": pair_meta.get("behavior_id"),
                            "behavior_type": pair_meta.get("behavior_type"),
                            "behavior_label": pair_meta.get("behavior_label"),
                            "scoring_family": pair_meta.get("scoring_family"),
                            "attack_method": pair_meta.get("attack_method"),
                            "expected_behavior": pair_meta.get("expected_behavior"),
                            "attack_goal": pair_meta.get("attack_goal"),
                        },
                        # Preserve the complete official row, including the untouched native trajectory JSON.
                        "raw_payload": source_row,
                    }
                    if record_id not in existing_ids:
                        output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        added_by_harness[backend] += 1
                    counts[backend] += 1
                    combo_counts[str(source_row["combo_id"])] += 1
            expected_roles = {("benign", "rows"): 300, ("attack", "rows"): 900}
            observed_roles = {(kind, "rows"): count for (kind, _), count in local_combo.items()}
            if observed_roles != expected_roles:
                raise ValueError(f"Unexpected role mix for {combo['combo_id']}: {observed_roles}")

    if len(seen_source_ids) != 24000 or len(combo_counts) != 20:
        raise ValueError(f"Incomplete source coverage: rows={len(seen_source_ids)}, combos={len(combo_counts)}")
    if any(pair_roles[(task, kind)] != 20 * (1 if kind == "benign" else 3)
           for task in pairs for kind in ("benign", "attack")):
        raise ValueError("Not every case has the expected number of released trajectories")

    # Atomically replace the canonical corpus with the old bytes plus only new IDs.
    combined = CORPUS / ".trajectories.actbench_combined.tmp"
    with combined.open("wb") as out:
        if CORPUS_JSONL.exists():
            with CORPUS_JSONL.open("rb") as old:
                shutil.copyfileobj(old, out, length=8 * 1024 * 1024)
        with staging.open("rb") as added:
            shutil.copyfileobj(added, out, length=8 * 1024 * 1024)
        out.flush()
        os.fsync(out.fileno())
    os.replace(combined, CORPUS_JSONL)
    staging.unlink(missing_ok=True)

    existing_report_path = CORPUS / "dataset_report.json"
    report = json.loads(existing_report_path.read_text(encoding="utf-8")) if existing_report_path.exists() else {}
    prior_count = int(report.get("record_count", 0))
    added_count = sum(1 for line in (CORPUS / "trajectories.jsonl").open(encoding="utf-8") if line.strip()) - prior_count
    by_harness = Counter(report.get("records_by_harness", {}))
    for harness, count in added_by_harness.items():
        by_harness[harness] += count
    by_source = Counter(report.get("records_by_source_kind", {}))
    by_source["actbench_published_trajectory"] += added_count
    report["record_count"] = prior_count + added_count
    report["records_by_harness"] = dict(sorted(by_harness.items()))
    report["records_by_source_kind"] = dict(sorted(by_source.items()))
    report["records_by_benchmark"] = dict(report.get("records_by_benchmark", {}))
    report["records_by_benchmark"]["actbench"] = report["records_by_benchmark"].get("actbench", 0) + added_count
    report["latest_import"] = {
        "dataset": "ActBench public release v1",
        "source": REPO_URL,
        "source_revision": "31bd732e9c083ddeb19bf152048386d38e511c90",
        "task_pairs": 300,
        "paired_cases": 600,
        "released_trajectory_rows": 24000,
        "rows_appended_this_run": added_count,
        "quality_filter_applied": False,
        "role_rows_in_release": {"benign": 6000, "attack": 18000},
    }
    report.setdefault("purpose", "append-only raw corpus; not filtered and not directly trainable")
    report.setdefault("required_next_step", "joint quality filtering, family-disjoint split assignment, then TRL conversational conversion")
    existing_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    import_report = {
        "dataset": "ZJUICSR/ActBench",
        "source": REPO_URL,
        "license": "MIT",
        "release_id": "actbench-public-v1",
        "manifest_row_count": manifest["row_count"],
        "combo_count": len(combo_counts),
        "task_pair_count": len(pairs),
        "paired_case_count": len(pairs) * 2,
        "trajectory_rows": len(seen_source_ids),
        "rows_appended_this_run": added_count,
        "role_rows": {"benign": 6000, "attack": 18000},
        "rows_by_harness": dict(sorted(counts.items())),
        "rows_by_combo": dict(sorted(combo_counts.items())),
        "score_json_availability": dict(score_available),
        "trajectory_rows_with_observable_assistant_turns": observable_count,
        "assistant_tool_calls": tool_call_count,
        "family_count": len(pairs),
        "quality_filter_applied": False,
        "task_pair_sha256": task_manifest["sha256"],
        "data_file_sha256_verified": len(expected_files),
        "source_revision": "31bd732e9c083ddeb19bf152048386d38e511c90",
    }
    (ARCHIVE / "import_report.json").write_text(json.dumps(import_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(import_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
