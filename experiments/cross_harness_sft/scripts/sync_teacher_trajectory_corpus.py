#!/usr/bin/env python3
"""Append unfiltered native teacher trajectories to the unified raw corpus.

Original source files remain untouched. Every normalized row embeds its exact
source payload so the unified JSONL is self-contained and auditable. The
watch mode tails active AgentHarm/SafeClawArena outputs and exits after all
known collectors stop and the sources have been idle for the configured grace
period. No quality, score, or safety filtering is performed here.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT = Path(__file__).resolve().parents[3]
EXP = PROJECT / "experiments/cross_harness_sft"
OUTPUTS = EXP / "outputs"
RAW = EXP / "data/raw/multi_harness_teacher_v1"
TRAJECTORIES = RAW / "trajectories.jsonl"
ATTEMPTS = RAW / "collection_attempts.jsonl"
REPORT = RAW / "dataset_report.json"
SYNC_REPORT = RAW / "source_sync_report.json"
LOCK = RAW / ".trajectory_sync.lock"
SAFECLAW = OUTPUTS / "safeclawarena_openclaw_glm53flash_raw"
AGENTHARM_GLOB = "agentharm*/test_public_v1.jsonl"
AGENTHARM_ERROR_GLOB = "agentharm*/test_public_v1.errors.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT.resolve()))
    except ValueError:
        return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_record_id(source_kind: str, harness: str, source_identity: str, raw: Any) -> str:
    blob = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(blob).hexdigest()
    return f"{source_kind}:{harness}:{source_identity}:{digest[:20]}"


def jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A collector may be in the middle of appending this line.
                continue
            if isinstance(value, dict):
                yield line_no, value


def normalized_tools(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed: dict[str, set[str]] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name:
                observed.setdefault(name, set()).update(args.keys() if isinstance(args, dict) else ())
    return [
        {"type": "function", "function": {
            "name": name,
            "description": "Observed tool in native trace. The original source payload is authoritative; upstream tool schema was not emitted in this transcript.",
            "parameters": {"type": "object", "properties": {key: {} for key in sorted(keys)}, "additionalProperties": True},
        }}
        for name, keys in sorted(observed.items())
    ]


def agentharm_rows(path: Path) -> Iterator[dict[str, Any]]:
    harness_hint = path.parent.name.removeprefix("agentharm_").removesuffix("_native")
    for line_no, raw in jsonl(path):
        meta = raw.get("metadata") or {}
        harness = str(meta.get("harness_id") or harness_hint).lower()
        if harness == "claudecode":
            harness = "claude_code"
        task_id = str(meta.get("task_id") or raw.get("benchmark_episode_id") or raw.get("episode_id") or line_no)
        family_key = str(meta.get("semantic_task_family_id") or meta.get("family_key") or task_id)
        messages = raw.get("messages") if isinstance(raw.get("messages"), list) else []
        teacher = meta.get("teacher_id")
        payload_digest = stable_record_id("agentharm-payload", harness, str(raw.get("id") or task_id), raw).rsplit(":", 1)[-1]
        yield {
            "record_id": f"agentharm:{harness}:{payload_digest}",
            "benchmark": "agentharm",
            "harness_name": harness,
            "harness_context": str(meta.get("skillrl_system_instruction") or ""),
            "teacher_model": teacher,
            "teacher_provider": meta.get("teacher_provider"),
            "source_kind": "agentharm_native_trajectory",
            "synthetic": False,
            "collection_status": raw.get("status"),
            "has_observable_turn_content": any(
                m.get("role") in {"user", "assistant", "tool"} and bool(m.get("content") or m.get("tool_calls"))
                for m in messages if isinstance(m, dict)
            ),
            "source_path": relative(path),
            "source_line": line_no,
            "task_id": task_id,
            "family": f"agentharm:{family_key}",
            "task_type": str(meta.get("task_type") or meta.get("scenario") or "unknown"),
            "official_score": meta.get("official_score") or meta.get("score") or meta.get("reward_breakdown") or {},
            "task_metadata": {k: meta.get(k) for k in (
                "benchmark_version", "domain", "scenario", "task_type", "seed", "family_key",
                "semantic_task_family_id", "skillrl_sft_data_source", "safety_module", "scoring_status",
            ) if meta.get(k) is not None},
            "messages": messages,
            "tools": meta.get("available_tools") or raw.get("tools") or [],
            "raw_payload": raw,
        }


def _part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        return str(part.get("text") or part.get("content") or "")
    return ""


def safeclaw_messages(raw: dict[str, Any]) -> list[dict[str, Any]]:
    harness = str(raw.get("harness") or "openclaw")
    messages: list[dict[str, Any]] = [{"role": "system", "content": "\n".join((
        "<harness_context>", f"name: {harness}",
        "interface: OpenClaw native local session", 
        "tool_contract: preserve the native OpenClaw tool-call transcript",
        "</harness_context>",
    ))}]
    transcript = raw.get("session_transcript_raw") or ""
    if not isinstance(transcript, str):
        return messages
    for line in transcript.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "message":
            continue
        message = event.get("message") or {}
        role = str(message.get("role") or "")
        parts = message.get("content") or []
        if isinstance(parts, str):
            parts = [{"type": "text", "text": parts}]
        if role == "toolResult":
            for part in parts if isinstance(parts, list) else []:
                if not isinstance(part, dict):
                    continue
                messages.append({
                    "role": "tool",
                    "tool_call_id": str(part.get("toolCallId") or part.get("tool_call_id") or part.get("id") or ""),
                    "name": str(part.get("name") or "unknown_tool"),
                    "content": str(part.get("text") or part.get("content") or ""),
                })
            continue
        if role not in {"system", "developer", "user", "assistant"}:
            continue
        texts, calls = [], []
        for part in parts if isinstance(parts, list) else []:
            if not isinstance(part, dict):
                texts.append(_part_text(part))
                continue
            kind = part.get("type")
            if kind in {"text", "thinking"}:
                texts.append(str(part.get("text") or ""))
            elif kind == "toolCall":
                calls.append({
                    "id": str(part.get("id") or part.get("toolCallId") or ""),
                    "type": "function",
                    "function": {
                        "name": str(part.get("name") or "unknown_tool"),
                        "arguments": json.dumps(part.get("arguments") or {}, ensure_ascii=False),
                    },
                })
        converted: dict[str, Any] = {"role": role, "content": "\n".join(t for t in texts if t)}
        if calls:
            converted["tool_calls"] = calls
        messages.append(converted)
    return messages


def safeclaw_rows() -> Iterator[dict[str, Any]]:
    paths = sorted(SAFECLAW.glob("*.json")) + sorted((SAFECLAW / "previous_attempts").glob("*.json"))
    task_root = EXP / "vendor/SafeClawArena/tasks"
    task_map = {p.stem: p for p in task_root.glob("*/*.json")} if task_root.is_dir() else {}
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        task_id = str(raw.get("task_id") or path.stem)
        harness = str(raw.get("harness") or "openclaw").lower()
        messages = safeclaw_messages(raw)
        raw_bytes = path.read_bytes()
        digest = sha256_bytes(raw_bytes)
        task = {}
        source_task = task_map.get(task_id)
        if source_task:
            try:
                task = json.loads(source_task.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                task = {}
        metadata = task.get("metadata") or {}
        status = str(raw.get("status") or "unknown")
        yield {
            "record_id": f"safeclawarena:{harness}:{task_id}:{digest[:20]}",
            "benchmark": "safeclawarena",
            "harness_name": harness,
            "harness_context": "\n".join((
                "<harness_context>", f"name: {harness}",
                "interface: OpenClaw native local session", 
                "tool_contract: preserve the native OpenClaw tool-call transcript",
                "</harness_context>",
            )),
            "teacher_model": raw.get("teacher_model"),
            "teacher_provider": "shengsuanyun",
            "source_kind": "safeclawarena_native_trajectory",
            "synthetic": False,
            "collection_status": "completed" if status == "captured" and raw.get("trace_complete") is not False else status,
            "has_observable_turn_content": any(m.get("role") in {"user", "assistant", "tool"} and bool(m.get("content") or m.get("tool_calls")) for m in messages),
            "source_path": relative(path),
            "source_line": None,
            "task_id": task_id,
            "family": f"safeclawarena:{task_id}",
            "task_type": "attack",
            "official_score": {},
            "task_metadata": {k: metadata.get(k) for k in ("dimension", "category", "category_name", "title", "tags", "targeted_component", "invariants_violated") if metadata.get(k) is not None},
            "messages": messages,
            "tools": normalized_tools(messages),
            "raw_payload": raw,
            "source_sha256": digest,
        }


def attempt_rows() -> Iterator[dict[str, Any]]:
    # Failed AgentHarm collection attempts remain in a separate audit ledger;
    # they are not turned into fake conversation trajectories.
    for path in sorted(OUTPUTS.glob(AGENTHARM_ERROR_GLOB)):
        harness = path.parent.name.removeprefix("agentharm_").removesuffix("_native")
        for line_no, raw in jsonl(path):
            record_id = stable_record_id("agentharm-error", harness, str(raw.get("episode_id") or line_no), raw)
            yield {
                "record_id": record_id, "record_type": "collection_attempt", "benchmark": "agentharm",
                "harness_name": "claude_code" if harness == "claudecode" else harness,
                "source_path": relative(path), "source_line": line_no,
                "status": "collection_error", "error": raw.get("error"), "raw_payload": raw,
            }
    # Include every SafeClawArena manifest event, including unsupported fixture
    # outcomes and retries; trajectory files are separately ingested above.
    manifest = SAFECLAW / "manifest.jsonl"
    for line_no, raw in jsonl(manifest):
        task_id = str(raw.get("task_id") or line_no)
        yield {
            "record_id": stable_record_id("safeclawarena-attempt", "openclaw", task_id, raw),
            "record_type": "collection_attempt", "benchmark": "safeclawarena", "harness_name": "openclaw",
            "task_id": task_id, "family": f"safeclawarena:{task_id}",
            "source_path": relative(manifest), "source_line": line_no,
            "status": raw.get("status"), "raw_payload": raw,
        }


def source_files() -> list[Path]:
    paths = list(OUTPUTS.glob(AGENTHARM_GLOB))
    paths += list(OUTPUTS.glob(AGENTHARM_ERROR_GLOB))
    paths += list(SAFECLAW.glob("*.json"))
    paths += list((SAFECLAW / "previous_attempts").glob("*.json"))
    paths += [SAFECLAW / "manifest.jsonl", SAFECLAW / "collector.log"]
    return [p for p in paths if p.is_file()]


def source_fingerprint() -> tuple[tuple[str, int, int], ...]:
    values = []
    for path in source_files():
        stat = path.stat()
        values.append((relative(path), stat.st_size, stat.st_mtime_ns))
    return tuple(sorted(values))


def active_collectors() -> bool:
    try:
        out = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return True  # fail closed; keep syncing if process visibility is unavailable
    for line in out.splitlines():
        if "sync_teacher_trajectory_corpus.py" in line:
            continue
        if "collect_safeclawarena_openclaw_raw.py --phase full" in line:
            return True
        if "experiments/cross_harness_sft" in line and "collect_native.py" in line:
            return True
        if "experiments/cross_harness_sft" in line and "agentharm" in line.lower() and any(x in line.lower() for x in ("collect", "resume", "scheduler")):
            return True
    return False


def scan_known_ids(path: Path) -> tuple[set[str], Counter, Counter, int]:
    ids: set[str] = set()
    by_benchmark, by_harness, by_kind = Counter(), Counter(), Counter()
    count = 0
    if not path.exists():
        return ids, by_benchmark, by_harness, by_kind, count
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed canonical trajectory at line {line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Canonical trajectory line {line_no} is not an object")
            rid = str(row.get("record_id") or "")
            if rid:
                ids.add(rid)
            by_benchmark[str(row.get("benchmark") or "unknown")] += 1
            by_harness[str(row.get("harness_name") or "unknown")] += 1
            by_kind[str(row.get("source_kind") or "unknown")] += 1
            count += 1
            if count % 5000 == 0:
                print(f"Indexed {count:,} existing raw records", flush=True)
    return ids, by_benchmark, by_harness, by_kind, count


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_report(by_benchmark: Counter, by_harness: Counter, by_kind: Counter, total: int,
                  attempts_count: int, attempts_by_source: Counter) -> None:
    report = {
        "dataset": "multi_harness_teacher_v1",
        "purpose": "append-only unfiltered raw corpus; not directly trainable",
        "record_count": total,
        "records_by_benchmark": dict(sorted(by_benchmark.items())),
        "records_by_harness": dict(sorted(by_harness.items())),
        "records_by_source_kind": dict(sorted(by_kind.items())),
        "collection_attempt_count": attempts_count,
        "collection_attempts_by_source": dict(sorted(attempts_by_source.items())),
        "quality_filter_applied": False,
        "required_next_step": "family-disjoint split assignment and unified trajectory quality filtering before SFT conversion",
        "updated_at_utc": utc_now(),
    }
    write_json_atomic(REPORT, report)


def sync_once(ids: set[str], by_benchmark: Counter, by_harness: Counter, by_kind: Counter,
              attempt_ids: set[str], attempt_stats: Counter) -> tuple[int, int]:
    new_rows: list[dict[str, Any]] = []
    rows_by_source = Counter()
    for path in sorted(OUTPUTS.glob(AGENTHARM_GLOB)):
        harness = path.parent.name
        for row in agentharm_rows(path):
            if row["record_id"] in ids:
                continue
            row["source_group"] = harness
            new_rows.append(row)
            ids.add(row["record_id"])
            rows_by_source["agentharm"] += 1
    for row in safeclaw_rows():
        if row["record_id"] in ids:
            continue
        new_rows.append(row)
        ids.add(row["record_id"])
        rows_by_source["safeclawarena"] += 1

    if new_rows:
        with TRAJECTORIES.open("a", encoding="utf-8") as out:
            for row in new_rows:
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                by_benchmark[row["benchmark"]] += 1
                by_harness[row["harness_name"]] += 1
                by_kind[row["source_kind"]] += 1
            out.flush()
            os.fsync(out.fileno())

    new_attempts = []
    for row in attempt_rows():
        if row["record_id"] not in attempt_ids:
            new_attempts.append(row)
            attempt_ids.add(row["record_id"])
    if new_attempts:
        with ATTEMPTS.open("a", encoding="utf-8") as out:
            for row in new_attempts:
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                attempt_stats[row["benchmark"]] += 1
            out.flush()
            os.fsync(out.fileno())

    current_total = sum(by_benchmark.values())
    update_report(by_benchmark, by_harness, by_kind, current_total,
                  sum(attempt_stats.values()), attempt_stats)
    if new_rows or new_attempts:
        print(f"{utc_now()} synced trajectories={len(new_rows)} attempts={len(new_attempts)} "
              f"totals={current_total} raw_attempts={sum(attempt_stats.values())} "
              f"sources={dict(rows_by_source)}", flush=True)
    write_json_atomic(SYNC_REPORT, {
        "updated_at_utc": utc_now(),
        "total_trajectory_records": current_total,
        "new_trajectory_records_this_sync": len(new_rows),
        "new_attempt_records_this_sync": len(new_attempts),
        "trajectory_rows_by_source_this_sync": dict(rows_by_source),
        "attempt_ledger": relative(ATTEMPTS),
        "trajectory_ledger": relative(TRAJECTORIES),
        "unfiltered": True,
    })
    return len(new_rows), len(new_attempts)


def load_attempt_ids(path: Path) -> tuple[set[str], Counter]:
    ids, stats = set(), Counter()
    if path.exists():
        for _, row in jsonl(path):
            rid = str(row.get("record_id") or "")
            if rid:
                ids.add(rid)
            stats[str(row.get("benchmark") or "unknown")] += 1
    return ids, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Continue syncing active collectors")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--idle-exit-seconds", type=int, default=600,
                        help="In watch mode, exit after no source changes and no active collectors for this long")
    args = parser.parse_args()
    if args.interval < 5 or args.idle_exit_seconds < 60:
        parser.error("interval must be >=5s and idle-exit-seconds >=60s")
    RAW.mkdir(parents=True, exist_ok=True)
    if not TRAJECTORIES.exists():
        TRAJECTORIES.touch()

    with LOCK.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another raw corpus sync process is already running")
        print(f"{utc_now()} indexing existing unfiltered corpus", flush=True)
        ids, by_benchmark, by_harness, by_kind, total = scan_known_ids(TRAJECTORIES)
        attempt_ids, attempt_stats = load_attempt_ids(ATTEMPTS)
        print(f"Indexed {total:,} trajectories across {len(ids):,} record IDs", flush=True)
        last_fingerprint = None
        idle_since = None
        while True:
            before = source_fingerprint()
            added, attempts_added = sync_once(ids, by_benchmark, by_harness, by_kind, attempt_ids, attempt_stats)
            after = source_fingerprint()
            collectors_running = active_collectors()
            changed = before != last_fingerprint or after != before or added > 0 or attempts_added > 0
            if changed or collectors_running:
                idle_since = None
            elif idle_since is None:
                idle_since = time.monotonic()
            last_fingerprint = after
            print(f"{utc_now()} watch heartbeat active_collectors={collectors_running} "
                  f"new_rows={added} new_attempts={attempts_added}", flush=True)
            if not args.watch:
                break
            if idle_since is not None and time.monotonic() - idle_since >= args.idle_exit_seconds:
                print(f"{utc_now()} all tracked collectors are idle; sync watcher exiting", flush=True)
                break
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
