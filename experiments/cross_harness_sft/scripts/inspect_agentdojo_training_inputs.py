"""Read-only inventory of native AgentDojo teacher trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        statuses: Counter[str] = Counter()
        types: Counter[str] = Counter()
        rewards: Counter[str] = Counter()
        keys: Counter[str] = Counter()
        checks: Counter[str] = Counter()
        identities: Counter[str] = Counter()
        first = None
        count = 0
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            count += 1
            row = json.loads(line)
            meta = row.get("metadata") or {}
            keys.update(meta.keys())
            statuses[str(row.get("status"))] += 1
            types[str(meta.get("task_type"))] += 1
            rewards[str(meta.get("reward_breakdown"))] += 1
            metrics = meta.get("env_metrics") or {}
            checks["safety_one"] += (meta.get("reward_breakdown") or {}).get("safety_reward") == 1.0
            checks["utility_one"] += (meta.get("reward_breakdown") or {}).get("utility_reward") == 1.0
            checks["final_success"] += metrics.get("final_success") is True
            checks["nonempty_reasoning"] += bool(str(meta.get("visible_reasoning") or "").strip())
            checks["nonempty_tools"] += bool(meta.get("available_tools"))
            checks["nonempty_history"] += bool(meta.get("tool_history"))
            checks["exact_context"] += bool(meta.get("skillrl_system_instruction") and meta.get("skillrl_user_content"))
            checks["think_tags"] += "<think>" in str((row.get("messages") or [{}])[-1].get("content") or "")
            checks["native"] += meta.get("native_harness") is True
            checks["no_invalid_calls"] += int(metrics.get("invalid_tool_call_count") or 0) == 0
            identities[f"{meta.get('harness_id')}|{meta.get('teacher_id')}|{meta.get('benchmark_id')}"] += 1
            if first is None:
                first = {
                    "top_keys": sorted(row),
                    "metadata_keys": sorted(meta),
                    "message_roles": [message.get("role") for message in (row.get("messages") or [])[:12]],
                    "episode_id": row.get("episode_id"),
                    "reward_breakdown": meta.get("reward_breakdown"),
                    "env_metrics": meta.get("env_metrics"),
                }
        print(json.dumps({
            "path": str(path), "rows": count, "status": dict(statuses),
            "task_type": dict(types), "reward_breakdown": dict(rewards),
            "metadata_key_counts": dict(keys), "checks": dict(checks), "identities": dict(identities), "first": first,
        }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
