"""Analyze trajectory statistics for both harness corpora."""
import json, statistics, sys
from pathlib import Path
from collections import Counter

base = Path(r"E:\AI Agent\harnessSafety\cross_harness_alignment\experiments\cross_harness_sft\outputs")

for name, sub in [("claude","claude_agentdojo_openrouter_gpt56_sol_full"), ("codex","codex_agentdojo_native_gpt56_sol_full")]:
    p = base / sub / "raw_native.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"{name.upper()} ({len(rows)} trajectories)")
    print(sep)
    
    resp_lens = [r.get("response_length",0) for r in rows]
    msg_counts = [len(r.get("messages",[])) for r in rows]
    event_counts = [len(r.get("native_events",[])) for r in rows]
    
    print(f"response_length: mean={statistics.mean(resp_lens):.0f} median={statistics.median(resp_lens):.0f} min={min(resp_lens)} max={max(resp_lens)}")
    print(f"messages count:  mean={statistics.mean(msg_counts):.1f} median={statistics.median(msg_counts):.0f} min={min(msg_counts)} max={max(msg_counts)}")
    print(f"native_events:   mean={statistics.mean(event_counts):.1f} median={statistics.median(event_counts):.0f} min={min(event_counts)} max={max(event_counts)}")
    
    statuses = Counter(r.get("status","?") for r in rows)
    print(f"status: {dict(statuses)}")
    
    benchmarks = Counter()
    tasks = Counter()
    scenarios = Counter()
    for r in rows:
        meta = r.get("metadata",{})
        benchmarks[meta.get("benchmark_id","?")] += 1
        tasks[meta.get("task_id","?")] += 1
        scenarios[meta.get("scenario_type","?")] += 1
    print(f"benchmarks: {dict(benchmarks)}")
    print(f"scenarios: {dict(scenarios)}")
    print(f"unique tasks: {len(tasks)}, top 10: {tasks.most_common(10)}")
    
    all_meta_keys = set()
    for r in rows:
        all_meta_keys.update(r.get("metadata",{}).keys())
    print(f"metadata keys ({len(all_meta_keys)}): {sorted(all_meta_keys)}")
    
    tool_names = Counter()
    tool_call_counts = []
    for r in rows:
        tc = 0
        for msg in r.get("messages",[]):
            if isinstance(msg, dict):
                content = msg.get("content",[])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_names[block.get("name","")] += 1
                            tc += 1
        tool_call_counts.append(tc)
    print(f"tool_calls per traj: mean={statistics.mean(tool_call_counts):.1f} median={statistics.median(tool_call_counts):.0f} min={min(tool_call_counts)} max={max(tool_call_counts)}")
    print(f"top 15 tools: {tool_names.most_common(15)}")
    
    roles = Counter()
    for r in rows:
        for msg in r.get("messages",[]):
            if isinstance(msg, dict):
                roles[msg.get("role","?")] += 1
    print(f"message roles: {dict(roles)}")
    
    # Sample one full row structure
    print(f"\n--- Sample row (first) ---")
    sample = rows[0]
    print(json.dumps(sample, indent=2, ensure_ascii=False)[:3000])
