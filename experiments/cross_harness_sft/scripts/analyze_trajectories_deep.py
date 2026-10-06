"""Deep analysis of trajectory structure for synthesis."""
import json, statistics, hashlib, random
from pathlib import Path
from collections import Counter

base = Path(r"E:\AI Agent\harnessSafety\cross_harness_alignment\experiments\cross_harness_sft\outputs")

for name, sub in [("claude","claude_agentdojo_openrouter_gpt56_sol_full"), ("codex","codex_agentdojo_native_gpt56_sol_full")]:
    p = base / sub / "raw_native.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"{name.upper()} DEEP ANALYSIS ({len(rows)} trajectories)")
    print(sep)
    
    # Tool calls from messages (both formats)
    tool_calls_per_traj = []
    all_tool_names = Counter()
    tool_sequences = []  # list of tool name sequences per trajectory
    
    for r in rows:
        tc_count = 0
        tc_seq = []
        for msg in r.get("messages", []):
            if not isinstance(msg, dict):
                continue
            # Check OpenAI-style tool_calls field
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {}).get("name", "")
                if fn:
                    tc_count += 1
                    all_tool_names[fn] += 1
                    tc_seq.append(fn)
            # Check Anthropic-style content blocks
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        fn = block.get("name", "")
                        if fn:
                            tc_count += 1
                            all_tool_names[fn] += 1
                            tc_seq.append(fn)
        tool_calls_per_traj.append(tc_count)
        if tc_seq:
            tool_sequences.append(tc_seq)
    
    print(f"tool_calls/traj: mean={statistics.mean(tool_calls_per_traj):.1f} median={statistics.median(tool_calls_per_traj):.0f} min={min(tool_calls_per_traj)} max={max(tool_calls_per_traj)}")
    zero_tc = sum(1 for x in tool_calls_per_traj if x == 0)
    print(f"  zero-tool trajectories: {zero_tc}/{len(rows)} ({100*zero_tc/len(rows):.1f}%)")
    print(f"top 20 tools: {all_tool_names.most_common(20)}")
    
    # Tool sequence length distribution
    seq_lens = [len(s) for s in tool_sequences]
    if seq_lens:
        print(f"tool sequence lengths: mean={statistics.mean(seq_lens):.1f} median={statistics.median(seq_lens):.0f}")
    
    # Domain distribution
    domains = Counter(r.get("metadata",{}).get("domain","?") for r in rows)
    print(f"domains: {dict(domains)}")
    
    # Task type distribution  
    task_types = Counter(r.get("metadata",{}).get("task_type","?") for r in rows)
    print(f"task_types: {dict(task_types)}")
    
    # Scenario distribution
    scenarios = Counter(r.get("metadata",{}).get("scenario","?") for r in rows)
    print(f"scenarios: {dict(scenarios)}")
    
    # Duration stats
    durations = [r.get("metadata",{}).get("native_duration_seconds",0) for r in rows]
    durations = [d for d in durations if d and d > 0]
    if durations:
        print(f"duration(s): mean={statistics.mean(durations):.1f} median={statistics.median(durations):.1f} min={min(durations):.1f} max={max(durations):.1f}")
    
    # Usage stats
    input_tokens = []
    output_tokens = []
    for r in rows:
        usage = r.get("metadata",{}).get("native_usage",{})
        if isinstance(usage, dict):
            it = usage.get("input_tokens",0) or 0
            ot = usage.get("output_tokens",0) or 0
            if it: input_tokens.append(it)
            if ot: output_tokens.append(ot)
    if input_tokens:
        print(f"input_tokens: mean={statistics.mean(input_tokens):.0f} median={statistics.median(input_tokens):.0f}")
    if output_tokens:
        print(f"output_tokens: mean={statistics.mean(output_tokens):.0f} median={statistics.median(output_tokens):.0f}")
    
    # Visible reasoning length
    vr_lens = [len(r.get("metadata",{}).get("visible_reasoning","") or "") for r in rows]
    print(f"visible_reasoning len: mean={statistics.mean(vr_lens):.0f} median={statistics.median(vr_lens):.0f}")
    
    # Reward breakdown
    rewards = Counter()
    for r in rows:
        rb = r.get("metadata",{}).get("reward_breakdown",{})
        if isinstance(rb, dict):
            for k,v in rb.items():
                rewards[f"{k}={v}"] += 1
    print(f"reward_breakdown samples: {dict(list(rewards.items())[:10])}")
    
    # Native command patterns
    cmds = Counter()
    for r in rows:
        cmd = r.get("metadata",{}).get("native_command","")
        if isinstance(cmd, list):
            cmds[cmd[0] if cmd else "?"] += 1
        elif isinstance(cmd, str):
            cmds[cmd.split()[0] if cmd else "?"] += 1
    print(f"native_command binaries: {dict(cmds)}")
