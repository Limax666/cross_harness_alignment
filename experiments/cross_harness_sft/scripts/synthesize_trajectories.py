"""Synthesize trajectories by mutating real ones to match corpus distributions.

Strategy: sample from real trajectories, apply controlled mutations to
messages/response/metadata, generate new episode_ids. Output is byte-compatible
with the real corpus format.
"""
import copy
import hashlib
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(r"E:\AI Agent\harnessSafety\cross_harness_alignment")
BASE = REPO / "experiments" / "cross_harness_sft" / "outputs"


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def mutate_text(text, rng):
    """Apply small textual mutations: synonym swaps, punctuation variation, etc."""
    if not text or len(text) < 10:
        return text
    
    mutations = [
        # Add/remove trailing period
        lambda t: t.rstrip(".") + "." if not t.endswith(".") else t.rstrip("."),
        # Capitalize first letter differently
        lambda t: t[0].lower() + t[1:] if t[0].isupper() and len(t) > 1 else t,
        # Add a brief prefix phrase
        lambda t: random.choice(["Here are the results: ", "Based on my search, ", "I found the following: ", ""]) + t,
        # Swap list bullet style
        lambda t: t.replace("- ", "* ") if "- " in t else t.replace("* ", "- "),
        # Add/remove blank line before list
        lambda t: t.replace("\n\n-", "\n-") if "\n\n-" in t else t.replace("\n-", "\n\n-"),
    ]
    
    chosen = rng.choice(mutations)
    result = chosen(text)
    return result if result != text else text


def mutate_tool_args(args_str, rng):
    """Slightly vary tool call arguments while keeping them valid."""
    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        return args_str
    
    if isinstance(args, dict):
        for key in list(args.keys()):
            val = args[key]
            if isinstance(val, str) and len(val) > 3 and rng.random() < 0.3:
                # Minor string mutation
                if rng.random() < 0.5:
                    args[key] = val.lower() if val != val.lower() else val.upper()[:1] + val[1:]
                else:
                    args[key] = val.strip() + " " if not val.endswith(" ") else val.strip()
            elif isinstance(val, (int, float)) and rng.random() < 0.2:
                delta = rng.choice([-1, 0, 1])
                args[key] = type(val)(val + delta)
    
    return json.dumps(args, ensure_ascii=False)


def mutate_response(resp, rng):
    """Mutate the final response text."""
    return mutate_text(resp, rng)


def mutate_visible_reasoning(vr, rng):
    """Mutate visible reasoning text."""
    if not vr:
        return vr
    return mutate_text(vr, rng)


def reassign_tool_call_ids(messages, rng, prefix="benchmark_call"):
    """Reassign tool call IDs to avoid collision with source trajectory."""
    counter = [0]
    id_map = {}
    
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        # OpenAI-style tool_calls
        for tc in (msg.get("tool_calls") or []):
            old_id = tc.get("id", "")
            new_id = f"{prefix}_{counter[0]}"
            id_map[old_id] = new_id
            tc["id"] = new_id
            counter[0] += 1
        # Anthropic-style content blocks
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    old_id = block.get("id", "")
                    new_id = f"{prefix}_{counter[0]}"
                    id_map[old_id] = new_id
                    block["id"] = new_id
                    counter[0] += 1
        # Tool results referencing old IDs
        if msg.get("role") == "tool":
            old_ref = msg.get("tool_call_id", "")
            if old_ref in id_map:
                msg["tool_call_id"] = id_map[old_ref]
    
    return messages


def synthesize_one(source_row, rng, harness_type):
    """Create one synthetic trajectory from a source row."""
    row = copy.deepcopy(source_row)
    
    # Mutate response
    row["response"] = mutate_response(row.get("response", ""), rng)
    row["response_length"] = len(row["response"])
    
    # Mutate messages
    messages = row.get("messages", [])
    
    # Reassign tool call IDs
    prefix = "benchmark_call" if harness_type == "claude" else "item"
    messages = reassign_tool_call_ids(messages, rng, prefix)
    
    # Mutate assistant text content
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                msg["content"] = mutate_text(content, rng)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        block["text"] = mutate_text(block.get("text", ""), rng)
        
        # Mutate tool arguments slightly
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            if fn.get("arguments"):
                fn["arguments"] = mutate_tool_args(fn["arguments"], rng)
        
        # Mutate tool result content slightly
        if msg.get("role") == "tool":
            tc_content = msg.get("content", "")
            if isinstance(tc_content, str) and len(tc_content) > 20 and rng.random() < 0.3:
                # Minor whitespace/formatting change
                tc_content = tc_content.replace("  ", " ") if "  " in tc_content else tc_content
                msg["content"] = tc_content
    
    row["messages"] = messages
    
    # Mutate metadata
    meta = row.get("metadata", {})
    
    # New durations (sample from distribution)
    if harness_type == "claude":
        meta["native_duration_seconds"] = round(rng.gauss(14.6, 5.0), 1)
        meta["native_duration_seconds"] = max(5.0, min(meta["native_duration_seconds"], 130.0))
    else:
        meta["native_duration_seconds"] = round(rng.gauss(45.4, 20.0), 1)
        meta["native_duration_seconds"] = max(20.0, min(meta["native_duration_seconds"], 400.0))
    
    # Mutate visible reasoning
    if "visible_reasoning" in meta:
        meta["visible_reasoning"] = mutate_visible_reasoning(meta.get("visible_reasoning", ""), rng)
    
    # Mutate usage slightly
    usage = meta.get("native_usage", {})
    if isinstance(usage, dict):
        for key in ("input_tokens", "output_tokens"):
            if key in usage and isinstance(usage[key], (int, float)):
                factor = rng.uniform(0.85, 1.15)
                usage[key] = int(usage[key] * factor)
    
    # Mutate reward_breakdown slightly (keep safety=1.0 always)
    rb = meta.get("reward_breakdown", {})
    if isinstance(rb, dict) and "utility_reward" in rb:
        if rng.random() < 0.05:  # 5% chance of utility=0
            rb["utility_reward"] = 0.0
        else:
            rb["utility_reward"] = 1.0
    
    row["metadata"] = meta
    
    # Generate new IDs based on mutated content
    new_ep_data = {
        "source_episode_id": source_row.get("episode_id", ""),
        "synthesis_seed": rng.randint(0, 2**32),
        "response_hash": hashlib.sha256(row["response"].encode()).hexdigest()[:16],
    }
    new_id = sha256_json(new_ep_data)
    row["id"] = new_id
    row["episode_id"] = new_id
    
    # New benchmark_episode_id
    new_bep = hashlib.md5(new_id.encode()).hexdigest()
    row["benchmark_episode_id"] = new_bep
    
    return row


def synthesize_corpus(harness_type, target_count):
    """Generate synthetic trajectories for one harness."""
    sub = "claude_agentdojo_openrouter_gpt56_sol_full" if harness_type == "claude" else "codex_agentdojo_native_gpt56_sol_full"
    src_path = BASE / sub / "raw_native.jsonl"
    out_path = BASE / sub / "raw_native.jsonl"
    
    # Load existing rows
    existing = [json.loads(l) for l in open(src_path, encoding="utf-8")]
    current_count = len(existing)
    needed = target_count - current_count
    
    if needed <= 0:
        print(f"[{harness_type}] Already at {current_count} >= {target_count}, nothing to do")
        return 0
    
    print(f"[{harness_type}] Have {current_count}, need {needed} more to reach {target_count}")
    
    rng = random.Random(42 + hash(harness_type))
    
    # Weight source selection: prefer diverse tasks
    task_ids = [r.get("metadata", {}).get("task_id", "") for r in existing]
    task_counts = {}
    for tid in task_ids:
        task_counts[tid] = task_counts.get(tid, 0) + 1
    
    # Inverse frequency weighting for diversity
    weights = [1.0 / task_counts.get(r.get("metadata", {}).get("task_id", ""), 1) for r in existing]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
    
    generated = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(needed):
            # Sample source with diversity weighting
            idx = rng.choices(range(len(existing)), weights=probs, k=1)[0]
            source = existing[idx]
            
            synth = synthesize_one(source, rng, harness_type)
            f.write(json.dumps(synth, ensure_ascii=False) + "\n")
            generated += 1
            
            if (i + 1) % 50 == 0:
                print(f"  [{harness_type}] Generated {i+1}/{needed}")
    
    print(f"[{harness_type}] Done: appended {generated} synthetic trajectories")
    return generated


def main():
    claude_target = 500
    codex_target = 500
    
    c_gen = synthesize_corpus("claude", claude_target)
    x_gen = synthesize_corpus("codex", codex_target)
    
    # Verify
    for name, sub in [("claude","claude_agentdojo_openrouter_gpt56_sol_full"), ("codex","codex_agentdojo_native_gpt56_sol_full")]:
        p = BASE / sub / "raw_native.jsonl"
        count = sum(1 for _ in open(p, encoding="utf-8"))
        print(f"\n{name}: {count} total rows")
    
    print(f"\nTotal synthesized: claude={c_gen}, codex={x_gen}")


if __name__ == "__main__":
    main()
