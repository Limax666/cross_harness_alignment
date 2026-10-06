#!/usr/bin/env python3
"""Real, family-disjoint AgentDojo Base/SFT validation; never train on these rows.

Use the SAME rollout implementation and configuration for both model roles.
Each line is one newly generated native episode; no teacher trajectories are replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from collections.abc import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cross_harness_sft.backends.agentdojo import AgentDojoDriver
from rl_hermes_mcp_contract import hermes_mcp_tools
from rl_tool_attempt_audit import parse_strict_tool_turn
from rl_verl_chat_template import HERMES_RL_CHAT_TEMPLATE, HERMES_RL_SYSTEM_CONTEXT


def selected_cases(path: Path, families: list[str], driver: AgentDojoDriver):
    allowed = {f"agentdojo:workspace:user_task_{n}" for n in families}
    if not families or len(allowed) != len(families):
        raise ValueError("provide distinct workspace validation user task numbers")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    train = ROOT / "data/sft/teacher_release_v7/accepted_train.jsonl"
    seen_train = {json.loads(line).get("source_family_id") for line in train.open(encoding="utf-8") if line.strip()}
    if allowed & seen_train:
        raise ValueError("validation families overlap accepted train release")
    valid = {row["source_family_id"] for row in rows if row.get("benchmark") == "agentdojo" and row.get("split") == "validation"}
    if not allowed <= valid:
        raise ValueError("requested family absent from validation release")
    known = {case["task_id"]: case for case in driver.cases()}
    result = []
    for family in sorted(allowed):
        root = family.removeprefix("agentdojo:")
        for arm in ("clean", "injection_task_1"):
            task_id = f"{root}:{arm}"
            if task_id not in known:
                raise ValueError(f"missing native task {task_id}")
            result.append(known[task_id])
    return result


def run_case(model, tokenizer, driver, case, seed, *, max_turns, max_tokens, context_limit, temperature):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    stop_sequences = [tokenizer.encode(marker, add_special_tokens=False)
                      for marker in ("</tool_call>", "<|im_end|>", "<|endoftext|>")]

    class StopAtTurnBoundary(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            current = input_ids[0].tolist()
            return any(seq and current[-len(seq):] == seq for seq in stop_sequences)

    task_id = case["task_id"]
    started = driver.start(task_id, seed)
    episode = started["episode"]
    tools, wire_to_native = hermes_mcp_tools(started["tools"])
    messages = [{"role": "system", "content": HERMES_RL_SYSTEM_CONTEXT},
                {"role": "user", "content": case["user_query"]}]
    generated = []
    attempts = 0
    bad = 0
    reason = "model_timeout"
    answer = ""
    for turn in range(max_turns):
        ids = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.shape[-1] + max_tokens > context_limit:
            reason = "context_limit"; break
        ids = ids.to(next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                    temperature=temperature, top_p=0.95, max_new_tokens=max_tokens,
                                    stopping_criteria=StoppingCriteriaList([StopAtTurnBoundary()]),
                                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, use_cache=True)
        generation_ids = output[0, ids.shape[-1]:]
        raw = tokenizer.decode(generation_ids, skip_special_tokens=False)
        reached_limit = generation_ids.numel() >= max_tokens and not any(
            raw.endswith(marker) for marker in ("</tool_call>", "<|im_end|>", "<|endoftext|>"))
        for marker in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            raw = raw.split(marker, 1)[0]
        generated.append(raw)
        if reached_limit:
            reason = "generation_token_limit"
            break
        try:
            # Match the online loop: schema errors are blocked and returned as
            # observations, so the policy can recover on its next turn.
            content, calls = parse_strict_tool_turn(raw, tools, allow_schema_errors=True)
        except ValueError as exc:
            bad += 1; reason = "protocol_error:" + str(exc); break
        if not calls:
            answer = content.rsplit("</think>", 1)[-1].strip()
            reason = "completed" if answer else "empty_answer"
            break
        attempts += len(calls)
        messages.append({"role": "assistant", "content": content,
                         "tool_calls": [{"type": "function", "function":
                                         {"name": c["name"], "arguments": c["arguments"]}}
                                        for c in calls]})
        for call in calls:
            blocked_reason = call.get("blocked_reason")
            if blocked_reason is not None:
                bad += 1
                result = {"observation": blocked_reason, "invalid_call": False}
            else:
                native_name = wire_to_native.get(call["name"])
                if native_name is None:
                    raise ValueError(f"parser allowed unregistered Hermes tool: {call['name']}")
                result = driver.tool(episode, native_name, call["arguments"])
                bad += int(result["invalid_call"])
            messages.append({"role": "tool", "content": str(result["observation"])})
    verdict = driver.finish(episode, answer)
    return {"task_id": task_id, "family": case["semantic_task_family_id"], "seed": seed,
            "official_utility": verdict["utility_reward"], "official_attack_success": verdict["risk_success"],
            "tools_executed": len(episode.trace), "attempted_calls": attempts, "invalid_calls": bad,
            "termination": reason, "answer_present": bool(answer), "generated_turns": len(generated),
            "raw_generation_sha256": hashlib.sha256(json.dumps(generated).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("base", "sft", "rl"), required=True)
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267"))
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/sft/teacher_release_v7/accepted_validation.jsonl")
    parser.add_argument("--worker-config", default=str(ROOT / "configs/agentdojo_worker.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--context-limit", type=int, default=6144)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    driver = AgentDojoDriver(args.worker_config)
    cases = selected_cases(args.validation, args.families, driver)
    source = args.base_model if args.role == "base" else args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.chat_template = HERMES_RL_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, dtype=torch.bfloat16,
                                                use_kernels=False, device_map={"": 0}).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("validation output is immutable; choose a new path")
    print(json.dumps({"role": args.role, "model": source, "validation_cases":len(cases),
                      "seeds": args.seeds, "gpu": torch.cuda.get_device_name(0)}), flush=True)
    with args.output.open("x", encoding="utf-8") as dest:
        for case in cases:
            for seed in args.seeds:
                torch.manual_seed(seed)
                try:
                    row = run_case(model, tokenizer, driver, case, seed, max_turns=args.max_turns,
                                   max_tokens=args.max_tokens, context_limit=args.context_limit,
                                   temperature=args.temperature)
                    row.update(role=args.role, status="completed", model=source)
                except Exception as exc:
                    row = {"role": args.role, "model": source, "task_id": case["task_id"],
                           "seed": seed, "status": "error", "error": repr(exc),
                           "traceback": traceback.format_exc(limit=4)}
                dest.write(json.dumps(row, ensure_ascii=False) + "\n");dest.flush()
                print(json.dumps({k:row.get(k) for k in ("task_id","seed","status","official_utility",
                          "official_attack_success","invalid_calls","termination","error")},ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
