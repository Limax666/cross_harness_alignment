#!/usr/bin/env python3
"""Run a real Qwen base or SFT model through the official AgentDojo verifier.

Both student roles use the same deterministic tool loop, prompts, cases, seed,
generation settings and verifier. This runner is intentionally identified as a
local Qwen tool loop; it is not Codex or Claude Code.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx


CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FUNCTION_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)


def parse_value(value: str) -> Any:
    value = html.unescape(value.strip())
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block in CALL_RE.findall(text):
        match = FUNCTION_RE.search(block)
        if match:
            args = {name: parse_value(value) for name, value in PARAMETER_RE.findall(match.group(2))}
            calls.append({"name": match.group(1).strip(), "arguments": args})
            continue
        try:
            value = json.loads(block.strip())
        except json.JSONDecodeError as exc:
            raise ValueError("malformed tool-call block") from exc
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise ValueError("tool-call block lacks a function name")
        args = value.get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict):
            raise ValueError("tool-call arguments are not an object")
        calls.append({"name": value["name"], "arguments": args})
    if "<tool_call>" in text and not calls:
        raise ValueError("unterminated tool-call block")
    return calls


def visible_text(text: str) -> str:
    text = CALL_RE.sub("", text)
    # Qwen3.5's generation prompt ends inside <think>; generated tokens often
    # contain only its closing tag, without a matching opening tag.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    return text.strip()


def generate(model: Any, tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             max_context: int, max_new_tokens: int) -> tuple[str, int]:
    import torch

    encoded = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                                            add_generation_prompt=True, return_tensors="pt")
    if isinstance(encoded, Mapping):
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
    else:
        input_ids, attention_mask = encoded, None
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    token_count = int(input_ids.shape[-1])
    if token_count > max_context:
        raise ValueError(f"context_overflow:{token_count}>{max_context}")
    device = next(model.parameters()).device
    inputs: dict[str, Any] = {"input_ids": input_ids.to(device)}
    if attention_mask is not None:
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        inputs["attention_mask"] = attention_mask.to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                use_cache=True)
    decoded = tokenizer.decode(output[0, token_count:], skip_special_tokens=False)
    for stop in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        decoded = decoded.split(stop, 1)[0]
    return decoded.strip(), token_count


def run_case(client: httpx.Client, base_url: str, planned: dict[str, Any], model: Any,
             tokenizer: Any, case: dict[str, Any], max_context: int, max_new_tokens: int,
             max_turns: int) -> dict[str, Any]:
    task_id = planned["task_id"]
    start = client.post(f"{base_url}/episodes/start", json={"task_id": task_id, "seed": planned["seed"]})
    start.raise_for_status()
    started = start.json()
    episode_id = started["episode_id"]
    tools = started["tools"]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": planned["system_prompt"]},
        {"role": "user", "content": planned["user_query"]},
    ]
    generated_turns: list[dict[str, Any]] = []
    tool_history: list[dict[str, Any]] = []
    final_answer = ""
    ended_reason = "max_turns"
    for turn in range(max_turns):
        raw, input_tokens = generate(model, tokenizer, messages, tools, max_context, max_new_tokens)
        turn_record: dict[str, Any] = {"turn": turn + 1, "input_tokens": input_tokens, "raw_generation": raw}
        generated_turns.append(turn_record)
        try:
            calls = parse_tool_calls(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            turn_record["parse_error"] = str(exc)
            final_answer = visible_text(raw)
            ended_reason = "invalid_tool_call"
            break
        if not calls:
            final_answer = visible_text(raw)
            ended_reason = "final_answer" if final_answer else "empty_answer"
            break
        assistant_calls = []
        for index, call in enumerate(calls):
            assistant_calls.append({"id": f"turn_{turn + 1}_call_{index + 1}", "type": "function",
                                    "function": {"name": call["name"], "arguments": call["arguments"]}})
        messages.append({"role": "assistant", "content": visible_text(raw), "tool_calls": assistant_calls})
        for call, shaped in zip(calls, assistant_calls):
            response = client.post(f"{base_url}/episodes/{episode_id}/tool", json=call)
            response.raise_for_status()
            result = response.json()
            tool_history.append({"name": call["name"], "arguments": call["arguments"],
                                 "observation": result.get("observation"),
                                 "invalid_call": bool(result.get("invalid_call")),
                                 "risk_event": bool(result.get("risk_event"))})
            messages.append({"role": "tool", "tool_call_id": shaped["id"], "name": call["name"],
                             "content": str(result.get("observation") or "")})
    finish = client.post(f"{base_url}/episodes/{episode_id}/finish", json={"final_answer": final_answer})
    finish.raise_for_status()
    verdict = finish.json()
    if not {"utility_reward", "safety_reward", "final_success", "risk_success"} <= verdict.keys():
        raise ValueError("official verifier returned incomplete scores")
    return {
        "schema": "agentdojo_9b_comparison.episode.v1", "task_id": task_id,
        "family": planned["family"], "task_type": planned["task_type"], "seed": planned["seed"],
        "status": "completed", "ended_reason": ended_reason, "final_answer": final_answer,
        "generated_turns": generated_turns, "tool_history": tool_history,
        "reward_breakdown": {"utility_reward": float(verdict["utility_reward"]),
                             "safety_reward": float(verdict["safety_reward"])},
        "official_verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("base", "sft"), required=True)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--benchmark-url", default="http://127.0.0.1:8111")
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real Qwen3.5-9B evaluation")
    if args.role == "sft" and not (args.checkpoint / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(args.checkpoint / "adapter_model.safetensors")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.role}.jsonl"
    errors = args.output_dir / f"{args.role}.errors.jsonl"
    plan = [json.loads(line) for line in args.plan.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = {json.loads(line)["task_id"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()} if output.exists() else set()
    base_url = args.benchmark_url.rstrip("/")
    with httpx.Client(timeout=120, trust_env=False) as client:
        health = client.get(f"{base_url}/health"); health.raise_for_status()
        version = health.json().get("version")
        if {row["benchmark_version"] for row in plan} != {version}:
            raise ValueError(f"benchmark version mismatch: worker={version}")
        cases_response = client.get(f"{base_url}/cases"); cases_response.raise_for_status()
        cases = {row["task_id"]: row for row in cases_response.json()["cases"]}
    for row in plan:
        case = cases.get(row["task_id"])
        if not case or case["semantic_task_family_id"] != row["family"] or case["user_query"] != row["user_query"]:
            raise ValueError(f"benchmark case mismatch: {row['task_id']}")
    print(json.dumps({"role": args.role, "planned": len(plan), "completed_before_start": len(completed),
                      "worker_version": version, "gpu": torch.cuda.get_device_name(0)}, ensure_ascii=False), flush=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype, use_kernels=False,
                                                 quantization_config=quantization, device_map={"": 0})
    if args.role == "sft":
        model = PeftModel.from_pretrained(model, str(args.checkpoint))
    model.eval()
    print(json.dumps({"role": args.role, "model_loaded": True}), flush=True)

    with httpx.Client(timeout=120, trust_env=False) as client:
        for number, planned in enumerate(plan, 1):
            task_id = planned["task_id"]
            if task_id in completed:
                continue
            print(f"[{args.role}] {number}/{len(plan)} start {task_id}", flush=True)
            try:
                row = run_case(client, base_url, planned, model, tokenizer, cases[task_id],
                               args.max_context, args.max_new_tokens, args.max_turns)
                row["role"] = args.role
                row["base_model"] = args.base_model
                row["checkpoint"] = str(args.checkpoint) if args.role == "sft" else None
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush(); os.fsync(handle.fileno())
                completed.add(task_id)
                print(f"[{args.role}] {number}/{len(plan)} done {task_id} "
                      f"utility={row['reward_breakdown']['utility_reward']} "
                      f"safety={row['reward_breakdown']['safety_reward']} "
                      f"tools={len(row['tool_history'])}", flush=True)
            except Exception as exc:
                error = {"role": args.role, "task_id": task_id, "error": str(exc), "traceback": traceback.format_exc()}
                with errors.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(error, ensure_ascii=False) + "\n")
                print(f"[{args.role}] {number}/{len(plan)} error {task_id}: {type(exc).__name__}: {exc}", flush=True)
    print(json.dumps({"role": args.role, "completed": len(completed), "planned": len(plan)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
