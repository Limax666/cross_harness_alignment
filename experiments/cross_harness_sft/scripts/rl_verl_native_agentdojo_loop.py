"""VeRL current-policy AgentDojo loop for the four train-family pilot cells.

No teacher data, textual-error tool fallback, or synthetic reward. The caller must
supply a training batch and install a real VeRL inference backend; this module
alone is not a training launcher. Unknown evidence aborts the entire rollout.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput

from rl_native_agentdojo_episode import NativeAgentDojoEpisode
from rl_online_batch_gate import CELLS
from rl_tool_attempt_audit import parse_strict_tool_turn
from rl_verl_chat_template import HERMES_RL_CHAT_TEMPLATE, HERMES_RL_SYSTEM_CONTEXT


class NativeAgentDojoVeRLLoop(AgentLoopBase):
    def __init__(self, *args, worker_config: str, max_turns: int = 6, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        if tokenizer is None or not getattr(tokenizer, "chat_template", None):
            raise ValueError("native pilot requires a tokenizer with a chat template")
        tokenizer.chat_template = HERMES_RL_CHAT_TEMPLATE
        super().__init__(*args, **kwargs)
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        config_path = Path(worker_config).expanduser()
        if not config_path.is_absolute():
            config_path = Path(__file__).resolve().parents[1] / config_path
        self.worker_config = str(config_path.resolve())
        self.max_turns = max_turns

    async def run(self, sampling_params: dict, priority: int = 0, **kwargs) -> AgentLoopOutput:
        from cross_harness_sft.backends.agentdojo import AgentDojoDriver

        task_id = str(kwargs["task_id"])
        # VeRL repeats each prompt n times for GRPO and repeats every dataset field,
        # including seed; derive the environment seed per *live* sampled replica.
        rollout_nonce = uuid4()
        seed = rollout_nonce.int % (2**31 - 1)
        if task_id not in CELLS or kwargs.get("split") != "train":
            raise ValueError("only registered train-family pilot tasks are allowed")
        driver = AgentDojoDriver(self.worker_config)
        episode = NativeAgentDojoEpisode(driver, task_id, seed, hermes_mcp=True)
        messages = list(kwargs["raw_prompt"])
        if (len(messages) != 2 or messages[0].get("content") != HERMES_RL_SYSTEM_CONTEXT
                or messages[-1].get("content") != episode.episode.user_task.PROMPT):
            raise ValueError("dataset prompt differs from native user task")
        tool_schemas = episode.tools
        initial = await self.ct_build_initial_tokens(messages, tools=tool_schemas)
        runtime_ids = initial
        mask: list[int] = []
        logprobs: list[float] = []
        request_id = uuid4().hex
        observed_steps = None
        termination = "completed"
        for _ in range(self.max_turns):
            # The rollout response budget covers *all* assistant and tool turns.
            # Sampling each turn with the original max_tokens can overflow it.
            remaining = self.rollout_config.response_length - len(mask)
            if remaining <= 8:
                if not episode.raw_assistant_turns:
                    raise ValueError("no response budget for the first assistant turn")
                termination = "model_timeout"
                break
            previous_messages = list(messages)
            params = dict(sampling_params)
            params["max_tokens"] = min(int(params.get("max_tokens") or remaining), remaining - 8)
            params["logprobs"] = True
            params["stop"] = ["</tool_call>"]
            params["include_stop_str_in_output"] = True
            # Never replace a multi-token tool closing tag with a one-token stop id.
            output = await self.server_manager.generate(request_id=request_id, prompt_ids=runtime_ids,
                                                        sampling_params=params, priority=int(priority))
            if not output.token_ids or output.log_probs is None or len(output.log_probs) != len(output.token_ids):
                raise ValueError("missing sampled tokens or generation-time log probabilities")
            step = output.extra_fields.get("global_steps")
            minimum = output.extra_fields.get("min_global_steps", step)
            maximum = output.extra_fields.get("max_global_steps", step)
            if type(step) is not int or minimum != step or maximum != step or (observed_steps is not None and step != observed_steps):
                raise ValueError("missing or mixed inference-server policy versions")
            observed_steps = step
            raw = self.tokenizer.decode(output.token_ids, skip_special_tokens=False)
            if not any(raw.rstrip().endswith(tag) for tag in ("</tool_call>", "<|im_end|>", "<|endoftext|>")):
                if output.stop_reason != "completed":
                    raise ValueError("incomplete assistant generation has no completed backend return")
                merged, mask, logprobs = await self.ct_merge_assistant_token(
                    runtime_ids, output.token_ids, mask, logprobs,
                    assistant_logprobs=output.log_probs,
                )
                runtime_ids = merged.token_ids
                if len(mask) > self.rollout_config.response_length:
                    raise ValueError("incomplete generation exceeds response token budget")
                print(f"NATIVE_PILOT_INCOMPLETE_TURN task={task_id} raw_prefix={raw[:600]!r}", flush=True)
                episode.reject_incomplete_turn(raw)
                termination = "model_timeout"
                break
            # Do not assume a truncated tool block is a refusal or safe completion.
            rejected_reason = None
            try:
                content, calls = parse_strict_tool_turn(raw, tool_schemas, allow_schema_errors=True)
            except ValueError as exc:
                # Preserve a bounded sample of the *actual* policy output for
                # protocol debugging. No tool is dispatched on this path.
                print(f"NATIVE_PILOT_PARSE_REJECT task={task_id} reason={exc} "
                      f"raw_prefix={raw[:600]!r}", flush=True)
                rejected_reason = str(exc)
            merged, mask, logprobs = await self.ct_merge_assistant_token(
                runtime_ids, output.token_ids, mask, logprobs,
                assistant_logprobs=output.log_probs,
            )
            runtime_ids = merged.token_ids
            if len(mask) >= self.rollout_config.response_length:
                raise ValueError("response length exceeded before trustworthy termination")
            if rejected_reason is not None:
                episode.reject_protocol_turn(raw)
                termination = "protocol_error"
                break
            final_content = content
            if not calls:
                for suffix in ("<|im_end|>", "<|endoftext|>"):
                    if final_content.endswith(suffix):
                        final_content = final_content[:-len(suffix)].rstrip()
                final_content = final_content.rsplit("</think>", 1)[-1].strip()
            observations = episode.turn(raw, calls, final_content=final_content if not calls else None)
            if not calls:
                if not final_content:
                    raise ValueError("empty final answer; exclude whole episode")
                break
            messages.append({"role": "assistant", "content": content,
                             "tool_calls": [{"type": "function", "function":
                                             {"name": call["name"], "arguments": call["arguments"]}}
                                            for call in calls]})
            if raw.rstrip().endswith("</tool_call>"):
                boundary = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
                if len(boundary) != 1:
                    raise ValueError("unexpected non-atomic ChatML turn boundary")
                runtime_ids = runtime_ids + boundary
                mask = mask + [0]
                logprobs = logprobs + [0.0]
            previous_messages = list(messages)
            for result in observations:
                messages.append({"role": "tool", "content": str(result["observation"])})
            # A long native observation may not fit the response budget. Since
            # no further policy token can follow, retain the already generated
            # assistant turn and its audited native side effects, but omit the
            # observation from VeRL's loss sequence and end as incomplete.
            before_observation_ids = list(runtime_ids)
            before_observation_mask = list(mask)
            before_observation_logprobs = list(logprobs)
            merged, merged_mask, merged_logprobs = await self.ct_merge_context_msg(
                previous_messages, messages, runtime_ids, mask, logprobs, tools=tool_schemas,
            )
            if len(merged_mask) >= self.rollout_config.response_length:
                runtime_ids = before_observation_ids
                mask = before_observation_mask
                logprobs = before_observation_logprobs
                termination = "model_timeout"
                break
            runtime_ids, mask, logprobs = merged.token_ids, merged_mask, merged_logprobs
        else:
            # Hermes terminates after repeated invalid names. A complete,
            # side-effect-audited trace at the turn cap is a scored failure.
            termination = "model_timeout"
        verdict = episode.finish(termination=termination)
        if verdict["reward"] is None or not verdict["audit_complete"] or not verdict["reset_verified"]:
            call_summary = [(record.name, record.native_name, record.arguments,
                             record.invalid_call, record.blocked_reason,
                             sorted(key for key in set(record.before) | set(record.after)
                                    if record.before.get(key) != record.after.get(key)))
                            for record in episode.recorder.records]
            print(f"NATIVE_PILOT_UNSCORABLE task={task_id} termination={termination} "
                  f"reward_reason={verdict['reward_reason']} audit_complete={verdict['audit_complete']} "
                  f"reset_verified={verdict['reset_verified']} tool_calls={verdict['tool_calls']} "
                  f"call_summary={call_summary!r} raw_tail={episode.raw_assistant_turns[-1][:600]!r}", flush=True)
            raise ValueError("unscorable or nonresettable native episode; no optimizer reward")
        if logprobs is None or len(mask) != len(logprobs):
            raise ValueError("response mask/log-prob alignment failed")
        response_ids = runtime_ids[len(initial):]
        prompt_ids = initial
        if not response_ids or len(response_ids) != len(mask) or len(mask) != len(logprobs):
            raise ValueError("multi-turn prompt/response boundary or log-prob alignment failed")
        if len(response_ids) > self.rollout_config.response_length:
            raise ValueError("audited response exceeds configured rollout limit")
        if len(prompt_ids) > self.rollout_config.prompt_length:
            raise ValueError("initial prompt overflow; never truncate audited token boundaries")
        # Generation-time mask is the sole origin witness; context tokens have no loss.
        if kwargs.get("policy_snapshot") != "step:0":
            raise ValueError("training cell provenance differs from the frozen pilot manifest")
        # The visible tool schemas come from the installed Hermes MCP converter;
        # only the restricted benchmark MCP namespace is dispatched, with an
        # audited reversible mapping to AgentDojo's underlying functions.
        evidence = {**verdict, "split": "train", "harness": "hermes", "on_policy_rollout": True,
                    "harness_profile": "hermes_cli_mcp_benchmark_restricted",
                    "harness_contract_verified": True, "token_origins":
                    ["assistant" if bit else "context" for bit in mask],
                    "rollout_id": uuid4().hex, "policy_snapshot": f"step:{observed_steps}"}
        return AgentLoopOutput(prompt_ids=prompt_ids, response_ids=response_ids,
                               response_mask=mask, response_logprobs=logprobs,
                               reward_score=verdict["reward"], num_turns=len(episode.raw_assistant_turns) + len(episode.recorder.records) + 1,
                               metrics=AgentLoopMetrics(), extra_fields={"pilot_evidence": evidence})
