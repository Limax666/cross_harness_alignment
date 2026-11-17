# Genuine v5 single-GPU online GRPO

`scripts/train_hsrl_v5.py` consumes only `hsrl_v5.dataset.Dataset` READY releases. It trains the official Qwen3.5-4B `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` with language-only rank16/alpha32 LoRA on physical GPU0. GPU1 remains a fixed-weight baseline lane; GPU2/3 are forbidden. No fallback dataset, source simulator, model API, native product harness or CHS-PO implementation is used.

This is implementation documentation. The focused CPU checks do not establish CUDA/kernel feasibility, real optimizer correctness, memory fit, effective policy improvement or successful recovery. Those observations must come from the authorized genuine run.

## Frozen training contract

The final release has 7K ClawGym benign and 7K AgentDoG conditions (5500 attack/1000 clean/500 query_target). Each step has16 groups×G8, half general and half safety. The sampler repeats the same groups under the peer configuration at the next step, after the first optimizer update. K2 is successive current-policy execution, never a4+4 normalization group. Each rollout opens a new isolated `Dataset.episode`; source state is retained only inside that episode.

`CONFIG` in the driver is the authoritative frozen configuration: AdamW lr1e-5, betas(.9,.999), eps1e-8, weight_decay0; first actual optimizer update uses0.5×lr, update2 onward uses full lr; gradient norm clip1; PPO clip.2; one optimization epoch; reference loss coefficient.001. BF16 base remains frozen; FP32 language LoRA parameters and optimizer state train. Reference scoring disables the same model's adapters. No second model or inference engine is loaded.

Generation uses temperature.7/top_p.95/top_k0. Old/current/reference scores use untempered, untruncated model logprobs. This is explicitly a raw-policy PPO surrogate, not an exact ratio of nucleus-sampling probabilities. The run manifest names this convention. All sampled action IDs, including terminal EOS, remain in scoring/masks; text extraction removes only the sampled terminal EOS, without globally dropping special-token business data.

GRPO-S uses the frozen registered-call reward vector and applicability. Judge safety, actual harm and persistence remain N/A in this release. GRPO-U explicitly masks malicious query-target PG and retains reference loss on its action tokens. PG is normalized by all PG-applicable action tokens in the admitted128-rollout step; reference loss by all admitted action tokens. Normalization never averages segment/microbatch means. Source output, prompt/schema, summaries and observations receive no action loss. Group std uses population variance and epsilon1e-6. Genuine zero-variance groups retain zero PG advantages and cost accounting; no reward differences are invented. An optimizer call with unchanged adapter bytes is recorded separately from an effective parameter update.

The cap is38,400 actual attempts/300 complete128-attempt steps, including failures. This samples at most2400 task-condition positions and is not a full14K pass. A full sampler pass would need224,000 episodes. Source/infra failures stop the driver with a durable blocked state; invalid groups never become smaller groups or synthetic reward0.

## Shared rendering and external context

```python
from hsrl_v5.runtime import prepare_interface, CONTEXT_TOOL, CONTEXT_SCHEMA
prepared = prepare_interface(tokenizer, public, row, workspace)
prompt_ids, public_messages, compaction = prepared.view.render()
tools = prepared.mapping.public_tools
```

`public` contains only public `messages` and `tools`. The return is `PreparedInterface(view,mapping,context)`. `CONTEXT_TOOL='hsrl_read_context'`; its schema takes an empty object and has no paths or write operation. It is prepended identically to every C configuration before the T mapping, and source name collisions are rejected. The fixed context resource is `hsrl://context_summary`. Reads return `{"available":false,"content":null}` until a summary exists. C1/C2 host compaction updates collector-owned memory; reads never enter source state, workspace files, snapshots or graders. Source tools remain unchanged. Workspace is an API-compatible evidence location; this renderer does not read/write it.

The final READY context certificate must be generated using this exact interface. Earlier audits without the shared reader have different tool/prompt hashes. Initial render fit alone does not verify dynamic C1/C2 compaction. The first actual histories crossing the thresholds must leave before/after token counts, hashes, exact segment prefixes and real context-read evidence. C1 retains5 rounds at8192; C2 retains3 at6144; the extractive summary's measured cap is2048 tokens. No prompt is silently truncated to force fitting.

JSON tool-call framing parses a whole JSON object before checking the closing marker, preserving literal reasoning/tool markers inside business strings. Duplicate keys, nonfinite numeric values, invalid names, incomplete framing and public schema violations fail as protocol errors. XML syntax remains supported; JSON should represent arbitrary string data containing XML delimiters. Parser errors execute no calls, and their audited protocol cost is merged into the source result before the frozen reward API.

## Handoff and command

The root owner must preserve pending baseline/judge/repair work in a persistent ledger and complete the GPU0 handoff first. Idle resident inference service is not sufficient. The driver never stops services, moves baseline queues or edits cron.

Prepare a private handoff JSON with this schema:

```json
{
  "schema": "hsrl-gpu0-training-handoff/v1",
  "physical_gpu": 0,
  "exclusive": true,
  "gpu1_baselines_preserved": true,
  "forbidden_gpus": [2, 3],
  "boundary_status": "stages_ended_preserved_unresolved_work",
  "old_service_pids": [3092154, 3092760],
  "baseline_ledger_path": "/absolute/path/to/the/preserved-baseline-ledger.json",
  "baseline_ledger_sha256": "the actual SHA256 of that ledger",
  "lock_path": "/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/rl/hsrl_gpu0.lock"
}
```

The paths/hash above describe required deployment inputs, not existing verified artifacts. `accepted_complete` is also allowed when actual acceptance supports it. Old service PIDs must be absent, the ledger hash must match, and GPU0 must show no compute process before model loading. The lease lock and per-run lock stay held for the run. The manifest binds lease bytes, preserved ledger identity, base/tokenizer artifact hashes, all v5 modules, policy/component/driver code, packages, dataset READY/manifest, reward/configuration and reference identity. Changes block continuation.

After the root has supplied actual READY/release, lease and new output paths:

```bash
cd /data/home/liumingxiao/hsrl_v3_worktree_20261116/experiments/cross_harness_sft
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts \
  /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python \
  scripts/train_hsrl_v5.py --dataset "$HSRL_READY_RELEASE" \
  --model /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/checkpoints/qwen35-4b-instruct-official-851bf6e \
  --out "$HSRL_NEW_RUN" --lease "$HSRL_GPU0_LEASE" --max-attempts 38400 --seed 267
```

Add `--resume` with the same arguments to continue the same run. The driver refuses overwriting an existing run without `--resume`. GRPO-U is an independently initialized arm selected with `--objective utility`; it is not a continuation of GRPO-S.

## Recovery and evidence

Before every real attempt, one atomic `journal.pt` stores the reserved attempt, sampler step/assignment, collected episode hashes, last committed checkpoint and Python/NumPy/Torch CPU/CUDA RNG. `state.json` is only a readable mirror. Completed episodes are durably linked before the next attempt. An interrupted in-flight attempt is charged and blocks automatic reexecution, requiring explicit reconciliation. A saved invalid episode also blocks continuation; it is never replaced by a successful retry without preserving identity/cost.

Each accepted optimizer call writes an atomic checkpoint containing adapter, optimizer, scheduler, RNG, version, adapter hash and complete controller state, with a `COMPLETE.json` content map. The journal checkpoint pointer is authoritative. If interrupted during update, reload the last committed adapter/optimizer and rebuild from retained exact episodes; do not rerun environments or use possibly partly updated weights. Completed checkpoints not referenced by the journal remain uncommitted evidence. The frozen-base version signature is a mutation guard over parameter metadata/version counters, not a weight-content hash; base artifact hashes are separately frozen.

Retain `run_manifest.json`, journal/state, per-segment IDs/logprobs, public interface/schema/mapping, canonical/source action evidence, private source grades, protocol audit, context reads/compactions, adapter transitions, real per-step gradient/loss/reward/token metrics, CUDA allocated/reserved peaks and failures. Resource gates stop new work below32GiB host available memory or100GiB disk free, or at60GiB of run growth. The current implementation retains checkpoints and never deletes history; rank16 full optimizer recovery every update can reach60GiB before300 updates. A larger budget or explicit current-run recovery retention policy requires a frozen follow-up change.

## Verification

Focused permitted checks, with no model/Torch import, API, source payload or Docker execution:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts \
 /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python \
 -m pytest -q tests/test_hsrl_v5_training.py tests/test_hsrl_v4_components.py
```

The real run must still establish finite real old/ref/current probabilities and gradients, effective adapter change, unchanged frozen base identity, peer rollout using the updated policy, actual C1/C2 compaction/read, checkpoint restore continuity, valid source reset/grading, real resource peaks and source/condition coverage. Failure or no effective update remains an observed failure/no-change result. The focused tests cannot substitute for these real-path observations.
