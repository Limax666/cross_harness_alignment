# 9B Cross-Harness Online RL Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution; the user has approved the experiment design and requested execution.

**Goal:** Start an honest 9B full-parameter online RL run with exact-task K=2 assignment, verified executable harness differences and frozen train-family provenance.
**Architecture:** Merge only SFT model shards into a validated HF actor/reference. Freeze benchmark-family/task assignments, execute current-policy groups through audited adapters, and wire fixed rollout iterations to VeRL with separate update counts.
**Tech Stack:** Existing Torch/VeRL/vLLM environments, official AgentDojo/AgentHarm, FSDP, tmux.
**Spec:** experiments/cross_harness_sft/docs/9b_cross_harness_rl_design_20261027.md

## Global Constraints

- Full-parameter 9B; no LoRA or 2B substitution. SFT optimizer upload does not block a new RL optimizer.
- Preserve user work; no broad git reset/clean, no unrelated GPU preemption.
- H_RL_train candidates Codex/Claude Code/Hermes; NanoBot/OpenClaw held out of updates.
- Match exact task/attack/fixture across K=2 rounds, independent rollouts, no reward-driven task replacement.
- No training claim until genuine current-policy rollout, independent scoring and parameter update complete.

## Review Focus

- Model files present but upload/sharding/content inconsistent: verify conversion and tensor shapes before launching.
- Runtime label differs while execution contracts remain identical: gate fidelity rather than claim native diversity.
- Family pair absent or validation identity unknown: never guess a pairing or override a split.
- Zero-advantage groups: fixed ledger, no dynamic replacement; record no-op/KL-only independently.
- Host RSS, mock-process leakage and weight-sync peak: bounded workers/teardown and measured preflight.

### Task 1: Model preparation and environment
- [ ] Inspect model/optimizer roles, disk/RAM/GPU and existing package versions.
- [ ] Convert model shards at low I/O priority with official VeRL merger; validate saved tensor inventory and checkpoint identity.
- [ ] Trace actual 9B SFT provenance and available frozen train-family fixture contracts.

### Task 2: Assignment ledger
- [ ] Add failing tests for exact task/payload/state reuse, split exclusion, balanced source/harness arms, deterministic resume and no-op phase advance.
- [ ] Implement a separate immutable ledger/sampler module; keep old 2B sampler behavior intact.
- [ ] Run focused tests and freeze a candidate ledger only from audited family pairs.

### Task 3: Runtime and trainer integration
- [ ] Verify source×harness common task intersections and fidelity evidence against actual runtime contracts.
- [ ] Add fixed-iteration no-op/KL-only semantics, group admission and recovery state for K=2.
- [ ] Parameterize 9B actor/reference, budget and resource launcher without changing historical run defaults.

### Task 4: Execution and evidence
- [ ] Run model/protocol/reset/reward gates and measured 9B memory preflight.
- [ ] Start a distinct tmux run if gates pass; verify first online scored update, no unsafe resource leak.
- [ ] Persist truthful status and independent curves; report exact remaining gate if one cannot be met.

## Execution ledger — 2026-10-27

- Task1: model-only merge completed; 760 tensor inventory/shape gate and BF16 GPU load/generation PASS. Actual 9B SFT view unknown; conservative v7 train allowlist frozen, no false provenance claim.
- Task2: immutable K=2 ledger, 300 rounds, fixed exact task/payload/state; no-op/cursor recovery tests PASS.
- Ruling: user-selected whitebox AgentDojo first supersedes real-product-first order. Native CLI labels cannot stand for product training fidelity; cost: this first run alone cannot establish native-product generalization.
- Ruling: C=0; only S/T/O/W vary in projected balanced 18 configs. Token-faithful compaction not implemented; cost: no full-five-factor reproduction claim.
- Task3: runtime/trainer/launcher wired; real state/schema integration regression RED→GREEN after hash correction. 49 related tests passed, 5 objective/sampler tests passed. Independent reviewer confirmed no remaining deterministic first-update blocker.
- Final: fixed local pandas shadowing, runtime/build state/schema mismatch, config numpy types, whitebox audit retry bias and assumed KL-only signal.
- Final: minor (deferred): 480s is a between-turn check, not a cancellable generation deadline; finite token/turn budgets still apply.
- Task4: first launcher rejected busy GPU3 before initializing Ray. Four-GPU/RAM-guard queue and low-I/O metrics/plot monitor installed. Actual first 9B online update and recovery verification remain pending resources. No parameter-update or efficacy claim.
- No broad git commit/push: unrelated dirty user work preserved.

- Ruling: two-card r2 stopped during model initialization, before rollout/update, after resolved FP32 master dtype invalidated BF16-only memory estimate. No model downgrade. Cost: startup attempt spent on model loading; no training data claim.
- Ruling: use FSDP2 CPUOffloadPolicy for actual CPU Adam, and frozen BF16 reference; the old manual optimizer_offload still loads all Adam states onto GPU. Preserve FP32 trainable masters. Conservative resource guard>190GiB RAM, two idle GPUs; cost: potentially longer resource wait, to be revised only from measured peaks.
- Task4 latest: r3 is queued on resources, no completed update; actual resource telemetry/proof must precede any 'training' claim. Related final focused tests remain green.

- Ruling: user freed GPUs0/2/3. Keep frozen BF16 reference on GPU (explicit opt-in, legacy default preserved); actor CPU FP32 Adam unchanged. 2 placement tests RED→GREEN. Actual text/vision tensor counts permit a 153GiB host first-update preflight; disable optimizer-state disk parking to avoid large per-step IO. Cost: tighter GPU transient headroom; this is a measured-preflight candidate, not a passed memory gate. r4 launched on0/2; actual first scored update pending.
