# FlashRSI Safety Baseline Completion Plan

**Goal:** Produce qualified, independently auditable 4B safety baselines across all seven upstream harness integrations and all 13 pinned FlashRSI methods.

**Architecture:** Preserve source-native harness, algorithm and scoring identities; implement per-cell adapters and safety admission with immutable policy, permissions, verifier and sandbox. First run bounded development diagnostics, then freeze family splits, budgets and snapshots before qualified validation/test. B0, each resource-evolution method and weight RL are separate conditions.

**Tech Stack:** FlashRSI `778e2d01a995491fe63db210dcf46db1723a4430`, Qwen3.5-4B `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, AgentDojo v1.2.2, Python, vLLM.

**Spec:** `experiments/cross_harness_sft/docs/safe_rsi_harness_scaling_experiment_20261111.md`; latest user request overrides the prior GEPA-only non-goal in `docs/superpowers/specs/2026-11-12-flashrsi-4b-gepa-security-design.md`.

## Constraints and steps

- [ ] Record all 7 harnesses × 13 methods × intended security sources, their actual edit scopes and admission status. Stock wildcard methods require confinement and immutable policy/tool/sandbox before running; never relabel a restricted method as untouched upstream behavior.
- [ ] Fix context failures with an explicitly frozen 16K configuration and independently verify overflow, 512-token truncation, protocol and reset; retain 8K and 16K separately. Diagnose all failure families without using final test for tuning.
- [ ] Implement case-family path oracles with independent malicious/benign controls and actual state evidence; unknown cells abstain. Keep official verifier and side-effect coverage separate.
- [ ] Complete reusable isolated per-harness environment adapters, bind actual evolved Snapshot to target runtime, preserve tool-schema identity, enforce permitted resource changes and admit only demonstrated reset, provenance and oracle controls.
- [ ] Integrate all 13 native algorithm implementations with their true prompt, skill or resource scope and method-specific budgets; verify proposals, candidate admission, actual evaluation and version lineage. Persist blocked or unsupported cells with cause; do not fake missing evolutions.
- [ ] Freeze family-disjoint benchmark splits and budgets; run initial and algorithm baselines across qualified harness/source cells. Record every attempted outcome and error, costs and model/worker versions. Use independent untouched held-out families for final measurements and report utility, ASR, safe-and-useful, path and verifier coverage with explicit denominators and intervals.

## Current status

MiniAgent GEPA made a rejected proposal and selected the initial Snapshot; it is not an evolved result. The existing 96-case test was viewed and cannot be treated as untouched for this iteration. Only calibration-family controls are available, not formal admission; expanding a diagnostic run does not change this status. 16K diagnostics on 15 cases from one previously observed family achieved 15/15 official scoring but 0/15 path-audit coverage. All other harness × method baselines remain unverified.
