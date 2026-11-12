# Full FlashRSI Security Baseline Implementation Plan

> Execute inline, preserving the benchmark and harness identities and the evidence for each case.

**Goal:** Evaluate the pinned 4B model across every FlashRSI target harness and the complete available security benchmark cases.

**Architecture:** Freeze a source- and split-aware case ledger before inference, execute each harness through its actual FlashRSI adapter and benchmark runtime under restricted access, persist immutable per-case trajectories and evidence, and report coverage and failures separately from scored outcomes. Handle unavailable combinations explicitly and retain final-test isolation from later development and model selection.

**Tech Stack:** FlashRSI 778e2d0, Qwen3.5-4B 851bf6e, vLLM, AgentDojo 0.1.35/v1.2.2, project AgentHarm and other security benchmark adapters.

**Spec:** `experiments/cross_harness_sft/docs/4b_safe_rsi_local_launch_plan_20261111.md` and user request for all FlashRSI harnesses × complete safety benchmark cases.

## Global Constraints

- Do not replace a native adapter with a different harness or present simulated tool execution as a native environment.
- Keep model, initial snapshot, task manifests, seeds, timeouts, schemas, sandbox and verifier versions identifiable; persist all attempts.
- Do not use held-out test results for evolution or model selection; abstentions and failures are not safe passes.
- Do not run unrestricted hostile model-generated code, disclose credentials, alter unrelated workloads or claim unaudited isolation is production-safe.

## Tasks

- [ ] Freeze an exhaustive, split-labelled AgentDojo ledger and the exact 7 FlashRSI Target adapter identities; enumerate project safety benchmarks and dependencies.
- [ ] Run every eligible AgentDojo case through the MiniAgent integration, preserve original outputs, errors, reset and attack evidence, and make runs resumable.
- [ ] Connect and execute each remaining FlashRSI adapter to real benchmark tools under confinement; separately identify embedded semantic ports and native CLI adapters.
- [ ] Connect and run complete cases from each project safety benchmark whose data and tool runtime are available; record unavailability and actual blockers for others.
- [ ] Aggregate by adapter, benchmark, case, split and arm; include error/abstention coverage and audit model and snapshot provenance.
