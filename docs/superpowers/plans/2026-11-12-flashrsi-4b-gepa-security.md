# FlashRSI 4B GEPA Security Baseline Implementation Plan

> **For agentic workers:** Implement this plan task-by-task. Preserve existing uncommitted work and B0 result directories.

**Goal:** Run official FlashRSI GEPA with a fixed Qwen 4B Target/Worker and produce an auditable eligible-cell full security benchmark × harness matrix.

**Architecture:** Reuse the pinned FlashRSI `GEPA`, `Agent`, `Snapshot`, and trajectory record contracts. Add a security-suite benchmark/runner adapter and a matrix ledger around the existing project task executors, with a hard safety/coverage gate for candidate admission. Use one independent search per harness × qualified benchmark; freeze the selected Snapshot before test execution.

**Tech Stack:** Python 3.12, pinned FlashRSI checkout, existing AgentDojo/AgentHarm environments, local vLLM OpenAI-compatible endpoint, JSONL/JSON run artifacts.

**Spec:** `docs/superpowers/specs/2026-11-12-flashrsi-4b-gepa-security-design.md`

## Global Constraints

- Model/revision: `Qwen/Qwen3.5-4B` / `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- FlashRSI commit: `778e2d01a995491fe63db210dcf46db1723a4430`.
- Algorithm: official `gepa`; one epoch, reflection minibatch size 3, and per-cell metric-call budget `2 * dev_count + 2 * min(3, train_count)`.
- Editable scope: `prompt` only. Safety policy, tools, permissions, model, sandbox, and verifier remain fixed.
- Train/dev/test are family-disjoint; test is not used by GEPA.
- Do not delete/overwrite B0 directories or credentials; do not claim unscored/blocked cells as safe.
- Run no tests unless the user asks; use runtime preflight and evidence audit for verification.

## Review Focus

- Benchmark cases with missing/invalid independent oracle: qualification must reject the cell before algorithm feedback.
- Candidate with better utility but higher ASR: safety gate must reject it.
- Candidate with lower verifier coverage: safety gate must reject it.
- Harness without FlashRSI `prompt` component: GEPA must fail closed and ledger the cell as blocked.
- Interrupted or resumed B1 cell: stable run/case IDs must prevent duplicate scoring and preserve parent Snapshot lineage.

---

### Task 1: Record B0 interruption and freeze the B1 matrix inventory

**Files:**
- Create: `experiments/cross_harness_sft/outputs/b0_pause_record_20261112.json`
- Create: `experiments/cross_harness_sft/outputs/b1_gepa_security_20261112/matrix_ledger.jsonl`
- Read: `docs/full_flashrsi_security_baseline_20261112.md`

**Interfaces:** The ledger has one row per 7 harnesses × every listed security source. Fields: `harness`, `benchmark`, `source_revision`, `qualification_status`, `train_cases`, `dev_cases`, `test_cases`, `reason`, `run_id`, `final_snapshot_id`.

- [ ] Record stopped collector roots, stop time, retained B0 output paths, and retained vLLM PID; confirm no B0 collector remains.
- [ ] Initialize all matrix rows; mark each cell `pending_qualification` rather than treating inventory as executable coverage.
- [ ] Pin and record all available source revisions and source counts; leave unverifiable revisions explicitly unknown and blocked.

### Task 2: Add family-safe composite benchmark adapters

**Files:**
- Create: `experiments/cross_harness_sft/scripts/flashrsi_security_benchmark.py`
- Read: `vendor/FlashRSI/flashrsi/core/benchmark.py`
- Read: existing `flashrsi_*_full_host*.py` task adapters

**Interfaces:** `load_security_cell(harness: str, benchmark: str, root: Path) -> SecurityCell`; `SecurityCell` supplies frozen `train`, `dev`, `test` `Task` tuples and a source-native grader/evidence validator.

- [ ] Map each source case to an immutable FlashRSI `Task` carrying original family ID, clean/attack arm, source revision, verifier ID, and fixture/reset identifiers.
- [ ] Split families before task/harness/arm expansion; reject leakage, duplicate IDs, missing oracle, or nonresettable cases.
- [ ] Implement `grade()` that retains source-native scores and derives utility, ASR, safe-and-useful, coverage, and observed side effects without assigning safety to missing scores.
- [ ] Add per-cell qualification command that executes positive/negative scoring controls and a reset/replay check; write evidence and gate result to ledger.

### Task 3: Integrate GEPA with the safety admission gate

**Files:**
- Create: `experiments/cross_harness_sft/scripts/flashrsi_gepa_security.py`
- Read: `vendor/FlashRSI/flashrsi/algorithms/gepa/__init__.py`
- Read: `vendor/FlashRSI/flashrsi/core/agent.py`

**Interfaces:** `run_gepa_cell(config: dict, cell: SecurityCell, target: Agent, worker: Agent) -> FrozenCellResult`.

- [ ] Instantiate the official GEPA implementation with its `prompt` evolve scope; fail closed when the Target does not expose that component.
- [ ] Configure fixed Qwen 4B Target and Worker through the existing vLLM OpenAI-compatible endpoint; keep secrets outside run records.
- [ ] Add a candidate admission wrapper that preserves upstream GEPA proposal/frontier logic but rejects any dev candidate with increased ASR, decreased verifier coverage, or invalid scoring; accept only strict improvement in the frozen safety-constrained primary score.
- [ ] Ensure Worker evidence contains train trajectories only and cannot receive dev/test labels, hidden oracles, or credentials.
- [ ] Persist Worker requests/responses, proposals, candidate Snapshot IDs and hashes, parent lineage, dev comparisons, decisions, and actual calls/tokens.

### Task 4: Build the resumable matrix runner and run budget accounting

**Files:**
- Create: `experiments/cross_harness_sft/scripts/flashrsi_run_gepa_security_matrix.py`
- Modify: `experiments/cross_harness_sft/docs/full_flashrsi_security_baseline_20261112.md`

**Interfaces:** CLI options `--matrix`, `--model-revision`, `--max-cells`, `--resume`, `--qualification-only`; defaults enforce pinned model/FlashRSI versions, one epoch/reflection minibatch size 3, and per-cell metric budget `2 * dev_count + 2 * min(3, train_count)`.

- [ ] Run only qualified cells and persist status transitions atomically (`pending` → `running` → `frozen` → `test_complete`, or `blocked/error`).
- [ ] Resume from stable cell/run IDs without duplicate task scoring; never silently change source revision, split, seed, or initial Snapshot.
- [ ] Freeze each final Snapshot before test; run all eligible test cases across the required harness matrix only after freeze.
- [ ] Aggregate results by source, family, harness, clean/attack arm, with numerator/denominator, invalid/unscored counts, confidence intervals, and per-cell budget.
- [ ] Update the experiment document with actual status, algorithm/Worker roles, and run artifact paths.

### Task 5: Runtime qualification and initial B1 launch

**Files:**
- Create: `experiments/cross_harness_sft/outputs/b1_gepa_security_20261112/preflight.json`
- Create: per-cell FlashRSI run directories under `experiments/cross_harness_sft/outputs/b1_gepa_security_20261112/runs/`

- [ ] Verify model endpoint/revision, FlashRSI commit, GPU/RAM, Worker/Target request limits, immutable resource scope, and one admitted cell's data/reset/scoring contract.
- [ ] Run the first qualified B1 cell with the frozen one-pass budget; inspect actual worker proposals, candidate decisions, snapshot hashes, test isolation, and resource usage.
- [ ] After the first-cell evidence is valid, continue the same frozen protocol through all other qualified cells; leave failed gates blocked and report exact reasons.

