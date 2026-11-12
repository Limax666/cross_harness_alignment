# FlashRSI 4B GEPA Security Baseline Design

## Intent

Replace the active fixed-Snapshot-only collection as the primary experiment with a real FlashRSI algorithm run. Keep the interrupted B0 outputs as an auditable fixed-Snapshot control. Run the official GEPA method with the fixed Qwen/Qwen3.5-4B model on eligible security tasks, then evaluate frozen selected Snapshots across the complete executable harness × benchmark case matrix.

## Goals

- Run upstream FlashRSI `gepa` and retain Worker proposals, candidate Snapshots, decisions, trajectories, metrics, and token/call budget records.
- Cover all seven FlashRSI Target harnesses and all project-listed security benchmark sources when each cell passes environment, grading, reset, and safety-evidence qualification.
- Keep task families disjoint across train, dev, and test. GEPA may use train evidence and dev selection only; evaluate test cases only after the cell's final Snapshot is frozen.
- Keep Qwen/Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` fixed as both Target and Worker. Start each cell from the same pinned safety-constrained initial Snapshot for its harness.
- Restrict GEPA edits to the Snapshot `prompt` component. Safety policy, tools, permissions, environment, verifier, sandbox, and model configuration are immutable.
- Use one GEPA epoch, reflection minibatch size 3, and a hard per-cell task-evaluation budget of `2 * dev_count + 2 * min(3, train_count)`, which funds baseline dev and at least one worst-case candidate iteration; record actual Target and Worker calls/tokens.
- Report every intended matrix cell as qualified/running/complete/blocked with exact case counts. Never count missing, unscored, or unqualified cases as safe.

## Non-goals

- Do not run all 13 FlashRSI algorithms in this phase.
- Do not modify model weights or use GRPO/CHS-PO.
- Do not overwrite or delete B0 outputs.
- Do not call unverified historical sources HarnessSafe or HarnessSecurity-Bench executable until pinned source, cases, environment, and scoring pass the same qualification gate.

## Approaches

1. Evaluate fixed initial Snapshots only. This is the existing B0 control and does not measure RSI; retain its partial results but do not use it as the primary result.
2. Run one independent GEPA search per harness × qualified benchmark, with a common fixed 4B Target/Worker and family-disjoint splits. This directly measures the selected official method and keeps benchmark-specific selection interpretable. **Selected.**
3. Run all official methods across the full matrix. This multiplies implementation and compute costs and is outside the agreed first algorithmic baseline.

## Experiment protocol

The target matrix is the seven FlashRSI harnesses (MiniAgent, Claude Code, Codex, Pi, Hermes, OpenClaw, OpenCode) crossed with the project security sources (AgentDojo, AgentHarm, HarnessAudit, HarnessRisk, ActBench, SafeClawArena, WildClawBench, plus any subsequently verified source). Each cell is admitted independently only after its task executor, reset behavior, tool mapping, scoring positive/negative controls, and auditable security outcomes pass qualification. A blocked cell remains visible in the ledger.

Each admitted cell runs FlashRSI GEPA from its pinned initial Snapshot. Use only train trajectories for GEPA proposals; candidate selection uses dev only and the safety-constrained primary score. Candidate admission must not accept an increase in dev attack success or a decrease in verifier coverage relative to its parent. GEPA can only edit the declared prompt file; a project guard rejects any candidate that removes or changes the pinned safety-policy block. Report benign utility, attack success rate, safe-and-useful, and each source-native side-effect metric separately. Freeze the selected Snapshot before any test execution. Then run every admitted test case through every intended harness; preserve case-level records and independent oracle evidence.

The same Qwen 4B endpoint serves Target and Worker. Calls are serialized if required by the existing single-sequence vLLM service. Credentials stay in environment variables and are not written to manifests. The pinned FlashRSI commit is `778e2d01a995491fe63db210dcf46db1723a4430`.

## Constraints and risks

- Current code has adapters and execution evidence for only part of the 7×benchmark matrix; full coverage is conditional on qualification and adapter completion.
- FlashRSI's stock GEPA expects a `prompt` component, a declared instruction file, nonempty train/dev splits, and unique dev task IDs. All seven pinned Target adapters expose a prompt component and instruction file, which GEPA requires.
- Stock GEPA uses scalar per-task fitness and dev selection; the security non-regression and coverage guard must be implemented at the candidate admission boundary without changing GEPA proposal/frontier policy, and recorded as a project safety gate.
- A source without reliable independent scoring or reset evidence is not eligible for algorithm feedback or a safety claim.
- The running B0 collection is interrupted by user instruction. Existing outputs remain intact and a stop record marks the interruption; the vLLM endpoint is retained for reuse.

## Success criteria

- B0 collectors are stopped and retained with interruption metadata.
- A real FlashRSI GEPA B1 run emits Worker traces, candidate Snapshots, accept/reject decisions, and fixed-budget records.
- The matrix ledger accounts for all harness/source cells, with no unqualified cell silently omitted.
- Every completed final test result traces to an immutable model, initial/final Snapshot, benchmark revision, verifier, and case ID.
- No safety result is claimed for missing, invalid, or unscored episodes.
