# ACE judge follower scoped rereview 1 — 2026-11-17

**Reviewed:** `2a20165..5731dd8`, supplied fix1 diff, current new follower/test, implementation report and retained `flashrsi_ace_judge_followup_v17_fix1_green.log`. No tests were rerun and no live/source/API/model/GPU actions were taken. Only this report was written.

**Verdict: both original Important findings are closed; one remaining Important terminal-error edge needs correction before launch.** Critical: none. Review scope is the two fixes and newly reachable finalization behavior, not the unchanged queue or broader experiment.

## Closed findings

- `scripts/flashrsi_ace_judge_followup_v17.py:20–22,58` now waits on immutable `(pid, start_ticks)`. A command change across `exec` no longer releases the wait, and PID reuse with a different start tick does. The new pure boundary test covers both comparisons.
- `drain_judges` at lines 25–36 requires an actual nonempty pending list before returning a lease/billing deferral. A completed spawned child with no pending jobs now drains successfully and reaches final admission instead of being mislabeled deferred. The new boundary test covers successful/unsuccessful child-state preservation and a pending lease case.
- The accepted source binding is checked again after the final poll and before returning from the drain. Admission, shared worker lease, billing guard, cache and no-overwrite dispatch remain unchanged by this diff.

## Important: final acceptance can hide a nonzero judge child exit

The repaired drain makes `run_followup` lines 79–82 reachable after a judge child exits. Those lines overwrite the task's outer status from `inspect_run` and return 0 whenever retained source/judge artifacts are complete, without checking the actual `judge_returncode` or failed `judge_status` retained by `Queue.poll_judge`.

Concrete path: the canonical judge writes a full successful `manifest.json`, then fails in `summarize` or output/teardown. This is possible in the actual order at `HarnessRisk/harness_adapter/scripts/eval_via_llm/llm_judge_multi_harness_batch_aiwave.py:1056–1059`: manifest write precedes summary and final return. `scripts/flashrsi_matrix_gpu1_queue_v17.py:290–295` correctly retains a nonzero return code and `judge_status=needs_judge_or_billing_repair`, even when the full cached case set is valid. But the new follower finalization sees the complete artifacts, sets outer `status=complete` and returns 0. The resulting task simultaneously claims completion and carries a failed judge control outcome.

Preserve these as separate facts: retained 128-case judge coverage may be complete, while this launched judge command failed. Do not return success or replace the failed task with a successful terminal state when its recorded child exit is nonzero. Keep the failed control task explicit, retaining its return code and accepted artifact metadata; recovery can reuse all cached case successes without API repetition.

Add one pure finalization/controller regression for **full artifact acceptance plus a nonzero spawned-child exit**, asserting a nonzero follower result and explicit failed/pending control status. Pair it with the full-acceptance/zero-exit success case. The current new test exercises only `drain_judges`; it does not call final `run_followup` admission and therefore cannot catch this edge. No expensive test or real judge request is needed.

## Evidence boundary

The retained green log contains **3 passed in 0.15s**. This reviewer read that evidence; the test run was performed by the implementer. It supports the retained-data admission and newly tested helper paths, not the untested finalization edge above. Real predecessor waiting, judge API completion and final cache admission remain unverified. The active strict ACE job can continue unchanged; this report does not clear the finite follower for launch yet.
