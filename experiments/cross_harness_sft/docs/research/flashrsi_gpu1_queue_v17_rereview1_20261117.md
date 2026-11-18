# GPU1 finite queue v17 fix rereview 1

Verdict: **Q-I1 ADDRESSED; Q-I2 ADDRESSED. No new Critical or Important finding, or other load-bearing issue, established within the fix diff.** Spec compliance and quality/privacy are approved for this scoped fix round. This is not evidence of an executed or completed queue.

Reviewed original `flashrsi_gpu1_queue_v17_review_20261117.md`, latest round1 implementation report, and exact `flashrsi_gpu1_queue_v17_fix1_review.diff` for MAIN `f818b483..0e7af3d8`. Scope was Q-I1/Q-I2 and new breakage introduced by this fix only. Relevant unchanged sequential rollout callers were read to check exception propagation. No tests, source/host execution, API/model/GPU/Docker action, process signal, service/cron change or subagent was run. Only this rereview file was written. Active ACEv17, its existing v17 benchmark/reference code and other live work were untouched.

## Q-I1 — ADDRESSED

Locations relative to this experiment: `scripts/flashrsi_matrix_gpu1_queue_v17.py:231–266,299–300,328,427,468–469,492–493`; new `flashrsi_harnessrisk_benchmark_v18.py:10–17`; guarded reference-v18/v19 entries.

The queue now consumes initial/algorithm host timeout, explicit cleanup-false and still-running owned-stage receipts. It writes a persistent `queue_cleanup_block.json`, marks dispatch readiness false, marks pending work blocked, and raises `QueueCleanupUncertain` before further dispatch. An empty name scan cannot clear that blocker. Checks occur after stages and before later model or new judge dispatch. Exact predecessor identities retain a bounded run scope whose receipts are also checked after predecessor exit, so disappearance of its process name cannot erase known timeout uncertainty.

The new benchmark-v18 wrapper raises `HostCleanupUncertain` immediately after the immutable v17 bridge has retained a timeout/cleanup-false receipt. Inspection of the unchanged `Agent.rollout` and algorithm `_evaluate` paths confirms sequential awaited execution without swallowing that RuntimeError: the next case is not started. Ordinary future stages use reference-v19; OpenCode context-fit GEPA reference-v18 composes the same guarded benchmark. This closes both the within-stage and later-queue dispatch paths while preserving active v17 code and original grading/status semantics.

The added guards cover initial timeout preventing Codex dispatch, algorithm timed-out receipts preventing the next algorithm, persistent blockade despite an empty lane scan, within-rollout abort before the second case and timeout evidence in an exact predecessor scope. Reconciliation/unblocking is deliberately external and identity-based; no automatic retry or claim of actual descendant absence was added.

## Q-I2 — ADDRESSED

Locations: `scripts/flashrsi_matrix_gpu1_queue_v17.py:100–159,330–332,345–349,362–376,383–392,406–410,442–444,468–482`.

H0 admission now takes an explicit expected contract derived from the trusted supplied raw case-manifest bytes, preserving those bytes verbatim at queue/harness roots. It requires the retained H0 parent manifest hash and the H0's declared hash to match that trusted identity. Required fields include source model/FlashRSI revision, exact harness/host, port18764, privacy/owner protocol, temperature, reply cap, applicable native turn/timeout limits and128-case count; missing fields fail, including the explicit null Codex turn cap. The older OpenCode v11 contract uses its actual available declared fields without inventing absent metadata or relaxing its existing12-turn4096-token contract.

The trusted source identity, expected contracts and retained reference binding enter the queue protocol and stage receipts. Algorithms revalidate H0 before dispatch and compare Snapshot/run-manifest/case-manifest bindings across the stage. Existing strict128 private/mock/rule checks and the separate exact128 judge gate remain in place. Same-ID/different-case bytes or wrong model/runtime revision, port, harness, temperature or budgets cannot pass the fixed gate. The added negative guards cover those mutations; no source or budget was changed to satisfy admission.

## New Critical / Important findings

None established within `f818b483..0e7af3d8`.

## Evidence and remaining limits

Read retained `flashrsi_gpu1_queue_v17_fix_round1_20261117.log`: **19 passed in 0.76s**, SHA256 `db3bae3dd10bb9a07e702b9009aa83e80b1dd6d1bd836972be9372bd1ae0b087`. No tests were rerun. The reviewed diff artifact SHA256 is `a94d8b94e809635b6fb87d36437bbfa445c1809cedb42f09c73004443ccc5917`. The report retains expected RED evidence and separately reports the raw RED transcript whitespace check failure; this rereview does not reclassify it as a pass.

Actual predecessor waiting/locks, descendant reconciliation, corrected full H0/algorithm behavior, native parser/context execution and judge/API results remain root-owned real-path verification. Full-objective completion remains false; failed and deferred tickets remain preserved. No actual queue completion, private-case scoring change, permission expansion or use of GPU0/2/3 is claimed.
