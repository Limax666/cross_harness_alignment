# GPU1 finite queue v17 independent review

Reviewed MAIN `e1cec998..f818b483`: new initial-v17, algorithm-reference-v18, GPU1 queue-v17, scanner-v17, focused tests and handoff documentation. Relevant existing benchmark/algorithm/reference/judge/acceptance callers and the v11 reference manifest were read to resolve their interfaces. No tests, host/source execution, model/API/GPU/Docker operation, process signal, service/cron change or subagent was run. Only this review file was written. Active predecessors 3839693/3839699 and CPU judge work were left untouched.

## Verdicts

- **Spec compliance: changes required** for unknown-cleanup dispatch blockade and matched initial-reference identity (Q-I1/Q-I2).
- **Quality/privacy: changes required** for Q-I1/Q-I2. No new private-case exposure or changed grading/model/permission/budget policy was established in the scoped diff. No Critical finding established.

## Critical

None established.

## Important

### Q-I1. Recorded unknown host cleanup does not block dispatch of the next harness

Locations relative to this experiment: `scripts/flashrsi_harnessrisk_initial_v17.py:22–29`; `scripts/flashrsi_matrix_gpu1_queue_v17.py:254–258,367–378`.

The initial bridge correctly writes `timed_out:true, cleanup_confirmed:false` and raises after the outer host timeout. That uncertainty exits the initial subprocess, but the queue classifies it as an ordinary failed H0, blocks only that harness's six algorithms and continues to the next harness. It never reads or promotes the cleanup-uncertain receipt into a queue-wide dispatch blocker. The next `execute()` only waits for `lane_clients()`, a command-name/port scan. Killing the direct host does not prove its native CLI/sandbox descendants have exited; those descendants need not carry a matching `flashrsi_harnessrisk_*` command argument. Thus the written uncertainty can coexist with new model work being dispatched before exact descendant reconciliation.

This violates the explicit unknown-cleanup-blocks requirement while still correctly refusing H0 acceptance. The unit test checks that the host wrapper writes the receipt, but not that the queue consumes that condition and forbids the next dispatch.

Required fix: carry a typed cleanup-uncertain outcome/receipt from host/stage execution into the queue and stop all further model dispatch until explicit identity-based descendant reconciliation proves terminal cleanup. Retain the failed H0 and blocked tickets, and do not treat an empty process-name scan as reconciliation. Cover queue-level timeout-to-next-harness behavior with an isolated controller test. Do not alter or signal the active predecessors to implement this fix.

### Q-I2. Reused H0 admission does not bind its case/model protocol to the queue contract

Locations: `scripts/flashrsi_matrix_gpu1_queue_v17.py:89–120,329–341,379–382`.

`initial_acceptance()` checks exact 128 IDs, host name, copied snapshot ID, completion/privacy/mock/rule evidence, but does not compare the reference's `run_manifest.case_manifest_sha256`, model revision, port or applicable target budget/protocol fields with the queue's frozen case/backend contract. The queue independently hashes current DATA against its supplied case manifest, then accepts a historical H0 using only that manifest's ID set. Consequently a reference evaluated under different source bytes or model protocol but the same 128 IDs and host name can authorize reuse. Initial snapshot before/after hashes prevent drift during a stage; they do not establish that the reference belongs to the same source/model contract.

The real v11 reference manifest already contains `case_manifest_sha256`, `model_revision`, `flashrsi_revision`, `model_port`, temperature and target budget fields. This is a missing admission guard, not a claim that the observed reference is actually mismatched. The present controller fixture omits those fields altogether and still passes the new H0 gate, so the retained tests do not establish the requested matched source/reference pins.

Required fix: pass a trusted expected case/model/target contract into H0 validation. Verify the reference's retained case-manifest bytes/hash and declared protocol against the actual queue manifest/backend/harness contract before reuse, and bind those identities into the queue's frozen protocol/receipts. Apply the applicable checks to fresh H0s too. Add same-ID/different-manifest and wrong model-port/revision rejection guards without rerunning real cases or changing source/grading limits.

## Checked behavior and evidence

- Initial-v17 compiles the original full recipe with the entry's real `__file__`; ROOT remains the experiment root. The initial subprocess command indexing used for case/log location matches the original bridge. Reference-v18 composes the existing v6 OpenCode context-fit GEPA transform with benchmark-v17; reference-v17 remains the other algorithm entry. The required bounded-owner review reference exists.
- Predecessor tracking retains PID/start ticks/command identity. The queue explicitly excludes itself from lane-client checks; the judge wrapper and its evaluator do not reacquire the queue's inherited worker-lock descriptor. No ordinary self-wait or judge-lock deadlock was established from these paths. Judge lease/billing unavailability remains pending instead of blocking model progress or becoming a model-stall claim.
- Queue ownership and existing lane locks, existing-directory rejection and exclusive stage logs preserve prior outputs. Initial and algorithm source failures remain failed tickets; full-five-harness completion remains false and the finite queue exits nonzero while deferred/incomplete work remains. Claude/Pi are blocked rather than blindly rerun. OpenClaw uses caseenv-v3; Codex checks the declared parser dependency versions before dispatch. This review did not import or exercise those parsers.
- The queue targets the existing 18764 service and sets child CUDA visibility empty. No service start/restart, GPU0/2/3 use, grader change, target budget increase or permission relaxation was found in these new files. Source checks and frozen code copies remain present, subject to Q-I2's initial-reference binding gap.
- Judge admission requires exact 128 accepted IDs; cached successes are reused without overwrite. Live CPU judge/worker-lock activity was not modified. Algorithm/source validity and judge completion remain separate statuses.

Read retained `flashrsi_gpu1_queue_v17_tests_20261117.log`: **14 passed in 6.91s**, SHA256 `141ed448c5fb1959d72c014ea2fed1cab9ce4506f03dad3130490c093fb62fef`. No check was rerun. The tests replace external process/acceptance boundaries and do not prove actual predecessor waiting, native descendant cleanup, corrected full H0 behavior, algorithms, API judge outcomes or complete matrix delivery.

After the scoped fixes, actual launch/wait/first-case evidence and strict full-run/judge acceptance remain root-owned. Existing failed tickets and unresolved baseline work must remain retained; no complete experiment or full objective is claimed by this review.
