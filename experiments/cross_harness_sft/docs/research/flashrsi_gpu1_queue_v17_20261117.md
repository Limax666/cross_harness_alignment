# Finite GPU1 queue v17 implementation handoff

New files only. Original initial/algorithm/benchmark/host/scanner files remain unchanged. No queue, actual host/source program, API, model/GPU or Docker operation was launched by this implementation. Root owns independent review and launch; the active OpenCode ACE predecessor is not duplicated or stopped.

The finite plan is OpenCode EvoPrompt and GEPA after the current ACE predecessor, then a fresh corrected OpenClaw H0 and a fresh parser-dependency-corrected Codex H0. Each new H0 must pass strict actual128 identity/privacy/mock/rule/completion checks before that harness's six algorithms may run. Claude Code and Pi retain `blocked_protocol_repair` tickets with sampled failures and the bounded-owner review reference; neither receives an unchanged broken rerun. `ready.json` and protocol coverage explicitly say the full five-harness objective is incomplete. Exit status remains nonzero while any task is incomplete/deferred. There is no automatic retry or validity/status reinterpretation.

## Entry points and isolation

- `flashrsi_harnessrisk_initial_v17.py` transforms exactly the original full128 initial recipe's host subprocess bridge and metadata. Every actual host streams stdout/stderr into `host_observability/<case>/`, records command/return code/timing, and preserves original strict source/Snapshot/resume checks. A 2460-second outer deadline writes cleanup uncertainty and aborts, requiring exact descendant reconciliation before replacement. Original host2400-second limit remains. Actual Claude native cap16, native timeout600 and sandbox900 are recorded; OpenClaw/Pi retain12 and360, and Codex reports no explicit turn cap with native/sandbox360 instead of inventing a12-turn limit. No actual budget is changed.
- `flashrsi_harnessrisk_algorithm_reference_v18.py` composes v17 durable host observation with the existing OpenCodeContextfitGEPA v6 Worker. All original algorithm configs and target budgets are unchanged. Other algorithms use existing reference_v17.
- `flashrsi_matrix_gpu1_queue_v17.py` binds the existing18764 service, records current32K context/4096 maximum reply contract, and sets child CUDA visibility empty (target inference remains the existing GPU1 HTTP service). It starts no service and touches no other GPU, daemon or cron. Exact predecessor PID/start-tick identities, matching lane clients, the legacy lane lock and a new queue ownership lock prevent duplicate dispatch. Waiting is recorded before source snapshot work. Already-existing output directories are rejected without modifying their evidence.
- `flashrsi_matrix_repair_scan_v17.py --scan` registers v17/v18 entries and reads historical direct roots plus one level of versioned queue harness directories. It prints an updated strict matrix and queue tasks without writing the historical matrix/repair tickets, invoking a source program or traversing copied evidence graphs recursively.

OpenCode reuses only the genuine v11 initial reference with128 strict scored cases and128 accepted judge records. The reference host must match exactly. OpenClaw uses existing `host_openclaw_caseenv_v3.py`; Codex uses native_v3 with declared installed `tree-sitter==0.25.2` and `tree-sitter-bash==0.25.1`, checked before dispatch. Missing/drifting dependencies block that harness and preserve all RSI tickets. Context/protocol/budget failures remain invalid under the existing strict contract; this queue does not claim the new H0s will all pass.

## Evidence and feedback

Before execution the queue freezes copies/hashes of FlashRSI scripts/runtime assets and the actual HarnessRisk adapter/evaluator source, verifies all128 original source case hashes, and pins each harness manifest. Every stage records its source protocol hash, manifest hash, initial-reference snapshot hash before/after, command, PID, return code and timing. Source/Snapshot drift raises a typed blocker. PID board and task state preserve live owned children if an unexpected exception leaves one running; no replacement is attempted.

Every blocked H0 keeps the partial complete128 attempt log, exact case evidence, classified errors, fingerprint, repair-needed flag and `unchanged_retry_allowed:false`; its six algorithms are explicitly blocked. Independent corrected harnesses can advance. Failed algorithm evidence remains retained. No failed/HTTP/context/dependency episode is silently scored zero or rewritten completed. Successful source stages need judge acceptance separately.

Judge work uses the existing shared `worker.lock`, existing billing cooldown and existing cached-success wrapper without `--overwrite`. Lease/billing unavailability leaves `needs_judge` and allows model work to continue. Only this queue's new successful batches are scheduled. Judge records require exact128 accepted case IDs; existing successful records do not incur repeated API calls. The queue drains eligible owned judge work at the end and leaves lease/billing blocked items explicit rather than claiming completion.

## Verification

Used the installed create-verification-skill workflow and existing controller verification pattern. No new skill was necessary. Pure tests replace external process/acceptance boundaries and use isolated fixture files; they do not supply experimental or dataset readiness evidence.

From MAIN experiment directory:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts:src:vendor/FlashRSI CUDA_VISIBLE_DEVICES='' .venv-rl-pilot/bin/python -m pytest -q tests/test_flashrsi_gpu1_queue_v17.py tests/test_flashrsi_harnessrisk_benchmark_v17.py
```

Final result is retained in `flashrsi_gpu1_queue_v17_tests_20261117.log` beside this report. Coverage includes unchanged128 transform, actual budget metadata, GEPA composition, retained failed/timeout host bytes, timeout cleanup blockade, strict H0 identity/mock/reset/extra-case checks, exact judge coverage, source drift, billing deferral, existing-output preservation and the finite dispatch order with blocked Claude/Pi and H0-gated RSI. Scoped diff check applies only to these new files because the shared main checkout has unrelated existing whitespace failures.

Real process waiting/locks, actual first corrected H0, kernel sandbox/tool/parser behavior, native context behavior, all algorithms, judge lease/API outcomes and durable task feedback over a full run remain unverified. Root's full authorized run must provide that evidence. Tests are not a formal proof or a claim of full matrix completion.

## Root launch after independent review

Use a fresh independent output and the exact active predecessor identities (omit a PID only after verifying it exited):

```bash
PYTHONPATH=scripts:src:vendor/FlashRSI CUDA_VISIBLE_DEVICES='' .venv-rl-pilot/bin/python -u scripts/flashrsi_matrix_gpu1_queue_v17.py --out outputs/flashrsi_gpu1_finite_queue_v17_20261117 --predecessor-pid 3839693 --predecessor-pid 3839699
```

This is a handoff command, not a command executed by this implementation. Read `launch_state.json`, `pidboard.json`, `remaining_tasks.json`, stage receipts and actual host logs; process existence/exit is not acceptance. Any future bounded-owner profile needs independent review plus a fresh matched H0 and RSI set; old strict or partial histories must not be mixed.

## Independent review fix round1: Q-I1 / Q-I2

Read `flashrsi_gpu1_queue_v17_review_20261117.md` in full and used the receiving-code-review workflow to verify both findings before changing the unused queue code. Active ACE's `benchmark_v17.py` and `algorithm_reference_v17.py`, all original runtime/initial/host files, live judge and CPU source QA were untouched. No actual queue, host, source program, process signal, model/API/GPU/Docker operation or native bounded-owner profile was run or implemented.

**Q-I1 fixed:** initial and algorithm timeout/explicit cleanup-false/live-owned-stage receipts now cause typed `QueueCleanupUncertain`, a persistent `queue_cleanup_block.json`, false dispatch-ready marker and queue-wide blocked tickets before the next dispatch. The queue consumes these receipts after every stage and before further model or new judge dispatch. An empty process-name scan cannot clear the sticky blocker. Existing-output restart remains forbidden. There is deliberately no automatic reconciliation or unblock; root must retain and reconcile exact owned process/descendant identities before authorizing a fresh queue. This implementation does not pretend a receipt or a name scan proves descendant absence.

The inherited algorithm benchmark could otherwise continue its next case before returning control to the queue after a timeout. New tiny `benchmark_v18.py` subclasses the immutable v17 bridge and raises `HostCleanupUncertain` immediately after v17 retains a timed-out host receipt. Ordinary future stages now use new `algorithm_reference_v19.py`; unused OpenCode GEPA `reference_v18.py` composes the same guard with the existing context-fit Worker. The new scanner registers v19 too. Original active v17 code remains unchanged. The guard does not rewrite trajectory statuses, change rules or budgets, or fabricate zero scores.

**Q-I2 fixed:** H0 acceptance now takes an explicit trusted contract, including the SHA256 of the actual supplied case-manifest bytes, model revision, FlashRSI revision, harness, exact host, model port18764, temperature0, reply cap4096, applicable native turn/timeout limits, owner protocol, privacy protocol and128-case count. Both the retained H0 parent case-manifest bytes and the H0's declared manifest hash must equal that trusted raw-byte hash. Every required field must actually exist, including Codex's explicit null turn cap. The retained v11 manifest's declared12-turn4096-token protocol is checked using its available fields; fresh corrected H0s additionally require their explicit native/sandbox timeout metadata.

The actual trusted bytes are preserved verbatim at queue and harness roots. Expected H0 contracts, trusted source identity and retained OpenCode reference bindings enter the frozen queue protocol. Stage receipts bind the expected H0 contract and retained Snapshot/run-manifest/case-manifest hashes before and after execution. Algorithm stages revalidate the complete H0 contract immediately before launch and reject mid-stage reference drift. Same128 IDs with changed source hashes, wrong model/runtime revision, wrong port, harness, temperature or budgets cannot authorize reuse.

TDD retained expected failures: `flashrsi_gpu1_queue_v17_fix1_red_20261117.log` (two missing queue-blocker failures, one existing pass) and `flashrsi_gpu1_queue_v17_fix2_red_20261117.log` (one missing trusted-contract guard failure). Those remain failure evidence; they are not counted as passes.

Final affected command, from MAIN experiment directory:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts:src:vendor/FlashRSI CUDA_VISIBLE_DEVICES='' .venv-rl-pilot/bin/python -m pytest -q tests/test_flashrsi_gpu1_queue_v17.py tests/test_flashrsi_harnessrisk_benchmark_v17.py
```

Result: **19 passed in 0.76s**, retained `flashrsi_gpu1_queue_v17_fix_round1_20261117.log`. Code/report-only scoped diff check passed. The broader staged check remains failed on two trailing-whitespace formatting lines in the raw fix2 RED pytest transcript; that original failure evidence is preserved unchanged. Focused new guards exercise initial timeout preventing Codex dispatch, algorithm timed_out-only receipts preventing even GEPA dispatch, persistent blockade despite empty lane scans, immediate within-rollout timeout abort, real-shaped complete H0 contracts, same-ID/different-manifest bytes, wrong port/revisions/harness/temperature/turn/reply caps, and guarded entry composition. Only process/controller boundaries and isolated fixtures were used; no original source bytes were altered.

Real descendant cleanup/reconciliation, predecessor wait/locks, full corrected initial/RSI behavior, native context contracts and judge outcomes remain root-owned and unverified here. Current strict validity and blocked Claude/Pi tickets remain unchanged. Independent rereview is required before the actual queue launch.

Explicit owned predecessor identities now retain only the safe algorithm run scope derived from PID/start-tick-bound harness/output/algorithm metadata. After that predecessor exits, its bounded host receipts are checked before the first new dispatch too. A historical external v17 timeout therefore cannot be forgotten merely because its process name disappeared. The additional isolated guard verifies this external-scope blockade; no active predecessor was inspected through a source program, modified or signaled.
