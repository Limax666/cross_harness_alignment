# Future GPU1 queue v21 registration checkpoint — 2026-11-17

This is a future controller prepared for independent review. **Claude v9's own independent review is pending; registration is not native runtime acceptance or permission to bypass that review.** Root controls deployment after both reviews. No queue/workload/native agent/source task/model/API/network/GPU/Docker operation was launched. Only owned new v21 code, tests, report and verification evidence were written. Reviewed v20/Pi/Claude implementations and old results remain unchanged.

## Actual integration contract

`scripts/flashrsi_matrix_gpu1_queue_v21.py` exactly composes reviewed v20 (`cfa44ec`) in its own module globals and registers Claude after Pi. Protocol, queue launcher and private launcher lock become `gpu1-finite-strict-matrix/v21`, `scripts/flashrsi_matrix_gpu1_queue_v21.py`, and `.flashrsi_gpu1_finite_queue_v21.lock`. The common GPU1 lane lease, exact predecessor identity wait before source/resource freeze, existing endpoint18764, frozen source/case/H0/profile/resource checks, post-wait pre-dispatch validation, sticky lane-wide cleanup barrier and finite cached-judge/billing/worker-lease behavior remain inherited. Importing the module does not invoke a queue or native entry point.

The model harness order is OpenCode → OpenClaw → Codex → Pi → Claude Code. OpenCode retains its two pending EvoPrompt/GEPA tasks and the reviewed exact existing H0. Each of the other four harnesses has a fresh H0 followed by all six unchanged names: `opro`, `gepa`, `evoprompt`, `ace`, `skillopt`, `evo_harness`. All 30 tasks remain explicit. No harness is deferred by this future registration; full-five-harness objective completion still remains false until actual evidence proves it. Root can inspect the explicit `remaining_tasks.json`, `repair_protocol.json`, `predecessors.json`, `ready.json`, stage receipts and pidboard, and the new read-only v21 scanner described below.

Claude's actual current files were read, and the implementation agent confirmed stable interfaces:

- Host: `flashrsi_harnessrisk_host_claude_native_owner_v9.py`.
- Initial: `flashrsi_harnessrisk_claude_native_owner_initial_v9.py --base FRESH_BASE --snapshot ORIGINAL_RESOURCE`, with `HR_HARNESS=claude_code`, exact host and port18764. The fresh base receives the trusted manifest bytes.
- RSI: `flashrsi_harnessrisk_claude_native_owner_reference_v9.py --algorithm NAME`, using that fresh `HR_INITIAL_REFERENCE`; strict evolution, one case attempt and Worker1024 are inherited.
- Trusted callbacks: initial `source_pins()`, reference `validate_reference(initial)`, helper `PROFILE`, `CONTRACT`, `validate_case(...)` and `snapshot_source_map(...)`. These are actual trusted modules; no Snapshot/native candidate code is imported.

The current Claude initial manifest exactly fits shared native admission: `max_turns_per_owner_turn=16`, `native_timeout_seconds_per_owner_turn=600`, `sandbox_wall_seconds_per_owner_turn=900`, reply4096, temperature0, port18764, original model/runtime/case-manifest bindings, execution protocol and full native contract. Its attempt rows use `returncode` and `measurement_valid`; its completion receipt uses `status=all_cases_attempted`,128 attempts/scored,0 failed/unscored, exact Snapshot ID and `all128_measurement_valid=true`. Shared admission checks all those fields and the exact128 case/attempt identities, then invokes the real Claude reference verifier with scoped/restored `HR_OUTPUT_ROOT`; that verifier rechecks source pins and all128 source/owner/native/private/host receipts.

The actual Claude reference transform emits `target_execution_protocol`, **`target_contract`**, `initial_reference_dir` and initial Snapshot ID. v21 additionally requires its exact `target_contract` during selected proof. Existing generic selected acceptance and v20 native selected source/Snapshot/lineage/strict-worker/per-case evidence revalidation remain. Both native inventories retain the known canonical judge artifact directory exception while rejecting foreign cases.

## Original resource bindings

Claude resource is `outputs/flashrsi_harnessrisk_4b_private_claude_cli_projection_v8_gpu1_20261115/claude_code_initial_full/initial_snapshot.json`. Its SHA-256 is `45d6efd89a4b0f3696ec2ad79bcc06c43c6e125890ed43251a0d462c4580713d`, Snapshot ID `26a1fb7c061e3aec310e5fb266d9083c5b5ab8a754dec4b4ae292a1792357f3b`, agent type `claude_code`. The canonical path is identical to that configured path. Pi's reviewed resource path, canonical path, SHA and ID remain unchanged. These agree with root's `outputs/flashrsi_gpu1_baseline_resume_20261117/native_h0_resource_bindings_20261117.json`.

Only Snapshot resource bytes are read from those old runs; old cases, native observations and scores are never imported. Trusted core loading checks content-addressed schema/type/ID. The shared controller retains both resources and exact profile source pins only after predecessor/lease waiting, verifies original/retained bytes and profile source pins around every candidate, and records observed original resource bindings before/after. Neither resource's embedded code is executed by admission.

The narrow v20 review diagnostic observation is addressed only in v21's composed body: `failure_feedback` now recognizes successful native `returncode=0, measurement_valid=true` rows as well as legacy `exit_code=0`, and still requires a completed case result. A valid bounded native outcome therefore does not produce a false diagnostic failure. Invalid native rows and existing failure evidence remain preserved; this condition is diagnostic only and does not change acceptance or retry policy.

## Read-only scanner registration

`scripts/flashrsi_matrix_repair_scan_v21.py` extends the reviewed v19 scanner without changing it. It registers finite queue roots v17 through v21 and examines exactly one harness-directory level inside each queue root. It reports queue protocol, launch state, complete retained task manifest and launcher identity status. It never scans copied sources/snapshots/cache/evidence recursively and never writes matrix state, calls a tick/repair loop, schedules work, or launches a subprocess.

Both actual native reference names are included in active-run registration. Their PIDs are attached only through the trusted identity-derived `run_scope`, which requires the recognized harness/algorithm, port18764 and actual `HR_OUTPUT_ROOT`. New native host/profile markers or a Pi/Claude run under a v20/v21 queue require v21's strict selected proof. Historical non-native runs retain legacy inspection. Live future queued jobs are surfaced as `queued_launches`; a queue owns those pending jobs only when the current process identity exactly matches retained pidboard PID/start/command and its launcher script and `--out` scope match the root. A reused PID does not claim queue ownership. H0 tasks remain fully visible in the queue's retained task manifest.

The eventual read-only command is `PYTHONPATH=scripts:src:vendor/FlashRSI .venv-rl-pilot/bin/python scripts/flashrsi_matrix_repair_scan_v21.py --scan`. It was **not** executed against live outputs or processes here; root controls that empirical observation after deployment/review.

## Reproducible verification

The installed create-verification workflow was applied using the existing project controller-test workflow. Commands from the experiment root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts:src:vendor/FlashRSI .venv-rl-pilot/bin/python -m pytest -q tests/test_flashrsi_gpu1_queue_v21.py tests/test_flashrsi_gpu1_queue_v20.py tests/test_flashrsi_gpu1_queue_v17.py
.venv-rl-pilot/bin/python -m py_compile scripts/flashrsi_matrix_gpu1_queue_v21.py scripts/flashrsi_matrix_repair_scan_v21.py tests/test_flashrsi_gpu1_queue_v21.py
git diff --exit-code -- scripts/flashrsi_matrix_gpu1_queue_v20.py tests/test_flashrsi_gpu1_queue_v20.py
```

Observed **78 passed in 3.89s**, exit 0: 28 new v21 checks and all 50 existing v20/v17 checks. Compilation and reviewed-v20 unchanged check exited 0. Retained output is `flashrsi_gpu1_queue_v21_green_20261117.log`; `flashrsi_gpu1_queue_v21_red_20261117.log` records the expected missing-v21-module collection failure before implementation. `flashrsi_gpu1_queue_v21_scanner_red_20261117.log` records the four expected missing-scanner-module failures before scanner implementation.

The new tests cover actual registry/commands/budgets/manifest/reference field names, both actual trusted reference verifiers walking 128 inert fixtures with only the per-case native boundary stubbed, environment restoration after success/drift rejection, Claude host/profile/model/turn/port/attempt/completion/resource mismatch rejection, all-five finite routing with every failure retained, failed-H0 isolation, sticky timeout, selected target-contract/native proof, canonical judge artifacts, both resource and retained-copy bindings, exact predecessor waiting before freeze, imported main-guard/process/native-import inactivity, CLI help/existing-output preservation and native feedback schema adaptation. Scanner fixtures additionally prove native reference scope/PID mapping, both nested v20/v21 native runs discovered exactly once, strict native proof/legacy fallback, both queue state/task manifests retained, live queued ownership versus reused PID rejection, no recursive output traversal and no writes. Unit dispatch/profile/host/process boundaries are stubs; no fixtures are claimed as experiments.

`flashrsi_gpu1_queue_v21_resource_20261117.log` records a separate resource-only check of both actual configured Snapshots: trusted-core load/save to isolated temporary files preserved identical bytes/IDs, original bindings remained unchanged, and evidence survives temporary-directory cleanup. No native agents or candidate code were imported/executed. The owned review diff is `flashrsi_gpu1_queue_v21_review.diff`; it excludes unrelated workspace changes and its own copy.

## Remaining empirical gates

Claude implementation review is still pending at this checkpoint. Genuine queue CLI waiting/lease/supersession, Claude/Pi H0 and six RSI algorithms, selected full128, native continuation/compaction/context/tool/mock/provider/namespace cleanup and actual cached/paid judge outcomes remain unverified here. No process existence, helper test, source hash or resource roundtrip is presented as runtime/full128 acceptance or baseline improvement. Root must complete independent reviews and preserve the live predecessor/active ACE before any waiting-queue replacement or genuine dispatch.

## Fix round 1 — current queue completion guard for I1

The independent v21 review identified Important I1: historical `complete` precedence could leave top-level `all_complete=true` while an identity-matched live queue owned fresh queued/running/failed/blocked work. This round changes only the owned v21 scanner, its covering tests and this report/evidence. The reviewed v21 queue, v20/Pi/Claude implementations and older scanners are untouched. Review base for this fix is `5855389`; independent re-review is still required.

Historical job status/count precedence is preserved and explicitly labelled `aggregation_mode=best_historical_verified_run`, both globally and per job. Each historical complete job exposes its chosen verified run's `path` and `protocol` in `best_historical_verified_run`; `historical_all_complete` retains that coverage reduction. No historical result or run record is erased.

`current_queue_jobs` separately contains every retained task from exactly identity-matched live queues, including H0, with job ID, raw retained task/evidence, current status, queue path/protocol and matched launcher identity. `current_queue_counts` counts these current statuses; `current_queue_status` is `complete`, `incomplete`, or `no_identity_matched_live_queue`. Queued, running, failed and blocked states remain explicit regardless of historical completion. Multiple live queues retain separate rows. An empty/unreadable task manifest in an owned queue becomes `invalid_task_manifest`, which blocks completion rather than disappearing.

Top-level `all_complete` now requires **both** historical coverage and every identity-owned current task to be complete. A reused/orphan PID owns no current tasks; retained queue data still remains visible in `queues`, and the no-owned-live-queue status labels the historical-only case. A live process is never completion evidence. A retained `complete` algorithm row additionally requires its exact current run to pass actual scanner source/judge inspection; an already inspected matching run is reused without scanning again. A retained `complete` H0 requires the trusted queue-manifest contract, actual H0 admission and exact cached judge acceptance. These callbacks read evidence and retain all existing native/legacy admission gates. Missing/mismatched source or judge proof produces `current_completion_not_verified` while preserving the original retained task row.

The pre-fix targeted regressions recorded eight failures in `flashrsi_gpu1_queue_v21_fix1_red_20261117.log` (the prior scanner lacked the newly required source-proof callback imports/current view). The final reproducible commands were:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts:src:vendor/FlashRSI .venv-rl-pilot/bin/python -m pytest -q tests/test_flashrsi_gpu1_queue_v21.py tests/test_flashrsi_gpu1_queue_v20.py tests/test_flashrsi_gpu1_queue_v17.py
.venv-rl-pilot/bin/python -m py_compile scripts/flashrsi_matrix_repair_scan_v21.py tests/test_flashrsi_gpu1_queue_v21.py
git diff --exit-code 5855389 -- scripts/flashrsi_matrix_gpu1_queue_v21.py scripts/flashrsi_matrix_gpu1_queue_v20.py tests/test_flashrsi_gpu1_queue_v20.py
```

Observed **88 passed in 3.10s**, exit 0; compilation and unchanged reviewed-queue checks exited 0. The final log is `flashrsi_gpu1_queue_v21_fix1_green_20261117.log`. New pure fixtures cover all 40 accepted historical jobs plus fresh queued/running/repair/cleanup-blocked states; failed H0 and all six blocked algorithms; complete current H0/algorithms with successful versus failed source/judge proof; unreadable/empty owned task manifests; and reused PID rejection. Source/profile/judge callbacks and all process observations remain unit boundaries, not fabricated experiments. No actual scan, process inspection, native/source/model/GPU/API/network/Docker operation or queue launch was performed. The scoped fix patch is `flashrsi_gpu1_queue_v21_fix1_review.diff`, limited to scanner/tests/report changes against `5855389`.
