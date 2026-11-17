# Future-only GPU1 queue v20 implementation — 2026-11-17

Implementation and pure-controller verification checkpoint for independent review. No queue, model, native agent, candidate code, source task, Docker, GPU workload, API or network request was launched. No active/predecessor files, earlier queue, Pi/Claude implementation, vendor, old results, credentials or services were changed. The existing create-verification workflow was applied through the repository's reproducible controller tests; no new project skill was needed.

## Acceptance contract

`scripts/flashrsi_matrix_gpu1_queue_v20.py` composes the reviewed v19 body, retaining exact predecessor identity waiting before source freeze, the shared GPU1 lane lease, original128 source/manifest checks, Codex/OpenClaw actual32K profiles, existing OpenCode H0, immutable H0 checks around every candidate, sticky cleanup uncertainty, finite failure retention and cached judge/billing/worker-lease gates. The only endpoint is existing GPU1 port18764; child CUDA visibility is empty. No new service or budget is introduced. Cleanup uncertainty still blocks the whole lane, including continuation after failed H0; ordinary failed H0 blocks its matching six RSI tasks.

Pi now joins the same finite queue after OpenCode, OpenClaw and Codex. Its initial command is `scripts/flashrsi_harnessrisk_pi_native_owner_initial_v6.py --base FRESH_PI_BASE --snapshot ORIGINAL_RESOURCE`; the base receives the exact trusted case-manifest bytes. Every Pi algorithm uses `flashrsi_harnessrisk_pi_native_owner_reference_v6.py --algorithm NAME`, `HR_INITIAL_REFERENCE` pointing to that fresh H0, and `flashrsi_harnessrisk_host_pi_native_owner_v6.py`. The six names remain `opro`, `gepa`, `evoprompt`, `ace`, `skillopt`, `evo_harness`.

All seven Claude tasks remain explicitly `blocked_protocol_repair`; Claude is not counted as integrated or complete. `NATIVE_PROFILES` is the edit point for a later independently reviewed profile: add its real host/initial/reference, contract/profile, resource path/hash/ID, source-pins and admission/native-source callbacks. Routing, task deferral, coverage, source/resource binding and selected proof use that registry. Genuine Claude contracts and callback signatures must be reviewed before adding an entry. This version imports no Claude implementation.

## Original H0 resource and fresh admission

The explicit Pi input is `outputs/flashrsi_harnessrisk_4b_private_pi_gepa_interpreter_v4_20261114/pi_initial_full/initial_snapshot.json`. Its canonical path resolves to `outputs/flashrsi_harnessrisk_4b_private_pi_v3_20261114/pi_initial_full/initial_snapshot.json`; both paths are recorded. Pinned bytes SHA-256 is `285e34f76191883d544f9d3fc3397c62e6c8d1480bb3726beb9ab7f95718a201`, Snapshot ID `dd0db4bf42de4f0c6888fff1b287bb56d1d58832ceb7e21b285887d832f4ead2`, agent type `pi`. Only this resource is read from the old run; old cases and scores are never reused.

Trusted `flashrsi.core.snapshot.Snapshot.load` validates schema, content addressing, canonical file encoding and type without importing or executing Snapshot/native code. After the predecessor and lane wait, the queue retains the resource under its own `native_resources/`, records full profile source pins, and checks the original and retained bytes before and after each stage. Stage receipts include original resource bindings before/after plus inherited fresh-H0 Snapshot/manifest bindings. The final check records observed hash/type/ID even when drift causes rejection. The queue also rechecks cleanup, source cases, frozen sources, resource/profile and H0 immediately after endpoint waiting and before creating a child.

Pi H0 admission checks the complete trusted model/runtime/host/port/privacy/turn/time/reply/profile contract, exact case-manifest bytes and identities, valid original-resource Snapshot bytes/type/ID, complete128 receipt/count identity, exactly128 distinct valid attempt rows with no timeout/uncertain cleanup, and the case inventory. It then invokes the **actual reviewed `validate_reference`**, with `HR_OUTPUT_ROOT` temporarily set to the fresh Pi base and restored on both success and failure. That verifier recomputes code/manifest/source/owner/private/native per-case receipts for all128. Legacy `completed` alone cannot admit Pi.

The selected-run gate retains generic `inspect_run` and adds Pi manifest/host/initial-reference/lineage/strict-worker checks and actual `validate_case` admission for each selected case against its original source and selected Snapshot native-source map. The reviewed benchmark already applies these native receipt checks during evolution and full128; the queue rechecks selected evidence before judge eligibility. Canonical judging adds `llm_judge_multi_harness` beside cases, so the inventory permits that one known artifact directory while requiring all128 `harness_result.json` case directories and rejecting foreign directories/results. Root spot-review identified that concurrency interaction before handoff; both H0 and selected regressions cover it.

## Verification and evidence

The reproducible commands, from the experiment root, were:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts:src:vendor/FlashRSI .venv-rl-pilot/bin/python -m pytest -q tests/test_flashrsi_gpu1_queue_v20.py tests/test_flashrsi_gpu1_queue_v17.py
.venv-rl-pilot/bin/python -m py_compile scripts/flashrsi_matrix_gpu1_queue_v20.py tests/test_flashrsi_gpu1_queue_v20.py
```

Observed result: **50 passed in 1.72s**, exit 0: 34 new v20 checks plus 16 inherited v17 checks; compilation exit 0. Unit subprocess, host and native execution boundaries are stubs. One test invokes the actual trusted Pi reference verifier over 128 inert fixture cases while stubbing only its per-case native-evidence boundary; it verifies owner/source arguments and environment restoration. Other checks cover six dispatch names/commands, malformed/wrong-type/hash/ID H0 resources, contract/revision/host/attempt drift, selected native proof, canonical judge artifacts, exact predecessor waiting without freeze/dispatch, sticky cleanup, ordinary failed-H0 isolation, resource/profile/fresh-H0 drift and retained stage receipts, drift during endpoint waiting, cached judge reuse, finite billing retention, CLI help and existing-output preservation.

`flashrsi_gpu1_queue_v20_resource_20261117.log` separately records a resource-only check of the actual configured original Snapshot: trusted-core load/save to an isolated temporary directory preserved **identical bytes and ID**, and the original binding was unchanged after cleanup. No Snapshot code was imported/executed. That evidence remains after temporary-file cleanup.

Retained logs are `flashrsi_gpu1_queue_v20_green_20261117.log`, `flashrsi_gpu1_queue_v20_red_20261117.log` (expected missing-v20-module collection error), and `flashrsi_gpu1_queue_v20_environment_20261117.log` (first unqualified Python lacked pytest; verification then used the documented repository environment). The owned code/test/report review patch is `flashrsi_gpu1_queue_v20_review.diff`; it excludes unrelated workspace changes and its own generated copy.

## Remaining unverified behavior

No genuine queue waiting/handoff, model stage, Pi H0/full128, RSI candidate/full128, namespace/native compaction/tool-settlement path or paid judge was run. Controller tests and byte roundtrip do not establish empirical native runtime success or baseline improvement. Deployment/supersession belongs to the root after independent review, using fresh output and exact live predecessor identities. Old failures remain old failures; current full-five-harness coverage remains false.
