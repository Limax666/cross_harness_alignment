# Task 4 — guarded mixed release/export

Status: implementation delivered for review; **actual 14,000-record release is NOT complete or ready**. Both new full admissions, selected exported environment audits, and selected context audits remain release gates. No source tool/checker, Docker, model, GPU, API, optimizer or simulated experiment was executed by this task.

Owned files: `scripts/hsrl_v5/{dataset,reward,sampler,release}.py`, `tests/test_hsrl_v5_dataset.py`, `configs/hsrl_v6_dataset.yaml`, `docs/hsrl_v6_dataset_usage.md`, all under `experiments/cross_harness_sft`. Root drafts of dataset/reward/sampler were retained and hardened. Unrelated dirty files and running workloads were preserved. Root owns the verification skill, active QC launchers, Claw thread repair and runtime/context audit.

Implementation commit: `3ce0f71` — `feat(hsrl): gate real mixed release exports on complete admission and audits`. This report is recorded in the following dedicated documentation commit.

## Implementation

- Read-only `python -m hsrl_v5.release ... --status`, bounded optional CPU-file `--wait`, fail-closed guarded `--stage`, certificate-bound `--publish`.
- Fixed source manifests, all source metadata file hashes, protocol/source record identity, frozen QC code, actual successful evaluator/reset/state/history controls. Source errors remain pending; incomplete and engine-blocked jobs cannot authorize fast-case selection.
- Actual admitted selection: 7000 Claw benign, 5500 AgentDoG attack, 1000 same-base-family clean, 500 query targets; seed 20261116, declared candidates preferred, family round robin and stable IDs. No old v5 safety QC reuse, generated 14K rows, holdout borrowing or source-fold changes. Whole Claw benchmark-collision families are removed using cached public hashes, including family `35c7ebf6dc3a23022c6b0cb5`; cross-source active dev/test request hashes are excluded.
- Candidate package copies original unchanged Claw trees, source bundles, complete case evidence, QC frozen source snapshots, transport failure proof, private source records and grader/vendor provenance. Relocated records preserve semantics and only change allowed active references/runtime admission. `train.parquet` carries public prompts/tools and opaque references; its role/lookup/mask metadata is private to the collector.
- Frozen code hashes gate in-process consumers; a frozen-code subprocess audits candidate file identities, all public renders and all 875 sampler blocks. Candidate `Dataset.for_verification` retains real source/runtime guards and rejects both training objective adapters. Ordinary `Dataset` requires detached READY, final manifest and both certificates. Final publication checks exact immutable core/selection/source identities and normal frozen constructor before atomic rename.
- GRPO and CHS-PO share the same vector contract and masks. Query-target legal utility stays null with the utility policy-gradient mask false. Invalid/error/nonfinite/NA applicable values raise. No persistence/broad-judge/actual-harm training claim or zero imputation.

## Verification commands and observed outcomes

Working directory for commands below: `/data/home/liumingxiao/hsrl_v3_worktree_20261116/experiments/cross_harness_sft`.

Interpreter: `/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python` (the main `.venv` is a Windows layout; default shell Python and `.venv-rl-native` lack pytest). Initial interpreter discovery failures did not alter environments or install anything.

TDD: initial root draft tests passed; new release API tests failed first with `ModuleNotFoundError: hsrl_v5.release`. Candidate/objective/terminal guard tests failed on missing interfaces. The missing registered-risk-control regression failed with `DID NOT RAISE` before the proof guard was added. Runtime-byte and per-role mask API tests failed on their missing helpers before implementation. Read-only/mutating-mode conflict regression failed on the previous parser error before mutual exclusion was implemented. These are pure guards or real retained fresh-v6 source/proof checks, without fabricated experiment records.

```bash
PYTHONPATH=scripts /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q tests/test_hsrl_v5_dataset.py
```

Retained logs: `task-4-evidence/tests.log` (17 passed, 41.72s) and final `task-4-evidence/tests-final.log` (**18 passed in 42.99s**, exit 0).

```bash
PYTHONPATH=scripts /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q tests/test_hsrl_v5_dataset.py tests/test_hsrl_v5_agentdog.py tests/test_hsrl_v5_claw.py tests/test_hsrl_v5_workspace.py
```

Outcome: **109 passed in 86.01s**. Retained log `task-4-evidence/component-tests.log` SHA256 `313ff0da8de322cf572259e78fd7ccb98fbd69cf578a4465a14e284fa674c510`. The entire unrelated repository suite was not run: this assignment prohibits source/model/API/container execution, so verification used the inspected affected pure data/guard/component tests. This is scoped test evidence, not a full repository pass.

Real read-only status (stdout redirected to `task-4-evidence/status-interim.json`):

```bash
PYTHONPATH=scripts /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m hsrl_v5.release \
 --claw-source outputs/rl/hsrl_v5_claw_prepare_shell_contract_fix1_20261116 \
 --claw-quality /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/rl/hsrl_v5_claw_runtime_quality_boundary_v2_20261116T193656 \
 --agentdog-source /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/rl/hsrl_v6_agentdog_components_prepare_20261116 \
 --agentdog-quality /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/rl/hsrl_v6_agentdog_real_quality_20261116_r1 --status
```

Outcome: exit **0**, status **not_ready**, no proof identity errors. Snapshot counts: Claw 765 completed/388 admitted/6873 unprocessed; AgentDoG 3039 completed/2727 admitted/7407 unprocessed. Claw consumer execution bytes differ from the interim QA, as intended; the report explicitly requires new v4 Claw QC. Interim Claw is status evidence only and cannot be staged. SHA256 `5f773b0409ec05f7011518a3a992d33353a7c6b5e2c6aa2075dc1c8e5d0fd090`.

The first status attempt failed closed on an unpaired inventory row without a split. The active-holdout scan was corrected to explicit dev/test folds and the complete CLI rerun above succeeded; that earlier error is not represented as success.

Real guarded staging against the planned final QA directory (same AgentDoG/source arguments, stdout `task-4-evidence/blocked-export.json`):

```bash
PYTHONPATH=scripts /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m hsrl_v5.release \
 --claw-source outputs/rl/hsrl_v5_claw_prepare_shell_contract_fix1_20261116 \
 --claw-quality /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/rl/hsrl_v6_claw_real_quality_20261116_r1 \
 --agentdog-source /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/rl/hsrl_v6_agentdog_components_prepare_20261116 \
 --agentdog-quality /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/rl/hsrl_v6_agentdog_real_quality_20261116_r1 \
 --stage --out /data/home/liumingxiao/hsrl_v3_worktree_20261116/.superpowers/sdd/2026-11-16-hsrl-v5-ready-dataset/task-4-evidence/blocked-candidate \
 --grader /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/vendor/agentdog_runtime_v5_c8d803f/runtime_evaluators.py
```

Outcome: exit **2**, `full_source_quality_terminal_completion_required`. Final Claw protocol was absent at the read time; AgentDoG was still processing. Observed no candidate directory and no READY. The structured `blocked-candidate.shortfall.json` survives, byte-identical to stdout SHA256 `e26982a4b0840f6b99627875bc5dff8f8b21b75bb41507646d8d33dc50a5ae29`. Its role shortfall was Claw 7000, attack 3158, clean 625, query 280. These are moving-job snapshot values, not final failures or a final quality conclusion.

`git diff --check` passed. Evidence directories are retained and no cleanup deleted them. They are local ignored proof artifacts; the report records their paths and hashes.

## Remaining actual release gates and limits

1. Both complete new source jobs, source-terminal proof coverage, enough admitted cases in each role, and exact source-execution bytes matching frozen QC. Claw requires new v4 QC and unchanged v2 private boundary; superseded runtime proofs cannot pass final export.
2. Stage the genuine 14K artifact. The successful full package/Parquet path is **not yet empirically exercised**, since manufacturing data or selecting incomplete QC would violate the release contract.
3. Root's real selected exported loader/render/environment/score/reset/privacy audit, plus complete 875-block actual 14K coverage; root's selected context audit across all 18 configurations and fixed 16384/2048/8192/12 limits. Certificates must bind selection, source pins and immutable candidate core digest. No certificate was fabricated.
4. Publish atomically and run the normal consumer against the actual final paths. Final publication and training remain **unverified**. The config/docs give exact staging/publish/collector commands and APIs; they do not claim stock final-answer-only verifiers or old v3 actor compatibility, nor provide an unimplemented optimizer command.
5. Root's independent final code/spec/privacy review and verification-skill exercise remain root-owned. No separate agent was spawned by this task.

## Review round 1 — I1, I2 and I3

Review-fix implementation commit: `66b5afe` — `fix(hsrl): bind audited semantics and preserve immutable publication candidates`.

Read `task-4-review.md` and Task 6's report/commit `3478fa6`. The implementation changes only this task's owned release, dataset, tests and usage documentation. Task 6's runtime, grader-pool and QC files were read without modification; running source admissions were not stopped, replaced or driven by these tests.

### I1. Audit certificates do not bind runtime-defining manifest metadata

Added a cycle-free `hsrl-v6-release-semantic-core/v1` descriptor and `semantic_core_sha256`, required in BOTH certificates in addition to the existing file-map digest. It binds schema/reward contract, full source/selection identities and seed, quotas, limits, sampler/consumer compatibility, admission report, grader path and fixed SHA, every runtime module hash, the independent minimum module set and all immutable core file hashes. Status/READY/certificates are outside this identity, preventing a self-hash cycle while binding the fields that select runtime behavior.

The minimum runtime closure includes `grader_pool.py`, actual environment/worker/source modules, pure v3 dependencies and the AgentDoG source-control helper. Every packaged Python module under `code/` must be covered, and every independent required module must be present even if an edited manifest removes its file-map entry too. Loader/certificate validation independently verify the pinned grader's actual copied bytes; `episode()` repeats the pin check before use. `Dataset._load` recomputes selection identity from loaded train rows, rather than trusting the manifest's declared digest.

Tests use genuine source-record/retained-code metadata and copied pinned grader bytes. They reject changed grader selection, reduced or resealed runtime maps, inconsistent or edited row selection, changed limits, changed copied grader bytes, and a resealed semantic descriptor paired with an unchanged certificate. The one-record metadata inputs are pure negative guard fixtures: they are never Dataset/14K fixtures, cannot pass admission/sampler construction, and do not constitute an actual audit certificate or published release.

### I2. Publication makes the existing candidate training-ready before the final atomic rename

Publication now copies the source candidate into a fresh private `0700` directory under the final output's parent. All re-auditing, certificate installation, ready-manifest/READY assembly and normal frozen-consumer validation occur in that copy. The original known candidate is never rewritten, receives no certificates or READY, and needs no rollback or metadata repair after an interruption. Failed private staging is retained. Linux `renameat2(RENAME_NOREPLACE)` atomically installs the validated copy and refuses replacement of an existing destination, including an empty directory.

Filesystem transaction tests operate on copies of genuine retained cases, without a fabricated Dataset, source execution or 14K release. They inject interruption after both metadata writes, failure at the consumer-validation callback boundary and failed final installation, and verify unchanged original hashes and absent final output. An isolated trusted helper process is SIGKILLed after its private READY write: the original remains unchanged, the final path remains absent, the private directory remains `0700`, and a retry succeeds from the unchanged original bytes. Separate tests exercise real successful copy/install and no-replacement behavior. The SIGKILL targets only that test-created process; no running user/QC workload is signaled.

### I3. Admission checks flags without validating all corresponding retained private grader/reset evidence

AgentDoG validation now requires the actual initial negative grade, both first-episode private grades, their pinned grader/boolean contracts, exact source-control provenance and executed names/arguments, result/private-grade consistency, applicability/metric masks, source state-transition hashes, actual public observations, initial/source identity and distinct reset process identities. Attack/query cases require the retained reset private grade, reset result and private final state, matching actual risk-control calls and registered-risk-positive outcome. Clean reset verification uses its actual recorded replay and public results; the genuine runner records no clean reset grade/result and none is manufactured. Clean grades also enforce the source's `never` registered-risk rule.

Claw validation parses the retained raw checker response through the existing pure parser and matches the raw scores and utility to each actual/reset result, the reset equality claim and untouched-workspace grade. It checks source/public/interface and snapshot hashes, immutable references, actual source-read/private-visibility control, fresh actor container identities and confirmed cleanup. Pooled v5 cases additionally require case-local request/projection/response/source-snapshot/cleanup evidence; bind original and projected checker/snapshot manifests and hashes; match request nonces and child PIDs to the actual responses/result/proof; and require distinct actual/reset children/nonces in the same private pool container. Final export now requires the pooled Claw v5 protocol and confirmed dispatch-wide pool cleanup. Whole case evidence and pool lifetime logs are packaged.

Closed genuine evidence from `hsrl_v6_claw_real_quality_20261116_r2` was available: the first risk-metrics case (`d1da9ddfe86562747e63dfe389ae665bfa2608b636f65a9d4bc6469015c5cdb6`) has actual child PIDs 7/8 and confirmed exact-container cleanup. Tests copy that real case and reject deleted request/projection/response/checker-snapshot/cleanup files, and coherent mutation of all corresponding reset metadata to reuse the first request nonce or child PID. Fresh v6 AgentDoG cases are copied for deletion/corruption checks of initial, positive and reset risk grades/results and clean replay evidence. Claw v6 r1 real baseline cases cover malformed raw checker response, inconsistent reset utility and contradictory untouched-workspace score. No source checker, tool payload, container, judge, model, API, GPU or optimizer is executed.

Metadata reads use a read-only descriptor anchored to the already validated attempt directory. Intermediate and leaf entries are opened with `O_NOFOLLOW`; only regular metadata files are accepted. This avoids repeated full-host-path resolution for every private proof file and also rejects intermediate/leaf symlinks without a path-resolution race beneath the anchor. Genuine-copy symlink regressions pass. Source bundles are hashed and parsed from the same read bytes. Pending per-case pool-removal confirmations are reported as pending retained evidence during a running admission, never as admitted or silently failed cases.

### Review verification and evidence

Interpreter and work directory remain as above. All following test invocations additionally set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts CUDA_VISIBLE_DEVICES=''`.

Observed RED regressions before fixes: 17 targeted I1/I3 failures (missing semantic identity helper, missing reset grades/results/initial grade accepted, corrupted private outcomes accepted, clean replay corruption accepted, malformed/mismatched Claw grades accepted); 5 publication helper/atomic-install failures before I2; and a clean-`never`-rule corruption accepted before its guard. One intermediate implementation run had 7 real-case baseline failures because source control provenance includes an inert `source_path`; this was corrected by binding the complete provenance to the pure source helper and comparing executed name/arguments separately, rather than dropping provenance checks.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts CUDA_VISIBLE_DEVICES='' /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q tests/test_hsrl_v5_dataset.py -k 'missing_grade_controls or corrupt_private_outcomes or clean_real_reset or binds_raw_checker or private_publication or publication_install'
```

Outcome: **18 passed, 22 deselected in 11.78s**, retained in `task-4-evidence/review-guards.log`.

The genuine pooled-case deletion/nonce/PID and successful filesystem install subset passed **8 tests in 9.82s**. Semantic/certificate/SIGKILL/pinned-grader guards passed **7 tests in 1.50s**, followed by the three loaded-row/resealed-metadata guards passing in **5.54s**.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts CUDA_VISIBLE_DEVICES='' /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q tests/test_hsrl_v5_dataset.py -k 'not release_status'
```

Latest fast outcome: **55 passed, 1 deselected in 29.40s**, `task-4-evidence/review-fast-final-tests.log`. The excluded source-status test is a full sweep of all real source proofs, not waived acceptance; its separate covering run additionally asserts zero identity errors and that all 8,950 genuine AgentDoG admitted proofs match the completed source summary's 6,202 attack / 1,528 clean / 1,220 query counts.

Full source-covering and affected component commands/results are recorded below after completion. A pre-optimization full dataset run is retained separately in `review-dataset-tests.log`; slow metadata I/O was not converted into a pass or a source failure.

Actual 14K staging, exported environment execution, context/factor audits and final publication still require the completed matching Claw pooled source admission and root's real integration/audit work. The test-only publication filesystem transaction is not claimed as successful full release publication. This review fix does not claim source graders prove all tasks solvable or that tests provide a formal correctness proof.

Final full covering command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts CUDA_VISIBLE_DEVICES='' /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q tests/test_hsrl_v5_dataset.py tests/test_hsrl_v5_agentdog.py tests/test_hsrl_v5_claw.py tests/test_hsrl_v5_workspace.py
```

Outcome: **162 passed in 747.90s (12:27)**, exit 0. This includes the complete real AgentDoG source-status sweep: 10,446 terminal records, zero identity errors, all 8,950 source admissions accepted, exact 6,202 attack / 1,528 clean / 1,220 query counts. Retained `task-4-evidence/review-component-tests.log`, SHA256 `65685c8b9179b6040bea6a4c3e5540aeefeb00e9ff8ddb484f3de0172376dce4`.

The pre-optimization full dataset command (`-m pytest -q tests/test_hsrl_v5_dataset.py`) also finished: **50 passed in 1254.38s (20:54)**, exit 0. Retained `review-dataset-tests.log`, SHA256 `c3de80ad3da27dde3cbe80959fcae4d8676127db59792322023bc3e1c2fc585f`. Its smaller count reflects collection before subsequent additional regression tests; it is not substituted for the 162-test covering run.

After that covering run was started, the final additional clean-`never`-rule guard was exercised by the 55-test fast run and by:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts CUDA_VISIBLE_DEVICES='' /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q tests/test_hsrl_v5_dataset.py -k 'corrupt_private_outcomes or clean_private_grader'
```

Outcome: **4 passed, 52 deselected in 3.45s**, exit 0; `review-final-outcome-guards.log` SHA256 `4fea77f734394e644280614b89257917b292dadacc12cd26f1ac34d84ffe546d`. Every corruption test now first validates its genuine unmodified copy. The final fast log's SHA256 is `04054a43f9b6ad1b506c9a0148933ba8da00199b22d407c528abed4c1b809151`.

The final rule's actual input values were additionally checked across **all 1,528 admitted clean cases**, by a read-only metadata audit (no source/evaluator execution):

```bash
PYTHONPATH=scripts CUDA_VISIBLE_DEVICES='' /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
from hsrl_v5.release import GRADER_SHA256,SOURCE_PINS
from hsrl_v5.io import file_hash
base=Path('/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/rl')
source=base/'hsrl_v6_agentdog_components_prepare_20261116'
quality=base/'hsrl_v6_agentdog_real_quality_20261116_r1'
assert file_hash(source/'source_manifest.json')==SOURCE_PINS['agentdog']
summary=json.loads((quality/'summary.json').read_text())
checked=0
with (source/'accepted.jsonl').open() as stream:
 for line in stream:
  record=json.loads(line)
  if record['split']!='train' or record['condition']!='clean':continue
  case=quality/'cases'/hashlib.sha256(record['id'].encode()).hexdigest()/'quality.json'
  proof=json.loads(case.read_text())
  if not proof['ready']:continue
  grade=json.loads((Path(proof['attempt_directory'])/'first_episode/private_grade_1.json').read_text())
  assert grade['ok'] is True and grade['grader_sha256']==GRADER_SHA256
  assert grade['registered_call_goal'] is False
  checked+=1
assert checked==summary['ready_by_condition']['clean']==1528
print(json.dumps({'schema':'retained-clean-private-grade-never-rule-audit/v1','passed':True,'checked_clean_records':checked,'registered_goal_true':0,'source_execution_performed_by_audit':False,'source_manifest_sha256':SOURCE_PINS['agentdog'],'grader_sha256':GRADER_SHA256},sort_keys=True))
PY
```

Exit 0, all 1,528 clean grades match the fixed grader and `registered_call_goal: false`. Output retained as `review-clean-grade-audit.json`, SHA256 `6b5dac9b73ad9c59336ee5e9a365c6801f98da80d656d891514baa9f10dababf`. This scoped retained-outcome audit is not a 14K context/runtime readiness certificate.

`git diff --check` passed. No check failure was reclassified as a pass: expected RED and the corrected intermediate baseline mismatch are documented above; the source sweep's long I/O-bound run was allowed to finish. Root's independent review and actual 14K staging/runtime/context/publication gates remain outstanding. Any later runtime-critical workspace change must still match the final admission protocol's execution-byte closure; this fix does not relax that check.
