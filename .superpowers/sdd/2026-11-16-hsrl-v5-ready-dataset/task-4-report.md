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
