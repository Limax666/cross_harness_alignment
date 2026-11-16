# Task 6: private data-grader pool

Implemented against HEAD `2bd280d235c92d4b12187fe56b510073c7faa067` in `/data/home/liumingxiao/hsrl_v3_worktree_20261116`. Owned changes are `scripts/hsrl_v5/{grader_pool,workspace_worker,claw_backend,quality_claw}.py` and `tests/test_hsrl_v5_workspace.py` under `experiments/cross_harness_sft`. Included the existing root-owned library-thread pins, static boundary cache, priority-case option and 16-worker cap as assigned. Did not edit workspace/isolation/source/export implementations or root's other dirty files.

The optional backend pool reuses only the existing actual `WorkspaceWorker` private grader container. Actors still start separately for actual and reset episodes. Existing `readonly_private_grader_no_actor_code_v2` gating is repeated for each request and binds the admitted checker hash to copied bytes. Container construction remains the unchanged pinned rootless worker: both data/source projections have read-only bind mounts at the original workspace and `/grader` paths; network none, UID 65534, dropped capabilities, no-new-privileges, read-only rootfs, 1 GiB RAM and PID 64 remain intact.

The host controller serializes each pool, clears and replaces projection contents while retaining bind-root inodes, checks source/snapshot manifests before and after grading, rejects symlinks/special files/size violations, independently copies bytes, and verifies hashes. Every grade still starts a new actual Python child running unchanged `/grader/test.py` with `/root/.openclaw/workspace`, inherited nice 10 and single numeric-library thread limits. Private RPC returns the actual child PID and request nonce. The trusted worker clears its own `/tmp` before every grade. Actor tool response shape is preserved. Pools recycle after at most 64 distinct cases or 128 grade children, poison on transport/identity/contract failure, and cannot be reused after poisoning. There is no score cache.

QC has one pool per executor lane, closes the serial first-case pool before executor dispatch, verifies distinct actual/reset nonces and child PIDs, and closes every owned lane in `finally` before writing a completed summary. Uncertain removal preserves projections and errors, prevents a completed summary, and blocks replacement dispatch on resume until previous cleanup is confirmed. The existing bounded admission dispatcher still stops after infrastructure failures and drains current work.

Case-local evidence is under each `{actual_episode,reset_episode}/private_grader/`: `isolation.json`, `request.json`, `projection.json`, `response.json`, and `source_snapshot/` containing independently copied original reward-directory bytes. Existing episode `private_grader_response.json` and frozen `private_output_snapshot/` remain. `projection.json` binds container/generation, request nonce, actual child PID, original/projected manifests and digests, bind-root identities and case-local checker copy. Pool removal writes a new case-local `cleanup.json` only after exact-container removal is confirmed. Pool lifetime stderr/startup/removal records are retained under output `private_grader_pools/dispatch_*/lane_*/generation_*/worker/`; projection scratch alone is removed. QC summary names the dispatch cleanup manifest and requires confirmed cleanup. Export should copy whole case evidence; retain pool lifetime logs too if desired for shared-worker stderr.

## Verification

Used the installed `create-verification-skill` workflow with the repository's existing documented Python/pytest and full-QC commands. No new skill was needed. All local runs set `CUDA_VISIBLE_DEVICES=''`; unit tests replace the Docker transport boundary only, preserving real controller/projection/gate/lifecycle behavior. No Docker container, source checker on the host, model/API call, GPU work, smoke experiment, or limited runtime admission job was launched. Existing Claw 2008682, AgentDoG 1908852 and read-only auditor 2055729 were untouched.

Interpreter: `/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python`.

From worktree root, initial RED command (3 expected missing-feature failures):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=experiments/cross_harness_sft/scripts CUDA_VISIBLE_DEVICES='' /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q experiments/cross_harness_sft/tests/test_hsrl_v5_workspace.py -k 'private_projection or grader_identity or scratch_clear'
```

Retained log: `.superpowers/sdd/2026-11-16-hsrl-v5-ready-dataset/task-6-red.log`. Resume guard also had an expected RED, then GREEN (`-k quality_resume_refuses`; `task-6-resume-red.log`). These failures were corrected, not counted as passes.

Initial focused suite: 91 passed in 50.70s (`task-6-focused.log`). New-controller focused run: 11 passed in 2.56s (`task-6-pool.log`), followed by two passing cross-case/all-lane checks and the passing resume guard. Coverage includes 64-case recycle, distinct projection inodes from originals, stable bind roots, stale-file removal across cases, immutable prior-case checker evidence, nonce/PID contract rejection, poisoning, post-grade projection tampering, unsafe-checker admission rejection before startup, byte/size/symlink guards, scratch-symlink clearing, partial/uncertain cleanup policy, and actual two-child OS PID observation using harmless Python commands. Genuine risk-metrics checker bytes were copied and checked against the frozen original hash and AST gate without executing them.

From `experiments/cross_harness_sft`, broader command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts:src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q tests/test_hsrl_v5_workspace.py tests/test_hsrl_v5_claw.py tests/test_hsrl_v5_dataset.py tests/test_hsrl_v5_agentdog.py tests/test_hsrl_v6_agentdog_groups.py
```

Result: 126 passed in 178.65s; retained `task-6-v5-v6.log`. This ran before the final five additional controller regressions. Final affected-suite command and result are recorded below after completion. The entire historical repository test suite was not run: it includes unrelated model/runtime paths outside this task's authorized local verification scope.

## Runtime handoff and remaining limits

Root owns review, graceful old-job drain and launch of the new frozen full-production QC at `hsrl_v6_claw_real_quality_20261116_r2`. Actual first-case risk-metrics recovery, real kernel mount enforcement in this pooled mode, observed fresh source-checker child identities, all 7262 eligible source cases, measured throughput and final actual pool cleanup remain unverified by this implementation task. Do not claim real-runtime completion or publish the final 14K source dataset until that full run finishes with confirmed cleanup. Tests establish scoped controller behavior, not a formal proof.

Final affected suite, from worktree root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=experiments/cross_harness_sft/scripts CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q experiments/cross_harness_sft/tests/test_hsrl_v5_workspace.py experiments/cross_harness_sft/tests/test_hsrl_v5_claw.py
```

Result: 99 passed in 50.22s (`task-6-final.log`), including all final 14 new regressions. `git diff --check` passed. A final post-review-style check of resume/nonce guards also passed as recorded in `task-6-last-guards.log`. No runtime or throughput claim is inferred from these tests.
