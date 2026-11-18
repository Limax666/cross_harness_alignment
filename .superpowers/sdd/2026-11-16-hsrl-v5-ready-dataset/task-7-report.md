# Task 7: acknowledged workspace creation and safe admission recovery

Owned files: `experiments/cross_harness_sft/scripts/hsrl_v5/{workspace,quality_claw}.py`, `tests/test_hsrl_v5_workspace.py`, and this report. No edit to `isolation.py`, AgentDoG runtime, pinned source/graders, actor worker, pool implementation, or running services. No Docker, source checker/program, model/API/GPU work, smoke run, or production admission was executed by this task.

## Cause and acceptance

Read the ready-dataset plan, Task 6 report, independent review O-I1, and inherited lifecycle. The evidence-backed cause is asynchronous `docker run` creation outliving an early absence check. Inherited cleanup checked absence before killing/waiting its client, so a confirmed-absent marker could precede late creation of that exact UUID. Reduced dispatch alone cannot establish terminal cleanup.

Acceptance: settle creation before an absence claim; unknown creation blocks replacement/resume; preserve exact ownership, unchanged resource/mount/network/user boundaries, original checker bytes and fresh actual/reset actors; retain command/timing evidence; retry only a bounded proven transport failure after confirmed actor cleanup, never source or identity failure; leave real admission to the root's independent full run.

## Implementation

WorkspaceWorker creates the same pinned container with the same flags and read-only mounts using synchronous `docker create`, requiring a successful 64-hex container-ID acknowledgement before issuing `docker start --attach --interactive`. The create timeout remains the existing Docker-control bound of 20 seconds; worker readiness remains 30 seconds. Create command/output/error/duration, attach command/start time, lifecycle creation state, selected daemon, readiness duration, and cleanup commands/output/duration are retained locally. Original isolation evidence fields and private-grader child identities are preserved.

WorkspaceWorker-specific cleanup first kills/waits the attach client, then removes and inspects the exact owned UUID. Start cannot recreate a removed container. A timed-out, failed, interrupted or malformed create acknowledgement remains unknown; even apparent absence cannot produce `removal_confirmed:true`. Unknown/failed cleanup writes a false marker, preserves projections/evidence and raises CleanupUncertain. Partial startup remains visible to the existing pool retirement path because `process` is initialized before creation.

Optional host configuration `HSRL_WORKSPACE_DOCKER_SOCKET` defaults to the existing owned socket. Its absolute resolved path must remain under `/run/user/<uid>`, its inode must be an owned socket, and the same rootless engine/pinned image checks apply. This changes WorkspaceWorker only. Host environment is still explicitly whitelisted; the override and other host secrets do not enter the actor. Root owns the separate daemon and its lifecycle, leaving the shared daemon untouched.

QC protocol v6 binds lifecycle/recovery/socket and all current code hashes, preventing old-protocol mixing. Resume refuses any previous actor lifecycle without an identity-matched confirmed cleanup, in addition to existing pool guards. One retry maximum is allowed only for exact `IsolationError:source_worker_deadline` or `IsolationError:source_worker_closed:*`, after all started actor cleanup is confirmed and the private pool is healthy. Each attempt uses its original independent directory and retains `quality.json`, traceback and started proof; recovery history references both attempts. No score cache or source/error coercion was added. The unchanged pool still permits at most 64 cases/128 fresh grader children and two fresh actors per successful case.

## Verification

Used installed create-verification-skill, systematic-debugging, test-driven-development and verification-before-completion instructions and the existing `.agents/skills/verify-hsrl-dataset` workflow/feature map. Existing pytest verification is adequate; no new skill was generated. All checks set CUDA_VISIBLE_DEVICES empty. Transport-specific tests replace only Docker/client boundaries; no container/source execution occurred.

From `experiments/cross_harness_sft`, interpreter `/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q tests/test_hsrl_v5_workspace.py tests/test_hsrl_v5_claw.py
```

Final result: **111 passed in 16.22s**, retained `task-7-final.log` beside this report. `git diff --check` passed. Focused lifecycle/socket/recovery selection: **12 passed in 1.19s**, `task-7-lifecycle.log`.

TDD evidence: `task-7-red.log` retains four expected missing-feature failures; `task-7-recovery-red.log` retains six expected missing-recovery failures. First affected run `task-7-affected.log` retains **1 failed, 109 passed**: the historical static cache correctly rejected changed workspace identity. The regression now explicitly requires that rejection and creates a small fresh cache by actually applying the current AST gate to genuine source bytes; it verifies source-record drift and actual checker hash checking without mutating original bytes or replaying a full 7638-record AST sweep. The production stale-cache policy remains strict; a new full runtime run must perform fresh static admission.

New regressions cover acknowledged create-before-attach, command and duration evidence, unknown creation refusing absent-cleanup confirmation, attach settlement before terminal inspection, resume blockade, socket confinement and environment whitelisting, successful bounded recovery with distinct attempt directories, and no retries for uncertain cleanup, poisoned pool, source failure or identity mismatch.

## Remaining real-path verification

Real kernel/container create/start/attach/removal behavior, selected daemon isolation and pinned-image enforcement, measurable throughput, genuine source checker/reset observations, entire eligible source inventory and terminal all-pool cleanup remain for the root's independent full production QC. Unit results do not prove those runtime outcomes or dataset readiness. Root must independently review and launch the new full run; old r1/r2 QA is not mixed into it. Existing complete safety QA remains unchanged. Failure logs and this report survive scratch cleanup; no running workload was touched.
