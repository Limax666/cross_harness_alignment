# Fresh-run checkpoint recovery retention — 2026-11-17

Scope: external trainer `experiments/cross_harness_sft/scripts/train_hsrl_v5.py` (previous trainer commit `fac7d83`), new external `scripts/hsrl_checkpoint_retention_v1.py`, owned filesystem tests and this evidence. No `hsrl_v3/` or `hsrl_v5/` package edits, baseline/FlashRSI changes, dataset/certificate alterations, model/API/source benchmark execution, GPU/Docker work, training lease, real training or historical checkpoint cleanup. Existing independent publication work is outside this commit.

## Acceptance and implementation

The unchanged serializer stores optimizer, RNG, scheduler and checkpoint state in `resume.pt`; its `COMPLETE.json` validates the exact full inventory and SHA256s. Retention is therefore external. A fresh run initializes `checkpoint_retention.json` only before any checkpoint or journal exists, bound to its finalized run-manifest hash and exact policy. Resume retains the existing code/config/lease/model/data identity checks; `_code_identity()` additionally hashes the helper, and CONFIG contains the retention policy. No migration or adoption of historical runs is provided.

`commit_checkpoint()` saves and durably installs a checkpoint, then persists the authoritative journal and readable mirror, before calling retention. Retention independently loads the authoritative journal on CPU and requires its pointer to equal the current checkpoint. Only subsequently registered checkpoint receipts are eligible; uncommitted/staging directories are never scanned for candidates. All current files/hashes must validate. Initial, current, last two fully valid complete committed points, and explicit persistent best/dev/terminal anchors remain full recovery points. A normal terminal budget completion explicitly anchors its checkpoint. The policy does not change GRPO, model, dataset, reward, safety seed 267, G8/K2, attempt/update budgets or any resource gates; the run cap remains 60 GiB.

For an eligible old checkpoint, all adapter and other original files remain. Before removing the sole recovery blob `resume.pt`, a trusted hash-validated CPU decode preserves every non-optimizer/non-RNG field in `ARCHIVED_STATE.json`, including scheduler, state/statistics, version, adapter_hash and frozen_signature. Unexpected payload fields or non-JSON metadata fail before deletion; no generic tensor dropping occurs. The strict `ARCHIVED.json` records recovery_unavailable, policy, original COMPLETE SHA, retained metadata SHA, removed file SHA/size and action. COMPLETE is renamed to `RECOVERY_MANIFEST.json` and fsynced before unlink. The complete original inventory is rechecked after metadata extraction. Blob unlink, marker updates and owning directory entries are fsynced. Parent directories are also synced.

Any archival intent/preserved recovery manifest is conservatively left untouched on subsequent calls, including interrupted archival. An unfinished action remains `prune_pending`, never falsely reported complete. This deliberately retains extra recovery bytes when uncertain. Incomplete, unrecognized, changed, symlinked, outside and uncommitted points cannot be deleted. Repeated successful calls are byte-idempotent.

## Verification workflow and evidence

Applied installed `create-verification-skill` and existing `.agents/skills/verify-hsrl-dataset/SKILL.md`; reused its interpreter, isolation and CPU test workflow rather than generating another skill. New executable behavior is exercised through real `Policy.save` / trainer `commit_checkpoint` / durable torch journal / retention with genuine torch CPU serialization and owned tiny adapter-file fixtures. These are filesystem/serialization tests, not synthetic experiments or evidence of model quality. No model is constructed and no optimizer update runs.

From `/data/home/liumingxiao/hsrl_v3_worktree_20261116/experiments/cross_harness_sft`:

```bash
PYTHONPATH=scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q tests/test_hsrl_checkpoint_retention_v1.py tests/test_hsrl_v5_training.py tests/test_hsrl_episode_durability.py
```

Result: **29 passed in 2.01s**, exit 0. Exact output survives at `checkpoint-retention-20261117-evidence/focused.log` beside this report. Initial red run before helper implementation produced five expected missing-helper failures; subsequent focused runs passed as behavior was added.

Coverage: initial/current/last-two/explicit anchors and all adapters; metadata retention; unknown payload fail-closed; unowned/outside, uncommitted, incomplete, symlink and drift preservation; idempotency; six interruption barriers (intent fsync, rename, renamed-directory fsync, unlink, unlink-directory fsync, final receipt); authoritative-journal mismatch; real trainer commit ordering; genuine CPU checkpoint/journal roundtrip; current resume blob drift rejection; failed journal commit never entering retention; original adapter drift during metadata extraction preventing unlink.

Broader documented CPU workflow: **INCOMPLETE / deliberately aborted**, exit 2 (`KeyboardInterrupt`), after 146 tests passed in 251.45s. This is not a passing complete check. It was stopped only in its owned known PID 2680937 after verifying `/proc/<pid>/cmdline`, while read-only hashing legacy AgentDojo inventory/private assets, to avoid competing with the active genuine R2 data audit. No existing source, workload or data-audit process was signaled. The completed focused verification above is proportionate to this external trainer/helper change. The partial output survives at `checkpoint-retention-20261117-evidence/verification.log`.

Exact broader command (same cwd):

```bash
PYTHONPATH=scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q tests/test_hsrl_checkpoint_retention_v1.py tests/test_hsrl_v5_training.py tests/test_hsrl_episode_durability.py tests/test_hsrl_v5_workspace.py tests/test_hsrl_v5_claw.py tests/test_hsrl_v5_agentdog.py tests/test_hsrl_v6_agentdog_groups.py tests/test_hsrl_v5_dataset.py tests/test_hsrl_v6_initial_context.py tests/test_hsrl_v4_components.py
```

`git diff --check`: exit 0. `git diff -- scripts/hsrl_v3 scripts/hsrl_v5`: empty; frozen source package code remains unchanged.

Helper SHA256: `8f32e35e27b3ea08811776bc753ded0a26da1f4de08637cf3f85b3841120870f`.
Trainer SHA256: `8b764a910b7652ba45a0620c5e4b624e52785eb9ed019f51df6d16949a6909d2`.

## Remaining unverified behavior

No production/GPU training, actual model/LoRA resume, full CUDA RNG restoration, development selection or 300-update/38400-attempt run was executed. The unchanged Policy.load path still requires exact COMPLETE inventory/file hashes and adapter identity before continuation; the filesystem tests exercise those guards and CPU state bytes, not actual GPU model restoration. Retention reduces recovery-state growth but cannot prove that all retained adapters, episode evidence and explicit anchors fit within 60 GiB; the existing gate continues to stop the run at its registered cap. Tests support scoped empirical claims, not a formal proof.
