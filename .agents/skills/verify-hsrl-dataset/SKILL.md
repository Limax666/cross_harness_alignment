---
name: verify-hsrl-dataset
description: Verify the private Harness Scaling mixed online RL dataset CLI, admission, release loader, sampler and real isolated environments. Use for hsrl_v5 source/runtime/export changes.
---

# Verify the mixed online RL dataset

Read [the feature map](features/README.md) first. The primary surface is the
`hsrl_v5.release` CLI and `Dataset` Python API. There is no web server to start.
The current source/runtime evidence is genuine CPU data admission, not model
performance or an RL optimizer update. Never replace it with synthetic runs.

## Launch

Use this existing isolated checkout and installed interpreter:

```bash
export HSRL_MAIN=/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft
export HSRL_EXP=/data/home/liumingxiao/hsrl_v3_worktree_20261116/experiments/cross_harness_sft
export HSRL_PY="$HSRL_MAIN/.venv-rl-pilot/bin/python"
export HSRL_EVIDENCE="$HSRL_MAIN/outputs/rl/verify_dataset_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -m 700 "$HSRL_EVIDENCE"
cd "$HSRL_EXP"
export PYTHONPATH=scripts
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
```

Do not create another admission run, model server or timer merely for verification.
These existing directories identify the current recipe:

- Claw source: `outputs/rl/hsrl_v5_claw_prepare_shell_contract_fix1_20261116` in this checkout.
- Claw QC: `$HSRL_MAIN/outputs/rl/hsrl_v6_claw_real_quality_20261116_r2`.
- Safety source: `$HSRL_MAIN/outputs/rl/hsrl_v6_agentdog_components_prepare_20261116`.
- Safety QC: `$HSRL_MAIN/outputs/rl/hsrl_v6_agentdog_real_quality_20261116_r1`.

## Doctor

Read `AGENT.md`, the QC `launch.json`, `protocol.json`, `run_state.json`,
`progress.json` and final `summary.json` where present. Verify the exact recorded
PID against its command before acting. A PID, partial progress, summary filename
or an existing Docker socket is not evidence that admission passed.

```bash
"$HSRL_PY" -c 'import sys; from hsrl_v5.release import SOURCE_PINS; print(sys.executable); print(SOURCE_PINS)'
```

For environment work, verify the owned rootless socket `/run/user/1026/docker.sock`
and pinned image `sha256:a5babb29ecb8d6cc68066e19adfae7ac5569a44162e0e97162edd90f46bf676c`.
Check that no earlier owned worker has uncertain cleanup. Do not use anyone
else's daemon, loosen mounts/network/limits, or stop existing GPU workloads.

## Drive

Use [admission](features/admission.md) for the read-only real CLI. Use
[release](features/release.md) only after all prerequisites pass. Use
[environments](features/environments.md) for the real exported consumer boundary.
The usage reference is `experiments/cross_harness_sft/docs/hsrl_v6_dataset_usage.md`.

Affected automated checks (additional to the real user path):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$HSRL_PY" -m pytest -q tests/test_hsrl_v5_workspace.py tests/test_hsrl_v5_claw.py \
  tests/test_hsrl_v5_agentdog.py tests/test_hsrl_v6_agentdog_groups.py \
  tests/test_hsrl_v5_dataset.py tests/test_hsrl_v6_initial_context.py tests/test_hsrl_v4_components.py
```

## Evidence

Keep exact arguments, exit code, stdout/stderr and before/after filesystem facts
under `$HSRL_EVIDENCE`. Admission side effects include real tool histories,
state/output snapshots, private raw evaluator responses, reset identity and exact
container removal evidence. A ready export additionally needs independent context
and runtime certificates bound to its selection, files and semantic core.

Status returning `not_ready` is a valid observation of the gate; it does not
verify successful publication or training readiness. Never fabricate certificates
from expected constants. Document which features remain unexercised.

## Cleanup

The admission/status journey starts no persistent service and needs no process
cleanup. Check that its evidence survives. Environment contexts close only their
own workers; inspect each exact recorded removal result. Preserve failed attempts
and do not blanket-kill Docker or Python processes. Retain publication staging
failures, candidate/proof files and all evidence. Never delete a source, grader,
checkpoint, finished evaluation or other user's files during cleanup.

## Helpers

No extra helper is required; use the existing CLI/API above. Generated workflow
validation is recorded in `verification_record.md`; unexercised features there
remain unverified. This is empirical verification, not formal correctness.
