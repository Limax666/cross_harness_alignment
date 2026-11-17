# Authorized GPU1 trainer handoff report

Implementation is limited to `experiments/cross_harness_sft/scripts/train_hsrl_v5.py`, focused CPU tests `experiments/cross_harness_sft/tests/test_hsrl_v5_gpu_handoff.py`, and this report. No frozen package/source modules, model, source dataset, running service, GPU workload or baseline process was changed or launched.

## Acceptance and behavior

`--physical-gpu {0,1}` defaults to physical GPU0. The actual `CUDA_VISIBLE_DEVICES` must exactly match the selected index (no list, alias or leading zero). The initial compute-process gate queries only `nvidia-smi --id=<selected>`, and refuses any reported compute PID. Existing memory/disk/growth floors and all algorithm, reward, optimizer, runtime and retention configuration remain unchanged. The run manifest's identity now explicitly records `physical_gpu`, as well as its existing full resource handoff. Existing GPU0 lease schema/status fields remain accepted. Old pre-change run manifests have a different code identity and cannot resume, consistent with existing strict resume semantics.

GPU1 uses a separate schema; the historical `gpu1_baselines_preserved` flag is forbidden because it could imply continued running. Both old service and controller PID lists are checked for absence. Nonpositive or noninteger PIDs are refused. The old service list must be nonempty for GPU1. The lease and ledger are validated before and again under the exclusive lease lock before initial loading. The ledger bytes must match the recorded hash; the caller owns creating the real immutable task snapshot and ensuring the PID lists enumerate all old owned processes.

## Exact GPU1 lease schema for root

The following is a schema recipe, not an actual lease or evidence. Root must instantiate real values from its independently verified handoff:

```json
{
  "schema": "hsrl-gpu1-training-handoff/v1",
  "physical_gpu": 1,
  "exclusive": true,
  "forbidden_gpus": [2, 3],
  "boundary_status": "stages_ended_preserved_unresolved_work",
  "baseline_status": "paused_with_preserved_results",
  "old_service_pids": [123456],
  "old_controller_pids": [123457],
  "baseline_ledger_path": "/absolute/path/to/immutable-real-baseline-task-ledger.json",
  "baseline_ledger_sha256": "actual SHA256 of those ledger bytes",
  "lock_path": "/absolute/path/to/exclusive-gpu1-training.lock"
}
```

Replace example PIDs with actual former owned service/controllers; each PID must be absent at validation. Controllers may be `[]` only if none existed. Services must be nonempty. Do not include `gpu1_baselines_preserved`. Launch selection must pair `CUDA_VISIBLE_DEVICES=1` with `--physical-gpu 1` and this actual lease. This report does not authorize a launch; root's independent review and formal R1 gate remain required.

## Verification and surviving evidence

Reused `.agents/skills/verify-hsrl-dataset/SKILL.md`'s installed interpreter, CPU isolation and retained-evidence policy, adapting its automated command to trainer/retention tests. No new verification skill was needed. The resource subprocess boundary is replaced in focused tests; no real `nvidia-smi` command or GPU/model/API execution occurred.

Exact command from the isolated checkout:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q experiments/cross_harness_sft/tests/test_hsrl_v5_gpu_handoff.py experiments/cross_harness_sft/tests/test_hsrl_v5_training.py experiments/cross_harness_sft/tests/test_hsrl_checkpoint_retention_v1.py
```

Observed exit code 0; terminal output:

```text
.........................................................                [100%]
57 passed in 2.28s
```

Coverage includes real CLI parsing through the environment guard for default GPU0 and explicit GPU1, parser refusal of GPUs2/3, unsafe schema/status/PIDs/ledger/lock rejection, changed ledger bytes, live current-process IDs, selected-device query argument, any unexpected compute PID, legacy training accounting/checkpoint behavior and checkpoint retention. Filesystem fixtures are isolated pytest temporary paths; no real lease/model/output path is opened by CLI failure tests. `git diff --check` on owned code/tests passed. Self-review confirmed no settings changes and no frozen-module edits in this commit.

Trainer SHA256 for independent approval:
`2ae1f5bc18ce1c3cf91f290fa4f2974e086731cc479bf9582c5b2a2e867bf9af`.

Evidence is embedded in this committed report and survives test fixture cleanup; raw test transcript also remains at `/tmp/hsrl_gpu_handoff_tests.log`. No persistent verification process was started. Real CUDA mapping, actual selected-device availability, immutable real task-ledger provenance, process handoff completeness, model loading, optimizer updates and formal R1 remain unverified here and belong to root's authorized run.

## Round1 Important correction: downstream constructor

The independent review identified a real gap in the first implementation: its tests did not reach the frozen Policy constructor, which still required physical GPU0. Do not launch the earlier trainer hash above. Following root's explicit scope extension, added `scripts/hsrl_selected_gpu_policy_v1.py`; frozen `hsrl_v3/policy.py` and published package/certificates remain unchanged. No installed skill named `fix-loop` was found in local skill catalogs; the installed receiving-code-review workflow was applied to verify and correct the finding, then rerun verification.

`SelectedGPU1PolicyV1` subclasses the frozen Policy, explicitly copies its initializer, changes only the initial environment guard from physical0 to physical1, and inherits every other method. It never changes `os.environ`. The logical CUDA device remains 0 under the real exact physical1 visibility mask. Runtime class selection validates the original source SHA256 pin and compares the complete initializer AST (signature, decorators and body), omitting only its first guard; the selected first guard is separately compared with the exact expected AST. Any original source byte drift, remainder drift or selected guard drift refuses GPU1 selection before loading. GPU0 still returns the original Policy class. The trainer calls this selector for its actual constructor and records selected class/version/source SHA256 plus the wrapper file in its code identity.

New approved-source pin, unchanged frozen policy bytes:
`a73973caf58668ce14f20f3e34d5487a381bbaad3b6529df525acb3bfa913b47`.

New trainer SHA256 requiring independent approval:
`7a0fdb391c4bc92f79339c4bdc21fbce37e7a076bbdb6b96c298b84afb1ea2b8`.

Wrapper version: `hsrl-selected-physical-gpu1-policy-v1`.
Wrapper SHA256: `a5971ecb8480d1803ee693961e08d38ad8ea3f9d1e96944f13bc2813de71920e`.

Reran the exact three-file pytest command documented above, saving stdout/stderr to `/tmp/hsrl_gpu_handoff_fix_tests.log`. Observed exit code 0:

```text
...............................................................          [100%]
63 passed in 5.38s
```

New coverage invokes the actual chosen original/subclass constructors with matching environments and intercepts the very first `torch` import by raising a sentinel exception. This proves both constructors cross their real guard without importing torch, issuing CUDA calls or loading any model. Opposite device environments fail at the actual constructor guard. Tests also prove original-source byte drift, copied-initializer rank drift and GPU guard drift fail validation, and verify inheritance and wrapper manifest/code identity. This is focused boundary verification, not a substitute for actual formal R1 GPU/model execution. `git diff --check` passed. The GPU1 lease schema remains exactly as documented above.

The first staged diff check flagged an extra blank line at wrapper EOF. Removed it and reran the complete three-file suite (63 passed above); final diff whitespace check passes. No behavior changed in this formatting correction.
