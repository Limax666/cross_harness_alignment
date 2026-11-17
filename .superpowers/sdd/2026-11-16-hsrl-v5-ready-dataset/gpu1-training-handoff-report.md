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
