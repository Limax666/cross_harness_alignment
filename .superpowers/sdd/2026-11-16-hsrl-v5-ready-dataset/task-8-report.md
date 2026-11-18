# Task8 implementation report

Implemented the approved read-only reconnaissance recommendations in owned files only: new `hsrl_v5/runtime.py` collector and `train_hsrl_v5.py` driver; `hsrl_v3/components.py` framing/context fix; `hsrl_v3/policy.py` masked objective and durable recovery; focused tests and `docs/hsrl_v5_training.md`.

Shared final-auditor contract is `prepare_interface(tokenizer,public,row,workspace)->PreparedInterface(view,mapping,context)`. `hsrl_read_context` takes `{}` and is present before T mapping in every C variant. It reads collector-owned memory only; C1/C2 update the same store. No v5 source/backend/workspace/dataset/release/quality file was edited. The previous initial audit does not bind this changed interface.

The new driver requires READY14K, the exact official4B path and explicit GPU0 lease/registered baseline ledger. It uses real Dataset episodes, complete128-rollout steps/16×G8, successive updated-policy K2, explicit malicious-request PG mask, globally computed PG/KL action denominators, the frozen registered-call reward and protocol cost, AdamW with the approved constants, two-update warmup, clipped gradients, raw-policy logprob convention and unchanged sampling. It saves exact IDs and old/ref arrays, adapter versions/hashes, immutable artifact identities, per-step metrics and partial-batch journals. In-flight uncertain attempts and invalid episodes stop rather than repeat or become reward0. Adapter metadata version guards are labeled honestly, separate from actual artifact content hashes.

TDD evidence: first focused run reproduced6 failures (actual parser marker bug plus missing interface/controller); the following focused run passed12. Added nonhashable tool-name/nonfinite numeric regression then reproduced1 failure; fixed framing input validation and that focused run passed13 tests in0.68s. Final focused regression added existing frozen reward/mask/sampler/candidate guards:19 passed in1.32s. Published official checkpoint JSON metadata was checked without importing tokenizer/model/Torch. Python compilation of the four owned modules passed. Commands are in the training documentation. These are controller/parser/interface/metadata checks, not an experiment.

Implementation commits:55d6649 (collector/driver/policy/components/tests/docs),354579c (actual initialized LoRA targets/count/dtypes, tokenizer template and EOS bound to run manifest). A repeat test invocation from repository root collected no tests because its relative path was wrong; the corrected experiment-root invocation passed13 in0.39s. The final19-check run passed in0.95s; retained stdout is `.superpowers/sdd/2026-11-16-hsrl-v5-ready-dataset/task-8-focused.log` and survives cleanup.

No GPU/model/API/source-payload/Docker/actual rollout/optimizer run was performed by this agent. Root owns genuine final-data audit, publication, exclusive GPU0 handoff and real budgeted training. Real kernel/forward/backward fit, optimizer/adapter mutation, dynamic context read, successor-policy sampling and checkpoint restore remain unverified until that run.

Remaining material implementation constraint: retaining every r16 full optimizer recovery point can exceed the registered60GiB run cap before300 steps; the driver stops rather than deleting evidence. Root was notified before handoff. Full-budget completion requires an explicit frozen current-run recovery retention policy or disk budget change. Baseline repair/judge work must be preserved on GPU1 before the resident GPU0 service is retired.

## Review fix round1: T8-I1 / T8-I2

Read the independent review and `superpowers/subagent-driven-development/references/fix-loop.md` before changing the code. Both Important findings apply to the implemented guards; no plan exception was claimed.

T8-I1: retained canonical checkpoint `.cache/huggingface/download/*.metadata` provides revision851bf6e and11 official load-relevant object IDs. The two weight LFS SHA256s are `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61` and `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188`. New code-owned `configs/hsrl_official4b_851bf6e.json` freezes that independent source identity and provenance hashes. Initial model acceptance requires the exact trusted resolved canonical path, exact load-relevant file map and byte agreement with each pinned LFS SHA256/Git blob SHA1. The manifest records the approved identity/hash separately from verified actual file SHA256s. Same-basename locations, changed small official config bytes and changed weight-byte fixtures are rejected. Source revision is taken from verified approved identity instead of inferred from a name.

T8-I2: checkpoint save now fsyncs populated adapter/nested directories bottom-up, staging root and destination parent before installation; after rename it fsyncs the destination parent and its parent before the shared commit controller can write the authoritative journal. New-run output parent is also fsynced. Restore and filesystem-only tests share `validate_checkpoint_files`. Focused serialization-boundary tests preserve/select a real previous private checkpoint when either the adapter-directory barrier or post-install parent barrier fails. An installed but unreferenced checkpoint is retained and never selected automatically. No historical checkpoint was deleted.

TDD: the first amended-guard run had4 expected failures for missing identity/commit APIs. Adding shared loader validation coverage then had2 expected failures for the absent extracted validator. After the fixes, scoped amended guards passed4 checks, and the full owned focused file passed12 in2.02s. The final retained run/compilation evidence follows below. These tests use small private serialization-boundary byte fixtures, no actual model or optimizer; they do not claim real Torch serialization, CUDA execution or host power-loss validation. All11 source metadata provenance files and the actual small official config were checked. Full9GB model-byte verification remains an actual-launch guard and was not run here. Root's PID3834678 admission work was untouched.

Root's future retention authorization (preserve all adapters/metrics/evidence, full resume only last2 plus registered development/terminal) was recorded but deferred from this narrow review fix. The existing60GiB stop guard remains; no pruning was implemented or historical evidence removed.

Covering command, from `WT/experiments/cross_harness_sft`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q tests/test_hsrl_v5_training.py
/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m py_compile scripts/train_hsrl_v5.py scripts/hsrl_v3/policy.py
```

`MAIN=/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft`. Retained output: `.superpowers/sdd/2026-11-16-hsrl-v5-ready-dataset/task-8-fix1-focused.log`. Runtime model loading, real optimizer change, successor-policy rollout, dynamic context evidence and real checkpoint restore remain root-owned and unverified here.

Final output: `12 passed in 5.75s`, pytest exit0; both amended modules compiled with exit0 and no output. No broader suite or source-I/O admission run was launched for this fix.
