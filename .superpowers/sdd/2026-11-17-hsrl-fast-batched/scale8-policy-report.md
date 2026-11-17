# P3 scale8 policy

New owned implementation: `experiments/cross_harness_sft/scripts/hsrl_batched_policy_v2.py`; focused tests: `experiments/cross_harness_sft/tests/test_hsrl_batched_policy_v2.py`. Live v1 policy/source/runtime/packages/checkpoints/trainer were not edited.

`hsrl_batched_policy_v2.BatchedPolicy` subclasses v1. Initialization, fast-kernel requirements, all original helpers, RawLogprobCapture, score/reference, generate(single), backward/loss, RNG, LoRA and checkpoint methods are inherited. Only generate_batch is copied: AST comparison confirms exactly cap2->8 and its validation message changed; all actual HF sampler/logprob/EOS/left-padding/reference/telemetry/error-cleanup semantics stay identical. Accepts one through eight equal-budget requests; ninth fails before model generation. Broker remains responsible for the32768 padded-token budget; policy never truncates/drops inputs.

Runtime identity calls inherited fast_runtime_identity (actual bindings and complete prior package/source provenance retained), then records schema `hsrl-fast-runtime/v2`, policy `hsrl-fast-batched-policy-v2`, inherited_policy `hsrl-fast-batched-policy-v1`, max_batch_size8, policy_class `hsrl_batched_policy_v2.BatchedPolicy`, and current v2 policy source path/SHA256. Class VERSION is explicit. Driver must additionally bind both v1 helper source and new v2 source; P4 was notified. policy.batch_events interface remains identical to v1, including failed-call evidence and pre-warp logprob contract.

## Scoped empirical verification

Reused existing verification workflow/interpreter/isolation; no installation, GPU/source/API/checkpoint/RL calls. Actual tiny randomly initialized CPU HF GPT2.generate batches4 and8 each invoke model generation once, use unequal left padding, include early EOS and different per-row output lengths, and compare every captured raw old logprob to independent unpadded teacher-forced raw scoring within1e-6. Check one pending B*V maximum, unchanged sampler/compile-disabled/output-logits flags, exact EOS-visible text and token counts, cap9 rejection, and distinct v2 identity preserving inherited provenance. Existing v1/Qwen3.5+PEFT/trainer fixtures also pass.

Exact final command from `experiments/cross_harness_sft`:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=scripts /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q tests/test_hsrl_batched_policy_v2.py tests/test_hsrl_batched_policy_v1.py tests/test_hsrl_v5_training.py > ../../.superpowers/sdd/2026-11-17-hsrl-fast-batched/scale8-policy-final-tests.log 2>&1
```

Exit0, exact summary `38 passed in 8.40s`, no warnings. Focused initial green:4passed7.98s. Test-first red retained in `scale8-policy-red.log`:1failed because missing v2 module (exit1). An initial construction command used a repository-relative path from experiment cwd and failed before writing source; its subsequent expected missing-module test failures are retained in `scale8-policy-first-green.log` (1failed3errors). Corrected path, without relaxing tests. Logs remain in the plan directory; no cleanup deletes evidence. Read-only AST comparison printed `generate_batch AST: only cap 2->8 and validation message changed`. Git staged whitespace check passed.

Hashes:

- v2 policy: a6ceffc0640820101db08065f635476a67be29cab42aa678bb9cf0640ec4ae28
- v2 tests: 3753e279591829e7c58aba44df96dec2bad4672285a61e03fc57e053cf2c482d
- unchanged v1 policy: 7bfe4948f5c4c417d87bbd00f7908e6b61526818eaebce45c04bf1ec4843559b

Unverified: actual eight-row4B CUDA/kernel execution, GPU/RAM fit under32768 tokens, BF16 numerical tolerances, real environment throughput and full128-episode/optimizer-update outcome. Root owns independent review and the genuine formal new run. No GPU utilization improvement is claimed from these CPU tests.
