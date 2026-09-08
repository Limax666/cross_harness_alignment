# Native Cross-Harness Rejection-Sampling SFT

This is the formal implementation of the cross-harness safety-alignment experiment. The production
path has **no mock flag, no controlled harness profile, no direct Python call to the teacher API, and
no heuristic replacement for benchmark scoring**.

## Experimental unit

```text
official AgentDojo / AgentHarm task + state + verifier
                         ^ HTTP
             per-episode stdio MCP proxy
                         ^ MCP
Codex CLI / Claude Code / OpenClaw / Hermes + teacher model
                         |
        native events + independently audited tool calls
                         v
 strict rejection filter -> multi-turn <think> SFT -> Qwen3.5-4B QLoRA
                         |
 base student vs SFT student vs teacher on identical task/harness cells
```

The benchmark worker owns task enumeration, environment state, tool execution and scoring. The MCP
proxy only transports calls and writes an append-only audit. A run is rejected if the harness binary
fails, emits no machine-readable events, bypasses audited benchmark tools, or the official verifier
does not return a complete verdict.

## Real integrations

- `backends/agentdojo.py` loads the four official suites, executes the shipped `FunctionsRuntime`,
  and calls the task's own utility/security methods.
- `backends/agentharm.py` loads AgentHarm through Inspect Evals, instantiates the shipped target tools
  and invokes the shipped grading function. Harmful partial completion is conservatively a risk.
- `native_harness.py` starts the actual `codex`, `claude`, `openclaw`, or `hermes` executable. Every
  process has an isolated config/state/work directory and the same per-episode MCP server.
- `mcp_proxy.py` performs the pre-tool check. Harmful-query calls are always blocked and recorded;
  additional forbidden benchmark tools can be listed in `pretool_guard.deny_tools`.

AgentHarm asks that harmful examples be used for evaluation rather than capability training. The rule
`harmful_query + any tool attempt -> reject` therefore admits only safe refusal/no-tool rows.

## Strict rejection and SFT format

`verify.py` delegates to `core/skillrl/build_skill_use_sft.py::rejection_reason`. It rejects incomplete
runs, parser/invalid-call/max-turn failures, safety failures, successful attacks, benign/injection
utility failures, missing tool history, and harmful-query tool attempts. Provenance is mandatory.

`build_rft.py` repeats that gate, requires `native_harness=true`, deduplicates, caps semantic families,
and splits by family with zero train/validation overlap. It preserves:

```text
system -> user -> assistant + tool_call -> tool observation -> ... -> assistant final
```

The final teacher target is `<think>visible reasoning</think> + final answer`. `train_student.py` uses
the Qwen tokenizer's native chat template and computes loss only on every assistant span; system,
user and tool-observation tokens are masked with `-100`.

## Setup

Use Linux and immutable versions:

```bash
cd /root/SafeEvolve
bash experiments/cross_harness_sft/scripts/setup_server.sh

export AGENTDOJO_REF='<commit-or-tag>'
export INSPECT_EVALS_REF='<commit-or-tag>'
bash experiments/cross_harness_sft/scripts/setup_benchmarks.sh

export CODEX_VERSION='<exact-version>'
export CLAUDE_CODE_VERSION='<exact-version>'
export OPENCLAW_VERSION='<exact-version>'
export HERMES_REF='<commit-or-tag>'
bash experiments/cross_harness_sft/scripts/install_harnesses.sh
```

Keep API credentials in the environment/native login stores. Set exact model IDs:

```bash
export CODEX_TEACHER_MODEL='...'
export CLAUDE_TEACHER_MODEL='...'
export OPENCLAW_TEACHER_MODEL='...'
export OPENCLAW_TEACHER_PROVIDER='...'
export HERMES_TEACHER_MODEL='...'
export HERMES_TEACHER_PROVIDER='...'
```

## Collect and build rejection-sampled data

```bash
bash experiments/cross_harness_sft/scripts/start_benchmark_workers.sh
curl --fail http://127.0.0.1:8111/health
curl --fail http://127.0.0.1:8112/health
bash experiments/cross_harness_sft/scripts/run_native_rft.sh
```

Set non-zero `max_cases_per_benchmark` for a pilot; zero means the full set. Raw, rejected, accepted
and filter-report files are all retained. Formal scripts never reference portable/mock sources.

## Train Qwen3.5-4B

Pin the training stack, run `scripts/setup_training.sh`, then:

```bash
bash experiments/cross_harness_sft/scripts/train_qwen35_4b.sh
```

The trainer follows the real CUDA-only pattern from `chapter8/cot-distillation/train_student.py`, uses
NF4 QLoRA, and writes a data-hashed `training_manifest.json` plus a real PEFT checkpoint.

## Same-task comparison

Run `scripts/run_eval_role.sh` three times with an otherwise identical config, changing only
`EVAL_ROLE`, `EVAL_MODEL_ID`, and the endpoint serving teacher, base Qwen3.5-4B, or the SFT checkpoint.
Claude Code needs an Anthropic-compatible facade (for example a pinned LiteLLM proxy); the other
harnesses use the OpenAI-compatible endpoint. Do not rejection-filter evaluation.

```bash
python -m cross_harness_sft.compare_models \
  --teacher outputs/eval/teacher.jsonl --base outputs/eval/base.jsonl \
  --sft outputs/eval/sft.jsonl --output outputs/eval/comparison.json
```

The join key is benchmark/task/harness/safety/seed. The report gives utility, safety, Pareto success,
paired improve/regress/tie counts, reasoning-behaviour rates, and
`(SFT - base) / (teacher - base)` capability recovery. Missing cells are reported, never imputed.

## No mock boundary

The standalone repository does not ship the former direct-API collector, controlled harness profiles,
or mock trajectory generator. Preflight fails if `harness_profiles` is present or any real executable,
model ID, or benchmark health endpoint is missing.
