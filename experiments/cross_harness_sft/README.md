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

## Two-stage collection and rejection sampling

The Codex + AgentDojo path separates raw trajectory collection from SFT filtering. Collection does
not drop a completed episode because it contains an invalid tool attempt, fails utility/safety, or
would otherwise be rejected for training. Native events, normalized messages, independently audited
MCP calls, official verifier results, and provenance are written before any rejection decision.

### Stage 1: collect all raw teacher-Agent trajectories

The current teacher Agent is `gpt-5.6-sol + Codex CLI`. In the full configuration,
`max_cases_per_benchmark.agentdojo: 0` selects every case exposed by the pinned AgentDojo worker
across its four official suites:

```bash
cd /root/cross_harness_aligement
bash experiments/cross_harness_sft/scripts/run_codex_native_agentdojo_full.sh
```

The launcher starts the worker when necessary, runs preflight, streams Codex JSON events, and calls
only `collect_native`. It does not invoke `verify.py` or `build_rft.py`. Completed rows are appended
to:

```text
experiments/cross_harness_sft/outputs/codex_agentdojo_native_gpt56_sol_full/raw_native.jsonl
```

Runtime failures are retained separately in `raw_native.errors.jsonl` in the same directory. Resume
is enabled by default: after an interruption, run the same command to skip completed episode IDs and
retry unsuccessful episodes. Do not use `--no-resume` for a continuing full run because it
intentionally re-executes completed cells and appends duplicate IDs.

Each raw row preserves both the unmodified Codex process stream in `native_events`, `native_stdout`,
and `native_stderr`, and a training-facing `messages` view reconstructed in native event order. The
messages retain assistant text, real MCP call IDs, audited arguments and observations, and the final
`<think>...</think>` target. Metadata records exact model/harness/benchmark identities, token usage,
the per-episode `tool_history`, official verifier output, and rewards. `status=completed` proves that
collection finished; it does not mean the row is eligible for SFT.

### Stage 2: filter strictly, then build SFT data

After Stage 1 finishes, run `verify.py`. It delegates training eligibility to
`core/skillrl/build_skill_use_sft.py::rejection_reason` and adds native-harness provenance checks:

```bash
EXP=experiments/cross_harness_sft
OUT="$EXP/outputs/codex_agentdojo_native_gpt56_sol_full"

"$EXP/.venv/bin/python" -m cross_harness_sft.verify \
  --input "$OUT/raw_native.jsonl" \
  --accepted "$OUT/accepted.jsonl" \
  --rejected "$OUT/rejected.jsonl" \
  --report "$OUT/filter_report.json" \
  --max-turns 20 \
  --max-response-length 16384
```

This rejects incomplete/parser/invalid-call/max-turn trajectories, safety failures, successful
attacks, utility failures, missing audited tool history, harmful-query tool attempts, and missing
provenance. Accepted and rejected rows and aggregate rejection reasons are all retained.

Build the family-disjoint SFT dataset only from `accepted.jsonl`:

```bash
"$EXP/.venv/bin/python" -m cross_harness_sft.build_rft \
  --input "$OUT/accepted.jsonl" \
  --output-dir "$EXP/data/sft/codex_agentdojo_native_gpt56_sol_full" \
  --validation-fraction 0.1 \
  --max-turns 20 \
  --max-response-length 16384
```

`build_rft.py` repeats the rejection gate defensively, deduplicates messages, applies any configured
semantic-family cap, and writes `train.jsonl`, `validation.jsonl`, and `build_report.json` with zero
semantic-family overlap.

### Combined multi-harness pipeline

The existing combined multi-harness pipeline remains available:

```bash
bash experiments/cross_harness_sft/scripts/start_benchmark_workers.sh
curl --fail http://127.0.0.1:8111/health
curl --fail http://127.0.0.1:8112/health
bash experiments/cross_harness_sft/scripts/run_native_rft.sh
```

Unlike the staged Codex command, `run_native_rft.sh` runs collection, verification, and SFT export
sequentially. Use the two-stage procedure to retain the complete raw AgentDojo teacher corpus before
making filtering decisions.

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
