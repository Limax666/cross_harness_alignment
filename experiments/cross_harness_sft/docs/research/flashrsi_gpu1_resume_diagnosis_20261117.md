# GPU1 FlashRSI × HarnessRisk resume diagnosis, 2026-11-17

Read-only diagnosis; no model/API/GPU/source-program/Docker execution or shared code edit. Only this report is written. Root owns actual launch. GPU0 remains reserved for RL; the original shared Docker engine must not be restarted or changed.

## Confirmed failure boundary

`flashrsi_harnessrisk_algorithm_reference_v7.py` reads the normal algorithm source and replaces its initial-directory expression with the explicit real `HR_INITIAL_REFERENCE`. It preserves target construction and `target._env_runner=benchmark.env_runner`; its transform guard matches the inspected source. No concrete reference-path bug was established.

`HarnessRiskBenchmark._run_once` invokes the configured real host with five argv entries, correct cwd and inherited environment, and captures stdout/stderr into pipes. On failure it returns an error Trajectory containing the last 2000 stderr characters **only in memory**. Neither stdout, stderr nor host return code is persisted. Strict `grade` then raises on all invalid cases. `Algorithm._evaluate` records trajectories only AFTER grade returns, so this exception also prevents the error trajectories from being saved. This is the concrete shared diagnostic defect: the only useful underlying host errors are discarded. The final generic `unscorable trajectories` traceback cannot identify the failed host operation.

`Agent.rollout` executes environment cases serially (`core/agent.py`, lines 239–246); these failures are not caused by launching 30 concurrent dev hosts in that loop.

Saved OpenCode remaining-run queue times: EvoPrompt 2026-11-17 01:26:19–01:26:25 (+08), ACE 01:26:25–01:26:43. EvoPrompt has 30 evidence folders and zero case directories. ACE has 30 evidence folders and 29 empty cases directories; `evidence/00000024/cases/setup_009` contains started mock-adapter files and `adaptive_context_adjustments.json`, but no privacy audit, model evidence, harness result or rule evaluation. Both trajectories directories are empty. Thus claims that no host ever ran are too strong: one reached genuine mock startup, but none reached a retained score.

Hermes remaining OPRO has 272/272 case evidence directories with harness/rule files; EvoPrompt has 248 evidence folders, 232 nonempty with rule/harness files. Subsequent ACE and SkillOpt have 30/30 empty cases directories; EvoHarness has four/four empty. The same hosts previously completed real episodes, followed by rapid failures around the same time as OpenCode. This supports a shared environment/startup transition rather than an intrinsic ACE/SkillOpt logic issue. Exact host exception remains **unrecoverable from these retained artifacts**; a namespace/OS/model/path cause must not be invented. Current namespace count limits are 1,029,881 each, and current filesystem free space is 7.0 GiB on `/` and 453 GiB on `/data`; neither current reading proves the historical state.

## Best actual initial controls

Searched only immediate `outputs/flashrsi_harnessrisk_*` children and their named `<harness>_initial_full` directories; no recursive graph traversal. `completed.json` and `attempts.jsonl`, not `result.json`, define initial progress. File presence alone overstates validity because failed episodes can still have rule/harness files.

| Harness | Actual best initial directory below outputs/ | Attempts | completed_scored | Judge accepted |
|---|---|---:|---:|---:|
| OpenCode v11 | `flashrsi_harnessrisk_4b_private_opencode_opro_suffix_v5_20261116/opencode_initial_full` | 128 | 128 | 128/128 |
| Claude Code v8 | `flashrsi_harnessrisk_4b_private_claude_cli_projection_v8_gpu1_20261115/claude_code_initial_full` | 128 | 50 | 0 |
| Codex v6 | `flashrsi_harnessrisk_4b_private_codex_v6_20261114/codex_initial_full` | 128 | 17 | 0 |
| OpenClaw | `flashrsi_harnessrisk_4b_private_openclaw_gepa_environment_v2_20261114/openclaw_initial_full` | 128 | 13 | 0 |
| Pi | `flashrsi_harnessrisk_4b_private_pi_gepa_interpreter_v4_20261114/pi_initial_full` | 128 | 18 | 0 |

Judge acceptance is read from `cases/llm_judge_multi_harness/manifest.json`, requiring status `judged` or `existing`; OpenCode v11 has all 128. Claude/Codex/OpenClaw/Pi best controls have no such manifest. OpenCode GPU1 v4 control also records 128 scored cases, but its host is v10, so it must not substitute for the v11 control. Older 124-scored controls and snapshot-only reference copies are not full genuine v11 controls.

## Small versioned next change and verification proposal

Preserve all original/live/archived imports. Add a versioned benchmark wrapper and reference entry rather than patching existing files. Persist a case-local host command, selected host/protocol/snapshot identity, timestamps, return code, stdout, stderr, and trajectory status/error BEFORE strict grading. Do not persist the whole environment or credentials. On an unscorable evolution episode, stop after that first actual failed case with its evidence path, rather than serially producing 30 indistinguishable errors. Keep strict scoring, privacy, target budgets, host confinement and same-protocol real-reference checks unchanged.

Controller regression tests should fake only the external host boundary, returning a concrete stderr/exit code and verifying retained bytes after ranking fails; also verify successful episodes preserve score/identity behavior, timeout evidence survives, no credentials enter records, and no reference result files are copied. Root can then start the authorized full real OpenCode missing-algorithm job with this instrumented entry; its first genuine failed host yields the exact next repair target without a smoke/simulated experiment. A successful full job proceeds normally. No broad queue should blindly retry an unchanged failure fingerprint.

Durable GPU1 sequencing should bind port 18764, use OpenCode v11 actual initial reference, skip only independently validated complete algorithm+judge cells, then repair/rebuild the other four incomplete initial controls before their six algorithms. Every stage must retain its concrete attempt and acceptance outcome; a process exit alone is not completion. Underlying host startup repair cannot yet be specified honestly until the discarded exception is captured in that real path.

## Implemented handoff: v17 subprocess evidence

At root's subsequent instruction, added only new `scripts/flashrsi_harnessrisk_benchmark_v17.py`, `scripts/flashrsi_harnessrisk_algorithm_reference_v17.py` and `tests/test_flashrsi_harnessrisk_benchmark_v17.py`. Original imports/files remain unchanged. The new benchmark subclasses the original and preserves its grade, splits, source metrics, host argv/cwd, 2460-second timeout and success/identity rules. Stdout/stderr stream directly to private case-attempt files before completion, so timeout or strict ranking failure cannot erase their already-written bytes. `host_execution.json` binds command/cwd/case/snapshot/PID/return code/status/timing; only harness and model port are recorded from the environment. No credentials or complete environment are serialized. The new entry transforms exactly one original benchmark import inside v7's guarded real-reference transform; all original algorithms/configurations/budgets remain unchanged. The full actual algorithm still evaluates the same full case set; no diagnostic-only case cap or score-zero fallback was introduced.

Pure transport-boundary checks ran from MAIN experiment directory:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts:src:vendor/FlashRSI CUDA_VISIBLE_DEVICES='' /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m pytest -q tests/test_flashrsi_harnessrisk_benchmark_v17.py
```

Result: **3 passed in 1.62s**. These cover concrete failing stderr surviving a subsequent strict-grade exception, genuine-shaped successful snapshot/official metric preservation, safe environment fields, and timeout bytes/exit evidence. No real subprocess/source/model/GPU/Docker/API was executed by these tests. A repository-wide `git diff --check` encountered preexisting unrelated CRLF/trailing-whitespace files; the scoped new-file check is the applicable check for this change. Real host behavior and underlying historical host-error cause remain unverified until root launches the authorized genuine full algorithm using v17.
