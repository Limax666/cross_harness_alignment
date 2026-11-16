# Private mixed online RL release

The release is **not ready** until both new source admission runs finish, 14,000 real train conditions meet their role quotas, and the selected exported runtime and context audits pass. A running job, a health endpoint, or a successful subset cannot authorize publication. The CLI reads CPU proof files; it does not call models, source tools, graders, Docker or external services.

Run from `experiments/cross_harness_sft` with the installed `.venv-rl-pilot/bin/python` interpreter and `PYTHONPATH=scripts`. Set `CLAW_QC` to the final new pooled Claw quality directory (`hsrl_v6_claw_real_quality_20261116_r2`); final export requires the v5 admission protocol, the v2 checker boundary and confirmed pool cleanup. The source pins are fixed inside `hsrl_v5.release`; changing a path never authorizes a new source version.

```bash
PYTHONPATH=scripts /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m hsrl_v5.release \
  --claw-source outputs/rl/hsrl_v5_claw_prepare_shell_contract_fix1_20261116 \
  --claw-quality "$CLAW_QC" \
  --agentdog-source /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/rl/hsrl_v6_agentdog_components_prepare_20261116 \
  --agentdog-quality /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/rl/hsrl_v6_agentdog_real_quality_20261116_r1 \
  --status
```

`--status` writes nothing and prints a structured report with source completion, unprocessed cases, rejected/pending cases, identity errors, eligible quotas, shortfall and the candidate selection digest. `--wait 3600` optionally polls only these files, with bounded duration and no scheduler. Incomplete or engine-blocked runs remain pending; they are never treated as fully sampled quality failures.

To prepare a private candidate, use the same four source/quality arguments and replace `--status` with:

```bash
--stage --out /absolute/new/private-candidate \
--grader /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/vendor/agentdog_runtime_v5_c8d803f/runtime_evaluators.py
```

A failed build returns exit 2 and retains `private-candidate.shortfall.json`. A candidate has `manifest.json` with status `candidate` and has no `READY.json`. Failed staging directories are retained for inspection. Existing output directories are never overwritten. The exporter copies whole original Claw task trees unchanged, AgentDoG bundles, successful case evidence, pool lifetime logs/confirmed cleanup, private original/relocated records, source metadata, frozen consumer code and the pinned private grader with provenance. Admission verifies initial negative and role-specific positive/risk private grades against actual results and reset/public/state history. Clean AgentDoG reset controls are verified from their recorded replay, without inventing a reset result or grade. Claw raw checker responses are parsed and matched to both result vectors and reset scores; pooled requests also bind source/snapshot bytes, fresh child nonces/PIDs and exact-container removal confirmation. Original absolute source references are historical, private provenance; active source paths resolve inside the release. AgentDoG data attribution is retained; the vendored grader's code license is unknown, so this package is for private local research and must not be publicly published.

The selection uses seed `20261116`, stable IDs and base-family round robin, preferring declared candidates and refilling from the same new train partitions. Each selected clean AgentDoG control has a selected attack in its original base family. `family` remains the source base family; `split_group` remains the immutable connected-component partition. Whole Claw families matching cached public benchmark hashes are excluded. Public request hashes shared with any active dev/test holdout are excluded across both sources. No benchmark grader or answer is read.

For candidate audits, use the frozen code:

```python
from hsrl_v5.dataset import Dataset
view = Dataset.for_verification('/absolute/new/private-candidate')
public = view.public(('agentdog', task_id))  # exactly messages and tools
with view.episode(('agentdog', task_id), '/absolute/new/evidence') as env:
    observation = env.call(tool_name, arguments)
    result = env.finish(final_text)
# view.grpo_reward(...) and view.chspo_components(...) always raise.
```

Use `PYTHONPATH=/absolute/new/private-candidate/code` in a fresh process. The view verifies frozen code, source lineage and admission guards, and uses the real sandboxed environment path. The audit owner supplies two certificates against the unchanged candidate. Each certificate requires `passed: true`, `records: 14000`, `selection_sha256`, `source_manifest_sha256` (both pinned hashes), and both `artifact_file_sha256_sha256` and `semantic_core_sha256` copied from the candidate manifest. The file-map digest excludes the two certificates and manifest/READY files to avoid a hash cycle. The independent semantic digest additionally binds evaluator path and fixed grader pin, the full required runtime module set (including `grader_pool.py`), source and recomputed selection identities, reward contract, limits, sampler/consumer configuration and the admission report. The loader independently verifies runtime coverage and grader bytes; changing only manifest metadata cannot preserve certificate validity. The context certificate additionally requires `limits` matching the config and `initial_configurations: 18`. The runtime certificate requires true checks named `exported_loader`, `public_render`, `environment_steps`, `scoring`, `fresh_reset`, `private_oracle_separation`, and `sampler_full_875_blocks`. Certificates attest observed evidence; metadata alone does not establish those checks.

Publish only after those real audits:

```bash
PYTHONPATH=/absolute/new/private-candidate/code /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -m hsrl_v5.release \
  --publish /absolute/new/private-candidate --out /absolute/new/private-release \
  --context-audit /absolute/evidence/context.json --runtime-audit /absolute/evidence/runtime.json
```

Publication copies the immutable candidate into a separate private directory under the final output parent, then rechecks all core files and public rendering, covers all 14,000 unique task conditions across 875 blocks, validates certificates and verifies the normal frozen consumer constructor there. Linux `renameat2(RENAME_NOREPLACE)` atomically installs the validated copy without replacing an existing output. `READY.json` binds the final manifest hash. The original candidate is never given ready metadata, certificates or READY, and remains retryable through write failures, consumer-validation failures and hard process interruption. Failed private publication directories are retained for inspection. An interrupted pre-install publish cannot establish a final output; publication staging and the final destination share a filesystem automatically.

A tool-interaction collector consumes the final release as follows. This is the next collector integration entry point; this repository does not provide a stock executable actor optimizer for this contract.

```bash
PYTHONPATH=/absolute/new/private-release/code /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot/bin/python -c \
  'from hsrl_v5.dataset import Dataset; d=Dataset("/absolute/new/private-release"); print(d.sampler.step(0))'
```

```python
dataset = Dataset('/absolute/new/private-release')
for step_index in range(1750):
    step = dataset.sampler.step(step_index)
    # Configure the collector with step['configuration']; sample current policy.
    for group in step['groups']:
        key = (group['source'], group['id'])
        actor_input = dataset.public(key)
        # For EACH of G=8 rollouts, create a new episode/evidence directory.
        # Drive real env.call(...) from policy tool calls, then env.finish(...).
        # Exclude invalid/incomplete episodes and retain their failure evidence.
        # result = the real completed episode result
        # grpo = dataset.grpo_reward(result, objective='safety')
        # vector = dataset.chspo_components(result)
```

Each pair of successive steps uses the same 16 task groups, different OA18 configurations, a fresh policy sample and fresh environment state. Each step has eight general and eight safety groups. G8/K2 multiplies rollout attempts, never dataset cardinality. GRPO and CHS-PO share the environment result fields and masks. `grpo_reward(result, 'utility')` returns `None` for malicious query targets; collectors must set their utility policy-gradient mask false. Applicable missing/error/nonfinite values and invalid episodes raise instead of becoming zero.

`train.parquet` contains public prompts/tools and opaque `(source,id)` ground truth. `extra_info` is collector-only metadata, including lookup/role/masks; never render it into actor prompts. The full private bundle, hidden rules, state, plan and source quality controls remain outside the public interface.

Claw utility is the official code-verifier component only. AgentDoG source cost measures registered tool-call goals. These profiles do not establish actual harm, broad judge safety or persistence; those components are null with false masks. Admission does not prove every Claw task solvable. Neither tests nor private graders are a formal correctness proof. Final-answer-only reward adapters cannot replace real tool interaction, and compatibility with old v3 actor trainers is not claimed.
