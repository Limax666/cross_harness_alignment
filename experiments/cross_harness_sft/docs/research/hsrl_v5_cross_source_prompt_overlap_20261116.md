# HSRL v5 cross-source public prompt overlap audit

Audit UTC: 2026-11-16T11:43:15.807638+00:00

Read-only comparison of the pinned Claw inventory, all paired AgentDoG records, and ClawBench200 cached public prompt hashes. No source or grader execution. Only these report files were written.

## Method

Normalize AgentDoG published scenario `user_query` with NFKC, casefold, collapse whitespace to single spaces, then SHA-256; compare against Claw normalized prompt hashes. All assigned dev/test records are references, including unqualified holdouts. AgentDoG query_target uses its inventory scenario `clean`. Unpaired records are outside frozen lineage and have no split/family. Private evaluation subtrees were skipped lexically, not decoded. AgentDoG paired private bundle file SHA-256 values and pinned inventory/cache hashes were verified.

## Counts

- source_records: 31176
- compared_records: 28614
- claw_inventory_records: 15402
- agentdog_inventory_records: 15774
- agentdog_paired_compared_records: 13212
- agentdog_unpaired_not_compared_records: 2562
- all_train_records: 24483
- all_dev_test_records: 4131
- unqualified_dev_test_records: 767
- unique_prompt_hashes: 18730
- clawbench200_public_prompt_hashes: 200
- train_vs_holdout_or_benchmark_collision_hashes: 135
- train_vs_holdout_collision_hashes: 131
- train_vs_benchmark_collision_hashes: 4
- train_train_duplicate_hashes: 472
- train_train_cross_source_duplicate_hashes: 0
- holdout_duplicate_hashes: 165
- extraction_failures: 0

## Train versus every source holdout

| Train source | Holdout source | Hashes | Train records | Holdout records |
|---|---|---:|---:|---:|
| agentdog | agentdog | 131 | 6172 | 1771 |
| agentdog | clawgym | 0 | 0 | 0 |
| clawgym | agentdog | 0 | 0 | 0 |
| clawgym | clawgym | 0 | 0 | 0 |

## Whole-family removal for holdout and benchmark contamination

Affected whole train families: 1105; affected train records: 7218; affected qualified train records: 7204. Remaining structurally qualified train capacity: 10724.

| Source | Condition | Affected whole-family records | Affected qualified records | Remaining qualified capacity |
|---|---|---:|---:|---:|
| agentdog | attack | 4940 | 4940 | 1808 |
| agentdog | clean | 1101 | 1098 | 928 |
| agentdog | query_target | 1172 | 1165 | 351 |
| clawgym | benign | 5 | 1 | 7637 |

## Conservative whole-family removal for every exact collision including train/train

Affected whole train families: 1584; affected train records: 9830; affected qualified train records: 9809. Remaining structurally qualified train capacity: 8119.

| Source | Condition | Affected whole-family records | Affected qualified records | Remaining qualified capacity |
|---|---|---:|---:|---:|
| agentdog | attack | 6748 | 6748 | 0 |
| agentdog | clean | 1580 | 1571 | 455 |
| agentdog | query_target | 1497 | 1489 | 27 |
| clawgym | benign | 5 | 1 | 7637 |

## Release assessment

Under the conservative prompt-only whole-family exclusion rule, the frozen approximately 14K / 7K safety + 7K general recipe cannot be met. This conditional capacity is not proof of exact underlying stateful task duplication. All AgentDoG conditions, including clean controls, count toward safety: 3,087 safety records remain (1,808 attack + 928 clean + 351 query_target), plus 7,637 Claw general records. The stricter union including every train/train exact duplicate group leaves 482 safety (455 clean + 27 query_target) and 7,637 general. Attack/query_target-only versus clean role counts are distinct from this frozen recipe. No resplit, target reduction, or runtime readiness claim is authorized by this audit.

All four direct Claw train matches to ClawBench200 normalized public prompts are already unqualified: the source direct-prompt guard correctly excludes them. The family of `auto_update_cd4d7c438758` (`clawgym:full_source_family:35c7ebf6dc3a23022c6b0cb5`) contains a different qualified sibling. Whole-family exclusion removes that additional sibling, leaving 7,637 structurally qualified general train records. This is a whole-family sibling gap, not an admitted direct prompt match; final release requires exclusion of the entire affected family.

## Verification

All five recorded input fingerprints rechecked unchanged, including four validated against the supplied external pin or source-manifest hashes; all 13,212 paired AgentDoG private bundle file SHA-256 checks passed; zero query extraction failures. Whole-family exclusion sets and remaining capacity were independently reconciled against inventory records. Initial qualified train capacity is 17,928; each reported exclusion plus remaining capacity sums to that baseline. The normalizer matches `scripts/hsrl_v5/claw_source.py:44-45`.

## Collision evidence

Full hash groups, source IDs, families, split, condition, qualification flags, per-collision whole-family impact counts by source/condition, and family-removal lists are in the companion JSON. No query text is included.

## Limits and release risks

- Normalized exact public prompt equality only; semantic and near-duplicate independence is not established.
- Capacity uses existing qualified flags as structural capacity, not runtime admission.
- Claw 292+32-task privacy gate is a later filter; this audit does not claim old qualifications runtime ready.
- Whole train families excluded in reported counterfactual capacities; no datasets, splits, targets, services, or jobs modified.
- No source execution, model/API/Docker/GPU calls, benchmark graders, answers, or evaluation of private graders.
- HarnessRisk public prompt decontamination not performed: no obvious public prompt file discovered in source data filename scan.

## Input fingerprints

- `/data/home/liumingxiao/hsrl_v3_worktree_20261116/experiments/cross_harness_sft/outputs/rl/hsrl_v5_claw_prepare_shell_contract_fix1_20261116/source_manifest.json`: SHA-256 `1cc77e86b9cd144864180b972498abdd0d1a3ab6de9ec364717934839717e4ac`
- `/data/home/liumingxiao/hsrl_v3_worktree_20261116/experiments/cross_harness_sft/outputs/rl/hsrl_v5_claw_prepare_shell_contract_fix1_20261116/inventory.jsonl`: SHA-256 `e5a59b6f0c3c947aa528c59a8108aea8c14cfbe2554b646ed71f9ec51f70f53b`
- `/data/home/liumingxiao/hsrl_v3_worktree_20261116/experiments/cross_harness_sft/outputs/rl/hsrl_v5_claw_prepare_shell_contract_fix1_20261116/benchmark_public_hashes.json`: SHA-256 `795d1074e145cea11cf3a6352c3402a1c50d9f616c23b5c313a0e6998c4dcce2`
- `/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/rl/hsrl_v5_agentdog_prepare_20261116/source_manifest.json`: SHA-256 `7fead9f6a7d0e543fd1d3ba182599b0d0010e4ebe3fef2d6e08e1f4cd492611f`
- `/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/outputs/rl/hsrl_v5_agentdog_prepare_20261116/inventory.jsonl`: SHA-256 `0f31d4ff5eb83fd11694efefbafcaf2cdba6ed72805b6036f55ff665d63cfcd1`

## Source task identity diagnosis

Source-defined state, allowed public interfaces, implementation hashes, success specification and legal/risk rule hashes were compared without actor execution. These are own researcher-created pre-model internal folds; no model/test outcome informed this diagnosis. Exact equality does not establish behavioral equivalence.

Of 131 same-query train/holdout hash groups, 0 contain an exact full selected task match, and 0 contain an exact underlying base clean task match. Full selected task cross-fold duplicate signature groups: 0; matching train records: 0.

| Signature dimension | Equal train/holdout record pairs | Different pairs | Train records with any holdout match |
|---|---:|---:|---:|
| public_interface_exact | 221579 | 85882 | 4419 |
| public_interface_normalized_query | 223102 | 84359 | 4443 |
| tools_schema | 223102 | 84359 | 4443 |
| base_clean_initial_state | 0 | 307461 | 0 |
| selected_state_and_overlays | 0 | 307461 | 0 |
| tool_impl_sources | 0 | 307461 | 0 |
| tool_dispatch | 223102 | 84359 | 4443 |
| legal_success_spec_and_eval_rule | 212204 | 95257 | 4558 |
| risk_eval_rule | 1940 | 305521 | 586 |
| full_selected_task | 0 | 307461 | 0 |
| underlying_base_clean_task | 0 | 307461 | 0 |

## Component topology and fixed-seed split hypothesis

Union-find uses all 13,212 paired records, every existing base-family edge, and every equal normalized public-query edge.

Existing families: 2931; unique normalized selected public queries: 3328; components: 1456; largest component: 723 records / 122 families.

| Largest component | Records | Base families | Conditions |
|---|---:|---:|---|
| `f4df3549bd6be02b` | 723 | 65 | {"attack": 511, "clean": 65, "query_target": 147} |
| `42e792fa7fe48ec6` | 712 | 31 | {"attack": 508, "clean": 31, "query_target": 173} |
| `cbac35c17faa5812` | 604 | 38 | {"attack": 435, "clean": 38, "query_target": 131} |
| `72a3eade0d36e744` | 478 | 96 | {"attack": 382, "clean": 96} |
| `bcecd92af7ae05e6` | 465 | 122 | {"attack": 212, "clean": 122, "query_target": 131} |
| `e57f28f53663fd65` | 433 | 71 | {"attack": 243, "clean": 71, "query_target": 119} |
| `f1dc1f1a07f9c241` | 420 | 69 | {"attack": 276, "clean": 69, "query_target": 75} |
| `eeb0f9c0f7ce5c2c` | 380 | 72 | {"attack": 308, "clean": 72} |
| `aded67b1397cb7fa` | 370 | 14 | {"attack": 280, "clean": 14, "query_target": 76} |
| `3d0eb61b07202001` | 365 | 38 | {"attack": 260, "clean": 38, "query_target": 67} |

Hypothesis only, not implemented: fixed seed 20261116; sort component IDs by SHA256(`20261116|component_id`), first floor(80% of components) train, next until floor(90%) dev, rest test. Component ID hashes the canonical sorted base-family list. Exactly one seed trial; no optimization. Components per fold: 1164/146/146.

| Hypothetical fold | Condition | All records | Qualified records |
|---|---|---:|---:|
| dev | attack | 1011 | 1011 |
| dev | clean | 282 | 244 |
| dev | query_target | 240 | 238 |
| test | attack | 544 | 544 |
| test | clean | 262 | 217 |
| test | query_target | 106 | 106 |
| train | attack | 6843 | 6843 |
| train | clean | 2387 | 2073 |
| train | query_target | 1537 | 1530 |

Hypothetical AgentDoG safety train capacity: 10446; unique full selected task hashes among qualified train records: 10446. With original Claw benchmark-family exclusions, general capacity remains 7,637 and total hypothetical structural train capacity is 18083.

A documented new protocol/version and root scientific ruling are required before any split change. The component hypothesis does not authorize borrowing official holdouts, choosing a favorable seed, dropping hard cases, or modifying the fixed target. Full per-record private hashes, component membership, per-query comparisons and exact duplicate groups are in the companion JSON.

## Diagnostic interpretation and reconciliation

Same normalized wording is a prompt-overlap signal, not proof of an exact repeated stateful task. Under the explicit fingerprints, all 131 crossfold same-query groups differ in full selected task and underlying clean task; all compared crossfold pairs differ in clean state and implementation mapping. Hash inequality does not prove behavioral or semantic independence. The current prompt-family exclusion capacity and hypothetical component resplit capacity answer different policy questions; neither changes the frozen protocol.

Independent graph traversal over family vertices reproduced all 1,456 union-find components. Every paired record appears exactly once; every existing family and normalized query stays within one hypothetical fold. Hypothetical condition counts were recomputed from member IDs, all 13,212 full selected task hashes are unique, and all five original input fingerprints remain unchanged.
