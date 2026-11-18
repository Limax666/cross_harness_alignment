# Harness Scaling RL v5 ready mixed dataset

**Goal:** Prepare the actual mixed general+safety dataset, filter unsupported or invalid records, and deliver source assets, frozen family splits, online environment/verifier adapters and algorithm-neutral GRPO/CHS-PO views ready for real online RL. A candidate index alone is not completion.

**Spec:** docs/harness_scaling_rl_training_pool_v5_20261116.md; active user goal takes precedence. Target approximately14K training conditions, equal general/safety; target7000Claw and7000AgentDoG. If genuine filtering removes cases, retain reasons and refill only from the same train partition after the same checks, never by replication/test borrowing. Do not shrink the objective to old2K or30instances.

**Architecture:** New versioned scripts/hsrl_v5 package in existing isolated worktree. Main checkout source files/data/env are read-only assets; outputs go to new independent preparation/release directories under main experiment data/rl and outputs/rl. No mutation of active FlashRSI imports/services. Task-facing public interface and host-only environment/verifier assets are separate. New package may reuse reviewed pure v3 helpers, but not old mixture defaults or unverified actor code.

## Global Constraints

- Workspace /data/home/liumingxiao/hsrl_v3_worktree_20261116, branch codex/hsrl-v3-20261116. Main assets /data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft. Interpreter mainexp/.venv-rl-pilot/bin/python.
- Source pins: ClawGym HF bd9043cff88b913922a165d9c022a4796aee3f3a, JSONL SHA0455f13c4f7a4839d39c3c1da9704c4959046ed529040d44f46f8f9bd07f7774,15402records; AgentDoG HF7c263216d05afcd2a31185f6d102b35504418b3c, gzip SHAd6d51b20a6a7d6597305af13e1a775233b94fd4c6da7a4ad79b3ad1632e7ee0c,15774catalogrecords. Existing7000safetycandidate index69c289ffabf39c8ea0c88754743b15658f84dfbb0faf0090f81843e112dd75ea.
- Source catalog/fixtures/graders/risk rules remain host-side. Policy sees only public instructions/schema and whitelisted business tool results. No target-readable oracle, reward_info, hidden initial/full state, host secret or source directory. Fake fixture secrets are data, not hostenv.
- Real source environment states and graders must reset, execute and score; no model mocks/simulated experimental evidence. Full data preparation/quality validation using genuine source assets is required. Do not invoke source payloads on host or disable sandbox to pass.
- Do not touch any GPU/model server/FlashRSI queue/cron. Rootless Docker is ours at /run/user/1026/docker.sock, image sha256:a5babb29ecb8d6cc68066e19adfae7ac5569a44162e0e97162edd90f46bf676c. No host mounts beyond declared workspace/private grader in isolated respective processes. Preserve old sources/evidence and failed preparation records.
- Split whole families before condition/harness expansion, seed20261116. Bench200 stays excluded; only public prompt/input hashes may be computed for contamination checks, never expose evaluation answers to target. AgentDoG paired2931families80/10/10 is frozen; malicious-query originalgoal must not be positive lawfulUtility. Safety call-goal is not physicaleffectASR.
- Judge errors/missing source-oracle evidence are invalid, not0/safe. Any deterministic-only adapter must be declared as a versioned profile, not silently substitute broadjudge. Preserve applicability masks and components for GRPO/CHS-PO.
- Do not claim runtime readiness from AST/PID/manifest alone; verify real dataset CLI/backend paths and retained side effects. Syntax/unit checks are supplementary. No automatic spending or duplicatejudge calls in preparation; judge capability can use explicitly configured existing approved service only when necessary to verify actual training-facing path.

## Interfaces

Source preparation writes inventory.jsonl (allrecords/reasons), accepted.jsonl (structurally admitted only), source_manifest.json (pin/hash/counts/status), private assets per task and split/provenance evidence. Each accepted record at least source,id,family,split,condition,source_path,source_sha256,source_tree_sha256,source_identity,source_contract,capability_contract. Not marked runtime_ready until actual validation.

Unified data runtime (Task2/3) opens records into fresh episode state; start/public tools, canonical call, finish components, close/reset. Record exact applicability/validity and scalar/cost vectors separately; source originalmetrics preserved. Dataset export uses stable task keys and host-only references; public Parquet/JSONL prompt never includes private grader/state/oracle. GRPO and CHS-PO consume identical task identities/components and differ only objective in the trainer.

### Task 1: full ClawGym preparation and evidence-backed admission

Owner files: new scripts/hsrl_v5/{__init__,claw_source}.py and tests/test_hsrl_v5_claw.py, optionally a focused claw_contracts.py. Do not edit v3/source modules or active code. Root owns common runtime/export.

- Parse all15402 fixedrawJSONL records, verify bytes/pin, safe unique relative paths and original checker contents. Preserve source task IDs/line identities/metadata instead of pretending officialselected2k is full corpus. Extract only source assets into a new independent full-source directory; private grader separate from public input assets.
- Inspect full-source metadata origin/template/primaryskill info and normalizedprompt/source asset duplicates. Derive reproducible grouped90/10 train/dev, no within-family crosssplit; record known near-duplicate scope without claiming complete semantic independence. Exclude exact/normalizedpublicprompt collisions with officialBench200; only publicinputs used for that guard.
- Reuse v3 capability/write/reference contracts where sound, recording version. Unknown network/host/dependency/checker contracts are pending/unsupported with evidence, not model utility failures. Do not blanket-disqualify legitimate in-place edits merely because an input changed: audited write scopes and immutable gold references must be distinct. Preserve unresolved cases, do not invent missing fixtures or change goals/checkers.
- Include allrecords in inventory and compute accepted counts; target>=7000train structurally admissible, but do not fabricate if insufficient. No source executors/model/judge/GPU runs in this subtask. Data CLI actually processes allrawrecords, retention/newdir behavior protects evidence, consumed bytes bound back to source. Actual runtime grade admission handled by root later.
- Write meaningful real-source TDD regression checks for source drift/missing files, split grouping/identity, public/private asset separation, capability reasons; run focused tests and applicable source suite. Run full package data preparation on real fixedrawassets, retain command/log/stats/fingerprint in outputs/rl/hsrl_v5_claw_prepare_20261116. Commit only owned code/tests, write report.

### Task 2: AgentDoG source preparation and isolated runtime

Owner root or next implementer: scripts/hsrl_v5/{agentdog_source,agentdog_worker,agentdog_backend}.py; tests/test_hsrl_v5_agentdog.py.

- Consume fixedcatalog+pairing+candidate index; validate allIDs/families/rules/source hashes and preserve poolreserve. Materialize private per-task state/tool implementations/rule specs; target only public task/schema. Check schema/dispatcher/callreachability/dependency/rule consistency, preserve allfilterreasons.
- Implement real isolated CPU state tools and source rule evaluation from the pinned published contracts (dataembeddedimpls preserved); use independent clean process/container, fresh initialstate each episode, bounded wall/memory/output lifecycle. No unsafehostexec/thread sandbox claims.
- Enforce JSONschema/canonical mapping, publicresult whitelist, output-size bounds; full sourceepisode and grade never reach policy. Preserve real state and history across calls, reset between episodes. Unknown/failing tools/infrastructure distinct from invalidmodelcalls; unsafe-call target vs actualeffect separate.
- Freeze legalUtility/malicious mask and source risk applicability; judge profile independent, failclosed/nofallback. Actual dataset quality path executes genuine sourceplans/tool actions/graders where sourceprovides positive/reference state and proves reset/privateoracle/score invariants, no fabricated model transcripts.

### Task 3: full runtime quality and balanced release selection

Owner root: scripts/hsrl_v5/{quality,claw_backend,release}.py, tests/test_hsrl_v5_quality.py.

- Verify genuine source paths: all releasecandidate taskreset and sourcechecker/runtime with retainedpercase evidence. ParallelCPU boundedworkers/uniquecontainer IDs; failures retained, safe resumable no repeatcompleted work. Modelperformance failures are not sourcequality exclusions; inability to validate infrastructure stays pending.
- Preserve private graders and immutableinput gold references. Use unchanged sourceClaw check.py only in independentread-only output snapshotgrader. Dependencies/images pinned; sourcegrader crashes invalid, no score0coercion. Source utility/risk mappings checked on applicable positive/negative genuine controls; do not claim everytask solvable fromonecontrol.
- Final selection equal sourcecounts near7K/7K only from owntrain partitions; reserve sourcefamily holdouts untouched. Safety roles5500attack/1000semanticmatchedclean/500malicious target; replacement followsfrozenstrata/familyroundrobin and logs exclusions, never copy IDs or use test. If postfilter genuinecounts reduce targets, report shortfall and repair/expand train ratherthan covertly redefining success.

### Task 4: online-RL exports, integration and delivery

Owner root: scripts/hsrl_v5/{dataset,reward,export}.py, configs/hsrl_v5_dataset.yaml, .agents/skills/verify-hsrl-dataset/SKILL.md and featuremap, docs/hsrl_v5_dataset_usage.md.

- Export selfcontained versioned assets+JSONL/Parquet release, source/version/manifest hashes, perrecord lineage/split/quality/compatibility and masks. Onealgorithm-neutral environment/reward API consumed by both GRPO/CHS-PO; task_id references cannot accidentally render hiddenoracle into prompt.
- Deterministic balanced sampler8general/8safety groups, G8/K2 schedule slots with same task nextconfiguration independentreset; source/vectormetric fields identical across algorithmviews. Implement and verify actual dataset load, publicrender, environment steps, scoring and reset through exported artifact paths. Do not implement or launch a new actor optimizer as a substitute for dataset readiness.
- Package provenance/QCfailure/accepted/coverage/candidate-vs-ready report and exact nexttrainercommand/config. Exercise reusable project verification workflow; generatedskill remains draft until launch/doctor/drive/evidence/cleanup exercised. Evidence survives cleanup; no duplicate modelservice/smokeexperiment for verification.
- Fresh independent code/spec/privacy review; fixImportantfindings; relevanttest commands+realproduction datasetCLI checks pass. Preserve unrelateddirtymain files anddon'tpush/mergewithoutprior scope.

## Completion audit

Not complete until delivered current-state sourceassets+validrelease manifests, true50/50train balance at requiredscale or explicitly resolved sourcequality shortfall, family/benchmarkdecontamination, actual environment+grader reset/sideeffect+privacy proof, frozenvalidity/masks, workingalgorithmneutral GRPO/CHS-PO export/loader/runtime and reproducible usercommands. Runtime/judge unverified cases cannot be marked ready. Scope excludes running RL optimizer, newbenchmarkmeasurement, replacing currentmodelservers, cloudpublishing orcron.
