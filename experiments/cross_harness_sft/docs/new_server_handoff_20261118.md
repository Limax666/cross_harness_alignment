# 新服务器 Codex 接手：4B Harness Scaling 安全 RSI

先读 [当前实验规格](harness_scaling_safe_rsi_current_20261118.md)，再按本文操作。Git 提供代码/配置/文档，不提供模型权重、READY 原始资产、私有 grader、完整运行证据或 `.venv`。源机在 v9 第 10 步采样时阻塞，v10 已准备从第 9 步 checkpoint 续训。接手前确认源机是否已恢复。2026-11-18 11:39 用户取消小时巡检，禁止恢复旧 cron/Codex 唤醒。

## 1. 克隆和代码入口

```bash
git clone https://github.com/Limax666/cross_harness_alignment.git
cd cross_harness_alignment
git log -1 --oneline
cd experiments/cross_harness_sft
```

| 用途 | 文件 |
|---|---|
| 源机续训与单条补采 | [v10 故障恢复](hsrl_online_rollout_recovery_20261118.md)，入口 `scripts/train_hsrl_v10.py`；v9 保留为父运行驱动 |
| 参数更新与 checkpoint 实现 | `train_hsrl_v5.py`、`hsrl_v3/policy.py`、`hsrl_checkpoint_retention_v1.py` |
| 8路真实交互/批量模型 | `hsrl_slot_recovery_v10.py`、`hsrl_batched_rollout_v2.py`、`hsrl_batched_policy_v2.py` 及其 v1 依赖 |
| 源执行修复/恢复契约 | `hsrl_source_repair_v1/`、`hsrl_source_recovery_v1.py` |
| 正式数据消费者/评分 | `hsrl_v5/dataset.py`、`reward.py`、`runtime.py`、`sampler.py`；训练用 patched Dataset |
| 数据发布/迁移封装 | `hsrl_v5/release.py`、`audit_hsrl_v6_candidate.py`、`publish_hsrl_v6_parallel.py` |
| 五维白盒组件 | `hsrl_v3/components.py`；不要退回固定 C0 或把产品 harness 当 OA18 |
| FlashRSI × HarnessRisk | `flashrsi_harnessrisk_*`、`flashrsi_matrix_*`、judge/host/supervisor 模块 |
| 本项目验证流程 | 根目录 `.agents/skills/verify-hsrl-dataset/SKILL.md` |

`train_hsrl_v5/v7/v8` 是保留的原版/中间版本，不应不看 manifest 就启动。旧 SkillLife/v3 小池也不是当前14K任务池。当前源码保持冻结字节，不能为了方便把一个端口/GPU/路径直接改进它们再叫 exact resume。

## 2. 安装依赖和隔离前置

1. Python3.11；先查看 `requirements-hsrl-observed-20261118.txt` 和 `configs/hsrl_environment_snapshot_20261118.json`。这是源机实际环境，不是任何显卡都能直接 pip 安装的通用 lock。当前 torch `2.11.0+cu129`、transformers `5.9.0`、peft `0.21.1`、FLA core `.5.2`、causal-conv1d `1.7.0`。
2. 新建本机独立 venv；按目标驱动/CUDA 安装相符的 torch，安装项目及所列依赖。causal-conv1d 从同版本可信源码为目标 GPU 编译；不要把旧SM架构 `.so` 当通用二进制。保留wheel/source SHA和实际导入函数身份。不要只通过 `import` 就宣称 fastpath 或完整训练可用。
3. 用当前用户独占的 rootless Docker，确认镜像和只读/网络/资源控制。AgentDoG 使用 `/run/user/<UID>/docker.sock`，Claw workspace 使用 `HSRL_WORKSPACE_DOCKER_SOCKET` 指定的 owned socket。源机后者为 `/run/user/1026/hsrl-v7/docker.sock`。不要沿用源机UID或别人socket。
4. 精确镜像为 `sha256:a5babb29ecb8d6cc68066e19adfae7ac5569a44162e0e97162edd90f46bf676c`。可从源机 `docker --host=unix://<owned-socket> image save <image> -o <private-image.tar>`，私有传输后 `docker image load -i ...` 并核对实际Id。Dockerfile重建不保证得到相同Id；改变镜像要重新记录身份和真实隔离/评分验收，不改常量绕过门禁。
5. GPU独占、可用RAM≥32GiB、磁盘可用≥100GiB、当前run增长<60GiB。所有GPU检查按真实物理UUID；不要让旧服务器PID数字指向新服务器的其他进程。

第三方来源在 `configs/safe_rsi_source_pins_20261118.json`。按固定commit获取：

```bash
git clone https://github.com/RUC-NLPIR/FlashRSI.git vendor/FlashRSI
git -C vendor/FlashRSI checkout 778e2d01a995491fe63db210dcf46db1723a4430
git clone https://github.com/ClawGym/ClawGym-Agents.git vendor/clawgym_agents_hsrl_v3_0e8526d
git -C vendor/clawgym_agents_hsrl_v3_0e8526d checkout 0e8526d8f5e0999896c74cee4d6b998fb8a0429e
```

HarnessRisk 放仓库根的 `HarnessRisk/`：checkout `07671f25b515fb936e0312dfdd3ca60b4bdf2f92` 后，先 `git apply --check` 再应用 `patches/harnessrisk_current_runtime_20261118.patch`（patch路径相对实验根）。它是本机完整tracked差异，别再叠加另一个重叠judge补丁。安全训练使用的 AgentDoG 私有 evaluator 不在 actor 挂载中，也未随Git重发，固定来源和SHA见source pins/READY release。

## 3. 必须私有迁移的资产

源机主实验根：`/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft`。

| 资产 | 源机位置/要求 |
|---|---|
| 最终14K READY发布 | `outputs/rl/hsrl_mixed_14k_50x50_v6_20261117/`，整个目录，包括assets/code/vendor/certificates/manifest/READY，不只复制train.jsonl |
| 官方4B | `checkpoints/qwen35-4b-instruct-official-851bf6e/`；所有文件与 `configs/hsrl_official4b_851bf6e.json` 独立authority核对 |
| v9 父运行与第 9 步 checkpoint | `outputs/rl/hsrl4b_harness_scaling_grpo_s_r1_sourcefix_v9_20261118/`，包括有效和拒绝轨迹、journal 和清理凭证 |
| v10 恢复证据和运行 | `outputs/rl/hsrl_v10_slot_recovery_20261118/`，其中 `run/` 仅在启动后出现；取最新恢复点前核对源机 current_run |
| v8父运行/源Bug证据 | `outputs/rl/hsrl4b_harness_scaling_grpo_s_r1_fast_v8_retry1_20261117/`；不得把旧partial拼进新运行 |
| 源修复continuation proof及真实API证据 | `outputs/rl/hsrl_hourly_monitor_20261117/audits/20261118_0600_19d7f87f-410c-4fa7-a8a3-40b018420843/`；proof引用的所有文件都需保留 |
| 租约/交接/快速算子证据 | `outputs/rl/hsrl_fastdeps_20261117/`，包括lease引用的handoff ledger；另保留 `outputs/rl/hsrl_codex_takeover_20261117/` |
| 当前指针和小时证明 | `outputs/rl/hsrl_hourly_monitor_20261117/`；`disabled`表示用户取消巡检，不表示训练完成 |

READY manifest SHA固定为 `220521662b8c40127929dbe07ad74a631d2d564a700954d1c1382e0615373c99`。copy后先核对字节和正常消费者，不要重新下载成不确定的latest或重新挑14K。若准备从原始源重建，使用既有release的stage→真实audit→publish流程，复用有效源QC；不要为迁移重复全量已经验收的源执行。

```bash
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$PWD/scripts"
python -m hsrl_v5.release --help
python scripts/train_hsrl_v10.py --help
python -c 'import os; from hsrl_source_repair_v1.adapter import Dataset; d=Dataset(os.environ["HSRL_DATASET"]); print(len(d.index)); print(d.sampler.step(0)["G"])'
```

`HSRL_DATASET` 必须由新机Codex设置为实际已迁移READY目录；输出应为14000和8。上面没有启动模型、源episode或训练，只证明真实发布消费者/CLI入口。已有READY不能手写，也不能对manifest/certificate路径全局sed后补哈希伪造身份。移动导致证书绑定失败，应通过正规的独立封装/验收恢复，保留原发布和失败证据。

## 4. 续训与目标服务器适配：先判断身份，不能假装通用入口已经完成

v10 是源机绑定的续训入口。`--physical-gpu` 只接受 1，policy、lease、official model authority 和 continuation proof 绑定源机路径、UID、隔离 socket、硬件和内核身份。新机按下方步骤做目标主机适配，并复用已修复的采样逻辑。

新机Codex的具体下一步：

1. 先只准备环境、READY、官方模型和版本依赖；读取源机当前run与进程真实身份。源机仍活跃时不启动第二个同名R1。需要实际转移训练时，在同一完整更新/全部源已清理的边界暂停领取新batch，并保留最后checkpoint、journal/RNG、累计attempt及源进程清理证明。不要在live目录rsync后声称它是原子可恢复备份。
2. 同硬件/路径/代码/lease身份且无inflight可使用原CLI `--resume`；仍需传入原 `--continuation-proof`。旧lease引用的ledger和父proof依赖也必须齐全。严格身份不匹配是失败，不能删掉判断。
3. GPU0、UID、目录、GPU型号或编译内核变化时，建立**新的版本化入口和独立运行cohort**，复用 v10 同槽位补采及现有 128 采样/GRPO/commit/retention 主体；参照 `hsrl_selected_gpu_policy_v1.py` 的受控适配方式，显式绑定真实物理UUID、owned socket、官方模型authority和新代码/内核身份。保持源数据、reward/mask、G8/K2/50:50、单条预算和优化器配方不变，原运行不改字节。这项主机适配要在目标机实现和验收，不能宣称当前Git版已在未知GPU上测试。
4. 从可信完整checkpoint在CPU核对adapter、AdamW、scheduler和RNG；延续累计已耗attempt/update，不从base静默重训、不把预算清零。跨硬件不承诺bitwise数值重放；显式记录变化，原未知inflight禁止重放。只有 `COMPLETE` 且完整resume.pt的点可恢复，`ARCHIVED`只有adapter/元数据不能exact resume。
5. 实际目标机验收包括真实rootless隔离、既定源控制/官方grader和清理、当前4B真实轨迹的token/logprob/mask，以及第一次真实backward/adapter变化/完整checkpoint。不要用mock/模拟/冒烟实验替代它，也不要用“有进程/health正常”当完成。新机脚本中涉及物理GPU/路径的修改要走项目验证流程和独立审查，然后直接继续正式RL。

源机 v10 启动成功且生成自身 journal 后，同机严格身份续训命令如下。首次从 v9 迁至 v10，改用全新输出目录并省略 `--resume`。

```bash
python scripts/train_hsrl_v10.py --resume \
  --physical-gpu 1 --dataset "$HSRL_DATASET" --model "$HSRL_MODEL" \
  --out "$HSRL_RUN" --lease "$HSRL_LEASE" --continuation-proof "$HSRL_CONTINUATION_PROOF" \
  --max-attempts 38400 --seed 267 --objective safety \
  --episode-workers 8 --model-batch-size 8 --max-batch-tokens 32768
```

源机使用venv及隔离site-packages的PYTHONPATH，见真实run manifest。新机不得仅靠上述占位变量产生看似成功的launch；必须先完成前面的身份和迁移边界验收。

## 5. 结果与后续实验

早期发布观察见 `reports/hsrl_source_handoff_observation_20261118.json`，记录当时 update7 和最近独立小时验收 update6。后续已在第 10 步采样中断处校验第 9 步 checkpoint。v10 补采预算沿用累计 1,304 次尝试，包含旧的 120 次隔离开销和这次拒绝。终点仍须审计完整 K2 与等预算比较。

先完成4B当前GRPO-S与匹配的单配置/效用奖励对照，再接DAPO/Dr.GRPO/CHS-PO和9B，不预设结果。固定同一FlashRSI初始/演化Snapshot分别评估θ0与θRL；保留未见harness/家族隔离测试。训练源注册目标不等于HarnessRisk的LLM judge ASR。LLM judge密钥只从本机私有配置读取，永不提交Git；RL本身无需该API。

小时巡检已由用户取消。迁移不要复制旧cron安装命令、不要自动唤醒当前或新会话、不要顺手恢复旧FlashRSI维修队列。下一步是目标机环境与受控入口迁移，不是开更多定时任务。
