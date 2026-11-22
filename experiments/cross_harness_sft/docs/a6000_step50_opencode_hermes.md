# A6000 上的 step‑50 × OpenCode / Hermes 配对评测

本机训练好的同一份 Qwen3.5‑4B LoRA 用于两个 harness。训练入口为 `scripts/train_hsrl_v10.py`。本页的入口运行 FlashRSI 的原生 Python headless 循环，对同一个 Snapshot、同一案例分别关闭和启用 adapter。每个独立配置各有 128 对、256 条轨迹。机器分工为源机 MiniAgent、A6000 OpenCode 和 Hermes。

| Harness | 独立运行的 Snapshot | 选回同一 Snapshot | 缺少选中快照 | 不适用 |
|---|---|---|---|---|
| OpenCode | 初始、ACE v4 | OPRO v5、EvoPrompt v4 与初始相同 | GEPA | SkillOpt、Evo-Harness |
| Hermes | 初始、GEPA v14、OPRO v14、EvoPrompt v14、ACE v3、SkillOpt v3 | — | Evo-Harness | — |

各组原运行名、Snapshot ID 和文件 SHA‑256 在 `configs/step50_hermes_opencode.json` 中。同算法的其他历史版本单列。两个 harness 共需 1,024 对。只对 8 个独立配置推理和评分。

## 取得固定输入

在本项目的 `experiments/cross_harness_sft` 下操作。指定实验目录和本机已有的权重目录。以下命令中的 `DATA_ROOT` 可指向 A6000 上任意有写权限的实验存储。

```bash
export EXP="$PWD"
export DATA_ROOT=/path/to/a6000-eval
mkdir -p "$DATA_ROOT"
export FLASHRSI="$DATA_ROOT/FlashRSI" BENCH="$DATA_ROOT/HarnessRisk"
export BASE=/path/to/Qwen3.5-4B
export ADAPTER=/path/to/step_0050
```

若已有独立的固定版本检出和对应评分器，可直接复用。以下命令针对新目录。

```bash
git clone https://github.com/RUC-NLPIR/FlashRSI.git "$FLASHRSI"
git -C "$FLASHRSI" switch --detach 778e2d01a995491fe63db210dcf46db1723a4430
git clone https://github.com/Baiyajing/HarnessRisk.git "$BENCH"
git -C "$BENCH" switch --detach 07671f25b515fb936e0312dfdd3ca60b4bdf2f92
git -C "$BENCH" apply "$EXP/patches/harnessrisk_step50_rule_reader.patch"
```

评分器补丁沿用源机读取 Hermes 等轨迹格式的改动。入口对评分器文件和每个案例单独核验。仓库随附的是 128 个案例的身份清单和 Snapshot 重建材料。程序用固定 FlashRSI 检出的字节和已选中的提示词、技能变更，在临时目录逐字节重建历史 Snapshot，并先验证原文件 SHA‑256。已有交接归档的机器也可传 `--artifacts /path/to/unpacked/harnessrisk` 直接读取原件。

如需补取模型，基座为 `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。适配器为 `Limax11/harness_alignment@8efba1d7714196e2a69216d8cfda5944dc2a21b0` 中的 `online_rl/a6000_r1_lora_2g8_300_20261121/step_0050/`。保留目录中的四个发布文件。入口复核模型分片、分词器和 adapter，在加载后逐项比对 496 个 LoRA 张量。

## 运行和计分

使用 A6000 的 4B 环境。源机核验环境为 Python 3.11、PyTorch 2.11、Transformers 5.9、PEFT 0.21.1、safetensors 0.8、httpx 0.28.1、openai 3.22.1、PyYAML 6.0.3。环境应能导入 `sqlite3`。使用 Conda 时，若系统 libstdc++ 缺少 ICU 所需的 CXXABI，可在当前命令环境优先使用该 Conda 环境自己的 lib。以下操作使用 Bash。

```bash
export PY="$(command -v python)"
COMMON=(--flashrsi "$FLASHRSI" --bench "$BENCH" --base "$BASE" --adapter "$ADAPTER")
for h in opencode hermes; do
  "$PY" scripts/step50_native_pair.py prepare --harness "$h" "${COMMON[@]}" --output "$DATA_ROOT/step50-$h" || break
done
```

`prepare` 核验输入，并用每个独立配置的原生内核和真实 tokenizer 走到 128 个案例的首个模型请求。输出 `input_audit.json`。这些请求只做本地编码，正式轨迹和 judge 的分母均从零开始。

选择分配给本实验的空闲卡。下例用物理卡 0。模型进程要求仅有 loopback 的独立网络命名空间。也可在平台已配置的 `--network none` 单卡容器内直接运行 Python 入口。A6000 上若仍有训练进程，先完成卡的交接。建议在 tmux 中顺序运行两组。

```bash
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)"
export HF_HUB_OFFLINE=1
for h in opencode hermes; do
  unshare -Urn -- "$PY" scripts/step50_native_pair.py run --harness "$h" "${COMMON[@]}" --output "$DATA_ROOT/step50-$h" || break
done
```

新协议 `flashrsi-native-case-documents-v1` 固定 BF16、贪心生成、关闭思考、每个案例 3 轮所有者消息、每轮最多 12 次模型请求、每次最多 4096 个新 token。原生循环保留调用与停止语义，历史所有者消息和最终回答在三轮间传递。工具只读当前案例的离线材料，可将普通文本写到该案例内存中的 `workspace/outputs`。Hermes 可读取该 Snapshot 的一份文本技能。每个案例重新创建状态。应比较同一协议下的新 base 和 step‑50 数据，历史原协议 ASR 单列。每个组内，案例交替先跑 base 或先跑 step‑50。

每条有效轨迹产生 `identity.json`、`calls.jsonl`、`owner_outcomes.json`、`privacy_audit.json`、`trajectory/`、`official_rule_evaluation.json` 和 `complete.json`。原目录重启复核已有完成回执，并保留未完整目录等待核对。程序会在首个有效结果前连续三条失败时退出，或在全组扫描后以非零退出码报告缺口。先检查 `error.json`；协议或代码变化后用新的输出目录，并保留前一批证据。

judge 在允许联网的独立进程中运行，使用当前环境已配置的 `SHENGSUANYUN_API_KEY`。有关凭据的环境变量只需提供给 judge。每个 harness 产生 `run_manifest.json` 后即可在另一个终端启动相应 judge；下例可在两个 rollout 队列结束后依次执行。

```bash
for h in opencode hermes; do
  "$PY" scripts/step50_native_judge.py "$DATA_ROOT/step50-$h" --bench "$BENCH" || break
done
```

judge 为 `openai/gpt-5.4-nano`。已保存回执可复核。已发起且缺回执的请求保留待核账。用同一脚本离线读取各臂的规则覆盖、judge 覆盖、共同案例数，以及交集上的 judge ASR、Utility、Persistence 和 Detection。

```bash
python scripts/step50_native_judge.py "$DATA_ROOT/step50-opencode" --bench "$BENCH" --summary
python scripts/step50_native_judge.py "$DATA_ROOT/step50-hermes" --bench "$BENCH" --summary
```

独立配置验收要求两侧各有同一套 128 个案例的身份、隐私、官方规则和 judge 回执，共同配对数为 128。覆盖不足保留实际分母。重选初始的组保留别名，不能算作另一次独立复现。
