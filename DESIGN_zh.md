# 跨 Harness Agentic Safety 拒绝采样 SFT：第一版正式实验

## 研究问题

同一个安全对齐后的策略模型，能否脱离单一 agent harness，在 Codex CLI、Claude Code、
OpenClaw、Hermes 中都保持较高的任务效用与安全性？本实验先用“强教师模型 + 多个真实
harness”产生跨 harness 安全工具轨迹，再对 Qwen3.5-4B 做拒绝采样 SFT，最后在同一任务
集合上比较教师、原始学生和 SFT 学生。

代码位于 `experiments/cross_harness_sft/`，操作说明见该目录的 `README.md`。

## 三阶段流程

### 1. 真实教师 Agent 轨迹采集

教师 Agent 的定义是 `真实 harness 进程 + harness 中配置的真实模型`，不是 Python 脚本直接
请求模型 API，也不是模仿各 harness 文风的 system prompt。

AgentDojo 和 AgentHarm worker 分别加载官方任务、工具、状态与评分器；每个 episode 通过
独立 stdio MCP 服务把同一套 benchmark tools 暴露给四个 harness。采集器同时保存：

- harness 原生 JSON/JSONL 事件、退出码、版本、模型 ID、session ID、usage；
- MCP 侧独立记录的实际工具名、参数、返回、错误、副作用与风险标记；
- 官方 benchmark 的 utility、safety、final-success 和 risk-success；
- task/family/benchmark/harness/safety-condition/seed 完整 provenance。

正式配置为 `configs/formal_native.yaml`。缺少二进制、模型 ID、worker 健康检查或原生事件时
fail closed，不回退到 mock。

### 2. 严格拒绝采样与 SFT

`verify.py` 复用 SafeEvolve `core/skillrl/build_skill_use_sft.py` 的规则，过滤未完成、安全分
不为 1、benign/injection utility 不为 1、攻击成功、非法调用、解析错误、最大轮数、缺少工具
历史，以及 harmful-query 中出现工具尝试的轨迹。

`build_rft.py` 再次执行该门禁，并要求 `native_harness=true`、非空可见 reasoning、真实工具
注册表；随后精确去重、按任务族限额，并按 semantic family 划分训练/验证集，禁止同族泄漏。

每条训练样本保留完整多轮结构：

```text
system
user
assistant + tool_call
tool observation
...
assistant: <think>可对外展示的简短决策依据</think> + final answer
```

这里只蒸馏模型主动输出的可见 rationale，不声称获取或训练 provider 私有的隐藏
chain-of-thought。

`train_student.py` 参考 `E:\AI Agent\ai-agent-book\chapter8\cot-distillation\train_student.py`
重写：使用 Qwen 原生 chat template，并把 system/user/tool observation 的 label 设为 `-100`，
只监督所有 assistant span（含工具调用与最终答案）。默认在单张 RTX 5090 上使用 NF4 QLoRA。
训练没有 CPU/mock 成功分支，完成后必须生成 PEFT checkpoint 与绑定数据 SHA-256 的
`training_manifest.json`。

### 3. 同题配对评估

分别用教师、base Qwen3.5-4B、SFT checkpoint 在完全相同的
`benchmark × task × harness × safety condition × seed` 单元上采集评估轨迹。评估数据不做
拒绝采样，避免把失败样本过滤掉。

`compare_models.py` 输出：utility、safety、Pareto success、SFT 相对 base 的配对改善/退化/
持平数、反思/回溯/验算行为比例，以及：

```text
capability_recovery = (SFT - base) / (teacher - base)
```

分母为零时返回 `null`，缺失单元单独报告而不插值。

## 安全条件

首版同时收集以下条件，以便之后做跨 harness 与 harness-safety-module 交互分析：

- `none`：无额外安全模块；
- `static_prompt`：固定安全 system instruction；
- `safeevolve_skillbank`：按 task metadata 从 SafeEvolve SkillBank 检索运行时技能；
- `pretool_guard`：调用执行前检查；harmful-query 一律禁止工具调用，也可配置禁止工具名与参数值。

这些条件不会取代 benchmark 原生 verifier。

## 正式验收门槛

- 四个 harness 均有真实二进制版本和真实原生事件证据；不可用 harness 标记失败，不补模拟值；
- AgentDojo/AgentHarm worker 返回官方版本，工具调用由 benchmark runtime 实际执行；
- 原始、接受、拒绝轨迹与逐原因统计全部保留；
- train/validation semantic family overlap 为 0；
- checkpoint 具有训练 manifest 和数据哈希；
- 教师/base/SFT 使用相同评估单元，缺失单元明确披露；
- 结论同时报告 utility 和 safety，不以单一平均分掩盖安全退化。

## Mock 边界

独立仓库不包含 mock teacher、portable case 或 controlled harness profile。正式配置、安装
脚本、采集脚本、训练脚本和评估脚本只接受真实 harness 与官方 benchmark worker。
