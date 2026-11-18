# 4B FlashRSI × HarnessRisk 已完成结果

核查时间：2026-11-14 18:54 CST。当前为固定 4B 权重下的 FlashRSI 资源演化基线，尚不是 CHS-PO 参数后训练或完整三阶段安全 RSI 的结果。下表全部使用修复 oracle 文件泄露之后的运行；完整128案例仍包含演化训练/开发暴露，不能当完全未见泛化测试。

## 同协议、全量有效的主要对照

| MiniAgent 系统 | 有效规则评分/计划 | 规则二值攻击目标达成↓ | 严格效用 oracle 均值↑ | 含回答fallback效用均值↑ | Judge攻击成功率↓ | Judge效用↑ |
|---|---:|---:|---:|---:|---:|---:|
| 初始 Snapshot |128/128|69/128，53.91%|43.03%|86.13%|32.03%|64.84%|
| OPRO |128/128|77/128，60.16%|49.67%|83.46%|37.50%|66.41%|
| EvoPrompt |128/128|64/128，50.00%|42.64%|84.11%|35.16%|64.06%|

三组 judge 均 128/128 已评分。规则完整攻击目标由 `details.attack_success.passed` 汇总，区别于官方 `attack_success_rate` 可能含谓词部分分；严格效用是 `utility_oracle_score` 的均值，不是二值任务成功率；fallback 是最终回答回退参与后的效用。规则与 judge 不混合，不是 Pass@k。

OPRO：规则攻击成功率 +6.25 个百分点、严格效用 +6.64 个百分点；judge 攻击成功率 +5.47、效用 +1.56 个百分点。单个已暴露数据运行中出现效用提高伴随安全下降的迹象，不证明所有 RSI 都如此。EvoPrompt 的规则攻击成功率降低 3.91 个百分点，但 judge 攻击成功率提高 3.13 个百分点，不能据此宣布安全改善；两类效用均略降。没有多 seed 和显著性结论。

## 仅完成尝试或部分评分的参考结果

| 系统 | 有效规则评分/计划 | 完整攻击目标达成/有效案例 | 严格效用均值 | Judge攻击成功率 | 解释 |
|---|---:|---:|---:|---:|---|
| MiniAgent ACE |127/128|66/127，51.97%|48.10%|33.86%|选回初始 Snapshot，非演化提升|
| MiniAgent Evo-Harness |127/128|66/127，51.97%|45.34%|33.86%|选回初始 Snapshot，非演化提升|
| Hermes v3 初始 |127/128|38/127，29.92%|14.83%|18.11%|缺1例；算法未完成|
| OpenCode strict v2 初始 |124/128|59/124，47.58%|42.07%|28.23%|独立适配协议|
| OpenCode strict v2 ACE |121/128|61/121，50.41%|40.70%|32.23%|选回初始 Snapshot，覆盖集不同|
| OpenClaw 初始 |13/128|3/13，23.08%|76.28%|46.15%|覆盖仅10.16%，不能代表全量|
| Claude Code native v3 初始 |23/128|9/23，39.13%|39.49%|39.13%|覆盖仅17.97%，不能代表全量|
| Codex v6 初始 |17/128|3/17，17.65%|1.96%|0%|覆盖仅13.28%，judge效用也为0|
| Pi v3 初始 |18/128|8/18，44.44%|62.50%|55.56%|覆盖仅14.06%，不能代表全量|

参考组 judge 只覆盖各自有效轨迹，未评分案例不能当安全。不同harness、协议和有效支持集不能直接横向排名。选回同一初始 Snapshot 的重复运行仍可能出现分数差异，不能归因于演化。

MiniAgent GEPA、SkillOpt；OpenCode strict v2 OPRO/GEPA/EvoPrompt；Hermes OPRO/GEPA/EvoPrompt；Claude Code 六种方法目前失败或待修复。独立重跑及其他未完成单元持续由修复循环追踪。OpenCode SkillOpt/Evo-Harness 因上游 Snapshot 无 skill 组件不适用。

证据目录统一在 `outputs/flashrsi_harnessrisk_4b_private*`；主要对照是 v1 下 MiniAgent initial_full_v2、opro_full、evoprompt_full，各自 result、selected_cases 官方评分和 judge manifest。最新维修状态见 `outputs/flashrsi_harnessrisk_4b_private_v1_20261113/monitor/matrix_loop/matrix.md`，表中历史数值不拼接新协议。
