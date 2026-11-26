# Harness Scaling RL 的 CHS-PO v2

CHS-PO v2 检验两个问题：模型完成一部分合法工作时，哪些证据足以给它正优势？同一组里有几条轨迹都完成了合法任务时，怎样鼓励避开已登记风险的做法？v2 在经过审计的两类轨迹上调整优势。离线回放已经测到训练信号的变化；FlashRSI 与 RSIHub × HarnessRisk 上的 ASR、Utility 仍待正式配对评测。[冻结实验配置](../configs/chspo_v2_design.json)

## 与 GRPO-S 匹配的训练条件

两个训练臂分别从官方 Qwen3.5-4B 的相同初始参数开始，各自初始化 rank 16、alpha 32、dropout 0 的语言层 LoRA 和 AdamW。固定比较第 200 步。每步从 ClawGym 和 AgentDoG 各采一个同任务、同配置的 G8 组，共 16 条当前策略轨迹。K=2 表示相邻更新在同一任务上轮换 OA18 配置。每个方法的预算为 3,200 条有效轨迹，另外记录无效尝试和生成 token。采样 seed 为 267、temperature 为 0.7、top-p 为 0.95。两臂均使用学习率 1e-5、两步预热、PPO clip 0.2、参考策略 KL 系数 0.001、梯度范数上限 1。按原 GRPO 日程，前 25 步使用终止符 248044，第 26 步起使用 248044 和 248046。

GRPO-S 先计算合法效用减来源风险代价、再减 0.1 倍协议代价，随后在 G8 内标准化。CHS-PO 先在组内标准化合法效用，得到 Aᵁ（总体标准差加 1e-6）；不适用或组内无差异时，效用优势为 0。两种方法都只在模型生成的有效动作 token 上计算梯度。

## 从原生回执到最终优势

1. **核验回执。** 对采样账本中的槽位、策略版本、环境重置、执行和评分证据做一致性检查。记录解析失败、被阻断的调用与实际执行的协议违规。当前两个来源均无可用的持久化判定器，该通道不计入分母。[原生证据核验](../scripts/a6000_vm/chspo_audit.py)
2. **恢复有证据的部分效用。** 对 ClawGym 的同一 G8，寻找最终工作区摘要等于 reset 摘要的同伴，以其官方分数估计初始分。候选轨迹须有工作区写入证据，正常结束，满足 0 < Utility < 1、总分高于参照、每个可比分项均无回退，且协议代价为零。只有原审计中“是否合法参与”这一项未知时，才补全该判断。没有参照或同一工作区的官方分数相互矛盾时弃权。该证据支持轨迹级归因；原始任务分数继续用于计算 Aᵁ。[初始分对照](../scripts/a6000_vm/chspo_progress.py) · [v2 优势规则](../scripts/a6000_vm/chspo_v2.py)
3. **计算约束优势。** 依数据源与 harness 配置分别维护 attack、persistence、protocol 三个乘子。初值依次为 0.25、0.50、0.10，下界为 0.25、0.50、0，预算为 0、0、0.10，上界均为 1。每步按本批可观测样本更新一次，增量为 0.08 ×（观测速率 − 预算）。使用更新后的乘子。确认违规时，优势为 min(Aᵁ, 0) − min(1, Σλc)；审计不完整时取 min(Aᵁ, 0)；确认完全未参与合法任务时取 min(Aᵁ, 0) − 0.3。其余轨迹使用 Aᵁ。无观测的通道保留原乘子。[约束目标实现](../scripts/hsrl_a6000_chspo.py)
4. **奖励同效用下的安全成功。** 对 AgentDoG 中合法效用为 1、正常完成、合法参与及协议执行均已核验、无已观测风险的轨迹 i，只选择同一 G8 内其他满分且有来源风险观测的同伴。令 p 为这些同伴的平均登记风险代价，以 max(Aᵢ, λ_attack × p) 作为最终优势。没有可比同伴则沿用原值。例如，λ_attack 为 0.3、三个可比同伴中有两个发生登记风险，则该轨迹的优势下限为 0.2。该规则不为单纯拒绝合法任务提供奖励。[同伴比较实现](../scripts/a6000_vm/chspo_v2.py)

## 历史回放

在第 1–179 步的既有 GRPO 轨迹上，对 v1、v2 使用相同的批次与对偶状态。第 146 步有一份评分后发生变化的快照，两个版本均剔除整批。其余 178 批的 2,848 条轨迹结果如下。[聚合回放回执](../reports/chspo_v2_replay_0179.json)

| 回放量 | 数量 |
|---|---:|
| 两来源参与比较的轨迹 | 各 1,424 |
| 新取得合法部分参与证据的 ClawGym 轨迹 | 105 |
| 其中优势由非正变为正的轨迹 | 101 |
| 新获得正优势的 AgentDoG 安全成功轨迹 | 15 |
| 两个版本中负优势完全一致的已确认风险轨迹 | 135 |

这些计数衡量固定轨迹上的优势赋值。模型实际更新后生成的轨迹、攻击成功率和任务效用需要独立测量。

## 正式验收

使用两条冻结评测线：FlashRSI 的 OpenCode ACE 选定快照，以及 RSIHub × HarnessRisk 矩阵中的冻结 H0 和各自搜索选中的配置。每个可用模型层（GRPO-S 与 CHS-PO v2 checkpoint）在相同的 26 个封存 test 案例上按规则评分与 LLM judge 分开报告。RSIHub 四种方法的搜索预算、候选谱系、覆盖要求和矩阵分母以[正式方案](rsihub_harnessrisk_baseline_20261126.md)为准；不再沿用旧 Harness-S sibling 救援的 104 条轨迹计划。单案例 evaluator/task-vector smoke 不是算法结果，正式算法闭环和 matrix 尚待运行。旧 Harness-S 方案已[归档](archive/harness_s_harnessrisk_baseline_superseded_20261126.md)。

官方规则和固定 LLM judge 分表，按案例配对，报告分母、缺失、四项指标以及 CHS-PO 与 GRPO-S 的差值。RSIHub 搜索配置和 seed 以[正式方案](rsihub_harnessrisk_baseline_20261126.md)为准：三个预登记 seed，逐 seed 报告，再给均值、标准差和按 case ID 配对的 bootstrap 95% 区间。比较只在相同模型身份、harness、search 条件、seed、案例和评分来源内配对；不可把不同 RSI 算法或未完成的 checkpoint 合并成一个效果。CHS-PO step 200 权重完成并冻结后才能进入该层；此前单案例 smoke 只说明 adapter contract 通过，不构成模型效果。历史评测已覆盖这 26 例，须另行披露，并核对本轮训练、快照和 checkpoint 选择的暴露。[配对验收实现](../scripts/score_chspo_v2_harnessrisk.py)

### CPU 复核

在实验目录、已安装项目测试依赖的环境中执行：

```bash
PYTHONPATH=scripts CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_chspo_v2_credit.py tests/test_chspo_progress.py \
  tests/test_native_chspo_audit.py tests/test_hsrl_a6000_chspo.py \
  tests/test_score_chspo_v2_harnessrisk.py
```
