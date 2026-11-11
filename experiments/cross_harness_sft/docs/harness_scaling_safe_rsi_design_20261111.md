# Harness Scaling × 安全 RSI：实验设计

状态：研究设计草案，替代先 SFT 再 RL 的活动设想；尚未实施或得出结果。

## 核心假设

以相同模型起点，先验证 Harness Scaling 与现有 RSI 组合是否提升通用任务能力，并测量其安全效用是否下降；再检验安全对齐 RSI（候选为加入安全约束的 CHS-PO）能否在保持通用能力的同时改善安全效用。**通用能力提升、安全下降均是待验证假设，不是预设事实。** 若下降不存在，应如实报告，并研究边界条件而不是改口径寻找退化。

不再收集教师轨迹做项目专门 SFT。优先选择具备基本工具调用能力的公开 instruction 模型作为起点；其上游 instruction tuning 和安全对齐仍须如实披露，不等于预训练 base。若使用 base 模型，单独报告无工具能力导致的执行无效，不能与 instruction 起点混淆。ClawLoop 原方法包含自蒸馏 SFT 与 GRPO：原版复现作为保真基线；禁用 SFT 的 RL-only 版本作为符合本项目约束的适配消融，不能称其为原版。

## Harness 与数据

将运行器配置为五维：系统指令 S、工具 T、观察 O、上下文 C、工作流 W；构造并冻结训练配置、配置验证集、未见组合测试集。对同一语义任务分配多个 harness，固定任务目标、攻击载荷、工具权限和环境初始状态，允许协议呈现变化；不得借组件变化改变安全权限或给模型泄露评分器。真实外部 harness（如 Gemini CLI、OpenCode、gptme 等）单列迁移评测，是否未见取决于模型后训练、设计和选模暴露审计；不声称预训练未见。

通用任务、良性安全任务和攻击任务分别按原始 family 划分 train/validation/test。AgentDojo、HarnessAudit 的可复位、可验证 train families 可作为安全训练候选；HarnessSafe、HarnessSecurity-Bench 等仅在去重、暴露和留出审计通过后用于独立测试。HarnessSecurity-Bench 的安全机制开关会影响模型可执行动作，需固定及分层，不得把环境机制效果冒充模型安全性。现有适配器/评分器的可用性逐项验证，不假定已接入。

## RSI 基线与实验矩阵

| 组 | 组合 | 用途 |
|---|---|---|
| B0 | 固定起点、无 RSI、固定 harness | 能力和安全基线 |
| B1 | Harness Scaling + 标准在线 GRPO，无递归任务/资源更新 | 单独识别 harness 多样性效果 |
| B2 | RSI 基线 + 单一 harness | 单独识别 RSI 效果 |
| B3 | Harness Scaling + RSI 基线 | 检验通用能力与安全变化 |
| B4 | B3 + CHS-PO 安全对齐 | 核心方法，评估安全收益与能力代价 |

三个候选并非等价算法。ClawLoop 合成任务、训练并诊断反馈，其参数训练路径最适合作为首个 RSI 主基线；FlashRSI 是统一平台，需要明确选择其中某个具体算法、可修改资源、目标 harness 和 worker，不能只写“FlashRSI 算法”；Dream-RSI 主要优化探索编排、基础 agent 不变，适合固定模型的资源/探索基线或独立扩展，而非直接视为参数 RL 对照。对每条基线分别记录模型参数是否更新、harness/prompt/skill 是否更新、任务是否合成，以及 oracle 是否外部提供。

每个基础模型独立比较；至少两个可实际运行的不同规模或系列模型，具体由资源与协议预检确定。每模型记录权重 hash、chat template、原有 post-training、上下文上限。最少三种独立训练种子作为目标，资源不足则标注 pilot，不声称跨模型普适。

## CHS-PO 的 RSI 闭环

候选流程：安全训练集构造/变体 → 当前策略在多 harness 真实在线 rollout → 独立任务/攻击/副作用 verifier → 安全与通用反馈分解 → 按数据源×harness 约束的 CHS-PO 更新 → 仅使用训练家族的安全诊断反馈改进下一轮任务。模型动作 token 才进入 policy loss；工具输出和隐藏 oracle 不进入策略监督。真实未授权动作、持久化和泄露必须来自沙箱状态与确定性证据；judge 不能覆盖确定性违规。评分不可用、环境故障单独记为 unscorable。

每组固定任务/环境初始化、harness 分配、rollout 数、训练 token、优化步数、生成预算和评测预算；RSI 的任务生成、worker、候选筛选、诊断与校验调用单独计入总计算量/成本。对于无法同时匹配每项资源，预注册主预算并列出实际差异。CHS-PO 奖励、约束、dual 更新、拒绝/不行动惩罚需通过真实 verifier 的正负例校准；保留原有版本作可追溯起点，不宣称已有严格约束保证。

## 评测与关键消融

通用能力采用有任务成功 oracle 的 Utility / Pass@1；随机独立采样时可另报 Pass@k，声明分母和生成设置。安全任务分别报告攻击成功率 ASR、恶意持久化、未授权副作用、可观测 Detection、benign Utility、safe-and-useful 联合成功，以及运行/评分 coverage、超时与协议错误；不同数据集指标不直接混成统一 ASR。按 family 配对报告置信区间、最差 harness、宏平均与样本量；无效轨迹不能计为安全。

优先完成 B0–B4 的同模型、同种子、同任务预算比较，保留单组件对照以分离 harness 多样性、递归改进与安全优化。进一步消融安全 verifier、dual 约束、递归诊断与合成任务来源；固定 harness 资源，和允许资源自修改的实验分开。模型选择仅用 train-harness validation，未见 harness/独立 benchmark 的测试结果不参与奖励、任务生成或选 checkpoint。

## 阶段与门禁

1. 固定模型起点和已有暴露清单，审查三种 RSI 的许可证、源码版本、训练与资源变更；将 ClawLoop 原版与 RL-only 差异登记。
2. 构建五组件配置与严格 holdout，建立跨任务/攻击家族去重及语义等价审计；完成 Harness Scaling 和 RSI 的安全 sandbox 与 evaluator 能力核验。
3. 在小预算下证明当前策略 rollout、独立 reset、verifier、参数同步、RSI 下一轮反馈真实发生；每批保存可追溯证据。
4. 在同一模型和固定预算跑 B0–B4 先导；预注册 benign Utility 容忍界限、ASR/Persistence 风险阈值、coverage 下限及模型选择规则。
5. 扩展到第二模型和独立种子；锁定 checkpoint 后，在未见配置、真实外部 harness 和独立安全 benchmark 测试。

来源：[ClawLoop](https://arxiv.org/abs/2610.14118)、[FlashRSI](https://github.com/RUC-NLPIR/FlashRSI)、[Dream-RSI](https://arxiv.org/abs/2609.14858)、[Harness Scaling](https://arxiv.org/abs/2610.29458)。当前为研究方案，未证明任何基线产生安全退化或 CHS-PO 产生收益。
