## 当前停用状态（2026-11-18 11:39，用户明确要求）

用户已要求删除本巡检loop。小时cron已删除，monitor/disabled存在；禁止依据下方历史契约重新创建定时器或因旧唤醒消息自动恢复巡检。此停用不等于训练完成，训练继续。只有用户新的明确授权才可恢复。

## 2026-11-18 已验证的v9恢复链（优先于下方v8历史描述）

当前运行从current_run.json读取，已迁移v9 sourcefix；不可把v8路径/PID当当前。新协议只修复AgentDoG工具执行namespace的普通异常类，原官方grader与奖励未变。父update4模型/AdamW/scheduler保留，采用已对账、全部关闭后的最终failed journal RNG；632次已耗不清零，112valid但未更新+8invalid共120次旧partial保留隔离，不能再拼入新128。恢复已通过真实新轨迹/官方评分/joint提交、精确checkpoint状态及实际隔离验收，证据路径在current_run。

新run的training_metrics.jsonl只含global step4以后的新更新；父run持有前4次。统计需沿continuation父链，不能用新run episode数假定等于128*全局updates，也不能把父隔离112条算作修复后的有效训练。dispatch_approval.json采用绝对路径pins，不沿用旧approval.code_sha256格式。旧inspect_evidence.py含v8固定PID和扁平metrics假设，不可原样运行。后续必须对修复后AgentDoG正式轨迹、下一真实梯度/裁剪/base mutation/checkpoint验收，已有源API回归不能冒充全300更新验证。较宽回归的慢inventory fixture状态看本次audit/scoped_regression_status.json，未完成不能称通过。

38,400累计尝试/最多300更新不变；120次隔离开销意味着逐更新预算上界299，299不是K2配对完整的承诺。末端审计完整128更新与K2覆盖，如预算不足不得静默重置尝试、删难例或加预算。遵循本契约原完成条件，未完成继续整点巡检，无新timer。

# Harness Scaling RL 每小时 Codex 巡检与修复契约

2026-11-17 用户新授权：每小时唤醒当前Codex会话，全方位检查当前Harness Scaling RL；正常继续训练，出错保存证据、定位修复并安全恢复。此授权覆盖历史“不得恢复定时任务”仅针对本条新RL小时任务；不能恢复旧FlashRSI/双卡/30分钟维修cron。当前任务不创建新的Codex会话，不重建定时器。

## 入口与当前运行

先读主项目AGENT.md，再读本文件与监控目录`outputs/rl/hsrl_hourly_monitor_20261117/current_run.json`。所有相对路径以experiments/cross_harness_sft为根。该指针记录当前实际v8 retry1；若后续正式修复迁移，由成功恢复的同一轮更新指针、记录父子运行和旧证据，不把PID和旧路径永久当作真相。监控目录同上，小时报告`hourly_report.md`，机器状态`status.json`；投递回执`last_queued.json`只说明已入队，不等于巡检完成。

每次先写本轮开始时间、请求ID、实际运行目录到status.json。同一会话串行执行，收到后续小时请求时若本会话正在维修则合并检查，不另建会话/重复维修器。检查实际活维修进程、其他actor、worker.lock、运行run.lock/GPU租约；有其他维修器持锁或正在改相同代码时先核对其进度，不并行编辑或启动相同模型。调度器wake.lock仅保护投递过程，不是假装跨工具调用持有的维修锁。

## 必须实际检查

1. 真实PID/starttime/UID/命令、GPU UUID/显存/利用率/错误、CPU/内存/swap/磁盘、线程及源容器、actor/controller存活与log。只操作已核对身份的自有进程；当前授权训练GPU1，其他卡先读最新用户授权和AGENT，不杀他人进程/抢卡/静默改物理索引。锁和租约必须对应真实物理设备。当前资源门槛可用RAM32GiB、磁盘100GiB、运行增长60GiB，不能移除门禁。
2. state.json可读状态与可信自有journal.pt、run_manifest、官方模型身份、14K READY及data/code/kernel/lease批准、G8/K2、16组128条、50:50、seed、objective、已耗300update/38400attempt预算。按实际manifest核对，不把任务池14K当已训练14K，不把reserved/进程存在当完成。
3. 每个wave/episode身份、source valid/done、官方checker及utility/cost/mask、privacy/reset/cleanup证据、真实模型动作/工具输出白名单（无隐藏grader/oracle）、旧/参考logprob与response/action mask逐token对齐及有限性。`private_grader_boundary.json`是AST静态检查，`source_execution_performed=False`为预期；真实执行必须核对独立`private_grader_response.json`及隔离/清理回执与最终分数，不能混淆两者。模型不完整工具调用/任务失败可以是正常负奖励；Docker/checker/隔离/接线错误必须invalid，不能伪装0分。禁止用LLM judge替代本来已冻结的源奖励或重复任何已完成judge/API调用。
4. 实际batch、active环境、采样/model/reference/source阶段耗时、吞吐、GPU样本、最近波/segment修改时间，和此前报告比较。GPU低或进程存在本身都不能判定卡死；先区分采样、源工具、编译、反传、持久化、等待资源。异常长时无任何轨迹/波/梯度进展或CPU/GPU无工作时，取实际栈/日志/子进程证据定位；不能盲目增加token、缩G/波或删难例来提速。
历史episode须绑定其各自step对应policy/adapter/config/metrics，不能要求旧step轨迹等于当前最新policy；只有待更新的同一组才禁止混合policy。

5. 满128后的prepare_update/梯度/AdamW step：PG/KL tokens、组内方差/零优势组、有限loss与grad norm、实际adapter前后hash/effective_updates、冻结base mutation guard、policy_version、K2下一配置同任务当前策略重采样。zero variance与0有效更新如实记录，不能捏造提升或强迫权重变化。
6. checkpoint和恢复：journal指向已完整安装的COMPLETE、adapter/optimizer/scheduler/RNG和身份一致；initial、最近两份及显式best/terminal恢复点保留，所有adapter和轨迹保留。核对retention ledger与ARCHIVED标记，不能从已清optimizer点声称可精确恢复。训练末尾必须有实际metrics/最后checkpoint和真实完成审计。

## 出错必须形成修复闭环

先保留错误、源码/配置版本、当前state/journal及inflight、进程和源容器身份、日志与已有结果，写本轮诊断与repair action。按installed create-verification-skill和项目真实工作流做定位、定向检查、实际受影响CLI路径验证；不做模拟/冒烟实验，不绕过沙箱/隐私/评分/设备门禁，不泄露密钥，不错误计零/混协议/删样本/重置预算。未明确授权的奖励/任务池/配比/G/K/预算变更不作为bugfix。

- 无inflight、同一代码/模型/数据/lease身份且checkpoint完整时，使用真实--resume；不得新开重复actor。
- 有inflight必须先确定每个已reserved槽实际开始/完成/评分/关闭情况。未知状态不能自动重新执行。先核对源进程/容器清理和已有episodehash/策略身份，保存明确reconciliation；缺失采样后RNG等恢复证据不能冒充精确恢复。需要新版本时采用独立目录，绑定父checkpoint/代码/optimizer/scheduler/RNG及已耗预算，不静默从base重训或拼接旧污染轨迹。
- 修改代码前确认无其他编辑者，保留活动代码或在新版本入口实现，适当独立审查；恢复后必须观察真实新轨迹/源评分/参数更新或具体正在执行阶段，再写“恢复成功”，只有Popen/health不够。
- 若资源被他人占用或无安全修复路径，记录具体阻塞和证据、保留未完成任务；下一小时继续核查，不能把失败当完成、不能反复报告但不诊断，也不能为启动杀他人进程。已完成的源任务/模型/API请求不作无理由重复。

正常就保持当前训练运行，不为了巡检重启。每轮追加hourly_report.md：时间、run/policy身份、真实reserved/已评分/已提交/更新/effective更新、当前配置、错误/资源/吞吐、诊断、修改与验证、实际恢复结果及剩余不确定项；结束时写status.json的finished/blocked和请求ID。运行迁移成功后更新current_run.json、AGENT.md和恢复链证据。

## 完成条件和停用

当前R1达到已批准预算，controller真实退出0、state预算完成、最后完整128更新/有限梯度与adapter/checkpoint/冻结base证据齐全、无未知inflight/源进程/隔离缺口后，生成completion_audit.json并最终报告真实update/effective计数及能力评测限制。若没有另一个明确授权的后续训练臂，创建监控目录disabled并仅移除crontab中`HSRL_HOURLY_CODEX_LOOP_20261117`这一行，保留报告和所有训练结果。不要把“当前预算结束”写成安全能力已提升。未完成/失败不得提前创建disabled。

## 实际调度与验证

本机cron每小时整点（服务器Asia/Shanghai）执行scripts/watch_hsrl_hourly_codex.sh，经codex queue向固定本会话投递短提示，再读取本契约。cron和Codex本地app-server/认证须可用；断机/服务离线不能保证模型被唤醒，错误写wake.log/queue_events.jsonl，后续小时会再尝试。只投递到同一会话，不新建会话，不另起模型服务。脚本支持disabled和flock防重入。

复核命令：bash -n scripts/watch_hsrl_hourly_codex.sh；systemctl is-active cron；crontab -l（确认唯一标记行）；检查last_queued.json与requests回执。脚本disabled/锁竞争路径可在尚未启用cron前用真实本地文件验证；最终真实codex queue回执才证明投递链路可用。投递成功不是巡检完成；完成以hourly_report/status和真实修复/恢复证据为准。验证不运行任何额外RL或外部judge/API。
