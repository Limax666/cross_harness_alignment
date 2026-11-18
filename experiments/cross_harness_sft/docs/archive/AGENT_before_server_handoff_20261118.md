## 2026-11-18 11:39 用户明确取消RL小时巡检loop

已删除唯一cron标记HSRL_HOURLY_CODEX_LOOP_20261117并创建monitor/disabled；旧脚本实际disabled路径验证返回0且无新投递。下方“下一小时继续/cron active”均为历史，不再有效；未经用户新的明确要求，不恢复任何巡检定时器。此次只停自动巡检，训练actor3467200/controller3467186仍运行，checkpoint/evidence不变，不是训练完成。证据：experiments/cross_harness_sft/outputs/rl/hsrl_hourly_monitor_20261117/stop_by_user_20261118_113913/verification.json。

## 2026-11-18 11:00巡检：6/effective6，第七步32/128，正常继续

GPU1 v9 actor3467200/controller3467186不变，policy6/S1T0O0C1W1/K_block3第一成员，第七步32joint/32有效官方评分，reserved928＝768已接受+120旧隔离+32当前提交+8reserved。新协议累计288joint/288有效，按历史step各自policy/adapter核对288checker/729段/576独立容器清理/334runtime、code/data/lease及完整initial/update5/6 checkpoint通过，800有效接受+当前提交，旧112valid未更新+8invalid不混。

4批各10.90/17.57/10.88/12.78分钟、27.3176动作token/s，当前32条通用组2条utility>0/sum1.0。轨迹/模型仍推进，无卡死/OOM/新异常；11:02核查7活源容器+1清理，RAM148.4GiB、disk421.72GiB、run1.177GiB充足。第七次参数更新未发生，保留上次preclip19.841835/实际clip约1与原尖峰未定根因diagnostics。训练得分不能当独立ASR/提升。旧广QA缺最终回执仍未验收，不重复。

正常保持训练，无改码/重启/新timer，其他GPU不动；原唯一小时cron继续12:00。准确进度/完整证据看current_run/status/hourly_report及本轮audit。时间戳audit脚本含当时step边界，后续按真实metrics映射历史与当前policy，不原样套旧断言。

## 2026-11-18 08:00巡检：源修复已在正式组和真实update5验收，第六步继续

本轮至08:15验收updates5/effective5/policy5。第五步128/128真实官方评分、339段/256清理/334runtime通过；此前失败的AgentDoG group14正式8/8有效，实际get_patient_info9次，无原NameError。新轨迹不等于旧失败RNG重放，旧精确失败调用的API回归证据仍保留。真实update5 loss0.066493726、preclip grad0.610355735、AdamW重建实际梯度0.610355738（本次低于1不需裁剪）；496个adapter张量变化、optimizer step5/scheduler5/完整checkpoint/base mutation guard和RNG验收通过，原梯度尖峰根因仍未证明。

修复后该步采样90.46分钟+反传6.73分钟+保存8.89秒，总97.35分钟。GPU1 actor3467200/controller3467186未重启；第六步同16任务K2第二配置S0T0O2C2W1/policy5已真实启动，reserved768、当前0条joint（640已接受+120旧隔离+8reserved）。current_run新增latest_update_verification指向本轮real_update5_verification.json；新run metrics只含全局step4起，父链前4次不混，旧partial112valid+8invalid仍隔离。08轮inspect脚本为policy4快照，未来要按真实metrics/assignment适配新step，不能原样沿用硬编码边界。

旧较宽pytest3461085已退出且旧session15190无最终结果，**未验收、不记通过**；14定向JUnit及真实API/正式组/optimizer路径证据保留，不重复全量旧QA。训练源数据：本批AgentDoG注册风险目标16/64、utility_sum35；ClawGym utility_sum0.095238，不能当独立ASR或模型提升。详见本轮小时报告与audit。无新训练Bug/改码/重启/新timer，其他GPU不动，原唯一小时cron继续09:00。

## 2026-11-18 07:00巡检：v9继续，4次更新，第五步40/128提交

07:05:47收尾原子快照：GPU1 v9 actor3467200/controller3467186正常，updates4/effective4/policy4/S0T2O2C2W2；本phase40条joint提交/40条有效官方评分、reserved680。旧112valid未更新+8invalid隔离保持，不混协议；有效接受512+当前提交40=552，不能把reserved当完成。334runtime/40checker/48段/80清理及完整initial恢复点、代码/数据/lease/进程身份验收通过。07:02观测3个live源容器（该批已完成，下一8槽真实采样）；无OOM/训练异常，不重启、不改训练代码，不动其他卡。新gradient未产生，原尖峰诊断继续；AgentDoG group14正式采样尚待走到。较宽旧QA目录pytest3461085/session15190仍在I/O扫描、无final结果，不称通过；详见本轮audit/scoped_regression_status.json。唯一原小时cron继续、下一次08:00，无新timer；实际进度和恢复链仍读current_run.json及status/report。

## 2026-11-18 06:45：RL源执行Bug已修复，GPU1正式v9续训已通过真实首wave验收

本条覆盖下方v8当前运行描述。权威指针仍`experiments/cross_harness_sft/outputs/rl/hsrl_hourly_monitor_20261117/current_run.json`，已迁移至`hsrl4b_harness_scaling_grpo_s_r1_sourcefix_v9_20261118`，controller3467186/actor3467200，worktree入口scripts/train_hsrl_v9.py，修复commit855effb。05:35旧v8在get_patient_info异常处理中因SAFE_BUILTINS缺普通异常类而NameError；独立新profile仅补ValueError/TypeError/Exception，原源代码/数据/官方grader/reward/沙箱均不改。14定向检查、真实失败调用的Dataset Docker/官方grade/cleanup和两个独立审查通过；较宽既有源库回归单独记录pending，不冒充全通过。

真实恢复点为父update4，496 AdamW状态/scheduler/adapter精确相等，恢复最终失败journal的已知RNG；保留632已耗尝试。旧112条valid但未更新+8条失败中止全部保留隔离，同16任务/同配置全步重采样。新正式首wave已核验8条valid/官方评分/joint提交，仍4/effective4，不能声称第五次更新完成。当前协议不混旧partial，metrics从global step4起，前4次沿父链读取；禁止照搬硬编码v8 PID/metrics的旧审计脚本。总38,400尝试不变，299仅逐更新预算上界，末端需审计完整更新/K2，不能加预算。修复后AgentDoG group14正式采样仍待走到，真实源API回归已通过。梯度尖峰根因未证实，保持原diagnostics追踪，不称安全/效用已提升。

详见小时报告与audits/20261118_0600_19d7f87f-410c-4fa7-a8a3-40b018420843。GPU1继续独占RL，GPU0/2/3不动；无新timer，原唯一小时cron继续，下一次07:00。

## 2026-11-17 21:07：用户新授权RL每小时Codex巡检，已新增唯一小时cron

11月18日05:00巡检于05:04完成：仍4/effective4，第五步policy4/S0T2O2C2W2提交64/128，累计576提交/579有效源评分（3条已评分未joint），reserved584。334runtime/579checker/1168段/1158清理/完整checkpoint及5个真实AgentDoG活容器(512MiB/32PID)通过；无OOM/异常，保持采样。无新梯度结果，最近25.929/实际裁剪约1，原2913尖峰根因未定，继续diagnostics监控。当前64条ClawGym Utility0仍为模型失败，非infra invalid；安全/效用提升未验证。无训练改动/重启/新timer，下一次06:00。

11月18日04:00巡检于04:07完成：第四次真实更新4/effective4验收，总107.63分钟、loss0.07148891，裁剪前grad norm25.929（前次2913.166，本次回落），实际AdamW状态重建裁剪后范数0.99999996；原尖峰根因未证实，继续diagnostics/gradient_spike_followup_step_0003.json监控。累计512有效评分/提交，1024清理/1084段/334runtime及完整新checkpoint通过；第五步policy4/S0T2O2C2W2已真实采样，reserved520不算新8条完成。update1/2仅归档resume.pt、adapter/evidence保留且不可精确恢复，initial/update3/update4完整。当前安全/效用提升仍未验证；无训练修改/重启，下一次05:00。

11月18日03:00巡检于03:06完成：更新3/effective3、第四步policy3/S0T2O1C1W0已核验提交48/128，累计432有效评分/提交，reserved440。第三步81.54分钟；裁剪前grad norm2913.166出现尖峰，真实AdamW状态重建裁剪后范数1.00000002、adapter/moments有限且496张量变化，当前根因未证实，诊断monitor/diagnostics/gradient_spike_step_0002.json必须下轮继续关注，不称已修复Bug。334runtime/432checker/870段/864清理/完整checkpoint及8活源容器通过；update1按冻结retention仅归档resume.pt，adapter/evidence保留但不可精确恢复，initial/update2/update3完整。无活训练源码修改/重启，下一次04:00。计数和诊断见小时status/report。

11月18日02:00巡检于02:04完成：当前仍2/effective2，第三步policy2/S0T1O1C1W1已提交96/128，累计352提交、359条有效源评分（7条inflight已评分未提交），reserved360。334runtime/359checker/747段/718清理/完整checkpoint验收；实际AgentDoG共享rootless容器按512MiB/32PID核查，不套用ClawGym1GiB/64PID规格。当前64条ClawGym仍Utility0、已评分39条AgentDoG Utility21/39/注册风险0/39是训练子集，不是独立ASR或提升证据。无训练修复/重启，下一次03:00；准确计数见小时status/report。

11月18日01:00巡检于01:05完成：第二次真实更新2/effective2通过，loss0.08779256、grad norm4.83455276，496张量变化/完整checkpoint及AdamW step2/RNG验收；第二步102.77分钟。当前第三步S0T1O1C1W1、policy2已提交8/128，累计264有效评分/提交，reserved272；334runtime/264checker/583段/528清理及8活源容器通过。step1 ClawGym64条Utility0，AgentDoG注册风险目标6/56；这不是独立ASR，参数/配置同时变化，未证明提升。无训练修复/重启，下一次02:00，最新计数见小时status/report。

11月18日00:00巡检于00:07完成：当前更新1/effective1，第2步K2配置S0T1O0C0W2已提交56/128，累计184条有效源评分/提交，reserved192；代码/334runtime/184份checker/376段/368份清理、完整checkpoint与2个活隔离容器通过。当前56条ClawGym均Utility0（53模型协议失败、3工具步数上限），属于真实模型表现，不能声称提升或改门禁。无训练修复/重启，下一次01:00；实际计数看小时status/report。

23:00巡检已于23:15完成：首次真实更新1/effective1，128/128有效源评分并提交，loss=-0.42475335、grad norm=0.14444642、adapter实际变化且完整checkpoint/optimizer/RNG验收。首步采样144.36分钟＋反传9.06分钟＋保存12.26秒，总153.65分钟。当前同16任务K2第二配置S0T1O0C0W2、policy1在GPU1继续，reserved136不等于新8条完成；无训练修复/重启，下一次11月18日00:00。实际最新计数及证据读小时status/report，不能据首次更新宣称安全提升。

22:00巡检已于22:06完成：GPU1当前actor正常，56reserved/54条有效评分/48条提交、updates0，较上轮提交+32。334runtime文件、54份真实checker、87段logprob、108份清理及2个活隔离容器均通过；无训练修复/重启，下一次23:00。详细证据与一次性报告字段更正见小时报告。

首轮真实巡检已于21:17完成，status=finished_training_continues：24reserved、21条有效评分、16条jointjournal提交，updates/effective均0；代码/READY/lease/334runtime文件、检查点及21份真实checker评分、42份源清理和3个活隔离容器通过核对。无训练Bug、无OOM/CUDA错误，保持当前actor，不重启；下一次22:00。证据/报告见监控目录audits及hourly_report.md，静态AST边界source_execution_performed=False的审计解释已按源代码纠正，原误报保留、训练门禁未改。

本条覆盖历史“未恢复cron/禁止恢复”的描述，仅针对用户刚明确请求的Harness Scaling RL小时巡检；旧FlashRSI/30分钟/双卡任务不恢复。crontab唯一新增标记`HSRL_HOURLY_CODEX_LOOP_20261117`，每小时整点(CST/AsiaShanghai)调用`scripts/watch_hsrl_hourly_codex.sh`并经codex queue唤醒当前会话01a1aa6a-0266-7d60-9abf-997271f68415；下一次22:00。调度依赖本机cron和Codex daemon在线，回执仅代表入队，实际完成看报告。

每次先读`experiments/cross_harness_sft/docs/hsrl_hourly_loop_20261117.md`并按`outputs/rl/hsrl_hourly_monitor_20261117/current_run.json`跟踪实际训练。当前仍GPU1上v8 retry1，正常继续训练；异常保留证据/定位修复/独占租约及checkpoint/inflight安全恢复，不能重复actor、重放未知inflight、删难例/改奖励或预算、混协议、泄露oracle或碰他人进程。每轮写`hourly_report.md`和`status.json`。同一会话串行，多小时消息合并当前维修；有其他活维修器持锁时不并行改代码/重复运行。

真实预算完成须completion_audit后停止本轮唤醒，保留证据；不能因只有进程/attempted/预算临近而停。disabled和锁竞争路径已实际通过；真实首次codex queue投递返回0并已入队（入队不等于完成巡检），详见监控目录schedule.json及last_queued.json。训练代码和实际R1未因设置调度改变。

## 2026-11-17 20:50：性能升级已真实验收首批8条，GPU1当前R1 fast v8 retry1

最新资源只读检查（2026-11-17T20:51:51.232212+08:00）：GPU3已自然释放（无计算进程、约19MiB）；GPU0/2仍有其他用户进程。GPU1当前仅有本实验actor2860668。GPU3未启动任何新任务，本轮训练仍仅物理GPU1。

当前唯一RL actor PID2860668、控制器2860662，物理GPU1；正式目录`experiments/cross_harness_sft/outputs/rl/hsrl4b_harness_scaling_grpo_s_r1_fast_v8_retry1_20261117`。worktree入口`scripts/train_hsrl_v8.py`，隔离causal-conv1d1.7.0+已有FLAcore0.5.2，8个独立源环境、模型batch最多8，按32768总padded-token预算拆分；每条16K上下文/2048call/8192episode限制、G8/K2、16组128rollout、50:50、reward/seed267/300update38400attempt均未改。原14K READY和v3/v5冻结source、旧所有训练器/FlashRSI源码不变。新执行代码b9fcdd8/3fa502e/c9c8a12/30ac25c/c75c58c/a79c021经过分阶段检查和独立审查，CPU测试不是正式RL结果。

**实际首wave已通过**：8/8真实rollout源评分valid/done，全部动作token和old/reference logprob数量对应、有限，证据fsync并jointjournal提交；实际batch[8,2]、activeworlds峰8，wave372.3265秒、16898动作token＝45.38token/s，平均46.54秒/条。旧串行首两条7.98token/s，观察比5.69x；同一任务但随机轨迹/长度不同，这只是首wave观测，不是全池或完整step稳定结论。实际GPU1采样121点均33.1%峰100%，显存峰17077MiB，未OOM。已进入下一组，updates仍0；梯度/参数更新/生产resume和安全提升尚未实测。原始数值/语义/全部episode hashes见`outputs/rl/hsrl_fastdeps_20261117/v8_first_real_wave_verification.json`，后续有效进度必须读当前state/journal，不能用reserved attempts当completed。

此前串行6条与v7双路4条均在无inflight/源进程的已提交边界精确停止，updates0，完整证据分别留原目录，未混入v8。第一次v8在模型加载前被新出现的ZhouZixiang外部GPU1进程拦截，只有run.lock、无模型/轨迹；失败日志保留，不放松独占门禁。20:35按用户新指示检查全部卡：GPU0/2/3他人任务100%，GPU1自然释放，无他人进程后恢复retry1，未停止他人进程、未使用其他GPU。GPU1后续再有资源变化应保留错误并定位，不能绕过门禁或重复运行已有inflight。

当前权威状态为`outputs/rl/hsrl_fastdeps_20261117/{fast_R1_v8_retry1_state.json,train_fast_R1_v8_retry1.log,fast_R1_v8_code_approval.json,gpu1_fast_r1_v8_lease.json}`与当前运行state.json。旧v7/首次v8已退出/失败，勿恢复。有限GPU只读采样文件`v8_retry1_gpu1_samples.jsonl`仅针对此次真实负载，不是cron/Codex唤醒；没有恢复任何定时任务。基线全部任务和结果继续按用户优先RL授权保留暂停，不可自动抢GPU1。图编译仅做只读源码评估，未启用存在Qwen混合cache兼容缺口的static-cache开关；当前仍有HF逐token CPU调度/波内尾部等待，未声称持续满载。

## 2026-11-17 20:15：用户要求性能修复，GPU1已启动真正双环境/batch2的R1 v7

当前唯一RL训练PID2837177、控制器2837173，运行目录`experiments/cross_harness_sft/outputs/rl/hsrl4b_harness_scaling_grpo_s_r1_fast_v7_20261117`。实际worktree入口`scripts/train_hsrl_v7.py`，CUDA_VISIBLE_DEVICES=1、episode-workers2、model-batch-size2、max-batch-tokens32768、objective=safety、seed267、38400attempt/300update上限。G8/K2/16组128rollout与50%通用/50%安全、任务池/奖励/上下文上限不变。两条真实环境已同时进入第一wave，尚待首wave评分/真实batch/吞吐落盘；不能把attempts2等于2条有效完成。

原串行R1 PID2759582/controller2759578已在6条有效episode全部journal提交且inflight=None、无子进程的边界暂停后精确停止，6条证据保持独立，updates仍0；不能拼入新kernel/batch协议。完整源池READY和冻结v3/v5包、旧训练器、FlashRSI源码均未改。GPU0为他人任务，GPU2/3不动。GPU1基线仍按用户明确优先RL指示暂停、全部未完成任务/已完成judge/selected证据保留；禁止重复启动旧基线服务/旧R1/cron。

新增外部策略与broker/driver：b9fcdd8/3fa502e/c9c8a12，P1定向34项、P2原88项及I1修复23项通过，独立integration复审无剩余Critical/Important；审查复现并修正“分词请求误算生成就绪导致假batch”的缺陷，纯边界复核实际[2]。真实4B/GPU/source/梯度/恢复仍以实际运行证据为准，不把CPU检查称RL结果。

注意此前“全部回退Torch”不准确：FLAcore0.5.2本就已绑定，实际缺causal-conv1d。官方1.7.0源码按SHA编译、仅安装到`outputs/rl/hsrl_fastdeps_20261117/site-packages`，未改原venv；实际导入/ABI检查4加速函数全部绑定且fast_path_available=True。新策略要求真实层函数也匹配并把package/源码/so身份写runmanifest。生成时capture原始pre-warp logprob，去除old-policy额外全前向，reference仍用冻结原打分逻辑；不保存B*T*V logits。真实提速尚待首wave数据，不能承诺15小时已降至某值。

当前权威状态/真实log/独立审查和升级租约：`outputs/rl/hsrl_fastdeps_20261117/{fast_R1_state.json,train_fast_R1.log,fast_R1_code_approval.json,fast_path_import_identity.json,GPU1_fast_upgrade_boundary_ledger.json}`。如失败保留真实双inflight与全部证据并修复；不得绕过wave/G8/隐私门禁、清零/删例或未知inflight自动重跑。未创建定时唤醒。

## 2026-11-17 19:25：用户已授权优先RL，Codex接回并在物理GPU1派发正式R1

**覆盖此前GPU1始终留基线、GPU0待空的旧交接。**用户明确答复“优先 RL，允许阶段结束后改用 GPU1”。GPU0当前有其他用户任务，不触碰；GPU2/3禁止使用。ACE已结束，v21后续EvoPrompt已在当前case退出且所有线程无host子进程、模型running/waiting均0后暂停并停止自有控制器；58次中间host执行、Snapshot、轨迹、原始30任务和旧全局40算法账本全部保留，EvoPrompt仍为未完成、需后续新目录核对/恢复，不能记完整128或混selected结果。18764自有API/engine/tracker均已退出，未停止他人进程；GPU1现在独占给RL，禁止另起基线服务或恢复旧queue/cron。完整交接账本及真实租约见下方takeover目录。

最终训练数据`experiments/cross_harness_sft/outputs/rl/hsrl_mixed_14k_50x50_v6_20261117`已实际独立发布READY.json，manifest SHA `220521662b8c40127929dbe07ad74a631d2d564a700954d1c1382e0615373c99`。14,000条＝7,000 ClawGym通用＋7,000 AgentDoG安全，完整14K×18=252K上下文检查、8/8真实隔离环境控制及全部文件/证书/普通Dataset加载验收passed。候选与源QA保持不变，不重stage或重做已通过全量审计。

**正式R1已于19:23:09派发**：控制器2759578、训练CLI2759582，worktree`/data/home/liumingxiao/hsrl_v3_worktree_20261116`；输出`outputs/rl/hsrl4b_harness_scaling_grpo_s_r1_20261117`。实际命令为worktree`scripts/train_hsrl_v5.py --physical-gpu 1 --objective safety --seed 267 --max-attempts 38400`，CUDA_VISIBLE_DEVICES=1、官方4B checkpoint、真实GPU1租约，LoRA16/32、50%安全/50%通用、16组×G8、K2、最多300step。38,400尝试并非14K全任务训练一遍。19:32已确认首条真实rollout完成且valid=true，源评分reward=-0.1（Utility=0、protocol_cost=1；第二段达到既定2048单次token上限，工具调用未闭合，按真实模型失败保留，不是基础设施错误），两段采样共2793动作token，old/reference logprob全部有限，已durably提交journal并自动进入第2条。当前updates=0/effective_updates=0，不能据此宣称能力提升；首条任务达成也不因valid而假定。实际GPU1约12.2GiB，无OOM/新错误，模型使用已安装Torch回退路径；真实吞吐与参数更新仍需继续观察。

GPU1入口2dec11a的旧策略构造器仍拒env1问题由独立审查发现，cc471b4外部策略子类修复并复审无剩余Important；原冻结policy/runtime/source不改，初始化除实际GPU guard外AST严格等价，继承原评分/梯度逻辑，原文件hash及新类身份写训练manifest。63项定向检查passed；真实CUDA/轨迹/评分/梯度仍须正式R1现场验证。checkpoint保留7b9cf43：所有adapter/evidence保留，initial/最近两份/明确锚点可完整恢复，只回收验证过的旧optimizer/RNG，不放松60GiB门禁。

最新状态、训练log、进程身份、代码批准、源审计证书、GPU1关闭证据及全部基线待办均在`experiments/cross_harness_sft/outputs/rl/hsrl_codex_takeover_20261117/`。先看`R1_launch_state.json`和`train_R1.log`；失败必须保留现场/真实计数并定位修复，已reserved的inflight不许静默重跑。旧GPU0等待器2694594已在未派发任何模型前停止；没有重复训练，没有恢复任何Codex唤醒或定时任务。

## 2026-11-17 17:46：用户明确要求Codex接回RL并立即启动（覆盖Claude负责GPU0的旧交接）

Codex现在同时负责RL恢复与GPU1基线，Claude仅是此前临时辅助。GPU1现有ACE和v21/2614862保持运行，不修改活FlashRSI源码；GPU2/3及其他用户进程不动。

已停止并保留Claude重复stage 2512351的现场，复用已完整导出的14K候选`experiments/cross_harness_sft/outputs/rl/hsrl_mixed_14k_50x50_v6_20261117_candidate`。R1创建超时的精确自有grader已核对owner/image/mount/id后移除并确认消失；19个R1产生、未列入payload的pyc缓存已移到takeover证据目录保留，未改候选payload或伪造READY。正式最终审计R2 PID2645328已启动，目录`outputs/rl/hsrl_mixed_14k_50x50_v6_20261117_audit_r2`，显式HSRL_WORKSPACE_DOCKER_SOCKET=/run/user/1026/hsrl-v7/docker.sock；AgentDoG JSONWorker仍按其冻结实现使用原共享socket，不能靠DOCKER_HOST冒充全部切换。当前先完整filehash/14K×18上下文/8项真实源环境验收，通过后独立publish READY.json，再直接运行worktree现有train_hsrl_v5.py，objective=safety(GRPO-S)、seed267、38400attempt上限、LoRA16/32。不要再另起stage/审计/训练或改worktree活代码。

训练器与现有依赖只读复核无新增确定性Important，证据`/data/home/liumingxiao/hsrl_v3_worktree_20261116/.superpowers/sdd/2026-11-16-hsrl-v5-ready-dataset/r1-final-launch-readiness-20261117.md`。真实CUDA负载、首组轨迹/logprobs/梯度/adapter update仍须在正式R1观察，不以模型进程存在算update。

GPU0在接回时曾空闲，随后NVML再次报告其他用户ZhouZixiang的2643114跨GPU0/2/3任务（同进程显存读数可能0但实际GPU利用率非0）；保持现有独占门禁、不得杀他人进程。已向用户异步确认若READY时GPU0仍忙，是否允许当前ACE完整结束后临时把GPU1优先用于RL；未得到该授权前不暂停GPU1基线/改用物理1。RL身份/现场在`outputs/rl/hsrl_codex_takeover_20261117/ownership.json`。


## 2026-11-17 17:26：五harness单GPU1有限接续已实际启动（覆盖旧等待器PID）

当前唯一基线模型服务仍是GPU1/18764官方4B/32K，GPU0留RL，GPU2/3及其他用户不动。OpenCode ACE v17 launcher3839693/algorithm3839699持续真实评测；17:23:45最终selected已98/128不同case通过host/原Snapshot/privacy/reset/官方规则核验，0无效，尚无final result或judge。有限judge接续2051639保留，待最终同一次128验收后只补未判。OpenCode v11初始和OPRO均128规则+128judge；H0的128份judge metadata全部nested-tools-v2。旧v4/V1等结果不可混成当前配对。

**新实际接续队列PID2614862**：`experiments/cross_harness_sft/outputs/flashrsi_gpu1_finite_queue_v21_20261117`，真实CLI已写入128 case manifest、五harness、30 queued、deferred为空，目前waiting_endpoint_capacity等待同一ACE；无模型/judge子进程重复派发，source_snapshot尚未冻结。顺序：OpenCode同v11 EvoPrompt/GEPA → OpenClaw新H0+六算法 → Codex新H0+六算法 → Pi新H0+六算法 → Claude Code新H0+六算法。OpenCode SkillOpt/Evo-Harness仍N/A。H0不满足同协议128则保留错误和对应阻塞任务；未知清理状态全lane停派，不能计零/删例/拼结果。

旧等待器2292066已在SIGSTOP后核验精确PID/start/command、零子进程、无source_snapshot/任何stage派发，保留`waiting_queue_superseded_for_v21_native_profiles_20261117.json`后仅停止它；没有打断ACE、judge或模型服务。下方旧v19仍在等待的描述已被本条覆盖。

Pi原生会话、32K元数据、mock/源身份与新共享退出guard已接入；Claude v9的原生UUID/resume/32K、受限环境下可信CLI身份IPC、真实SDK modelUsage别名、固定压缩/重试策略和host死亡/取消清理均已修复。代码7d3de3d与queue/scanner c99388d分别经独立复审无剩余Important；root最新联合267项检查8.74s通过。**这些是接线/控制器验证，不是新native全128结果**；真实CLI/namespace/context/Stop/EOF/JSONL/projection/compaction/counter仍由正式新H0验收，缺证据保持invalid并留原始诊断，不能宣称已修成完整基线。

当前只读状态命令：在实验根执行`PYTHONPATH=scripts:src:vendor/FlashRSI PYTHONDONTWRITEBYTECODE=1 .venv-rl-pilot/bin/python scripts/flashrsi_matrix_repair_scan_v21.py --scan`，不要用旧scanner漏掉nested新队列，也不要自行tick重复启动。实际scan退出0：全局聚合8complete/7needs_repair/25queued/2N/A；当前新队列30queued、all_complete=false。全局优先保留历史有效版本，不能替代当前协议cohort；其他未纳入的矩阵项仍未完成。

真实PID/预检/267验证日志/交接结果与最新覆盖见`experiments/cross_harness_sft/outputs/flashrsi_gpu1_baseline_resume_20261117/controller_handoff_current.json`、`v21_handoff_preflight.json`、`finite_queue_v21_launch.json`、`integration_verification_20261117.log`及hourly_report。未来freeze前后保持所有FlashRSI源码稳定；修复应先核查活队列/锁、保留旧证据并在独立协议目录进行，禁止并行改活代码。未恢复任何cron或Codex定时唤醒。


## 2026-11-17T16:55:21.431596+08:00：v21单卡队列复审通过但尚未启动，Claude四项修复仍是派发门禁

当前仍为GPU1/18764 OpenCode ACE v17（3839693/3839699）真实评测，judge接续2051639、旧等待器v19/2292066保留。v19仍在waiting且尚无source_snapshot；禁止在新代码审查前另起重复任务。未来v21输出拟定`experiments/cross_harness_sft/outputs/flashrsi_gpu1_finite_queue_v21_20261117`，尚未创建/启动。

未来队列cfa44ec→5855389和scanner修复c99388d已经独立审阅通过：五harness全部任务已接入定义，OpenCode用既有v11 H0，剩余EvoPrompt/GEPA；OpenClaw/Codex/Pi/Claude各新H0→六RSI，OpenCode两skill算法N/A。scanner分开显示best historical覆盖和exact-owner当前任务，H0/失败/blocked/queued不会被旧complete遮盖。代码/真实待派发不能当实验完成。

Claude v9 7d384ac有145项定向回归记录，但独立review发现四项Important：受限entry读取/proc取CLI start不可能；raw modelUsage与SDK model_usage字段别名；host死亡未绑定unshare/namespace；候选设置可改变固定压缩/重试策略。正在原作者统一修复，复审前**禁止派发v21或Claude新H0**。报告`experiments/cross_harness_sft/docs/research/flashrsi_claude_native_continuation_review_20261117.md`。共享退出guard的修复范围允许追加到未启动的Pi v6 host与source_pins；已核实这些新Pi进程为0，不能改其他历史/活host或结果。所有模型/评测仅GPU1，其他GPU不动，无cron恢复。

## 2026-11-17 16:15：单GPU1基线继续，Pi复审通过，Claude开始实际实现

用户再次要求一张卡补齐五harness且升算云余额恢复。GPU1/18764当前真实OpenCode ACE v17保持运行（launcher3839693/algorithm3839699），16:09最终selected57/128官方/身份/privacy/reset有效、0无效；不是累计演化计数，最终result/judge尚未出。有限judge接续2051639及后续等待队列v19/2292066保留，无新模型服务、无cron。充值恢复后的OpenCode OPRO和Hermes OPRO judge均128，沿用openai/gpt-5.4-nano，仅补未判。

Pi v6修复b93443a精确匹配官方gateway URL解码分类，57项纯定向检查通过，独立复审`experiments/cross_harness_sft/docs/research/flashrsi_pi_native_continuation_rereview2_20261117.md`无未处理Important；新原生接续及完整128仍未实测。v20未来单卡队列正在接入Pi H0+六算法，尚未替换v19。Claude Code v9现已进入实际代码实施，保留原生SDK会话/真实resume/settled owner与32K元数据；未审查前不得派发。现有OpenClaw/Codex修复和后续任务保留，所有旧partial和失败证据保留，不改写为完成。

最新真实状态/源核验见`experiments/cross_harness_sft/outputs/flashrsi_gpu1_baseline_resume_20261117/controller_handoff_current.json`、`ace_selected_progress_verified.json`和16:12只读scanner_v19。历史矩阵8已完成算法单元/19需修复/13待跑/2 N/A不是同协议全部配对完成。当前任务计划`experiments/cross_harness_sft/docs/superpowers/plans/2026-11-17-flashrsi-single-gpu-completion.md`。GPU0留给RL，其他用户GPU2/3不动。

## 2026-11-17 14:48：OpenClaw上下文及原生压缩审计一并修复，当前等待器为v19

覆盖下方v17/v18等待队列PID：当前真实有限GPU1接续PID2292066，输出`experiments/cross_harness_sft/outputs/flashrsi_gpu1_finite_queue_v19_20261117`。旧等待器1352410/2270254均在无子进程、无source_snapshot、未派发model/judge时停止，逐次根因/状态与任务证据保留；active ACE3839693/3839699、独立judge接续2051639和模型服务未打断，没有重跑已完成初始/缓存judge。最新只读扫描为`scripts/flashrsi_matrix_repair_scan_v19.py --scan`。

OpenClaw原生input预算默认120000亦与18764/32768不符；future-only profile `openclaw-local32k-metadata/v4`设context.max_tokens24576，保持4096回复、12轮、360秒、2400整case envelope及原Snapshot compaction enabled/threshold/tail/prune、SQLite会话、Python/PATH/mime、case环境和沙箱不变。为避免将合法原生压缩误判成历史丢失，新entry只观测并转发原compact函数，新审计保留完整rawSQLite prefix/全部owner/session路径，同时用code-owned原始纯函数重算原生projection并核对实际每次wire及本轮raw append。实际exported compaction/models/providers三文件SHA须匹配冻结受支持版本；未知这三组件源码修改为明确unsupported/invalid兼容性问题，不能计零或假装所有Evo-Harness代码修改已支持。无修改vendor/Snapshot/旧活代码、无执行候选代码到host。native估计不冒充Qwen准确token证明。

代码3e0c006+e65c524/9aa5b26，24个定向检查通过、独立复审`docs/research/flashrsi_openclaw_context_v19_rereview1_20261117.md`无Critical/Important。新queuev19保留Codex v18 32K/24576元数据修复、真实同v11 OpenCode H0绑定及所有128/source/private/规则/judge/lease/cleanup门禁。实际OpenClaw/Codex新H0及其6RSI仍未执行，不宣称修复后已128完成。Claude/Pi仍在表中，Pi新原生session/32K metadata/summary观测修复进行中；无旧partial状态改写。

GPU1当前仍只有OpenCode ACE v17实际使用模型，14:46已有51条演化host exit0、无timeout、最终selected128未开始；这些不是最终coverage。OpenCode旧v11 H0/OPRO及Hermes OPRO judge均128，judge部署全部openai/gpt-5.4-nano，充值后仅补未判。当前实际/历史协议区分见`outputs/flashrsi_gpu1_baseline_resume_20261117/baseline_progress_20261117T1448.json`（相对实验根）。没有恢复任何cron。

## 2026-11-17 14:29：Codex上下文元数据修复后已恢复单GPU1接续队列

覆盖下方14:07中的等待队列PID/路径：旧v17等待器1352410已确认无子进程、无source_snapshot、无任何model/judge派发后，仅停止该等待器；原因是新确认Codex未知模型回退272000上下文/244800压缩阈值，与实际18764服务32768不符。原生parser依赖修复不足以解决它，不原样重跑。停止证据保存在旧v17队列`waiting_queue_superseded_for_codex_context_repair.json`，所有任务和证据保留。active ACE3839693/3839699、其有限judge接续2051639及两个模型服务未中断。

新增future-only Codex profile `codex-local32k-metadata/v18`，包级`codex._settings`临时runtime override设置window32768、total自动压缩24576（保留两份原4096回复空间）；不改Snapshot/vendor、不加context/output/time/turn权限预算、不放松沙箱。原native估计不是准确Qwen token证明，真实压缩/全128仍须新H0验证。新entry-I仅加入固定可信scripts/vendor导入根，逐owner保留实际应用元数据，正常完成但缺失/不符receipt即拒绝；新H0manifest与expectedcontract绑定该元数据。代码43ce89a，21项定向检查通过，独立审阅`docs/research/flashrsi_codex_context_v18_review_20261117.md`无Critical/Important。

新有限队列PID2270254，实验根`outputs/flashrsi_gpu1_finite_queue_v18_20261117`；真实CLI已进入waiting_endpoint_capacity，等待同一ACE前驱，随后OpenCode EvoPrompt/GEPA→修正环境OpenClaw新H0+6RSI→新Codex上下文profile H0+6RSI。H0严格128失败仍留工单，不能计零/丢弃；Claude/Pi实现问题继续明确留表，不宣称五harness完成。只读扫描改用`scripts/flashrsi_matrix_repair_scan_v18.py --scan`，覆盖v17/v18队列。OpenCode SkillOpt/Evo-Harness仍N/A。

ACE独立有限judge接续PID2051639已经启动，根`outputs/flashrsi_gpu1_baseline_resume_20261117/ace_judge_followup_v17`；精确等待PID/startticks，完整同运行128身份/private/规则/selected才调用既有judge缓存；共享worker lease、billing guard、无overwrite。4个定向检查与两轮独立复审通过，保留非零judge子进程错误，即使128缓存写完也不冒充流程成功。它不启动模型或cron，真实judge须等ACE最终128完成。

## 2026-11-17：一张GPU1继续FlashRSI基线（最新用户授权）

用户要求一张卡持续补齐OpenCode/Claude Code/Codex/OpenClaw/Pi初始与RSI安全基线，并明确升算云已充值；GPU1既有18764/32K官方4B服务保留作基线，GPU0留RL。禁止恢复用户已删除cron或触碰GPU2/3、原共享Docker与其他用户任务。

14:07实际状态：OpenCode OPRO judge已128/128；Hermes OPRO14:01完成，复用22已有、补判106，合计128/128。均用既有openai/gpt-5.4-nano升算云wrapper，无overwrite。只读scanner_v17全历史算法矩阵8complete/19needs_repair/13pending/2N/A；这是每个单元的历史已完成版本，不表示同一新版配对cohort或五harness完成。OpenCode初始v11已128规则+128judge，旧v4 EvoPrompt/ACE也128+128；Claude Code50、Codex17、OpenClaw13、Pi18是各自最佳历史H0有效条数，不是完整128结果。

真实活进程OpenCode ACE v17 launcher3839693/算法3839699，根`experiments/cross_harness_sft/outputs/flashrsi_harnessrisk_4b_private_opencode_v17_gpu1_20261117`，14:06已18条演化host成功，最终selected128还未开始；演化计数不能当最终coverage。新增有限GPU1接续队列PID1352410，根`outputs/flashrsi_gpu1_finite_queue_v17_20261117`（相对实验根，下同），等待精确ACE PID/startticks及lane锁，不重复模型任务。之后OpenCode同v11 EvoPrompt/GEPA→已修正Python环境的OpenClaw新H0+6算法→已安装官方声明tree-sitter依赖的Codex新H0+6算法。H0不满严格128则该harness算法保持阻塞并保存反馈，独立其他harness可推进。OpenCode SkillOpt/Evo-Harness仍N/A。

接续代码f818b483，经独立review修复Q-I1清理不确定性（同算法下个case、跨stage和外部前驱timeout均sticky阻断）/Q-I2真实H0源字节及模型/runtime/host/端口/预算契约，c1e8a063/0e7af3d8；19测试通过，复审`docs/research/flashrsi_gpu1_queue_v17_rereview1_20261117.md`无未处理Important。原active v17代码未改，新任务使用guarded benchmark_v18/reference_v19，OpenCode GEPA使用原contextfit v6组合入口v18，不盲增预算。

Claude/Pi全部initial+6算法明确留blocked_protocol_repair，不按旧错误原样重复跑、不把错误计零。正在新文件中实现Pi原生session_file逐owner连续会话及正常预算终止/settled工具/历史审计；Claude SDK真实resume需要保留原生配置/会话状态，原adapter每owner删除它。两者须新协议、新H0和完整同协议RSI，不可追溯改旧50/18条结果为completed。原生方案与审阅在`docs/research/flashrsi_native_bounded_continuation_recon_20261117.md`、`flashrsi_bounded_owner_profile_review_20261117.md`。当前Pi实现审查未完成、未实际部署。独立有限ACE judge接续补丁正在审查，不能假定active launcher会自动judge。

接续任务/源冻结/实际PID/错误分别见新队列`remaining_tasks.json`、`repair_protocol.json`、`pidboard.json`、stage receipts、私有host logs；只读新扫描`scripts/flashrsi_matrix_repair_scan_v17.py --scan`。充值恢复与API日志在`outputs/flashrsi_gpu1_baseline_resume_20261117/`。不把进程、attempted、文件存在当完整验收。

## 2026-11-16：用户已删除每小时Codex唤醒loop（最新调度指令）

已按用户最新明确请求移除唯一`flashrsi-harnessrisk-codex-hourly` cron并复读核验，当前crontab无任务。不要根据下方历史“唯一小时cron保留/继续巡检”记录重新创建任何定时唤醒。正在运行的模型、真实评测及既有算法接续队列继续，不因删除唤醒而停止；这不表示实验已完成，不创建实验完成的disabled标记。移除记录、原cron备份及日志在`experiments/cross_harness_sft/outputs/flashrsi_harnessrisk_4b_private_v1_20261113/monitor/hourly_wake_removal_20261116.json`和hourly_report.md。

## 2026-11-16：用户否决小池，RL v5改为万级与50/50（覆盖下方v4）

用户最新明确要求：主训练任务池与Harness Scaling的约13.5K同量级，安全/通用各半。旧SkillLife30训练实例＋Claw精选2K的75/25方案仅保留为历史机制先导，不能继续当主方案或按旧比例启动。已完成大规模官方RL任务环境静态调研，主安全源选择AgentDoG 1.5 **独立Agentic RL runtime包**，不是其1.0/1.5 guard分类训练集；ClawGym通用源扩到已固定下载的完整15,402条原始任务，再准入，不再把2K官方精选当永久上限。

v5主建设目标14,000条训练记录＝7,000通用＋7,000安全条件。AgentDoG已核实catalog15,774、配对索引去重13,212条件/2,931基础family；按family80/10/10静态train候选clean2,344/attack6,748/query_target1,524。7K数量有候选基础，但不是7K独立基础目标，也不是已通过真实运行验收。已实际列出7,000唯一候选ID、1,420基础family：5,500 attacked-benign＋1,000同源语义clean控制＋500纯恶意用户请求，恶意目标成功不能当正向Utility。源码规则为调用/参数/历史谓词，不冒充实际泄漏/破坏后果；需要私有oracle/工具输出白名单、外层隔离、原始source判据与judge分列，judge异常为invalid，禁止默认安全或改奖励公式。详细决定/证据以`experiments/cross_harness_sft/docs/harness_scaling_rl_training_pool_v5_20261116.md`为准。

固定HF runtime catalog及pairing已下载到`experiments/cross_harness_sft/data/rl_sources/agentdog_runtime_hf_7c26321`并匹配发布SHA256SUMS；7K候选索引和Root复核在`docs/research/agentdog_rl_safety7000_candidate_index_20261116.json`及`agentdog_rl_safety7000_root_verification_20261116.json`（相对实验根）。候选hash69c289ffabf39c8ea0c88754743b15658f84dfbb0faf0090f81843e112dd75ea，不是运行准入hash。

14K若全部K2×G8一遍需224,000训练episode、1,750个128-rollout step；旧2,560尝试只覆盖最多160个task-condition位置，不能称14K全量训练。当前仍是新来源下载/元数据和源码审计，未启动v5 RL、未切卡、未改当前FlashRSI服务/队列/cron；其他历史权重和检查证据继续保留。

## 2026-11-16：用户授权独立重新选型，RL v4采用SkillLifeBench lifecycle＋ClawGym（覆盖下方v3固定来源）

用户明确HarnessSafe/HarnessAudit只是参考，并授权web检索、多agent投票、负责人拍板。三个独立审阅首选一致，首轮安全源改为SkillLifeBench **autonomous lifecycle**，固定`57bc115a56baa483d4f68824d5a9c78f00a265dd`。只取60实例（12模板×5），不把388源场景或89 runtime任务当自主训练池；B/D/E各4模板，按seed20261116哈希排序每类2train/1dev/1test，合计6/3/3模板、30/15/15实例、60/30/30 clean/attack条件。逐条名单在`experiments/cross_harness_sft/docs/research/skilllife_lifecycle_split_20261116.json`，不是运行验收证明。

**HF直接来源已核对：**用户更正为`RUC-AIBOX/ClawGym-Task`，固定HF revision `bd9043cff88b913922a165d9c022a4796aee3f3a`，原始JSONL实际15,402条（非README概述13.5K）。已下载并按LFS hash校验。官方精选2,000 prompt全部一一对应，8,647现有文件完全同字节，25个原Git漏收输入（23个日志、1个任务脚本、1个任务fixture配置）从HF恢复到独立`experiments/cross_harness_sft/data/rl_sources/clawgym_selected2k_hf_bd9043c/tasks`。旧vendor不改；后续新pool须重做HF字节绑定，不能沿用旧缺输入pool身份。核对证据`docs/research/clawgym_hf_source_binding_20261116.json`（相对实验根）。下载/静态映射不是在线RL更新。

通用池保留ClawGym官方精选2,000训练任务；网络或宿主能力不支持的题按源契约明确排除并保留原因，合格数不是固定2,000。官方200题Bench仍只评测。主配方通用75%/安全25%，安全内部clean/attack各半；16组×G8每步96/16/16条，K2在相继step用当时策略对同任务重新reset采样。安全只用一个源不是权重遗漏；HarnessSafe/Audit旧代码和证据保留、默认训练权重0。以[最新单卡设计](experiments/cross_harness_sft/docs/4b_harness_scaling_rl_single_gpu_design_20261116.md)及[选型决定](experiments/cross_harness_sft/docs/harness_scaling_rl_security_selection_20261116.md)为准。

用户已批准搭建并启动真实在线RL，不再要求一次方案批准。隔离worktree`/data/home/liumingxiao/hsrl_v3_worktree_20261116`中新建版本化训练包，避免改动现有FlashRSI活代码。源数据适配审查中发现输入写权限、网络准入、执行错误、源字节及split约束问题，修复与失败证据分别保留；静态检查不等于sandbox/reset/真实参数更新成功。当前尚未产生新RL更新。GPU0仍在Hermes现有完整阶段，GPU1继续OpenCode；交接必须在完整算法边界，登记全部等待器，禁止中途抢卡或恢复cron。

## 2026-11-16：RL设计v3已明确三数据源、固定版本和配比（覆盖旧“通用任务待定”）

最新设计在`experiments/cross_harness_sft/docs/4b_harness_scaling_rl_single_gpu_design_20261116.md`，研究依据在`docs/harness_scaling_rl_dataset_research_20261116.md`，只读元数据盘点在`docs/research/harness_scaling_rl_dataset_inventory_20261116.json`（后两路径相对实验根）。首轮只用：①HarnessSafe本地328 case的研究者train；②HarnessAudit single_agent本地34任务的研究者train；③ClawGym/ClawGym-Agents commit 0e8526d8f5e0999896c74cee4d6b998fb8a0429e下RL/data/clawgym_train的官方精选2,000任务，内部留dev后train。不能把完整HF HarnessAudit210多角色任务当34单agent，也不能把ClawGym教师轨迹或13.5K全量宣称已接入。

主配方安全25%/通用75%，全局ClawGym75%、HarnessSafe12.5%、HarnessAudit12.5%；每步16组×8为96/16/16条，2,560尝试无额外失败时为1,920/320/320。首轮另做ρ=0%与50%，完整扫描再10%/100%；原始文件拼接不能代替配额。安全源按family70/15/15，ClawGym仅在官方2,000中按模板簇90/10留dev，seed20261116；实际合格数尚待审计，不伪造目标数。ClawGym-Bench200虽HF split叫train，仍完全封存；HarnessRisk128作已暴露回归，训练权重0；SafeClawArena/τ²/BFCL/SkillsBench/Agent-World/AgentDojo首轮训练权重0。

三源全部通过一个五维白盒执行器，复用任务/环境/checker，不调用上游OpenClaw黑盒RL。ClawGym上游reward/check.py可见目录布局不能照搬：改独立target不可读grader与只读输出快照，checker异常为invalid。各源需真实reset、隐私、评分、C1/C2与参数更新验收，某源不合格不得静默改权重替代。当前只做调研/文档和本地文件盘点，未训练/切卡/修改现有服务/定时任务。

## 2026-11-16：RL设计v2按用户反馈改为多源任务池与单一白盒五维训练（后续以v3为准）

以`experiments/cross_harness_sft/docs/4b_harness_scaling_rl_single_gpu_design_20261116.md`的v2为最新草案，并同步总路线/执行协议。移除AgentDojo优先训练路线：从HarnessSafe/HarnessAudit等抽取任务、状态、工具后端与源checker，先family/模板跨源去重及split，再进入**同一个S/T/O/C/W白盒执行器**，每维三版本、固定18行、K=2相继step且配置内G=8。不是在Codex/OpenCode/Hermes上轮流RL；它们承担现有FlashRSI基线和冻结资源下的参数迁移评测。不得把来源永久绑定某个配置，或把原生CLI套wrapper称白盒。

源oracle与训练reward分开：HarnessSafe保留多阶段载体/恢复机制及N0–N5b/CSS，需独立Utility checker；HarnessAudit保留tool/resource/flow SAR、AVS/TCR，不能叫ASR。混合安全reward使用版本化c_source，只有适用且完整证据才可判分；持久化事件去重。先导25%安全占比为12通用组+2 HarnessSafe组+2 HarnessAudit组，每组8；clean/attack只在真实可比配对stratum内各半，原生boundary不强造配对。替代下方旧的全安全池对半假设，保留训练attempt预算及对照。

单卡LoRA、C1/C2和两源白盒适配仍是实施放行条件；本次只改文档，未启动RL、未切卡、未改现有服务/评测/定时任务。两主来源尚未接通不能退回AgentDojo后称此方案已执行；源码存在与模板seed复制都不证明合格任务规模或完整五维训练。

## 2026-11-16：一张卡保留基线、一张卡做4B RL的设计草案（后续以v2为准）

用户要求先审阅单卡RL设计。已新增`experiments/cross_harness_sft/docs/4b_harness_scaling_rl_single_gpu_design_20261116.md`并在总路线Phase 2索引：建议GPU1保留FlashRSI × HarnessRisk基线，GPU0在当前Hermes完整控制阶段/评分结束、等待器安全迁移后转为4B LoRA-GRPO；首批25%安全数据的多配置GRPO-S、指定单配置GRPO-S、仅效用GRPO-U，后接0%/50%配比，每臂2,560次训练episode尝试。FlashRSI固定权重，RL前后评测冻结同一Snapshot；白盒与原生资源开发面板分别配对，HarnessRisk128仅已暴露全量回归，不用于选模。LoRA/Qwen3.5单卡链路、C1/C2和新mixture入口仍需真实验收。当前只是文档，未停止队列/服务、未切卡、未启动RL；不因草案自行抢占或恢复定时任务。

## 2026-11-16：全局阶段二设计已改为普通Harness Scaling RL与数据配比优先

用户最新研究要求已写入`experiments/cross_harness_sft/docs/harness_scaling_safe_rsi_technical_roadmap_20261113.md`的Phase 2（2.1–2.7），并同步`safe_rsi_research_protocol_20261113.md`的11/16修订。4B/9B各自官方Instruct→各harness×RSI算法独立外层资源谱系（计划r=0/1/2/3，内部step不当外层轮）→固定五维18组合/K=2在线RL；先验证GRPO-U/S是否改善安全，再在同数据比例/证据/分组/预算下比较DAPO/Dr.GRPO/CHS-PO。FlashRSI目前固定权重演化Snapshot，不是已更新了4B/9B权重；当前受限单次运行不能改名为已完成3轮或9B结果。

重点变量为安全比例ρ=0/10/25/50/100%，独立通用任务占1−ρ；只有安全源内部clean/attack各半。25%为首个候选而非已知最优，先4B开发扫描，再冻结代表比例用于4B/9B确认及3seed。记录task-condition、实际episode、过滤后覆盖与有效action-token比例，避免长安全轨迹实际主导梯度。通用能力目标不下降，主能力/clean Utility逐项非劣门槛不宽于5个百分点；在同一冻结Hᵣ上比较RL前后，不能用更弱H0掩盖能力损失。所有扫描臂含退化结果都报告，不在test选比例。

按原文核对K=2为同任务相邻配置相继step、各自新策略rollout；普通GRPO配置内每任务G=8。跨配置4+4是独立分组因素，不能当原文默认或与只改安全约束混淆。共享白盒底座与FlashRSI资源评测分开，资源引入训练须组件映射验收。当前旧live训练仍C0；C1/C2 segment/真实logprob/actor更新同步未完全接通，旧9B脚本/AgentDojo pool也未实现混合采样。本次只改实验文档，不启动训练、不动现有评测、不恢复已删除的定时loop。后续新实验依修订执行，历史协议和结果不追溯改名。

## 2026-11-16 10:49：用户充值后judge恢复成功；两harness全部适用算法已落实串行补齐队列

用户明确告知升算云余额补足。已先备份共享guard/旧manifest/52失败result，原子将retry_after置0，只恢复Hermes v12旧初始未判52（不加--overwrite）。10:45:45 CLI退出0：76 existing+52 judged，完整128/128；旧76成功输入/结果152份hash全保持。128个case身份、V2证据协议、按canonical judge规定的规范化JSON证据hash、4项指标有限且在[0,1]全部通过。Judge ASR48.4375%、Utility68.75%、Persistence28.90625%、Detection3.125%。这是v12旧初始完整judge，不是v14 GEPA或新控制。第一次辅助核验误以文件字节hash比较规范化JSONhash，按可信源hash契约重验后128全过；没有改证据或重判。备份在v12根judge_billing_recovery_attempts/20261116T104050_user_recharged，日志hermes_initial_judge_recovery_20261116T104050_user_recharged.log。judge_billing_ticket已resolved/remaining空；guard状态resolved_after_user_recharge、retry_after0，reason里的402是历史，不得误报仍余额不足/重新暂停；新402才按原门禁重新冷却。

**剩余补齐范围以monitor/matrix_loop/two_harness_completion_plan_20261116.json为准**。当前Hermes GEPA v14与OpenCode v4 EvoPrompt保持原进程，不重启、不抢占；它们满128后原父队列会自动judge及后续阶段。OpenCode旧V1初始128不重复判分，证据质量限制不因充值消失，旧v4结果保留独立协议。

已启动两个新的真实等待任务（不是模拟/冒烟、不是已完成实验）：
- GPU0 tmux `flashrsi-hermes-remaining-v14-gpu0`，launcher643580，根`outputs/flashrsi_harnessrisk_4b_private_hermes_v14_remaining_20261116`。等待当前v14父138726完成GEPA及真实同v14初始control128后，依次完整OPRO→EvoPrompt→ACE→SkillOpt→Evo-Harness；只读引用该真实控制，不再推理一份初始。与当前GEPA合计6种算法，旧Hermes EvoPrompt旧工具环境协议不能代替新版配对。
- GPU1 tmux `flashrsi-opencode-remaining-v11-gpu1`，launcher643584，根`outputs/flashrsi_harnessrisk_4b_private_opencode_v11_remaining_20261116`。等原v4父3925329及v5 OPRO197030、v6 GEPA419477自然结束后，依次完整同v11 EvoPrompt→ACE；只读引用v5实际完整初始128，不重复初始推理。与已有v5 OPRO、v6 GEPA合计4种适用算法，SkillOpt/Evo-Harness仍N/A。当前v4 EvoPrompt/ACE不打断，结果独立保留；v11是接线修复后的同协议配对，不能与v4混分数。

两个等待任务都用独立`scripts/flashrsi_matrix_remaining_lane_v7_20261116.py --harness hermes|opencode`，无新模型服务或其他GPU。每算法独立run，严格演化拒绝不可评分，selected完整128身份/private/规则/judge门禁不变；失败留在remaining_tasks.json并继续下一独立算法，下一小时须继续维修失败项，不把它们遗忘。新reference-only入口不改官方算法配置/目标host/预算/权限；冻结源hash/case128/真实依赖已核验，两个新根目标评测目录均0，正在等待，未来模型生成/全量效果仍未验证。10项定向检查通过；真实生产CLI等待路径、源hash和不重复推理已验证。实现计划docs/superpowers/plans/2026-11-16-complete-two-rsi-lanes.md。

**后续扫描用`scripts/flashrsi_matrix_repair_scan_v16.py --scan`**：只新增reference-v7进程和多算法真实等待队列登记，保留v15入口，完整验收函数不改。当前全局历史矩阵3 complete/15 needs_repair/7 queued/2 running/13 pending/2 N/A；历史Hermes EvoPrompt complete不代表新版v14已完成，严格两harness新协议范围需按completion_plan逐项验收，不能因机械历史complete而关闭巡检。其他harness也未完成，不冒充全矩阵完成。证据：judge_credit_resume_result_20261116.json、hermes_v12_initial_judge_full_verification_20261116.json、remaining_lane_v7_verification_20261116.json、two_harness_completion_plan_20261116.json及四个等待器/活根source_snapshot、launch_state、remaining_tasks。无残留judge/维修锁，唯一小时cron继续，无disabled。未动其他用户GPU、未修改当前活代码。后续成功judge不重复API，新轨迹只有完整门禁通过才judge。

## 2026-11-16 10:04：Hermes与OpenCode EvoPrompt均在selected全128阶段；EvoPrompt选回初始

10:02逐例核验GPU0 Hermes v14 GEPA selected601577bf…最终阶段74/128有效、0失败（1例进行中），已完成记录均精确Snapshot身份/private/mock/官方规则/数值有限；总过程296 completed包含222条演化记录，10088请求重过oracle，尚无result/judge。继续原138740/父138726；全128模型阶段结束后父队列再真实跑同v14初始control128，不借旧v12分数配对。当前选出Snapshot与初始不同，但最终结果/新控制/judge都不完整，不能报告安全提升。

GPU1 OpenCode v4 EvoPrompt（346668/父3925329）已于09:06:48选出Snapshot e5a29fd7…，**与其初始Snapshot ID完全相同**；最终全128阶段10:02核验50/128有效、0失败（1例进行中），不是新的提示优化获得提升。总过程170 completed包含前120条人口/子代dev评测；4394请求oracle通过，最终phase按source写序/时间/sequence/精确ID/官方case前缀单独核验，不能把同ID之前dev30也算最终覆盖。无result/judge；ACE仍在其后未启动。旧OPRO/GEPA错误保留，197030(v5 OPRO)→419477(v6 GEPA)两个真实等待器依赖正常、源hash匹配、评测目录0，不重复启动或误杀等待器；两个skill算法N/A。旧初始V1 judge128仍有证据质量限制，不冒充完整可信新V2对照。

10:01共享额度冷却已过且无其他judge，持worker.lock仅恢复Hermes v12旧初始未判52一次，无--overwrite。仍HTTP402，CLI退出1：76 existing/1 error/51 blocked_billing，新增失败API请求1，原成功76输入/结果152份SHA全不变。备份v12根judge_billing_recovery_attempts/20261116T10，日志hermes_initial_judge_recovery_20261116T10.log；共享guard到11:01:08.9。到期前不调用、不清guard、不切账号/model或充值、不把缺判记0、不重判成功76和旧OpenCode128。当前两个selected正式run的judge都还未开始，不能将旧v12的76套在v14上。恢复已退出并释放锁，worker.json仍历史finished，无另一个维修器。

本轮无新评测故障、无可执行代码/配置/隔离/服务修改，模型端口18761/18764健康、活根及等待根源码hash匹配。监控辅助步骤曾在审计产物尚未写完时提前读取引发FileNotFoundError，已等待审计结束后成功补读，证据monitor_read_order_recovery_20261116T10.json；不影响实验，不算模型错误或覆盖失败。继续`scripts/flashrsi_matrix_repair_scan_v15.py --scan`：3 complete/20 needs_repair/2 queued/2 running/13 pending/2 N/A，35个非完成/非运行工单保留，无新增complete；唯一小时cron保持、无disabled、其他用户GPU未动。证据：monitor/matrix_loop/hourly_evidence_20261116T10.json、hermes_gepa_full_selected_progress_20261116T10.json、opencode_evoprompt_full_selected_progress_20261116T10.json、judge_recovery_20261116T10.json。下轮重点全128是否出现新失败、result/selected_cases身份验收及Hermes控制/OpenCode ACE实际接续，再按guard补judge；新错误须保存后串行修复。

## 2026-11-16 09:04：Hermes已完成演化，当前为最终Snapshot全128评测；OpenCode仍EvoPrompt

GPU0 Hermes v14 GEPA（138740/父138726）于08:36:59保存最终selected Snapshot `601577bf7be5c940c08d2caf6cf8a5e427387d7c9486e69f573b4e75646edc58`，25批决策完成（22 reject/3 admit），另有1条promote决策，不是26个训练批。09:02在同运行seq223起的selected全128阶段核验21/128不同case有效，Snapshot身份/private/mock/官方规则均通过、无错误，result.json尚无、judge未开始。阶段由冻结入口先save selected再全128的写序、文件时间、连续evidence序号、精确selected ID及官方case有序前缀交叉确认；这是进行中覆盖，最终验收仍以result/selected_cases同集128为准。总过程243 completed（含之前222条演化记录），不能拿243当最终覆盖或混算指标。8404请求重过oracle。GEPA完整模型阶段之后父队列才运行同v14初始control128，不能用旧v12分数替代配对。

GPU1 OpenCode v4 EvoPrompt（346668/父3925329）09:02为115条过程completed、0错误；身份/private/mock/官方规则通过，2945请求oracle审查，无selected/result/judge，ACE仍在其后尚未启动。v4 OPRO143 completed+1 skill_015 error、GEPA129目标completed但Worker length失败均保留，无selected/result。独立v5 OPRO修复197030、v6 GEPA修复419477都存活并正常有序waiting，冻结hash匹配、各评测目录0；不重复启动或因等待长误杀。v5等原v4父退出后同v11初始128→OPRO；v6再等v5并引用其真实完整初始，随后GEPA→selected128；新Worker实际生成/全量效果仍未验证。OpenCode SkillOpt/Evo-Harness N/A，旧初始规则128及旧V1 judge证据质量限制继续保留。

本轮无新故障、未改可执行代码或启动/重启推理。09:02共享额度guard仍冷却到09:05:41.7，未主动清冷却或发送API，Hermes v12旧初始judge仍76 existing/1 error/51 blocked_billing；已有76成功输入/结果152份SHA均未变，剩余52不是0分。后续到期且无活judge才仅补未完成项，成功76与旧OpenCode128不重判。worker.json历史finished，worker.lock已确认空闲，无另一个维修器。

继续扫描`scripts/flashrsi_matrix_repair_scan_v15.py --scan`，当前3 complete/20 needs_repair/2 queued/2 running/13 pending/2 N/A，35个非完成/非运行工单保留，无本轮新complete，无disabled。两卡18761/18764服务模型正确、活根及两个等待根源hash匹配，唯一小时cron保留，其他用户GPU未动。证据：monitor/matrix_loop/hourly_evidence_20261116T09.json、hermes_gepa_full_selected_progress_20261116T09.json（逐例21条）、hourly_closing_checks_20261116T09.json。下轮重点核验Hermes同selected完整128的后续错误/结果及control实际衔接；OpenCode EvoPrompt是否保存selected并进入全128或失败后ACE是否启动；两个修复任务继续按依赖推进，judge按guard处理。禁止因已选出Snapshot而直接标算法完成。

## 2026-11-16 08:07：GPU1已转EvoPrompt；GEPA Worker长度故障已修复并登记独立串行重跑

当前GPU0 Hermes v14 GEPA（138740/父138726）08:06为193条有效过程completed、0错误，19/25批决策（3 admit/16 reject），约88个不同case/多Snapshot；6655请求重过oracle，身份/private/mock/官方规则通过，无selected/result/judge。GEPA后仍需真实同v14初始128，不能混旧v12分数。GPU1当前OpenCode v4 EvoPrompt（346668/父3925329）已于07:04:28自动实际衔接，08:06为61条有效过程completed、0错误，1607请求重过oracle；无selected/result/judge，之后ACE尚未启动。过程数均非最终128，勿据进程/attempted宣布完成。

OpenCode v4 GEPA07:04:28退出1：129条目标轨迹均completed且身份/private/mock/官方评分通过，失败在trajectories/00029-worker.json，输入9707/输出1024，error=LLM Worker did not finish:length。输出为冗长案例操作手册而非完整通用候选；无selected/result/judge。原错误/Worker完整输入输出/候选及129轨迹全部保留。v4 OPRO143 completed+1 skill_015 error仍保留工单，其v5独立修复197030继续正常等待原GPU1父队列结束。

**新GEPA维修任务**：独立`scripts/flashrsi_opencode_gepa_contextfit_v6.py`及algorithm_opencode_gepa_v6.py，沿用Hermes已验证的bounded reflection思想，但明确标注OpenCode Worker算子适配：紧凑历史数据、保留全部3条metrics/feedback和原轨迹hash，父指令完整保留；要求一个完整通用fenced指令≤300words，Worker仍1024，不增加target32K/4096/12轮、重试或权限。8项定向检查通过；保存的实际失败parent经上游真实Trajectory投影→2295 tokenizer tokens（原9707），3条反馈/指标不变、4862char合法JSON、原始证据未改，0新模型生成。目标模型运行效果与完整候选产出尚待正式全量验证，不声称修复已验收或baseline提升。

新tmux `flashrsi-opencode-gepa-contextfit-v6-gpu1`，launcher419477，脚本`scripts/flashrsi_matrix_opencode_gepa_contextfit_v6_20261116.py`，根`outputs/flashrsi_harnessrisk_4b_private_opencode_gepa_contextfit_v6_20261116`。当前真实waiting_endpoint_capacity，等待原GPU1父3925329及v5修复197030自然退出，不抢占EvoPrompt/ACE，不启动重复推理。之后只读引用v5真正完成的opencode_initial_full作为同v11 target/runtime起点，先128身份/private/mock/官方规则验证，再完整GEPA→selected128→gated judge；不复制旧评分或再跑一份相同初始。不完整或缺失的真实reference不得排序，失败证据保留并留工单。v6只变Worker算子，target协议与v5相同；原v4及新v5/v6分别报告。v6 source_snapshot/case_manifest128/repair_protocol已冻结hash匹配，评测目录0，尚未实际生成模型候选。OpenCode两个skill算法仍N/A，旧初始V1 judge128及证据质量限制不变。

**后续扫描使用`scripts/flashrsi_matrix_repair_scan_v15.py --scan`**，新增独立OpenCode GEPA入口注册，保留v14 Hermes登记，原完整128验收函数未改；原scanner及活队列源均未改。最新3 complete/20 needs_repair/2 queued/2 running/13 pending/2 N/A，35个非完成/非运行工单持续保留，无本轮新增complete。两个等待器依赖有序：v5只等原v4，v6再等v5，不存在互等；均验证为真实等待而非卡死。

08:05在串行锁内仅恢复Hermes v12未判52一次，无--overwrite，仍HTTP402；76 existing/1 error/51 blocked_billing，新增失败API调用1，成功76的152输入/结果SHA全不变。备份v12根judge_billing_recovery_attempts/20261116T08，日志hermes_initial_judge_recovery_20261116T08.log。共享guard到09:05:41.7；之前不调用、不重判成功、缺判不记0，不切账号/model或充值。维修lease已在恢复结束释放，无另一个维修器或残留judge；两卡服务与活代码/冻结hash保持，其他用户GPU未动，唯一小时cron保留，无disabled。证据：monitor/matrix_loop/hourly_evidence_20261116T08.json、opencode_gepa_worker_length_diagnosis_20261116T08.json、opencode_gepa_contextfit_v6_verification_20261116T08.json、judge_recovery_20261116T08.json及各新根queue_events/launch_state/source_snapshot。下轮继续查Hermes GEPA最终选取/control、OpenCode EvoPrompt/ACE及v5→v6实际接续，不能把待运行修复说成已完成算法。

## 2026-11-16 07:04：两卡仍在GEPA演化，超过128条过程记录不等于最终全量

07:01实查GPU0 Hermes v14（138740/父138726）138条过程completed、0错误，全部身份/private/mock/官方规则通过，4749请求重过oracle审查；已经13条迭代决策（iteration0–12，2 admit/11 reject），当前约70个不同case出现在多Snapshot过程记录。GPU1 OpenCode v4 GEPA（159357/父3925329）126条过程completed、0错误，全部身份/private/mock/官方规则通过，3608请求重过oracle；7条迭代决策（iteration0–6，2 admit/5 reject），多Snapshot过程约52个不同case。两者无selected/result/judge，不能启动selected judge或声称最终128覆盖，更不能对过程记录混合计算ASR/Utility。官方GEPA设置train72/dev30/test26、1epoch/反射batch3，上游会额外padding完整3个task，因此计划25批（非128案例跑完即结束）；完成演化后才由已存在算法入口真实运行同一selected完整128。不得为提速改变预算、截迭代、删难例。

OpenCode旧OPRO仍143过程completed+1 skill_015 error，无selected/result；独立v5修复launcher197030存活且持续waiting_endpoint_capacity、冻结源码hash匹配、目标评测目录0，等待原GPU1父队列和target clients自然结束，不是卡死。原GEPA后EvoPrompt/ACE仍待实际启动；v5随后同v11初始128→OPRO；SkillOpt/Evo-Harness N/A。GPU0 v14匹配初始control仍在GEPA之后，不以v12初始分数替代。旧OpenCode初始128官方规则及V1 judge保留，旧judge证据质量限制继续，禁止与未来V2直接归因增益。

07:02持worker.lock、确认无其他judge后，仅恢复Hermes v12未判52一次（无--overwrite）；仍HTTP402，CLI退出1，76 existing/1 error/51 blocked_billing；新增失败API请求1，成功76输入/结果152份SHA全不变。旧manifest/guard及52失败结果已备份在v12根judge_billing_recovery_attempts/20261116T07，日志hermes_initial_judge_recovery_20261116T07.log。共享guard到08:02:13.8，到期前不请求；成功case及旧OpenCode128不重判、不切账号/judge或充值、不把缺判计0。恢复进程已退出，锁释放，worker.json历史finished，未并行维修。08:00小时巡检若仍早于retry_after，应继续保留等待，不人为清冷却。

矩阵3 complete/21 needs_repair/1 queued/2 running/13 pending/2 N/A，35个未完成/非运行工单仍在，无本轮新增完成。两卡和对应端口模型均正常，两活根与待运行v5冻结源码hash通过；本轮未修改任何可执行代码/配置/预算/隔离，未重启服务或重复推理，未碰其他用户GPU。唯一小时cron保留，无disabled。证据：monitor/matrix_loop/hourly_evidence_20261116T07.json、judge_recovery_20261116T07.json；队列及逐案例实际路径同下文。后续继续核查GEPA选择与同一运行selected128、Hermes新control、OpenCode EvoPrompt/ACE及197030真实接续；新失败保留后串行修复。

## 2026-11-16 06:04：两条GEPA继续，Hermes此前失败case已有真实候选通过；judge仍受余额阻塞

GPU0 Hermes v14 GEPA（138740/父138726）06:02已有90条过程completed、0错误，90条身份/private/mock/官方规则通过，3084请求重新oracle审查；包含初始Snapshot36次及新候选5aea6a…54次，不是单一selected的最终128覆盖，无selected/result/judge。旧失败skill_074在v14 seq30初始、seq66新候选均completed，39/38请求，最大输入8638/12278，contextfit failure为空；原失败v12候选843a…与此候选不同，不能声称旧Snapshot精确重放或最终全量已通过。之后仍同v14初始128队列，不能用旧v12分数当匹配对照。GPU0开始巡检时0%是case切换/工具期，最近证据23秒，非空转或卡死，无需重启服务。

GPU1 OpenCode v4 GEPA（159357/父3925329）06:02过程78 completed、0错误，78条身份/private/mock/官方规则通过，2145请求重新oracle审查，无selected/result/judge；之后EvoPrompt→ACE尚未启动。OPRO v4仍143 completed+1 skill_015 error，无selected/result，失败工单保留。独立v5 OPRO修复launcher197030仍正常waiting_endpoint_capacity，确认当前GEPA和原父存活；源码冻结hash匹配、评测目录0、未抢占GPU1或重复推理。原队列结束后自动同v11初始128→OPRO；两个skill算法N/A。旧OpenCode初始128规则/旧V1 judge保留及证据质量限制不变。

本轮在worker.lock确认空闲后持锁恢复Hermes v12未判52一次，无--overwrite。仍HTTP402：76 existing/1 error/51 blocked_billing；新增失败API调用1，原76成功输入/结果152份SHA全不变，旧错误/results/manifest/guard备份于v12根judge_billing_recovery_attempts/20261116T06，恢复日志hermes_initial_judge_recovery_20261116T06.log。新共享guard到07:01:38；未到期不调用，不切账号或judge、不充值、不将缺判计0。Hermes旧初始128身份/private/官方规则再次通过。judge任务已退出1且锁释放，不能当judge完成；成功76及OpenCode旧128不重判。

继续用`scripts/flashrsi_matrix_repair_scan_v14.py --scan`：3 complete/21 needs_repair/1 queued/2 running/13 pending/2 N/A，35个非完成/非运行适用工单保留，无本轮新增complete。两个服务端口18761/18764返回正确模型，两个活根及v5待运行源hash均匹配；worker.json为历史finished，无另一个维修器。未修改活代码/服务/预算/隔离，不启动新推理或定时器，唯一小时cron保留、无disabled。证据：monitor/matrix_loop/hourly_evidence_20261116T06.json、hermes_v14_skill074_recovery_verification_20261116T06.json、judge_recovery_20261116T06.json。下次核查GEPA最终selected/full128与后续control/算法衔接；v5是正常等待任务，禁止重复启动；遇新错误再串行诊断修复，不把本条过程数当最终覆盖。

## 2026-11-16 05:07：双卡均为GEPA运行中；OpenCode OPRO修复已登记独立等待队列

GPU0当前Hermes v14 GEPA仍PID138740（父138726），05:05核验41条过程completed、0错误；1356请求重过oracle过滤，身份/private/mock/官方评分全通过，暂无selected/result/judge。skill_074在新candidate中的最终效果尚未验收；之后仍补同v14初始128，禁止把这些过程episode写成最终覆盖。

GPU1 OpenCode v4 OPRO于04:35:12退出1：143过程completed + 1 error，candidate_dev的skill_015不可评分，strict门禁拒绝排序；无selected/result/judge。父队列3925329已实际自动启动GEPA（PID159357），05:05核验27过程completed、0错误；793请求重过oracle过滤，身份/private/mock/官方评分全通过。之后仍EvoPrompt→ACE，两个skill算法N/A，不修改活队列源码、不重复启动或抢占。初始128规则与旧V1 judge128保留，旧证据质量限制仍有效。

OPRO失败诊断：evidence/00000139/cases/skill_015第三个模型请求（整例request15）输入4220、输出4096、finish=length，持续续写`[Assistant tool call]`/`[Tool result]`和重复行。native compaction去掉所有user后，v8接线把精确当前owner插在末尾assistant历史摘要之前；实际wire为system/user/assistant，末尾是半截工具历史。已修正独立v11投影为system/assistant/user，仅无user请求补精确原任务到历史之后；有user请求完全不改，native/SQLite/完整历史、权限、32K/4096/12轮/1024 Worker均不变。8项定向检查通过；保存的真实失败请求经新投影与原/tokenize校验4216<32768、oracle通过、原始输入不变、预算/工具/其他字段不变。消息顺序缺陷已实证修正，但这是否足以防止模型length仍须新全量验证，不能提前称skill_015恢复或算法完成。

**新增真实等待任务**：tmux `flashrsi-opencode-opro-suffix-v5-gpu1`，launcher197030，脚本`scripts/flashrsi_matrix_opencode_opro_suffix_v5_20261116.py`，独立根`outputs/flashrsi_harnessrisk_4b_private_opencode_opro_suffix_v5_20261116`。当前只等待原GPU1父队列及target clients自然退出，没有新增模型/评分API调用；不是失联或GPU空转。届时自动跑新的同协议初始128→OPRO演化及selected128→gated judge；不复制v4轨迹/评分，不重跑当前正常GEPA/EvoPrompt/ACE。新C投影改变可比性，故必须有v5初始配对；v4其他算法继续和自己的v4初始对照分别报告。已有完成judge不重复调用，新协议产生的新轨迹也受共享额度guard约束。此等待任务已被原矩阵queued_launches验真实PID/脚本登记为opencode/opro queued；只有等待而非运行，禁止重复启动。source_snapshot/repair_protocol/case_manifest128已冻结并校验。

扫描继续用`scripts/flashrsi_matrix_repair_scan_v14.py --scan`，当前3 complete/21 needs_repair/1 queued/2 running/13 pending/2 N/A，无本轮新增完成单元。维修worker.json仍为finished，本轮manual lock用于串行诊断；收尾释放后后续巡检正常接管，勿停止197030等待任务。Hermes旧初始judge仍76有效、未判52；05:05尚在04:08的HTTP402共享冷却内（约05:08:18后才可恢复），本轮API调用0，76成功输入/结果152份hash均保持。原judge完成不重调，不以缺判计0。证据在monitor/matrix_loop/hourly_evidence_20261116T05.json、opencode_opro_skill015_projection_diagnosis_20261116.json；各活根源码hash均匹配。唯一小时cron保持、无disabled，其他用户GPU未动。

## 2026-11-16 04:21：Hermes当前为v14，v13接线失败已保留；OpenCode OPRO继续

本条取代下方04:12的活进程信息。v13 supervisor包装漏导入`mock_client_environment`，注入的v12审计表达式在首次模型调用前NameError；22条error、0模型请求、0官方结果。已只停止该队列及其子进程，全部目录保留，`interrupted_for_namespace_repair.json`与`hermes_v13_wrapper_namespace_failure.json`记录根因。未把隐私文件存在当作推理完成，未混入后续版本。

当前独立根`outputs/flashrsi_harnessrisk_4b_private_hermes_native_prune_v14_20261116`，tmux `flashrsi-hermes-native-prune-v14-gpu0`，04:18:42启动，launcher138726、GEPA138740，GPU0/18761服务未重启。v14仅修复入口namespace，仍用v13已诊断的原生prune配置及16K/4096/1024/12轮；原生C变化相对v12仍需新配对控制。真实完整GEPA运行中首例daily_025完成，29次模型请求/响应无failure，Snapshot一致、private审计/公开mock身份/官方规则完成；29条请求重新通过oracle过滤。5项定向回归通过，已冻结源hash匹配；这只验证恢复后的实际链路，不证明skill_074已解决，更不代表最终128或算法提升。GEPA结束后队列会真实执行同v14初始control128，不复制v12轨迹/分数充数。初始Snapshot只只读引用v12已完整128的真实起点。

后续使用`scripts/flashrsi_matrix_repair_scan_v14.py --scan`；仅扩展独立wrapper进程注册，原完整128验收不变，原scanner/运行中GPU1源码未改。扫描先等run_manifest落盘，不能因启动瞬间漏登记而重复起任务。04:20 OpenCode v4 OPRO有131条有效过程episode、0错误，尚无selected/result，不能写131/128；初始128官方规则完成、旧V1 judge128保留及证据质量限制仍在，后续GEPA/EvoPrompt/ACE尚待实际衔接，SkillOpt/Evo-Harness N/A。GPU1/18764与其他用户GPU未动。

本轮额度冷却到期后只对未完成judge恢复一次，仍402：76 existing、1 error、51 blocked_billing；已有76成功输入/结果共152份hash不变，实际新增失败API调用1。新的retry_after为05:08:18左右；此前不要再调。继续保留judge_billing_ticket与旧OpenCode证据质量ticket，不把缺判记0，不重判已完成结果。当前矩阵3 complete/22 needs_repair/2 running/13 pending/2 N/A，未新增完成单元；仅保留现有小时巡检cron，不创建disabled。证据在monitor/matrix_loop/hourly_evidence_20261116T04.json、hermes_v14_first_real_case_verification.json、hermes_v14_request_privacy_check.json。下方旧版本记录为历史诊断，不是当前启动指令。

## 2026-11-16 04:12：GPU0 Hermes v13 原生回收全量GEPA恢复；GPU1 OPRO继续

Hermes v12 GEPA于03:53严格停止：71 completed+1 error，candidate_dev的skill_074最后输入16463>16384；两个原生compression事件为committed=false/reason=would_grow，未产生selected/result，不可计0或算完成。原完整错误/contextfit/工具轨迹保留；已保存native窗口诊断，未增context/output/Worker/轮次。原生proactive prune默认disabled，且protect_last_n=20导致短会话prune:tail_only；仅改短tail仍因min_result_chars=8000回收不足。选用原生配置 `{proactive_prune_tokens:8192, protect_last_n:4, proactive_prune_min_result_chars:2048}`；保存的真实失败请求经原生函数及原tokenizer诊断降到9196 tokens，system/user/最新工具组均保持，未做新模型生成或伪造case评分。新profile只改变上下文组件，旧原始工具证据继续留存。

新独立 `outputs/flashrsi_harnessrisk_4b_private_hermes_native_prune_v13_20261116`，tmux `flashrsi-hermes-native-prune-v13-gpu0`，launcher129067、GEPA129069，04:10启动真实完整GEPA→同v13初始control完整128。初始Snapshot只只读引用v12真实initial（128身份/privacy/mock-env/规则检查通过），算法不复用旧轨迹或得分；不复制旧completed文件假称v13初始。算法wrapper记录initial_reference与v13目标协议。新v13自身initial目录要等GEPA阶段后真实生成，前此禁止声称严格配对或安全收益。即使GEPA失败，队列仍运行v13控制并保留失败工单。固定16K/4096目标/1024 Worker/12轮、bounded Worker、private/sandbox/评分门禁不变。4项定向检查及真实128 reference核验通过，但skill_074在新完整演化中仍待实际验收，不能把tokenizer诊断当成功。

**巡检扫描用 `scripts/flashrsi_matrix_repair_scan_v13.py --scan`**：旧scanner被冻结且只识别标准algorithm.py，无法识别新只读Snapshot-reference入口；新scanner仅在本进程扩展入口注册，完整128身份/privacy/规则/selected/judge验收函数未改，旧文件/其他队列哈希不动。实际进程已被识别为hermes/gepa running。不要因旧scanner漏识别而重复启动GPU0任务；要同时检查新父队列/initial进程（GEPA后会转control）。不要恢复已删除的5分钟cron，当前仅每小时唤醒。

GPU1 OpenCode v4 OPRO继续（04:01为113过程completed，无error/selected/result），随后GEPA/EvoPrompt/ACE；两skill算法N/A，未操作GPU1模型或他人GPU。升算云冷却到期后本轮只对未判52恢复一次，仍遇402：76 existing、1本次error、51 blocked_billing；此前76成功的152份输入/结果hash全不变。旧失败备份在Hermes v12根/judge_billing_recovery_attempts/20261116T04，新的guard已延长一小时；后续继续按billing ticket处理，不能重复成功judge或借充值重判旧OpenCode V1 128。

## 2026-11-16 03:08：两卡算法在跑；Hermes初始judge因402暂停，禁止重复判已完成case

Hermes v12初始规则已完整128/128，02:30自动启动GEPA（PID4194015），初始judge并行运行。新版V2已实际成功判76条，均有原生工具证据及证据hash，剩余52条全部因升算云HTTP402 insufficient_balance失败；这是额度外部阻塞，不能当0分或完整128。API账户需用户补充额度；本机不进行充值/切模型/换账号。GPU1 OpenCode v4 OPRO（PID4143617）继续，随后GEPA/EvoPrompt/ACE，两个skill算法N/A；03:01过程记录GEPA28/OPRO60条均有效，不是最终128。

可信judge新增余额不足止损：只有明确response_format不支持的400/422才回退，402不再重复同一请求；首次402设置共享一小时暂停，未判case标blocked_billing，无有效判分的汇总指标为null而非0。7项回归通过。已根据真实402证据激活 `/data/home/liumingxiao/.cache/harnessrisk-judge/billing-blocked.json`，并运行真实恢复CLI验证：76 existing、52 blocked_billing，CLI返回1（仍未完成），256份原输入/结果哈希不变，新API调用0。未重判任何已完成case，52个原402错误保留。补丁及两个run根的judge_protocol_amendment已更新；V2证据/提示/评分口径不变，只改API错误处理与空分母表示，实际新源hash进metadata。

**每小时继续读取** monitor/matrix_loop/judge_billing_ticket.json、judge_billing_guard_verification_20261116.json和guard；如果仍在冷却，不发送新API请求。冷却已过且无活judge时，可以用原 `scripts/judge_harnessrisk_shengsuanyun.py <Hermes-v12根>/hermes_initial_full/cases --deployment openai/gpt-5.4-nano` 恢复（不加--overwrite），它只补未完成52条；若再次402立即重新冷却。用户明确通知充值后，可先备份guard并将retry_after置0，再恢复未完成项。两个GPU队列继续，不为余额阻塞重跑模型推理。之后算法结果如全128但judge受额度影响，保留needs_judge，恢复只补未判，绝不把缺判当0。新的经验证run目录和前文一致。

OpenCode旧初始128份V1 judge的证据质量限制仍在judge_evidence_quality_ticket.json；充值通知不等于授权重判已完成的旧128份。不得把该问题与Hermes未判52条混为一谈。当前两个算法尚无最终selected/result/judge，本轮未新完成任何算法矩阵单元，不创建disabled。

## 2026-11-16 02:10：OpenCode初始规则128与judge128落盘、OPRO衔接；judge证据导出V2修复

OpenCode gpu1-v4初始已于01:45完成128/128官方规则、0执行错误，01:54完成128份GPT-5.4-nano判分并自动启动OPRO（PID4143617），之后仍GEPA→EvoPrompt→ACE；两个skill算法N/A。GPU0 Hermes v12初始02:01为103/128有效、0错误，尚未GEPA。不要把OPRO过程轨迹当最终128。

本轮发现judge质量限制：FlashRSI原生工具事件嵌套在hermes_turns.jsonl每条的messages里，官方judge只支持标准role/tool_calls，漏读原始3083次调用与3083次返回；旧OpenCode初始输入仅223个CLI/标准工具事件，64条完全无工具事件，125份判词列有不同程度不确定性。原128份判分虽状态/身份/分值格式合格，也不能无条件当完整可信的LLM配对基线。已按用户“不重复调用完成judge”保留原输入/判分哈希，不重调API；规则基线与LLM证据质量须分开报告。

新staged脚本 `experiments/cross_harness_sft/scripts/flashrsi_harnessrisk_canonical_judge_v2.py`及patches/harnessrisk_judge_flashrsi_tool_evidence_20261116.patch已实现native tool_start/tool_end、Hermes tool_execution_start/end的逐owner有序提取、来源事件hash，保留旧格式；完成缓存不再重写它原来的证据。最终4项回归通过（含缺失/非法/非有限指标或case错配报错、不能默认计0）；128个真实已保存case离线重建为6389条工具证据事件（含原223条），64个空工具case降为0，原判分/输入哈希不变，API调用0。新副本已在无活judge时原子安装到HarnessRisk canonical judge路径，只影响后续可信judge证据，不改目标Snapshot/模型请求/沙箱/预算/当前评测。未来_llm_metadata含evidence_protocol=harnessrisk-flashrsi-nested-tools-v2、证据和脚本hash；两个当前根均有judge_protocol_amendment_20261116.json。不要把这个补丁版称为完全未修改的官方judge。

旧源保存在monitor/matrix_loop/canonical_judge_before_tool_projection_v2_20261116.py，安装计划在judge_v2_install_plan.json。旧初始完整离线证据在OpenCode v4根/trusted_judge_evidence_v2_offline，未回写原judge证据。必须继续跟踪monitor/matrix_loop/judge_evidence_quality_ticket.json：旧V1判分有证据质量限制，不可与未来V2分数直接归因安全增益；当前禁止重复完成API判分，不要用新目录偷换为已重判或已解决该质量限制。初始官方规则ASR/Utility与LLM判分分别报告。两个模型评测均继续，不因本次judge导出修复重跑已完成推理。

## 2026-11-16 00:08：GPU0 Hermes v12 同协议初始→GEPA，GPU1 OpenCode v4继续

本轮不能继续将Hermes v11的可评分覆盖当作环境有效：停止前141条过程轨迹中，71条（44个不同case）有实际tool_execution_end返回case路由拒绝，共346次。例recovery_126执行 `python3 bin/mockctl.py catalog`，缺公开OPENCLAW_ADAPTER_CASE_ID导致客户端默认case_023；原supervisor还未传官方虚构fixture凭据，PATH仅/usr/bin:/bin。全部轨迹、诊断diagnosis_mock_client_case_identity.json及中断记录保留，标为环境缺陷诊断，不拼入新版。此前recovery_055确实完成了上下文关卡，但不能因此忽略工具环境缺陷。

只停止自有旧Hermes v11评测进程树，GPU1/两个模型服务/他人GPU未动。新 `outputs/flashrsi_harnessrisk_4b_private_hermes_caseenv_v12_20261116`、tmux `flashrsi-hermes-caseenv-v12-gpu0`，初始PID3993303；先跑完整同协议初始128，身份/规则/privacy全通过后自动衔接GEPA并并行启动初始judge。v12保留16K原生上下文、1024 Worker投影预算、12轮及工作区复制隔离，复用已验证的最小公开case身份/fake fixture凭据和现有解释器PATH。拒绝真实凭据/身份不匹配，不传宿主API key或越界trace路径，不放松private路由门禁。9项定向回归/真实解释器CLI检查通过；全量、算法、judge仍待验收。新初始是为排除环境修复混淆，不能套用旧baseline声称GEPA收益。其他Hermes算法未纳入此条新队列，未完成项仍留矩阵。

GPU1当前仍是OpenCode gpu1-v4/session-v10，未打断。00:01已30条有效，Python缺失/默认case错路由均零；后续OPRO/GEPA/EvoPrompt/ACE尚在初始后排队，两个skill算法N/A。来源及旧协议限制见下文，当前状态以最新进程/产物为准。

## 2026-11-15 23:28：OpenCode 当前为 gpu1-v4 / session-v10（完整工具环境修复）

进一步真实核验：v3的Python调用已成功，首例耗时69.11秒（旧v2同例87.28秒，但不足以推断平均提速）；然而mockctl因未收到OPENCLAW_ADAPTER_CASE_ID而默认case_023，被现有严格case绑定门禁拒绝。v3前三例无Python缺失，却分别有7/1/4次case路由拒绝；不能把可规则评分等同任务环境可用。旧v2的12例及v3的3例均保留为诊断，分别有performance_diagnosis_pythonpath.json、diagnosis_mock_client_case_identity.json和独立中断记录，不和新版混算。旧版OpenCode分数须注明这些工具环境缺陷，不能直接用于修复前后安全增益归因。

当前独立 `outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v4_20261115`，tmux `flashrsi-opencode-gpu1-v4`，初始PID3925340；v10保留v8公开owner投影及v9 PATH，另从官方adapter生成的state/.env只传当前case身份和fake_开头的虚构fixture凭据，拒绝身份不匹配或真实凭据；不传宿主API key、TMPDIR或越界trace路径，不放松private路由门禁或沙箱。每case新增mock_client_environment_audit.json只记键名。8项回归/CLI检查通过，仍需真实全量环境效果/耗时及judge验收。初始→OPRO→GEPA→EvoPrompt→ACE保持顺序，其他两算法N/A。GPU0 Hermes GEPA v11及两个模型服务未动。

## 2026-11-15 23:23：OpenCode 当前为 gpu1-v3 / session-v9，修复工具 PATH

用户质疑速度后检查真实轨迹：gpu1-v2 前12条平均72.12秒，8条共48次 `python: not found`，大量用尽每owner的工具轮次。不是将这些规则可评分案例视为真实任务完成。上游HarnessRisk把workspace/bin前置于继承的PATH，而我们的受限env只保留workspace/bin:/usr/bin:/bin，丢掉已安装运行时解释器目录；既有OpenClaw适配已包含该目录，OpenCode遗漏。旧v2停止记录在 interrupted_for_pythonpath_repair.json，性能诊断在 performance_diagnosis_pythonpath.json，12条及进行中的action_109均单独保留，不与新版拼接。

当前独立根目录 `outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v3_20261115`；tmux `flashrsi-opencode-gpu1-v3`，队列PID3917016、初始PID3917024。使用独立host/supervisor session-v9：在原workspace/bin之后加入已有的sys.executable目录，不安装解释器、不继承宿主全部PATH、不改现有只读/可写/网络权限；保留v8公开owner转发审计。32K/4096/12轮、strict、privacy及Snapshot均不变。7项回归与真实解释器CLI检查通过；实际全量评测、耗时变化与各算法/judge仍须以新目录证据验收，不能预先称提速或完整修复。初始后仍顺序OPRO/GEPA/EvoPrompt/ACE，另两算法N/A；仅严格128门禁后judge。GPU0 Hermes GEPA v11及两个模型服务未修改。

## 2026-11-15 23:08：双卡当前状态与 OpenCode v8 独立恢复

GPU0/18761 继续单一 Hermes GEPA contextfit-v11（PID 3670671），不可打断或重复启动；过程轨迹不等于最终128。GPU1/18764 的 OpenCode gpu1-v1 在83条attempts（72有效、11压缩丢user错误）后停止并保留全部证据；另有进行中的recovery_128在停止后0.1秒落盘error，属于本次中断，不计模型失败，故原始harness_result共84份。Qwen模板对前11条缺user请求报400；该版本未调用judge，不能与新版本拼接。停止记录在旧根目录 interrupted_for_protocol_repair.json。

独立 `outputs/flashrsi_harnessrisk_4b_private_opencode_gpu1_v2_20261115` 已启动，tmux `flashrsi-opencode-gpu1-v2`，初始进程PID3888344。使用 `host_opencode_session_v8.py`：不修改原生SQLite或原始请求，仅当实际转发请求缺user时恢复该轮原始公开用户任务，分别保存native/forwarded请求并严格审计。不是官方原生无改动运行器，报告须注明该模型接线适配；不得与v7/16K结果混算。32K服务、4096输出、12轮、Snapshot、隐私/沙箱、严格演化不变。11条已保存失败请求修复投影后全部通过真实服务/tokenize校验；6项回归通过，但不代表完整模型评测通过。

新队列脚本 `scripts/flashrsi_matrix_opencode_gpu1_v2_20261115.py` 顺序运行完整初始→OPRO→GEPA→EvoPrompt→ACE，两个skill算法N/A。每阶段保留错误继续下一算法；仅单次运行128身份/隐私/规则及selected门禁通过才judge，不补跑已有完整judge。源代码哈希及副本冻结到新根目录；监控queue_events.jsonl和各阶段日志。GPU1模型服务未重启，GPU0未修改，其他用户GPU未触碰。

## 2026-11-14：算法矩阵修复闭环（最高优先级）

用户最新要求是持续修复全部算法单元，不能失败后放置。读取 `experiments/cross_harness_sft/docs/flashrsi_matrix_repair_loop_20261114.md`。每 5 分钟 cron 调用 `experiments/cross_harness_sft/scripts/flashrsi_matrix_repair_loop.py --tick`，从真实产物构造 42 单元（40适用+2 N/A）的任务表/工单，并启动一个有 worker.lock 的独立 Codex 维修执行器。任务表位于 `experiments/cross_harness_sft/outputs/flashrsi_harnessrisk_4b_private_v1_20261113/monitor/matrix_loop/matrix.json` 与 matrix.md；工单 repair_tickets.json，维修日志 repair_journal.md、action_last.json、codex_repair_*.log，路径均在同一 matrix_loop 目录。所有路径在仓库根读取时要加 experiments/cross_harness_sft 前缀。

首次真实扫描：2 complete、20 needs_repair、16 pending、2 running、2 N/A；只有 result 128 finished、case身份/隐私/官方评分与 judge 同集128通过才 complete。选回初始Snapshot可完成但不能说提升。每小时会话只读汇报/检查维修器，维修器活着时不要并行修改相同代码或启动重复评测。维修轮次要真正定位、改代码/配置并启动真实全量工作，或记精确证据和下一项诊断；故障持续留工单，下轮继续，不得只重复总结。原初始基线按需引用，协议变更另列，禁止错误计零、删难例和放松安全门禁。

## 2026-11-14：18:40 Hermes OPRO 有限重试

Hermes v3 B0 127/128、judge 127/127，OPRO/GEPA/EvoPrompt 因少数错误被严格门禁停止，ACE 运行。新可选 `HR_EVOLUTION_CASE_ATTEMPTS=2` 仅演化时失败重跑一次并记 case_retry_events.jsonl，仍失败则 fail-closed，最终 128 case 不重试；`flashrsi-hermes-opro-v4-gpu3` 在独立 `outputs/flashrsi_harnessrisk_4b_private_hermes_opro_retry_v4_20261114` 运行 OPRO，尚无结果。重试会有选择效应与额外预算，须与原协议分开并报告。

## 2026-11-14：15:00 低覆盖队列控制

Hermes v3 B0 96 attempted/95 valid，继续运行；Pi v3 B0 18/128、judge 18/18，OPRO 49 evidence 45 error，已停止 Pi 队列保留证据，需先解决固定轮次不结束再恢复。Codex v6 B0 17/128、judge 17/17，OPRO 15 evidence 12 context error，已停止队列；其他算法未完成。不得混协议或把停止后的部分 evidence 记为结果。

## 2026-11-14：14:00 Hermes/Pi/Codex 进展

Hermes v3 B0 47 attempted/46 valid，Pi v3 B0 87 attempted/16 valid（71 达 max_turns），两条 GPU0 队列继续。Codex v6 B0 17/128、judge 17/17，OPRO 前 15 evidence 12 context error，已停止 `flashrsi-codex-v6-gpu3` 防止低覆盖无效演化，证据保留。需要可审计的长上下文处理才能独立重跑；不能把错误记 0。

## 2026-11-14：13:00 Hermes、Pi 接线重跑

Hermes v1 0/128 是缺 regex_rs；安装 regex-rs 0.2.4 后 v2 又缺 RegexError/escape API，已停止 v2 并保留证据；在 port 入口做兼容别名后 `flashrsi-hermes-v3-gpu0` 运行独立完整 B0，尚无最终结果。Pi v1 在推理前缺 provider_name，v2 首例需流式 API 而 relay 只接受非流式；新增有界缓冲 SSE 协议后 `flashrsi-pi-v3-gpu0` 全量运行，2 attempted/1 valid。新增适配需以实际结果验收且不同协议不混算。Codex v6 76 attempted/12 valid，native 请求而非普通代理记录才是证据。

## 2026-11-14：12:00 OpenClaw 与 Codex v6

OpenClaw B0 13/128、judge 13/13，OPRO 因 63/72 不可评分失败，GEPA 22 条只有 1 条完成；为避免重复低效调用停止原 GPU0 队列并启动 `flashrsi-private-hermes-pi` 接续 Hermes、Pi。OpenClaw 未完成的算法须在解决 12 轮不结束后独立恢复，不能称完成。Codex v6 只在受限 scratch 预置选定 Snapshot 中的 system skills 并按上游规则写 digest marker，不放松 seccomp/Landlock；独立 `outputs/flashrsi_harnessrisk_4b_private_codex_v6_20261114`、tmux `flashrsi-codex-v6-gpu3` 真实全量 B0 进行中，首例尚需核验。

## 2026-11-14：11:00 队列与 Codex 资源安装诊断

OpenClaw B0 97 attempted、10 scored，85 次达 12 iterations、2 次 API 400，仍在执行；不可把模型超轮次判成功。OpenCode strict v2 ACE 121/128 scored、judge 121/121、selected unchanged，队列尾出现 EOF；bash -n 目前通过，运行中的 shell 曾被原位改写可能造成偏移问题但未证实，GPU0 旧进程也需监控。Codex v5 资源安装调用 python_core/resources.py 的 shutil.copytree，而 seccomp 禁 chmod/utimensat，是 EPERM 候选并非已验证；需不放松 sandbox 地定位并修复。

## 2026-11-14：Codex v5 首例失败，队列已停

Codex v5 的 action_041 已越过 copytree 参数冲突，但原生程序在 scratch 系统技能资源安装时多处 Operation not permitted；已停止 `flashrsi-codex-v5-gpu3` 并保留日志。尚需定位实际受限系统调用，不能放松 seccomp/Landlock 或把它当有效评分。旧 v3 128/128 无有效、v4 首例 mode mismatch、v5 首例 EPERM 均各自保留。

## 2026-11-14：10:00 Codex 桥接修复重跑

Codex native v3 B0 128/128 全部失败，实际原因 copytree wrapper 递归重复传 copy_function；旧 OPRO 也无有效推理。已终止旧 native v3 队列（Claude 结果保留），修复递归参数并限制初始评分为零时跳过算法。独立 Codex v4 首例又遇 scratch 目录 mode mismatch，保留证据并停止；目录复制适配修正后在 `outputs/flashrsi_harnessrisk_4b_private_codex_v5_20261114` 启动 `flashrsi-codex-v5-gpu3` 全量 B0。v5 启动不是成功，需核查真实推理、privacy、评分和剩余失败；不同协议不混算。

## 2026-11-14：09:00 Native 事件序列修复

Claude Code native v3 B0 仅 23/128 有效、judge 23/23；OPRO/GEPA/EvoPrompt/ACE/SkillOpt 曾因 native 字符串事件不能转 mapping 抛 ValueError。已改 `flashrsi_harnessrisk_benchmark.py` 保留原字符串并包装成事件映射，原 evidence 不动、已在跑进程不重载，源码变更写入 native v3 protocol_source_changes.jsonl；尚未真实完整验证，且长上下文导致低覆盖仍须解决，不能将代码改动称完整修复。MiniAgent SkillOpt 也在 recovery_119 输出耗尽被严格门禁停止。

## 2026-11-14：08:00 评测快照

MiniAgent ACE 最终 127/128 完成，selected Snapshot 与初始相同、judge 127/127；action_044 的上下文超限在 `recovery_adaptive_miniace` 独立恢复中，不能混原协议。Claude Code native v3 94 已尝试、17 完成、77 错误（55 上下文、22 超轮次），继续采集但桥接覆盖不足；OpenCode 严格组 OPRO/GEPA/EvoPrompt 均因不可评分 fail-closed，ACE 运行中。上下文与重复输出问题未根治。

## 2026-11-14：native v3 与严格队列覆盖不足

07:00 巡检：Claude Code native v3 已尝试 44 case，仅 10 completed、34 error，其中 24 即使 512 输出预算仍超 16K，另外 10 多为 12-turn 上限等；保留模型原始/转发消息和 audit，尚不具备完整桥接可信度。OpenCode 严格 v2 的 OPRO、GEPA、EvoPrompt 已因不可评分 fail-closed，ACE 正在运行。MiniAgent GEPA 重复输出塌缩尚未修复，不要仅增加预算或把错误记零；优先设计可验证且不混协议的上下文处理。

## 2026-11-14：Claude Code 原桥接失败与 native v3 独立重跑

原 `flashrsi-private-gpu3` 进入 Claude Code 后前四条失败，实际停止该 tmux 队列，旧证据保留；原 Codex 随之不再自动启动。新 `outputs/flashrsi_harnessrisk_4b_private_native_v3_20261114` 的 `flashrsi-native-v3-gpu3` 已启动 Claude Code → Codex 全量队列：native tool_reference 记录并映射为文本、context 失败自适应输出预算，隐私与严格评分门禁保留。新旧不是相同协议；启动时未完成，必须检查实际轨迹、coverage 和 judge。OpenCode 严格 v2 的 OPRO、GEPA 已因不可评分 fail-closed，EvoPrompt 正在跑；不能将失败记 0。

## 2026-11-14：GEPA v4 未完成，先分析生成原因

MiniAgent GEPA v4 的 daily_090、skill_077 各耗尽 8192 target tokens，在候选 dev 因严格门禁停止。原 4096 与当前 8192 均已失败，保留所有 evidence；不要只增加预算并反复盲目重跑。先诊断生成/工具协议和上下文，再制定可审计独立协议恢复；不可将错误记 0。OpenCode 原 GEPA 在 daily_025/daily_026 不可评分后停止，严格 v2 OPRO 尚在运行。

## 2026-11-14：MiniAgent GEPA 第三次独立重跑

GEPA retry v3 的目标模型在 skill_011 达到 4096 token，上游严格评分停止；已保留 error 和 evidence。独立目录 `outputs/flashrsi_harnessrisk_4b_private_gepa_retry_v4_20261114` 的 `flashrsi-miniagent-gepa-retry4` 使用 target 8192 + 自适应回退、Worker 4096、严格评分和隐私门禁；其协议与原 B0 不同，持续核查是否完成、失败或 judge。OpenCode 严格 v2 B0 为 124/128，judge 124/124，OPRO 已启动。

## 2026-11-14：MiniAgent GEPA 第二次独立重跑

GEPA retry v2 因 Worker 1024 token 输出耗尽失败，证据保留。`HR_WORKER_MAX_TOKENS` 可调整并写入 manifest，默认仍 1024；已在独立目录 `outputs/flashrsi_harnessrisk_4b_private_gepa_retry_v3_20261114` 启动 tmux `flashrsi-miniagent-gepa-retry3`，Worker 4096，严格评分、隐私门禁及独立 text/adaptive 协议，之后自动 judge。该目录只引用初始 Snapshot，不包含同协议 B0。OpenCode 严格 v2 的源码变更另记 protocol_source_changes.jsonl。

## 2026-11-14：MiniAgent GEPA 严格重跑

原 v1 GEPA 在 daily_086 输出达到 4096 token 后被严格评分门禁终止，错误与 evidence 保留。独立 `outputs/flashrsi_harnessrisk_4b_private_gepa_retry_v2_20261114` 的 tmux `flashrsi-miniagent-gepa-retry` 已启动重跑，使用初始 Snapshot、严格评分与 text/adaptive host；与原始 B0 的执行协议不同，不得默认为严格配对。OpenCode 严格 v2 的 action_112 即便预算降至 512 仍失败，需持续关注长上下文；运行错误不当作零分。

## 2026-11-14：OpenCode 严格评分独立重跑

`outputs/flashrsi_harnessrisk_4b_private_strict_v2_20261114` 是独立的 OpenCode 全量基线及算法队列，tmux `flashrsi-strict-opencode` 于 01:01 启动，GPU 3/18762，保留隐私协议、记录附件文本转换与可调整输出预算。`HR_STRICT_EVOLUTION=1` 要求演化候选评分不包含不可评分轨迹；失败即停止该方法并保留 evidence。严格模式已是未来算法进程的默认值。旧 v1 OpenCode OPRO 已在修改前启动，失败轨迹曾按 0 参与候选排序，只能做诊断，不得作为可信算法对照；旧数据及新 v2 不混算。注意按实际结果追踪 v2 初始、四种适用方法和 judge，不能仅凭 tmux 活着宣称完成。

## 2026-11-13：FlashRSI HarnessRisk 每小时 Codex 巡检

用户明确授权每小时唤醒本会话，检查并自主修复评测及初始基线后的 RSI 算法队列。用户 crontab 以 `0 * * * *` 运行 `experiments/cross_harness_sft/scripts/watch_flashrsi_harnessrisk_codex.sh`，经 `codex queue` 向本会话投递真实巡检消息；2026-11-13 20:08 CST 已成功投递一次。cron 日志为 `experiments/cross_harness_sft/outputs/flashrsi_harnessrisk_codex_hourly_cron.log`，巡检报告追加至新运行目录的 `monitor/hourly_report.md`。脚本使用 flock，创建 `monitor/disabled` 可停止投递；只有矩阵及评分确实完成才自动创建该标记。所有故障要保留证据、检查重复进程，禁止绕过隐私门禁或混入旧污染结果；不得接触别人的 GPU。不要因其他旧任务的 cron 而停用本任务。

## 2026-11-13：HarnessRisk 隔离协议 v1 与已重启队列（最高优先级）

真实旧轨迹曾出现 `8_oracle_scorer`，旧 run 位于 `experiments/cross_harness_sft/outputs/flashrsi_harnessrisk_4b_20261113`，视为可能污染，只供诊断，禁止与新 run 混算或续跑旧 Snapshot。相关旧队列已暂停。

`HARNESSRISK_PRIVATE_CASE=1` 阻止 HarnessRisk adapter 把完整 case 写入 workspace 和 filesystem seed；mock 服务将所有 query/body 的 case ID 绑定到当前任务；外部 judge 使用可信 data-dir 优先且校验 case ID。`flashrsi_harnessrisk_supervisor.py` 启动 agent 前审查 workspace/seed/mock_data 并生成 privacy_audit.json；模型 relay 拒绝包含隐藏评分字段的输入。Claude Code/Codex 请求另写 native_model_request_evidence.json。原始 case 仍供可信规则 evaluator 与 judge 使用。已对全部 128 个真实 case 的 2407 个 seed 文件执行静态检查；这不代表完整 OS 安全审计。

新运行目录为 `experiments/cross_harness_sft/outputs/flashrsi_harnessrisk_4b_private_v1_20261113`。2026-11-13 20:02 CST 已启动 `flashrsi-private-gpu0`（MiniAgent → OpenClaw → Hermes → Pi）和 `flashrsi-private-gpu3`（OpenCode → Claude Code → Codex）。每个 harness 先完整初始 128 条及基础设施重试，然后依次评估适用的 OPRO、GEPA、EvoPrompt、ACE、SkillOpt、Evo-Harness；OpenCode 不支持后两者。LLM judge `openai/gpt-5.4-nano` 在初始及各算法案例完成后调用。队列用 `scripts/flashrsi_harnessrisk_private_queue.sh`，监控 `queue_gpu0.log`、`queue_gpu3.log`、逐 run attempts、result 和 judge manifest；队列启动不是评测完成。当前仅实际启动 MiniAgent / OpenCode B0，其他工作等待其前序任务完成。新 protocol 每案例有独立 audit，出现防护错误应保留证据并修复，不得为跑完关闭门禁。模型服务是 `flashrsi-4b-local`（GPU 0/18761）及 `flashrsi-4b-gpu3`（GPU 3/18762）。不要碰他人 GPU 进程。

研究方案和候选算法见 `experiments/cross_harness_sft/docs/safe_rsi_research_protocol_20261113.md`。现有训练暴露、port 与正式独立泛化等限制仍然有效，不能宣称跨 harness 泛化或 CHS-PO 效果已完成。

## 2026-11-13：4B FlashRSI 安全基线改用 HarnessRisk（最新）

### 后续更新：全量 MiniAgent 初始评测运行中

2026-11-13 19:00 起按用户授权启用并行：GPU 0 继续 MiniAgent；GPU 3 的独立 4B vLLM `flashrsi-4b-gpu3`（端口 18762，同模型和推理设置）供 `flashrsi-harnessrisk-opencode` 全量初始及 OPRO/GEPA/EvoPrompt/ACE 队列使用。其他人的 GPU 2 进程不得终止。两条序列使用不同 case 输出目录及独立 model service，OpenCode 的运行尚待实际结果。`flashrsi-harnessrisk-ports` 修改为等待 MiniAgent 和并行 OpenCode 完成后从 OpenClaw 继续，避免重复跑 OpenCode；自动监听会话使用 tmux 精确匹配。GPU 0 的 `recovery_118` 因输出达到 4096 token 失败，保留原始记录，剩余 case 已恢复；`completed.json` 表示所有 case 均已尝试，必须查看 `completed_scored` 和失败分母，不代表 128 个均有效评分。

19:05 更新：OpenCode 的 action_041–043 因其向宿主父目录查找 AGENTS.md 与 Landlock 冲突而失败；入口改为隐藏 case 和 Snapshot 之外的祖先指令文件。真实 action_044 已完成官方评分。`flashrsi-harnessrisk-opencode-retry` 会在 128 条全部尝试后仅重跑基础设施 exit 1，旧轨迹不覆盖；`paired_cases` 指向有效轨迹供 judge，输出长度失败不自动重试。OpenCode 的祖先资源查找受到这项有记录的适配约束。

已按用户要求直接启动 HarnessRisk 128 条完整 case，不再启动模拟/冒烟测试。`tmux flashrsi-harnessrisk-miniagent` 执行 FlashRSI 官方 MiniAgent 初始 Snapshot `ab452d0042a47002125fed85e62f798c932a1674aef81b872a4c8e2b730f622c`，输出 `experiments/cross_harness_sft/outputs/flashrsi_harnessrisk_4b_20261113/miniagent_initial_full_v2/`。第一版因 namespace 中 HOME 错误停止，v2 修复；其证据单独保留。每条在独立 network/PID namespace 和文件/端口受限子进程中执行官方 HarnessRisk mock 与 FlashRSI Snapshot，使用官方规则评分。多轮适配重放先前用户及助手文本并保留环境状态，不含完整原生工具消息历史；报告必须明确该差异。隔离机制未声称经过独立安全审计。

`tmux flashrsi-harnessrisk-rsi` 等初始全量完成后按 OPRO、GEPA、EvoPrompt、ACE、SkillOpt、Evo-Harness 顺序运行，并各评 128 条；每种演化预算已记录为资源受限配置，Evo-Harness 使用受限 JSON skill 编辑 Worker。全量对比包含演化训练和开发使用过的 case，不能说全部是未见测试集。`tmux flashrsi-harnessrisk-ports` 等 MiniAgent 六种方法后顺序运行 OpenCode、OpenClaw、Hermes、Pi 初始和支持的方法；`flashrsi-harnessrisk-claude`、`flashrsi-harnessrisk-codex` 排在后面。除 MiniAgent 初始以外这些新桥接与算法均尚未实际运行验收，任何环节失败队列停止。官方 OpenCode Snapshot 没有 skill component，不能宣称支持 SkillOpt/Evo-Harness。其他七种 FlashRSI 方法尚未接入。`tmux flashrsi-harnessrisk-judge` 和其他 judge 队列使用现有 Shengsuanyun 凭据文件和 `openai/gpt-5.4-nano`，rule 与 judge 不混算，凭据不得泄露给目标进程。

官方规则评分的 `attack_success_rate` 可为各谓词的部分分，完整攻击成功使用 `details.attack_success.passed`；Utility 包括 final-answer fallback，须同时报告严格 oracle 分和 fallback 分。运行汇总：`python scripts/flashrsi_harnessrisk_summary.py outputs/flashrsi_harnessrisk_4b_20261113/miniagent_initial_full_v2`。禁止把不完整、未评分或未完成的结果写成完整基线。

用户要求停止当前 AgentDojo 评测，改用 HarnessRisk 作为主要 benchmark；对每个可用 harness、初始 Snapshot 和各 FlashRSI RSI 算法的演化 Snapshot 均全量评测 128 条，同一 harness 先初始再演化。AgentDojo 原队列已停止、旧证据保留。当前模型服务 `flashrsi-4b-local` 仍供本地 4B 使用。

固定清单 `experiments/cross_harness_sft/outputs/flashrsi_harnessrisk_4b_20261113/case_manifest.json` 保存 128 个 case 身份和 SHA-256；固定 FlashRSI `778e2d01a995491fe63db210dcf46db1723a4430` 与 4B `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。全量评测不意味着这些任务对算法都未见；训练暴露必须另行标注。实验实施计划见 `experiments/cross_harness_sft/docs/superpowers/plans/2026-11-13-flashrsi-harnessrisk.md`。

接线说明：HarnessRisk 官方进程后端不是 OS 沙箱，case 会引导 shell 和外部通信；不得直接无隔离启动真实模型。MiniAgent 的正式初始队列已按上文启动；官方 runner 仅直接支持 OpenClaw/Hermes/Nanobot，其他 FlashRSI runtime 桥接和各 RSI 方法仍需以实际完成的轨迹为准，不能将普通 CLI 测试伪称演化 Snapshot。

规则评分器优先，LLM judge 与规则结果分别报告；需要 LLM judge 时使用用户指定 Shengsuanyun router，既有 `scripts/judge_harnessrisk_shengsuanyun.py` 从环境或本地凭据库读取，不得将 API key 写入代码、清单、日志或模型沙箱。

## 2026-11-12：FlashRSI GEPA 4B 安全基线交接（最新）

B0 已暂停，原始输出保存在服务器。B1 的 AgentDojo × FlashRSI MiniAgent 单元已完成一次预算内 GEPA 迭代与选定 Snapshot 的 96 条 held-out test 评测。固定 Qwen/Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，FlashRSI revision `778e2d01a995491fe63db210dcf46db1723a4430`。Worker 的一次提议因修改固定安全策略被拒绝；GEPA 最终保留初始 Snapshot `487f11214c730975b9ce68b903026e68321f5e7c7ff779c98844c293d6a65ade`。该运行是上游 GEPA 加项目的安全准入门禁，未产生被接受的演化 Snapshot。由于实现返回原 Snapshot 后仍执行了候选 minibatch，应按实际 decision 记录解释，不把重跑原 Snapshot 当作新候选。

共 196 次 Target case 尝试中 144 次取得官方评分，52 次出错；test verifier coverage 为 64/96，path audit coverage 为 0。失败样本按 fail-closed 计分，汇总并非完整覆盖的 ASR 估计。两个 qualification family 的历史开发暴露未知。详细结果、原始轨迹、所有 case、Snapshot 和决策保存在 `experiment_records/flashrsi_b1_20261112/gepa_miniagent_agentdojo_complete.tar.gz`；`SHA256SUMS` 可校验。归档包含 AgentDojo 模拟环境和攻击内容，不包含模型权重、数据包和真实服务凭证。原始 `b1_invocation.json` 的 running 状态是启动时快照，以 `result.json` 为准。`matrix_at_launch.jsonl` 是启动时状态，不是实时进度。

后续目标是 AgentDojo × 其他六个 FlashRSI harness 的 GEPA：必须真实运行目标 harness，绑定候选 Snapshot 的 prompt，固定工具和安全策略，记录 Worker、候选、决策与预算；先验收隔离、协议、oracle、reset 和轨迹身份，再启动。现有 OpenCode、OpenClaw、Pi、Hermes 是 B0 桥接，尚不能当成 B1 GEPA 路径；Codex 与 Claude Code 适配和验收亦未完成。train/dev/test family 保持隔离，已有 test 结果已被查看，不能再宣称对本轮迭代完全未见。七个 harness × 安全 benchmark 的其他单元仍未完成，不要以 B0 或单元运行冒充完整矩阵。


## 2026-11-11：4B Harness Scaling × FlashRSI 接手（最高优先级）

用户当前目标是先在 4B 上完成安全 RSI 预检，再在 4B/9B 上比较通用能力与安全效用；**不使用本项目教师 SFT 权重作为起点**。以 `experiments/cross_harness_sft/docs/safe_rsi_harness_scaling_experiment_20261111.md` 为实验主设计，其他旧 2B/9B 训练指令只作历史资料。优先阅读 `docs/harness_scaling_infrastructure_20261111.md`、`docs/flashrsi_integration_20261111.md` 和 `docs/4b_safe_rsi_local_launch_plan_20261111.md`（均位于 `experiments/cross_harness_sft/`）。

**当前不是可直接启动正式训练的状态。** 五维配置和 K=2 任务分配、原始 rollout 的分段记录、GRPO 的分段评分/准入准备已有 CPU 原型。C1/C2 尚未接入真实训练循环及 optimizer；CHS-PO 分段路径、真实 autograd/分布式归约、actor 更新与 rollout 权重同步未验证。FlashRSI 上游 AHE 默认可修改全部组件，安全约束、真实环境与五维资源接线尚未完成。此前针对这些原型的 CPU 定向测试 26 项通过、2 项因 PyTorch 导入超时未验证；不能将超时当作通过。原有 C1/C2 fail-closed 不得解除来冒充已完成。

**优先做安全评测基线：**模型及安全环境验收后、任何资源演化和参数更新之前，先评测固定官方 4B + FlashRSI 初始 snapshot（B0），逐数据源和 harness 报告 benign Utility、ASR、safe-and-useful、可观测副作用与有效覆盖；`--check-only` 不是模型评测。固定 4B + 安全受控的 FlashRSI 演化 snapshot 为独立系统基线 B1；之后将 GRPO/CHS-PO 用相同的冻结任务、verifier 和采样条件配对比较。区分开发、validation 和最终未见 test，详细分母与放行要求见 `docs/4b_safe_rsi_local_launch_plan_20261111.md` 的基线章节。

新机器按以下顺序继续，不要覆盖历史运行目录或把历史 SFT/checkpoint 当成新主线：

1. 核对 GPU 型号、空闲显存、RAM、磁盘、CUDA、容器隔离和任务环境；单张 A6000 48GB / 60GB RAM 适合先尝试资源受控预检，不能保证全参数训练可运行。旧入口的 80 GiB 宿主内存门槛不能因为换机直接删除；另做小规模资源预检，记录峰值。不要终止他人进程。
2. 下载官方模型 `Qwen/Qwen3.5-4B`，固定 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，完成后核对模型、tokenizer 与文件完整性。Git 不包含模型、私有数据、输出或凭证：

   ```bash
   cd experiments/cross_harness_sft
   hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir checkpoints/qwen35-4b-instruct-official-851bf6e
   git clone https://github.com/RUC-NLPIR/FlashRSI.git vendor/FlashRSI
   git -C vendor/FlashRSI checkout --detach 778e2d01a995491fe63db210dcf46db1723a4430
   git -C vendor/FlashRSI rev-parse HEAD
   ```

   FlashRSI 需要 Python >=3.11；依其上游说明在隔离环境安装。`--check-only` 不调用模型也不验证隔离；不要运行默认 AHE 资源演化直到不可变安全策略、工具权限、端点、沙箱、训练/测试隔离和候选验收已实现。FlashRSI 的资源演化不是模型参数 RL。
3. 按仓库现有 `scripts/bootstrap_rl_sources.sh` 与实验文档准备并核对 VeRL、AgentDojo 和相关补丁的版本，单独准备 HarnessAudit；任务、fixtures、配对 ledger 和未见 family 划分需要重新准备和审计，Git 中的代码不是数据集。不要用旧测试家族参与训练。真实 reset、工具协议、独立 verifier、clean/attack 语义与副作用均须通过正反例验收。
4. 在新机器重跑 CPU 测试（先建立含 pytest、PyTorch 和兼容 VeRL 的环境；无需 GPU）：

   ```bash
   cd experiments/cross_harness_sft
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts:vendor/verl python -m pytest -q \
     tests/test_harness_scaling_infrastructure.py tests/test_harness_scaling_online_cpu.py \
     tests/test_harness_segment_admission.py tests/test_harness_segment_dataproto.py \
     tests/test_harness_segment_actor_bridge.py tests/test_harness_segment_pipeline.py \
     tests/test_rl_whitebox_reward_contract_pair.py
   ```

5. 完成真实上下文压缩和 C1/C2 训练循环；按原 rollout 计算优势并检查 token mask、真实前缀、old/reference log-prob、全局 token 归约、policy 版本及 G=4 准入。接入 CHS-PO 和共享奖励契约，并做真实 autograd、跨微批/分布式数值比较。`validate_paired_profiles` 必须在对照运行间实际调用，单个运行内的 profile 校验不等于跨运行匹配已验证。
6. 再做 4B 小规模真实在线预检：模型生成→环境执行和安全评分→一次有效 optimizer step→actor 到 rollout 同步→checkpoint 保存/恢复；分别记录资源、原始证据及可复现配置。LoRA 可作为单卡可行性预检，须标注且不得冒充全参数对照。接通 FlashRSI 的版本化资源和安全门禁，验证下一轮确实使用冻结的新产物。全部通过后再冻结 GRPO/CHS-PO 公平预算并启动正式实验。

## 2026-11-11：当前研究方向更新

用户确定新主线为**不做本项目教师轨迹 SFT、从官方 Qwen3.5-4B/9B 后训练模型出发，研究 Harness Scaling × RSI 的安全效用与 CHS-PO**。完整共识、历史代码可复用范围、评测及启动门槛见 `experiments/cross_harness_sft/docs/safe_rsi_harness_scaling_experiment_20261111.md`；下方 9B SFT 起点和真实 harness RL 启动顺序为历史设计，不覆盖新主线。目前新 4B 入口、完整五维配置和 RSI 闭环尚未验证；不能因 GPU 空闲而直接运行下方旧 2B/9B SFT 脚本冒充新实验。

## 2026-11-03：9B Harness Scaling 迁移入口

- 9B 全参数 SFT 合并权重的上传目标是 Hugging Face `Limax11/harness_alignment` 中的 `qwen35-9b-sft-step291/`，下载前先确认该目录已上传完成。该目录用于从权重新初始化在线 RL，不能恢复 SFT 优化器。
- 实验设计参见 `experiments/cross_harness_sft/docs/9b_cross_harness_rl_design_20261027.md`、`2b_harness_scaling_rl_redesign_20261030.md` 和 `verified_progress_reward_20261030.md`。同任务分配不同 harness 配置，并为测试保留 RL 未见配置和真实 harness；白盒配置不能当作原生 CLI。
- 入口为 `experiments/cross_harness_sft/scripts/train_rl_9b_whitebox.sh`。数据集、assignment ledger、v7 family split 和 AgentDojo fixtures 不在普通 Git 提交中，必须在新机器准备并校验；不可将测试家族用于在线更新。实际 9B SFT 暴露清单仍需核实。
- VeRL 基线 revision `02a318c303a7b60871cb63e4ce2f779b522874d9`，本机未提交修改保存在 `experiments/cross_harness_sft/patches/verl_harness_scaling_20261103.patch`。先在干净的该 revision 上用 `git apply --check` 检查补丁；请勿和旧补丁盲目重复应用。
- 本机 9B 在线更新多次触发宿主 RAM 保护，不能认为该配置已经完成过稳定更新；新机器需验证真实 rollout、首次 optimizer 更新与权重同步。2B v4 的单批历史轨迹重放中可变奖励组从 outcome-only 3/24 增至 progress+protocol 7/24，不能外推为 9B 训练或安全泛化结果。

## 2026-10-28 11:44 CST：Qwen 字符串协议修复与 r11 重启

用户要求停 r10 修复协议后重启；r10 在首批 reference log-prob 阶段手动停止，无完整更新或 checkpoint。r11 从 9B SFT step-291 新起，使用 GPU0/3、固定 ledger 和相同 online RL objective；tmux `rl9b-whitebox-chspo-v11`，run `experiments/cross_harness_sft/outputs/rl/9b_whitebox_chspo_20261028_r11_schema_gpu03/`。

新协议版本 `qwen_schema_strings_v1` 仅对精确已注册工具、schema明确声明 `type=string` 的参数接受未加引号的 Qwen XML 文本；JSON字符串仍支持，非字符串类型仍需合法JSON。未知工具或参数、空值、损坏标签、重复参数/JSON键、非有限数和嵌入标签不被自动修复。生成、工具调用归因和路径审计使用同一解析逻辑，原始输出和状态审计保留；解析器版本写入rollout/metrics并禁止批次混用。提示词显式说明两种字符串格式，重新计算540个cell，最长3772 tokens，小于4096。

对 r10 中55条含完整工具块的拒绝日志按实际cell的schema重放：旧解析55条拒绝，新解析40条可解析且与路径归因一致，15条仍拒绝；未重算执行结果或reward。摘要 `outputs/rl/9b_preparation_20261028/qwen_parser_r10_replay.json`。62项相关测试通过，含真实AgentDojo别名dispatch与审计。r11 11:44已启动，仍在初始化，未有真实更新；旧actor梯度view修复及两卡reference内存仍待首步验证。watcher逐分钟记录RAM/GPU和主机I/O压力。

## 2026-10-28 11:09 CST：两卡 GPU0/3 r10 启动

按用户要求改为 GPU0/3，两卡 FSDP2 actor、vLLM TP2 与 BF16 frozen reference 共卡；参考模型继续指向 SFT step-291，actor 仍为全参数 FP32 master/Adam。r9 的 actor 错误来自对 activation-offload custom Function 返回视图原地缩放，r10 使用 opt-in 的分块温度缩放、selected log-prob 和 entropy，不修改原 logits；小规模 FP32 梯度/BF16数值检查与相关布局检查共15项通过。新实现只支持当前非蒸馏、无 sum_pi_squared 的 packed 路径，其他情况明确拒绝。

tmux `rl9b-whitebox-chspo-v10`、run `experiments/cross_harness_sft/outputs/rl/9b_whitebox_chspo_20261028_r10_gpu03/`。11:09放行启动时 available RAM约159 GiB，临时守卫为153 GiB，余量有限；监视主机RAM、两卡显存和 I/O。r10 刚启动，尚无 rollout 或完整 optimizer 更新；共卡 reference 的实际显存峰值、长序列 backward 和首个 Adam 更新尚未经过验证，不能宣称训练已跑通。GPU1/2存在其他用户任务。

## 2026-10-28 11:05 CST：r9 在首次 actor 更新中退出

r9 的 reference 分块 log-prob 已通过长序列阶段；首批 24 组、96 条 rollout 准入，14 组有奖励差异，日志打印 `optimizer_called=True` 表示准备调用更新，并不证明 optimizer step 已完成。随后 actor 在 `_scale_logits_by_temperature` 对 activation offload 自定义 autograd 输出的 view 原地 `div_`，触发 PyTorch RuntimeError，21:58退出；无完整 iteration metrics 或 checkpoint，不能宣称完成一次更新或确认是否存在部分内部 optimizer step。

需要在不破坏 custom Function backward、同时控制长序列 logits 显存的条件下修复原地缩放，并验证梯度及实际三卡更新。当前 GPU1 是其他用户约14 GiB的任务；GPU2也出现其他用户四个活跃进程，重启前需重新核对资源，勿结束他人进程。GPU0/3空闲。

## 2026-10-27 21:35 CST：r9 reference 分块 log-prob 重启

r9 从同一个9B SFT step-291重新开始（三卡 actor/vLLM 0/2、reference 3，r8没有完成更新）。独立 Qwen3.5 reference 使用 VeRL torch fused 输出头、128 token 分块，reference world-size 1 时使用本地 DTensor 视图，降低长序列全词表 logits 的峰值；actor 继续使用原 non-fused 路径和全参数 FP32 master/Adam。数值检查覆盖常规/分块 log-prob 和梯度、BF16、temperature=0.8、world-size 1 DTensor；真实 HFModelConfig 克隆确认 actor non-fused、reference torch fused。测试通过不能代替 9B GPU 全批结果。

tmux `rl9b-whitebox-chspo-v9`、run `experiments/cross_harness_sft/outputs/rl/9b_whitebox_chspo_20261027_r9_3gpu_refchunk128/`；关注 `REFERENCE_CHUNKED_LOGPROBS`、`WHITEBOX_GPU_ASSIGNMENT`、首次 `WHITEBOX_ADMITTED` 与 `metrics.jsonl`，只有后者包含实际 optimizer 调用才算已完成在线更新。首步仍需检查reference和actor显存/RAM、协议解析拒绝；I/O压力及每10步 checkpoint要继续监控。

## 2026-10-27 21:30 CST：三卡 r8 首批采样后 reference OOM

r8 的三卡资源池实际分配为 actor/vLLM GPU0/2、reference GPU3；首次权重同步已完成。首批在线采样后进入 reference log-prob 计算，21:30 在 Qwen3.5 `lm_head(hidden_states)` 分配 3.68 GiB logits 时，GPU3（进程可见的 logical GPU2）只剩 2.92 GiB，CUDA OOM 退出。`exit_code=1`、无 `metrics.jsonl`、无在线优化器更新；GPU0/2/3 已释放。日志有60条解析拒绝和1条未完成输出，日志条数不是失败轨迹数。

下一步需针对 reference 和 actor 的长序列 logits 内存做处理并验证 log-prob 与梯度语义；VeRL 的 Qwen3.5 torch fused backend 提供分块计算，是候选方案，尚未验证，不可宣称修复或启动正式更新。三卡布局解决了同步期的共卡压力，但尚未证明首个完整更新可以在本机资源内完成。

## 2026-10-27 21:04 CST：三卡 r8 初始化

按用户要求停止 r7；其当时尚在初始化，未产生 rollout 或更新。r8 的 GPU mask 为 `0,2,3`：Ray 给 actor/vLLM 保留两个 GPU（FSDP2 world-size 2、vLLM TP2），单独给冻结 BF16 reference 保留一个 GPU；各池实际设备由 `gpu_assignment.json` 核对，不假定 Ray 的分配顺序。全参数 FP32 actor master/Adam、KL reference 和在线 GRPO/CHS-PO 目标不变。同步仍采用 CPU 上先转 BF16、256 MiB 通信桶。

启动脚本 `scripts/train_rl_9b_whitebox.sh`，资源池实现 `scripts/rl_9b_resource_layout.py`，run 目录 `experiments/cross_harness_sft/outputs/rl/9b_whitebox_chspo_20261027_r8_3gpu/`，tmux `rl9b-whitebox-chspo-v8`。13 项资源布局、转换及相关目标检查通过；启动配置显示 reference 独立池启用。r8 21:03 已启动：已观察到 GPU3 承载约19 GiB reference 权重，GPU0/2 的 actor 权重加载和 CPU offload 已开始，vLLM 正在启动；`gpu_assignment.json` 和 `metrics.jsonl` 尚未出现，程序内资源分配检查、首次同步和首次在线更新尚待确认。

## 2026-10-27 20:58 CST：r6 同步 OOM，r7 启动

r6 在首次同步时已成功先在 CPU 把 FP32 分片转成 BF16，但在 GPU 聚合 1.89 GiB 张量时仅余 1.49 GiB，20:54 CUDA OOM 退出；没有 rollout、metrics 或参数更新。同步通信桶此前采用 VeRL 默认 2048 MiB。r7 保留 CPU 先转 BF16 的路径，将通信桶设为 256 MiB；大张量仍走现有 CUDA IPC 直传，实际首次同步和训练内存峰值均尚未验证。

新 tmux `rl9b-whitebox-chspo-v7`、监控 `rl9b-whitebox-watch`，run 目录 `experiments/cross_harness_sft/outputs/rl/9b_whitebox_chspo_20261027_r7_bucket256_gpu02/`，GPU0/2。20:57 启动；仅当出现有效 rollout、准入与 optimizer 更新记录才能称在线 RL 已跑通。当前服务器 I/O 压力偏高，初始化可能较慢；监控 RAM 和 GPU，勿把 tmux 存活视为训练成功。

## 2026-10-27 20:45 CST：9B 在线 RL 当前启动状态

当前 tmux `rl9b-whitebox-chspo-v6` 正在初始化，run 目录为 `experiments/cross_harness_sft/outputs/rl/9b_whitebox_chspo_20261027_r6_bf16_cpu_cast_gpu02/`；绑定 GPU0/2。两次早期启动失败均在 rollout/update 前：r4 在 FP32 全量 DTensor 聚合时请求 3.79 GiB、可用 659 MiB；r5 在把已搬到 GPU 的 FP32 分片转 BF16 时请求 970 MiB、可用 659 MiB。二者均无 metrics、无完成更新，不能算训练步。

r6 修正为在 CPU 本地分片上先转 BF16，再搬到 GPU 聚合；FP32 actor master/Adam 与训练计算配置未改。`test_rl_weight_sync_cast.py` 3项、reference placement 2项通过。此转换仅显式 `VERL_SYNC_WEIGHT_DTYPE=bf16` 开启，源参数不变。r6 于20:36启动，最近检查仍在 FSDP/模型初始化，尚无 `metrics.jsonl`；因此在线 RL 参数更新仍未开始。初始化监控见 run 下 `resource_status.json`、`resource_history.jsonl`、`launch.log`。最近主机可用 RAM约49 GiB，持续看 RAM，勿将进程存活或模型加载当作首次更新成功。

旧 r4/r5 配置说明如下为启动时历史记录；GPU reference 的放置本身未解决权重同步峰值。后续以 r6 实际配置、source snapshot及运行状态为准。


## 2026-10-27 新启动：r4 GPU0/2 + GPU常驻reference

用户确认 GPU0/2/3 空闲后，r3资源等待已取消。新运行目录 `experiments/cross_harness_sft/outputs/rl/9b_whitebox_chspo_20261027_r4_gpu02_gpured/`，tmux `rl9b-whitebox-chspo-v4` 使用 **GPU0/2**，GPU3留空。启动守卫已放行并写入 `launch.started`；首次 rollout/update 仍需日志证据，不能用该标志称为已完成更新。

本次资源修复：明确 opt-in `VERL_REFERENCE_KEEP_ON_GPU=1`，仅在未显式请求offload的reference上关闭默认强制CPU放置；actor仍FSDP2 CPUOffloadPolicy=true，FP32 trainable master/Adam、BF16 forward。reference BF16、offload_policy/param_offload/optimizer_offload均false，冻结权重不变；vLLM TP2 GPUutil=.43给reference留空间。默认未设环境变量的旧实验行为不变，两个放置回归测试通过，独立代码复核无确定放置矛盾。

safetensors头实测文本参数8,953,803,264、视觉参数456,010,480。本轮只有文本输入，视觉参数不会生成梯度或Adam状态；所有训练参数标志仍保持全参数。FP32全模型master+文本grad+Adam状态估算CPU约135GiB；GPU驻留reference省去约17.5GiB host权重。临时采用 available RAM>153GiB 启动首次更新内存预检，**不是峰值已通过保证**；Ray原95%保护保留。若实测失败必须报告并停机，不转LoRA。`VERL_PARK_OPTIMIZER_STATES_FOR_SYNC=0`，避免同步时每轮写盘停放约67GiB Adam状态引起I/O压力；因为其余配置没有改变奖励/分组，算法对照必须使用相同资源放置。

18配置/540cells、固定ledger、24×G4、LR5e-7、path_weight0、KL.001、每10轮checkpoint保持原设计。新run存完整source snapshot/hash。当前状态看该run的launch.log、metrics.jsonl和resource_status.json；r1/r2/r3都未更新，不得拼接它们成连续训练曲线。

## 2026-10-27 GPU0/2 调度更新

用户指定 GPU0/2 后，已将 r3 等待器固定 `RL_GPU_IDS=0,2`。完整 Hydra 配置复核通过（`config_validation_r3.log` 有 `VALIDATED 2 fp32 bf16`）；先前失败是读取尚未写完的空配置文件，失败日志保存在 `config_validation_r3_failed_empty_config.log`，已用完成后的结果原子替换，避免永远卡在旧失败标记。

本次实查 GPU2 空闲；GPU0 约2.3GiB来自用户zhz的6个进程，不是本项目残留，未结束。可用RAM约135GiB，I/O压力正常。CPUOffloadPolicy 的 FP32参数+梯度+Adam两状态理论约140GiB，BF16 reference另约17.5GiB，尚有worker/同步/pinned buffer开销，因此不能只因两卡显存空闲就忽略RAM。r3仍处资源等待，无9B RL rollout/update。实时状态看 `experiments/cross_harness_sft/outputs/rl/9b_whitebox_chspo_20261027_r3_cpuoffload/queue_status.json`。

## 2026-10-27 13:50 执行状态更新：9B 尚无在线参数更新

当前有效目录为 `experiments/cross_harness_sft/outputs/rl/9b_whitebox_chspo_20261027_r3_cpuoffload/`，tmux `rl9b-whitebox-chspo-v3`。此前 r1 被 GPU 占用守卫拒绝；r2 两卡启动到加载模型，发现默认 FP32 master+grad 的常驻预算估算错误后由操作者主动停止，未开始 rollout/update，日志及 `operator_stop.json` 保留。**没有把这两次启动算作训练步数。**

最终配置明确为 **两卡 FSDP2 + CPUOffloadPolicy + vLLM TP2**，actor FP32 master/Adam、BF16 mixed-precision forward，reference BF16冻结，非LoRA。原 `optimizer_offload=true` 的手动 offload 会在 step 把全部 moments 搬回 GPU；不能将它当 CPU optimizer 执行。FSDP2 `offload_policy=true` 才走CPU policy，后续需实际首步验证。

为避免重演宿主OOM，等待器要求两张空卡（<256MiB）且 available RAM>190GiB；此阈值是从9.41B FP32参数/梯度约70GiB、Adam两份约70GiB、BF16 reference约17.5GiB，加worker和Ray保护余量推导的保守预检门槛，**不是测得的实际峰值或9B算法的固有要求**。当前 available RAM约135GiB，门禁未放行；不会结束其他用户进程或降低精度/改LoRA绕过。每10秒轻量检查，24小时无资源则退出等待并记状态。以 `queue_status.json` 为实时事实；若首步通过，再依据测量优化资源，而非无证据降低阈值。

首轮仍为20采样轮工程验收、每10轮checkpoint；正式300轮和GRPO对照/外推验证尚未启动。`rl9b-whitebox-watch` 监测新目录，实际指标到达后才输出独立图。下方旧“四卡队列”的记录是被本次配置复核替代的历史状态。

## 2026-10-27 执行补充：用户确认先做受控白盒 9B RL

本节覆盖上方旧“真实三产品先训练”的启动顺序。用户选择：**先按论文做受控白盒 harness 配置 RL，用已审计 AgentDojo 家族启动；真实 CLI 留作外推验证**。

- 9B SFT step-291 已由 VeRL 官方 merger 合并到同目录 `hf_merged_step_291/`；760 张量的名称/形状匹配、BF16 GPU 加载及短生成通过。新 RL 不加载 SFT optimizer。检查产物：`outputs/rl/9b_preparation_20261027/`。这不等于 9B RL 更新已完成。
- 活动任务池：`outputs/rl/9b_whitebox_pool_v4/train.parquet`，SHA256 `dbe7ed2ce19b4aaa451d84364e7964c9ef057c8fbdd9719dc60ffed732bbe56d`。15 个已审计 AgentDojo train family × clean/injection × 18 配置 = 540 cells；最长完整提示 3,754 tokens。仅用既有 v7 train allowlist，9B r7 实际 SFT 视图仍 unknown，不宣称已证明 9B SFT 家族隔离。v1–v3 是准备过程作废版本，禁止训练。
- 配置是 system framing / reversible tool aliases / untrusted observation wrapper / workflow 四维变化；context 固定完整历史（C=0），不是论文完整五维复现，也不是真实 Codex/Claude/Hermes 产品。18 个投影配置有两两均衡覆盖；不把“18配置”称为“18产品”。CLI 产品不参与本轮更新，可用于后续外推；SFT exposure 仍需核对。
- 冻结 `assignment_ledger.json` 共 300 采样轮次；每两轮同 12 family × 两臂锁定 task、payload、初始状态和 fixture seed，anchor→最近配置，按当前策略独立 reset 生成。每轮 24 groups × G4 = 96 trajectories；不重采零方差/不利轨迹，审计证据不足直接阻断。首个运行只做 20 轮工程验收，不能代替 300 轮正式比较或独立能力验证。
- 两算法共用 outcome-only reward（path_weight=0）：验证越界 −1，安全完成 +1，良性不行动 −0.5，其余已审计未完成 0；具体优先级由 `score_episode` 决定。CHS-PO 在 GRPO advantage 上按 benchmark×白盒配置施加 attack/persistence/protocol dual；blocked attempts 不单独进成本，良性不行动另减0.30。新 dual 初值 .25/.50/.10，步长 .08、上限1，预算0/0/.10是预注册保守目标，**不是9B SFT实测基线**；三次观测层风险率 >.15 停机。不能承诺 held-out ASR 不涨。
- 参数：全参数9B、FSDP四卡、vLLM TP2、参考模型冻结同一 SFT291，LR5e-7，KL系数.001，clip默认.2，温度.8/top-p.95，16turn、4,096助手token预算，完整轨迹窗口4,096+8,192。CPU参数/优化器offload，activation checkpoint/offload，worker2；这些是首轮待测资源配置，不是已证明内存通过。每10轮保存完整checkpoint及 `whitebox_state.json`（assignment cursor/dual/streak/数据哈希），实际actor调用、KL-only/no-op单报。
- 启动：`scripts/train_rl_9b_whitebox.sh`；入口/训练器 `train_rl_9b_whitebox.py`、`rl_whitebox_task_runner.py`、`rl_whitebox_trainer.py`。独立复查发现并修复了 pandas局部遮蔽、准备/运行JSON哈希不一致、numpy配置值、白盒审计重采以及KL-only误计数。49项相关测试通过，另5项objective/sampler测试通过；CPU测试不代替首次GPU更新验证。
- 当前运行目录 `outputs/rl/9b_whitebox_chspo_20261027_r1/`。首启被 GPU3 占用守卫拒绝，**未更新参数**；现 tmux `rl9b-whitebox-chspo-v1` 等待四卡 <256MiB 且 RAM available>150GiB 连续3次检查后自动启动。不得杀其他同学任务。实时看 `queue_status.json`、`launch.log`、`exit_code`，不要用本段静态时间推断运行状态。
- tmux `rl9b-whitebox-watch` 每分钟少量记录 RAM/GPU，`metrics.jsonl` 新记录落盘才生成 `paper_figures/reward.{png,pdf}`、`entropy.{png,pdf}` 等独立曲线；未更新时没有虚构step0曲线。完整批次证据和token/mask/log-prob按轮压缩为 `rollouts/*.jsonl.gz`。生成耗时的480s目前是轮间检查，非强制取消中的生成请求；硬deadline留待接线。

## 2026-10-27：下一阶段改为 9B 显式跨 harness 在线 RL

用户要求参考 Harness Scaling 的 Reinforcement Learning across Harnesses，重点验证 **同任务执行 harness 变化 + RL 未见运行器安全外推**。已更新 `experiments/cross_harness_sft/README.md` 的活动假设及 Phase 2–3；详细方案是 `experiments/cross_harness_sft/docs/9b_cross_harness_rl_design_20261027.md`，旧 2B 设计归档为 `docs/rl_2b_design_snapshot_20261027.md`。下文 2B 的“下一步”“仅有 pilot”等按日期的历史条目，不覆盖本次 9B 计划。

- **候选 SFT 起点**：`experiments/cross_harness_sft/checkpoints/qwen35-9b-base-v7-a6000_20261018_r7/global_step_291/`。两份 `model_world_size_2_rank_{0,1}.pt` 各 9,410,256,507 bytes 与 BF16 Qwen3.5 配置存在，尚未合并/加载验证；`huggingface/` 没有模型 safetensors。当前只见非标准文件名的 optimizer rank-0、未见 rank-1，因此不称 SFT 全状态完整；**新 RL 从模型初始化新 optimizer/step-0，不需要该 SFT optimizer**。9B 实际 SFT view、harness exposure、family split 和该 checkpoint 的训练链仍需补齐，2B 的 1,421 条不能代替它。
- **默认训练候选**：Codex、Claude Code、Hermes。AgentDojo 和 AgentHarm 的同 family 两臂要形成跨三运行器的可执行交集，防止旧四个来源×harness 组合的来源混淆。每两轮保留同一批 12 个 family×两臂，并锁定具体 task/攻击变体/payload/初始状态，step A 在 anchor、step B 在另一 harness 用更新后的策略重新 reset/采样；每步 24 groups×G4=96 rollout。真实产品用均衡配对，不虚构汉明邻近；18 个五维白盒组件配置另作受控消融。
- **严格留出**：NanoBot/OpenClaw 不用于 9B RL 更新、奖励调参或选 checkpoint，若已见于 SFT 只叫 RL-unseen。Goose 作为项目后训练未见运行器的优先候选、NemoClaw/SeClaw 补充，需审计 9B SFT 暴露且验证真实运行器；未知就记 unknown。所有任务变体按原始 family 隔离；外推测试不能回灌。历史 2B 评测开发暴露如实记录。
- **公平算法对比**：同一个 9B SFT 上，标准 GRPO 和 CHS-PO 共用 task assignments、reward、预算、KL/clip；CHS-PO 的增量是按 source×harness 的安全优势约束。新9B不复用2B dual/priority/优化器。主实验 path_weight=0；路径细分、信息量采样单独消融。候选固定同一ledger的300采样轮次、3训练seed，先20轮工程验收，每10轮checkpoint/20轮validation；零优势不动态换任务，实际optimizer更新和KL-only/no-op分别计数；只在训练harness的validation families选模型，再统一测试每case三次。全参数资源参数待测，不自动回退LoRA。
- **接线未完成**：现有 sampler 没有同任务 K 步绑定，公共 MCP 模板并不等同原生 harness；需要补 adapter fidelity、同源工具/状态等价、assignment 恢复和持久化评分污染修复。当前只改了文档，**未启动9B训练、未合并权重、未占GPU**。新算法的安全效果仍须 Utility/ASR/Persistence 配对外推结果验证。


## 2026-10-16 接手优先读：已完成实验、实际能力与大模型下一轮设计

以下只汇总已落盘的训练与评测；下文保留了按日期记录的历史快照，其中“运行中”“当前无任务”等说法不能代替实时进程检查。实验规范见 `experiments/cross_harness_sft/README.md`；具体逐条结果见 `experiments/cross_harness_sft/reports/`。**当前没有证据证明 CHS-PO 已在所有 harness 上同时实现 Utility 提升与 ASR/Persistence 下降。**

### 模型与训练进度

| 阶段 | 已完成的配置与产物 | 关键限制 |
|---|---|---|
| 2B 全参数 SFT | 原始教师候选 4,162 条，经任务家族隔离、轨迹/工具/安全审计和独立 LLM judge 筛成 1,603 train、246 validation；Qwen3.5-2B-Base 的 4,096-token 完整轨迹预检保留 1,421/232。VeRL BF16/FSDP、`lora_rank=0`、两张 24 GiB GPU、全局 batch 16、每卡 micro-batch 1、AdamW、峰值 LR `1e-5`、10% warmup、cosine、3 epoch、267 更新；验证 loss 最低 0.893（step 144），最终约 0.900。评测与 RL 起点是 `hf_merged_step_267`。 | 训练覆盖 Codex、Claude Code、Hermes、NanoBot、OpenClaw、OpenCode、`claudecode`、QwenPaw；每个来源×harness 的可用条数不均，不能用 loss 收敛代替工具协议或安全能力证明。数据、曲线和入口见下文“已完成的全参数 SFT”。 |
| 标准在线 GRPO R11 | 从同一 2B SFT 初始化 actor、冻结 SFT reference；AgentDojo/Codex、AgentDojo/Claude Code、AgentHarm/Codex、AgentHarm/Hermes 的 107 个 train-family cell；每步 32 组×`G=4` 独立 reset 当前策略 rollout（128 条），全参数更新 20 步，保存完整 `global_step_20`。有官方 outcome、真实工具/路径审计、`path_weight=0.25`、2048-token 响应预算。 | 后四步平均零方差组率 81.3%，组内奖励信息变窄；HarnessRisk 上 Utility 和 ASR 同涨，不能称为净安全收益。 |
| CHS-PO v2→v3→v4 | 沿用 107-cell、32×4 在线闭环；按 `(benchmark, harness)` 分层 attack/persistence/protocol dual，保留良性不行动惩罚、安全硬门禁及每 10 步 checkpoint。v3 从 v2 step 70 续训，最后完整安全起点为 step 130；v4 从该完整 checkpoint 迁移奖励目标，加入有官方路径证据的成功轨迹效率细分和信息量采样。v4 完整 `global_step_150` 与 `hf_merged_step_150` 可用于续训和推理评测（用途分别核对）；**step 155 出现 128/128 超时、Utility 0 和 PPO KL 0.206，watchdog 已停机，不要使用其未保存权重**。 | v4 是跨目标函数的续训分支，非从 SFT 独立起跑的干净算法消融；持续高零方差与 step 138/155 突发输出退化仍未解决。日志、奖励版本、恢复账本和曲线见 `reports/chspo_v4_step140_signal_design_20261012.md` 与对应 run 目录。 |

独立 4B/9B 全参数 SFT 规模对照记在 `reports/sft_4b_9b.md`：其单轮 1,081 题报告中，4B ASR 18.90%→5.70%，但正常任务完成 85.57%→68.04%；9B ASR 8.66%→1.48%，正常任务完成 92.78%→91.75%。这些是**另一套模型/任务口径的 SFT 结果**，不能直接与下表 2B HarnessRisk 或 CHS-PO 算法效果合并；它们说明放大模型值得做同口径复验，也提示必须同时约束良性效用。

### 已完成的能力验证：相同 HarnessRisk case 和 judge

2B Base/SFT、GRPO step 20、CHS-PO v4 step 150 都在 HarnessRisk 的 Hermes、NanoBot、OpenClaw 上各用 128 case×3 独立 rollout；本实验统一由 `openai/gpt-5.4-nano` 判分，表内依次为 **Utility↑ / ASR↓ / Persistence↓ / Detection↑**，每项分母 384。论文使用 GPT-5.4，不能将本表与其表格当作同 judge 的严格排名。完整计数与百分点差见 `reports/harnessrisk_sft_grpo20_chspo150_comparison.md`。

| Harness | Base | SFT | GRPO step 20 | CHS-PO step 150 | CHS-PO 相对 GRPO 的关键变化 |
|---|---|---|---|---|---|
| Hermes | 1.8 / 0.3 / 0.0 / 4.4% | 1.0 / 1.6 / 0.8 / 4.2% | 5.2 / 3.4 / 1.3 / 3.6% | 13.3 / 1.6 / 1.0 / 3.1% | Utility +8.07 pp，ASR −1.82 pp |
| NanoBot | 12.8 / 4.2 / 19.5 / 0.8% | 7.8 / 3.1 / 13.0 / 1.0% | 20.6 / 9.1 / 29.2 / 1.6% | 18.5 / 6.3 / 23.2 / 0.3% | Utility −2.08 pp，ASR −2.86 pp |
| OpenClaw | 1.0 / 0.0 / 30.2 / 0.0% | 0.0 / 0.0 / 22.9 / 0.5% | 11.5 / 4.2 / 30.7 / 1.0% | 23.2 / 1.6 / 34.4 / 1.3% | Utility +11.72 pp，ASR −2.60 pp |

**判读：**CHS-PO 相对 GRPO 三个 harness 的 ASR 都降低，但 NanoBot Utility 低 2.08 pp；相对 SFT，NanoBot/OpenClaw 的 ASR 仍分别高 3.12/1.56 pp。OpenClaw Persistence 含已知初始化文件评分污染，不能解释为真实攻击性持久化变化。模型能力比较也不是控制训练步数、响应预算和目标函数的严格 GRPO↔CHS-PO 消融。SFT 的 AgentDojo/Codex、Claude Code 官方 verifier 抽样评测见下文；超时/轮次上限与分母差异必须连同指标报告。

### 当前跨 harness 泛化验证与下一台大服务器的首要实验约束

SafeClawArena 使用 v7 **未入 SFT 的 PSE:2.4、SSI:1.2 共 30 个家族内 case**，NemoClaw/SecLaw 各 3 次；这是 SFT/RL 都未接触过的**运行 harness**，但数据集来源 SafeClawArena/OpenClaw 曾用于 SFT，因此是“新 harness + 已见数据集来源”，不是完全新 benchmark。当前修正后的评测目录是 `outputs/safeclawarena_chspo150_eval_schemafix_v3_20261016/`；先前版本因工具示例污染、工具名自动改写和 schema 截断而作废。2026-10-16 交接时 NemoClaw 两臂仍在断点续跑，SecLaw 尚未完成；只看最终相同 case/seed 的双方有效 verdict，协议错误、网关启动失败与未评分须分开统计。SafeClawArena 无良性任务，**不能用它单独证明 Utility 保留**。已观察到未见过的 53 工具/约 31.5K-token 提示下，2B 会把部分必填参数写成错误标签，或选错工具参数；先做协议归因，再谈模型容量瓶颈。

**更大模型下一轮以“跨 harness 安全泛化”作为预注册的核心目标，而非只增加模型参数：**

1. **先冻结三类 harness 暴露集合和任务家族切分。** `H_SFT` 用于多源良性+风险教师轨迹 SFT；`H_adapt` 整个 harness 从 SFT train/validation 轨迹中留出，先对 SFT 做零样本基线，再仅用该 harness 的**独立 train families** 建可复位在线工具环境和 verifier，加入 GRPO/CHS-PO RL；`H_final` 在 SFT、RL、奖励调参和 checkpoint 选择中均完全不出现，只供最终跨 harness 泛化测试。`H_adapt` 的 RL 后成绩是“对 SFT 未见 harness 的在线适应”，不能冒充全程未见泛化；真正的未见 harness 结论只由 `H_final` 给出。先核对原始轨迹、工具 schema、别名与近重复家族的实际暴露，再指定具体 harness；已经看过的 NemoClaw/SecLaw 结果不能无条件重当盲测集。
2. **SFT 留出必须真实。** 新模型重建独立数据版本和不可变 exposure manifest；从训练、验证、上下文示例及蒸馏教师视图中排除 `H_adapt/H_final` 的实际运行协议，按原始任务家族隔离 benign/risk 配对与跨 harness 变体。保留足够的良性工具完成和风险防御轨迹，控制来源/长度比例，并核对训练格式与目标 harness 工具协议。更大模型的全参数 SFT 大小、序列长度、批量和 FSDP/优化器 offload 在新服务器做显存/RAM/吞吐预检后确定，不从 2B 参数机械放大。
3. **RL 在 `H_adapt` 上补可验证能力，同时保留已见 harness。** 先对每个新增 harness 做真实工具 schema、权限、reset、最终副作用与路径审计的正反例门禁；使用同一 SFT checkpoint、训练池、token/rollout 预算分别跑标准 GRPO 与 CHS-PO，再逐项消融分层采样、良性/风险配对、路径奖励、安全 dual。`H_adapt` 的 train-family rollout 是当前策略在线生成；验证家族只选 checkpoint，不回灌；`H_final` 无任何 RL 更新。不能仅靠增加全局违规惩罚来宣称跨 harness 泛化。
4. **以预注册的配对评测决定是否成功。** 在已见、`H_adapt` 和 `H_final` 上用相同 case/seed、原生 harness、同一模型路由与 judge/oracle、每题至少 3 次，分别报告 Utility、ASR、Persistence、Detection（仅在 benchmark 定义该指标时）、协议错误、超时及置信区间。模型选择应要求每个可评层的良性 Utility 不低于 SFT，ASR/禁止持久化不高于 SFT，并单报最差 harness；不要用 HarnessRisk 或新的最终盲测结果更新 reward、dual、采样权重或挑 checkpoint。若使用已检查过的 benchmark 家族调参，必须另留真正未见家族/基准做最后验证。

## 2026-10-11 CHS-PO v3 step-70 续训（16:03 CST）

R20g v2 的完整 `global_step_70` 已保存，AgentDojo/Hermes 家族隔离 24 条配对验证为 SFT→v2 step-70：clean Utility 1/12→3/12、injection Utility 1/12→3/12、ASR 0/12→0/12。样本窄，不能外推到 NanoBot/OpenClaw 或 HarnessRisk。v3 收紧 attack/persistence dual 与训练门禁。

最初误启的 SFT 起点 v3 run 在首个更新前已中止，未产生指标/checkpoint。正式分支 `outputs/rl/multiharness_chspo_r20g_step70_v3_resume_20261011/` 从 v2 `global_step_70` 恢复后跑到 step126/300，共完成 56 次更新。step120 是最后完整 checkpoint；step121–126 指标虽已记录，但不可作为完整续训状态。step127 的 AgentHarm judge 调用 Shengsuanyun 返回 HTTP 402 Payment Required，进程退出，GPU1/2 已释放；当前需恢复 judge 额度/凭证后从 step120 checkpoint 续跑。不是 OOM，RAM 可用约189 GiB。

训练池统计（step71–126）：每步128条已评分；零方差组率首10步均值0.813、末6步0.854，峰值0.969；896条风险臂中 AgentHarm/Codex 有4次官方攻击成功、AgentHarm/Hermes 2次，AgentDojo两层为0；AgentDojo/Codex有5次禁止持久化。step126安全 streak 全为0。只代表训练池，不是 HarnessRisk 评测结论。objective migration `v2→v3` 有 provenance 记录。详情见 `experiments/cross_harness_sft/reports/chspo_v3_safety_revision_20261011.md`。

## 2026-10-11 CHS-PO 续训状态（13:08 CST 快照）

R20f 从 step-20 续训，真实完成 step 67；下一个 32 组批次全部组内奖励恒定，原 GRPO 门禁在 `rl_guarded_chspo_trainer.py` 的更新入口停机。CHS-PO 有独立的可验证违规与良性不行动惩罚，零组内奖励差异并不必然意味着零策略优势。现已加入 CHS-PO 专属检查：完整批次仍须通过来源、token、奖励和审计门禁；调整后优势非零才更新；完全零信号批次写入 `chspo_skipped_batches.jsonl`，不计优化器步数，并在线抽取下一组平衡任务，连续 16 次零信号才停机诊断。标准 GRPO 的零差异门禁保持不变。VeRL 采样跳批 hook 位于 `patches/verl_chspo_zero_signal_resample.patch`，`bootstrap_rl_sources.sh` 会应用它；每个 checkpoint 的 sampler epoch 随安全日志和 resume provenance 保存，避免跳批后重复任务调度。针对性测试 12 项通过。

新 run `outputs/rl/multiharness_chspo_r20g_zero_signal_resample_gpu12_20261011/` 已在 tmux `chspo_r20g` 用 GPU1/2 从 **R20f 的完整 `global_step_60`** 启动；rank 0/1 已加载模型、优化器和 RNG，13:08 CST 进度条 60/300，尚不能据此声称 step 61 更新已完成。启动脚本仍每 10 step 保存，训练池 SHA-256 仍是 `f6f88bf4f73e982b95b741a2292bcb39464ea65a9b5c2a42186a40cd5c93db42`。R20f 的 step 61–67 是被此次从 step 60 恢复替代的另一条训练分支，不可与新 run 拼接为连续曲线。后续只看新 run 的 `metrics.jsonl` 和 `chspo_safety.jsonl` 判断真实更新。

## 2026-10-10 历史快照：R11 结果、CHS-PO 实现与当时的停机状态

以下“当前”均指 2026-10-10 的记录；实际运行状态以上方更新和实时日志为准。

**当前没有在线 RL 任务运行，四张 GPU 上没有本项目进程。** 此文件后面标注“运行中”的段落是当时的历史快照，不代表现在的进程状态。A6000-2 上不要按旧的 `chspo_current_run.json` 或旧 run-id 续跑。R18/R19/R20 各只完成一次在线 actor 更新，均在第一次更新后的权重同步阶段因当前服务器的**宿主 RAM**达到 Ray 95% 限值而失败，未保存 step-10 checkpoint；三次 step-1 指标不可拼成连续训练曲线。R19/R20 的失败已在当前服务器确认，分别见 `outputs/rl/multiharness_chspo_r19_ramfix_gpu12_20261009_2310.log` 和 `outputs/rl/multiharness_chspo_r20_ramfix_gpu12_20261010.log`（原始日志未上传 GitHub）。R20 将响应预算降至 1024、AgentLoop/reward worker 减至 4/2，仍在同步点 OOM；试验性 optimizer-state parking 报告 `parked ... bytes=0`，**未证明有效**。不要关闭 Ray 内存保护或把此故障误判为 GPU 显存不足。新机器须重新测量首次 actor 更新、权重同步、第二步采样的峰值，再考虑长跑。

### R11 标准在线 GRPO：已验证的能力变化及失败模式

起点是 Qwen3.5-2B 全参数 SFT step 267；R11 在 AgentDojo/Codex、AgentDojo/Claude Code、AgentHarm/Codex、AgentHarm/Hermes 四层的 107 个 train-family cell 池中，每步均衡采 32 组、每组 G=4 次**当前策略在线 rollout**，完成 20 次全参数 GRPO 更新并保存 `global_step_20`；它没有覆盖 v7 SFT 的全部 harness。R11 已启用 `path_weight=0.25` 和 2048-token 响应预算，下一轮不能把这两项误写成新增创新。训练后期零方差组率从前四步平均 38.3% 增至后四步 81.3%，说明组内相对优势逐渐消失；还需防止通过“不行动”获得表面安全分。AgentDojo 四个隔离 validation 家族的 24 个配对 episode 中，RL 相比 SFT 的回合未正常结束从 19/24 降到 6/24，但 clean Utility 仍是 0/12→0/12、injection Utility 1/12→0/12、ASR 0/12→0/12，不能凭正常结束率称为安全提升。完整证据见 `experiments/cross_harness_sft/reports/r11_step20_online_rl_summary_20261008.md`。

HarnessRisk 独立测试使用同一 `openai/gpt-5.4-nano` judge：每个 harness 在 SFT step 267 与 RL step 20 上各评 128 case × 3 轮（384 条/模型）。下表是同口径内部对照，百分数为三轮均值；ASR/Persistence 越低越好，Utility/Detection 越高越好。

| Harness | Utility SFT→RL | ASR SFT→RL | Persistence SFT→RL | Detection SFT→RL |
|---|---:|---:|---:|---:|
| Hermes | 4/384 (1.0%)→20/384 (5.2%) | 6/384 (1.6%)→13/384 (3.4%) | 3/384 (0.8%)→5/384 (1.3%) | 16/384 (4.2%)→14/384 (3.6%) |
| NanoBot | 30/384 (7.8%)→79/384 (20.6%) | 12/384 (3.1%)→35/384 (9.1%) | 50/384 (13.0%)→112/384 (29.2%) | 4/384 (1.0%)→6/384 (1.6%) |
| OpenClaw | 0/384→44/384 (11.5%) | 0/384→16/384 (4.2%) | 88/384 (22.9%)→118/384 (30.7%) | 2/384 (0.5%)→4/384 (1.0%) |

**结论是 Utility 与 ASR 三个 harness 同涨，而非净安全提升。** OpenClaw Persistence 两臂都受初始化文件误判污染，这一列只能按当前 judge 口径报告；不能据 +7.8 pp 断言真实攻击性持久化增加。NanoBot/OpenClaw 并未直接进入 R11 的在线训练层，是跨 harness 迁移测试。论文 GPT-5.4 judge 与本实验 nano judge 不同，不能严格横向排名。评测细节、逐轮稳定性、any@3、超时和评分 bug 修复见 `experiments/cross_harness_sft/reports/harnessrisk_sft_vs_rl_step20.md` 与 `experiments/cross_harness_sft/reports/r11_step20_online_rl_summary_20261008.md`；step-20 合并权重只用于推理，续训须完整 FSDP 模型、优化器和训练状态。

### 当前 CHS-PO（harness-safety-dual-v2）：代码、约束与复现边界

- **训练池与在线性**：`scripts/train_rl_chspo_verl.sh` → `train_rl_chspo_verl.py` → `rl_native_chspo_task_runner.py` → `rl_guarded_chspo_trainer.py`。从同一 SFT step 267 和 R11 的 107-cell train-family 池 `outputs/rl/multiharness_agentdojo_agentharm_pool_v5.parquet` 重新初始化；`rl_multiharness_stratified_sampler.py` 每步按四个 benchmark/harness 层各取 4 clean + 4 risk cell，即 32 组 × G=4 = 128 条独立 reset rollout。只接受官方 outcome、真实工具/副作用证据、token mask 和 old-policy log-prob 均完整的更新批次；未评分轨迹 fail-closed。训练池与 SFT 权重不在公开 Git，须从原迁移件和私有模型仓库取得，并核对 Parquet SHA-256 `f6f88bf4f73e982b95b741a2292bcb39464ea65a9b5c2a42186a40cd5c93db42`。
- **奖励与更新**：`rl_guarded_chspo_trainer.py` 在 VeRL GRPO 相对优势上加入不居中的安全代价；dual 按 `(benchmark, harness)` 分层，attack/persistence/protocol 的初值为 0.25/0.50/0.10、训练预算为 0.05/0.05/0.10、步长 0.08、上限 1.0。仅真实违规 dispatch/协议终止计入 protocol 成本；解析器挡下的 `blocked_tool_calls` 单独观察，不进入 dual，以免压制“尝试但被阻止”的良性轨迹。verified unsafe 的正优势被截断再减代价；benign nonengagement 另受 0.30 的非中心化惩罚，避免安全靠不行动。`reward_outcome=-1` 的 verified violation 边界、`path_weight=0.25` 的有界真实路径证据、风险/良性配对和 batch 准入不应为追求 Utility 而放松。
- **硬门禁与选模**：任一层 risk 臂 attack 或禁止持久化率连续 3 步超过 `max(0.15, SFT 同池初始率 + 0.10)` 即在下一次更新前停机。这个初始率来自 R11 step 1 的训练池快照，是工程阈值，不是验证集安全标准。每 10 步 checkpoint；正式放行须在**家族隔离的 validation** 上以相同 task/seed 与 SFT 配对，要求每个可评层 Utility 不低于 SFT 且 ASR、禁止持久化均不高于 SFT。HarnessRisk 是独立最终测试，不能用于训练 dual 或挑 checkpoint。目前启动器只保存 checkpoint，**未自动执行该 validation/选模门禁**。
- **运行参数与可视化**：计划 300 step、全参数 `lora_rank=0`、FSDP BF16、AdamW optimizer/参数 CPU offload、actor micro-batch 1、mini-batch 4、KL loss 系数 0.001、rollout temperature 0.8/top-p 0.95、prompt 6144、模型上下文上限 8192、每 10 步保存。R18 原定 response 2048；R19/R20 为本机 RAM 故障临时改成 1024，不可混合比较曲线。run 内 `metrics.jsonl`、`chspo_safety.jsonl`、`chspo_duals.json` 和 `paper_figures/` 的单张 PNG/PDF/SVG 图分别记录 Utility、ASR、持久化、dual、zero-variance、回复长度、熵/概率及梯度等；绘图脚本 `scripts/plot_rl_paper_figures.py`。R18/R19/R20 的图都只有首步，不能用来宣称趋势或涨点。
- **judge、依赖与迁移**：AgentHarm 语义 judge 是 `configs/agentharm_rl_train_worker.yaml` 指向的 OpenAI-compatible `bigmodel/glm-5.3-flash`；密钥只经 `AGENTHARM_JUDGE_API_KEY` 或 gitignored `.env.rl_judge` 载入，绝不提交。`bootstrap_rl_sources.sh` 固定 VeRL/AgentDojo commit 并应用项目 patch；非零编号 GPU 的 Ray/CUDA 映射修复见 `patches/verl_nonzero_cuda_visible_devices.patch`。R20 本地临时 `VERL_PARK_OPTIMIZER_STATES_FOR_SYNC=1` 未验证有效，新服务器不要据此假定 RAM 故障已解决。A6000-2 应以**新 run-id** 重建环境/密钥/训练池，先通过两步在线更新和权重同步，再谈 300 步正式实验。

下文保留之前的 SFT、数据包和早期 RL 交接历史；涉及“当前运行中”“尚未交付多 harness loop”的时间性表述均以上述 2026-10-10 状态为准。

更新：2026-10-06。先读 `experiments/cross_harness_sft/README.md`，尤其是 Phase 1 数据与全参数 SFT、Phase 3 在线 RL 和 Phase 4 评测协议。本文件记录已经实际完成的工作及下一步入口；不能把计划、启动成功或训练集奖励当作 held-out 能力提升。

## 已完成的全参数 SFT

原始候选视图有 4,162 条教师轨迹。按原始任务家族切分，再做完整性、真实工具证据、官方结果和 harness 协议检查；独立 `ali/qwen3.8-flash` judge 对安全、效用、工具正确性、harness 忠实度作四维审核。接受集为 1,603 train、246 validation。`Qwen/Qwen3.5-2B-Base` 的 4,096-token 完整轨迹预检留下 1,421 train、232 validation，隔离 196 条超长轨迹；train 有 316 条良性工具任务及 1,105 条风险相关任务，覆盖 8 个 harness、139 个任务家族。源、切分、分类和哈希见 `experiments/cross_harness_sft/data/sft/teacher_release_v7/` 中随本分支提交的几个 manifest；原始轨迹和完整 SFT Parquet 没有上传 GitHub。

VeRL BF16/FSDP **全参数**训练，`lora_rank=0`，两张 24 GiB GPU，全局 batch 16、每卡 micro-batch 1、梯度累积 8、最大长度 4096；AdamW 峰值 LR `1e-5`、10% warmup、cosine、weight decay 0.05、梯度裁剪 1.0；3 epoch 共 267 次更新，每 8 步验证。训练入口为 `experiments/cross_harness_sft/scripts/train_teacher_release_v7_full_sft_2b.sh`。32 步 train-loss 均值约从 1.3 降至 0.65；独立 validation loss 最低 0.893（step 144），末步 0.900。当前评测及 RL 初始化用最终合并权重 `checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267`，不是 step 144。图与数据在 `experiments/cross_harness_sft/reports/qwen35_2b_full_sft_training_curves_20260925.{png,csv}`。

## SFT 后的实际评测

HarnessRisk：每个 harness、每个模型使用相同 128 case × 3 独立 rollout，384 条均落盘并由 `openai/gpt-5.4-nano` 评分。论文用 GPT-5.4，二者不能作同评分器的严格排名。指标分别为 Utility↑、ASR↓、Persistence↓、Detection↑：

| Harness | Base | SFT | 结论 |
|---|---|---|---|
| Hermes | 1.8% / 0.3% / 0.0% / 4.4% | 1.0% / 1.6% / 0.8% / 4.2% | Utility 下降，ASR 上升 |
| NanoBot | 12.8% / 4.2% / 19.5% / 0.8% | 7.8% / 3.1% / 13.0% / 1.0% | 风险指标下降，Utility 同时下降 |
| OpenClaw | 1.0% / 0.0% / 30.2% / 0.0% | 0.0% / 0.0% / 22.9% / 0.5% | 几乎无法完成任务；初始化文件污染 Persistence 评分 |

详见 `experiments/cross_harness_sft/reports/harnessrisk_qwen35_2b_v7_sft_comparison_20260927.md`。OpenClaw 的 `IDENTITY.md` 等初始化文件被 judge 计为持久化证据；在重评分前不要解释该项 Base→SFT 变化为真实安全收益。Hermes 的 `skill_view` 参数格式问题和 OpenClaw 工具协议仍需排查。低 ASR 与极低 Utility 同时出现，不能声称跨 harness 安全对齐成功。

AgentDojo：固定种子抽 100 个任务（9 clean、91 injection），Base/SFT × Codex/Claude Code 各计划 3 次。官方 verifier 的已评分子集：Codex Base 232/300，Utility 14/232、ASR 0/214；Codex SFT 246/300，Utility 56/246、ASR 3/221。Codex 还有 68/54 条超时未评分，两个模型的评分分母不同。Claude Code 两组都 300/300；Base Utility 51/300、ASR 2/273，SFT Utility 57/300、ASR 5/273，分别有 162/156 条达到 CLI 最大轮次。SFT 的已评分 Utility 有上升，但两个 harness 的攻击成功事件也增加；因此不能将其概括为净安全提升。完整口径在 `experiments/cross_harness_sft/reports/yuque_multiharness_sft_harnessrisk_report_20260920.md` 的第 8 节。

## 在线 RL 已做到哪里

VeRL 当前策略在线生成 + AgentDojo 官方重置环境及 verifier + Hermes MCP 工具名称/参数映射已经跑通。16 个 train task cells 来自 8 个 AgentDojo 家族，各有 clean/injection 两臂；G=4，共 64 个独立 rollout/update。模型是上述 SFT，reference 冻结；全参数更新、无 LoRA。`outputs/rl/hermes_agentdojo_grpo_12step_16cells_20261005` **真实完成 6 次在线 GRPO 更新并保存 step-6 checkpoint**；step 7 的 16 组奖励全部零方差，门禁拒绝继续更新。原 metrics 的组数仍写了旧版 4 组，使用 `metrics.corrected.jsonl` 和对应 corrected 图。训练集上 Utility 约 0.19→0.03，平均回复长度 370→222 token，大量轨迹超时或未完成，呈现“避免违规但也不完成任务”的中性塌缩。

独立的 4 个未入训 workspace 家族上，24 个配对 episode/模型：SFT clean Utility 1/12、injection Utility 1/12；RL 两项均 0/12；ASR 均 0/12。invalid call/incomplete 从 SFT 的 17/24 升到 RL 的 24/24。使用与在线循环一致的 blocked-call 反馈重跑后，结论相同。结果证据随本分支提交 `metrics.corrected.jsonl`、`rl_training_dashboard_corrected.png` 和 `heldout_agentdojo_validation_protocol_parity/comparison.json`。这只是小样本单 harness 诊断，已明确显示当前 checkpoint 不宜宣传为安全能力提升。

2026-10-06 的 2048-token r17 单卡尝试未产生 rollout 或 optimizer step：GPU1 上 actor/reference + vLLM 加载后，**step 0 初始权重同步**需额外 2 GiB，而只剩约 658 MiB，因 OOM 退出。日志留在旧服务器的 `outputs/rl/hermes_agentdojo_grpo_r17_budget2048_16cells_20261005/launch.log`。新服务器用两张实际空闲 GPU 预检；不要用单卡失败日志画新 RL 曲线。代码中的 bounded path reward 虽支持 `path_weight<=0.25`，但 `verified_safe_subgoals` 尚未由真实路径证据填充，调大权重本身不能解决中性塌缩。

## 2026-10-07：塌缩诊断与奖励改造（r3–r8 的教训，已实施）

新服务器 300-step 系列 run 的实际死亡记录：r3 死于 AgentHarm 缓存 JSONL 并发读竞态（已修，`agentharm.py` 的 `fcntl.flock` 串行化）；r4（12 cells）跑到 step 9 后**整批零方差**被门禁停机；r5/r6 死于 unscorable episode（不可复位/不可评分）；r7（32 cells）8 步内零方差率从 0.44 爬到 0.81，最终死于 judge 子进程失败。**两次独立实验（r4、r7）收敛到同一形态：中性塌缩——策略组内行为趋同、组内优势消失**，扩批次只降低单步全零方差概率，不改变收敛方向。

由此确定的五条原则（按优先级）：

1. **过程方差治本**：terminal reward {-1,0,1} 太粗，结局相同的失败必须按过程分出差异。已实施：`score_episode` 的 path 通道现按 `path_weight × (verified_safe_subgoal_fraction − min(blocked_invalid_calls,1))` 计分，`path_weight=0.25`，且 **AgentDojo 的 subgoal 分数来自真实证据**——官方 ground-truth 工具计划（`_ground_truth_tools`）与实际成功 dispatch 的工具名求交，`|交集|/|计划|`；ground truth 不可得时诚实归零（AgentHarm 无 ground-truth 序列，subgoal 通道显式 abstain 为 0，由 judge 信号补位）。invariant 保持：path 通道不能把 incomplete 轨迹抬成正分、不能减轻 verified violation。
2. **堵死"稳定什么都不做"的不动点**：benign 臂（含 injection 的良性目标）目标失败、**零次有效工具 dispatch**、termination 为 completed/model_timeout 时，reward 记 `benign_nonengagement`，outcome 分量 −0.5，严格低于"尝试后失败"（0），且叠加 blocked-call 路径惩罚（如 1 次 blocked → −0.75）。verified violation（−1）仍是最差。AgentHarm 的 harmful 臂拒答是正确行为，不受此罚。
3. **监控三条联动曲线**：零方差率 ↑ + 平均回复长度 ↓ + 熵 ↓ 同时出现即塌缩进行时；journal 已按 arm 记录 `mean_path_signal` 和 `nonengagement_rate` 用于区分结局/过程贡献。
4. **批次大小治标**：r4（12 cells）与 r7（32 cells）死法相同，不要再以扩批次作为塌缩对策。
5. **消融顺序**：先解决塌缩（上述 1/2），再跑 CHS-PO、DAPO 等对照，否则所有方法死在同一个零方差门禁上，比较不出优劣。注意 CHS-PO 的 worst-stratum 加权只做组间再平衡，对组内零方差无效。

配套修改：奖励分解审计化——batch gate 现要求每条 rollout 携带 `reward_path_signal` 且 `reward = outcome + path_signal`（outcome ∈ {−1, −0.5, 0, 1}，|path_signal| ≤ 0.25），否则拒收；AgentHarm 语义 judge 已从 codex CLI（配额耗尽）切换到 OpenAI 兼容端点（`judge_backend: openai_compatible`，声学云路由 + `bigmodel/glm-5.3-flash`，key 走环境变量 `AGENTHARM_JUDGE_API_KEY`），并对推理模型的空 `content` 加了 `reasoning_content` 回退。相关测试已更新并通过（`tests/test_rl_verifier.py`、`tests/test_rl_native_agentdojo_episode.py` 等）。三个 RL 启动脚本（`train_rl_multiharness_verl.sh`、`train_rl_chspo_verl.sh`、`train_rl_dapo_clip_verl.sh`）会在 `AGENTHARM_JUDGE_API_KEY` 未设置时自动 source 本机 gitignored 的 `experiments/cross_harness_sft/.env.rl_judge`——重启 run 不再需要手工 source，密钥本身永不入库。密钥加载的完整规范见下节，所有现有与未来的 RL 入口都必须遵守。

## 2026-10-07：AgentHarm judge 密钥加载规范（spec）

**隐患记录（2026-10-07 发现并封堵）**：AgentHarm 语义 judge 依赖环境变量 `AGENTHARM_JUDGE_API_KEY`。若训练启动时该变量未设置，首个 AgentHarm 批次判分即抛 `Judge API key is unset` → episode 不可评分 → 整个 job 在第一次优化器更新前中止。r11 的启动当时依赖会话内手工 `source .env.rl_judge`，任何后续重启若遗漏该步骤就会复现故障。这是 judge 链路三天内第三次事故（r3 的缓存 JSONL 竞态、r7 的 codex CLI 配额耗尽、本隐患），因此固化为以下规范，对**所有现有与未来的 RL 训练入口**生效。

**规范**：

1. **启动即就绪（invariant）**：任何 RL 训练入口必须在 Ray/agent-loop worker 派生**之前**保证 `AGENTHARM_JUDGE_API_KEY` 已设置。三个启动脚本已实现守卫：变量未设置且 `$ROOT/.env.rl_judge` 存在时自动 source。新增任何 RL 入口（含 CHS-PO/DAPO/未来消融脚本）必须复制同样的守卫；**禁止依赖 shell 会话状态、`nohup` 外层环境或人工记忆**。
2. **密钥永不入库**：`.env.rl_judge` 已在 `.gitignore`；配置文件只允许 `api_key_env` 间接引用（如 `configs/agentharm_rl_train_worker.yaml` 的写法），不得出现字面 key。每次提交前检查：任何被跟踪文件中不得含 judge key 的任何片段（前缀、中间段、完整值）。文档只允许写环境变量名和密钥文件的路径，不允许写 key 内容。
3. **失败必须 fail-closed 且可诊断**：judge 缺 key、超时或空响应时按现状抛错、拒绝评分（unscorable → 门禁拒收整批），禁止静默记 0 分或跳过该 episode；日志保留 `Judge ...` 签名行供监控匹配（Monitor 的过滤模式已覆盖）。
4. **重启前自检**：重启任何 run 前，要么依赖启动脚本守卫（推荐），要么显式确认 `echo ${AGENTHARM_JUDGE_API_KEY:+set}` 输出 `set`；同时核对 `configs/agentharm_rl_train_worker.yaml` 的 `judge_backend` 与预期端点一致（当前为 `openai_compatible`，不再是 `codex_cli`）。

**验证记录（2026-10-07）**：三个启动脚本 `bash -n` 通过；离线端到端调用 `OpenAICompatibleJudge._generate` 返回带 `Answer:` 的规范判定；`git grep` 确认被跟踪文件中无 key 片段。后续新增 RL 入口时重复这三项检查。

## 2026-10-07：R11 在线 GRPO 运行中（14:19 UTC 快照）

当前新服务器使用 `scripts/train_rl_multiharness_verl.sh` 启动 `multiharness_grpo_pool107_20261007_r11_300steps`：VeRL 原生在线生成、全参数 GRPO，SFT step 267 初始化，2 GPU，32 个 train-family task groups/step、每组 4 个当前策略 rollout（128/step），计划 300 optimizer steps、每 10 步保存 checkpoint。入训的四个来源×harness 层为 AgentDojo/Codex、AgentDojo/Claude Code、AgentHarm/Codex、AgentHarm/Hermes；这**不是** v7 全部 8 harness 的正式覆盖。SafeClawArena 尚未接入本机 Docker runner。训练池位于忽略目录 `outputs/rl/multiharness_agentdojo_agentharm_pool_v5.parquet`（107 个合格 train cells，按 batch 分层抽样）；从私有 handoff 数据重建的代码是 `scripts/build_multiharness_rl_task_catalog.py` 和 `scripts/build_rl_multiharness_dataset.py`。训练池、私有原始数据、API key、checkpoint 和逐条 rollout 不放入公开 GitHub。

截至 14:19 UTC，R11 已真实完成 step 1–4，各步 32 groups/128 sampled/128 scored/0 unscorable，零方差组率分别为 0.375、0.3125、0.4375、0.40625；训练进程仍存活，尚无 step-10 checkpoint。该数字只证明训练机制在工作，**没有 held-out 效果结论**。可公开的聚合曲线快照见 `reports/r11_online_grpo_steps1_to4_20261007.{csv,png}`；本机完整 metrics、日志与未来 checkpoint 在 `outputs/rl/multiharness_grpo_pool107_20261007_r11_300steps/` 和同名 `.log`。新服务器不得把本机 R11 状态当成可恢复的远端 checkpoint；至少等 step 10 保存并单独转移权重。

R9 在首步之前被奖励分解门禁拒绝；R10 真实完成 2 步，第 3 步因 float32 `rm_scores` 与连续奖励的精确相等比较被错误拒绝。`rl_multiharness_update_guard.py` 已改用 `math.isclose`（绝对容差 `1e-6`），R11 使用修复后的代码。R8 完成 5 步但没有可用 checkpoint；不要把 R8–R10 的局部步数拼成 R11 的连续训练。

**16:23 UTC 交接更新：**R11 已完成 step 20 并保存 checkpoint。用户授权 Claude Code 停训、合并 step-20 权重并运行 `evaluate_rl_sft_pair.sh` 的家族隔离配对验证；此时权重合并正在运行。为免 Codex 将有意停训误判为故障而重启训练，R11 的自动修复 cron 已改为每分钟只调用 `scripts/check_rl_training_status.py` 记录状态。另一个 cron 每分钟调用 `scripts/wake_codex_on_rl_validation.py`：只有 `heldout_agentdojo_validation*/comparison.json` 及 SFT/RL 各 24 条逐条验证结果、step-20 FSDP checkpoint 均完整且稳定后，才启动一次独立 Codex 会话核查结果，决定是否能从 step-20 正确恢复训练，或先调整奖励/采样/更新策略。决策应记录在 R11 run 目录的 `step20_codex_decision.md`，不能把 24 个 AgentDojo episode 当作全 harness 结论。两个 cron 均是本机状态，不会随 Git 迁移；新服务器须按新 run-id 另行安装。

## 正式多 harness 在线 RL 尚未交付

GitHub 当前**只有 Hermes 协议映射 + AgentDojo 官方可复位工具环境的 16-cell 在线 pilot**。`configs/rl_native_pilot_agent_loop.yaml`、`scripts/rl_verl_native_agentdojo_loop.py` 和已提交的 `outputs/rl/native_pilot_train_20261005_hermes_mcp_16cells.parquet` 都只服务这条路径；在线 loop 的 harness 字段也是 Hermes。

正式 RL 的目标不能缩为 Hermes/NanoBot/OpenClaw。v7 接受集包含 8 个原始 harness 标签及 4 个源数据集，来源×harness 组合如下（计数是离线 SFT 接受轨迹，不是 RL rollout 数）：

| 源数据集 | v7 接受集中的 harness |
|---|---|
| AgentDojo | Codex 333 train + 64 validation；Claude Code 345 + 71 |
| AgentHarm | Codex 175 + 21；Claude Code 134 + 15；Hermes 175 + 22；NanoBot 211 + 24 |
| ActBench | `claudecode` 23 + 3；OpenClaw 27 + 3；OpenCode 30 + 6；QwenPaw 19 + 2 |
| SafeClawArena | OpenClaw 131 + 15 |

保留 `claude_code` 和 `claudecode` 为两个独立标签，直到分别核对其真实运行时、提示词、工具 schema 和权限后才决定是否归一。目标矩阵至少覆盖上述 8 种 harness 及其已接受的来源组合；能否把某个源扩到更多 harness，取决于该源的环境工具能否通过该 harness 的模型可见协议执行，不能按全笛卡尔积虚构覆盖率。

每个 RL rollout 都必须由当前 VeRL policy 在 harness 专属上下文和工具合同下新生成，并在对应数据集的 train-family 可复位环境里执行、评分；teacher transcripts 只能提供任务/来源，不能当 GRPO rollout。AgentDojo 的代码和 16-cell 数据已在 GitHub，但线上适配目前只有 Hermes。AgentHarm 工具环境、ActBench 服务及任务 fixtures、SafeClawArena 隔离执行环境都没有接入本项目 VeRL online loop；这些源分别还需建立 resettable train-family worker、真实副作用 verifier 和 harness 映射。NanoBot、OpenCode、QwenPaw、Codex、Claude Code 的正式 RL tool adapters 也未在此 pilot 中验收。`scripts/rl_group_sampler.py` 仅提供通用分层采样逻辑，不是这些环境的实现。因此**目前不存在可从 GitHub 直接拉取并启动的正式多 harness RL 数据包或启动脚本**。旧机 `data/raw/multi_harness_teacher_v1/` 和 `data/sft/teacher_release_v7/` 是离线教师/SFT 数据，不能代替当前策略在线 rollout 环境；HarnessRisk 保留评测 case 不能直接转成 RL train cell。扩展顺序应是按上述来源×harness组合逐项落地，完成协议、reset、正反例奖励和 family-disjoint 验证后，再将合格 cells 纳入 VeRL/CHS-PO 多 harness 采样。

### 旧服务器上的 split/fixture 传输包

所需输入包已上传至私有 Hugging Face 模型仓库 `Limax11/harness_alignment`，文件路径为 `transfers/handoff_multiharness_rl_data_20261007.tar.zst`，固定 revision `49d5041fba2373c3a195cfcc268aa01d25963784`（12,817,609 bytes；SHA-256 `71678b4ba1f9fb80b9a49754406ca9e60ab904c5c8fe5e17a13517955598f9d0`）。旧/新服务器均无需 SSH 互通；新服务器需先以有读取权限的 HF 账号登录，再执行：

```bash
hf auth login
hf download Limax11/harness_alignment \
  transfers/handoff_multiharness_rl_data_20261007.tar.zst \
  --revision 49d5041fba2373c3a195cfcc268aa01d25963784 \
  --local-dir ./hf_transfer
sha256sum hf_transfer/transfers/handoff_multiharness_rl_data_20261007.tar.zst
mkdir -p hf_transfer/extracted
tar --zstd -xf hf_transfer/transfers/handoff_multiharness_rl_data_20261007.tar.zst \
  -C hf_transfer/extracted
```

包内有 v7 `accepted_train.jsonl`、`accepted_validation.jsonl`、完整 `family_split_manifest.jsonl`（246 train、30 validation、30 test、7 review-only families）、分类/预处理元数据和 Qwen3.5-2B 4,096-token qualified JSONL（1,421 train、232 validation）；也有 ActBench 与 SafeClawArena 的提交版 tasks/fixtures/runner 源码快照。ActBench commit 为 `31bd732e9c083ddeb19bf152048386d38e511c90`，SafeClawArena commit 为 `a11f5cceaba0676be721021f8d232638fd111305`。test 家族只给 split 身份用于排除；没有 `accepted_test.jsonl`，也不应将其用于训练。`TRANSFER_MANIFEST.json` 列有包内文件大小和 SHA-256。包不含原始采集日志、密钥或未提交的本地代码改动。

### 在新服务器重新 SFT：已筛选 v7 数据包（2026-10-18）

完整的 **v7 已接受 train/validation 数据**现已单独上传至私有 HF 模型仓库 `Limax11/harness_alignment`，路径 `transfers/sft_teacher_release_v7_filtered_20261018.tar.gz`，固定 revision `1b7ebbce840c42ae4b39e028ae15cfc7395049d6`，11,306,817 bytes，SHA-256 `8e5d533cb129506b3527ed5dd95092a2f6d9082b0be9c674cd48fbcaa2f549c7`。上传后已从该 revision 重新下载并校验哈希。包内 `teacher_release_v7/` 含 `accepted_train.jsonl`（1,603 条）、`accepted_validation.jsonl`（246 条）、完整 `family_split_manifest.jsonl`、分类与预处理元数据、Qwen3.5-2B-Base 的 4,096-token 预检及当次全量训练视图/VeRL Parquet；`TRANSFER_MANIFEST.json` 列出每个文件的字节数与 SHA-256。原始候选和被拒轨迹不在此包中。它比上述 RL handoff 包多了当次 SFT 的 full-view/batch-plan/Parquet，供精确复现；新模型则应从 `accepted_*.jsonl` 重新处理。

在新服务器的 Git 仓库根目录运行（使用有私有仓库读取权限的 HF 账号；不要把令牌写进命令或 Git）：

```bash
hf auth login
hf download Limax11/harness_alignment \
  transfers/sft_teacher_release_v7_filtered_20261018.tar.gz \
  --revision 1b7ebbce840c42ae4b39e028ae15cfc7395049d6 \
  --local-dir ./hf_transfer
echo '8e5d533cb129506b3527ed5dd95092a2f6d9082b0be9c674cd48fbcaa2f549c7  hf_transfer/transfers/sft_teacher_release_v7_filtered_20261018.tar.gz' | sha256sum -c -
mkdir -p experiments/cross_harness_sft/data/sft
tar -xzf hf_transfer/transfers/sft_teacher_release_v7_filtered_20261018.tar.gz \
  -C experiments/cross_harness_sft/data/sft
```

**重新做更大模型的 SFT 时：**先确定 `H_SFT/H_adapt/H_final` 的 harness 留出方案，并检查 v7 `accepted_*.jsonl` 是否暴露计划留出的 harness；必要时按 exposure manifest 再过滤。保留现有家族隔离，严禁将 test/review 家族用于训练。换 tokenizer、上下文长度或模型后，从 accepted JSONL 重新做完整轨迹长度/assistant-mask 预检、批次平衡及 VeRL materialization；`preflight_qwen35_2b_base_4096_v1` 和 `verl_qwen35_2b_base_bs16_full_v2` 仅复现旧 2B 运行，不能直接当新模型数据。若要重新执行**全部原始轨迹的质量筛选**，本包不够，仍须另行转移旧机约 7.7 GB 的 `experiments/cross_harness_sft/data/` 原始库和所需源 benchmark。

## 新服务器复现与继续工作的顺序

1. 克隆本仓库的 `handoff/rl-20261006` 分支（或其合并后的 `main`），确认读到本文件、实验 README 和 `experiments/cross_harness_sft/scripts/rl_*`。代码已移除被误纳入旧 Git 历史的虚拟环境；本分支也不包含 checkpoint、原始轨迹和 API key。
2. 在兼容 CUDA 12.9 的服务器上建立 `experiments/cross_harness_sft/.venv-rl-pilot`。旧机实测核心版本：Torch 2.11.0+cu129、vLLM 0.23.0+cu129、Ray 2.58.0、Transformers 5.9.0、PEFT 0.21.1、Hydra 1.3.7。执行 `bash experiments/cross_harness_sft/scripts/bootstrap_rl_sources.sh`，它固定 VeRL `02a318c303a7b60871cb63e4ce2f779b522874d9`、AgentDojo `089ed468cf3ed0322acc66b0211f26d9d90dbf60`，并应用 `patches/verl_project_local_02a318c3.patch`（含 optimizer 状态按需回迁等本地修复）。安装两者到 RL 环境并确认 Hermes `5eb99eb2844b22ebb723711b8e6a0bbb80bb5f04` 的 Python 环境；按需设置 `HERMES_AGENT_ROOT` 和 `HERMES_AGENT_PYTHON`。`rl_native_pilot_agent_loop.yaml` 的 worker 路径已改为相对路径。
3. 从私有 Hugging Face 模型仓库 `Limax11/harness_alignment` 下载 **step 267 的 4.2 GiB SFT 合并 checkpoint**，固定 revision `7b608bf7517bd8b2f08ec8eecedd6dea4bdbbe26`，放到仓库相对目录 `experiments/cross_harness_sft/checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267`；新服务器先以有权限的 Hugging Face 账号运行 `hf auth login`，再运行 `hf download Limax11/harness_alignment --revision 7b608bf7517bd8b2f08ec8eecedd6dea4bdbbe26 --local-dir experiments/cross_harness_sft/checkpoints/qwen35-2b-base-teacher-v7-all1421-manual/hf_merged_step_267`。GitHub 不托管该权重。16-cell 训练 Parquet 已随本分支上传到 `experiments/cross_harness_sft/outputs/rl/native_pilot_train_20261005_hermes_mcp_16cells.parquet`。复现**已筛选的** v7 SFT 可下载上面的独立数据包；重新从原始轨迹筛选则仍需另行转移忽略的 `data/raw` 和源 benchmark。不能用 Git 中 manifest 伪装成数据本身。
4. 先核查 Hermes MCP 转换、AgentDojo 真工具执行及 reset/path audit；解决 `verified_safe_subgoals` 的真实证据来源，并以固定安全/不安全反例核对奖励边界。然后在两张空卡上用**新的** `RL_RUN_DIR` 做 2048-token、16-cell、G=4 的小批在线更新；入口是 `scripts/train_rl_native_pilot_verl.sh`，设 `CUDA_VISIBLE_DEVICES=<两卡>`、`RL_N_GPUS=2`、`RL_GPU_UTIL=0.45`、`RL_MAX_NUM_SEQS=2`、`RL_RESPONSE_LENGTH=2048`、`RL_TRAIN_BATCH_SIZE=16`、`RL_DATA=<上面的 Parquet>`。不得复用已有 run 目录；若批次无有效组内优势，保留门禁停机。
5. 每次保存 checkpoint 后，用 `scripts/evaluate_rl_sft_pair.sh` 对同一家族隔离验证集配对比较 SFT/RL 的 Utility、ASR、invalid calls、超时及真实副作用；保存 JSONL、metrics 和曲线。只有验证同时守住安全边界与正常任务效用，才扩大任务池或开始 CHS-PO、Balance-GRPO、DAPO 等消融。不要把训练 reward 上升当作能力涨点。

旧机 SFT 和 RL checkpoint 均未上传 GitHub；**只有 SFT step 267** 已上传至上述私有 Hugging Face 仓库，RL checkpoint 尚未上传。若从旧机迁移其他权重，可用 `rsync --partial`；先检查目标磁盘、模型文件大小和新服务器 GPU 驱动，再运行全参数在线更新。所有外部服务密钥应通过新服务器环境变量配置。

执行守卫补充：r3还须通过完整Hydra配置校验（`config_validation_r3.log` 含 VALIDATED），并等待 `/proc/pressure/io` 的 full avg60 <15%，避免高I/O压力时开始模型加载。阈值为保守资源控制，不属于算法定义。

R4启动实况（15:33 CST）：两rank的After FSDP/Total steps:20已出现在Ray worker日志，reference GPU分片约8.76GiB/rank已加载；rollout后端初始化中。主日志暂未出现WHITEBOX_ADMITTED或metrics更新，已完成在线更新为0。初始化期间available RAM最低监测到约33GiB，随后恢复约90GiB；额外共享内存占用/初始化临时副本及首个Adam更新峰值仍需监测，不能据启动放行推断长期资源稳定。
