## 2026-10-09：用户同意下一试为V10条件头修正版7轮

实现完成，尚未运行：实验 `pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_7ep_seed44`，独立 `pathvqa/train_v10_head_fixed_7ep.py` 默认epochs7/save5,6,7，启动脚本 `bash pathvqa/run_v10_head_fixed_7ep_seed44.sh`。共用训练/回调按真实预算保存并记录，原五轮入口默认不变。先绑定修正版五轮的20261009_102158_643548及原V1五轮seed44的20260926_2；新运行从零训练，无resume/权重加载。完成后CPU绘图→依次完整评估epoch5/6/7→各自对两既有基线配对10000/seed42（Overall/Yes-No/Free-form/what/where）；预定epoch7为主，5/6观察趋势，不择优。保留epoch_scores.json、最终报告/阶段状态/退出码/日志/ledger_fragment。独立产物 `pathvqa/outputs/v10_head_fixed/<7ep_experiment>_<timestamp>/`。

绘图兼容Windows/Linux和训练原目录trainer/trainer_state.json；横轴上限取实际max_steps，warmup从实际max_steps及报告ratio计算，分epoch及末轮窗口按真实训练轮数分组，无3075/93硬编码。最小CPU语法/Bash检查及既有五轮日志绘图兼容检查通过，未跑模型。仅授权这一试，GPU用户执行，失败即停，不重试/关机/Test/其他seed，结果回传后补两账本。

计划PathVQA model seed44/data seed42，从零训练7轮，沿用条件头修正版结构/初始化/分组基础LR/归一化/batch2累积16/clip1/3%warmup；线性衰减覆盖完整7轮，因此与原五轮轨迹不同。保存并完整Validation评估epoch5/6/7，预定epoch7主结果，其他轮仅轨迹诊断；与修正版五轮epoch5（58.6835）及V1五轮seed44 epoch5（59.3386）做图像簇配对。既有入口硬编码五轮及saved_epochs=[3,4,5]，需独立7轮入口及训练配置，不能只改名称。独立产物，不覆盖历史，复用训练诊断并按真实总步数/warmup绘图，禁止硬编码3075/93。尚未实现或运行，GPU用户执行；失败即停，无自动重试/Test/其他seed/关机。余下最后一试不自动安排；完成后补两账本，账本无独立commit/push。

## 2026-10-09：既有epoch3/4 Validation补评估已完成

用户回传同一条件头修正版seed44的epoch3/4/5 Overall57.1337/58.0284/58.6835、Free-form25.2712/26.2285/27.2176。已补两账本；本次补评估授权完成，覆盖下方待运行状态，固定epoch5主checkpoint不变。不自动延长训练、增大LR、增加seed或跑Test；剩余两次训练尝试仍待用户选择。后续若选择延长，应独立记录完整训练时长与scheduler/恢复方式，不能将重排衰减的从零训练与原epoch5简单视为同轨迹。

## 2026-10-09：条件头修正版已完成；剩余两试暂不指定

新增用户授权只评估既有epoch3、4：入口 `bash pathvqa/run_v10_head_fixed_epoch3_4_validation.sh`，精确绑定条件头修正版seed44的20261009_102158_643548产物，按3→4串行完整Validation，原生成评分协议不变；新目录epoch3_4_validation_<timestamp>，保存预测/summary/日志/退出码及epoch_scores.json。epoch5既有summary仅复用汇总，仍为原主结果，不以最高分重新挑主checkpoint。失败即停，不重训、不跑Test/其他seed/重试/关机，GPU用户执行；结果回传后补两账本。

曲线分析完成：SSH只读下载唯一已完成实验 `..._20261009_102158_643548` 的4份日志，本机纯CPU绘制loss+重建组LR、裁剪前/后梯度、LN/偏移尺度、有效融合地图/层权重/摘要RMS。产物 `pathvqa/outputs/cpu_curve_audit_20261009/`，完整口径与缺失项见report.md和两账本；不能用训练下降直接决定延长训练，不按梯度或地图熵单独判贡献。未跑模型/新实验/GPU状态命令、未修改服务器文件；剩余训练候选仍待用户决定。下方“尚未绘图”是历史状态，由本条覆盖。

用户回传PathVQA seed44固定epoch5 Validation Overall58.6835：对原V10 +0.9746pp、配对CI[+0.1117,+1.8222]；对V1 -0.6551pp、CI跨零，where探索性下降5.3790pp。两账本已更新。本条覆盖下方同名尚未运行状态，授权单次实验结束；不追加Test/seed/重试。接下来讨论训练曲线辅助分析，先读已有trainer/trainer_state.json和v10_diagnostics.jsonl；目前只有汇总报告，没有实际轨迹，尚未绘图或实施脚本。余下两试待用户选择，不擅自安排LR扫描或结构修改。

## 2026-10-09：已授权单次V10条件头修正版，代码准备、尚未运行

实验 `pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44`。仅组合三项：融合地图读取共同Value后的2560维摘要加可学习LayerNorm；Meta-Net两层权重/bias均为默认Linear初始化；Meta-Net LR3e-4、新LN LR1e-4。原V10模型及存档不改；薄子类在原V10初始化后，用同seed独立CPU随机流重建默认头，共有参数（含Meta-Net第一层）逐张量校验不变。新config/weights/backend独立，预计参数1,691,043（原1,685,923＋LN5,120），启动逐组实数核验。

沿用原五轮协议：44/42、batch2/accum16、3% warmup/linear、clip1、归一化False、保存3/4/5，固定epoch5完整Validation6259题/832图。独立入口 `bash pathvqa/run_v10_head_fixed_seed44.sh`；原V10 seed44依据完整report/config/协议及57.7089分数定位唯一产物，不按最新目录选择；可用 `--original-v10-run` 明确指定。另绑定原V1 seed44目录20260926_2（59.3386）。分别使用既有预测图像簇配对10000/seed42，输出Overall/Yes-No/Free-form/what/where；组合实验不能分别归因三项。

保留已有诊断并增加summary_pre_ln_rms/summary_post_ln_rms；首批梯度为裁剪前，Trainer grad_norm为裁剪前总梯度，on_pre_optimizer_step各组为裁剪后。复用原真实Train预检及保存重载/cache检查，不增加测试套件。仅本机语法/参数算术CPU检查，真实张量初值/梯度核验随用户启动的GPU预检执行，助手未启动。产物 `pathvqa/outputs/v10_head_fixed/<experiment>_<timestamp>/`；失败即停，不重试、不关机、不跑Test/其他seed/剩余候选。结果回传后更新两账本和计划，文档不单独提交。

## 2026-10-09：CoCoOp代码核对后的三次候选（仅讨论，未授权实施/运行）

已核对本地CoCoOp-style：post-merger原生视觉token均值→2560/160/2560 ReLU Meta-Net→共享偏移加到前置P20；无问题条件/Visual18/摘要LN，Meta-Net默认Linear初始化、LR3e-4。V10有Visual18，输出Normal(0,1e-4)/bias0、Meta-Net LR1e-4、固定5轮；CoCoOp历史PathVQA是3轮且累积归一化历史证据不足，不能仅归因地图。PathVQA同Validation三seedCoCoOp56.3136±1.1367，V10为56.7556±0.9428；不能混用CoCoOp Test56.9529或单seed57.4053宣称V10均分更低。

优先候选：1）V10仅在融合后的2560维摘要、Meta-Net之前加LayerNorm，其余固定；2）在候选1基础上采用CoCoOp的Meta-Net默认输出初始化与LR3e-4作为一个明确的优化配置组合，不能分别归因初始化/LR；3）在前两项选定的明确参考配置上加入均匀全局池化路径，地图为(1-lambda)/N+lambda*sum(beta_l*a_l)，lambda=sigmoid(s)，初值0.1，融合摘要后单次LN及同一Meta-Net，保留Visual18，除混合门外不新增第二Meta-Net。lambda=0对应带LN且保留Visual18的CoCoOp-style结构极限，不等于历史原CoCoOp完整协议。三项均为PathVQA seed44、固定5轮Validation讨论候选，无自动训练/关机/其他seed；最终执行定义须等用户选择，不视为已排入执行队列。

更换层位7/15/23（自然第8/16/24层）暂不优先：目前没有证明5/11/17错误，且只影响Key/地图，不改变共同Value仍来自最终merger。原CLIP CoCoOp输入是L2归一化后的全局视觉向量、用于类别相似度分类；本地为未归一化post-merger token均值和生成式CE，只有条件共享偏移的思路迁移。相关结论应保持版本与任务边界。

## 2026-10-09：PathVQA V10三seed队列已完成，自动关机失败

用户回传固定epoch5 Validation：44/45/46 Overall57.7089/55.8236/56.7343，均值56.7556±0.9428，对同seed V1平均-2.1036pp，三项Overall配对CI均低于零。下方“尚未运行”为历史准备状态，由本条覆盖；已补两账本，真实时间戳产物路径/训练元数据待补。授权队列已经结束，不增加seed、重试、调参或其他数据集。自动关机实际调用报Exec format error，未成功，须由用户确认服务器状态并处理；助手未执行GPU或远程关机。

## 2026-10-08 用户扩展授权：PathVQA V10三seed串行，结束后自动关机

实现状态：代码已准备，尚未运行。复用 `slake.train_v10` 的模型、优化器、五轮保存和预检，仅新增 PathVQA 数据/Train图像读取以及 model seed 参数；独立入口 `bash pathvqa/run_v10_seeds44_45_46_shutdown.sh`。严格44→45→46，每项完整 Validation 后保存同seed V1配对统计（10000/seed42），全成功才输出三seed mean±sample std(ddof=1)。产物 `pathvqa/outputs/v10/suite_<timestamp>/`，各seed独立子目录，保存 seed_results、final_report、suite_status、ledger_fragment 和逐阶段日志。失败退出码与后续未运行seed落盘后停止。

关机延迟明确为600秒：报告与状态落盘后启动脱离会话的纯CPU计时子进程，实际等待十分钟才调用 `/usr/bin/shutdown`，不向可能忽略延迟参数的AutoDL封装传 `+10`。普通Python队列入口默认不关机；只有上述专用脚本显式启用。关机安排/调用失败另存 shutdown_status.json 和 shutdown.log。助手没有启动GPU或实际关机，结果回传后补两账本。

本条覆盖下方原单seed44及不自动关机设置；V10结构、初始化、学习率和五轮训练协议不变。新增完整队列model seeds44/45/46，data seed均42，各自从零训练并固定epoch5完整PathVQA Validation，不跑Test、不择优、不重训基线。实验名分别为pathvqa_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44、seed45、seed46，按44→45→46严格串行，各自独立输出。

逐seed配对同seed原五轮V1：44绑定pathvqa/outputs/visual_selection_prefix/pathvqa_v1_norm_fixed_5ep_seed44_20260926_2，45绑定pathvqa/outputs/visual_selection_prefix/pathvqa_v1_norm_fixed_5ep_seed45_20260926，46绑定pathvqa/outputs/visual_selection_prefix/pathvqa_v1_norm_fixed_5ep_seed46_20260926，均读取eval_validation/epoch_5既有预测并核验身份，不按最新目录猜测。保存逐seed图像簇配对CI与Overall/Yes-No/Free-form/问题类型；成功后汇总三seed均值±样本标准差（ddof=1），不选最高seed作为主结果。

用户明确授权本次串行入口结束后自动关机。入口由用户亲自启动，助手只本机准备代码/CPU检查/相关提交推送，绝不直接执行任何GPU或关机操作。成功跑完全部训练、评估、配对和汇总后先落盘总报告/日志/退出码再调用已有AutoDL关机方式。失败仍按既定规则停止，不自动重试；记录失败seed、未运行队列和错误退出码后也关机，避免空闲计费；关机调用失败另行记录。检查stdout重定向/pipefail与退出处理，防止漏记失败或报告尚未写完就关机。结果回传后更新两账本和计划，不做仅账本提交。

## 2026-10-08 用户授权：PathVQA V10单seed五轮Validation

计划实验pathvqa_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44，尚未执行。完整复用已跑通的SLAKE V10结构、共有初始化及V1五轮训练协议，不调参、不改地图/层权重/Meta-Net，仅迁移到PathVQA。model seed44/data seed42，batch2/accum16，save3/4/5，固定epoch5完整Validation6259题/832图，不跑Test或其他seed。主对照为pathvqa/outputs/visual_selection_prefix/pathvqa_v1_norm_fixed_5ep_seed44_20260926_2/eval_validation/epoch_5（59.3386），做Overall/Yes-No/Free-form/what/where图像簇配对bootstrap10000次seed42；只读取既有基线预测。独立输出，不覆盖SLAKE或原V1；本机实现/CPU检查并提交推送，所有GPU检查、训练、评估由用户亲自启动，失败即停，无重试或自动关机。结果回传后更新两账本及计划，不做账本独立提交。既有CoCoOp可作三轮历史参考，须保留协议差异，不因本次自动安排其重训。

## 2026-10-08 用户授权：V10在SLAKE单seed试验，训练沿用五轮V1

> **2026-10-08 V10已完成，用户回传固定epoch5完整Test及配对结果：** seed44，1632/2094，Overall77.94，OPEN73.45/CLOSED84.69。对同seed五轮V1 +2.3878pp，图像簇95%配对CI[+0.8889,+3.8911]；对三轮CoCoOp +0.9074pp，CI[-0.3792,+2.2212]。完整记录已补EXPERIMENT_RESULTS.md和result.md；下方“未运行/仅计划”均为此前准备状态，由本条覆盖。简化头部在此配置下有效，不证明64维冗余或梯度阻塞，未确认超越CoCoOp；五轮/三轮及Visual18差异保留。实际train_report与激活/梯度轨迹仍待回传。此次单次授权已完成，不自动安排其他seed、数据集或新变体，无GPU操作由助手执行。

> **实现准备完成，真实训练/Test尚未运行：** 独立`slake/visual_selection_v10.py`、`visual_selection_v10_interface.py`、`slake/train_v10.py`；新config/weights文件名和backend`v10-weighted-map`，原V1不变。入口`bash slake/run_v10_weighted_map_metanet_seed44.sh`，先绑定两条精确基线及9834有效Train/2094 Test，用户启动后先CPU张量检查，再真实Train batch的梯度/单次视觉与LLM/labels/保存重载和cache预检，通过后从头五轮→固定epoch5全语言Test→相对CoCoOp与V1的10000/seed42图像簇配对。失败即停、不重试、不关机。独立输出`slake/outputs/v10/runs/`；预检查放`prechecks/`，不当作训练结果。
>
> 已验证：本机4项NumPy/AST数值和协议测试及Python编译；服务器GPU完全不可见、内存载入本机代码的2项CPU mock张量测试，逐组实数1,685,923，共有28个张量逐一等于同seed原V1，CPU RNG相同，输出std0.0000998816/bias0，混合图像grid概率守恒/层门控与地图梯度/独立save-load通过。只读baseline预核验通过，CPU产物`slake/outputs/v10/prechecks/slake_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44_20261008_173524_696069`；未加载真实骨干、未使用GPU或进行训练/正式推理。新头用CPU独立generator状态，不调用全局torch.manual_seed重置CUDA；旧LN/投影/输出头只在构造原V1保留初值时临时存在，最终模型/优化器/存档彻底删除。

状态：仅计划，尚未实现或执行。本条覆盖本轮先前提出的三轮训练、Meta-Net学习率3e-4及CoCoOp初始化建议；用户明确要求训练配置沿用V1。原V1和历史基线保持冻结，本次仅授权以下单次V10，不自动增加seed、重试、其他数据集或参数搜索。

- 实验名：slake_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44；SLAKE官方完整中英train，从零训练，model seed44/data seed42，保存epoch3/4/5，固定epoch5完整Test（2094题），不跑Validation、不择优。独立产物目录拟用slake/outputs/v10/，不得覆盖既有结果。
- 结构：保留V1的P20、索引17的S8+Av10 Visual18、128维真实问题编码、索引5/11/17的独立Q/K地图、每图grid_thw概率合并及共同最终merger Value。保留问题条件层门控beta=softmax(layer_gate(u))。融合地图a=sum_l beta_l * merged_map_l，直接读取z=sum_j a_j * V_j；不再softmax，不额外除以3。删除三套Value LayerNorm、2560->64投影、分块拼接及旧192->2560输出头；改为一套带bias的2560->160->ReLU->2560 Meta-Net，输出共享偏移加到20个P20位置。原生视觉与DeepStack路径及真实问题mask不变。
- 训练：沿用V1的batch2/累积16、max_length2048、workers2、bf16、AdamW/weight_decay0/clip1、3%warmup及跨完整五轮的线性衰减。P20 LR0.3，S8 3e-5，Av10 1e-4，问题编码/QK/层门控/新Meta-Net均1e-4。P20 embedding-row与Visual18 Normal(0,0.02)初始化沿用V1；Meta-Net第一层常规初始化，最终输出层Normal(0,1e-4)、零bias，保留小幅非零初值。显式model_accepts_loss_kwargs=False，Accelerate累积1，不额外手工除loss。
- 预计参数：1,685,923（P20 51,200，S8 8,192，Av10 10,240，问题编码345,088，地图448,896，层门控387，Meta-Net821,920）；实现后须逐组审计，不把预估当运行实测。独立method/config/checkpoint及推理加载接口，禁止残留旧头参数进入优化器或存档。
- 比较：主对照为既有CoCoOp seed44完整Test77.03，绑定slake/outputs/cocoop/slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44_20261008_143843_694315/eval_test/epoch_3；历史参考为V1 seed44五轮Test75.55，绑定slake/outputs/visual_selection_prefix/slake_v1_norm_fixed_5ep_seed44_20260928/eval_test/epoch_5。保存预测、Overall及OPEN/CLOSED/KVQA/VQA/EN/ZH，官方图像img_name簇配对bootstrap10000次/seed42。披露V10五轮且有Visual18、CoCoOp三轮且无Visual18，不把胜负全部归因于地图。只使用既有基线预测，不安排基线重训。
- 执行边界：本机编辑/CPU检查/相关代码commit及push；服务器仅用户运行source /etc/network_turbo后git pull --ff-only同步。所有GPU预检、训练、推理与评估由用户亲自执行；助手不启动。准备一条独立启动命令，失败即停，无自动重试或关机。结果回传后同任务更新EXPERIMENT_RESULTS.md、result.md与本计划，文档不做单独提交。

> **2026-10-08 SLAKE CoCoOp完整Test评估已回传：** 2094题1613正确，分项及评估路径已补两账本。复用既有epoch3，无新训练；仅配对统计内容待补。冻结V1/不自动加seed；计时聚合输出token数疑含P20，效率表使用前先CPU审计。

> **2026-10-08 SLAKE CoCoOp seed44 Test成绩已回传：** 三轮77.03，同seed V1五轮75.55；结果已记账，不重训。分项/配对CI和最终评估路径待回传；45/46未排期。本轮不据Test调参或改变冻结V1。

## 2026-10-08 用户要求SLAKE CoCoOp沿用PathVQA正式Test的参数

> **用户授权继续推理，准备仅评估入口：** `--evaluate-run slake/outputs/cocoop/slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44_20261008_125232_604532`严格核对完成训练的身份/协议/epoch3 checkpoint，只评估2094题完整Test并配对V1；绝不进入训练分支。有效训练计数复用既有split/空答案过滤（9834，排除qid1622），保存排除清单，不改变监督或已训权重。新评估和状态放独立目录，保留原失败日志，GPU仍由用户亲自执行，无关机。

> **运行状态更正：训练已完成，Test被启动器计数核验阻断。** 用户运行`..._20261008_125232_604532`完成924步/epoch3，全部checkpoint存在；9835原始条目包含空答案qid1622，原loader实际训练9834，启动器未复用过滤口径。已CPU只读确认，无GPU操作、不重训、不直接绕过断言。后续应仅修正有效样本审计并准备指定epoch3的评估续跑入口，由用户执行；本次诊断未实施修复。

> **CPU核验已通过：** 本地两项无torch导入的协议/配对统计测试及Python编译检查通过；SSH只执行标准库/纯评分的`--precheck-only`，真实历史报告、完整中英Train/Test图片存在性、V1身份及2094题归档预测/参考答案/评分一致性通过。CPU产物`slake/outputs/cocoop/slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44_20261008_124627_689238`只有预检查记录，不是训练结果，不可当作已运行实验。GPU训练/推理仍等待用户执行。

> **实现状态：代码已准备，尚未运行。** 独立入口 `bash slake/run_cocoop_style_seed44.sh`，实验 `slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44`；启动先只读核验历史报告/配置/日志及明确绑定的V1，失败即停、不重试、不关机。2026-10-08 SSH只读确认历史报告873120参数、LR0.3/3e-4、44/42、epoch3，日志WD0/warmup0.03/linear；历史batch覆盖值、执行commit、库版本及归一化未记录，未知项不当作已验证一致。原始引入源码7b94673的batch2/累积16/max_length2048/workers2/AdamW/bf16是源码旁证，不等于历史实际运行证明。V1精确参照为`slake/outputs/visual_selection_prefix/slake_v1_norm_fixed_5ep_seed44_20260928/eval_test/epoch_5`，2094题、75.55、三轮对五轮预算差异必须保留。新运行归一化False、完整中英Train与epoch3 Test；结果及失败保存在独立目录与ledger_fragment，用户回传后立即追加两份Windows账本。助手未启动GPU操作。

只准备/补SLAKE原CoCoOp-style seed44/data42，三轮固定epoch3全语言官方Test；不是五轮方案，不自动加45/46。以PathVQA正式Test对应历史P20/H160训练定义为模板：冻结同Qwen3-VL骨干，P20 embedding-row初始化，原生post-merger视觉token均值，正常初始化2560->160->2560 ReLU Meta-Net，共享P20偏移、位于完整chat前，无Visual18/问题条件；873120参数。P20 LR0.3/MetaNet3e-4，batch2/累积16，max_length2048，AdamW betas0.9/0.999 eps1e-8 WD0，warmup3%/linear，clip1，bf16，workers2。核对历史train_report或实际日志覆盖值，不能只用当前默认值宣称完全一致。

新入口使用已验证的累积归一化修正，并记录Trainer/Accelerate版本与实值；PathVQA旧基线归一化历史若无法证实一致，明确为协议差异，不照搬已知bug或悄悄声称逐位复现。只改SLAKE数据/官方评估，保持原问句和中英样本；不跑Validation，不按Test择优或调参，不自动关机。由用户执行全部GPU操作，工作AI本地实现/CPU检查/提交推送/提供独立命令。精确参数/初始化/runtime审计、Test分项和配对现有V1 seed44（固定epoch5，75.55，轮数不同须披露）均输出；结果完成后记双账本。代码和产物本轮尚未创建。

> **2026-10-08 更正与论文范围：** 电气LoRA-r8为3轮/seed47，V1同为seed47；此前seed44及V1 seed待补状态被此条覆盖。论文用汇总移除退休方法与开发历史，后续正文/表格不再引用。主账本保留历史以追溯。未新增训练或调参。

> **2026-10-08 电气V1成绩已回传：** 675/954=70.75，完全沿用PathVQA配置。电气LoRA-r8补齐3轮/seed44。电气V1不再列为无成绩；精确seed/运行名称/输出路径/报告及配对统计仍待补。不新增训练或调参。

> **2026-10-01 五任务已全部完成：** suite_20261001_134036_879815；PathVQA V1三seed最终Test59.1209±0.7169，RSVQA静态84.4362/CoCoOp86.1455。两账本已记全结果；RSVQA V1对LoRA及新基线的配对统计已补齐。以上任务不再列待跑，不重训、不据Test追加调参。剩余为电气V1/主表基线协议审计及缺口、效率汇总等此前计划；本轮无新增GPU任务。

## 2026-10-01 用户确定五项串行收尾实验

顺序：PathVQA最终归一化五轮V1 seed44/45/46各现有epoch5 checkpoint完整Test（仅评估、禁止重训或择优），随后RSVQA-LR静态Visual18+P20、原CoCoOp-style各seed44/data42从头五轮并固定epoch5完整Test。RSVQA两项均microbatch4/累积8，有效batch32，匹配已运行RSVQA V1/LoRA的batch设置；LoRA三轮与本次五轮仍需单列说明。静态69632参数，P20@0.3、S8@3e-5、Av10@1e-4；CoCoOp873120参数，无Visual18/问题条件，P20@0.3、MetaNet@3e-4，沿用历史2560->160->2560共享偏移定义。两项显式修正累积归一化，其他已定初始化/优化/评分不变。RSVQA官方active过滤及count区间评分复用既有接口。

实现时先以metadata明确绑定三seed正确checkpoint，核对缺失/身份错误即停止，不用目录最新匹配猜测。五任务串行失败即停，不自动重试，不覆盖产物；默认不自动关机。完成产物只有在身份与协议完整匹配后才可跳过。仅用户亲自执行GPU操作；助手准备本地代码/CPU检查/提交推送/一键命令。收集PathVQA每seed及均值±样本SD、RSVQA各组OA/AA/题型及现有V1/LoRA配对图像簇CI；所有完成及失败任务即时记两账本，状态更新plan。本批不增加Validation评估、其他seed或调参。

> **2026-10-01 LoRA-r2配对统计已补齐：** V1−LoRA Overall+2.2368，95%图像簇配对CI[1.2824,3.2133]；两份账本已更新，不再列为待提取。不新增GPU实验。

> **2026-10-01 LoRA-r2 b1a32已完成：** seed44固定五轮Validation57.1018，V1同seed高2.2368；已更新两份账本。覆盖旧待运行状态，不重跑。配对文件仅提供服务器路径，尚缺内容；不启动额外训练/推理，后续可只读提取。

> **2026-09-30 三项消融已完成：** C-static56.7343、C-qmap58.7314、C-noVisual57.9805，均seed44五轮Validation，完整记录见两份账本。原队列不重跑、不自动补seed；uniform仍延期至审稿要求。LoRA-r2代码准备状态不等于已运行，本轮未启动任何GPU工作。

## 2026-09-30 LoRA-r2 OOM后重跑：batch1/累积32，待用户执行

状态更正：下方原batch2准备记录对应运行已被用户执行并在837/3075步OOM。原运行无epoch5结果。独立脚本仍为`pathvqa/run_lora_full_model_attention_r2_norm_fixed_5ep_seed44.sh`，现在启动新实验`pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44`，microbatch1/累积32、有效batch32；从头五轮，其余配置及固定epoch5 Validation、10000次seed42配对均不变。重跑尚未启动，所有GPU操作由用户执行；不加入消融队列，不自动关机或重试。保留原目录和历史记录。此条更新上方旧的LoRA-r2准备状态。

## 2026-09-30 PathVQA LoRA-r2近似等参数预算对照：原batch2准备记录（后续OOM）

独立实验`pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_seed44`，不插入或更改当前C-static→C-qmap→C-noVisual队列。复用现有LoRA全attention训练器，24层ViT qkv/proj及36层LLM q/k/v/o，共192目标；标准r2/alpha4/dropout0.05、rsLoRA关闭，无Prompt。预计并强制审计1,769,472参数，比V1 1,864,963少约5.12%，仅称近似等参数预算。model44/data42，从头5轮、batch2/累积16、LR1e-4、原AdamW/clip1/3% warmup线性衰减、`model_accepts_loss_kwargs=False`；保存epoch3/4/5，仅固定epoch5完整Validation，对原五轮V1 59.3386做图像簇10000次seed42配对总体及题型统计。记录版本、执行提交、参数量、训练耗时和峰值显存；历史r8三轮/未知历史归一化仅作带协议差异标注的参考。独立脚本`pathvqa/run_lora_full_model_attention_r2_norm_fixed_5ep_seed44.sh`，输出`pathvqa/outputs/lora/`；GPU全部由用户执行，无自动关机、Test、其他seed或搜索。当前只实现与本地CPU静态检查，结果回传后再追加两份账本。

## 2026-09-30 用户确定三项PathVQA消融

实现状态：已准备串行目标`pathvqa_v1_c_ablations_static_qmap_noVisual_5ep_seed44`，等待用户亲自运行。顺序C-static→C-qmap→C-noVisual，失败即停、强制不关机。未找到现有完全匹配结果；运行时仍按完整产物与训练协议核验复用。独立名称`pathvqa_v1_c_static_norm_fixed_5ep_seed44`、`pathvqa_v1_c_qmap_norm_fixed_5ep_seed44`、`pathvqa_v1_c_no_visual_norm_fixed_5ep_seed44`，产物位于`pathvqa/outputs/visual_selection_prefix/ablations/`。参数69,632/1,815,811/1,846,531；C-qmap三条128维向量初始化为原query Linear的bias（Uniform[-1/sqrt128,+1/sqrt128]，即原query在u=0的输出），不加温度或尺度修正，共有参数和全局RNG保持原V1。保存epoch3/4/5、固定epoch5 Validation，按图像簇10000次seed42配对原59.3386。预检核验目标路径切断、保留分支梯度、原生embedding及20-token注入、缓存和保存重载一致；日志保留梯度、训练时间和峰值显存。结果尚未产生，用户回传后立即追加两份本地账本，不将代码准备记为已完成实验。

本轮只准备C-static、C-qmap、C-noVisual，按此顺序串行，均model seed44/data seed42，归一化五轮、固定epoch5 Validation，从头训练，对照原V1 59.3386。C-static即静态Visual18+P20，同数据集同协议不重复命名跑两遍；不能以RSVQA结果替代PathVQA。C-qmap每层以独立、跨样本共享的可学习query向量替换原问题生成query，其余keys/maps/共同Value/问题层权重/映射/P20/Visual18保留；query尺度应按原实现初始化口径审计，不额外调温度。C-noVisual仅移除Visual18，保留原生5/11/17特征读取与完整动态分支。C-uniform延期至审稿要求；不补其他seed或Test。保留共有参数初值、数据顺序与原五轮优化协议，删除模块不扰动共有初始化；记录实际参数与单因素边界。提供串行失败即停入口，不自动重试，不继承旧任务自动关机设置；本轮未要求关机，默认关闭。所有GPU工作由用户亲自启动。此条覆盖上方旧建议优先级。

## 2026-09-30 剩余论文补实验清单（规划，不自动运行）

优先级更新：不再继续结构搜索，已有A/B及视觉布局负结果直接复用。以下覆盖旧矩阵的执行优先级，不删除历史。

1. 先完成CPU协议/产物审计：PathVQA/SLAKE/电气各基线的归一化、split、epoch与精确运行身份；已有兼容产物复用，只列出确实缺失的重训项。补V1B对原V1、RSVQA V1对LoRA的图像簇配对CI。
2. RSVQA补静态Visual18+P20和原CoCoOp-style（无Visual18），seed44，固定五轮、相同数据/评分/归一化；共2次训练。不据Test继续调参，不补RSVQA其他seed。
3. PathVQA最终固定epoch5 V1 checkpoints seeds44/45/46做Test，仅评估不重训；明确当前58.8592均值来自Validation。主表基线经第1项审核后补齐，不混旧3轮未知协议和新5轮结果作纯架构归因。
4. 核心消融统一PathVQA seed44、归一化五轮、固定epoch5 Validation，从头训练：C-static移除动态分支保留P20+Visual18；C-qmap以与问题无关的可学习query替换地图query，保留问题条件层门控；C-uniform三层地图固定均匀保留其余映射；C-noVisual移除Visual18保留动态分支。优先前两项，后两项补完整机制表；这不是推理时关模块。若已有同协议结果则复用。C-static/C-qmap追加45/46属于预定可选稳定性补证，不按掉分大小挑选。
5. 电气补冻结V1，先seed47与历史主seed对应；先核验有效样本/设备场景划分与基线协议，再决定补48/49。不得复用缺图自动计正确口径。SLAKE V1三seed已完成，不重训；其基线按审计决定缺口。
6. 效率表：先提取已有训练时长、峰值显存、参数量；不同时硬件/精度的延迟不横比，缺失且论文需要时由用户统一环境补测。
7. 可选单层17（保留192维条件容量）、最终五轮checkpoint机制干预和定性图；不排新层数/Prompt数/学习率/瓶颈宽度搜索，不新增数据集。A/B不补多seed作为默认任务。

> **2026-09-30 完成状态更新：** V1B seed44五轮Validation57.4533，RSVQA LoRA-r8 b4a8 seed44 Test87.0552，两项已记账，不重跑。保留原V1；未新增训练。LoRA本地配置三轮、V1五轮，比较须披露预算差异；现有预测可后续做配对图像簇CI，当前未回传。此条取代旧待运行状态。

> **2026-09-29 RSVQA-LR full-attention LoRA-r8已排期，V1B自动关机已取消。** 新LoRA仅做单独seed44实验：官方Train从头训练3 epochs，ViT 24层qkv/proj与LLM 36层q/k/v/o共192个目标，r8/alpha16/dropout0.05/LR1e-4，预计并强制核验7,077,888参数；microbatch2、累积16、有效batch32、3% warmup线性衰减、裁剪1，并显式`model_accepts_loss_kwargs=False`。固定epoch3完整Test，训练与推理为同一实验的两个阶段，不与V1B或其他实验串联。独立脚本无自动关机、无额外GPU冒烟；全部GPU操作由用户亲自启动。V1B普通目标继续保留；旧`_shutdown`别名仅兼容调用且强制不关机。

> **2026-09-29 V1B独立证据token消融已实现，等待用户亲自运行（关机安排已取消）。** 命名固定：已完成的879,364参数去瓶颈共享偏移为V1A；新V1B保留原V1全部1,864,963参数及初始化/学习率，把`b=prefix_output(ReLU(c))`从P20广播偏移改为序列`[P20;b;原chat]`。第21前缀位置attention有效、label=-100，不进入独立问题条件源；预检强制核对mRoPE、视觉mask、DeepStack、原生embedding不变、完整条件分支梯度、单次prefill注入与KV-cache/save-load一致。固定seed44/data42、五轮、epoch5 Validation，配对原V1，并在V1A产物存在时补B对A；两者同时改变映射与接口，不作单因素归因。GPU全由用户启动，不跑Test/其他seed/额外配置。

> **2026-09-29 两项结果已回传。** RSVQA-LR V1 b4a8 seed44固定epoch5 Test OA85.0360/AA86.0812，相对基座OA+27.4590；该运行完成不重复启动。PathVQA方案A去瓶颈保留偏移55.9035，对原五轮-3.4351且配对CI低于零，记为负消融，不自动调alpha或重训。方案B结果尚未回传；不得推断运行状态或补启动串行任务。继续收集已有约定的B与RSVQA静态/原CoCoOp结果；LoRA未新排期。

> **2026-09-29 用户确认补A与修改版B两项消融。** A恢复为待运行：已实现去瓶颈但保留共享偏移；B保留原V1全部瓶颈/层权重/输出头，仅将生成向量改为独立序列token。建议最小B为[P20;b;原chat]共20+1、静态P20不加b，保持原输出bias一次、1,864,963参数；20+3会另改变分层输出保留方式，不混为纯注入对照。B具体20+1为本轮建议，尚未实现。两项均PathVQA seed44/data42五轮固定epoch5，GPU全部由用户执行。直接摘要独立插入的旧B暂不排期；A/B失败不能以独立因素假设推断组合必败。保留小非零原输出头初始化时须诊断独立token尺度和梯度，初始化策略改变则另标协议变量，不事后据分数调尺度。

> **2026-09-29 用户纠正消融意图：独立证据token插入，不是摘要偏移。** 上方/下方已实现`pathvqa_v1_direct_summary_norm_fixed_5ep_seed44`仍为p_i+alpha*s，不能代表本次用户意图；该待运行方案暂缓，不自动启动或覆盖其代码/历史。用户想保留静态P20，将地图加权的2560维摘要以独立证据token（如20+3）注入。历史笔记B5提出过三个额外soft prompt但未找到完成实验记录；旧QDPT的CA生成Z10、GRASP全局原型token不等同于当前共同Value摘要直注；V1/V2/V3均为P20偏移。新候选保留三份摘要对应3个token，若先融合成一份则为20+1，二者需明确选定；不采用旧偏移幅度作为独立token的当然尺度标准。具体注入位置、归一化/尺度与层权重待方案明确，本轮仅纠正语义，不实现或启动新GPU实验。

> **2026-09-29 PathVQA五轮V1去瓶颈读写投影消融已实现，等待用户亲自运行。** 独立目标`pathvqa_v1_direct_summary_norm_fixed_5ep_seed44`保持三层地图、共同Value、各层LN、问题摘要/softmax层权重、P20、原Visual18和注入位置，只删除三套2560->64 `value_blocks`及192->2560 `prefix_output`/ReLU，改为`b=alpha*sum_l beta_l LN_l(z_l)`。仅新增FP32标量alpha@1e-4，总可训练879,364，较1,864,963减少985,599（52.8482%）；其余保留张量按原V1构造顺序逐张量核验。用户启动的首个真实Train batch先在eval/no_grad下计算旧V1输出和新摘要RMS，校准一次alpha0并恢复CPU/CUDA RNG，再以同一batch做训练态梯度预检；旧头只作为非参数临时校准张量并在校准后删除。model seed44/data seed42、batch2/累积16、归一化修正、五轮、保存3/4/5及固定epoch5 Validation均不变，配对原五轮V1 59.3386。所有GPU校准/训练/评估由用户执行，不跑Test/其他seed/额外配置，不自动关机。

> **2026-09-29 RSVQA-LR V1吞吐版已实现，等待用户运行。** 因用户实测32GB RTX5090仅约14,948MB显存、GPU峰值约47%，新增独立目标`rsvqa_v1_norm_fixed_5ep_b4a8_seed44`：microbatch2->4、累积16->8，有效batch仍32，epochs仍5，其余结构、初始化、学习率、调度、数据顺序与固定epoch5 Test均不变。真实batch预检同步扩大为4，先检查显存和梯度；该配置预期减少microbatch调度但并非逐位等价优化轨迹，结果须保留独立实验名。原batch2/累积16目标与产物不覆盖；GPU运行由用户执行，不自动关机。

> **2026-09-29 RSVQA-LR归一化五轮V1 seed44已实现，等待用户亲自运行。** 只做数据集迁移：原五轮V1结构、1,864,963参数、model seed44/data seed42、batch2/累积16、两档Visual18学习率、3% warmup线性衰减和`model_accepts_loss_kwargs=False`均不变。使用官方Train 57,223题从头训练，保存epoch3/4/5并固定epoch5完整Test；训练监督原始发布答案，Count评估再映射官方区间。训练与推理共享题型短答prompt，但条件分支只读原始真实问题。GPU预检/训练/Test只由用户启动，不自动关机，不补其他seed。

> **2026-09-29 RSVQA-LR规范化接口及基座Test已完成。** `rsvqa_lr_base_qwen3vl_test`在完整Test 10,004题/100图上得到OA57.5770、AA58.8947；四类为rural/urban69.0000、presence63.0457、count29.8948、comparison73.6382，图像簇95%CI[56.3944,58.7048]。这是不训练、不加载checkpoint的固定参照；`RSVQA/`的数据record契约和官方区间评分继续作为后续训练/评估唯一接口。结果已写两份账本，账本修改留本地随下一次相关代码提交，不单独提交。

> **2026-09-29 用户纠正并锁定CoCoOp基线：不加Visual18。** 已核对主账本2026-09-09 PathVQA CoCoOp-style P20/H160及slake/cocoop_prompt_tuning.py：仅P20与2560->160->2560 Meta-Net可训练，总873,120参数；原生post-merger视觉token均值生成共享P20偏移，无视觉Prompt、无问题条件。RSVQA沿用这个方法定义及原P20/Meta-Net学习率0.3/3e-4，不擅自加模块。下方助手建议CoCoOp+Visual18明确撤回，不实施。四组固定为原始Qwen3-VL、静态Visual18+P20、原CoCoOp-style（无Visual18）、完整V1；共同五轮与归一化/评分协议按既定计划。

> **2026-09-29 RSVQA-LR数据已由用户解压，拟定三训练四评估。** 用户回报服务器根`/root/autodl-tmp/dataset/RSVQA/6344334/`，Images_LR含772张256x256 RGB TIFF；split有效images/questions/answers分别Train572/57223/57223、Val100/10005/10005、Test100/10004/10004。只作用户回报，尚未全量复核。三个split文件均须先筛active=true再按ID关联并检查一致性；禁止把inactive其他split记录纳入训练，不用all_questions/all_answers重划分。实施前CPU全量校验图像ID跨split互斥、答案反向question_id、有效ID唯一、路径与评分计数类别。
> 用户拟比较原始Qwen3-VL零训练、Visual18+P20静态、CoCoOp-style与冻结V1，共三训练四次完整评估。建议seed44/data42，五轮固定epoch5；零训练不插入随机Prompt。为视觉适配条件一致，建议CoCoOp-style也保留同Visual18并明确命名为该变体，使用全局视觉池化条件P20，不声称严格复现原CoCoOp或单变量隔离全部V1机制。该CoCoOp细节仍为本轮建议，待实施确认。不预排RSVQA LoRA；遥感部分定位为冻结骨干Prompt家族迁移验证，不声称此数据集上优于LoRA，也不依据最终Test是否好看来决定补对手。统一归一化/评分/输入协议，报告Overall、题型准确率和AA；所有GPU操作由用户执行，本次只更新计划。
> **2026-09-28 用户安排明天补RSVQA-LR单seed跨域实验（计划日期2026-09-29）。** 默认model seed44/data seed42，沿用冻结归一化五轮V1及原Visual18两档学习率，固定epoch5；仅适配数据与官方评分，不扩展多seed/结构/LR搜索。实施前核验实际下载版本、图像级train/val/test划分、问题/答案处理和计数等评分口径，不凭清单推定数据已就绪。今天不启动；明天先准备与检查入口，再交付用户亲自运行全部GPU预检/训练/评估的命令。此为实验排期，不创建自动运行或提醒；本次不修改结果账本。

> **2026-09-28 SLAKE V1三seed结果已回传，覆盖等待运行状态。** 按当前全语言Test固定epoch5计划，44/45/46为75.55/76.31/78.37，均值76.7433±1.4591；精确运行身份/路径/样本数待补。三seed不重复训练；保留冻结V1，不据Test重开调参。性能均值接近历史旧QDPT但观测seed波动增大，论文不可声称跨域稳定性普遍改善。继续既定电气、基线协议审计和核心消融补证据规划，未启动任何新GPU操作。

> **2026-09-28 当前已实现，等待用户亲自运行：SLAKE固定V1三seed Test。** 完整冻结归一化修正五轮V1（P20、索引17 S8@3e-5+Av10@1e-4、5/11/17条件选择、共同Value与共享P20偏移，1,864,963参数），只将训练数据换为SLAKE官方train、评估换为全语言官方Test；model seeds44/45/46、data seed42严格串行，保存epoch3/4/5并固定epoch5，不跑Validation、不按Test挑checkpoint。新增目标`slake_v1_norm_fixed_5ep_seeds44_45_46_test`，任一seed失败即停止后续seed；GPU预检、训练与评估仅由用户启动，不自动关机。

> **2026-09-28 新阶段：冻结结构，规划论文补实验。** 用户要求列出SLAKE、自建电气与消融清单；此前停止的是新结构涨分探索，本次只规划最终V1的跨数据集验证与证据补全。以下为建议矩阵，不恢复深层Prompt/数量/LR扫描；其中SLAKE固定V1三seed现已获授权并进入上方待运行状态。

## 2026-09-28 论文补实验建议矩阵

### P0：冻结与协议审计（先CPU）
- 固定原归一化V1：P20+索引17 S8@3e-5/Av10@1e-4，5/11/17问题条件视觉选择、共同Value、共享条件偏移；1,864,963参数，五轮固定epoch5。保留所有负结果。
- 补齐PathVQA seed45/46精确输出身份；审计各旧基线实际归一化、初始化、训练预算、mask、分辨率、评分与版本。不可把旧3ep未知归一化直接当新5ep公平对照；通过审计且协议一致的产物复用，其余标历史或重跑匹配版。
- 新V1入口目前为PathVQA专用；SLAKE/电气旧入口不是新版迁移已完成的证据。迁移仅适配数据/评估，不改结构；分别验证问句mask、多图grid映射、语言/选项保留与答案排除、prefill缓存。
- SLAKE保持既定语言/OPEN-CLOSED/KVQA-VQA覆盖，Train/Validation/Test不混用。电气冻结有效样本ID与切分，旧972含18条缺图自动计正确，历史纠正为954；若新版数据变化需版本化不能硬套旧分母。按设备/场景/同源多视图检查泄漏与统计聚类单位。

### P1：跨数据集主结果
- SLAKE：固定V1，seeds44/45/46，data seed42，五轮固定epoch5；超参由PathVQA迁移，不在Test调参。报告Overall/OPEN/CLOSED/VQA/KVQA/EN/ZH。
- 电气：固定V1，建议主seed47与历史一致，复现seeds48/49在运行前固定；三seed方法间一致，data seed42。报告Overall及已有有效任务分组，按实际设备/场景/图像相关单元聚类。
- 三数据集主比较：Static P20、CoCoOp-style（标注近似复现）、冻结V1、Full-Attention LoRA-r8。冻结骨干零训练结果可作为背景。旧QDPT作为版本演进参考，协议不匹配不得声称单变量改进；旧SLAKE LoRA明显更强的结果必须保留。
- 主方法与关键基线采用匹配三seed及预先声明训练预算；各方法专属优化配置透明，非强制相同LR。统一训练/评估协议以外的历史结果单列。PathVQA新版主模型三seed训练已完成；锁定比较配置后再做最终Test，不能拿Validation58.8592当Test结果。SLAKE/电气也是配置锁定后最终评估，不按Test挑配置。
- 同设备/软件/精度/输入预算记录参数、训练时间、峰值显存、TTFT/TPOT；复用已有匹配测量，不另扩展性能测试矩阵。

### P2：PathVQA重训练消融（不是推理时关分支）
- A1 去动态分支：仅P20+原Visual18，原位置与各自LR不变；直接匹配静态底座，已有P20-only不能替代。
- A2 去问题引导地图：问题生成的三层query换成图像无关/问题无关可学习query，视觉keys/values及问题条件层门控保持原样；仅检验地图查询的问题依赖。
- A3 固定均匀地图：每张图有效token归一化均匀权重，保留原共同Value/三分块/问题层门控/输出路径，从头训练；检验非均匀选择，不能用已训练checkpoint uniform_map干预代替。
- A4 单层17视觉选择：只用层17生成地图，保留最终条件192维与输出容量，避免同时缩小瓶颈；报告真实参数变化，检验多层来源的必要性，不宣称严格等参数。实现细节另行审计。
- A5 去Visual18：移除视觉Prompt插入，其他动态选择/P20保持不变并从头训练；检验当前视觉适配作用，不用旧纯视觉Prompt或深层替换结果代替。
- 五项先在预先固定seed44以同五轮协议完成完整表，全部结果保留；论文核心多seed消融优先预定A1+A2补45/46，不以哪个掉分最大决定补seed。A3-A5若只跑单seed须显式标注。
- 不在三数据集复制全部消融；跨域机制如需要，仅在电气预先固定A1+A2作为补充，待预算决定。

### P3：复用与可选
- 已有三轮V1问题错配/均匀地图干预作为对应checkpoint依赖性证据；若图表声称最终五轮模型机制，应在冻结五轮checkpoint重放少量既定干预，不新增搜索。
- 复用学习率低LR、Visual20、Deep5、Deep10+10、位置V3、混合V2、3->5epoch已有结果，注明协议/单seed/联合变量。不得将旧QDPT或V0消融直接挂到新版V1。
- 可选层门控固定均匀、无问题卷积、无P20仅在论文明示其为核心创新时补；不排新Prompt数/宽度/层位/LR扫描或压缩训练。
- 所有新GPU预检/训练/评估均由用户亲自启动。此矩阵只是规划，不自动执行、提交或推送。

> **2026-09-28 最后一次探索已完成，探索正式结束。** 深层每层10+10 seed44固定epoch5为57.2935/89.4720/25.2074；对原五轮V1 Overall-2.0451，配对CI[-2.8918,-1.2314]，对Deep5也下降0.9107。保留原归一化五轮V1与已有三seed结果。遵守用户“无论如何最后一次”的决定，所有尚未执行的探索候选均不再排期，不补seed、不重试、不调参、不跑Test或压缩；后续仅整理已有结果。下方待运行及候选计划均为历史。

> **2026-09-27 最后一次探索实验已实现，等待用户亲自运行。** 独立目标`pathvqa_v1_deep_visual20_split_lr_l16_23_norm_fixed_5ep_seed44`：彻底移除Visual18，在ViT索引16～23各放独立低LR P10和高LR P10，分别为真实优化器组3e-5/1e-4，按低10后高10当层插入并立即移除。入口强制24 blocks、160个视觉Prompt、两组各81,920参数和总参数2,010,371；私有CPU随机流不改变V1共有初值，索引17条件特征在剥离临时Prompt后读取。保存epoch3/4/5、固定epoch5，并分别对原五轮V1 59.3386及Deep5 58.2042做10000次seed42图像簇配对比较。GPU由用户执行，命令显式启用脚本退出后自动关机；成功或失败均结束探索，不重试、不调参、不补seed、不跑Test或启动其他实验。

> **2026-09-27 用户明确指定最后一次实验：深层每层10+10，结束后无论结果不再追加探索。** 索引16-23共8层，每层独立10行LR3e-5+10行LR1e-4，当层插入移除，完全替换原Visual18；Normal(0,0.02)，独立初始化流不改变共有参数初值。共160个视觉Prompt、预计总可训练2,010,371。归一化V1 model seed44/data seed42，从头五轮，保持其他结构/训练/评估协议，固定epoch5 Validation。主对照原V1五轮seed44 59.3386，次对照逐层5 58.2042；图像簇配对10000次seed42。只交付干活AI实施指令，所有GPU预检/训练/评估由用户亲自启动。完成或失败均停止，不自动重试、调参、补seed、跑Test或开始后续实验；结果回传仅记录与总结。

> **2026-09-27 原Visual18统一低LR已完成，负结果。** 57.9805/88.5760/27.4729，对原五轮seed44 Overall-1.3580、Yes-No-2.3040，配对CI均低于零；保留原两档LR五轮V1。仅降低Av10 LR的同布局对照支持原配置，但不能判定机制或排除其他统一LR。三项视觉替代/消融均未改善；深层每层10+10仍为讨论候选，未启动，本次不自动增加运行。下方待运行状态为历史；今晚截止及用户执行全部GPU操作的边界不变。

> **2026-09-27 原版Visual18统一低学习率消融已实现，等待用户亲自运行。** 以归一化修正五轮V1 seed44为基准，保留索引17原始S8+Av10两张参数表、8后接10的排列、Normal(0,0.02)初始化及当层插入/移除前向；唯一训练改动是将Av10 LR从1e-4降至3e-5，使S8/Av10均为3e-5。总参数继续强制核验1,864,963，共有初值仍由既有V0/V1逐张量审计保证。独立目标`pathvqa_v1_visual18_uniform_lr3e5_norm_fixed_5ep_seed44`，保存epoch3/4/5并固定epoch5完整Validation，对原五轮V1 seed44 59.3386/90.8800/27.8877做10000次seed42图像簇配对比较。本次只检验原18布局中的两档LR与统一低LR，不概括其他LR设置；不跑深层10+10、Test、其他seed或额外配置。全部GPU操作由用户执行。

> **2026-09-27 用户纠正：撤销下方单层10+10解读。** 用户实际希望ViT深层索引16-23每层独立10个低LR3e-5 +10个高LR1e-4，当层插入移除，替换整个旧Visual18；不是单层20。另提出必须考虑统一低LR对照。建议先在原18布局仅将Av10的LR1e-4降为3e-5，所有初值/其余协议不变，直接检验原两档LR是否必要；该对照尚待用户确定执行顺序。若要证明深层10+10中的两档LR价值，还需同深层20布局统一低LR的匹配对照，不能用原18统一低LR代替。此轮仅修正计划与讨论，不实现或启动新实验；所有GPU操作仍由用户执行。

> **2026-09-27 新增消融方向：单层Visual20双学习率10+10。** 按用户“不同学习率10+10”理解为ViT索引17单层20个，前10行LR3e-5、后10行LR1e-4，不是深层每层20个。复用失败的统一Visual20的全部20行初值及共有参数初值，从头五轮seed44/data seed42，除视觉LR分配外保持其协议不变；预计1,867,011参数。双主参照为统一Visual20 57.7249（检验LR分配）与原8+10 V1 59.3386（替代方案性能）。5+5仅讨论备选、不排期；此前统一18建议暂不执行。本次交付干活AI实施要求，尚未修改代码或启动运行；所有GPU操作由用户本人执行。

> **2026-09-27 实验2已完成，负结果，覆盖下方待运行状态。** 深层16-23逐层5为58.2042/89.5680/26.9304；对原五轮seed44 Overall-1.1344、Yes-No-1.3120，配对区间均低于零。保留原五轮V1，不自动启动叠加或补seed。第三次预算尚未指定；若用于机制拆分，可考虑原18仅统一LR1e-4，但不是已授权运行。今晚截止、不做第三天；全部GPU操作由用户亲自执行。

> **2026-09-27 实验2已实现，等待用户亲自运行。** 以归一化修正五轮V1为基准，完整移除索引17的S8+Av10，改为ViT代码索引16～23共8层各自独立的P5；每层只在本block输入前按图像/帧分段插入，输出后立即移除，不跨层传递。入口强制骨干恰为24个block，否则训练前停止；索引17的条件特征钩子重挂在剥离Prompt后的真实视觉输出。40,960个深层视觉Prompt使用独立CPU随机流Normal(0,0.02)，不推进共有参数初始化随机流，统一LR1e-4；预计并强制核验总参数1,887,491。独立目标`pathvqa_v1_deep_visual5_l16_23_norm_fixed_5ep_seed44`，保存epoch3/4/5并固定epoch5 Validation，对五轮V1 seed44 59.3386做图像簇配对比较。只评价整套视觉适配布局，不单独归因于深度、数量或LR；不跑Test/其他seed。全部GPU预检、训练与评估由用户执行。

> **2026-09-27 实验1已完成，负结果。** 统一Visual20五轮seed44为57.7249/90.5920/24.9521；相对五轮V1 Overall-1.6137，配对CI[-2.4277,-0.8014]。原V1保持主候选。数量与前8行LR同时改变，不能认定18->20单独致损。建议将18个统一1e-4列为优先拆变量候选，深层逐层5仍待决策，叠加不自动进行；未授权新具体运行。今晚截止及用户亲自执行全部GPU操作的约束不变。下文等待重试为历史状态。

> **2026-09-27 Visual20首次启动失败，已修复，等待用户重试。** 失败发生在模型构造参数审计：子类删除`visual_s8/visual_av10`后仍先调用父类分组函数，父类访问旧字段触发`AttributeError`；未进入真实batch预检、训练或评估。修复仅改为Visual20子类直接返回六个实际参数组，不改变结构、初值、LR、五轮预算或评估协议。失败已记入两份账本；GPU重试仍只能由用户执行。

> **2026-09-27 截止与压缩方向更新（覆盖此前三天预算）。** 用户决定今晚结束探索，不做第三天；五轮V1保持主候选。今晚优先讨论/获取参数冗余证据：先查大矩阵权重谱，再用真实输入激活与压缩敏感性筛选，避免凭参数量或权重范数直接删层。当前仅确定诊断方向，具体压缩秩与训练尚未排定；后处理压缩后的存储参数量不得冒充从头训练的可训练参数量。所有涉及GPU的诊断、前向、评估与训练仍由用户亲自执行。

> **2026-09-27 五轮V1第三seed结果已回传。** 按当前seed46计划为58.7314/90.3360/27.2176；三seed Overall58.8592 +/-0.4299、range0.8309，参数1,864,963。seed45/46完整身份/输出路径及配对拟合明细待补，不重复训练。五轮三seed可作为当前主候选，1M压缩尚未排期，不自动增加训练/Test；所有GPU操作用户亲自执行。

> **2026-09-27 当前实验1已实现，等待用户亲自运行：五轮V1统一Visual20。** 以归一化修正、五轮seed44 V1为基准，仅把ViT代码索引17当层插入/移除的分组`S8@3e-5+Av10@1e-4`替换为单组静态Visual20@1e-4；初始化均为Normal(0,0.02)。前18行严格复用同seed原S8后接Av10的初值，新增2行使用独立CPU生成器且不推进全局CPU/CUDA RNG，从而保持P20、问题条件分支及输出头等共有参数初值一致。预计并强制核验1,867,011可训练参数。独立目标`pathvqa_v1_visual20_lr1e4_norm_fixed_5ep_seed44`，保存epoch3/4/5、固定epoch5完整Validation，并对五轮V1 seed44 59.3386做10000次seed42图像簇配对比较。本实验联合改变视觉Prompt数量18->20和学习率分组，只评价整套简化方案；不跑Test、其他seed或额外配置。全部GPU预检、训练和评估由用户执行。

> **2026-09-26 当前已授权并已实现：五轮V1 seed46复现，由用户亲自运行。** 在五轮seed45入口上仅将model seed改为46；data seed42、结构和1,864,963参数、初始化规则、各组LR、batch2/累积16、AdamW、裁剪1、`model_accepts_loss_kwargs=False`、5 epochs、3% warmup+线性衰减、保存epoch3/4/5及固定epoch5完整Validation全部不变。独立目标`pathvqa_v1_norm_fixed_5ep_seed46`；与旧三轮seed46做10000次seed42图像簇配对比较，并复用同一固定256+256清单分别审计三轮/五轮seed46。不跑Test、不补其他seed、不自动调参；所有GPU操作由用户执行。

> **2026-09-26 五轮seed45成绩节选已回传。** 按当前实验上下文为58.5077 Overall、90.4320 Yes/No、26.6752 Free-form；完整身份/精确输出路径及配对统计待补。五轮44/45均值58.9232、分差0.8309。下一候选原样五轮seed46，尚未授权；GPU操作均由用户执行。

> **2026-09-26 当前已授权并已实现：五轮V1 seed45复现，由用户亲自运行。** 唯一训练变量为model seed44->45；data seed42、V1结构、1,864,963参数、各组LR、batch2/累积16、AdamW、裁剪1、`model_accepts_loss_kwargs=False`、5 epochs、3% warmup+线性衰减到新终点、保存epoch3/4/5及固定epoch5完整Validation均与五轮seed44一致。独立实验`pathvqa_v1_norm_fixed_5ep_seed45`；与旧三轮seed45做10000次seed42图像簇配对比较，并用同一固定256+256清单分别审计三轮/五轮seed45的生成准确率和答案正文CE。不跑Test、不补其他seed、不自动调参。所有GPU操作由用户执行，助手只提供命令。

> **2026-09-26 五轮V1 seed44已完成。** 固定epoch5 Overall59.3386，对三轮+0.7669，CI[0,1.5370]；where+5.6235、what+0.0785。保存五轮单seed候选与三轮三seed参照，不混算。拟合探针输出manifest因新增来源元数据而文件SHA变化，但比较器已逐项核对两侧各256题的question ID、问题、答案、图像ID和分层，样本内容一致。下一训练为上方已授权的原样5ep seed45。所有涉及GPU的操作只能由用户本人执行。

> **2026-09-26 当前授权：归一化修正 V1 seed44 5-epoch 训练对照。** 从头训练，唯一实验变量是总预算3→5 epochs，3% warmup+线性衰减到新终点；保持修正V1架构、初始化、各组LR、batch2/累积16、AdamW、裁剪1、监督/生成及`model_accepts_loss_kwargs=False`。独立实验`pathvqa_v1_norm_fixed_5ep_seed44`，保存epoch3/4/5，固定epoch5为主结果；对原3-epoch V1做10000次seed42图像簇配对比较，并显式复用SHA256为`5ba0ae68...dfda930`的旧256+256样本清单比较生成准确率与答案正文CE。本地实现、检查、提交推送后已授权服务器运行；完成后停止，不加seed/Test/自动调参。

> **2026-09-26 V1 seed44 小规模拟合检查已完成。** Train/Validation各分层256题（不是官方Overall）为61.7188/53.1250；what仅25.0000/20.8333，where75.0000/64.5833，Yes/No96.8750/87.5000。开放回答在Train也明显欠拟合，罕见答案训练题尤差；同时epoch平均loss 0.8822→0.7100→0.6490，末20%均0.6389，未见反弹。唯一优先候选是保持V1结构/监督不变，做一次更长训练的单seed受控实验（建议5 epochs）；尚未授权，不自动启动。详细记录见两份账本，下文待执行状态为历史。

> **2026-09-26 V3已完成，保留V1。** 后置P20 seed44 Overall58.2361，对V1 -0.3355（CI[-1.0751,0.3930]），Free-form -1.0530；无收益，不自动补seed或扫描位置。精确输出路径、终点训练诊断待补；下一实验尚未排定。下文V3待运行为历史。

> **2026-09-25 当前授权：V3 后置条件 P20 单次 seed44。** 以归一化修正版 V1（非 V2）为唯一基准，保持全部 1,864,963 个参数和训练/评估协议，只将原有 20 个条件 Prompt 从完整 chat 前移动到完整图像段结束标记之后、真实问题之前。复用项目 `Qwen3ProcessorWithMMRL` 继承链，V3 processor 在原生图像展开和分词后预留占位与 `prompt_mask`，模型只做 embedding 替换和条件注入；不得在模型里再次插 token。先审计真实边界及 Qwen3-VL 的 mRoPE/DeepStack/KV-cache，完成本地实现、静态检查、提交推送；服务器由用户运行独立目标 `pathvqa_v3_postvisual_prefix_p20_norm_fixed_seed44`，真实 batch 预检通过才训练 3 epochs 并固定 epoch3 完整 PathVQA Validation。对归一化 V1 seed44 58.5717 做图像簇配对比较；不跑 Test、其他 seed、位置扫描或自动关机。成功或失败后将精确结果追加两份账本，本次只交付用户运行命令，不代启动。

> **2026-09-25 V2 alpha只读审计完成。** alpha有更新但接近全1，变化RMS0.01365、有效映射位置差异R约0.944%；没有样本级激活记录。V1继续作为参照，V2暂缓，无新增训练或推理排期。

> **2026-09-25 当前只读审计：V2 seed44 alpha 位置差异。** 已准备离线入口，仅加载既有epoch3 checkpoint和每20步训练JSONL，输出完整alpha、相对全1变化、共同层缩放与跨位置分解、按输出头W加权的有效映射R及早中晚裁剪后梯度。若未保存样本级c/b_l/逐位置偏移，明确记为缺失，不补模型前向。用户自行在服务器运行；收到机器可读结果后更新两份账本和设计笔记的候选结论，V1继续为参照，V2暂不继续、不排新训练。
> **2026-09-25 V2已完成，暂不继续。** 逐位置alpha[20,3]单seed44为58.0764 Overall，对修正V1 -0.4953（配对CI[-1.2189,0.2228]），Free-form -1.0849。未见收益，继续冻结V1为参照；不自动补seed/调alpha/增加输出头。可读已有alpha及有效偏移日志，但不是新训练前置要求；当前无新增训练排期。下文V2授权待运行状态为历史。
> **2026-09-25 当前授权实验：V2 P20逐位置三层混合系数，单次 seed44。** 以归一化修正版V1为基准，仅新增FP32自由alpha[20,3]全1初值，对现有输出头按64维输入块取三层分量，bias只加一次；预计总参数1,865,023。保持model seed44/data seed42、其余初始化/LR/Trainer归一化/3 epochs/固定epoch3 PathVQA Validation不变。先本地数值/接口预检并提交推送，服务器快进同步后仅运行本次目标；若预检不通过则停止，不解释为结构分数。主对照修正V1 seed44 58.5717，图像簇配对bootstrap10000次seed42；不跑Test、不补seed、不自动调整。完成或失败均立即记两份账本。
> **2026-09-25 最新完成：修正V1三seed齐全。** seed46 Validation57.58/90.1760/25.08（Overall/Yes-No/Free-form），where64.5477；三seed Overall约57.77 +/-0.72、range1.40，精确summary待补。原样复现阶段完成，下文seed46待运行状态为历史，不重复启动。当前冻结V1为参照；设计笔记新增“每个P20分别混合三层输出分量，+60参数”及备选，仅待讨论、未排期或授权，不启动训练/Test。
> **2026-09-25 当前实验：已准备修正 V1 的单次 model seed46 复现，须由用户在服务器亲自启动。** 唯一训练变量为初始化 seed45→46；data seed42、结构、1,864,963参数、归一化、训练与固定epoch3 Validation协议均沿用seed44/45。独立目标 `pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed46`，不跑Test、不补其他seed、不自动关机。结果出来后记录精确分数并与seed44/45共同报告，不能用三seed直接断言稳定性已解决。
> **2026-09-25 最新完成：修正V1 seed45。** 固定epoch3 Validation57.17 Overall、88.9920 Yes/No、25.43 Free-form、where63.5697；两seed均值约57.87、分差约1.40。下文seed45待运行状态为历史，不重复启动。待补精确summary、seed45配对统计及较大推理速度差异的环境/配置核查。建议下一训练候选原样seed46，尚未获本次运行授权；不改结构、不自动跑Test。账本留本地随下一相关代码提交。
> **2026-09-25 当前实验：已授权归一化修正版 V1 的单次 model seed45 复现。** 唯一训练变量为模型初始化 seed44→45；data seed42、架构、参数量、学习率组、优化器、batch2/累积16与已验证归一化、3 epochs、固定 epoch3 PathVQA Validation 及生成评分协议不变。从头训练，独立输出 `pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed45_*`；不跑 Test、不补其他 seed，不自动关机。与 seed44 的58.5717做同协议比较，完成后按实测更新两份账本；单个新 seed 不足以充分估计稳定性。
> **2026-09-25 最新状态：V1归一化修正版诊断及已有产物补读全部完成。** 正常58.5717；对原V1/修复V0/CoCoOp/Static的Overall配对CI均排除0。同类型问题错配-2.8759、跨类型-6.6784、均匀地图-1.2143，各自配对CI均为负；四个探索性子组、128题地图/摘要/偏移统计、同图换问题及同问题换条件图像的共同99题比较、60张既有分歧图均已读并记入两份账本。无需重复全量推理。现保留问题条件P20与非均匀地图；11/17虽最相近但不足以指定删层。下文待运行/重试状态为历史，不新增训练/Test/seed，不据单seed断言稳定性。账本修改留本地，按AGENTS.md随下一次相关代码提交推送，不做仅账本提交。
> **2026-09-25 诊断重跑状态：首轮在均匀地图小样本预检因`KeyError: grid`停止，未进入三项完整Validation干预。** 已定位为探针网格元数据只在特征捕获模式下写入的诊断接口错误；修正为每次启用探针均记录当前grid，仍仅在明确捕获时复制完整视觉特征。原checkpoint和约58.57分不变。提交推送后由用户在服务器运行同一只读目标，重新通过全部预检才继续全量干预；失败结果已追加两份账本，不解释为结构负结果。
> **2026-09-25 当前优先项：修正 V1 seed44 epoch3 只读机制诊断（本地代码已准备，服务器结果待运行）。** 先用已有正常 Validation 预测重算评分并核对6259题/832图像簇，按图像簇配对bootstrap（10000次、seed42）比较原V1、修复V0、CoCoOp、Static P20；固定128题前向探针审计地图/共同Value摘要/条件偏移及同图换问题、同问题换条件图像；小样本正常贪心预测必须复现已有结果，且三种干预只改变条件源，才能跑同类型问题错配、跨类型问题错配、均匀地图三项完整Validation。输出独立目录，不训练、不更新参数、不跑Test或其他seed。Windows无AutoDL SSH，本地提交推送后由用户运行服务器入口；收到机器可读结果后再补两份账本和机制结论。旧基线归一化协议仍未知，干预只是checkpoint依赖性而非重训消融。
> **2026-09-24 最新完成状态：V1归一化修正复跑已完成。** `pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed44`固定epoch3 Validation58.57 Overall、90.2080 Yes/No、27.03 Free-form、where66.7482；对原V1约+1.02 Overall。下文“已授权下一次运行”现为历史状态，不再重复启动。当前待办是提取已生成summary的精确分数、新旧V1及既有基线的配对比较、最终训练metadata/诊断；尚无本次配对CI，不复用旧CI。原57.5491偏差版本保留，旧基线协议未知。不新增训练、其他seed或Test。
> **2026-09-24 已授权下一次运行：V1 seed44仅修正累积归一化。** 独立实验名`pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed44`；从头训练，唯一训练改动为初始化后设置`trainer.model_accepts_loss_kwargs=False`。保持V1结构、初值、数据seed42、模型seed44、batch2/累积16、3 epochs、学习率组、AdamW、裁剪1、3% warmup/线性衰减和原图文/生成协议；固定epoch3完整Validation，不跑Test或其他seed。启动前强制读取已通过的K16/K3诊断JSON，核对当前PyTorch/Transformers/Accelerate与审计环境一致，不一致则先重新审计而非直接训练。第一配对参照为旧V1偏差版本57.5491，再列V0/CoCoOp/Static P20并明确旧协议未知。输出独立目录，保留训练loss、裁剪前总范数、裁剪后各组范数、偏移/P20、地图、耗时和显存；新旧loss不可直接按日志数值比较收敛。完成后立即追加两份账本并停止，不自动调整或补实验。当前Windows不能SSH直连AutoDL，本地提交推送后由用户执行给定命令。

> **2026-09-24 已完成前置条件：V1数值审计通过；历史五项协议证据不足。** 同logits有效token平均CE与`outputs.loss`均1.68652809；带/不带窗口`num_items_in_batch=228`的模型loss相同。完整K16原Trainer裁剪前梯度为手工等权参照15.9949倍，单设`model_accepts_loss_kwargs=False`后1.0001倍；DataLoader实际尾K3为2.9981/0.9999倍。数值检查均通过，诊断不执行optimizer step、不重训、不评估。实际审计提交`e42ae4608bcbb93137b3c9e410a565f03800237e`，环境PyTorch2.8.0+cu128/Transformers5.0.0/Accelerate1.12.0。历史V0/CoCoOp/Static/QDPT/LoRA seed44产物均未提取到各自当时的提交或运行版本，不能确认同V1协议或判其结果无效；当前源码只提供风险线索。原V1 Validation57.5491保持“梯度累积归一化偏差版本”标记，不修改历史分数。

> **2026-09-24 V1完成状态（历史记录）：** seed44、固定epoch3 Validation已完成，Overall57.5491、Yes/No89.4080、Free-form25.78、where62.8362；输出后缀`20260924_1`，保留梯度累积归一化偏差标记。两份账本已记录完整结果和配对CI。当时下一步为手工CE、完整/尾部累积窗口梯度数值复核，以及旧V0/CoCoOp/Static/QDPT/LoRA实际版本和loss路径审计；前两者已由上条完成，历史部分仍待证据。修正复跑须获得该次运行授权；当前不新增结构扫描、其他seed或Test。

> **2026-09-24 V1只读审计实施中：** 用户明确授权必要的服务器诊断前向/反向，但当前Windows环境无法SSH直连AutoDL；本地完成诊断代码、静态检查和提交推送后由用户在服务器运行，返回JSON和摘要才更新审计结论。诊断以完成的`_20260924_1/checkpoints/epoch_3`为唯一模型来源，复用实际Trainer构造的DataLoader批采样器，检查完整16个和实际尾窗口，不执行optimizer/scheduler更新或真实梯度裁剪。机器可读结果必须含同logits手工CE、带/不带`num_items_in_batch`的loss、每组及整体梯度比较、裁剪副本、参数哈希、源码/版本定位和五个历史seed44产物的证据表。已预备独立的`pathvqa_v1_loss_corrected_seed44`入口，只在显式提供通过审计的JSON时才允许启动；其代码只设置`model_accepts_loss_kwargs=False`，本次不调用、不修改历史分数。服务器审计完成后立即按证据追加两份账本，并把本段改为完成状态。
# QDPT 返修计划：优先解决多随机种子稳定性

> **2026-09-24 最高优先级：V1 loss/梯度累积归一化审计。** 当前已在服务器运行的`pathvqa_v1_visual_selection_prefix_p20_seed44`（输出目录后缀`_20260924_1`）保持原提交和训练配置，完成epoch3与原定Validation；标记为“梯度累积归一化偏差版本”，不覆盖日志。服务器报告Torch2.8.0+cu128、Transformers5.0.0、Accelerate1.12.0，直接forward首批loss3.89144063。先用同一logits和扩展labels核对因果错位有效token平均CE，再在固定参数、相同16个microbatch及末尾实际3个microbatch的窗口中，对照手工等权microbatch均值、现有Trainer路径、仅设置`model_accepts_loss_kwargs=False`的候选路径，核对裁剪前梯度范数和方向，不执行optimizer step。数值通过后才安排单独的修正复跑；不手工再除16、不改Accelerate累积配置、不改为token加权窗口目标。另逐一审计历史V0、CoCoOp、Static P20、QDPT、LoRA的实际运行版本和loss路径；旧结果在未核实前只列风险，不批量宣布失效。数值诊断与后续训练结果按项目规则补两份账本。

> **审计配置补充：** 当前V1训练提交为`4636416ee99768c667377f0c66b5809daf48ffcd`，完整输出为`pathvqa/outputs/visual_selection_prefix/pathvqa_v1_visual_selection_prefix_p20_seed44_20260924_1`。本地已追到V0、CoCoOp、Static P20、QDPT的`forward(**kwargs)`封装和Trainer继承路径，说明在Transformers 5.0.0下有同类风险，但这些旧run当时的实际库版本尚未证实；LoRA采用PEFT包装，尤其不能按Prompt封装推断。数值脚本在V1完成后只读检查其checkpoint，并从各历史`train_report.json`/`train.log`提取明确记录的版本和提交；没有版本证据就标记未知，不能把诊断时的当前环境版本追溯到旧run。修正复跑须等待上述全窗口、末尾不足16步窗口和手工CE复核通过，且须由用户明确授权该次服务器训练。

> **当前优先项（2026-09-24）：修复V1首跑误杀并重跑同一配置。** `pathvqa_v1_visual_selection_prefix_p20_seed44` 首次在49/1845步被BF16反算增量的固定0.01阈值错误中止，无checkpoint和Validation分数；这是监控口径问题。仅移除错误阈值、保留该值为描述性日志及有效的有限值/注入范围检查，不改V1前向计算、初始化、优化器或总预算。重新从step0运行model seed44/data seed42、3 epochs、固定epoch3完整Validation，仍不扫配置、不补seed、不跑Test。对照修复V0 56.7503、CoCoOp 57.4053、Static P20 54.8650。

> **V0 v2诊断状态：** 修复后正常基线56.7503；`offset_off`49.2251、`condition_off`51.9412已经写入两份账本。128样本前向探针统计尚未收到，不阻塞已获授权的V1单次实验；未来若回传，再据实补录。

> **2026-09-23 V0问题mask修复v2（已实现，待服务器正常Validation）：** v1已证明边界token虽然位置数一致，但完整prompt会把训练的末尾问号ID30改分词为跨换行ID5267，因此“目标位置ID必须等于训练ID”本身不可成立。v2将条件源与注入位置显式拆开：条件分支始终读取独立分词的真实问题ID，严格复现训练来源；偏移写入完整prefill中与问题字符范围相交的位置，且两边token数必须一一对应，否则立即失败。完整生成prompt和解码协议不变。下一步仍只用原seed44 epoch3 checkpoint运行正常PathVQA Validation并与旧55.6479预测作图像簇配对比较；不运行`offset_off`/`condition_off`、不训练、不评估Test或其他seed。目标仍为`pathvqa_v0_seed44_mask_fixed_validation`。

> **2026-09-23 V0 单次实验已完成：** 独立实现问题引导三层视觉选择＋共同原生 Value＋ADePT 风格真实问题偏移。固定 `r_q=128`、`r_delta=192`、三块各64、depthwise k3＋pointwise 残差卷积、P20＋索引17的S8/A_v10 Visual18；偏移输出层 `Normal(0,1e-4)`、零 bias。PathVQA seed44/data seed42、3 epochs、固定epoch3 Validation 为 **55.6479 Overall**，实核可训练参数 **2,356,675**。结果已计入两份实验账本；下一步只读提取首批梯度/偏移与地图诊断，并对照性能缺口。暂不排 Test、其他seed或候选消融；后续服务器实验须重新获得该次明确授权。

> **2026-09-23 阶段调整说明：** 用户将当前探索重点调整为低参数新接口的可用性与明确性能收益，随后再考虑多 seed 稳定性。此探索不以 59/59.5/60.8 或旧 QDPT 多 seed 门槛作为第一轮硬要求。具体设计见 `QDPT_DESIGN_DISCUSSION_NOTES.md`；V0 配置已在上文冻结，冻结 Q/K 等替代接口仍是未排期候选。下文保留旧返修阶段规则与已完成实验历史；旧规则中的“先稳定再压参/不考虑新接口”不作为本次已授权实现的限制，正式主模型替换仍须另行验证。固定 10% 晚启动路线继续关闭，仅保留为未来新结构的未排期退路。

> 更新日期：2026-09-22
> 当前状态：旧实验计划已完成；返修阶段只聚焦 QDPT 稳定性。
> 当前唯一可执行项：利用现有 checkpoint 定位 seed 方差来源，不先训练新模型。
> 事实来源：完整实验以 `EXPERIMENT_RESULTS.md` 为准，简要结论以 `result.md` 为准。

## 1. 旧计划极简摘要

### 2026-09-23 新接口探索的可选消融（未排期）

- 当前 V0 采用 `QDPT_DESIGN_DISCUSSION_NOTES.md` 的完整公式：轻量问题上下文/池化 → 三层问题条件地图 → 共同原生 Value 的三份摘要 → 分层条件 ADePT 问题偏移。配置与本地实现已获授权；服务器运行另需用户明确授权。
- 用户要求登记的后续候选按笔记 ID 管理，均未排期：A1～A5 为视觉条件/地图/问题引导/卷积/多层必要性对照；B1～B6 为层权重、提前融合、初始化、容量、额外 Prompt 接口和冻结 Q/K 替代；C1～C6 为各层 Value、多槽、后段视觉 Prompt、出口视觉 Prompt、视觉残差及其问题条件版本。等级与触发条件以笔记表格为准，不批量执行。D 类保持归档。

- [ ] 在轻量新建 Q/K 打分器形成可用基线之后，可比较复用对应 ViT 层冻结 Q/K 的替代读取器。用户明确要求保留此项；不是首次主配置、当前不启动。须匹配读取位置、问题条件化、摘要/偏移网络和训练预算，并核实原生归一化、多头及位置处理，分别报告可训练参数与计算成本。具体公平控制仍待设计，不因“复用”预设收益。

上一轮计划已经完成方法冻结、三随机种子复现、PathVQA/SLAKE/电气数据集评估、Prompt 基线、LoRA 参考、机制消融、稳定性分析以及训练吞吐测试。历史过程不再保留在本文件中，完整记录见实验账本。

当前论文方法与边界：

- 最终方法为 Dense D768 Sandwich QDPT，共 7,805,184 个可训练参数。
- PathVQA Validation 三 seed 为 60.7765 / 57.2935 / 59.3865，均值 59.1522 +/- 1.7528。
- PathVQA Test 三 seed 均值为 58.8827 +/- 1.8141；LoRA-r8 约为 59.6628 +/- 0.0232。
- SLAKE Test 上，QDPT 为 76.95 +/- 0.44，LoRA-r8 为 81.82 +/- 0.22。
- QDPT 的明确优势是受控训练吞吐约为 LoRA-r8 的 1.881 倍，峰值 allocated 显存低 18.18%；当前不存在参数量优势。
- Static Prompt Overall 标准差为 0.2244，CoCoOp-style 为 1.1367，QDPT 为 1.7528；高方差是当前条件动态 Prompt 路线的实际缺陷，不是所有 Prompt 方法的共同问题。
- 输出头降秩、增加视觉动态写回、延长训练和追加 Prompt 排列均未得到可靠改进，不再恢复这些旧方向。

旧计划到此归档。历史结果不得通过本文件改写；任何更正只能追加到实验账本。

## 2. 返修目标与优先级

### 2.1 主目标

在不降低 QDPT 多 seed 平均性能的前提下，显著降低 PathVQA 的随机种子方差。

预注册成功标准：

- PathVQA Validation 三 seed Overall 样本标准差从 1.7528 降至 **不高于 0.70**；理想目标不高于 0.50。
- 三 seed Overall 极差从 3.4830 降至 **不高于 1.50**。
- 三 seed Overall 均值相对 59.1522 的下降不超过 **0.30 个百分点**；优先要求不下降。
- Free-form 均值不得出现超过 0.50 个百分点的系统性下降。
- 不能依靠删除低 seed、报告最佳 seed、改变数据划分或根据 Test 选择配置获得稳定性。

若稳定性明显改善但均值下降 0.30 至 0.60，记为灰区结果，不直接替换论文方法；必须结合逐样本配对统计和 Free-form 表现决定。下降超过 0.60 则判定失败。

### 2.2 次目标

稳定版本成立后，尝试在不扣分、不重新引入高方差的前提下减少参数量。

- 第一门槛：参数量不高于现有 7.805M。
- 优秀目标：低于 LoRA-r8 的 7.078M，同时保持主目标中的均值与方差门槛。
- 参数压缩不是本轮投稿的必要条件；任何压缩只允许在稳定版本之后进行一次受控尝试。

### 2.3 当前不作为返修主线的问题

- Static Prompt 为什么能释放冻结 MLLM 的已有能力；
- 视觉证据利用与答案空间校准的详细分解；
- SLAKE 上追平 LoRA；
- 全新 Prompt 接口或完整下一代模型。

这些问题保留为论文讨论或后续研究，不得拖延本轮稳定性返修。

## 3. 总体原则

1. **先定位、后干预**：先利用现有三 seed checkpoint 分析方差来源，再决定最小修改。
2. **只改一个因素**：每次只控制初始化、优化阶段或动态输出约束中的一项。
3. **不从文献挑模块拼装**：近期论文只帮助解释现象，不直接成为候选结构。
4. **优先训练策略和初始化修正**：在现有结构上能解决的问题，不升级为新架构。
5. **先跑极端 seed**：新方案先验证当前最好 seed44 与最差 seed45；只有差距明显收敛且均值不降，才运行 seed46。
6. **不使用 Test 选方案**：开发和稳定性判断只用 PathVQA Validation；最终配置冻结后才运行 Test。
7. **不在 SLAKE 调参**：PathVQA 三 seed 通过后，原样迁移 SLAKE 检查是否引入回归。
8. **完整留痕**：完成、失败和负实验都同时写入 `EXPERIMENT_RESULTS.md` 与 `result.md`。
9. **Windows 为唯一事实来源**：本地编辑、测试、提交和推送；服务器只在用户明确授权具体运行后 fast-forward pull。

## 4. 阶段一：用现有结果定位不稳定来源

本阶段不训练新模型，不修改 checkpoint。

### 4.1 样本级方差地图

对最终 QDPT seed44/45/46 的 Validation 预测进行统一分析：

- 三 seed 预测文本一致率与正确性一致率；
- 三 seed 全对、全错、仅一个 seed 正确、仅两个 seed 正确的样本数；
- Overall、Yes/No、Free-form及 how/what/where/when/why/other 的 churn；
- 按图像簇统计方差，确认波动是否集中于固定图像或固定问题类型；
- 区分真正语义错误、答案格式差异和 Exact Match 表达差异；
- 计算 seed44 对 seed45、seed44 对 seed46、seed45 对 seed46 的 exclusive-correct counts 与配对统计。

目标：确认方差主要是全局性能漂移，还是少数开放题/空间题的能力交换。

### 4.2 训练轨迹与终点表征

统一提取三个 seed 的：

- P20、文本 Anchor、Q10、Z10 与动态 Prompt 的范数轨迹；
- 视觉注意力熵、slot cosine、Cross-Attention delta/query、text delta/anchor；
- P20、Anchor 和生成器参数的有效秩、方向相似性及跨 seed 对齐；
- 训练损失、梯度裁剪频率和各分支梯度范数。

不能仅凭“seed45 范数较小”得出因果结论。需要判断较差 seed 是：

1. 从训练早期进入不同尺度轨迹；
2. 生成了方向不同的动态 Prompt；
3. 视觉检索更加弥散；
4. 静态 P20 与动态分支形成了不同的互相补偿关系。

### 4.3 最小模块交换

先只使用 seed44 与 seed45 checkpoint，执行推理级模块交换：

- `P20_44 + rest_45`；
- `P20_45 + rest_44`；
- `Anchor_44 + generator_45`；
- `Anchor_45 + generator_44`。
- `Visual18_44 + rest_45`；
- `Visual18_45 + rest_44`。

必要时再增加“P20+Anchor整体交换”，但不预先展开完整组合矩阵。

解释边界：

- 换 P20 后性能随 donor seed 转移：静态 Prompt 轨迹是主要来源。
- 换生成器后性能随 donor seed 转移：动态生成器是主要来源。
- 换 Visual18 后性能随 donor seed 转移：Layer17 静态视觉 Prompt 的初始化或优化轨迹是主要来源。
- 所有交换都明显崩溃：主要问题是强共适应，不能归咎于单个模块。
- 交换影响很小：继续检查优化随机性、数据顺序或无法由终点参数解释的轨迹差异。

模块交换只用于诊断共适应，不作为公平性能模型，也不进入主结果表。

执行状态（2026-09-22）：六项交换均已完成。P20与文本Anchor双向交换显著破坏性能，说明文本侧存在强seed特异共适应；Visual18双向交换仅下降0.3036/0.2077，交换后仍保留原seed差距约97%，因此排除保存的`S8+A_v10`视觉Prompt终点张量为方差主因。该诊断不能继续区分P20、Anchor和动态生成器中的唯一根因。

### 4.4 阶段一交付物

- 三 seed 样本级稳定性报告；
- 训练轨迹与终点表征对照；
- seed44/45 最小模块交换结果；
- 一项明确结论：方差主要来自静态起点、动态生成器、视觉检索，还是模块强共适应；
- 两个实验记录文件同步更新。

阶段一没有得到可区分结论时，不直接设计复杂结构；优先补最小诊断。

## 5. 阶段二：按诊断结果选择唯一稳定化干预

本阶段不预先绑定某篇论文或某个模块，只根据阶段一结论选择一条路线。

### 5.1 若主要来自 Static P20 轨迹

优先验证“稳定领域锚点 + 条件增量”的解耦训练：

1. 先按现有 Static Prompt 协议训练 P20；
2. 将训练后的 P20 作为 QDPT 的领域锚点；
3. 第二阶段冻结 P20 或使用显著更低的学习率，只训练条件分支；
4. 每个模型 seed 必须包含自己的完整两阶段流程，不能用一个最佳 seed 的 P20 初始化全部运行后再把结果冒充独立多 seed。

冻结与低学习率只能选择一个作为首次控制，不同时扫描。

### 5.2 若主要来自动态生成器

保持 Prompt 数量、位置和视觉接口不变，只控制一种自由度：

- 统一、数据相关但 seed 可复现的初始化；或
- 对动态残差施加明确的尺度边界；或
- 分阶段打开动态分支。

首次实验只选择其中一项。不得同时增加新损失、低秩头、门控和额外归一化。

### 5.3 若主要来自模块强共适应

优先采用分阶段优化，让静态领域校准先稳定，再训练条件增量。目标是减少 P20、Anchor 和生成器相互代偿，而不是继续增加模块。

### 5.4 若主要来自 Prompt 接口扰动

这意味着问题超出低风险返修范围。只有前述初始化与分阶段优化均失败时，才考虑改变接口；任何长度保持或注意力内注入方案都视为下一代方法，不与本轮快速返修混在一起。

## 6. 阶段三：两 seed 快速门槛

选定唯一干预后，先运行 seed44 与 seed45，保持：

- PathVQA 官方训练与 Validation；
- 数据 seed42；
- 三 epoch 总训练协议，除非干预本身是明确的两阶段训练；
- 相同图像处理、模板、生成设置和评价脚本；
- 不使用 Test。

进入 seed46 的必要条件：

- seed44/45 平均 Overall 不低于当前两 seed 平均 59.0350 超过 0.30 个百分点；
- 两 seed 差距从当前 3.4830 降至不高于 1.50；
- Free-form 不出现两个 seed 同方向明显下降；
- 训练、梯度和参数均有限，无隐藏的 checkpoint 选择或额外调参。

不满足即停止该干预并记录负结果；不连续叠加补丁抢救。

## 7. 阶段四：三 seed 正式稳定性验证

两 seed 门槛通过后补 seed46，并正式计算：

- Overall、Yes/No、Free-form的三 seed mean +/- sample SD、range和worst seed；
- 与原 QDPT 的逐 seed、均值和方差对比；
- 三 seed 合并后的样本级 bootstrap 或分层分析；
- 参数量、训练时间、吞吐和峰值显存；
- 改进是否来自修复最差 seed，而不是牺牲最佳 seed 后机械压缩方差。

只有同时达到第2.1节的均值与方差门槛，才能替换论文主方法。

配置冻结后，对三个 checkpoint 各运行一次 PathVQA Test。不得根据 Test 结果回到 Validation 修改方案。

随后原样迁移 SLAKE 三 seed，只检查明显回归，不针对 SLAKE 调参。电气数据集是否重跑由论文需要决定，不作为稳定性结论的必要条件。

## 8. 阶段五：可选参数优化

仅在稳定版本完成后考虑。目标是减少参数而不破坏已经获得的稳定性。

规则：

1. 只允许一个由稳定化结果直接支持的压缩假设。
2. 先跑 seed44/45；平均下降超过 0.30 或两 seed 差距重新超过 1.50，立即终止。
3. 不重复已失败的简单 R256 输出瓶颈，也不扫描多个 rank。
4. 参数少于 LoRA-r8且性能、方差不下降，视为优秀加分项；否则保留未压缩稳定版本投稿。

## 9. 暂缓的视觉利用诊断

“Frozen Base/Static Prompt × 正确图像/错配图像”的差分实验仍有研究价值，可用于区分视觉利用与答案空间校准，但不再是本轮返修 P0。只有以下情况才恢复：

- 稳定化干预需要判断 Static Prompt 的具体功能；
- 论文审稿意见明确要求机制解释；
- 稳定性问题解决后准备下一代方法。

## 10. 当前执行顺序

- [ ] 审计三 seed 最终 QDPT Validation 预测、summary、diagnostics 和 checkpoint 是否齐全。
- [x] 编写只读的三 seed 样本级稳定性分析，不改模型（待服务器实际运行）。
- [ ] 编写训练轨迹与终点表征对照。
- [x] 按 checkpoint 保存契约实现并完成 seed44/45 的P20、文本Anchor与Visual18双向模块交换。
- [x] 为模块交换加入严格参数来源审计；实际六项运行均通过形状与来源检查。
- [ ] 本地完成静态检查和测试。
- [x] 用户授权后完成服务器推理级模块交换，并将完整结果计入两份实验记录。
- [x] 阶段一现有诊断与首个稳定化控制结果已更新至 `EXPERIMENT_RESULTS.md` 与 `result.md`。
- [x] 首个稳定化控制已停止并关闭：seed45冻结同seedStatic P20后Overall为55.0248，较原QDPT seed45下降2.2687且几乎退化至Static P20基线；seed44因旧checkpoint缺少seed元数据在训练前失败。按止损规则不修该兼容问题、不补跑seed44/46，不连续叠加补丁抢救。

## 10.1 第二个稳定化控制：动态分支晚启动（已实现，待授权运行）

- [x] 在最终Dense D768 Sandwich QDPT上只改变动态分支启动时间：总预算仍为3 epochs、data seed42，不加载预训练P20，不改变结构、Prompt数量/位置、初始化、各组基础学习率或全局3% warmup+linear-decay scheduler。
- [x] 定义文本动态Prompt为`D=A_t+g(t)Delta_theta(I,q)`；前10% optimizer steps取`g=0`，只允许P20、A_t10和Visual18更新。动态生成器参数在这段时间梯度置为`None`，从而不执行AdamW状态、动量或权重衰减更新；之后一次性切换为`g=1`并恢复原联合训练，不重启scheduler、不补步数。
- [x] 已实现启动与边界审计：文本输出头末层权重/偏置必须严格为零；动态分支参数量必须为7,709,952；开启前参数SHA-256必须不变且输出头仍为零；首次开启前原始动态残差与残差/Anchor比必须为零。审计文件同时记录边界前后loss、静态/动态分支梯度、动态残差/Anchor比和视觉注意力熵。
- [x] seed45 epoch3 Validation为57.8048 Overall，较原seed45回升0.5113并通过首轮门槛，因此按预注册顺序继续运行seed44；未评估Test。
- [x] seed44为57.2136；两seed均值57.5092、分差0.5912、Free-form均值25.1595。虽然seed45回升且分差达标，但Overall均值和Free-form均值均未达门槛，稳定主要来自seed44下降3.5629，判定路线失败且不补seed46。
- [x] 关闭固定10%晚启动路线：不扫描比例、不叠加Gate/正则/损失、不评估Test。结果已同步至两份账本；动态边界审计精确值仍待从服务器审计文件提取补全。

## 11. 停止规则

1. 阶段一未完成前，不训练稳定化变体。
2. 一次只运行一条稳定化路线，不并行扫描多种初始化、学习率、Gate或损失。
3. 不通过牺牲最佳 seed、大幅降低均值来制造较小标准差。
4. 不以两 seed 稳定直接代替三 seed 正式结果。
5. 不使用 Test 选择结构、训练轮数或 checkpoint。
6. 不把注意力图、范数或相关性单独当作因果证明。
7. 不因结果不符合预期而更换数据划分、评价指标或后见阈值。
8. 不删除、覆盖或静默修正旧实验；所有更正追加记录。
9. 未经用户明确授权，不在服务器启动训练或推理实验。
## 2026-10-01 五项固定Test串行：代码准备，待用户启动

顺序固定PathVQA原五轮归一化V1 seed44/45/46 epoch5完整Test（不训练）→RSVQA静态Visual18+P20 seed44五轮epoch5 Test→RSVQA原CoCoOp-style seed44五轮epoch5 Test。入口`run_five_task_test_suite.sh`，不加入其他队列，失败即停，无自动重试或关机，不覆盖旧产物。全部GPU由用户执行。

PathVQA seed44绑定账本`pathvqa_v1_norm_fixed_5ep_seed44_20260926_2`；seed45/46账本尚缺精确目录，入口只枚举精确实验名日期/后缀，逐一检查训练报告、归一化、种子、5epochs、原版布局/参数、原Validation成绩及固定epoch5文件，并要求唯一合法匹配；不按最新选。执行前保存绑定清单及权重SHA256，身份缺失或歧义在首项GPU之前停止。

RSVQA两项microbatch4/累积8、有效32，seed44/data42、5epochs/3%warmup线性、显式loss kwargs=False。静态69632参数，原S8/Av10与P20三档LR；CoCoOp873120参数，只P20+原2560→160→2560 Meta-Net，无Visual18/问题条件，LR .3/3e-4。本条覆盖早期建议CoCoOp加Visual18的草案，不实现该变体。复用active过滤、原始答案监督、count区间评分。原LoRA是3epochs，本次和V1是5epochs，比较明确不同预算。

所有产物在`outputs/five_task_test_suite/suite_*`，逐项保存状态、退出码、执行commit、配置/报告、预测、成绩、成本及两账本/计划追加片段。为守Windows唯一源规则，不在服务器修改受Git管理账本；回传后用CPU-only `--import-results`在本机追加，不做仅账本提交。已匹配完整Test结果可复用，严格核验协议并记录，部分/身份不符结果不能跳过。PathVQA三seed报告均值±样本SD；RSVQA两新基线vsV1及V1vs3epLoRA做10000次seed42图像簇配对OA/AA与四类统计。此次未运行GPU，尚无新实验成绩。
## 2026-10-01 五项入口预检兼容修复：待用户重启

首次用户启动在seed44原报告缺`visual_prompt_mode`处停止，未进入GPU，0/5任务完成；失败记录`precheck_failed_20261001_133024_919726`已追加两份账本。最初五轮schema另缺Av10 LR与分组LR，已一并兼容：明确原V1报告/checkpoint method确认原布局，学习率从原训练日志或报告记录的训练commit取证，缺失或矛盾仍停止。CPU回归检查后提交推送，仍用`run_five_task_test_suite.sh`；原顺序、五轮、固定Test、失败即停与不自动关机全部不变。全部GPU由用户执行。
## 2026-10-01 SSH核实后的精确绑定（覆盖此前路径未知状态）

用户授权SSH只读确认，已验证三套真实原五轮V1 epoch5。入口固定seed44=`pathvqa_v1_norm_fixed_5ep_seed44_20260926_2`、45=`pathvqa_v1_norm_fixed_5ep_seed45_20260926`、46=`pathvqa_v1_norm_fixed_5ep_seed46_20260926`，保留逐项元数据和CRC/SHA256审计。CPU真实产物校验通过，RSVQA两套已有Test评分也已复核；不导入torch、不执行GPU、不写服务器源码。兼容修复与精确路径即将本机提交推送；等待用户同步后亲自重启原五项入口，无自动关机。
## 2026-10-09：七轮实验已完成，负结果；剩余一试待选

用户回传5/6/7轮完整Validation及配对，Overall56.4307/55.7437/55.8396，固定epoch7主结果；对原五轮修正版 -2.8439pp，CI[-3.7909,-1.9147]。本次授权结束，覆盖下方尚未运行状态。已补两账本；不追加轮数、重训、seed或Test。保留原五轮修正版及V1；最后一试待用户明确选择。
## 2026-10-09 五轮/七轮既有轨迹CPU对比已完成

精确绑定102158_643548及151030_637257，报告与10张叠加图/机器JSON在pathvqa/outputs/v10_5ep_7ep_cpu_comparison_20261009/，两账本已追加。原五轮/七轮均已完成，覆盖下方历史“尚未运行”。仅读取既有产物；不同调度在前五轮已分岔，七轮末段loss改善不伴随开放回答改善，未证实具体根因。无GPU操作，不补缺失探针、不安排新训练，等待用户决定。CPU分析脚本及账本保留本地，随下一次相关代码提交，不做仅账本提交。
## 2026-10-09：最后一试授权，修正版五轮仅降低P20 LR

用户同意单变量探索：PathVQA model seed44/data seed42，从零5轮，唯一改动P20基础LR0.3→0.1；完全沿用成功修正版5轮结构、初始化、其余组LR（Meta-Net3e-4、S8 3e-5、Av10及其余1e-4）、batch2/accum16、归一化False、3%warmup/五轮linear及clip1。预定实验名 `pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_p20lr0p1_norm_fixed_5ep_seed44`；尚未实现/运行。保存3/4/5，固定epoch5完整Validation为主结果；对精确绑定的修正版五轮20261009_102158_643548（58.6835）及V1五轮seed44 20260926_2（59.3386）做图像簇配对10000/seed42和五项分组。保留相同曲线诊断并与原五轮CPU对比，不改结构/初始化/轮数/其他LR，不做混合地图候选。实现必须使用独立配置/入口，不修改历史默认值或覆盖产物；所有GPU操作用户执行。失败即停，不追加seed/Test/重试/关机。本次为最后一试，结果回传后补两账本及计划，不自动延伸实验。
## 2026-10-09 最后一次实验：V10修正版五轮P20 LR0.1（已准备，尚未运行）

独立实验 `pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_p20lr01_norm_fixed_5ep_seed44`，仅P20基础LR .3→.1，从零PathVQA44/42、五轮保存3/4/5、固定epoch5完整Validation，1691043参数。复用原模型/初始化/训练模块；Meta-Net3e-4、S8 3e-5、Av10及其他组不变，batch2/累积16、归一化False、clip1、3%warmup/五轮linear不变。独立薄训练入口和脚本，原默认配置保持。精确绑定修正版五轮102158_643548（58.6835）及原V1五轮20260926_2（59.3386），复用已有配对10000/42、Overall/Yes-No/FF/what/where流程；记录实际组LR与初始化审计。

所有GPU预检/训练/评估用户执行：`bash pathvqa/run_v10_head_fixed_p20lr01_seed44.sh`。独立v10_head_fixed/<新实验>_<timestamp>输出，失败即停，无重试/关机/Test/其他seed。完成后两账本追加，复用CPU曲线比较原五轮（重点早期P20/偏移、层权重、loss、有效地图熵），不新增模型探针。最后一次探索，不自动追加。
## 2026-10-10：最后一试P20 LR0.1完成，停止追加探索

修正版五轮PathVQA seed44 P20 LR0.1 Overall58.3160；对原修正版 -0.3675pp，CI[-1.1936,0.4482]，未获明确收益；对V1 -1.0225pp，CI[-1.8899,-0.1622]。已补两账本，覆盖下方准备/待运行状态。本次为用户指定最后一试，授权已完成，不自动追加训练/seed/Test/关机/重试。V1保留为论文主方案，修正版五轮58.6835为探索结果；学习率根因未证实。
## 2026-10-10 V10修正版跨数据集：SLAKE→RSVQA-LR（准备完成，尚未运行）

用户放弃PathVQA P20 LR0.1方向：删除两份专用入口及共用调度中的low-LR分支，保留历史58.3160记录、预测和checkpoint；代码可由Git恢复，不删除实验产物。新授权仅两项：`slake_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44` → `rsvqa_lr_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44`，均从零44/42、五轮、保存3/4/5、固定epoch5官方完整Test（SLAKE双语），不额外Validation或seed。沿用PathVQA成功修正版模型、初始化、1691043参数、P20 .3/Meta3e-4/S8 3e-5/Av10 1e-4/LN及其他1e-4、batch2/累积16、clip1、3%warmup/linear、归一化False；不是RSVQA旧4/8配置。

最简实现：两个薄训练入口，共用slake.train_v10增加RSVQA已验证Dataset/Collator及预检prompt适配；复用v10-head-fixed加载器/官方评估。RSVQA active过滤、原始答案监督、count官方区间评分不变。用户启动 `bash run_v10_head_fixed_slake_rsvqa_shutdown.sh`；Bash按顺序train→epoch5Test，失败即停、不重试。独立slake/outputs/v10_head_fixed、RSVQA/outputs/v10_head_fixed产物，suite日志/阶段退出码/最终状态与汇总在outputs/v10_head_fixed_slake_rsvqa/<stamp>/。EXIT trap先落盘并sync，再Bash调用/usr/bin/shutdown，成功或失败均立即关机，调用失败日志另存；不使用Python直接exec或十分钟延迟。全部GPU用户执行；回传后补两账本。不自动追加其他任务。
