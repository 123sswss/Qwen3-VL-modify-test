# 一周投稿实验计划

## 2026-08-27 方向修正：Dynamic Multimodal Prompt

- 当前主线从视觉层内部 MMRL 转为冻结完整 VLM 主干的 Dynamic Multimodal Prompt Tuning；旧 MMRL、Relation 与 Utility Gate 计划保留为历史方案，不再阻塞新方法验证。
- 方法必须单阶段训练：20个 Static Prompt 与图文条件动态 CA 从同一初始状态联合优化，禁止加载训练完成的 Static Prompt checkpoint 作为主方法 warm-start。
- Static Prompt 同时作为 LLM 前缀基底与 CA Query；图像 Merger 后视觉 Token 和仅问题侧文本 Token 分别 Mean Pooling 为两个 Memory，监督答案 Token 必须从文本 Memory 严格排除。
- Dynamic CA 首版使用256维、8头和零初始化输出投影，产生20个 Prompt 残差；最终仍只向 LLM 入口插入20个 Token，不增加 Static Prompt 基线之外的序列长度。
- 视觉编码器、视觉 Merger 和 LLM 全部冻结；Static Prompt 使用LR0.3，动态 CA 首轮使用LR3e-4，两组共享3轮线性调度与validation选模，训练参数预计2,685,440。
- 首个决策实验固定为 PathVQA seed44。成功线为显著超过 Static Prompt55.83，重点检查 Free-form是否超过21.27并接近或超过last8 Visual LoRA23.41；若不超过Static Prompt，不追加多seed。
- 必须记录动态残差/静态Prompt范数比、图文注意力比例、注意力熵、两组梯度范数、参数量和延迟；首步必须审计动态残差精确为0。
- 回退仅保留为后续附加性质：可报告Attention Mask关闭Prompt后的约98% Base输出一致性，但不作为当前方法成立条件，也不在首轮实验中实现KV-cache受损的动态屏蔽。
- PathVQA seed44 首轮已完成：Dynamic Prompt 为56.54/89.83/23.21，超过 Static Prompt55.83/90.33/21.27，主要收益是 Free-form+1.94；但配对图像聚类 Overall 差值95% CI为[-0.10,+1.55]，单seed不足以声称稳定总分提升。
- 训练诊断显示两Memory注意力在首轮早期迅速硬化，末期熵约0.00010、视觉注意力质量固定为44.375%。该现象不等于动态分支失效，因为Value与残差仍依赖样本，但禁止在因果验证前声称样本级动态路由。
- 三项 checkpoint 因果干预已完成：`delta=0` 为49.29，前32样本平均residual为43.19，lag32错配Memory为48.58，均显著低于正常56.54。正常相对错配Memory的聚类配对95% CI为[+6.99,+8.89]，证明当前样本条件有效；近零注意力熵应解释为固定模态分工，而不是动态分支失效。
- 下一正式优先级是同seed配对运行 PathVQA Static Prompt seed45 与 Dynamic Prompt seed45；两者保持3 epochs和现有超参，先验证Free-form与Overall趋势能否复现。只有seed45仍支持动态收益，才进入SLAKE迁移或额外seed。epoch3 validation仍上升，但5 epochs重训排在seed45复现之后，避免根据seed44 test继续调参。
- 不再追加 seed44 的结构小改。视觉Memory与文本Memory的单独错配可作为后续机制消融，但当前联合错配、零残差和均值残差已经足以证明样本条件性，不阻塞多seed验证。
- 2026-08-28 受控追加一次 seed44 视觉协同小试：保留 Dynamic Prompt20+CA256 主干不变，只在原生视觉层5/11/17（0-based）加入独立插入/立即剥离的局部 Rep 分支。共享8个1024维 Rep，使用CA128/4头读取当前视觉段均值与问题文本均值，以零初始化 `tanh(gamma_l)` 融合，并对三层局部输出施加平均 Relation0.05。新增1,201,155参数、总计3,886,595参数；若不超过当前56.54则停止视觉协同扩展，若有任何正收益再考虑seed45。
- 首轮零初始化运行在step460仍只有 `sparse_visual_delta_ratio_mean=7.69e-6`，Relation缩放项为 `2.72e-11`；分支有梯度但实际视觉改变量落在BF16近零区。重跑时仅把实际初始融合比例从0改为0.05（内部参数为`atanh(0.05)`），Sparse LR仍保持3e-4，其余配置不变；实验名加入`init0050`以避免与首轮混淆。
- `init0050` 重跑已完成：PathVQA seed44 为56.02/88.88/23.12，低于 Dynamic Prompt 的56.54/89.83/23.21。Sparse 分支已有效激活但由第11层主导，未形成主指标互补；不追加seed45，不再调该视觉协同架构。
- 纠正（2026-08-28）：上述“停止视觉协同扩展”结论忽略了完整三轮轨迹。Sparse Visual 在epoch1相对原 Dynamic Prompt 同轮提升Overall +2.01、Yes/No +0.74、Free-form +3.29，随后epoch2坍缩并在epoch3恢复，更符合分支联合优化不稳定而非结构完全无效。允许追加一次严格的学习率诊断，除此之外仍不扩结构、不加seed。
- 待运行 `pathvqa_dynamic_prompt_sparse_visual_layers5_11_17_slots8_ca128_init0050_relation0050_sparse_lr3e5_seed44`：唯一变量是Sparse Visual LR由3e-4降至3e-5；Prompt LR0.3、Dynamic CA LR3e-4、初始化0.05、Relation0.05、结构、seed和数据顺序保持不变。主要诊断是epoch2是否不再从epoch1下降3.23个百分点；若训练轨迹稳定且最终接近或超过56.54，支持“分支更新时间尺度不匹配”的改进方向，否则停止单纯LR调节。
- 上述LR诊断已完成并达到成功线：Validation为54.53 ->55.14 ->56.85，Test为57.51/89.62/25.35；相对原 Dynamic Prompt 的配对聚类95% CI为[+0.20,+1.73] Overall、[+0.84,+3.44] Free-form。仅降低Sparse LR便消除epoch2坍缩并取得显著净增益，确认后续方向是轻量、慢更新的视觉协同，而不是增强视觉分支幅度。
- 旧双路径Sparse Visual的`checkpoints/`与`final/`已按用户要求删除，仅保留日志和评估证据，因此原57.51 checkpoint不再能运行Sparse-off干预。
- 新实现改为标准单路径Rep Token注入：每个锚点只执行一次原生视觉Block，直接计算`Strip(Block([R;h]))`；彻底移除Residual Scale、Relation及其诊断和参数。Sparse参数1,201,152，总训练参数3,886,592，继续固定LR3e-5。新运行目标为`pathvqa_dynamic_prompt_sparse_visual_single_pass_seed44`，实验名显式包含`single_pass`；该实现尚未运行，不能继承57.51分数。
- 算力恢复后的第一优先级是运行上述single-pass seed44，先判断去掉双路径后能否保留低LR收益与降低训练/推理耗时；取消继续扫描Sparse LR。只有single-pass成立后才追加额外seed和同checkpoint Sparse-off干预。

### 今晚单Seed共享S递进实验（已完成）

- 三项全部固定PathVQA seed44/data_seed42、3 epochs、相同数据顺序、20个LLM前缀位置、Dynamic CA256/8头、Sparse Visual层5/11/17、8个1024维共享S、Visual CA128/4头、Sparse LR3e-5、Text CA LR3e-4和validation选模。每项只测试所选最佳epoch一次，不扫描学习率。
- 实验1 `pathvqa_dynamic_prompt_sparse_visual_single_pass_seed44`：单路径新基线。保留独立可学习P（LR0.3）和现有Text CA，不加入S到文本侧的直接桥梁。目标是判断`Strip(Block([R;h]))`能否接近历史双路径57.51，并建立后两项的同实现对照；它不是旧57.51的严格复现。
- 实验2 `pathvqa_dynamic_prompt_shared_s_memory_keep_p_seed44`：保留实验1的独立可学习P；捕获第17层最后视觉锚点的8个条件化Rep，经冻结的普通`visual.merger`按原生2x2规则映射成2个LLM入口空间向量，再按样本均值汇聚为额外S-Memory并加入Text CA Memory。P仍作为20个Query和残差基底。普通Visual Merger的输出原生用于替换LLM入口视觉Token；不使用面向LLM第0/1/2层后残差的`deepstack_merger_list`。冻结Merger不增加训练参数，S通过视觉路径与文本路径共同接收CE梯度，但始终使用自己的3e-5学习率。
- 实验3 `pathvqa_dynamic_prompt_shared_s_memory_no_trainable_p_seed44`：与实验2完全相同，但移除独立可学习P。保留同一初始化得到的20个冻结P0作为Text CA Query/长度骨架，最终前缀为`P0 + TextCA(P0, [visual, text, S-Memory])`；这样唯一核心变量是51,200参数的独立P是否可训练，不额外混入前缀长度或新Token扩展器差异。
- 主要比较依次为实验2-实验1（共享S跨视觉/文本桥梁是否有用）和实验3-实验2（独立可学习P是否必要）。报告Overall、Yes/No、Free-form、问题类型、三轮Validation、聚类配对CI、逐样本fix/break、参数量、训练时间、TTFT、Sparse分支总梯度、共享S参数梯度及S-Memory接口梯度；后两者用于确认文本桥梁参与反传，不冒充严格的逐路径梯度分解。
- 决策规则：实验2若不低于实验1且Free-form有正向净修复，则保留共享S桥梁；实验3若与实验2差距不超过0.5 Overall，可采用更纯粹的共享S叙事，否则保留独立P并将硬共享版本作为必要消融。三项完成前不追加seed、不修改S数量、不自建新MLP。
- 三项实现完成后使用单一串行目标`pathvqa_dynamic_prompt_shared_s_suite_seed44`依次运行，任一项训练、validation选模或test失败即停止后续实验，避免在基线失效时继续消耗整夜算力。
- 实验1完成：single-pass baseline Validation55.06 ->56.21 ->57.52，Test **57.79/89.77/25.77**，超过已删除checkpoint的历史双路径57.51点估计。标准单路径成立并成为新参考。
- 实验2完成：keep-P共享S-Memory为 **56.14/89.02/23.21**，相对实验1下降1.65 Overall和2.56 Free-form，聚类配对Overall CI为[-2.43,-0.92]。桥梁梯度健康且获得23.75%注意力，因此按预设规则删除该方向，不追加seed或结构修补。
- 实验3完成：冻结P0为 **54.71/87.45/21.92**，相对实验2再降1.43 Overall，聚类配对CI为[-2.25,-0.65]；Dynamic residual/base末期均值76.8x。独立可学习P被确认为必要组件，纯共享S叙事停止。
- 后续候选仅保留single-pass baseline的额外seed与同checkpoint Sparse-off机制干预；是否立即运行seed45由下一轮讨论决定，不自动启动。共享S-Memory实验2/3不再追加预算。
- 更正（2026-08-29）：上述实验2/3把第17层视觉Block输出的条件化Rep经Merger回流到Text CA，偏离了原始“未经Visual CA的共享参数S直接分叉到视觉侧与文本侧”设计。56.14/54.71只否定末层Rep回流捷径，不能否定raw-S共享。旧结果保留但撤销“共享S终止”的外推结论。
- 纠正版实验A（本方版本）：保留57.79基线的独立P20、原图文Dynamic CA256和视觉层5/11/17注入不变；仅将raw S经冻结普通`visual.merger`变成2个领域Token，再由独立零输出初始化CA128读取为20个额外残差。最终前缀为`P + Delta_sample + Delta_raw_S`，新增约1.324M参数，S仍只属于Sparse LR3e-5组。
- 纠正版实验B（用户版本）：完全删除独立P及冻结P0；raw S经同一冻结`visual.merger`原生得到2个LLM入口Token，直接作为前缀基底和现有图文Dynamic CA256的Query。视觉侧仍从同一个raw S生成层5/11/17 Rep。禁止增加扩展MLP或偷偷补回20个Query位置。
- 两次固定PathVQA seed44/data_seed42、3 epochs、相同数据顺序与Sparse LR3e-5，串行目标为`pathvqa_dynamic_prompt_raw_shared_s_suite_seed44`；不重复57.79基线。重点记录raw-S文本接口梯度、独立Shared-S CA梯度、两种delta/base、Validation轨迹、Overall/Yes-No/Free-form及逐样本配对结果。
- 纠正版实验A已完成：54.95/88.31/21.54，相对57.79基线下降2.84 Overall和4.23 Free-form，聚类配对Overall CI[-3.67,-2.04]。raw-S残差仅约P的3.95%但梯度健康，视觉CA层11/17转为96.5%/91.2%视觉质量；结果强烈支持跨支路优化干扰，但尚未用stop-gradient区分反向梯度冲突与前向残差内容有害。
- 纠正版实验B已完成：51.64/87.33/15.91，相对基线下降6.15 Overall和9.86 Free-form，聚类配对Overall CI[-7.10,-5.20]。2-token raw-S基底范数约70，Dynamic残差达到其2.83倍；S虽有强CE梯度但仍处于3e-5视觉学习率组，无法替代LR0.3的P20文本锚点。
- 决策：停止raw-S硬共享结构，不追加seed、不扫描S学习率、不增加S数量或Token扩展器。当前支持架构仍是57.79的独立P20 + 图文Dynamic CA + 单路径Sparse Visual；后续若继续追求论文核心，应研究优化层面的软耦合/梯度协调，而不是让视觉与文本直接共用同一个可训练参数。
- 评估协议更新（2026-08-29）：后续PathVQA Dynamic Prompt开发实验固定训练3 epochs，只对`epoch_3`执行一次全量Validation；不再比较epoch1/2、不再自动选择最佳epoch，也不运行Test。`selected_result.tsv`必须标记`fixed_epoch3_validation`，Test仅在架构最终锁定后恢复。Dynamic Prompt因果干预同样默认只运行Validation。
- Shared-S方向按新证据重新开放一次、但不恢复已否定的raw视觉S硬共享：新增`pathvqa_dynamic_prompt_asymmetric_shared_s_seed44`，令20x2560的文本Prompt本身成为共享S并保持Prompt LR0.3；文本侧直接更新S，视觉侧只能读取`detach(LayerNorm(S))`，经可学习20→8 Token Mixer与2560→128→1024低维Adapter生成视觉Rep基底。视觉Adapter、Visual CA和层嵌入保持Sparse LR3e-5，现有图文Dynamic CA保持3e-4；不使用`visual.merger`，不保留另一套独立P或视觉S。
- 上述非对称Shared-S只运行PathVQA seed44，固定3 epochs并仅评估epoch3 Validation，不运行Test。对照为同seed的single-pass基线Validation57.5172；若明显低于基线则彻底停止Shared-S，若持平或产生正收益才讨论seed45。必须记录S梯度、视觉Adapter梯度、Token Mixer熵、Adapter输出范数、Dynamic delta/base及三层Visual CA诊断，以确认“文本写、视觉只读”的优化机制实际成立。
- 非对称Shared-S seed44已完成：固定epoch3 Validation为56.8142/88.2240/25.4946，相对精确基线57.5172/89.7280/25.3989为-0.7030/-1.5040/+0.0957；Overall聚类配对95% CI[-1.4045,+0.0161]，Yes/No退化显著而Free-form完全打平。梯度隔离和Adapter均正常，但Visual CA三层视觉质量由约68%/50%/61%降至29%/27%/29%，Rep/input反而显著增大，说明纯替换把视觉分支文本化并放大。该结构不追加seed45；若仍抢救共享锚点，只允许“保留独立视觉S + 有界stop-gradient文本S残差”这一项新假设。
- 强标准对照`pathvqa_lora_full_model_attention_r8_seed44`已完成：PathVQA seed44/data_seed42，LoRA rank8、alpha16、dropout0.05、LR1e-4，覆盖24层视觉Attention的qkv/proj与36层LLM Self-Attention的q/k/v/o，共192个Linear目标、精确7,077,888个训练参数。固定epoch3 Validation为**59.3386/92.1920/26.5795**，与Full Workspace的Overall统计打平；LoRA显著领先Yes/No1.5680，Workspace显著领先Free-form1.7869。
- 方向调整（2026-08-29）：停止用删除私有参数、低维瓶颈或硬共享换取小参数量。下一阶段先建立高容量性能上限，再做蒸馏、共享权重或低秩化。新结构必须完整保留57.5172 Validation基线的独立P20、原Text CA256、独立视觉S8、原Visual CA128和层5/11/17单路径注入。
- 新增唯一主实验`pathvqa_dynamic_prompt_full_workspace_seed44`：增加32x1024样本级共享多模态工作区Z；Z在视觉层5/11/17依次经过各自独立的全Token Cross-Attention、Self-Attention和FFN4096更新，Memory为该层完整视觉Token与完整问题Token，而非Mean Pooling。原文本与视觉私有支路继续独立工作；Z只通过独立Text CA1024/16头和三组Visual CA1024/16头提供附加残差，不拼入原CA Memory、不替换P或S。四个共享残差头输出投影严格零初始化，使训练第0步等价于当前基线。
- 上述工作区实验固定PathVQA seed44/data_seed42、Prompt LR0.3、原Dynamic CA LR3e-4、原Sparse Visual LR3e-5、Workspace LR1e-4、3 epochs；只评估epoch3 Validation、不跑Test。重点记录Workspace总梯度、三层Z的视觉/文本注意力质量和熵、Z范数、三层视觉共享残差比例、文本共享残差比例及原私有分支诊断。只有Validation超过57.5172且主要收益不只来自Yes/No，才追加seed45并进入降本消融；失败时先根据独立残差头诊断定位，不扫描槽数或维度。
- Full Workspace seed44已完成：固定epoch3 Validation **59.4504/90.6240/28.3663**，相对精确基线为+1.9332/+0.8960/+2.9675；聚类配对95% CI为[+1.1747,+2.7132] Overall和[+1.8602,+4.1721] Free-form，达到预设成功线并成为新开发基线。该结果不跑Test，正式Test仍等待架构与seed复现锁定。
- 下一顺序调整为：（1）不改Workspace结构和核心超参，将Full Workspace迁移到SLAKE并先跑seed44，验证跨数据集泛化；（2）SLAKE方向成立后再复现PathVQA seed45；（3）对PathVQA seed44同一checkpoint增加Workspace Text-off、Visual-off以及Visual residual scale `0/0.1/0.25/0.5/1.0`推理干预，均只跑Validation；（4）跨数据集和额外seed方向一致且确认有效支路后才设计参数缩减。不得提前扫描Z槽数、宽度、FFN或学习率。
- SLAKE迁移实验`slake_dynamic_prompt_full_workspace_seed44`已完成：官方Test **78.37**，相对Static Prompt显著+3.96，OPEN/VQA/EN/ZH均有稳定净修复，跨数据集性能泛化成立。与此同时，Workspace深层更新几乎完全忽略文本且Z32读头近似均匀，76.90M参数效率与“共享多模态工作区”机制尚不成立；保留该checkpoint作为高容量教师和降本参考。
- 最简SLAKE层消融`slake_dynamic_prompt_full_workspace_17only_seed44`已完成：参数34,895,872，官方Test **78.37**，与三层版本逐题70/70互换、McNemar `p=1.0`；所有子组差异均不显著，训练时间下降9.42%。该结果证明层5/11可在重新训练后被第17层替代，但不证明三层checkpoint中的早期路径没有贡献；17-only末窗第17层共享残差/私有Rep升至2.434x，而三层版第17层为1.038x、原第11层为2.080x。
- `slake_dynamic_prompt_workspace_path_interventions`已完成：六次推理均复用三层checkpoint。层5/11 Z更新分别或同时旁路均不降分，双旁路为78.41；层5/11视觉写回分别关闭不降，同时关闭为78.08、仅-0.29且不显著。早期Z更新的因果必要性已被否定，早期视觉写回只保留一个很弱且未证实的联合效应。
- 主结构决策起点转向34.90M layer17-only，不再为保持三层叙事而保留42.00M早期Block/Reader参数。下一次结构实验仍以增效为先：围绕单一Z20同时生成20个视觉Prompt与20个LLM Prompt残差，比较同位置残差叠加与显式concatenate接口；具体结构和控制变量需先讨论定稿，不自动实现或运行。
- 新增受控增效实验`slake_dynamic_prompt_full_workspace_17only_s20_seed44`：以78.37的layer17-only `S8+Z32`为唯一参考，只将私有视觉基底/插入Rep数从8提高到20，Workspace维持Z32，训练参数由34,895,872增至34,908,160。固定seed44/data_seed42、3 epochs、仅epoch3官方Test；用于判断视觉写回是否受8个Rep Token表达空间限制。只有该实验持平或提升，才追加`S20+Z8`测试Workspace槽数压缩；失败则停止组合训练，避免两项改动互相抵消。
- 诊断修复列为无算力代码任务：定位为什么`DynamicPromptTuningModelInterface._capture_workspace_debug()`只保存Text-reader字段，恢复相邻Z的LayerNorm后余弦、每层更新增量余弦和三层视觉delta余弦。修复后不为补诊断单独重跑旧六项；只有后续获授权的新实验自然复用诊断。
- 机制待证：末段Workspace视觉残差约为私有Rep的5.25倍、私有Sparse梯度约为Workspace的1/9，但视觉Block输出范数稳定；所有下游Z读取熵约0.9998。当前可以主张高容量全Token共享工作区有效，不能主张32槽已经形成语义分工，也不能声称私有视觉分支对最终增益不可替代。

## 总体目标

- 主贡献：冻结 LLM，通过 128-slot MMRL 对视觉编码器进行图文条件化垂直专精。
- 安全扩展：Utility Gate 在混合部署中决定是否启用 MMRL，不负责提高垂直数据集上限。
- 主数据集：PathVQA，承担主结果、完整核心消融和视觉专精结论，各正式方法运行 seed44/45。
- 第二正式数据集：SLAKE，承担多语言、知识型与混合问题上的稳健性验证，各正式方法运行 seed44/45，并优先复用已有有效结果。
- 内部验证：自建数据集仅运行 seed44，不调参、不承担核心结论。
- 通用域：LLaVA-150K 提供 Gate 负例候选，VQAv2 固定验证子集评估能力保持。

## Day 1：定稿与缺口审计

- 锁定 MMRL：128-slot Attention Pooling、共享 CA、same-init layer MLP、Relation 0.05。
- specialist 训练必须保留现有 Stage 1 Gate、两套 pooling 及其初始化/随机数路径；不得通过删除 Gate、跳过 Stage 1 后再硬设 `G=1` 来训练。
- Stage 3 仅允许使用已经多 seed 验证过的 `open_full_ce` 协议：保留完整 Gate 路径但令专业 CE 训练的数据流全开。它与 `No gate (G=1)` 消融不是同一个实验。
- 核实 SLAKE Base、视觉 LoRA、Prompt Tuning、MMRL 的正式分数、参数量、日志和 checkpoint；缺失项进入补跑队列。
- PathVQA 原始文件完整性已核对：官方 7/3/3 个 Parquet 分片及 SHA-256 全部匹配，规模为 train 19,654、validation 6,259、test 6,719。
- 完成 PathVQA 内容审计：答案规范、问题类型分布、按图像哈希恢复同图 QA、重复样本和训练测试泄漏。
- 审计自建数据集：只记录规模、来源、划分、质量限制和不可开源原因。
- 整理视觉 Prompt、ViT Adapter、条件 PEFT、样本级路由和安全回退相关工作，产出碰撞对照表。
- 在验证集建立 Base/MMRL 逐样本对照，统计双方正确性转换和 Oracle 上限。
- 检查 always-on MMRL 是否明显损害 VQAv2；只有存在通用域退化时，Gate 才作为有效贡献。
- PathVQA 必须额外统计 Yes/No、What、非 Yes/No、问题类型 macro average，并按图像聚类计算 bootstrap 置信区间，防止 49.8% Yes/No 和同图多 QA 夸大收益。
- 当天产出最终实验矩阵、缺失 checkpoint 清单、数据审计报告和论文主张表。

## Utility Gate

- 每个垂直 specialist 配置独立 Gate，不训练跨 specialist 的统一路由器。
- 先按原始 Stage 1 + `open_full_ce` Stage 3 协议训练并冻结 specialist，再离线生成 Utility，最后只训练 Utility Gate；禁止 Utility Gate 与 MMRL 联合反向传播。
- 离线计算答案 token 平均损失：`utility = CE_base - CE_mmrl`。
- LLaVA-150K 只是负例候选池；仅选择 `utility < 0`、即 MMRL 确实有害的多模态样本作为负例。
- LLaVA 数量不超过垂直 Utility 样本数量，使用稳定样本 ID 和图像级去重，防止 150K 数据压倒垂直样本。
- Gate 继续使用现有图像加问题 `Task_classifier`，训练时使用连续 Utility 软标签。
- 保留已验证的 alpha-probability 瓶颈：`alpha = sigmoid(Task_classifier(...))` 是 Utility Gate 的连续训练变量；训练阶段不进行 Hard-Concrete 采样，也不把 alpha 提前阈值化成硬 `G`。
- specialist 冻结后，Utility Gate 的梯度只更新 Gate pooling 与 `Task_classifier`；连续 alpha 保证路由器本身有平滑梯度，但不得反向改写已经定稿的 MMRL specialist。
- 默认使用垂直训练集生成 Utility；若正负任一类低于 10% 或 probe `AUROC < 0.55`，自动改用 validation 留出监督。
- 阈值在 validation 上校准，目标是优先保证垂直 specialist 召回率；不确定样本默认开启 MMRL。
- 推理主结果采用由 alpha 校准得到的硬路由：`G=0` 完全跳过 MMRL，`G=1` 运行 specialist；额外报告 soft-alpha 推理作为诊断，但不把它作为“精确安全回退”的主结果。
- `always_off/always_on` 只用于同一冻结 checkpoint 的 Utility 采集、Oracle 和推理对照，不再通过两种模式分别重训 specialist。
- 保留“垂直数据全为正、LLaVA 全为负”的旧来源 Gate 作为必要对照，证明 Utility 监督不是普通数据集分类。

历史约束：`No gate (G=1)` 消融为 71.97，而保留 Stage 1 与完整 Gate 路径的 `open_full_ce` seed44 为 73.93。现有证据说明删除 Gate/跳过 Stage 1 会改变初始化与优化轨迹，不能把该联合干预简单归因于门值本身；后续实验必须保持训练路径一致，只在冻结模型上比较路由。旧 alpha-probability 实验为 `72.92 +/- 0.59`，Hard Concrete 为 `72.71 +/- 1.03`，说明连续 alpha 降低了路由训练方差，但相对 `open_full_ce` 的 `72.94 +/- 0.66` 没有稳定均值增益。

新增运行接口包括 `--gate-mode {always_on,always_off,source,utility}`、`--gate-threshold`、Utility 缓存路径和 LLaVA 候选数量。输出逐样本 `utility_records.jsonl`、阈值扫描、路由指标和 Gate checkpoint。

## 正式实验

| 实验组 | PathVQA | SLAKE | 自建 |
|---|---:|---:|---:|
| Base | 1次 | 1次 | 1次 |
| Visual LoRA | seed44/45 | seed44/45 | 不跑 |
| Static Prompt Tuning | seed44/45 | seed44/45 | 不跑 |
| MMRL | seed44/45 | seed44/45 | seed44 |
| Utility Gate | seed44 | seed44 | seed44 |
| Source-label Gate | seed44 | seed44 | 不跑 |
| No Relation | seed44 | seed44 | 不跑 |
| Mean Pooling | seed44 | seed44 | 不跑 |
| Independent Init | seed44 | 复用已有结果 | 不跑 |
| 归一化 Concat-MLP | seed44 | seed45 | 不跑 |

- PathVQA 已完成的全部24层 Attention LoRA-r128 仅作为更宽作用域的上界。正式匹配基线必须运行 seed44 的视觉层17-24 Attention qkv/proj LoRA-r128，使作用层范围与 MMRL 一致；在该结果完成前不比较两种架构的优劣。
- 当前优先队列固定为三项 seed44 串行实验：视觉层17-24 Attention LoRA-r128；极简 `7,927,808` 参数 MMRL（Mean Pooling、共享高维 `S` 直接进入共享 CA）Relation0；同一极简 MMRL Relation0.05。两项 MMRL 必须复用同一 Stage1 checkpoint 与初始 MMRL 状态，三项都按 validation 选择最佳 epoch 后仅测试一次。
- 上述三项已完成：last8 LoRA53.85，极简 MMRL Relation046.84，Relation0.0549.41。Relation 的干净配对收益为+2.57；若继续确认，只补一个额外 seed，不再扩展成完整2x2矩阵。
- 位置编码控制实验已完成：固定5x8网格得到47.67，低于共同原点的49.41；该方向停止，不追加可学习坐标实验。只有未来建立明确的 Rep Token-图像区域对齐监督后才重新考虑空间位置。
- PathVQA 传统静态 Prompt Tuning 已完成：20个共享软提示 token、51,200参数、seed44 得到55.83，超过 minimal MMRL49.41与 last8 LoRA53.85；Yes/No90.33，但 Free-form21.27仍低于 last8 LoRA23.41。下一架构方向转为图文条件化动态 LLM Prompt，目标是保留静态 Prompt 的二分类校准并补足开放题，不再继续扩展 ViT 内部位置或 slot 结构。
- 决胜实验已完成：minimal MMRL+Relation0.05+Static Prompt20 得到55.75，与 Static Prompt 单独55.83打平；Yes/No88.94、Free-form22.52，仍未超过 last8 LoRA 的 Free-form23.41。两分支梯度均健康，因此结论是当前内部ViT MMRL与静态LLM Prompt功能重叠且存在能力权衡，不再追加seed或组合小变体，下一实现直接转向图文条件化动态LLM Prompt。
- 自建数据集只报告 Base、MMRL、Gated 三项，使用主实验超参，不根据结果调整。
- 不再声称 128-slot Pooling 优于 Mean Pooling，除非 PathVQA 给出明确证据。
- PathVQA 上的 MMRL 增益必须同时出现在 What 或非 Yes/No 问题中；若增益只来自 Yes/No，不得归因于视觉表征改进。
- Gate 成功标准：垂直准确率相对 always-on MMRL 下降不超过 0.5 个百分点，并恢复至少 80% 的通用域性能损失。
- 若 always-on MMRL 在 VQAv2 上几乎不退化，或 Gate 无法满足上述标准，则 Gate 只作为负结果，不列为主要贡献。

## 时间与验收

- Day 2：实现冻结 checkpoint 的路由覆盖、Utility 采集、连续 alpha Gate、缓存格式、Gate 独立训练和监控。
- Day 3：完成 PathVQA 数据依赖与字段预检后直接运行 Base/MMRL seed44，随后完成 Base/MMRL 一致性测试和 seed44 Gate；不单独安排 smoke test。
- Day 4：双卡并行运行 PathVQA 与 SLAKE 正式 seed44/45。
- Day 5：补 Visual LoRA、Prompt Tuning 以及跨数据集核心消融。
- Day 6：完成 Gate、VQAv2 通用保持和自建数据集单 seed。
- Day 7：冻结结果，生成均值方差、参数量、训练成本、延迟、路由曲线、消融表和论文提纲。
- `G=0` 必须与 Base 确定性生成完全一致，并通过 logits 数值一致性测试。
- Utility 缓存必须验证样本 ID 唯一、答案 token 数非零、损失有限、数据划分无交叉。
- Gate 报告 AUROC、AUPRC、Brier、ECE、FPR、FNR、domain recall、general close rate 和 Oracle regret。
- 每个实验完成后立即更新 `EXPERIMENT_RESULTS.md`，记录失败结果，并提交、推送 Git 备份。
