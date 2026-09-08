# 五天论文收尾计划

> 执行周期：2026-09-02 至 2026-09-06
> 当前阶段：停止开放式架构探索，围绕已经验证的 Question-Guided Directional Prompt Tuning 完成方法冻结、对比实验、统计分析和论文初稿。
> 数据源约束：Windows 工作区是唯一事实来源；本地修改、测试、提交和推送后，服务器仅执行 `source /etc/network_turbo; git pull --ff-only`。

## 1. 总决策

1. **主模型采用 D768**：最终冻结为 `S8@3e-5+A_v10@1e-4`，7.805M 可训练参数；PathVQA Validation seed44为59.5622，三seed均值为59.0030。统一V20在严格同初始化控制下仍显著低1.0385分，已作为负消融终止。
2. **效率模型采用 D512**：PathVQA Validation 为 58.6835，4.592M 参数；作为主方法的低容量版本，用于展示准确率与适配容量之间的折中。
3. **D1024 只作为容量上界**：10.626M 参数，59.5782；不再作为默认主模型。
4. **D256 只作为容量下界**：2.033M 参数，57.1497；已经出现注意力近似均匀、跨模态更新不足的容量悬崖。
5. **宽度搜索结束**：不再追加 D128、D384、D640、D896 等中间配置，也不再扫描槽数、层数、Gate 或新融合模块。
6. **五天内完成投稿版本**：Day 1 后冻结架构；若多随机种子结果削弱现有结论，则降低论文措辞，不再用救火实验延长项目。
7. **DRAPE 作为首要 Related Work**：将其视为当前最接近、完成度最高的动态跨模态 Prompt 工作。正文先充分肯定其在多模态持续指令微调中的贡献，再明确 QDPT 聚焦完全冻结 MLLM 的单领域 VQA 适配。不得宣称首次提出“文本 Query + 视觉 K/V + 动态 LLM Prompt”。
8. **CoTBox-TTT 作为首要同领域证据选择工作**：该方法同样冻结生成式医学VLM并在PathVQA、SLAKE和VQA-RAD上使用连续Soft Prompt，直接支持“领域VQA的关键问题之一是选择问题相关视觉证据”的叙事。但它属于逐测试样本优化20轮的无标签Test-Time Training，依赖额外VisCoT定位器、裁剪重编码和EMA Teacher，且开放题使用Recall、SLAKE采用英文设置，因此只能做定性方法比较和原协议文献背景，不能把其分数与QDPT直接排名。
9. **GRASP 升级为必做直接基线，但不取代 DRAPE**：GRASP（arXiv:2601.17089v1）与QDPT同属冻结生成式MLLM中的问题引导动态Soft Prompt，并已在Qwen2.5-VL-7B的生成式遥感VQA上与Prompt Tuning、VPT、Adapter、LoRA和DoRA比较。DRAPE继续承担动态跨模态Prompt技术先例与创新边界审计；GRASP则必须移植到当前Qwen3-VL、PathVQA/SLAKE和统一评价协议下进入主表。
10. **先做QDPT收敛马拉松，再决定GRASP预算**：当前QDPT只训练3 epochs，而GRASP原文采用batch6、最多20 epochs并按Validation patience5早停。为避免只给竞争方法更多优化步，先固定最终QDPT-D768 seed44从头训练10 epochs，并从epoch3起每轮跑完整PathVQA Validation。逐轮结果实时写入`marathon_progress.tsv`；观察QDPT自身的收敛上限后，再确定GRASP的公平长程协议。线性调度器的总步数随之扩展到10 epochs，因此马拉松的epoch3代表长程日程的中段，不是旧3-epoch日程末端的逐位复现，两者不得混称同seed复现。

## 2. 论文定位与核心叙事

### 2.1 暂定题目

**Question-Guided Directional Prompting for Parameter-Efficient Domain Adaptation of Frozen Multimodal Large Language Models**

暂定方法名：**Question-Guided Directional Prompt Tuning（QDPT）**。正式使用前必须完成名称和方法碰撞检索，必要时更名，但不因此改动模型。

### 2.2 要解决的问题

冻结多模态大模型进行垂直领域适配时，现有两类方法各有明显缺口：

- Static Prompt Tuning 参数极少，但所有样本共享同一组提示，无法针对当前问题从当前图像中定向提取证据。
- 常规静态或单模态 Prompt 缺少显式、可审计的“当前问题如何选择当前图像证据”路径，难以同时表达领域先验和样本条件信息。

本论文不再主张“必须改写视觉编码器内部特征”，而是研究一个更具体的问题：**冻结视觉编码器已经产生了有用视觉证据时，能否通过当前问题定向读取这些证据，并把结果写入 LLM Prompt，从而显著改进传统静态 Prompt，并保留冻结骨干和可调适配容量？**

DRAPE 已在多模态持续指令微调中证明：任务级静态 Prompt 难以覆盖同一任务内部的样本差异，使用指令生成 Query、读取当前图像并合成实例 Prompt 是有效路线。QDPT 不重复宣称这一通用思想，而是把问题收窄到领域 VQA：在不更新视觉 projector、视觉编码器和 LLM 的条件下，研究内部视觉证据的读取位置、静态领域锚点与动态证据 Prompt 的分工，以及条件信息应写回视觉侧还是只写入语言入口。

### 2.3 核心洞察

领域 VQA 的瓶颈不一定是视觉特征完全缺失，也可能是模型没有根据当前问题选择和解释已有视觉证据。QDPT 使用完整问题 Token 生成查询，以冻结视觉特征为 K/V，只把匹配后的跨模态结果写入语言入口：

`当前问题 Q -> 当前图像冻结视觉 K/V -> Directional CA -> 动态 LLM Prompt`

这条路径把“任务先验”和“样本条件信息”分开：静态 Prompt 提供领域锚点，动态 Prompt 提供当前图文对齐后的增量信息。

大量受控实验进一步显示出一种**适配不对称性**：视觉特征仍是不可缺少的证据，但直接、反复改写视觉 Token 的边际收益很小；更有效的控制点是根据当前问题读取冻结视觉证据，再把条件化结果写入 LLM 入口。与 CLIP 类对称双编码器不同，当前“视觉编码器 + Merger + 自回归 LLM”生成式架构的任务输出权集中在 LLM，因此视觉侧增强往往只能通过影响语言侧决策间接生效。

### 2.4 预期贡献

1. **领域 VQA 中的完全冻结问题引导适配**：在 DRAPE 已验证的实例级跨模态 Prompt 方向上，QDPT 将完整问题聚合为 Q，读取冻结视觉编码器内部 Layer17 Token，并在不更新视觉 projector 或任何骨干权重的条件下生成动态 LLM Prompt。
2. **可调的宽度参数 D**：将跨模态适配宽度设计成明确的容量旋钮，实证得到 D256 性能悬崖、D512 效率点和 D768-D1024 性能平台。
3. **因果机制证据**：问题 Q 错配和视觉 K/V 错配分别造成 9.63 和 7.03 个点损失，证明模型依赖当前问题与当前图像的正确配对；动态视觉写回关闭无损，支持 read-only vision / write-only language 的收缩方向。
4. **Prompt 范式内的性能与效率证据**：D768 在 PathVQA 与 SLAKE 上均显著超过 Static Prompt，并通过宽度消融展示可控的准确率-容量边界；D512提供低容量折中点。
5. **生成式 MLLM 的适配不对称性**：受控比较和路径干预共同表明，领域视觉证据需要被保留和正确检索，但语言侧条件 Prompt 比视觉表示写回具有更高的边际适配价值。该结论作为实验发现和设计原则报告，不上升为所有 MLLM 的普遍定律。

### 2.5 必须克制的表述

- 不宣称首次提出跨模态动态 Prompt、指令/问题来源 Query、视觉 K/V 检索或实例级 LLM Soft Prompt；DRAPE 已公开覆盖这组通用设计。创新边界必须落在领域 VQA、完全冻结边界、内部视觉证据、静态领域锚点和机制干预组合上。
- 不宣称QDPT普遍优于权重空间PEFT；PathVQA上的接近表现不能外推到SLAKE。
- 不把错配实验的掉分直接解释成问题与图像各自的独立贡献，错配本身可能比关闭输入更具破坏性。
- no-static-visual 已证明静态视觉校准具有实质贡献；统一V20的受控复跑仍显著落后，因此最终保留旧`S8@3e-5+A_v10@1e-4`参数化。论文只把两组参数共同描述为18个Layer17静态视觉Prompt，不虚构不同语义功能，但如实报告其双学习率实现。
- Visual-only LoRA 明显超过 Frozen Base，因此不得写成“视觉增强毫无意义”；准确表述是其相对语言入口适配的边际收益较低，并且当前证据仅覆盖 Qwen3-VL、PathVQA 与 SLAKE。
- 不再把 Utility Gate、安全回退或 98% Base 一致性作为新路线主叙事；这些内容只属于旧路线背景或未来工作。
- 不宣称优于采用不同骨干、不同数据划分或不同评价脚本的同领域论文；这些结果只能作为文献背景。
- 不把参数更少等同于训练更快；实际训练时间还受冻结骨干前向和全视觉 Token CA 支配。

## 3. 最终方法结构与冻结条件

### 3.1 当前候选结构

1. 冻结基础 VLM。
2. 完整问题 Token 经 attention pooling 得到 `Q10`。
3. 冻结视觉编码器 layer17 的完整视觉 Token 作为 K/V。
4. Directional CA 在宽度 D 中计算匹配表示 `Z10`。
5. `Z10 -> MLP -> anchored dynamic LLM prompt`。
6. 动态 Prompt 与静态 `P20` 拼接后进入冻结 LLM。
7. 在layer17插入18个静态视觉Prompt，代码参数化为`S8@3e-5`与`A_v10@1e-4`两张表；二者都与`Z10`独立，动态`Z -> visual`写回已经删除。该双表保留是受控性能决策，不赋予两个子表未经消融证明的不同语义角色。

### 3.2 Day 1 必做的两项定型实验

#### A. 静态视觉插入消融

历史消融完全移除旧 `S8 + A_v10` 及其 layer17 insert/block/strip 路径，其余训练配置保持一致。

决策规则：

- 若 Overall 下降不超过 0.3，且图像聚类配对 95% CI 包含 0：删除静态视觉插入，最终方法定型为 **read-only vision / write-only language**。
- 若下降超过 0.3 且达到统计或跨子组一致的实质影响：保留静态视觉校准支路，但只描述为与方向性文本 Prompt 互补，不夸大其贡献。
- 若结果处于灰区：以结构简洁性为优先，结合参数量、训练时间、Free-form 和跨数据集表现决定；不追加新的视觉支路搜索。

#### B. learned static query 控制

重新训练 D768 seed44，将问题生成的 `Q10` 替换为同数量、同宽度的可学习静态 Query；视觉 K/V、Prompt 数量、MLP、训练步数与优化器不变。

目的：区分本方法与普通 Q-Former/learned-query 视觉聚合器，验证收益是否确实来自**当前问题条件化的查询**，而不是增加一组通用查询和参数。

#### C. 必做 question-only / w/o visual CA 控制

重新训练一个不读取视觉 K/V 的受控版本，保持问题 `Q10`、静态锚点、文本写入头、训练协议和输出 Prompt 接口不变，令 `Z=Q`，而不是 `Z=Q+CA(Q,V,V)`。视觉 K/V 错配只能证明错误证据具有破坏性，不能替代“完全不使用视觉 CA”的正交消融。该实验直接借鉴 DRAPE 的 `w/o Cross-Modal Attention` 设计，用来回答 QDPT 的动态收益是否确实包含问题引导的视觉检索，而非仅由问题侧条件 Prompt 产生。

### 3.3 架构冻结规则

Day 1 结束后不再增加：

- 新的 CA、Q-Former、Workspace Block、共享 S、Gate、分类器或 MoE。
- 新的视觉插入层、Prompt 槽数、注意力头数、MLP 深度或残差缩放扫描。
- 新的宽度 D。
- 为追回单个 seed 小幅掉分而设计的补丁。

## 4. 对比方法设计

所有对比必须分为三个层级：主文受控Prompt基线、附录跨范式PEFT参考、同领域论文原协议结果；禁止把不同协议结果混为“同表公平对比”。

### 4.1 主文：同一骨干、同一数据与评价协议的 Prompt 基线

主表围绕冻结生成式MLLM的Prompt适配问题组织，不再围绕能否战胜权重微调展开。保留以下同范式基线；除新增的GRASP直接竞争方法外，其余待补受控基线合计最多新增三个单seed实验，不做额外超参数搜索：

| 家族 | 配置 | 作用 | 状态 |
|---|---|---|---|
| Frozen Base | 不训练参数 | 适配增益下界 | PathVQA 已有 |
| Static Prompt Tuning | P20，51.2K 参数 | 样本无关 Prompt 基线 | PathVQA/SLAKE 已有 |
| Static Visual Prompt | 固定视觉Prompt，不生成LLM动态Prompt | 视觉侧Prompt基线 | 待补PathVQA seed44 |
| Dual Static Prompt | 静态视觉Prompt + 静态LLM Prompt | 排除收益仅来自双侧增加Prompt | 待补PathVQA seed44 |
| Image-conditioned Prompt | 图像池化后生成LLM Prompt，不使用问题Query | 样本条件但非问题引导的动态Prompt | 待补PathVQA seed44 |
| Learned-query Prompt | 静态Q10读取视觉K/V后生成LLM Prompt | Q-Former式通用Query对照 | PathVQA已有 |
| GRASP | 问题语义对固定空间块打分，以Entmax稀疏加权空间Prompt原型并生成1个全局Prompt | 同范式直接竞争方法 | **必做PathVQA/SLAKE统一协议复现** |
| QDPT-D512/D768 | 问题Q读取视觉K/V并写入LLM Prompt | 本文效率点与主模型 | PathVQA/SLAKE已有 |

公平性要求：

- 冻结同一 Qwen3-VL 基座，使用相同 train/val/test split、图像预处理、最大生成长度和官方归一化 exact match。
- 报告可训练参数、训练时间、TTFT、TPOT；不只比较准确率。
- 新增Prompt基线固定seed44和三epoch，不根据结果继续扫描长度、层数、学习率或宽度。
- question-only / w/o visual CA 提升为当前必做机制消融；IA3与补充语义指标仍保留在审稿后候补清单。
- GRASP优先使用作者正式开源仓库；若无代码，则严格按论文公式独立实现并标注为`GRASP reimplementation under our unified protocol`。固定`h=512`、`alpha=1.5`和低分辨率主配置`N=4`，先做seed44，不替对手进行额外超参数搜索。
- GRASP论文明确通过冻结LLM前向并mean-pool隐藏状态得到问题向量`q`。统一复现不得悄然替换为Token Embedding pooling；必须如实保留额外text-only LLM前向，并在TTFT与训练时间中单独报告其成本。

### 4.2 附录：不同适配范式参考

Full-Attention LoRA和Visual-Only LoRA保留为不同适配范式的强参考，但不再主导标题、摘要、贡献和主结果讨论。正文实验设置用一处简短文字说明“完整结果见附录”，附录如实报告PathVQA与SLAKE的所有seed、参数、时间和显著性；不得删除SLAKE负结果，也不得把LoRA错误表述为不同任务或不同领域方法。取消尚未运行的r4/r16扫描，除非审稿人明确要求权重空间PEFT的容量曲线。

### 4.3 同领域论文方法

相关工作与独立文献表计划纳入 **4 个核心方法，最多再加 2 个补充方法**：

核心候选：

1. **MEVF**：传统医学 VQA 表征融合方法。
2. **PubMedCLIP**：医学图文预训练视觉表征方法。
3. **M3AE**：医学多模态预训练方法。
4. **MedVInT**：面向医学视觉问答/指令适配的方法。

补充候选：

5. **LLaVA-Med**：仅在其确实报告可比 SLAKE/PathVQA 划分和指标时列入数值表，否则只写 Related Work。
6. **PMC-LLaVA**：同上，不满足协议可比性时不做横向数值结论。

纳入数值表前逐项审计：基础模型、数据划分、是否使用外部医学数据、答案生成或分类设置、评价归一化、报告的是 Validation 还是 Test。协议不同的结果放入“Reported results under original protocols”独立表，并明确**不可与受控 Qwen3-VL 实验直接排名**。

### 4.4 方法碰撞与相关工作审计

必须重点核对以下路线，目标不是继续改模型，而是划清贡献边界：

- **DRAPE**：当前最强且最接近的 Related Work。默认 `H=512`、`Lp=10`，由指令分段池化和文本注意力产生 Query，再对视觉 K/V 做 Cross-Attention并生成实例 LLM Prompt；同时面向持续学习加入任务专属生成器、共享 projector 的 null-space 梯度保护和 CLIP prototype 路由。
- **GRASP**（arXiv:2601.17089v1）：当前最接近且可做统一协议数值比较的直接竞争方法。它将冻结视觉Token网格划分为固定空间块，用冻结LLM提取问题向量，在`h=512`空间计算问题-区域相关性，经`Entmax(alpha=1.5)`得到稀疏权重，再对各空间块绑定的静态Prompt原型加权，生成单个全局Prompt Token写入视觉-语言接口。原文未给出正式代码地址，优先继续检索作者仓库；无仓库时按公式独立复现。
- **CoTBox-TTT**（arXiv:2511.12446v1）：当前最贴近医学VQA任务与数据集的证据选择工作。其24-token Evidence Prompt驱动冻结VisCoT进行两次框定位，32-token Answer Prompt在原图/裁剪图和EMA Teacher之间逐测试样本优化20轮；覆盖VQA-RAD、SLAKE和PathVQA，但不是一次前向的条件Prompt生成器。
- CoCoOp：条件 Prompt 的经典范式。
- MaPLe：多模态/深层 Prompt 学习。
- BLIP-2 Q-Former：learned query 读取冻结视觉特征。
- LION：双层视觉知识与 soft prompting。
- MASP：多方面视觉 Query 模块与静态 soft prompt。

需要回答的区别：Query 是否来自当前问题、K/V 是否来自当前图像、动态结果写入哪里、是否修改冻结骨干、是否提供逐样本错配证据、容量宽度是否可调。

DRAPE 与 QDPT 的正式边界：

- **任务不同**：DRAPE 解决 rehearsal-free multimodal continual instruction tuning 和灾难性遗忘；QDPT 解决单领域医学 VQA 适配。
- **冻结边界不同**：DRAPE 训练当前任务生成器和共享视觉 projector；QDPT 冻结视觉编码器、`visual.merger`/projector 与 LLM，只训练外接 Prompt 模块。
- **视觉来源不同**：DRAPE 读取 projector 后视觉特征；QDPT 读取 Layer17 内部视觉 Token，并用层位和静态视觉 Prompt 消融验证该选择。
- **部署形式不同**：DRAPE 每任务保存生成器并依赖 CLIP 路由；QDPT 每领域使用一个共享适配器，不需要任务标签、生成器池或路由。
- **机制证据不同**：DRAPE 提供去除 Cross-Attention、宽度/Prompt数敏感性和可视化；QDPT 提供 learned-query、图文错配、视觉写回关闭、静态视觉移除和层位敏感性。

论文写法采用“肯定后区分”：先肯定 DRAPE 证明了实例级跨模态 Prompt 在持续学习中的有效性，再指出完全冻结领域适配仍缺少对内部视觉证据位置、静态领域先验与动态样本证据分工、视觉写回必要性的系统研究。DRAPE 是 Related Work 中的首要技术近邻，但其原论文分数不进入 PathVQA/SLAKE 同协议主表。

GRASP 与 QDPT 的正式边界：

- **共同范式**：两者都冻结视觉与语言骨干，根据当前问题选择当前图像证据，并通过训练期CE学习动态Soft Prompt；因此QDPT不得宣称首次提出问题引导视觉Prompt或冻结MLLM动态Prompt。
- **动态内容来源不同**：GRASP的视觉块只生成标量路由权重，Value是样本无关的空间Prompt原型`p_i`，最终`p_global`受限于这些原型的稀疏加权组合；QDPT以Layer17真实视觉特征作为Cross-Attention的Value，直接生成携带样本视觉内容的`Z10`。
- **结构先验不同**：GRASP依赖固定二维网格和空间块绑定，适合稀疏遥感目标；QDPT不预设病变位置，通过由完整问题形成的10个方向查询对完整内部视觉Token做细粒度语义检索。
- **输出容量不同**：GRASP只写入1个全局Prompt Token；QDPT生成10个动态证据Token，并与`P20`静态领域锚点拼接，显式区分领域先验与样本证据。
- **计算路径不同**：GRASP按论文描述需要额外的text-only冻结LLM前向提取问题隐藏状态；QDPT在视觉编码前由问题Embedding完成attention pooling，不额外执行完整LLM编码。效率比较必须包含该差异。
- **正式比较规则**：GRASP的遥感原论文数字只进入原协议文献表；移植到同一Qwen3-VL、PathVQA/SLAKE、三epoch和官方评价脚本后的结果进入主表。若使用独立实现，方法名旁必须标注reimplementation。

CoTBox-TTT与QDPT的正式关系：

- **共同问题意识**：两者都认为医学VQA错误可能来自未选择问题相关视觉证据，而不只是骨干缺少领域知识。
- **适应时机不同**：CoTBox-TTT在每个测试样本上执行20轮前向/反向更新；QDPT在训练阶段学习共享适配器，测试时直接条件生成Prompt，不做反向传播。
- **证据选择形式不同**：CoTBox-TTT调用独立VisCoT预测框、裁剪图像并重新编码；QDPT用问题Q从冻结Layer17视觉K/V中检索潜在证据，不需要额外定位模型或裁剪路径。
- **比较协议不同**：其骨干不是Qwen3-VL，开放题报告关键词Recall而非QDPT官方归一化准确率，SLAKE设置也与当前全语言评估不同；论文不得用其PathVQA/SLAKE数字宣称QDPT胜负。
- **可复用叙事**：把QDPT描述为无需逐样本优化、无需外部定位器、单次常规推理的latent evidence-selection adapter，并在效率表中增加“测试时反向传播、额外模型、重复视觉编码、每样本适应步数”四列。
- **可信度边界**：截至v1未见公开代码、正式录用信息、多seed或显著性；其所谓cross-view loss在公式中表现为两个同视图Teacher-Student损失之和，不能未经复现直接沿用其因果解释。

CoTBox-TTT可以计入“同方向Related Work”的文献数量，但**不计入可直接进行公平分数对比的同协议基线数量**。若制作文献原协议表，必须与Qwen3-VL受控主表分离，并标注Test-Time Training、Open Recall及SLAKE子集差异。

### 4.5 从 DRAPE 迁移的实验设计

可以复用实验范式和分析方法，但必须独立实现、重新运行并用自己的文字与图表报告：

| DRAPE 实验 | QDPT 对应状态 | 决策 |
|---|---|---|
| Static Prompt vs Dynamic Prompt | PathVQA/SLAKE seed44 已有 | 必须进入主表，作为“实例条件化有什么用”的第一证据 |
| w/o Cross-Modal Attention | 当前只有视觉 K/V 错配 | **新增必做 question-only 重训**，区分正确视觉读取与错误视觉污染 |
| Learned Query | QDPT seed44 已完成，下降2.3806 | 直接进入机制消融，不再重复 |
| Mean Pooling / Query初始化变体 | 历史池化实验较多，但非最终结构同协议 | 不为此重开架构搜索；只在 Related Work 中讨论 |
| 隐宽 `H=256/512/768/1024` | D256/D512/D768/D1024 已闭环 | 直接形成容量曲线；承认宽度平台不是独家发现 |
| Prompt数量敏感性 | QDPT 尚无最终结构槽数扫描 | 非必做；审稿后再补，不占当前收尾窗口 |
| Prompt分布 t-SNE | 尚未制作 | 低成本必做分析，比较静态锚点与动态 Prompt，并按问题类型/样本分组 |
| Prompt-to-image注意力图 | 尚未制作 | 低成本必做分析，优先选择医学图像中同图不同问题案例 |
| 同图不同问题案例 | 可从 PathVQA/SLAKE 重复图像中筛选 | 必做定性图，展示问题变化如何改变视觉注意与答案 |
| 路由混淆矩阵 | QDPT 无任务路由 | 不适用，不照搬 |
| 遗忘、BWT、null-space分析 | QDPT 非持续学习 | 不适用，不照搬 |
| 效率表 | 参数、训练时间、TTFT、TPOT已有 | 采用其完整报告思路，但保留硬件和实现口径限制 |

**可选：DRAPE 的近似复现。** 实现一个 DRAPE-style late-feature generator：根据论文公开公式，由问题/指令生成 Query，读取 `visual.merger` 后视觉 Token，并直接生成 LLM Prompt；不引入持续学习专属的任务生成器池、CLIP 路由和 null-space 模块。由于官方项目代码尚未公开、原任务协议也不同，该实验必须标注为“根据论文描述实现的单任务近似版本”，不得称为官方 DRAPE 复现。

该可选项具有强制触发条件：先尽力检索并整理 **3至4个真正同方向的冻结生成式 MLLM Prompt 方法/可复现基线**。Full-Attention LoRA、Visual LoRA以及仅共享“参数高效适配用途”但不属于Prompt生成路线的方法，不计入这个数量。若最终无法凑齐至少3个可信的同方向正式对比，DRAPE近似复现自动升级为必做，用于避免主表只能依赖自建消融或与LoRA进行跨范式正面对比。

执行DRAPE近似复现时固定以下公平边界：使用相同Qwen3-VL骨干、PathVQA划分、三epoch和epoch3全量评估；明确报告与原文的所有偏差；主表名称使用`DRAPE-style (reimplemented)`或`DRAPE-inspired single-task baseline`。若选择冻结`visual.merger`以匹配QDPT冻结协议，应同时说明这不是原论文中可训练projector的完整设置；不得把近似版本的成绩归因给DRAPE作者。

## 5. 数据集与统一实验协议

### 5.1 数据集角色

- **PathVQA：主数据集。** 用于方法选择、宽度曲线、同范式Prompt对比、机制控制和主要统计结论。
- **SLAKE：跨数据集验证。** 最终架构必须原样迁移，不允许根据 SLAKE 重新搜索层数、宽度或槽数。
- **自建电气数据集：必做的补充应用案例。** 质量有限且不可开源，只允许最终方法 seed44 一次运行；不做多seed、消融或SOTA声明。

### 5.2 划分和 Test 使用规则

1. 架构选择只使用 Validation。
2. Day 1 冻结最终结构后，不再根据 Test 修改模型。
3. 每个最终 checkpoint 只执行一次正式 Test。
4. 旧 SLAKE Directional 结果包含已经删除的动态视觉写回或不同宽度，不能冒充最终 D768 架构的跨数据集结果。
5. 所有失败和负结果继续写入 `EXPERIMENT_RESULTS.md`，简要结论同步到 `result.md`。

### 5.3 随机种子

- QDPT-D768：PathVQA seed44/45/46，SLAKE seed44/45/46。
- QDPT-D768与附录Full-Attention LoRA-r8：PathVQA和SLAKE均已完成seed44/45/46。
- D256/D512/D1024、Static Prompt、Visual LoRA、结构消融与新增同范式Prompt基线：seed44。
- 自建数据集：最终 D768 seed44 一次。

### 5.4 指标与统计

主指标：

- 官方归一化 exact-match Overall。
- PathVQA：Yes/No、Free-form、问题类型。
- SLAKE：CLOSED/OPEN、KVQA/VQA、EN/ZH。
- 可训练参数、训练时间、TTFT、TPOT。

统计要求：

- 同一数据样本预测采用 exact McNemar 检验。
- PathVQA 按 image cluster 执行 paired bootstrap 95% CI，避免把同图多问当独立样本。
- 多 seed 报告 mean ± std，并保留每个 seed 的原始分数。
- Free-form token-F1、ROUGE-L 或 BERTScore 当前不实现；仅在审稿人明确要求语义指标时补做，且不得替代官方 exact match。

## 6. 最终实验矩阵

### 6.1 已完成、直接进入论文的实验

- [x] PathVQA D256/D512/D768/D1024 宽度曲线。
- [x] PathVQA D768与D1024的逐题统计对比；LoRA-r8对比移至附录。
- [x] PathVQA Full-Attention LoRA-r8 seed44/45/46，作为附录跨范式参考。
- [x] PathVQA Visual last8/all24 Attention LoRA-r128 seed44，作为附录视觉权重适配参考。
- [x] PathVQA Frozen Base 与 Static Prompt seed44。
- [x] PathVQA 问题 Q mismatch 与视觉 K/V mismatch。
- [x] 动态视觉写回 inference intervention，确认其可删除。
- [x] SLAKE Static Prompt seed44 和旧 Directional/Workspace 机制实验，作为研究轨迹与辅助证据保存。

### 6.2 PathVQA 必做

| 优先级 | 实验 | Seed | 目的 |
|---|---|---:|---|
| P0 | D768 移除静态视觉插入 | 44 | 已完成：58.4438，显著下降1.1184；静态视觉校准必须保留 |
| P0 | D768 learned static query | 44 | 证明 question-guided Q 的必要性 |
| P0 | D768视觉前缀改为`A_v10 + Proj(Z10)`硬拼接 | 44/45/46 | 单卡串行三seed；检验独立动态Z视觉Token能否替代重复的`S_v8+A_v10`静态前缀，并直接得到稳定性结论 |
| P0 | 最终 D768 复现 | 45/46 | 主方法均值与稳定性 |
| P0 | Static Visual Prompt | 44 | 新增同范式视觉侧Prompt基线 |
| P0 | Dual Static Prompt | 44 | 新增双侧静态Prompt基线 |
| P0 | Image-conditioned Prompt | 44 | 新增不使用问题Query的动态Prompt基线 |
| Appendix-complete | Full-Attention LoRA-r8复现 | 44/45/46 | 已完成，仅作为附录跨范式参考 |
| Cancelled | Full-Attention LoRA-r4/r16 | 44 | 不再运行；不扩展跨范式容量扫描 |
| P1 | 最终架构正式 Test | 44/45/46 最终 checkpoint | 冻结后仅运行一次 |
| Completed | question-only / w/o visual CA | 44 | 已完成：57.6290，相对完整QDPT显著下降1.9332；证明正确视觉K/V读取有独立增益，不追加seed |
| Conditional P0 | DRAPE-style近似复现 | 44 | 当前搁置；若最终不足3个可信同方向Prompt对比则自动升级为必做 |
| Post-review | IA3 单配置 | 44 | 当前不实现；仅在审稿人要求增加轻量PEFT时补做 |

### 6.3 SLAKE 必做

| 优先级 | 实验 | Seed | 目的 |
|---|---|---:|---|
| P0 | 与 PathVQA 完全一致的最终 D768 | 44/45/46 | 跨数据集泛化 |
| P1 | Static Prompt | 复用已有 seed44 | 静态 Prompt 基线 |
| Appendix-complete | Full-Attention LoRA-r8 | 44/45/46 | 已完成，仅作为跨范式边界参考 |
| P1 | 最终 checkpoint 官方 Test | 最终 seeds | CLOSED/OPEN、KVQA/VQA、EN/ZH |

### 6.4 补充数据集

- [ ] 自建电气数据集：**必做**。最终 D768 seed44 一次；只报告应用可行性，不进行多 seed、消融或 SOTA 声明。专用训练/评估入口已实现，等待运行。
- [ ] 不新增第四个公开数据集。PathVQA + SLAKE 已足以支撑主张，自建数据集只展示跨领域应用。

## 7. 论文表格与图

### 7.1 主表

1. **PathVQA受控Prompt主性能表**：Frozen Base、Static LLM Prompt、Static Visual Prompt、Dual Static Prompt、Image-conditioned Prompt、Learned-query Prompt、QDPT-D512/D768。
2. **参数-容量表**：D256/D512/D768/D1024 的参数、Overall、Yes/No、Free-form、训练时间。
3. **机制消融表**：question mismatch、visual K/V mismatch、learned static query、no-static-visual、question-only / w/o visual CA。
4. **跨数据集表**：最终D768与Static Prompt在PathVQA/SLAKE的统一协议结果；不要求三个新增基线重复训练SLAKE。
5. **文献结果表**：同领域论文原协议结果，和受控主表严格分离。
6. **效率表**：Prompt方法的参数量、训练时长、峰值显存、TTFT、TPOT。
7. **附录PEFT参考表**：Full-Attention LoRA与Visual-Only LoRA的完整结果，只在正文引用一次，不重复组织主叙事。

### 7.2 图

- 最终架构图：Question-Q -> Frozen Visual K/V -> Directional CA -> Dynamic LLM Prompt。
- Accuracy-Parameters Pareto 图：QDPT宽度点与同范式Prompt基线。
- 宽度曲线：D256 到 D1024 的性能平台与容量悬崖。
- 错配干预图：Matched、Question mismatch、Visual K/V mismatch。
- 可选能力分布图：Yes/No 与 Free-form 的方法差异。
- Static/动态 Prompt 分布图：参考 DRAPE 的 t-SNE，但按医学问题类型和图像簇重新设计。
- 同图不同问题注意力图：固定医学图像，改变问题，展示 Q10 对 Layer17 视觉 Token 的证据选择变化。

## 8. 五天执行表

### Day 1：方法冻结与文献边界（2026-09-02）

- [x] 实现 no-static-visual D768；参数审计固定为 7,786,752，已通过本地 Python 编译与脚本静态检查，PyTorch 单测由启动脚本在训练前强制执行。
- [x] 实现 learned-static-query D768；已随最终`S8+A_v10`参数化恢复为7,805,184参数。
- [x] 准备 PathVQA/SLAKE 最终统一启动脚本；实验名编码数据集、D、Query 来源、视觉模式与 seed，统一强制 3 epochs 且只在 epoch 3 全量评估。
- [x] IA3和补充语义指标已移至审稿后候补；question-only因DRAPE碰撞审计提升为当前必做。
- [ ] 完成 CoCoOp、MaPLe、Q-Former、LION、MASP、DRAPE与GRASP的碰撞矩阵。
- [ ] 审计 PathVQA/SLAKE 同领域论文的 split 与 metric。
- [x] 根据 no-static-visual与RNG-controlled V20结果冻结最终`S8+A_v10`结构和论文主张。

当日产物：最终方法配置、实验命令清单、相关工作差异表、冻结后的 Method 草稿提纲。

### Day 2：PathVQA 主实验（2026-09-03）

- [x] 完成D768 seed45/46复现，连同seed44得到Overall 59.0030 +/- 0.4859。
- [x] 完成Full-Attention LoRA-r8 seed45/46；连同seed44得到Overall 59.2640 +/- 0.0666，QDPT均值低0.2610且开放题均值近乎相同，最终按性能持平而非胜出表述。
- [x] 完成 no-static-visual seed44：显著下降1.1184，保留静态视觉校准。
- [x] 完成 direct-visual-Z concat seeds44/45/46：三seed均值58.8007，较原D768均值-0.2023且方差增大，拒绝作为最终结构。
- [x] 最终层位敏感性：Layer18-only为58.3959，共享Layer17+18+19为58.3799，均较Layer17的59.5622显著低约1.18分；多层与Layer18完全打平且进一步伤害`where`。最终固定Layer17-only，停止层数扫描。
- [x] 最后一次统一V20抢救：严格固定下游初始化后为58.5237，仍较旧8+10同seed显著低1.0385；永久保留旧8+10双速率视觉Prompt，V20只作为负消融，不再重复运行。
- [x] 完成 learned-static-query seed44：57.1817，较问题引导Q10显著下降2.3806，确认当前问题条件化Query的必要性，不追加seed。
- [x] 取消LoRA-r4/r16 seed44；已有r8足以作为附录中的跨范式强参考，不再用rank扫描消耗收尾时间。
- [x] 未发现GRASP作者正式仓库；已按论文公式完成Qwen3-VL近似复现、PathVQA/SLAKE/现有电气数据接口连接和CPU单测。保留冻结LLM的额外question-only前向，并明确标注为独立复现。
- [ ] 修正GRASP全局Prompt的视觉段注入位置与纯问题Token编码后，重做PathVQA seed44；初版45.0871因两项实现偏差仅作为失败记录，不进入主表，也不扫描`N/h/alpha`。
- [ ] 汇总每个实验的 Validation、参数、训练时间与预测文件。
- [x] 完成QDPT-D768 seed44十轮收敛马拉松：epoch3-10逐轮完整Validation，不跑Test。epoch6峰值58.7794、epoch10为57.1657，均未超过原3-epoch seed44的59.5622；终止QDPT长程训练与学习率扫参，保留原3-epoch协议。
- [x] 立即更新两个实验账本，不做账本单独提交。

当日产物：PathVQA多seed主结果、同范式Prompt主表和定型消融结论；权重空间PEFT结果单列附录。

### Day 3：SLAKE 跨数据集验证（2026-09-04）

- [x] 原样迁移最终D768并完成seed44/45/46：Overall 77.65/76.70/76.74，三seed 77.03 +/- 0.54；架构冻结，不按SLAKE结果调参。
- [x] 准备专用串行目标`slake_qdpt_d768_final_seeds44_46`：只运行SLAKE最终D768三seed，任一失败继续其余项，退出后自动关机，不重复PathVQA。
- [x] 准备`slake_lora_full_model_attn_r8_seeds44_46`：三个seed串行、失败继续、仅epoch3官方Test，并在启动前跳过已有完整结果。
- [x] 完成Full-Attention LoRA-r8 seed44/45/46：Overall 81.95/81.57/81.95，均值81.82 +/- 0.22；QDPT均值低4.79且三个同seed配对均极显著，停止跨数据集LoRA性能持平叙事。
- [ ] PathVQA修正版通过实现审计后再完成SLAKE GRASP seed44迁移；当前shell history中的启动命令没有生成实验目录或结果，不算已完成。
- [ ] 复核 Static Prompt 的 checkpoint、split 和评价结果。
- [ ] 禁止根据 SLAKE 分数修改 D、层数、Prompt 长度或训练策略。

当日产物：SLAKE 主表、多 seed 稳定性、跨数据集结论。

### Day 4：一次性 Test、统计与制图（2026-09-05）

- [ ] 锁定 Validation 决策后，对最终 checkpoint 运行一次正式 Test。
- [ ] 计算 multi-seed mean ± std、McNemar、image-clustered paired bootstrap CI。
- [ ] 生成主性能表、容量表、消融表、效率表和文献独立表。
- [ ] 生成架构图、Pareto 图、宽度曲线和 mismatch 图。
- [ ] 运行一次自建电气数据集 D768 seed44；该项为必做，但不扩展多seed或消融。

当日产物：全部定稿数字、图表初版、统计脚本与机器可读结果。

### Day 5：论文初稿与复现包（2026-09-06）

- [ ] 完成 Abstract、Introduction、Related Work、Method、Experiments、Analysis、Limitations、Conclusion。
- [ ] 把所有数字与 `EXPERIMENT_RESULTS.md` 逐项核对，禁止手工猜测或混用 Val/Test。
- [ ] 整理配置、启动命令、环境、seed、训练时长、硬件和 checkpoint 说明。
- [ ] 完成方法碰撞复核、声明强度复核和泄漏/Test 调参复核。
- [ ] 打包代码与复现说明，形成可投稿初稿。

当日产物：完整论文初稿、最终图表、复现 README、实验账本和投稿前问题清单。

## 9. 停止规则

1. Day 1 后停止架构搜索，只允许修复明确 bug。
2. 不再增加公开数据集，不做新宽度、层数、槽数、学习率或 Gate 搜索。
3. 可选基线实现失败不得拖延主方法实验。
4. 不根据 Test 结果选择 checkpoint、修改结构或调整措辞中的数值门槛。
5. 若 D768 多 seed 不稳定，诚实报告均值和方差，并把“优于”降为“具有竞争力”。
6. 不再追加权重空间PEFT容量扫描；已有强参考只用于界定方法边界，不为追赶其绝对分数修改QDPT。
7. no-static-visual已显著掉分；保留旧`S8@3e-5+A_v10@1e-4`静态视觉校准。统一V20受控复跑仍显著低1.04分，不再为结构外观设计替代支路。
8. 若 SLAKE 提升较弱，保留为跨数据集边界结果，不回到 SLAKE 做定制搜索。
9. 任何新想法先进入 Future Work，不在五天窗口内实施。

## 10. 论文结构

1. **Abstract**：问题、QDPT 路径、D 可调容量、两数据集结果与机制证据。
2. **Introduction**：从Static Prompt的样本不变性和现有条件Prompt缺少问题-视觉定向检索切入，不用LoRA组织主要矛盾。
3. **Related Work**：医学 VQA、MLLM PEFT、Prompt Learning、Q-Former/视觉聚合器。
4. **Method**：问题池化、Directional CA、动态 Prompt、宽度 D、冻结与训练参数。
5. **Experiments**：数据、协议、基线、主结果、效率。
6. **Analysis and Discussion**：宽度曲线、错配、learned query、视觉插入消融、能力类型差异，以及生成式 MLLM 与 CLIP 类对称双编码器的适配不对称性。用“视觉证据必要但视觉写回边际收益有限”概括，不称视觉编码器为附属挂件。
   补充机理表述：冻结视觉编码器已保留广泛视觉证据，问题条件更适合在编码后做定向检索并写入LLM Prompt。静态视觉Prompt提供稳定领域校准，而动态Z写回视觉编码器未带来稳定收益且增大seed波动。该结论仅限当前冻结生成式MLLM与受控实验，不声称对所有任务普遍成立。
7. **Limitations**：两公开数据集、exact match局限、训练加速有限，并承认强权重空间适配在部分数据集上具有明显准确率优势；完整数字指向附录。
8. **Conclusion**：强调可配置、可审计的样本条件Prompt适配，不讨论全面替代其他PEFT范式。

## 11. 最终复现检查清单

- [ ] 每个结果都能定位到唯一实验名、commit、seed、checkpoint、预测文件和 output path。
- [ ] `EXPERIMENT_RESULTS.md` 保存完整记录，`result.md` 保存简洁结论。
- [ ] 表格明确区分 Validation/Test 与受控结果/文献原协议结果。
- [ ] 参数量使用实际 parameter audit，不用理论估算替代。
- [ ] 训练时长、TTFT、TPOT 标明硬件和测量条件。
- [ ] 所有 paired 检验使用同一批样本，并保存 exclusive-correct counts。
- [ ] 代码默认配置与论文最终方法一致。
- [ ] 服务器只从本地仓库 fast-forward，同步前执行 network turbo。
- [ ] 不删除负结果，不覆盖历史更正。
