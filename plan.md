# 五天论文收尾计划

> 执行周期：2026-09-02 至 2026-09-06
> 当前阶段：停止开放式架构探索，围绕已经验证的 Question-Guided Directional Prompt Tuning 完成方法冻结、对比实验、统计分析和论文初稿。
> 数据源约束：Windows 工作区是唯一事实来源；本地修改、测试、提交和推送后，服务器仅执行 `source /etc/network_turbo; git pull --ff-only`。

## 1. 总决策

1. **主模型采用 D768**：PathVQA Validation 为 59.5622，7.805M 可训练参数；与 D1024 的 59.5782 统计等价，同时参数减少 26.54%。
2. **效率模型采用 D512**：PathVQA Validation 为 58.6835，4.592M 参数；参数少于 Full-Attention LoRA-r8，Overall 与其统计持平。
3. **D1024 只作为容量上界**：10.626M 参数，59.5782；不再作为默认主模型。
4. **D256 只作为容量下界**：2.033M 参数，57.1497；已经出现注意力近似均匀、跨模态更新不足的容量悬崖。
5. **宽度搜索结束**：不再追加 D128、D384、D640、D896 等中间配置，也不再扫描槽数、层数、Gate 或新融合模块。
6. **五天内完成投稿版本**：Day 1 后冻结架构；若多随机种子结果削弱现有结论，则降低论文措辞，不再用救火实验延长项目。

## 2. 论文定位与核心叙事

### 2.1 暂定题目

**Question-Guided Directional Prompting for Parameter-Efficient Domain Adaptation of Frozen Multimodal Large Language Models**

暂定方法名：**Question-Guided Directional Prompt Tuning（QDPT）**。正式使用前必须完成名称和方法碰撞检索，必要时更名，但不因此改动模型。

### 2.2 要解决的问题

冻结多模态大模型进行垂直领域适配时，现有两类方法各有明显缺口：

- Static Prompt Tuning 参数极少，但所有样本共享同一组提示，无法针对当前问题从当前图像中定向提取证据。
- LoRA 通过修改大量注意力投影获得较强性能，但其作用范围广、参数与训练成本随目标模块增加，而且缺少显式、可审计的“问题如何选择视觉证据”路径。

本论文不再主张“必须改写视觉编码器内部特征”，而是研究一个更具体的问题：**冻结视觉编码器已经产生了有用视觉证据时，能否通过当前问题定向读取这些证据，并把结果写入 LLM Prompt，从而以较小、可调的适配容量达到接近 LoRA 的性能？**

### 2.3 核心洞察

领域 VQA 的瓶颈不一定是视觉特征完全缺失，也可能是模型没有根据当前问题选择和解释已有视觉证据。QDPT 使用完整问题 Token 生成查询，以冻结视觉特征为 K/V，只把匹配后的跨模态结果写入语言入口：

`当前问题 Q -> 当前图像冻结视觉 K/V -> Directional CA -> 动态 LLM Prompt`

这条路径把“任务先验”和“样本条件信息”分开：静态 Prompt 提供领域锚点，动态 Prompt 提供当前图文对齐后的增量信息。

大量受控实验进一步显示出一种**适配不对称性**：视觉特征仍是不可缺少的证据，但直接、反复改写视觉 Token 的边际收益很小；更有效的控制点是根据当前问题读取冻结视觉证据，再把条件化结果写入 LLM 入口。与 CLIP 类对称双编码器不同，当前“视觉编码器 + Merger + 自回归 LLM”生成式架构的任务输出权集中在 LLM，因此视觉侧增强往往只能通过影响语言侧决策间接生效。

### 2.4 预期贡献

1. **问题引导的方向性交叉注意力**：不是使用与样本无关的 learned query，而是由完整问题 Token 聚合得到 Q，对当前图像的冻结视觉 Token 执行定向读取，并生成动态 LLM Prompt。
2. **可调的宽度参数 D**：将跨模态适配宽度设计成类似 LoRA rank 的容量旋钮，实证得到 D256 性能悬崖、D512 效率点和 D768-D1024 性能平台。
3. **因果机制证据**：问题 Q 错配和视觉 K/V 错配分别造成 9.63 和 7.03 个点损失，证明模型依赖当前问题与当前图像的正确配对；动态视觉写回关闭无损，支持 read-only vision / write-only language 的收缩方向。
4. **性能与效率证据**：D768 在 PathVQA 上与 Full-Attention LoRA-r8 Overall 统计持平，并显著提高 Free-form 与 `what`；D512 在参数少于 LoRA-r8 时仍与其 Overall 统计持平。
5. **生成式 MLLM 的适配不对称性**：受控比较和路径干预共同表明，领域视觉证据需要被保留和正确检索，但语言侧条件 Prompt 比视觉表示写回具有更高的边际适配价值。该结论作为实验发现和设计原则报告，不上升为所有 MLLM 的普遍定律。

### 2.5 必须克制的表述

- 不宣称 D768 在 Overall 上显著优于 LoRA-r8；当前结论是**统计持平、能力分布不同**。
- 不把错配实验的掉分直接解释成问题与图像各自的独立贡献，错配本身可能比关闭输入更具破坏性。
- 在静态视觉插入 `S8 + A_v10` 完成训练级消融前，不宣称视觉编码器内部干预完全无用或完全不需要。
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
7. 当前 checkpoint 仍保留 layer17 的静态视觉插入 `S8 + A_v10`；动态 `Z -> visual` 写回已经删除。

### 3.2 Day 1 必做的两项定型实验

#### A. 静态视觉插入消融

重新训练 D768 seed44，完全移除 `S8 + A_v10` 及其 layer17 insert/block/strip 路径，其余训练配置保持一致。

决策规则：

- 若 Overall 下降不超过 0.3，且图像聚类配对 95% CI 包含 0：删除静态视觉插入，最终方法定型为 **read-only vision / write-only language**。
- 若下降超过 0.3 且达到统计或跨子组一致的实质影响：保留静态视觉校准支路，但只描述为与方向性文本 Prompt 互补，不夸大其贡献。
- 若结果处于灰区：以结构简洁性为优先，结合参数量、训练时间、Free-form 和跨数据集表现决定；不追加新的视觉支路搜索。

#### B. learned static query 控制

重新训练 D768 seed44，将问题生成的 `Q10` 替换为同数量、同宽度的可学习静态 Query；视觉 K/V、Prompt 数量、MLP、训练步数与优化器不变。

目的：区分本方法与普通 Q-Former/learned-query 视觉聚合器，验证收益是否确实来自**当前问题条件化的查询**，而不是增加一组通用查询和参数。

#### C. 可选 question-only 控制

若 Day 1 前两项按时完成，可训练一个不读取视觉 K/V、只由问题生成动态 Prompt 的等容量控制。该实验用于排除纯问题捷径；若实现或运行会影响主实验进度，立即取消，因为现有视觉 K/V 错配已经提供了较强机制证据。

### 3.3 架构冻结规则

Day 1 结束后不再增加：

- 新的 CA、Q-Former、Workspace Block、共享 S、Gate、分类器或 MoE。
- 新的视觉插入层、Prompt 槽数、注意力头数、MLP 深度或残差缩放扫描。
- 新的宽度 D。
- 为追回单个 seed 小幅掉分而设计的补丁。

## 4. 对比方法设计

所有对比必须分为两个层级，禁止把不同协议结果混为“同表公平对比”。

### 4.1 同一骨干、同一数据与评价协议的可控 PEFT 基线

主表保留 **5 个基线家族**，其中真正需要新增运行的只有 LoRA rank 补点和最多一个可选轻量基线：

| 家族 | 配置 | 作用 | 状态 |
|---|---|---|---|
| Frozen Base | 不训练参数 | 适配增益下界 | PathVQA 已有 |
| Static Prompt Tuning | P20，51.2K 参数 | 样本无关 Prompt 基线 | PathVQA/SLAKE 已有 |
| Full-Attention LoRA | r4/r8/r16 | 与作用于视觉+LLM Attention 的强通用 PEFT 正面对比 | r8 seed44 已有；补 r4/r16 |
| Visual-Only LoRA | last8-r128、all24-r128 | 与只作用视觉编码器的方法比较 | PathVQA 已有 |
| IA3 或同等级轻量 PEFT | 单一标准配置 | 增加一个非 Prompt、非 LoRA 的通用轻量基线 | 可选，Day 1 中午前未完成实现即删除 |

公平性要求：

- 冻结同一 Qwen3-VL 基座，使用相同 train/val/test split、图像预处理、最大生成长度和官方归一化 exact match。
- 报告可训练参数、训练时间、TTFT、TPOT；不只比较准确率。
- LoRA rank sweep 只做 r4/r8/r16，不扩展到视觉模块组合搜索。
- IA3 不是论文成立的前置条件，禁止因实现困难拖延五天计划。

### 4.2 同领域论文方法

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

### 4.3 方法碰撞与相关工作审计

必须重点核对以下路线，目标不是继续改模型，而是划清贡献边界：

- CoCoOp：条件 Prompt 的经典范式。
- MaPLe：多模态/深层 Prompt 学习。
- BLIP-2 Q-Former：learned query 读取冻结视觉特征。
- LION：双层视觉知识与 soft prompting。
- MASP：多方面视觉 Query 模块与静态 soft prompt。

需要回答的区别：Query 是否来自当前问题、K/V 是否来自当前图像、动态结果写入哪里、是否修改冻结骨干、是否提供逐样本错配证据、容量宽度是否可调。

## 5. 数据集与统一实验协议

### 5.1 数据集角色

- **PathVQA：主数据集。** 用于方法选择、宽度曲线、LoRA 对比、机制控制和主要统计结论。
- **SLAKE：跨数据集验证。** 最终架构必须原样迁移，不允许根据 SLAKE 重新搜索层数、宽度或槽数。
- **自建电气数据集：补充应用案例。** 质量有限且不可开源，只允许最终方法单 seed 一次运行；若前四天资源紧张则取消，不影响主论文。

### 5.2 划分和 Test 使用规则

1. 架构选择只使用 Validation。
2. Day 1 冻结最终结构后，不再根据 Test 修改模型。
3. 每个最终 checkpoint 只执行一次正式 Test。
4. 旧 SLAKE Directional 结果包含已经删除的动态视觉写回或不同宽度，不能冒充最终 D768 架构的跨数据集结果。
5. 所有失败和负结果继续写入 `EXPERIMENT_RESULTS.md`，简要结论同步到 `result.md`。

### 5.3 随机种子

- QDPT-D768：PathVQA seed44/45/46，SLAKE seed44/45/46。
- Full-Attention LoRA-r8：PathVQA seed44/45/46；SLAKE 至少 seed44，资源允许则补 45/46。
- D256/D512/D1024、Static Prompt、Visual LoRA、LoRA-r4/r16、结构消融：seed44。
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
- 可选报告 Free-form token-F1、ROUGE-L 或 BERTScore 作为补充语义指标，但绝不替代官方 exact match，也不用于训练选型。

## 6. 最终实验矩阵

### 6.1 已完成、直接进入论文的实验

- [x] PathVQA D256/D512/D768/D1024 宽度曲线。
- [x] PathVQA D768 与 D1024、LoRA-r8 的逐题统计对比。
- [x] PathVQA Full-Attention LoRA-r8 seed44。
- [x] PathVQA Visual last8/all24 Attention LoRA-r128 seed44。
- [x] PathVQA Frozen Base 与 Static Prompt seed44。
- [x] PathVQA 问题 Q mismatch 与视觉 K/V mismatch。
- [x] 动态视觉写回 inference intervention，确认其可删除。
- [x] SLAKE Static Prompt seed44 和旧 Directional/Workspace 机制实验，作为研究轨迹与辅助证据保存。

### 6.2 PathVQA 必做

| 优先级 | 实验 | Seed | 目的 |
|---|---|---:|---|
| P0 | D768 移除静态视觉插入 | 44 | 已完成：58.4438，显著下降1.1184；最终结构保留18个Layer17静态视觉Prompt |
| P0 | D768 learned static query | 44 | 证明 question-guided Q 的必要性 |
| P0 | D768视觉前缀改为`A_v10 + Proj(Z10)`硬拼接 | 44 | 检验独立动态Z视觉Token能否替代重复的`S_v8+A_v10`静态前缀；超过59.5622才保留 |
| P0 | 最终 D768 复现 | 45/46 | 主方法均值与稳定性 |
| P0 | Full-Attention LoRA-r8 复现 | 45/46 | 公平多 seed 主基线 |
| P1 | Full-Attention LoRA-r4 | 44 | LoRA 参数-性能曲线 |
| P1 | Full-Attention LoRA-r16 | 44 | LoRA 参数-性能曲线 |
| P1 | 最终架构正式 Test | 44/45/46 最终 checkpoint | 冻结后仅运行一次 |
| P2 | question-only 等容量控制 | 44 | 排除纯问题捷径 |
| P2 | IA3 单配置 | 44 | 通用轻量 PEFT 补充对比 |

### 6.3 SLAKE 必做

| 优先级 | 实验 | Seed | 目的 |
|---|---|---:|---|
| P0 | 与 PathVQA 完全一致的最终 D768 | 44/45/46 | 跨数据集泛化 |
| P0 | Full-Attention LoRA-r8 | 至少44，优先44/45/46 | 同骨干强基线 |
| P1 | Static Prompt | 复用已有 seed44 | 静态 Prompt 基线 |
| P1 | 最终 checkpoint 官方 Test | 最终 seeds | CLOSED/OPEN、KVQA/VQA、EN/ZH |

### 6.4 补充数据集

- [ ] 自建电气数据集：最终 D768 seed44 一次；只报告应用可行性，不进行多 seed、消融或 SOTA 声明。
- [ ] 不新增第四个公开数据集。PathVQA + SLAKE 已足以支撑主张，自建数据集只展示跨领域应用。

## 7. 论文表格与图

### 7.1 主表

1. **受控主性能表**：Frozen Base、Static Prompt、Visual LoRA、Full-Attention LoRA、QDPT-D512/D768；PathVQA 和 SLAKE 分开报告。
2. **参数-容量表**：D256/D512/D768/D1024 的参数、Overall、Yes/No、Free-form、训练时间。
3. **机制消融表**：question mismatch、visual K/V mismatch、learned static query、no-static-visual、可选 question-only。
4. **LoRA rank 表**：r4/r8/r16 的参数与性能。
5. **跨数据集表**：最终 D768、Static Prompt、LoRA-r8 在 PathVQA/SLAKE 的统一协议结果。
6. **文献结果表**：同领域论文原协议结果，和受控主表严格分离。
7. **效率表**：参数量、训练时长、峰值显存、TTFT、TPOT。

### 7.2 图

- 最终架构图：Question-Q -> Frozen Visual K/V -> Directional CA -> Dynamic LLM Prompt。
- Accuracy-Parameters Pareto 图：QDPT 宽度点与 LoRA rank 点。
- 宽度曲线：D256 到 D1024 的性能平台与容量悬崖。
- 错配干预图：Matched、Question mismatch、Visual K/V mismatch。
- 可选能力分布图：Yes/No 与 Free-form 的方法差异。

## 8. 五天执行表

### Day 1：方法冻结与文献边界（2026-09-02）

- [x] 实现 no-static-visual D768；参数审计固定为 7,786,752，已通过本地 Python 编译与脚本静态检查，PyTorch 单测由启动脚本在训练前强制执行。
- [x] 实现 learned-static-query D768；以 10x2560 learned Query 等参数替换问题池化打分矩阵，总参数保持 7,805,184，已通过本地 Python 编译与脚本静态检查。
- [x] 准备 PathVQA/SLAKE 最终统一启动脚本；实验名编码数据集、D、Query 来源、视觉模式与 seed，统一强制 3 epochs 且只在 epoch 3 全量评估。
- [ ] 中午前判断 IA3 是否能低风险接入；不能则从计划删除。
- [ ] 完成 CoCoOp、MaPLe、Q-Former、LION、MASP 的碰撞矩阵。
- [ ] 审计 PathVQA/SLAKE 同领域论文的 split 与 metric。
- [ ] 根据 no-static-visual 结果冻结最终结构和论文主张。

当日产物：最终方法配置、实验命令清单、相关工作差异表、冻结后的 Method 草稿提纲。

### Day 2：PathVQA 主实验（2026-09-03）

- [ ] 两张 GPU 分容器串行/并行运行 D768 seed45/46 与 LoRA-r8 seed45/46。
- [ ] 运行 learned-static-query 和 no-static-visual seed44。
- [ ] 运行 LoRA-r4/r16 seed44。
- [ ] 汇总每个实验的 Validation、参数、训练时间与预测文件。
- [ ] 立即更新两个实验账本，不做账本单独提交。

当日产物：PathVQA 多 seed 主表、LoRA rank 表、定型消融结论。

### Day 3：SLAKE 跨数据集验证（2026-09-04）

- [ ] 原样迁移最终 D768，运行 seed44/45/46。
- [ ] 运行 Full-Attention LoRA-r8，至少 seed44；资源允许补 45/46。
- [ ] 复核 Static Prompt 的 checkpoint、split 和评价结果。
- [ ] 禁止根据 SLAKE 分数修改 D、层数、Prompt 长度或训练策略。

当日产物：SLAKE 主表、多 seed 稳定性、跨数据集结论。

### Day 4：一次性 Test、统计与制图（2026-09-05）

- [ ] 锁定 Validation 决策后，对最终 checkpoint 运行一次正式 Test。
- [ ] 计算 multi-seed mean ± std、McNemar、image-clustered paired bootstrap CI。
- [ ] 生成主性能表、容量表、消融表、效率表和文献独立表。
- [ ] 生成架构图、Pareto 图、宽度曲线和 mismatch 图。
- [ ] 若时间允许，运行一次自建电气数据集；否则取消。

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
6. 若 LoRA-r16 明显更高，不再追赶其绝对分数；转而报告可调容量、Free-form 能力与显式条件路径。
7. 若 no-static-visual 显著掉分，保留该支路并准确命名为静态视觉校准，不设计替代支路。
8. 若 SLAKE 提升较弱，保留为跨数据集边界结果，不回到 SLAKE 做定制搜索。
9. 任何新想法先进入 Future Work，不在五天窗口内实施。

## 10. 论文结构

1. **Abstract**：问题、QDPT 路径、D 可调容量、两数据集结果与机制证据。
2. **Introduction**：Static Prompt 的样本不变性与 LoRA 的广泛权重改写，引出定向读取冻结视觉证据。
3. **Related Work**：医学 VQA、MLLM PEFT、Prompt Learning、Q-Former/视觉聚合器。
4. **Method**：问题池化、Directional CA、动态 Prompt、宽度 D、冻结与训练参数。
5. **Experiments**：数据、协议、基线、主结果、效率。
6. **Analysis and Discussion**：宽度曲线、错配、learned query、视觉插入消融、能力类型差异，以及生成式 MLLM 与 CLIP 类对称双编码器的适配不对称性。用“视觉证据必要但视觉写回边际收益有限”概括，不称视觉编码器为附属挂件。
7. **Limitations**：两公开数据集、exact match 局限、训练加速有限、未证明所有领域都无需视觉写回。
8. **Conclusion**：强调可配置的样本条件 Prompt 适配，而非宣称全面替代 LoRA。

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
