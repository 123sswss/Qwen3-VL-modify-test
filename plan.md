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
