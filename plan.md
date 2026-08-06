# FROST-VL 英文稿完整整改计划

## 总体判断

以当前 [paper_en.tex](E:/AAA24105033/project/py/visionOnly/Qwen3-VL-modify-test/els-cas-templates/paper_en.tex) 和 [body_en.tex](E:/AAA24105033/project/py/visionOnly/Qwen3-VL-modify-test/els-cas-templates/body_en.tex) 为准，按 EAAI/ESWA 等严肃应用 AI 期刊的门槛重构。自建电力数据集继续承担主要应用贡献，SLAKE 从附录移入主实验，作为公开可复现证据。

### 对编辑部十条意见的实际命中情况

| 条目 | 当前问题 | 严重度 |
|---|---|---|
| 1. 贡献和技术深度 | Gate、Rep Token、Router、Adapter 和多种约束堆叠较多，但缺少诊断实验证明每个设计解决了什么问题 | 高 |
| 2. 对比方法 | 主要是通用 LoRA/DoRA/IA³；结果表没有文献引用；缺少近期视觉 PEFT 和可复现动态路由方法的同源比较。现有电力 VLM 通常没有可下载权重或足够的复现材料，不能强行要求直接实验 | 极高 |
| 3. 贡献表达/英文/简单组合 | 英文基本可读，但引言过早堆叠模块名，主线容易被理解为 MMRL、MoE 和 Adapter 的组合 | 高 |
| 4. 非参数统计 | 自建数据只有单次结果；SLAKE 的五次运行包含重复 Seed 44；没有显著性检验和置信区间 | 极高 |
| 5. 公平调参 | 只完整报告了 FROST-VL 参数；没有验证集和基线调参协议，容易被怀疑使用测试集选参 | 极高 |
| 6. 数据和划分 | 缺少验证集、类别分布、去重方法、标注一致性、生成题目流程和公开数据具体规模 | 极高 |
| 7. 应用分析 | 现有 98.40% Top-1 agreement 是很强的输出行为保持证据，但它只比较首个生成位置且没有任务正确答案，不能单独支撑“100% 保存原生通用能力”。现有 972 道领域 VQA 题也没有按题型汇总，尚不清楚改进主要来自设备识别还是缺陷判断 | 极高 |
| 8. 内容过多但浅 | 方法章节很长，Discussion 只有两段；多个 Loss 和 Router 机制缺少内部行为分析 | 高 |
| 9. 摘要 | 只写自建数据 63.73%，未给训练/测试规模、公开 SLAKE 结果和明确的比较差值 | 极高 |
| 10. Tortured phrases | 不是整体英语差，而是术语密度过高、部分句子不自然、因果断言超过证据；另有明显损坏的参考文献元数据 | 中高 |

## 实验和统计重建

### 1. 固定数据协议，消除测试集选参嫌疑

- 自建数据按原始图像分组去重，同一图像产生的所有 instruction 必须位于同一划分。
- 从现有训练图像中按设备和缺陷类别分层划出 10% 验证集；现有 972 张图像、972 道题的测试集保持完全冻结。
- 明确说明实际训练使用的是 20,000 个样本，而完整训练池包含 20,015 张图像和 36,700 条 instruction，并给出采样种子和抽样规则。
- SLAKE 使用官方 train/validation/test 划分，`language=all`；只在 validation 上选超参数，test 仅用于最终锁定配置后的评测。
- 固定并公布 LLaVA-150K 正负样本 ID、抽样种子、数量和 split，避免“completely at random”这种不可复现描述。

### 2. 建立公平的同源基线体系

所有主要比较统一使用 Qwen3-VL-4B、同一数据划分、输入模板、训练样本、精度和解码设置：

- Base、LoRA、DoRA、IA³、Adapter；
- Visual Prompt Tuning 和 VL-Adapter；
- Full-VLM LoRA，作为不受冻结语言模型约束的准确率上界；
- Vision Encoder 和 Last-8 Vision Encoder 两种与 FROST-VL 更接近的适配范围；
- 对 Dynamic LoRA Routing 做官方代码和配置核验，能够可靠复现时纳入同源比较；
- Power-LLaVA、DefectGPT 等电力 VLM 仅在 Related Work 中讨论。由于没有能够下载并按相同协议本地推理的权重、代码和数据，不设置直接实验，不引用其异构数据上的分数与本文结果比较，也不把“无法复现的电力 VLM 对比”列为待补实验。

每个表格的方法名直接附带文献引用。不同数据集、骨干或训练数据的结果不进行数值上的“谁优谁劣”结论。

### 3. 公平调参与重复运行

- 为每个方法预先确定验证集搜索空间和相同的试验预算，覆盖学习率、epoch、rank或瓶颈宽度、dropout等关键参数。
- 所有候选配置只根据验证集选取，论文和补充材料列出完整搜索空间、试验次数、最终值和 GPU 小时。
- SLAKE 当前使用 4 个独立初始化种子 `44, 45, 46, 47`，并额外重复一次 Seed 44。四个独立种子用于观察初始化敏感性，重复 Seed 44 用于检查同一名义种子下的运行重复性；不强制补 Seed 43。
- 自建数据的 FROST-VL、最强视觉 LoRA/DoRA、Full-VLM LoRA使用相同的一组独立种子。最低要求为 3 个独立种子，预算允许时与 SLAKE 一致使用 4 个。
- 关键消融至少运行 3 个独立种子。Single Adapter 的 32.70% 和 Relation Loss 的大幅下降必须复验。

### 4. 非参数统计分析

- 在同一测试题上的方法比较先使用 Cochran’s Q 检验；显著后，以精确 McNemar 检验进行 FROST-VL 与各关键基线的成对比较。
- 多重比较使用 Holm 校正。
- 报告配对 bootstrap 95% 置信区间、准确率差值和原始正确/错误计数，不只报告 \(p\) 值。
- 多种子结果报告均值、样本标准差、中位数和四分位距。
- 时延使用完全相同的 100 个请求、固定顺序和充分预热，保存逐请求数据；对 TTFT/TPOT 使用配对 Wilcoxon 检验和 bootstrap 置信区间。
- 在没有统计支持前，将“improves first-token responsiveness”“stable”“substantially”改为中性事实描述。

## 方法深度与应用证据

### 1. 把创新主线收束为一个问题

论文统一围绕以下主张展开：

> FROST-VL 在视觉和语言骨干冻结的条件下，通过任务门控的外部视觉残差支路，在需要领域知识时注入可路由偏移，在一般输入上保持原始路径。

MMRL、MoE、Adapter 和 Hard-Concrete 只作为技术来源，不再分别发展成四条并列叙事。新增最近工作对照表，比较：

- 是否冻结骨干；
- 是否具有独立领域支路；
- 是否按样本启用；
- 是否路由多个残差专家；
- 是否约束原始视觉关系；
- 是否支持缓存和模块化加载。

### 2. 补齐复现细节

- 给出最终八个注入层的准确索引、Rep Token 维度、Pooling 查询维度、Router/MLP 层数、Adapter 瓶颈宽度、激活函数和初始化。
- 补充 Hard-Concrete 的 \(\gamma,\zeta,\tau\)、推理阈值和温度调度。
- 明确 Shared Pooling 在两个阶段中的参数状态，以及冻结分类头但继续更新 Pooling 是否会改变最终门控行为。
- 增加训练和推理伪代码，明确 \(G=0\) 时是仅禁止残差写入，还是同时跳过专家支路计算。
- 分解 9.9M 可训练参数和 5.4M 部署参数，分别说明缓存后移除了哪些参数。

### 3. 增加内部机制诊断

- 报告四个 Adapter 输出之间的余弦相似度，验证“异构学习率减少方向集中”的主张。
- 报告 Router 的使用率、熵和不同设备/缺陷类别下的专家分配。
- 报告有效偏移比值 \(r_i\) 的分布及越界比例。
- 比较有无 Relation Loss 时的关系矩阵误差和 token dispersion。
- 增加 w/o Usage Loss、w/o Entropy Loss、w/o Effective-offset Loss、w/o Dispersion Floor、视觉单模态 Router、Force \(G=1\) 等消融。
- 在通用样本上比较正常 Gate 与 Force \(G=1\)，直接证明门控器而非单纯冻结骨干带来的行为保持效果。

### 4. 明确区分“输出行为保持”和“通用任务能力保持”

- 保留现有 500 个 LLaVA-150K 样本的首 token 分布实验。它可以严谨地支持以下结论：在这些样本上，FROST-VL 与 Base 的首步输出分布非常接近，Top-1 token 有 98.40% 一致。
- 该实验不能单独证明“原生通用能力 100% 保留”，原因不是 98.40% 不够高，而是它没有标准答案、只检查第一个 token。两个模型可能输出同一个错误答案，也可能第一个 token 相同而后续答案不同。
- 如果论文要保留“通用任务能力未下降”这一强结论，就必须增加一个带标准答案的通用 VLM benchmark。固定使用 MMBench EN dev，比较 Base、FROST-VL 和 Force \(G=1\)；三者使用相同提示词、图像预处理、解码参数和评分脚本。
- 对 MMBench 报告任务准确率、FROST-VL 与 Base 的逐题正确性变化、配对置信区间和 McNemar 检验。只有当差值及置信区间支持时，才能写“在该公开基准上未观察到可测量的能力下降”；不能写“100% 保存所有原生能力”。
- 如果不做通用 benchmark，则不删除现有 98.40% 结果，但全文统一降格为“output-behavior preservation”或“limited output perturbation”，不再把它表述为通用能力保持实验。

### 5. 增加与现有电力 VQA 任务匹配的应用分析，不做 YOLO 对比

- 不要求 FROST-VL 输出检测框，不计算 mAP，也不拿它与 YOLO 比较。本文任务仍然是 VQA/多项选择题，所有分析都基于现有 972 道测试题的答案正确性。
- 如果现有生成模板或元数据能够可靠区分题型，就把 972 道题分为“设备/部件识别”和“缺陷/异常判断”两组，分别汇总样本数和准确率。这只是对现有预测结果重新分组，不需要重新训练或增加模型推理。
- 只有在单个设备或缺陷类别具有足够样本且标签可靠时，才进一步报告类别级准确率；否则只保留上述两个大类，避免小样本类别产生误导。
- 错误分析限定为检查现有答错题，说明错误更常发生在设备识别、缺陷判断还是问题理解中。它是解释模型适用边界的定性分析，不扩展成目标检测实验。
- 门控器在最终完整 checkpoint 上报告 accuracy、precision、recall、F1、FPR、FNR 和混淆矩阵；困难负样本实验用于排除 Gate 只识别数据集风格的可能。
- 删除或弱化“边缘部署”动机，除非补充真实边缘硬件测试；RTX 5090 只能支持训练和单卡部署成本分析。

## 论文结构、摘要和语言

- 将 SLAKE 从附录移入主实验，摘要、引言贡献、Discussion 和结论同步报告最终五独立种子结果。
- 摘要按“问题—方法—数据—公平比较—公开结果—限制”重写，至少包含：
  - 自建数据的 36,700 条训练 instruction 和 972 道测试题；
  - 63.73% 及相对同范围 LoRA 的 8.07 个百分点优势，最终以重跑表格为准；
  - SLAKE 五独立种子均值、标准差及相对同范围基线的差值；
  - 不声称超过 Full-VLM LoRA 或总体 SOTA，除非新实验确实支持。
- 引言减少模块名堆叠，先定义“选择性领域适配”问题，再说明现有方法缺口和三项可验证贡献。
- 扩展 Discussion，讨论准确率—通用能力—参数—时延的 Pareto 权衡，并明确 Full-VLM LoRA 准确率更高。
- 删除冗余的 “Summary of experiments”；结论先总结已验证结果，再写多类别 Gate 的未来设想。
- 统一术语：Rep Token、Rep-Token generator、TAV-Gate、Router-Adapter、Shared Pooling；将参数表中的 “MMRL LR” 改为 “Rep-Token generator LR”。
- 按 humanizer 审阅标准逐段处理术语堆叠、被动句、空泛因果和重复的 write-back/candidate directions 表述，同时保留技术精度。
- 修复公式标点、大小写和不自然表达，例如 “the model expression writes no domain offset”。
- 所有数字从最终结果表统一生成或交叉核验，禁止摘要、正文和结论分别手工维护旧数字。

## 引用、数据和代码

- 对全部 38 条 BibTeX 逐条通过正式出版页面或 DOI 核验。
- 修复或删除明显损坏的 `SAME` 条目：当前作者为 “Multimodal Continual Instruction Tuning”、卷 83、页码 74--2，不可能作为有效引用保留。
- 补充 SLAKE 原始数据集论文引用，以及数据生成模型、公开基准和新增实验方法的正式引用。
- LoRA、Hard-Concrete、MoE 等优先替换为正式会议/期刊版本，不使用存在卷期页码错误的 Scholar 自动条目。
- 系统补充近期同行评审的视觉 PEFT、动态路由、电力 VLM 和抗遗忘研究。电力 VLM 只承担文献定位作用，不要求在缺少公开权重和代码时进行不可能的本地实验。优先寻找确实相关的 CAEE 论文，不为迎合编辑机械填充无关文献。
- 在结果表中显示方法引用；外部报告值注明来源，作者自行复现值注明 “our reproduction”。
- 下一次投稿前提供匿名化代码包或匿名仓库，包括环境、配置、split manifest、训练/推理命令、评分脚本和 SLAKE 权重。
- 私有图像不公开，但发布类别统计、数据 schema、生成模板、划分规则和允许公开的示例。
- 更新代码可用性声明：删除可能已经过期的“提交后两周”承诺，改为真实仓库状态；如果代码已经可用，直接提供匿名链接。

## 验收标准

- 测试集从未参与超参数选择，所有 split 和样本 ID 可追踪。
- 所有主要方法使用同一骨干、数据、划分和评测协议，并在表中附引用和最终超参数。
- 清楚区分独立种子和同种子技术重复；主要稳定性结果至少包含 3 个独立种子。SLAKE 当前按 4 个独立种子加一次 Seed 44 重复报告。
- 自建数据和 SLAKE 都有配对非参数检验、置信区间和逐种子结果。
- 摘要、正文、结论与表格数字完全一致，不存在 66.95 等旧结果。
- Gate、Router、异构 Adapter、Relation Loss 和缓存都有直接证据，不再只靠结构解释。
- SLAKE 位于主实验；附录只放详细超参数、逐种子结果、扩展消融和补充案例。
- 参考文献无损坏元数据、无缺失数据集引用，所有比较方法在表内可追溯。
- 英文完成技术校对和自然化审阅，没有术语堆叠、夸大结论或疑似 tortured phrasing。
- LaTeX 修改后由用户统一编译；不由 Codex 编译。

## 待补实验优先级

以下顺序按“对下一次投稿能否通过编辑初审的重要性”排列。完成一项后直接在本节将对应复选框划掉。

### A. 必须完成：不完成不建议投同等级期刊

- [ ] **SLAKE 同协议强基线比较**：在同一 Qwen3-VL-4B、官方划分和评分协议下，补齐能够实际复现的近期视觉 PEFT/动态路由基线；不包含无法下载的电力 VLM。
- [ ] **公平调参与验证集协议**：为 FROST-VL 和所有主要基线建立相同预算的验证集选参流程，锁定配置后再运行测试集。
- [x] **SLAKE 随机初始化与同种子重复性实验**：已完成 Seed `44, 45, 46, 47` 四个独立初始化实验，并重复运行一次 Seed 44；论文须按“四个独立种子、五次运行”报告，不能写成“五个独立种子”。
- [ ] **自建数据关键方法重复运行**：至少对 FROST-VL、最强同范围视觉 PEFT 和 Full-VLM LoRA 使用相同独立种子，避免主结论只来自单次训练。
- [ ] **主要准确率的非参数统计**：保存逐题预测，对自建数据和 SLAKE 进行 Cochran’s Q、McNemar、Holm 校正和配对 bootstrap 置信区间。
- [ ] **通用 VLM 能力 benchmark**：使用 MMBench EN dev 比较 Base、FROST-VL 和 Force \(G=1\)。如果决定不做，则必须把全文结论严格限制为“首 token 输出行为保持”，不能声称通用能力未下降。

### B. 强烈建议：用于证明方法不是模块堆叠

- [ ] **关键消融多种子复验**：复验 Single Adapter、Homogeneous Adapter LR、w/o Rep Token 和 w/o Relation Loss，尤其确认 32.70% 和 52.31% 不是偶然训练失败。
- [ ] **其余损失和路由消融**：补充 w/o Usage、w/o Entropy、w/o Effective-offset、w/o Dispersion Floor、视觉单模态 Router 和 Force \(G=1\)。
- [ ] **Adapter/Router 内部诊断**：统计 Adapter 输出余弦相似度、Router 使用率和熵，直接验证异构学习率与多专家路由的作用。
- [ ] **Relation/offset 诊断**：统计关系矩阵误差、token dispersion、有效偏移比值及越界率，给各约束提供直接证据。
- [ ] **最终门控器直接评测**：在最终 checkpoint 上报告完整二分类指标，并增加困难负样本检查数据集风格捷径。

### C. 有条件完成：增强应用解释，但不应阻塞核心实验

- [ ] **现有 972 道题的题型分组**：若元数据可靠，按设备识别和缺陷判断汇总现有 VQA 准确率；不做检测框和 YOLO/mAP 对比。
- [ ] **现有错误案例分析**：从答错题中归纳设备识别、缺陷判断和问题理解错误，不新增检测任务。
- [ ] **推理时延重复测量**：固定 100 个请求、预热和逐请求记录，补 Wilcoxon 检验与置信区间；当前缓存实现和均值表仍可保留。
- [ ] **真实边缘硬件实验**：只有继续强调 edge deployment 时才做；否则删除或弱化相关动机即可。

## 论文必须修复、增加或删减的内容

本节采用已经确定的“方法为主、应用为证据”路线。FROST-VL 不再被描述为电力专用方法，而是冻结 VLM 上的任务门控式低干扰外部适配框架。自建电力数据集保留为主要真实应用案例、潜在数据贡献和真实工作量；SLAKE 成为主要公开实验对象；后续增加一个通用能力 benchmark 和一个额外的公开垂直领域 benchmark。所有具体数字等实验锁定后再统一更新。

### A. 必须重构：锁定新的核心叙事

- [ ] **更换论文的中心命题**：全文统一回答“如何在不更新 VLM 主干的情况下增加可按任务启用的领域能力，同时限制对原始推理行为的干扰”。不再回答“如何为电力电气设计一种专用 VLM 适配方法”。
- [ ] **采用统一的一句话定义**：以“FROST-VL is a task-gated external visual adaptation framework that adds domain expertise to a frozen VLM while limiting changes to its original inference behavior”为基础，统一标题、摘要、引言、方法概览、Discussion 和 Conclusion。
- [ ] **将四类数据的职责彻底分开**：
  - 自建电力电气数据集：主要真实应用案例、已有完整消融、数据构建工作量和未来可能公开的潜在贡献；
  - SLAKE：主要公开实验对象，承担核心基线比较、多次运行、统计分析和主要可复现结论；
  - 额外公开垂直领域 benchmark：验证方法不是只对 SLAKE/医疗领域成立，只运行核心方法和关键基线；
  - MMBench EN dev：不参与领域训练，只用于比较 Base、正常 FROST-VL 和 Force \(G=1\) 的通用任务能力。
- [ ] **明确电力数据集的贡献边界**：可以强调采购、清洗、人工筛选、多模态 instruction 构建和真实评测工作量；在获得公开授权前，只能称为 in-house application benchmark，不能把“公开数据集贡献”写成已经完成的贡献。未来公开只能作为供应商授权后的潜在工作。
- [ ] **确定四个研究问题并据此组织结果**：RQ1 领域能力提升；RQ2 输出行为和通用能力的低干扰保持；RQ3 Gate、Rep Token、Router、Adapter 和约束项的机制作用；RQ4 参数、缓存和推理开销。
- [ ] **限制“通用框架”的表述边界**：当前只在 Qwen3-VL-4B 上实现，最多写成“在多个专业领域上验证的通用设计”或“domain-agnostic design”。在没有第二种骨干实验前，不写 model-agnostic、universal 或 standard framework。
- [ ] **限制多领域切换结论**：当前结果可以证明同一框架能够分别适配不同领域，不能证明多个领域支路已经在同一个模型中同时挂载并无缝切换。多头 Softmax Gate 和多分支共存仍属于 Future Work。

### B. 必须增加：解释为什么需要这套外部适配结构

- [ ] **把“为什么使用 Rep-Token 软提示词专家支路而不是 LoRA”写成最重要的设计依据**：在 Introduction 的研究缺口和 Methodology 的 Overview 中分别说明：
  1. TAV-Gate 使用第一次领域注入之前的前 16 层视觉输出判断当前输入是否需要领域知识，因此这些前缀表示必须来自冻结、未被领域适配污染的原始路径。
  2. 在前 16 层或整个视觉骨干中直接应用 LoRA 会破坏这一干净的门控依据。把 LoRA 限制到后 8 层可以保留 Gate 输入，但本文评测的标准 Last-8 LoRA 仍对所有样本使用同一组适配后的映射，也不会自然产生可与原始支路配对比较的专家状态。
  3. Rep Tokens 只进入后 8 层的并行专家支路，原始支路不接收这些 tokens；专家状态与原始状态之差能够显式定义为领域偏移，再由 Gate 决定是否写回。
  4. 当前同层范围结果为 Last-8 Full-Linear LoRA-r32 55.66%、Last-8 Attention LoRA-r32 54.09%、FROST-VL 63.73%。该证据只能说明在 frozen-prefix constraint 和当前协议下，后 8 层 LoRA 没有达到 FROST-VL 的效果。
- [ ] **同时写清 LoRA 比较的限制**：FROST-VL 与 Last-8 LoRA 参数量并不完全相等，Full-VLM LoRA 的领域准确率也更高。不得写成 FROST-VL 普遍优于 LoRA；应写成冻结语言模型、保持原路径、条件化启用和领域准确率之间的权衡。
- [ ] **正式定义“低干扰”而不是把它当形容词使用**：至少分为三层：结构上冻结主干并隔离领域支路；行为上使用 JS、Top-1 agreement 和 Top-10 overlap；能力上使用带标准答案的 MMBench。参数量和时延属于效率，不与能力保持混为一个概念。
- [ ] **增加结构隔离和条件执行说明**：明确 Gate 在第几层取得输入、Rep Tokens 从哪一层开始注入、两条支路如何并行、\(G=0\) 时是否跳过专家计算，以及缓存后的 Rep Tokens 在推理中的调用顺序。
- [ ] **增加训练和推理伪代码**：训练算法分开描述 TAV-Gate 专属阶段与领域支路训练阶段；推理算法展示 Gate、缓存读取、专家偏移、Router-Adapter 和条件写回。
- [ ] **增加最近工作差异表**：围绕冻结主干、独立支路、按样本启用、残差路由、结构约束和缓存部署进行比较。电力 VLM 只进入 Related Work，不进入无法公平复现的实验结果表。

### C. 必须重排：标题、摘要和章节结构

- [ ] **标题去掉电力限定**：工作标题改为“FROST-VL: Task-Gated Low-Interference Plug-In Adaptation for Frozen Vision-Language Models”。如果最终更强调 routed offset，可在实验完成后改为“FROST-VL: Routed Visual Offsets for Low-Interference Domain Adaptation of Frozen Vision-Language Models”。
- [ ] **摘要改成方法优先结构**：依次写冻结 VLM 的领域适配问题、低干扰外部支路、核心机制、SLAKE 公开结果、自建电力真实应用结果、通用能力/行为保持和适用边界。电力不再出现在摘要第一句，也不把尚未公开的数据集称为公共贡献。
- [ ] **Introduction 从通用问题开始**：先讨论专业领域适配为什么会影响共享路径和一般能力，再引出条件化外部支路；电力和医疗只作为两类代表性专业场景。删除“该结构源于电力设备特点”的暗示。
- [ ] **重写贡献列表**：第一项是任务门控的外部视觉偏移框架；第二项是 Rep-Token 专家偏移与多 Adapter 路由；第三项是低干扰约束和跨公开/真实场景验证。数据贡献仅写电力数据构建与应用评测，不承诺公开。
- [ ] **Related Work 改成方法相关分类**：保留视觉 PEFT、低干扰/抗遗忘适配、条件路由/MoE、软提示词与共享表示四条主线；电力 VLM 缩成应用背景段，说明公开权重和统一协议不足，不再承担实验比较压力。
- [ ] **Methodology 删除电力专用措辞**：将 domain、specialized input、general input 作为统一术语；除举例外，不把 Gate 写成只判断 power/non-power。所有公式和模块描述保持领域无关。
- [ ] **Experiment 按证据职责重排**：先给统一协议和数据角色，再依次呈现 SLAKE 主实验、额外公开垂直领域、自建电力应用案例、通用能力与输出行为保持、推理效率、消融和内部诊断。不同数据集不混在同一排名表中。
- [ ] **Discussion 围绕四个研究问题展开**：分别回答领域能力、低干扰、机制和效率，并明确 Full-VLM LoRA 准确率更高、FROST-VL 只验证了一个 VLM 骨干、私有电力数据不可公开等限制。
- [ ] **Conclusion 先总结已验证内容**：总结跨专业领域的公开与真实应用结果，再陈述低干扰证据；多头 Gate、N 个专家并存和无缝切换移到最后一段，并明确尚未实现。
- [ ] **同步修改所有投稿附件**：Highlights、Graphical Abstract、Cover Letter 和关键词都改成方法优先。图形摘要可以保留电力图像作为应用示例，但主体必须是可拆卸领域支路，而不是让读者误以为结构只适用于电力设备。

### D. 必须修复：实验协议、复现信息和统计表述

- [ ] **把 SLAKE 从附录移入主实验**：主文报告最终配置、主要基线和核心统计；附录保留逐次运行、详细子类别、完整配置和补充案例。
- [ ] **为第二个公开垂直 benchmark 预留独立小节**：正式选择前必须确认图像、标注、许可证和评分脚本均可下载；优先选择能够本地训练与自动评分、且不与 SLAKE 同分布的专业 VQA 数据，不选择只有论文没有数据的基准。
- [ ] **重写自建数据说明**：交代 20,015 张训练图像、36,700 条 instruction、实际抽取 20,000 条样本和 972 张独立测试图像之间的关系，以及去重、同图分组、人工检查和验证集用途。
- [ ] **重写基线公平性说明**：每个训练方法报告学习率、epoch、rank/瓶颈宽度、target modules、训练样本数、搜索预算、选参依据和随机种子；结果表的方法名直接附正式引用。
- [ ] **纠正 SLAKE 多次运行措辞**：统一写成“five runs across four initialization seeds, including one repeated run with Seed 44”。四个独立种子用于初始化敏感性，第二次 Seed 44 用于运行重复性；不能写 five independent seeds。
- [ ] **增加统计方法小节**：说明逐题正确性记录、exact McNemar、Holm 校正和 paired bootstrap；Cochran’s Q 作为多个方法的总体检验，可放正文或补充材料。
- [ ] **补齐网络与训练参数**：写明注入层索引、Rep Token/Pooling/Adapter 维度、MLP 结构、Hard-Concrete 参数、初始化、阈值，以及第二阶段中 Pooling 和 Gate classification head 的冻结状态。
- [ ] **修复参数和模块命名**：将 “MMRL LR” 改成 “Rep-Token generator LR”；统一 Rep Token、TAV-Gate、Router-Adapter、Shared Pooling、expert offset 和 routed offset。
- [ ] **修复参数量口径**：分别解释 trainable parameters、added parameters、deployment-loaded parameters 和缓存后可卸载的 Rep-Token generator，避免 9.9M/5.4M 被误解为前后矛盾。
- [ ] **逐项核验引用**：修复或删除损坏的 `SAME` 条目，补 SLAKE 原始论文和新增 benchmark 引用，把可获得的 arXiv 引用替换为正式出版版本，并在结果表中显示方法来源。
- [ ] **更新可用性声明**：按下一次投稿时的真实状态说明代码、SLAKE/其他公开 benchmark 权重和匿名仓库；删除可能已经过期的“投稿后两周公开”。私有电力数据继续按供应商协议说明限制和潜在公开条件。

### E. 必须删除或降格：避免旧叙事和过度断言回流

- [ ] **删除标题、摘要和方法贡献中的电力专用绑定**：保留 power equipment inspection 作为应用场景和关键词之一，但不再用它定义 FROST-VL 的结构来源或适用范围。
- [ ] **降低 MMRL 的叙事权重**：保留其对 shared representation 的启发，明确 FROST-VL 的核心问题是生成式 VLM 上的条件化外部偏移，不让方法看起来只是 MMRL 移植。
- [ ] **删除普遍优于 LoRA 或达到总体 SOTA 的暗示**：所有优势必须附带同骨干、同数据、同层范围或冻结前缀等条件；明确 Full-VLM LoRA 在领域准确率上的结果。
- [ ] **把 98.40% 严格限定为首 token 输出行为证据**：完成 MMBench 后才讨论带标准答案的通用能力；即使 MMBench 分数接近，也只写“在该基准上未观察到可测量下降”，不写“100% 保存全部原生能力”。
- [ ] **删除未经诊断支持的因果表达**：heterogeneous LR “reduces expert degradation”、Relation Loss “prevents drift”、缓存“improves TTFT”、少量运行证明“stable”等表述均须由新实验支持，否则改为中性观察。
- [ ] **弱化 edge deployment 叙事**：没有真实边缘设备结果时，只讨论单 GPU 训练、模块加载、缓存和推理开销，不把 RTX 5090 结果包装成边缘部署证据。
- [ ] **删除冗余 “Summary of experiments”**：把关键结果和限制合并到 Discussion，避免章节只重复表格数字。
- [ ] **删除或缩短与当前证据无关的宏大展望**：多领域 Softmax Gate、N 专家自动切换和跨模态文本支路只在 Future Work 简短保留，不作为当前贡献。
- [ ] **最后统一数字和英文**：摘要、正文、图注、结论、Highlights 和 Cover Letter 从最终表格同步数字；删除 66.95 等旧值，并按 humanizer 标准处理术语堆叠、重复 write-back、公式标点和不自然句子。
