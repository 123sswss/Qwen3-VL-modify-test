# Related Work 与方法碰撞记录

## DRAPE：当前首要技术近邻

- 论文：*Dynamic Cross-Modal Prompt Generation for Multimodal Continual Instruction Tuning*，arXiv:2605.10765。
- 方法名：DRAPE。
- 任务：rehearsal-free multimodal continual instruction tuning，重点解决顺序任务学习中的灾难性遗忘。
- 主干：冻结视觉编码器和 LLM；训练当前任务的跨模态 Prompt 生成器，并以更小学习率更新共享视觉 projector。
- 默认设置：生成器隐宽 `H=512`，实例 Prompt 长度 `Lp=10`，生成器学习率 `2e-4`，projector 学习率 `2e-5`。

### 核心信息流

1. 将当前指令 Token 投影到隐空间。
2. 把指令分为 `Lp` 段并分别做 masked mean pooling，得到 `Lp` 个初始摘要。
3. 以这些摘要为 Query，对完整指令 Token 做文本注意力，生成 instruction-aware `Q`。
4. 将当前图像视觉特征投影到同一隐空间，作为视觉 K/V。
5. 计算 `R = LN(Q + MHA(Q, V, V))`。
6. 使用两层 MLP `H -> 2H -> d_llm` 生成实例级 Soft Prompt，并拼到冻结 LLM 输入前。
7. 持续学习部分为每个任务保存独立生成器，以 null-space gradient projection 保护共享 projector，并用 CLIP 文图 prototype 选择推理生成器。

### 与 QDPT 的重合

- 当前文本/问题决定 Query。
- 当前图像特征提供 K/V。
- Cross-Attention 生成逐样本跨模态表示。
- 将表示映射为动态 LLM Soft Prompt。
- 使用模态不对称解释：文本说明需要什么，图像提供证据。
- 动态槽数均采用10，隐宽均被当作可调容量参数。

因此，QDPT 不得宣称首次提出上述通用结构或思想。DRAPE 必须在 Related Work 中作为最接近方法重点讨论，而不能只按“不同任务”一笔带过。

### QDPT 可守住的边界

- DRAPE 面向多任务持续学习；QDPT 面向单领域医学 VQA。
- DRAPE 更新共享视觉 projector；QDPT 冻结视觉编码器、visual merger/projector 和 LLM。
- DRAPE 读取 projector 后的视觉特征；QDPT 从冻结视觉编码器 Layer17 内部读取证据。
- DRAPE 每任务保存生成器并使用 CLIP 路由；QDPT 每领域使用一个共享适配器，不需要任务路由。
- QDPT 将静态领域锚点与动态样本证据分开，并保留经过消融验证的18个 Layer17 静态视觉 Prompt。
- QDPT 已有问题错配、视觉 K/V 错配、learned static query、动态视觉写回关闭、静态视觉移除和层位敏感性等干预证据。

### DRAPE 使用的数据集

DRAPE 原论文结果用于说明持续学习机制，不能与 PathVQA/SLAKE 数值直接排名。

- CoIN：ScienceQA、TextVQA、ImageNet、GQA、VizWiz、Grounding/RefCOCO、VQAv2、OCR-VQA 等顺序任务。
- UCIT：ArxivQA、CLEVR-Math、IconQA、ImageNet-R、VizWiz-caption、Flickr30k 等顺序任务。
- 主要持续学习指标：最终平均准确率、Backward Transfer 和 Mean Accuracy。

QDPT 不需要照搬这些数据集。PathVQA 与 SLAKE 继续承担医学领域主验证，自建电气数据集承担补充应用；DRAPE 的数据集只用于说明其任务范围和跨任务多样性。

### DRAPE 的对比方法

- Prompt持续学习：CODA-Prompt、DualPrompt、L2P、ModalPrompt。
- LoRA/模块持续学习：MoELoRA、ProgLoRA、O-LoRA、LoRA-FT、Continual LLaVA、CL-MoE。
- 其他持续学习方法：HiDe、SEFE。

这些方法多数不是 QDPT 的同协议直接基线。可用于扩充 Related Work 和解释 Prompt/LoRA 在持续学习中的发展，但不得把其原论文分数放入 PathVQA/SLAKE 受控主表。与 QDPT 最相关、可迁移的基线思想是 Static Prompt、Learned Query、Mean Pooling 和 w/o Cross-Modal Attention。

## 可直接迁移的实验与图表

### 必须进入正文

1. Static Prompt 与 QDPT：证明任务级固定先验不能覆盖逐样本图文差异。
2. Learned Static Query 与问题来源 Query：证明通用视觉摘要不能替代当前问题引导。
3. Question-only / w/o Visual CA：证明正确视觉检索相对纯问题条件 Prompt 的独立贡献。
4. D256/D512/D768/D1024：报告容量悬崖、效率点和性能平台；将其视为与 DRAPE 相互呼应的独立观察，不宣称独家发现。
5. 参数、训练时间、TTFT、TPOT：完整报告效率边界，不只给准确率。

### 使用现有 checkpoint 完成的低成本分析

1. Prompt t-SNE：比较静态锚点与动态 Prompt；动态 Prompt 按问题类型、答案类型或图像簇着色。
2. Prompt-to-image Attention：展示 Q10 对 Layer17 视觉 Token 的注意力，不把热力图单独当作因果证据。
3. 同图不同问题：固定一张医学图像，选择询问不同病变、位置或属性的问题，展示注意力和答案随问题变化。
4. 成功/失败案例：分别展示 Static Prompt 失败而 QDPT 成功、两者都成功、QDPT失败的样本，避免选择性展示造成过度宣传。

### 不应照搬

- CLIP prototype routing、路由混淆矩阵：QDPT 没有任务生成器池。
- Null-space gradient projection、Backward Transfer：QDPT 不是持续学习任务。
- 更新视觉 projector：可能提高准确率，但会破坏完全冻结边界并进一步靠近 DRAPE，当前不采用。
- 指令分段池化：可能改善槽多样性，但医学 VQA 问题较短，强制切为10段缺乏语义保证；不因此重启架构搜索。

## 正文叙事模板

先肯定 DRAPE：它系统证明了实例级跨模态 Prompt 能在多模态持续学习中适应任务内部的细粒度图文差异，并结合 projector 保护和任务路由缓解灾难性遗忘。

再指出缺口：持续学习设置允许更新共享视觉 projector，并依赖任务专属生成器和路由；它没有回答单领域适配中，当基础 MLLM 完全冻结时，应从哪个视觉阶段读取证据、是否需要把条件信息写回视觉编码器，以及静态领域先验如何与动态样本证据协作。

最后引出 QDPT：QDPT 将静态 Prompt 定义为领域级锚点，将问题引导的动态 Prompt 定义为样本级证据，通过 Layer17 内部视觉读取和 read-only-vision/write-only-language 结构完成完全冻结适配，并用图文错配与路径干预验证其工作机制。

禁止使用“首次”“首个”“前所未有”等优先权表述。采用“we study”“we instantiate”“we provide controlled evidence”等可验证措辞。
