# 旧论文引用回收审计

来源：`../els-cas-templates/references.bib`

本次只检查旧稿使用的引用键和 BibTeX 元数据，不沿用旧稿的方法、实验或结论。`body_en.tex` 共引用 38 个唯一条目，`references.bib` 也包含 38 个唯一条目；没有缺失引用，也没有未使用条目。

## 可直接进入 QDPT 文献框架

### PEFT 与 Prompt Tuning

- `houlsby2019parameter`：Adapter/参数高效迁移学习的基础文献。
- `hu2022lora`：LoRA，QDPT 的主要权重空间 PEFT 参照。
- `lester2021power`：Soft Prompt Tuning，Static Prompt 基线的直接来源。
- `li2021prefix`：Prefix Tuning，可用于交代连续 Prompt 的发展。
- `jia2022visual`：Visual Prompt Tuning，说明 Prompt 不只用于语言侧。
- `sung2021vl`：VL-Adapter，多模态参数高效迁移的早期代表。
- `zhou2024empirical`：MLLM 参数高效微调的实证比较，与论文定位直接相关。

### 多模态骨干与查询式视觉聚合

- `bai2025qwen3`：本文骨干 Qwen3-VL 的必要引用。
- `wang2024qwen2`：Qwen-VL 系列背景；若篇幅紧张，可以只保留 Qwen3-VL。
- `li2023blip`：BLIP-2/Q-Former，适合放在查询式视觉证据聚合部分。
- `dai2023instructblip`：问题/指令感知视觉建模的相关背景。
- `alayrac2022flamingo`：冻结视觉与语言组件之间的视觉语言连接方式。
- `liu2023visual`：LLaVA/Visual Instruction Tuning，交代生成式 MLLM 背景。
- `radford2021learning`：CLIP；只有在讨论视觉语言预训练或 DRAPE 的 CLIP 路由时才需要。

### 通用综述与电力电气应用

- `zhang2024vision`：视觉语言模型综述，可用于 Introduction 的背景句。
- `wang2025parameter`：PEFT 综述；更适合 Related Work，不替代原始方法论文。
- `wang2024power`：Power-LLaVA，电力巡检场景的重要已有工作。
- `rocha2025advances`：多模态大模型用于电网资产巡检。
- `lin2025defectgpt`：小样本电气缺陷识别，适合后续电气数据集部分。

## 视篇幅决定是否使用

- `dettmers2023qlora`、`liu2024dora`：LoRA 的后续变体。若实验只比较标准 LoRA，可在一处合并引用，不必展开。
- `liu2022few`：适合讨论少样本 PEFT，但不是 QDPT 的直接技术近邻。
- `sharshar2025vision`、`jeon2026edgev`：只有在论文强调边缘部署时使用。
- `hao2025power`：电网巡检应用背景，需先核验出版信息和论文质量。
- `wu2026domain`：光伏故障诊断与动态 LoRA 路由；可作为电气领域适配工作，但任务和 QDPT 不同。

## 当前不建议回收

以下文献主要服务于旧路线中的持续学习、灾难性遗忘、MoE、稀疏门控或 MMRL，不应为了增加引用数量而带入 QDPT：

- `zhai2023investigating`
- `zhu2024model`
- `tuning83same`
- `ge2025dynamic`
- `guo2025mmrl`
- `li2026fine`
- `gao2025enhanced`
- `lin2026moe`
- `louizos2017learning`
- `shazeer2017outrageously`
- `fedus2022switch`

其中 `tuning83same` 的作者、卷期和页码明显损坏，禁止复用。

## 复用前必须修正的 BibTeX 问题

1. `hu2022lora` 把 ICLR 写成了带卷期和页码的期刊条目，应改为正式会议格式。
2. `alayrac2022flamingo` 仍记为 arXiv，终稿宜更新为 NeurIPS 版本。
3. `lin2025defectgpt` 的 `booktitle = {BMVC.}` 不完整，需要核验正式会议信息。
4. 多数条目没有 DOI、URL 或正式出版页码，引用前应从论文主页、出版社或 DBLP 核验。
5. LoRA、Qwen、BLIP、MLLM、LLaVA 等专名需要用花括号保护大小写。
6. 2026 年条目需要逐篇确认是否已经正式公开，不能仅凭旧库中的年份和期刊字段引用。

## 已补入新项目的文献

- DRAPE：`Hu2026DynamicCP`。
- CoTBox-TTT：`Qian2025CoTBoxTTTGM`。
- PathVQA：`He2020PathVQA3Q`。
- SLAKE：`Liu2021SlakeAS`。
- CoCoOp/Conditional Prompt Learning：`Zhou2022ConditionalPL`。
- 已修正会议类型的 LoRA 与 Flamingo：`Hu2022LoRA`、`Alayrac2022FlamingoAV`。
- Qwen3-VL、Power-LLaVA 和四项电力电气相关工作。

新库位于 `references.bib`。用户提供的 Rocha 与 Hao 条目各重复一次，新库只保留一个。DefectGPT 根据 BMVC 2025 页面补全为七位作者；当前没有页码或论文编号，因此不虚构该字段。

## 仍需补充

- 其他与“按样本生成 Prompt”直接相关、且比 CoCoOp 更贴近生成式 MLLM 的条件 Prompt 工作。
- 后续实际使用的基线原始论文；不为凑数量预先加入。

## 建议的引用分配

- Introduction：VLM/MLLM 背景、LoRA、Prompt Tuning、MLLM PEFT 实证研究。
- PEFT Related Work：Adapter、LoRA、Prefix Tuning、Prompt Tuning、VPT、VL-Adapter。
- Cross-Modal Prompt Related Work：BLIP-2、InstructBLIP、DRAPE，以及后续补充的条件 Prompt 文献。
- Specialized-Domain VQA：Power-LLaVA、电网资产巡检、PathVQA、SLAKE；电气实验完成后再加入更细的电气缺陷与故障诊断文献。
