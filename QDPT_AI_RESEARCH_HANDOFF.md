# QDPT 长期研究规划交接文档

> 更新日期：2026-09-22
> 用途：交给负责长期规划、机制讨论和论文定位的高级 AI。本文不是实验账本，而是当前可信事实、争议、约束、文件索引和待决策问题的入口。
> 当前 Git 分支：codex/dynamic-rep-cross-attention
> 编写时 HEAD：a8219a643e6c2c5e8dbb9f425229c15582b31695

## 1. 委托目标

请基于现有证据为 QDPT 制定长期但可执行的研究路线，并与用户进行细节讨论。最终形成足够明确的技术方案、实验矩阵、停止规则和论文叙事，再交回当前工程代理实施。

三个主线任务按优先级排列：

1. 稳定性提升：这是当前投稿能否继续的核心。只要在不显著扣分的前提下解决随机种子稳定性，就具备再次投稿价值。
2. 参数量优化：重要加分项，但不能以明显掉分或重新引入高方差为代价。
3. 网络结构可解释性与核心创新提炼：需要把方法从“多个旧模块拼接”提升为具有清楚问题定义、归纳偏置和机制证据的完整贡献。

用户不希望继续无边界结构搜索。优先给出少量高信息量、可证伪的实验；每一步必须说明它区分什么假设，以及什么结果触发停止。

## 2. 项目身份与历史命名

- 正式方法名：Question-Guided Directional Prompt Tuning（QDPT）。
- MMRL 是早期主线和曾用名，因为撞名而更换。仓库仍大量保留 MMRL 命名。
- 根目录 MMRL*.py 是历史正式代码谱系，但当前 Dense D768 Sandwich QDPT 的真实执行代码主要位于 slake/ 与 pathvqa/。
- QDPT/ 是准备开源的极简版本，当前实验和返修暂时不用它。
- 项目经过大量迭代，存在旧配置、旧结果和已被推翻的解释。必须遵守本文第5节的可信来源优先级。

## 3. 论文、笔记和仓库位置

### 3.1 本地与服务器

- Windows 唯一事实来源：E:\work\py\Qwen3-VL-modify-test
- AutoDL checkout：/root/autodl-tmp/Qwen3-VL-modify-test
- Git remote：https://github.com/123sswss/Qwen3-VL-modify-test.git
- 服务器模型：/root/autodl-tmp/model
- PathVQA：/root/autodl-tmp/dataset/pathVQA
- SLAKE：/root/autodl-tmp/dataset/slake
- Electrical 数据根：/root/autodl-tmp/dataset

工程规则：Windows 编辑、检查、提交并推送；服务器只做 fast-forward pull。拉取前执行 source /etc/network_turbo。未经用户明确授权，不在服务器改源码或启动实验。

### 3.2 论文

- 用户指定的审稿期论文 PDF：sn-article-template/qdpt-sn-revised.pdf
- Springer 模块化工作源：sn-article-template/qdpt-sn-source.tex
- 单文件投稿源：sn-article-template/qdpt-sn.tex
- 中文模块化论文：QDPT-paper/main.tex
- 中文编译 PDF：QDPT-paper/main.pdf
- 各章节：QDPT-paper/sections/*.tex
- 参考文献：QDPT-paper/references.bib、sn-article-template/qdpt-references.bib
- 图：QDPT-paper/figures/ 和 sn-article-template/figures/

当前没有单独的审稿意见原文文件。用户概括为：方法和实验仍有进步空间；论文文笔较差，建议重写；主要薄弱点是多 seed 稳定性、参数/性能性价比、方法特色和核心创新不够突出。当前先解决研究问题，论文重写稍后处理。

### 3.3 文献笔记

- 外部笔记：C:\Users\11473\Desktop\prompt-tuning-论文笔记.md
- 覆盖2025至2026年顶刊顶会中的 PEFT、Prompt、视觉 Prompt、初始化、注入接口和理论工作。
- 笔记用于提供问题意识与设计原则，不是候选模块清单。用户反对从论文中直接挑模块拼装，最终必须形成自己的方法。

重点思想包括：稳定任务知识与样本条件增量解耦；数据/语义相关初始化；Prompt 拼接可能引起位置偏移和注意力稀释；可以调节真实 token 或注意力 K/V；低维空间优先承担选择/打分，不应轻率压缩最终证据表达容量；Prompt 更擅长选择冻结模型已有能力，而非稳定创造新知识。

### 3.4 结果与计划

- 完整唯一实验账本：EXPERIMENT_RESULTS.md
- 精简结论：result.md
- 当前返修计划和停止规则：plan.md
- 项目代理规则：AGENTS.md
- 统一调度中心：run_experiment.sh
- 相关工作草稿：RELATED_WORK.md

警告：QDPT_SOURCE_AND_CONFIG.md 停留在2026-09-08，仍记录旧非-Sandwich配置、旧分数和旧“最佳结果”，已被后续实验取代。它只可用于寻找代码入口，不可作为当前性能与结论的事实来源。

### 3.5 诊断与服务器输出

- 样本级seed稳定性：diagnostics/analyze_qdpt_seed_stability.py
- 训练轨迹：diagnostics/analyze_mmrl_diagnostics.py
- checkpoint终点表征：diagnostics/analyze_qdpt_checkpoint_representations.py
- PathVQA配对干预比较：diagnostics/compare_pathvqa_conditioning_mismatches.py
- PathVQA输出根：pathvqa/outputs/；其下按dynamic_prompt、prompt_tuning、cocoop、lora等方法分目录。
- SLAKE输出根：slake/outputs/。
- Electrical输出根：electrical/outputs/。
- 正式服务器绝对路径均以上述AutoDL checkout为根。完整实验名和唯一输出路径必须从EXPERIMENT_RESULTS.md读取，不要凭目录时间猜测。

## 4. 当前正式方法

### 4.1 骨干和训练边界

- Backbone：Qwen3-VL-4B-Instruct，约4.445B参数。
- 视觉编码器、视觉 Merger 和 LLM 原权重全部冻结。
- “视觉侧微调”指通过 Layer17 静态视觉 Prompt 适配视觉路径，不是更新 ViT 原权重。
- 训练3 epochs；model seeds 44/45/46；data seed42；固定epoch3评估，不按 Validation 选 checkpoint。

### 4.2 Dense D768 Sandwich 数据流

最终 LLM 顺序：P20；Visual Tokens；Z10；Question/Chat Tokens。

- P20：20个静态 LLM Prompt token，20 x 2560，LR0.3。
- S8+A_v10：合计18个 Layer17 静态视觉 Prompt token，宽度1024；S8 LR3e-5，A_v10 LR1e-4。
- 问题上下文经注意力池化形成10个 D768 query。
- query 对视觉编码器代码索引17进入 Block 前的完整视觉 token 做16头 Cross-Attention。
- Z10 x 768 经 LayerNorm、768到768、GELU、768到2560生成动态文本残差。
- 动态残差加到文本 Anchor A_t10 x 2560，作为Z10插入LLM。
- 最终版本关闭动态视觉写回；条件信息只写入 LLM Prompt。
- 总可训练参数：7,805,184。

核心实现：

- slake/directional_concat_workspace.py
- slake/dynamic_prompt_tuning.py
- slake/dynamic_prompt_tuning_interface.py
- pathvqa/train_dynamic_prompt.py
- pathvqa/pathvqa_official_eval.py
- run_experiment.sh 中的 run_qdpt_d768_final_dataset()

## 5. 事实来源优先级

冲突时按以下顺序判断：

1. EXPERIMENT_RESULTS.md 中较新的明确更正或完整实验记录；
2. 原始服务器 summary、predictions、train report 和 diagnostics；
3. result.md；
4. plan.md；
5. 当前论文；
6. QDPT_SOURCE_AND_CONFIG.md 等旧索引；
7. 文件名、注释和历史目录不得单独作为事实依据。

禁止静默删除或覆盖历史结果；错误必须追加更正。

## 6. 当前主要结果

### 6.1 PathVQA

最终 Dense D768 Sandwich Validation：

| Seed | Overall | Yes/No | Free-form |
|---:|---:|---:|---:|
| 44 | 60.7765 | 92.7360 | 28.9087 |
| 45 | 57.2935 | 90.6880 | 23.9949 |
| 46 | 59.3865 | 91.5200 | 27.3452 |
| Mean +/- sample SD | 59.1522 +/- 1.7528 | 91.6480 +/- 1.0300 | 26.7496 +/- 2.5107 |

官方 Test 三 seed Overall：58.8827 +/- 1.8141。

对比：

- Static Prompt P20 Validation：55.0780 +/- 0.2244；Test：56.0798 +/- 0.3636。
- CoCoOp-style Validation：56.3136 +/- 1.1367；Test：56.9529 +/- 0.8363。
- Full-Attention LoRA-r8 Test约：59.6628 +/- 0.0232。
- QDPT Test均值比 LoRA 低约0.78，但远不稳定。

### 6.2 SLAKE

- QDPT Test三 seed：76.95 +/- 0.44。
- Full-Attention LoRA-r8：约81.82 +/- 0.22。
- QDPT明显落后LoRA，不能宣称跨数据集普遍优越。

### 6.3 Electrical

同一私有固定 holdout、seed47、排除18个无图关联样本后：

- Static Prompt：69.50
- CoCoOp-style：70.34
- LoRA-r8：70.96
- QDPT：71.38

QDPT单seed高LoRA 0.42，但缺少多seed与分项，不得宣称统计显著。

### 6.4 效率

RTX5090公平短测，完全相同的3,200样本和1,202,879视觉token：

- QDPT：7.3048 samples/s，峰值allocated 15.805GiB。
- LoRA-r8：3.8831 samples/s，峰值allocated 19.317GiB。
- QDPT训练吞吐约1.881x，峰值allocated显存低18.18%。
- 这是受控100 optimizer-step训练吞吐，不是完整训练墙钟时间。
- 当前QDPT参数7.805M，比LoRA-r8的7.078M多约10.27%，不存在参数量优势。

## 7. 已确认的重要机制事实

### 7.1 视觉信息没有简单“丢失”

用户的重要经验判断：只微调视觉侧时收益远低于只在LLM入口使用Static Prompt。这提示视觉编码器可能保留了相当多任务细节，主要问题是LLM没有按问题读取或理解这些细节。

现有实验支持但不完全证明该判断：

- 去掉视觉Cross-Attention、仅保留问题条件路径，seed44下降1.9332，where下降9.78。
- 错配视觉K/V会大幅下降约7.03。
- 正确问题到正确视觉K/V的检索确实有用，尤其是开放题和空间题。

### 7.2 视觉Prompt必要，但不是seed方差主因

- 在匹配的非-Sandwich seed44控制中，去掉全部S8+A_v10下降1.1184 Overall，Free-form下降2.1378，where下降12.4694。
- 因此不能删除视觉适配；18个视觉Prompt只有18,432参数，也不构成重模块。
- Visual18跨seed双向交换仅下降0.3036/0.2077；交换后保留原seed差距约97%。
- 结论：Visual18终点张量可移植，基本排除为多seed方差主因；不等于所有视觉适配均无关。

### 7.3 Static Prompt本身稳定

- Static P20三seed Overall SD仅0.2244。
- 不能把“soft prompt天然不稳定”作为论文结论。
- 方差呈LoRA约0.0666、Static P20 0.2244、CoCoOp-style 1.1367、QDPT 1.7528的递增趋势。
- 更稳妥的陈述是：当前冻结Qwen3-VL协议下，高自由度条件动态Prompt表现出明显初始化敏感性。

### 7.4 P20和文本Anchor存在终点共适应，但交换不能定位根因

seed44/45跨checkpoint交换：

- P20双向交换下降6.4867/2.2208。
- 文本Anchor双向交换下降7.7488/7.9086。
- 更好的seed44组件放入seed45也没有提升。

这证明P20、A_t10和其余生成器形成seed特异坐标系，不是可拔插模块；但不能证明P20或Anchor中的某一个导致训练方差。用户已指出：共享工作空间、梯度与前向耦合使这种不可交换性本身很合理。

### 7.5 样本级不稳定

PathVQA三seed共6,259题：全对3,233；全错2,112；部分seed正确914；prediction agreement 61.0641%。方差明显影响Free-form和where，不是单纯二分类阈值问题。

### 7.6 Prompt位置的强机制叙事已被削弱

- Sandwich：59.1522 +/- 1.7528。
- All-after：58.9658 +/- 0.4799。
- Reversed：58.5450 +/- 0.5008。

Sandwich均值只比All-after高0.1864，却有大得多的方差；逐seed排序反转。因此不能宣称Sandwich位置稳定或普遍最优。保留Sandwich是因为它在补多seed前按seed44 Validation选定并已冻结完成Test/SLAKE，不应事后换主模型。

### 7.7 问题条件平均有效，但不是逐seed稳定优势

容量匹配Learned Query三seed均值58.3214 +/-0.6730，比问题引导QDPT平均低0.8308；问题条件在两seed获胜，但seed45反转。可主张平均有效，不能主张逐seed稳定优越。

## 8. 已失败或不应优先重复的路线

1. 简单R256输出头：参数降到6.101M，但seed44下降3.3552，where下降17.36。不要继续扫描R160或多个rank。
2. 动态视觉写回：增加参数和不稳定性，没有稳定收益。
3. 增加视觉层：Layer18和共享17/18/19均未超过Layer17。停止层搜索。
4. 统一V20替代双速率S8+A_v10：Sandwich中约55.70，比最终seed44低约5.08。
5. 延长训练：QDPT 10-epoch marathon未超过3-epoch版本；其他长训还发生数值发散。
6. Prompt位置单seed解释：多seed已排序反转。
7. 跨seed模块交换作为性能方案：只诊断兼容性。
8. 完整6倍率乘3seed残差缩放扫描：成本过高，也可能只压低最佳seed机械降低方差，已放弃为P0。
9. 直接照搬文献模块：用户要求形成自己的方法，不接受论文组件拼盘。

## 9. 当前真正难点

### 9.1 稳定性因果源未唯一定位

较有把握排除：Static P20本身普遍不稳定；Visual18终点张量是主因；单纯训练不足；单一范数即可解释性能。

仍可能的来源：

- P20在QDPT联合训练中被动态分支拖入不同坐标系；
- A_t10的高LR随机轨迹；
- 问题池化、视觉Cross-Attention与输出头的高自由度联合初始化；
- 动态Prompt方向而非幅度差异；
- 模块互相补偿，使多个不同内部解获得相近训练loss却不同泛化。

### 9.2 参数主要花在自由生成，不在Prompt本体

- P20+A_t10只有76,800参数；Visual18只有18,432参数。
- 其余问题投影、视觉投影、Cross-Attention和D768到LLM输出头约7.71M。
- 真正重的是条件聚合和全维Prompt生成器。
- R256失败说明不能简单压缩最终表达容量。更可能的方向是压缩证据打分/选择，保留全维视觉值或LLM表达。

### 9.3 创新叙事容易被看成旧模块拼装

当前结构是问题池化、Cross-Attention、MLP动态Prompt、Static视觉Prompt和Sandwich位置。各部件都不新，多seed又削弱了位置机制。

需要回答：

- QDPT解决冻结MLLM的哪个明确接口缺陷？
- 为什么该缺陷适合在Prompt空间而不是LoRA权重空间处理？
- 能否表述为“稳定任务先验 + 查询选择的视觉证据”，而不是自由生成前缀？
- 能否让动态Prompt追溯到真实视觉token/区域，形成可验证的证据选择机制？

## 10. 当前等待的首个稳定化实验

最新实现commit：a8219a6。

实验语义：

- seed44 QDPT加载并冻结独立Static P20 seed44 epoch3；
- seed45 QDPT加载并冻结独立Static P20 seed45 epoch3；
- 不是加载已与QDPT共适应的最终P20；
- A_t10、Visual18、问题池化、视觉投影、Cross-Attention和输出头全部按对应QDPT seed重新初始化并训练；
- 总可训练参数7,753,984，只少冻结的51,200个P20参数；参数下降不是目标。

启动目标：pathvqa_qdpt_frozen_static_p20_sandwich_seeds44_45。MMRL_SHUTDOWN_ON_EXIT必须为0。

状态：代码已提交推送，本文编写时尚未报告实验结果。

预注册判断：

- 两seed平均不得比原seed44/45平均59.0350低超过0.30；
- 两seed差距应从3.4830降到不高于1.50；
- Free-form不能两个seed同方向明显下降；
- 若差距缩小主要来自seed44大幅下降，判定失败；
- 不通过则停止，不补seed46，不叠加Gate、额外正则或学习率补丁抢救。

该实验判断：稳定Standalone P20在联合QDPT训练中继续漂移是否是重要反馈源。若差距仍巨大，应转向A_t10与条件检索/生成器内部。

## 11. 值得高级AI重点论证的长期方向

以下是待讨论方向，不是已决定方案。

### 11.1 稳定锚点与条件增量解耦

候选问题定义：Static Prompt提供稳定领域/任务先验，动态分支只表达当前图像和问题带来的条件增量。当前联合训练允许两者相互补偿并漂移。

需要论证：

- 冻结P20是否足够，还是A_t10也应来自稳定语义锚点；
- 如何避免共享同一最佳seed锚点而人为消除方差；
- 数据相关初始化、分阶段训练、幅度约束中哪项是最小单因素干预；
- 如何确保稳定性改善不是牺牲最佳seed。

### 11.2 从自由生成Prompt转向选择真实视觉证据

潜在统一方向：只在低维空间学习问题到视觉token的打分，Value保持Qwen已对齐的原生全维视觉表示；动态Prompt由真实视觉token加权组合形成，而不是用大MLP在整个2560维空间自由生成。

概念：

score = (Q Wq) (V Wk)^T
evidence = softmax(score) V
dynamic prompt = stable anchor + bounded(evidence)

可能同时带来：

- 较低自由度和更好的初始化稳定性；
- 只压缩选择空间而不压缩证据表达容量，避开R256失败；
- 减少Wv/Wo和D768到2560生成头参数；
- Prompt可映射回视觉区域，增强可解释性；
- 形成“selection instead of synthesis”的独立主张。

风险：这是比当前投稿修复更大的结构改造。高级AI需判断它属于当前返修、后续版本，还是论文讨论。

### 11.3 论文核心贡献重构

较有潜力的叙事：

> 冻结MLLM往往保留领域视觉细节，但语言解码未必会根据当前问题读取相关证据。QDPT在不修改骨干权重的前提下，用稳定领域先验约束适配空间，并把问题选择出的视觉证据重组为LLM可消费的Prompt。

必须接受的边界：

- 不能宣称普遍优于LoRA；
- 不能宣称Sandwich位置稳定最优；
- 不能宣称所有Prompt方法都不稳定；
- 不能宣称视觉Prompt无用；
- 可以强调训练吞吐、显存、任务偏好以及Prompt空间和权重空间的互补。

## 12. 希望高级AI最终交付

1. 当前不稳定性的因果假设树，明确已排除和仍开放项。
2. 按信息增益和成本排序的实验序列，每一步最多改变一个主要因素。
3. 每项实验的成功、灰区、失败门槛和停止条件。
4. 当前返修最小方案与下一代结构升级方案，不要混在一起。
5. 参数优化路线：压缩哪部分、为什么不会重复R256失败。
6. 可解释性方案：需要哪些因果干预、可视化和定量指标，而不只展示注意力图。
7. 论文主张清单：哪些能写、哪些必须降级、哪些需要新证据。
8. 对PathVQA、SLAKE和Electrical差异的统一解释。
9. 给工程代理的精确实现规格：输入输出、张量形状、初始化、冻结边界、参数组、学习率、日志、checkpoint兼容性和一键目标。

## 13. 协作偏好与禁区

- 用户熟悉项目历史，含义不清时直接询问，不要凭文件名猜。
- 一次讨论一个关键问题，不要未讨论完就提前下结论或开始大规模实现。
- 用已有实验约束新假设，不能只凭直觉推荐流行模块。
- 多seed均值和方差优先于最佳seed；禁止挑seed。
- Validation用于设计，Test仅在配置冻结后评估。
- 不在SLAKE调参；PathVQA方案冻结后原样迁移。
- 参数优化排在稳定性之后。
- 用户已授权工程代理完成代码后自动提交推送，但服务器实验仍由用户启动。
- 不附带自动关机命令。

## 14. 建议阅读顺序

1. 本交接文档；
2. sn-article-template/qdpt-sn-revised.pdf；
3. plan.md；
4. result.md；
5. EXPERIMENT_RESULTS.md 中2026-09-10之后的最终套件、稳定性、位置、多seed和模块交换记录；
6. 外部prompt-tuning-论文笔记.md；
7. 只有需要可实现细节时再读当前核心代码；
8. 不要从QDPT/极简版或旧QDPT_SOURCE_AND_CONFIG.md开始理解当前方法。
