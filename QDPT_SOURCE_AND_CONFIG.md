# QDPT 源代码与最佳配置索引

> 更新时间：2026-09-08  
> 用途：集中记录当前主方法的代码入口、真实执行结构、最佳已完成配置和复现位置。实验结论仍以`EXPERIMENT_RESULTS.md`为唯一完整账本。

## 1. 当前方法身份

- 方法名：Question-Guided Directional Prompt Tuning（QDPT）。
- 当前高性能版本：QDPT-D768。
- 骨干策略：冻结视觉编码器、视觉 Merger 和 LLM，仅训练 Prompt 与 Directional 模块。
- 当前最佳已完成单次结果：PathVQA Validation seed44，3 epochs，Overall **59.5622**。
- 当前可靠多 seed 结果：PathVQA Validation seeds44/45/46，Overall **59.0030 +/- 0.4859**。
- 10-epoch marathon已完成：最佳为epoch6的58.7794，未超过原3-epoch配置，因此不替代当前最佳配置。

## 2. 核心源代码位置

| 文件 | 关键对象或入口 | 作用 |
|---|---|---|
| `slake/directional_concat_workspace.py` | `DirectionalConcatWorkspaceVisual` | QDPT核心：问题注意力池化、视觉Cross-Attention、静态视觉Prompt插入和共享表示`Z`。|
| `slake/directional_concat_workspace.py` | `ZeroInitWorkspaceProjection` | 将`Z10 x D768`升维为LLM空间动态残差，并加到文本锚点`A_t10`。|
| `slake/sparse_visual_mmrl.py` | `SparseVisualInjectionBlock` | 在指定视觉Block前拼接视觉Prompt、只执行一次冻结Block、随后立即Strip。|
| `slake/dynamic_prompt_tuning.py` | `DynamicPromptTuningModel` | 总模型封装；冻结骨干、安装视觉钩子、构造LLM入口Prompt、保存和加载QDPT checkpoint。|
| `slake/dynamic_prompt_tuning.py` | `trainable_parameter_groups()` | 将可训练参数划分为文本Prompt、视觉私有Prompt和Directional Workspace优化器组。|
| `slake/dynamic_prompt_tuning_interface.py` | `DynamicPromptTuningModelInterface` | 从checkpoint重建QDPT并执行生成式推理。|
| `pathvqa/train_dynamic_prompt.py` | `DynamicPromptTrainer` | 创建分组AdamW优化器并为不同参数组设置学习率。|
| `pathvqa/train_dynamic_prompt.py` | `DynamicPromptCallback` | 写诊断日志并在每个epoch保存轻量QDPT checkpoint。|
| `pathvqa/train_dynamic_prompt.py` | `main()` | PathVQA训练入口、参数审计和`train_report.json`写出。|
| `pathvqa/data_pipeline.py` | `PathVQADataset`、`PathVQADataCollator` | PathVQA训练数据、chat模板、标签和上下文mask。|
| `pathvqa/pathvqa_official_eval.py` | `run_inference()`、`main()` | PathVQA完整Validation/Test生成与结果落盘。|
| `pathvqa/pathvqa_vqa_metric.py` | `evaluate_pathvqa_predictions()` | Overall、Yes/No、Free-form、问题类型和图像簇bootstrap指标。|
| `run_experiment.sh` | `run_qdpt_d768_final_dataset()` | 当前统一D768配置生成器及参数审计。|
| `run_experiment.sh` | `run_pathvqa_directional_concat_workspace_text_dynamic_only_compressed()` | 产生最佳3-epoch D768结果的原始PathVQA入口。|
| `test_dynamic_prompt_tuning.py` | Dynamic Prompt单测 | 检查Prompt注入、冻结边界、checkpoint和生成路径。|
| `test_sparse_visual_mmrl.py` | 视觉注入与Directional单测 | 检查单次Block执行、插入/Strip、Q/K-V路径及参数共享。|
| `test_pathvqa_directional_interventions.py` | 错配机制单测 | 检查问题Q错配、视觉K/V错配和干预范围。|

### Marathon附加代码

| 文件 | 作用 |
|---|---|
| `pathvqa/marathon_validation.py` | 使用训练中的同一模型在epoch3以后逐轮跑完整Validation，避免重复加载4B骨干。|
| `pathvqa/marathon_progress.py` | 实时写入精简的`marathon_progress.tsv`。|
| `test_pathvqa_marathon_validation.py` | 检查epoch3-10调度、日志字段和重复epoch保护。|

## 3. 当前最佳结构的真实数据流

### 3.1 问题到共享表示

1. 回答开始前的非视觉上下文Token为`T: L_text x 2560`。它包含问题文本及chat模板/角色控制Token；视觉特殊Token和监督答案Token被排除。
2. 可学习打分投影产生`Score: L_text x 10`，沿文本长度做Softmax。
3. 可学习Value投影将文本降维为`V_text: L_text x 768`。
4. `Softmax(Score^T) @ V_text`得到十个问题查询`Q: 10 x 768`。
5. 读取视觉编码器代码索引17处、进入该Block前的完整视觉Token：`H17: L_visual x 1024`。
6. 视觉投影得到`K,V: L_visual x 768`，执行16头Cross-Attention。
7. 共享条件表示为`Z = Q + CA(Q,H17,H17)`，形状`10 x 768`。

### 3.2 视觉侧

- 私有静态视觉Prompt：`S_v: 8 x 1024`，学习率`3e-5`。
- 静态视觉锚点：`A_v: 10 x 1024`，属于Directional Workspace组，学习率`1e-4`。
- 在视觉Block代码索引17之前拼接`[S_v8; A_v10; H17]`，共18个新增视觉Token。
- 冻结视觉Block只执行一次，随后立即Strip前18个Token，原视觉序列继续向后传播。
- 当前最佳配置关闭动态视觉写回：`Z`不会再次投影并写入视觉编码器。

### 3.3 LLM侧

- 私有静态文本Prompt：`P: 20 x 2560`，学习率`0.3`。
- 文本锚点：`A_t: 10 x 2560`，学习率`0.3`。
- `Z10 x 768`经过`LayerNorm -> Linear(768,768) -> GELU -> Linear(768,2560)`得到动态残差`Delta P10`；末层Linear零初始化。
- 十个动态文本Token为`A_t10 + Delta P10`。
- LLM入口实际拼接的是`[P20; A_t10 + Delta P10; 原始图文序列]`，因此**实际新增30个LLM Prompt Token**，不是20个。

## 4. 当前最佳3-epoch配置

### 4.1 结构参数

| 配置项 | 固定值 |
|---|---:|
| 私有文本Prompt `P` | 20 tokens x 2560 |
| 动态文本锚点 `A_t` | 10 tokens x 2560 |
| 问题Query `Q` | 10 tokens x 768 |
| Directional宽度 `D` | 768 |
| Directional Cross-Attention | 16 heads |
| 私有视觉Prompt `S_v` | 8 tokens x 1024 |
| 静态视觉锚点 `A_v` | 10 tokens x 1024 |
| 视觉插入位置 | 代码索引17，仅一层 |
| 动态视觉写回 | 关闭 |
| 视觉条件 | 当前回答前非视觉上下文生成Q，读取当前完整视觉K/V |
| LLM新增Prompt总长度 | 30 tokens |
| 可训练参数 | 7,805,184 |

### 4.2 视觉层编号警告

配置和运行审计使用Python的零基索引：`anchor_layers=[17]`，自然计数是视觉编码器第18个Block。现有实验记录沿历史命名常写“Layer17”。论文图、正文和代码说明必须统一采用一种编号，并明确说明是否零基，禁止把Layer17和第17个Block混用。

### 4.3 优化参数

| 参数组 | 学习率 | 备注 |
|---|---:|---|
| `soft_prompt` | 0.3 | 包含`P20`和`A_t10`。|
| `sparse_visual` | 3e-5 | 仅包含私有视觉`S_v8`。|
| `shared_workspace` | 1e-4 | 包含`A_v10`、问题池化、视觉投影、Cross-Attention和`Z -> LLM`投影。|
| `dynamic_prompt` | 3e-4配置值 | Directional模式下该参数组为空，此学习率对当前QDPT不产生更新。|

其他训练设置：

- 优化器：AdamW，`betas=(0.9,0.999)`，`eps=1e-8`，weight decay为0。
- 学习率调度：linear，warmup ratio为0.03。
- 最大梯度范数：1.0。
- micro batch：2。
- gradient accumulation：16。
- effective batch：32。
- 训练轮数：3。
- optimizer steps：1,845。
- model/training seed：44；data seed：42。
- 评测：仅epoch3完整PathVQA Validation，不跑Test。

## 5. 最佳已完成结果与文件

### 5.1 PathVQA seed44

- 实验名：`pathvqa_directional_concat_workspace_text_dynamic_only_z10_d768_l17_private_p20_s8_seed44_20260901`。
- Overall：**59.5622**。
- Yes/No：**91.0400**。
- Free-form：**28.1749**。
- 图像簇95% CI：**[58.0759, 60.9533]**。
- 训练时间：6,440.81秒。
- 训练loss：11.2930。
- 服务器输出目录：`/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_directional_concat_workspace_text_dynamic_only_z10_d768_l17_private_p20_s8_seed44_20260901`。
- checkpoint：上述目录下`checkpoints/epoch_3`。
- Validation结果：上述目录下`eval_validation/epoch_3`。

### 5.2 PathVQA三seed

| Seed | Overall | Yes/No | Free-form |
|---:|---:|---:|---:|
| 44 | 59.5622 | 91.0400 | 28.1749 |
| 45 | 58.6835 | 90.3360 | 27.1219 |
| 46 | 58.7634 | 92.2560 | 25.3669 |
| Mean +/- SD | 59.0030 +/- 0.4859 | 91.2107 +/- 0.9709 | 26.8879 +/- 1.4190 |

## 6. 复现入口

复现原始PathVQA 3-epoch D768 seed44：

```bash
RUN_TARGET=pathvqa_directional_concat_workspace_text_dynamic_only_d768_seed44 bash run_experiment.sh
```

当前10-epoch收敛诊断：

```bash
RUN_TARGET=pathvqa_qdpt_d768_marathon_seed44 bash run_experiment.sh
```

Marathon在epoch3-10逐轮评测完整Validation，实时精简表位于对应输出目录的`marathon_progress.tsv`。它采用10-epoch线性调度，故其中epoch3不是原3-epoch线性调度终点的逐位复现。该实验已于2026-09-08完成，epoch6峰值58.7794，epoch10为57.1657；保留原3-epoch配置。

## 7. 整理与写作时的事实边界

1. 单次最高分使用59.5622时必须明确是seed44；稳定主结果应报告三seed均值59.0030及标准差。
2. 论文可把`S_v8+A_v10`合称18个Layer17静态视觉Prompt，但必须如实说明它们属于两个学习率组。
3. 不得把LLM入口描述成只有20个Prompt Token；当前实现是20个私有Token加10个条件锚点，共30个。
4. 不得声称动态`Z`写回视觉编码器；当前最佳配置只把`Z`写入LLM Prompt。
5. `dynamic_lr=3e-4`是通用训练器遗留配置，当前Directional模式没有对应参数，论文超参数表不应把它写成有效QDPT学习率。
6. Marathon峰值低于原3-epoch结果，不得用其中任何分数替换当前最佳配置或三seed主结果。
7. 当前问题池化不是严格的raw-question-only编码；源码使用`mmrl_gating_mask`选取回答前上下文，再排除视觉模板Token。论文可称question-guided，但若写“only raw question tokens”则与实现不符。
