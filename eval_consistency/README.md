# 原模型输出扰动一致性评测

本目录用于量化不同视觉侧参数高效微调方法对 Qwen3-VL 原模型输出分布造成的扰动。

本实验的核心目标不是衡量微调模型与原模型是否“相似”，而是验证：

> 我们的方法对原模型通用能力造成的额外扰动，是否低至原模型重复运行产生的自然数值扰动水平。

因此，本实验中各项扰动指标均为**越小越好**。理想情况下，我们的方法与原模型之间的扰动应接近两次原模型运行之间的自然扰动。

## 一、实验组成

### 1. 原模型自然扰动

原模型在两个独立 Python 进程中分别运行一次：

- `base_run_1`
- `base_run_2`

每次运行都会重新加载相同的原模型 checkpoint。二者之间的差异用于估计原模型在真实运行环境中的自然数值扰动。

### 2. 七个视觉侧基线

本实验只比较修改视觉编码器的方法。修改 LLM 的全模型 LoRA、DoRA、IA3 和 Adapter 不属于本组核心实验。

七个基线分别是：

1. Vision Encoder LoRA rank 8
2. Vision Encoder LoRA rank 16
3. Vision Encoder LoRA rank 32
4. Vision Encoder DoRA rank 8
5. Vision Encoder DoRA rank 16
6. Last-8 Vision Encoder Full-Linear LoRA rank 32
7. Last-8 Vision Encoder Attention LoRA rank 32

对应 checkpoint：

```text
/root/autodl-tmp/Qwen3-VL-modify-test/loraTest/runs/lora_vision_attn/rank8/final
/root/autodl-tmp/Qwen3-VL-modify-test/loraTest/runs/lora_vision_attn/rank16/final
/root/autodl-tmp/Qwen3-VL-modify-test/loraTest/runs/lora_vision_attn/rank32/final

/root/autodl-tmp/Qwen3-VL-modify-test/loraTest/runs/dora_vision_attn/rank8/final
/root/autodl-tmp/Qwen3-VL-modify-test/loraTest/runs/dora_vision_attn/rank16/final

/root/autodl-tmp/Qwen3-VL-modify-test/loraTest/runs/lora_vision_last8/last8_full_linear_rank32/final
/root/autodl-tmp/Qwen3-VL-modify-test/loraTest/runs/lora_vision_last8/last8_attention_rank32/final
```

### 3. 我们的方法

`eval_ours.py` 已预留，但目前处于禁用状态。`run_all.py` 中对应的批量运行项也已注释，训练完成前不会运行。

## 二、文件说明

```text
eval_consistency/
├── config.py
├── common.py
├── analyze.py
├── eval_base.py
├── eval_vision_lora_rank8.py
├── eval_vision_lora_rank16.py
├── eval_vision_lora_rank32.py
├── eval_vision_dora_rank8.py
├── eval_vision_dora_rank16.py
├── eval_last8_lora_full_linear_rank32.py
├── eval_last8_lora_attention_rank32.py
├── eval_ours.py
├── run_all.py
├── run_all.sh
└── README.md
```

- `config.py`：原模型、数据集、图片目录及默认参数。
- `common.py`：统一的模型加载、数据读取、推理和原始 JSON 保存逻辑。
- `analyze.py`：计算自然扰动、方法扰动和汇总结果。
- `eval_*.py`：每个实验的独立运行入口。
- `run_all.py`：按照固定顺序批量运行所有实验。
- `run_all.sh`：Linux Shell 批量入口。
- `eval_ours.py`：我们的方法的预留入口，目前禁用。

## 三、默认配置

默认原模型：

```text
/root/autodl-tmp/model
```

默认评测数据：

```text
/root/autodl-tmp/dataset/llava_instruct_150k.json
```

默认图片目录：

```text
/root/autodl-tmp/dataset/gen/train2017
```

默认实验参数：

```text
样本数量：500
随机种子：114514
最大生成长度：64 tokens
生成方式：Greedy Decoding
do_sample：False
```

需要修改默认路径时，编辑 `config.py`。数据集和图片目录也可以通过命令行参数临时覆盖。

## 四、正式批量运行

进入项目根目录：

```bash
cd /root/autodl-tmp/Qwen3-VL-modify-test
```

使用 Python 批量入口：

```bash
python eval_consistency/run_all.py
```

也可以使用 Shell 入口：

```bash
bash eval_consistency/run_all.sh
```

程序将依次运行：

```text
1. Base run 1
2. Base run 2
3. Vision LoRA rank 8
4. Vision LoRA rank 16
5. Vision LoRA rank 32
6. Vision DoRA rank 8
7. Vision DoRA rank 16
8. Last-8 Full-Linear LoRA rank 32
9. Last-8 Attention LoRA rank 32
10. 统一统计分析
```

每个模型都在新的 Python 进程中加载和运行。这样可以避免模型复用造成的状态残留，并使两次 Base 运行能够反映重新加载模型后的实际自然扰动。

## 五、小规模检查

正式运行 500 条样本前，可以使用独立输出目录检查路径、依赖和显存：

```bash
python eval_consistency/run_all.py \
    --num-samples 2 \
    --max-new-tokens 8 \
    --output-root ./eval_consistency/smoke_results
```

小规模检查与正式实验必须使用不同输出目录，防止测试数据被误认为正式数据。

正式实验：

```bash
python eval_consistency/run_all.py \
    --num-samples 500 \
    --max-new-tokens 64 \
    --output-root ./eval_consistency/results
```

构建评测样本清单时，程序会检查图片文件是否真实存在。图片字段缺失、图片文件丢失或问题文本为空的样本会被自动跳过，并输出类似统计：

```text
[dataset] scanned=6, selected=2, missing_image_file=4, missing_image_field=0, empty_prompt=0
```

所有实验读取相同的数据集并执行相同的筛选规则，因此缺图样本不会导致不同模型之间发生样本错位。如果扫描完整个数据集后仍无法获得指定数量的有效样本，程序会停止并提示检查 `IMAGE_DIR`。如果绝大多数图片都被判定为缺失，通常代表图片根目录配置错误，而不是数据集中恰好丢失了大量图片。

## 六、单独运行某个实验

第一次原模型运行：

```bash
python eval_consistency/eval_base.py \
    --run-id base_run_1 \
    --output-root ./eval_consistency/results
```

第二次原模型运行：

```bash
python eval_consistency/eval_base.py \
    --run-id base_run_2 \
    --output-root ./eval_consistency/results
```

单独运行 Vision LoRA rank 8：

```bash
python eval_consistency/eval_vision_lora_rank8.py \
    --output-root ./eval_consistency/results
```

其他实验：

```bash
python eval_consistency/eval_vision_lora_rank16.py
python eval_consistency/eval_vision_lora_rank32.py
python eval_consistency/eval_vision_dora_rank8.py
python eval_consistency/eval_vision_dora_rank16.py
python eval_consistency/eval_last8_lora_full_linear_rank32.py
python eval_consistency/eval_last8_lora_attention_rank32.py
```

所有独立脚本均支持以下参数：

```text
--run-id              本次运行的唯一名称
--output-root         结果根目录
--num-samples         评测样本数量
--max-new-tokens      最大生成长度
--dataset-json        评测数据 JSON
--image-dir           图片目录
--seed                随机种子
```

## 七、中断后的继续运行

已经完成部分实验后，可以执行：

```bash
python eval_consistency/run_all.py --skip-existing
```

当以下文件存在时，该实验会被跳过：

```text
<output-root>/<run-id>/raw_results.json
```

例如：

```text
eval_consistency/results/base_run_1/raw_results.json
eval_consistency/results/vision_lora_rank8/raw_results.json
```

注意：当前实现按完整实验文件判断是否完成，不支持从单个实验的中间样本继续。若某个 `raw_results.json` 因异常而不完整，应删除该实验目录中的不完整结果，然后重新运行该实验。

## 八、只运行统计分析

模型推理完成后，可以单独重新计算全部指标：

```bash
python eval_consistency/analyze.py \
    --output-root ./eval_consistency/results
```

默认使用：

```text
base_run_1
base_run_2
```

作为自然扰动参考。如果使用了其他运行名称：

```bash
python eval_consistency/analyze.py \
    --output-root ./eval_consistency/results \
    --base-run-1 my_base_run_1 \
    --base-run-2 my_base_run_2
```

统计过程只读取已有 JSON，不重新加载模型。

## 九、输出目录结构

```text
eval_consistency/results/
├── base_run_1/
│   └── raw_results.json
├── base_run_2/
│   └── raw_results.json
├── vision_lora_rank8/
│   └── raw_results.json
├── vision_lora_rank16/
│   └── raw_results.json
├── vision_lora_rank32/
│   └── raw_results.json
├── vision_dora_rank8/
│   └── raw_results.json
├── vision_dora_rank16/
│   └── raw_results.json
├── last8_lora_full_linear_rank32/
│   └── raw_results.json
├── last8_lora_attention_rank32/
│   └── raw_results.json
└── reports/
    ├── base_natural_disruption.json
    ├── vision_lora_rank8.json
    ├── vision_lora_rank16.json
    ├── vision_lora_rank32.json
    ├── vision_dora_rank8.json
    ├── vision_dora_rank16.json
    ├── last8_lora_full_linear_rank32.json
    ├── last8_lora_attention_rank32.json
    └── all_experiments.json
```

## 十、原始 JSON 保存内容

每个 `raw_results.json` 的第一个字段都是 `conclusion`：

```json
{
  "conclusion": {
    "status": "raw_data_complete",
    "statement": "Raw evidence only; run analyze.py for the disruption conclusion."
  }
}
```

该结论表示原始实验数据已经保存，最终扰动结论需要由 `analyze.py` 计算。

### 实验元数据

`metadata` 中保存：

- 实验名称和运行 ID。
- 方法类型。
- 原模型路径。
- adapter checkpoint 路径。
- 数据集路径和 SHA256。
- 图片目录。
- 样本数量。
- 随机种子。
-生成配置。
- Python、PyTorch、CUDA、cuDNN 和 GPU 信息。
- 主机名、运行时间和进程 ID。

### 逐样本数据

每条样本保存：

- 稳定样本 ID。
- 数据集原始索引。
- 原始问题文本。
- 实际输入模型的 chat template 文本。
- 图片路径及图片 SHA256。
- 完整输入 token IDs。
- 输入 token IDs 的 SHA256。
- 首输出 token 的完整词表 logits。
- Top-50 token IDs 和对应概率。
- 完整生成 token IDs。
- 完整生成文本。

完整词表 logits 会转换为 FP16，然后编码成 Base64 写入 JSON。`analyze.py` 可以将其恢复为张量并重新计算指标。

## 十一、扰动计算方法

### 1. 自然扰动

对每条相同样本计算：

```text
D_natural = D(Base run 1, Base run 2)
```

它代表原模型在两个独立进程中重复运行时产生的自然数值扰动，是所有方法需要接近的目标下限。

### 2. 方法扰动

每种方法同时与两次 Base 比较：

```text
D_method =
    [D(Method, Base run 1) + D(Method, Base run 2)] / 2
```

这样可以避免只选择某一次 Base 作为参考而引入偶然偏差。

### 3. 额外扰动

```text
Excess Disruption = D_method - D_natural
```

解释：

- 接近 `0`：该方法造成的额外扰动接近自然扰动水平。
- 大于 `0`：该方法引入了超过自然扰动的额外污染。
- 数值越大：对原模型输出分布的破坏越明显。

### 4. 自然扰动倍数

```text
Natural Disruption Ratio = D_method / D_natural
```

解释：

- 接近 `1`：方法扰动约等于自然扰动。
- 明显大于 `1`：方法产生了额外扰动。
- 数值越大：相对自然噪声的污染越强。

如果自然扰动恰好为 `0`，程序会将该比例保存为 `null`，避免除零得到无意义结果。此时应主要解读绝对 JS 和 `Excess Disruption`。

## 十二、指标解释

### JS Divergence

衡量两个首 token 概率分布的差异，对称且数值稳定，是本实验的主要指标。

- 越小越好。
- `0` 表示两个概率分布完全相同。
- 论文主表建议优先报告平均 JS、P95 JS 和额外 JS。

### Symmetric KL

计算两个方向 KL 散度的平均值：

```text
[KL(A || B) + KL(B || A)] / 2
```

- 越小越好。
- 对概率分布尾部变化比 JS 更敏感。
- 适合作为 JS 的补充指标。

### Logits RMSE

衡量两个完整词表 logits 的均方根误差。

- 越小越好。
- 直接反映模型输出层数值变化。

### Logits Cosine Distance

衡量两个完整 logits 向量方向的差异。

- 越接近 `0` 越好。
- 可用于判断整体输出结构是否发生方向性变化。

### Top-1 Agreement

两个模型首选 token 相同的样本比例。

- 越高越好。
- `1.0` 表示所有样本首选 token 都相同。

### Top-10 Overlap

两个模型 Top-10 token 集合的平均重合比例。

- 越高越好。
- 可以发现 Top-1 未变化但候选分布已经变化的情况。

### Generated Exact Match

完整 greedy 生成 token 序列完全一致的样本比例。

- 越高越好。
- 对早期 token 的微小变化非常敏感。
- 适合作为直观结果，不应代替首 token 分布指标。

## 十三、统计字段解释

每项指标均保存：

- `mean`：所有样本平均值。
- `std`：样本间标准差。
- `median`：中位数。
- `p95`：95% 样本不超过的数值。
- `min`：最小值。
- `max`：最大值。

对于 JS、KL、RMSE 和余弦距离：

- `mean` 用于报告整体平均扰动。
- `median` 用于判断典型样本扰动。
- `p95` 用于判断大部分样本中的最坏扰动水平。
- `max` 用于定位极端异常样本。

对于 Top-1、Top-10 和 Exact Match：

- 应主要查看 `mean`。
- 其 `mean` 就是总体一致率或平均重合率。

## 十四、结果文件解读

### 1. 自然扰动报告

首先查看：

```text
results/reports/base_natural_disruption.json
```

关注：

```json
{
  "conclusion": {
    "mean_natural_js": 0.0
  }
}
```

该值是原模型自然扰动基线。后续方法都应与它比较。

### 2. 单个方法报告

例如：

```text
results/reports/vision_lora_rank8.json
```

文件开头直接给出：

```json
{
  "conclusion": {
    "mean_js": 0.0,
    "mean_natural_js": 0.0,
    "mean_excess_js": 0.0,
    "natural_disruption_ratio": 1.0
  }
}
```

重点解读：

- `mean_js`：该方法相对两次 Base 的平均扰动。
- `mean_natural_js`：Base 自身自然扰动。
- `mean_excess_js`：该方法超出自然扰动的部分。
- `natural_disruption_ratio`：该方法是自然扰动的多少倍。

### 3. 总汇总报告

查看：

```text
results/reports/all_experiments.json
```

其中 `ranking` 按平均 JS 从小到大排列：

```text
排名越靠前，表示对原模型造成的扰动越小。
```

论文中不应只比较七个基线之间谁最小，还应将每种方法与 `mean_natural_js` 比较。

## 十五、论文结果判断

### 支持“接近零污染”的理想结果

如果我们的方法满足：

```text
Ours mean JS ≈ Base natural mean JS
Ours excess JS ≈ 0
Ours disruption ratio ≈ 1
Ours Top-1 agreement ≈ Base-Base Top-1 agreement
```

同时七个视觉侧基线表现为：

```text
Baseline mean JS > Base natural mean JS
Baseline excess JS > 0
Baseline disruption ratio > 1
```

则可以说明：

> 我们的方法引入的输出分布扰动接近原模型重复运行的自然数值扰动，而传统视觉侧参数高效微调方法产生了可测量的额外扰动。

### 自然扰动完全为零

如果两次 Base 的所有指标均完全相同，自然扰动可能为零。这并不代表实验无效，而是说明当前硬件和推理配置下原模型推理是确定性的。

此时：

- 不解读扰动倍数，因为分母为零。
- 直接比较各方法的绝对 JS、KL 和 RMSE。
- 如果我们的方法也为零，可以表述为“在当前数值精度下未检测到额外扰动”。
- 不应仅凭打印出来的有限小数位声称数学意义上的绝对零。

### 均值较小但 P95 较大

这说明方法对多数样本影响较小，但会严重扰动少数样本。应结合 `per_sample` 定位高扰动样本，不宜只报告平均值。

### Top-1 不变但 JS 增大

这说明最终首选 token 没变，但候选 token 概率分布已经受到污染。该结果仍然代表模型内部输出行为发生变化，JS 能比 Top-1 更早检测到这种变化。

### 完整生成不同但首 token 扰动很小

自回归生成会累积微小差异。后续 token 的一个细微变化可能导致完整答案不同，因此应以首 token 分布指标作为主要数值证据，以完整生成一致率作为辅助证据。

## 十六、启用我们的方法

训练完成后：

1. 在 `eval_ours.py` 中导入你们现有的模型加载逻辑。
2. 将 `CHECKPOINT` 设置为最终 checkpoint。
3. 返回 `model` 和 `processor`。
4. 在 `run_all.py` 中取消以下代码的注释：

```python
# ("eval_ours.py", "ours"),
```

改为：

```python
("eval_ours.py", "ours"),
```

如果其他实验已经完成，可以运行：

```bash
python eval_consistency/run_all.py --skip-existing
```

程序会跳过已有的两次 Base 和七个基线，只运行我们的方法，然后重新生成全部统计报告。

## 十七、实验数据保存注意事项

- 不要手动修改正式实验 JSON。
- 保存完整的 `results` 目录，不要只保存汇总报告。
- 论文提交时应同时保留原始 `raw_results.json` 和 `reports`。
- 不同正式实验应使用不同的 `output-root`，避免混合不同样本数或不同环境的结果。
- 所有模型必须使用相同数据集、样本顺序、图片、生成参数和数值精度。
- `analyze.py` 会检查样本 ID、输入 token 哈希和词表形状，不一致时会停止分析，避免错误配对。
- 完整词表 logits 数据量较大，运行前应确认磁盘空间充足。
