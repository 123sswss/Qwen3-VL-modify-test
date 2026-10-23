# QDPT

视觉问答中的同一图像可以对应不同问答目标，条件提示需要随问题调整其视觉信息。基于全局汇聚的图像条件提示通过整图摘要生成适配上下文，其摘要独立于当前问题。本文提出问题驱动的提示微调方法 QDPT，在冻结生成式多模态骨干的条件下，利用问题引导的多层视觉选择汇聚原生视觉表征，并将所得摘要转换为静态前置提示的共享偏移。

QDPT 使用约 1.865 M 可训练参数，在 PathVQA 三个训练 seed 上取得 59.03% 的平均 Test 准确率，样本标准差为 0.24 个百分点，较静态提示与 CoCoOp-style 分别提高 2.95 和 2.08 个百分点。seed44 的 Test 准确率较近似同预算 LoRA-r2 提高 1.50 个百分点。同一结构也在 SLAKE 与 RSVQA-LR 上改善了未微调模型的表现。

## 使用

本目录提供 PathVQA 的训练与评分。准备好 Qwen3-VL 模型和 PathVQA 数据，数据目录中放置 `train-*.parquet`、`validation-*.parquet` 和 `test-*.parquet`。

在本目录安装依赖：

```bash
pip install -r requirements.txt
```

训练五轮，随后评估 Validation：

```bash
python main.py run --model /path/to/model --data /path/to/pathvqa
```

将 `run` 换成 `train` 可单独训练。训练配置已固定，仅保存第五轮权重，默认输出到 `outputs/v1_seed44/`。通过 `--output` 指定其他输出目录。

加载权重，在 Test 上推理评分：

```bash
python main.py eval --model /path/to/model --data /path/to/pathvqa \
  --checkpoint outputs/v1_seed44/checkpoints/epoch_5 --split test
```

将 `test` 换成 `validation` 可评估验证集。预测和评分分别保存在输出目录的 `eval_test/epoch_5/` 或 `eval_validation/epoch_5/`。
