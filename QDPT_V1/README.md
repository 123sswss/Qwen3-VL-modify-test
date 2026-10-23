# QDPT V1 / PathVQA

独立的五轮 V1，读写头学习率3e-4。依赖冻结的 Qwen3-VL-4B，视觉/语言宽度为1024/2560，使用 Transformers 5.0.0 的原生视觉、DeepStack和生成缓存。

```text
main.py               训练、推理和评分入口
model/qdpt.py         Visual18、三层选择和条件P20
pathvqa/data.py       Parquet读取、chat与答案监督
pathvqa/evaluate.py   贪心推理、官方归一化精确匹配和图像簇CI
```

模型固定为20个LLM Prompt、索引17的S8＋Av10、索引5/11/17的三张选择地图。三张地图各自读取共同merger Value，经LN和2560→64投影，问题条件层权重调制后拼接，192→2560输出共享偏移。可训练参数目标为1,864,963。

训练固定model seed44、data seed42、5 epochs、batch2/累积16、AdamW(betas=.9/.999, eps=1e-8, weight_decay=0)、clip1、bf16、workers2、3% warmup和五轮线性衰减。P20学习率.3、S8为3e-5、Av10和其余条件分支为1e-4，value_blocks与prefix_output为3e-4。Trainer显式设置`model_accepts_loss_kwargs=False`。仅保存epoch5，固定epoch5评分。

数据目录直接放官方`train-*.parquet`、`validation-*.parquet`、`test-*.parquet`。训练使用完整原始问句和答案，长度预算2048，保留完整序列；监督从assistant标记后的换行开始，包含答案和结束符。问题条件单独分词；推理在原问题后附加`Answer with only a short answer.`，使用贪心生成、最多32个新token和raw评分。图像簇CI固定2000次、seed42。

## 使用

```bash
cd QDPT_V1
pip install -r requirements.txt

# 从零训练，随后固定epoch5完整Validation
python main.py run --model /root/autodl-tmp/model --data /root/autodl-tmp/dataset/pathVQA

# 单独训练
python main.py train

# 单独推理评分；也可指向正式V1已有checkpoint目录
python main.py eval --checkpoint outputs/v1_seed44/checkpoints/epoch_5 --split validation
python main.py eval --checkpoint outputs/v1_seed44/checkpoints/epoch_5 --split test
```

输出为adapter权重、Trainer日志、train_report，以及划分独立的预测/逐题判分/summary。更换运行时指定新的`--output`目录。训练超参数直接写在源码中；CLI只选择动作、路径和评估划分。

此目录的Python模块只依赖本目录代码和requirements。旧项目的诊断、消融、调度、计时及预检没有迁入。本次按要求未执行测试、训练或推理；精简版与历史版本的数值一致性尚待实际运行核对。历史论文分数对应原正式入口。
