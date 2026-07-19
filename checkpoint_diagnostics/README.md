# Checkpoint Diagnostics

本目录只包含离线 checkpoint 诊断代码与输出，不修改训练流程。

## 运行高分 checkpoint

在项目根目录执行：

```bash
bash checkpoint_diagnostics/run_high_checkpoint.sh
```

脚本会执行视觉 prefill，但不会重新生成答案。答对/答错标签来自 checkpoint
旁边已有的 `eval/test.log`，因此明显快于重新跑完整生成测评。

可以先限制每个数据源的题目数做冒烟检查：

```bash
bash checkpoint_diagnostics/run_high_checkpoint.sh --limit-per-source 5
```

输出位于 `checkpoint_diagnostics/outputs/<实验名_时间>/`：

- `console.log`：完整运行日志。
- `samples.jsonl`：逐题 Gate、router、adapter、delta 指标。
- `summary.json`：按数据源以及正确/错误分组的统计。
- `vectors.npz`：逐题 MMRL/adapter/routed-delta 方向向量。

## 比较高低分 checkpoint

分别诊断两个 checkpoint 后执行：

```bash
python checkpoint_diagnostics/compare_runs.py \
  checkpoint_diagnostics/outputs/<高分目录> \
  checkpoint_diagnostics/outputs/<低分目录>
```

比较脚本会按题目对齐，自动搜索四个专家的最佳排列，并报告 router 一致率、
adapter 方向相似度和最终 delta 方向相似度。

## 高低分混合消融

完成方向比较并得到专家排列后，可以只替换 router 或 adapters 做两次完整评测。
当前高低分 checkpoint 的排列已经写入脚本，直接在项目根目录执行：

```bash
bash checkpoint_diagnostics/run_hybrid_ablation.sh
```

脚本不会复制或保存完整 checkpoint。两次评测都以高分 checkpoint 为底座：

- `high_adapters_low_router`：只安装对齐后的低分 router。
- `low_adapters_high_router`：只安装对齐后的低分 adapters。

结果位于 `checkpoint_diagnostics/outputs/hybrid_*/`，每轮包含 `test.log` 和
`summary.json`；后者同时记录总分与两个数据源的分数。

若 checkpoint 路径发生变化，可以用环境变量覆盖：

```bash
HIGH_CHECKPOINT=/path/to/high/final \
LOW_CHECKPOINT=/path/to/low/final \
bash checkpoint_diagnostics/run_hybrid_ablation.sh
```

## 固定高分专家 3

强制高分 checkpoint 的所有样本只使用专家 3：

```bash
bash checkpoint_diagnostics/run_fixed_expert3.sh
```

该评测只在内存中把 router 输出覆盖为 `[0, 0, 0, 1]`，不会修改或保存模型
权重。结果保存在 `checkpoint_diagnostics/outputs/fixed_expert_3_*/`。

## 扫描低分 checkpoint 的四个专家

按高分 checkpoint 的使用率从高到低扫描专家 `3 → 2 → 1 → 0`：

```bash
bash checkpoint_diagnostics/run_low_expert_sweep.sh
```

每个专家一旦出现无法提取 A-D 答案的输出便立即结束当前评测，并继续下一个
专家。结果位于 `checkpoint_diagnostics/outputs/low_expert_sweep_*/expert_*/summary.json`。
