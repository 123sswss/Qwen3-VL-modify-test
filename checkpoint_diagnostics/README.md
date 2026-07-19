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
