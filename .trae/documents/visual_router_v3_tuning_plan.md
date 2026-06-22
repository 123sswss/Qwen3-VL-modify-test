# visual_router_v3 调参分析计划

## Summary

目标：基于用户提供的 6 组实验结果、`train/train.py` 中的 `visual_router_v1` / `visual_router_v2` 参数，以及当前训练代码逻辑，只做只读分析，解释：

1. 为什么同为 `visual_router_v2`，`visual_router_v2_2` 可以达到 65.3 分，而其他 `v2` 大多在 59-60 分；
2. 为什么 `visual_router_v1` 版本结果波动更大且整体偏低；
3. 给出一个 `visual_router_v3` 的推荐参数块，目标优先追求“更稳”和“更可复现”，而不是赌单次高点。

## Current State Analysis

### 已确认的代码事实

- `visual_router_v1` 并没有在 `EXPERIMENTS` 中单独定义，它实际对应 `BASE_VISUAL_ROUTER_MODEL` 默认参数：
  - `adapter_usage_balance_loss_weight = 0.005`
  - `adapter_sample_entropy_loss_weight = 0.02`
  - `adapter_common_mode_loss_weight = 0.03`
  - `adapter_sample_entropy_target = 0.55`
  - `adapter_common_mode_target = 0.85`
- `visual_router_v2` 在 `train/train.py` 中显式覆写为：
  - `adapter_usage_balance_loss_weight = 0.003`
  - `adapter_sample_entropy_loss_weight = 0.05`
  - `adapter_common_mode_loss_weight = 0.07`
  - `adapter_sample_entropy_target = 0.55`
  - `adapter_common_mode_target = 0.80`
- Stage3/4 训练只使用 `expert-mm` 视图，见 `train/train_stages.py` 中 `stage34_views = ("expert-mm",)`。
- Stage4 的最终损失为：
  - `CE loss`
  - `adapter_usage_balance_loss * weight`
  - `adapter_sample_entropy_loss * weight`
  - `adapter_common_mode_loss * weight`
- `adapter_sample_entropy_loss` 的作用是：当 router 的样本级熵低于 target 时施加惩罚，推动每个样本不要太早“一把梭”到单 adapter。
- `adapter_common_mode_loss` 的作用是：当 pooled delta 的公共模态比例高于 target 时施加惩罚，抑制“所有图都往同一个残差方向塌缩”。
- 训练存在明显随机性来源：
  - `train/train.py` 里虽然记录了 `seed = 42`，但仓库内未检索到真正的全局 `torch.manual_seed` / `transformers.set_seed` 设置；
  - `deterministic_sampling` 默认是 `False`；
  - `DataLoader(..., shuffle=True)`；
  - `StageScheduleCallback.on_epoch_begin()` 每个 epoch 都会 `dataset.resample_data()`。

### 从用户提供结果得到的现象

- `v1` 组：
  - `visual_router_v1`：62.79
  - `visual_router_v1_1`：54.82
  - `visual_router_v1_2`：60.48
- `v2` 组：
  - `visual_router_v2`：59.43
  - `visual_router_v2_1`：59.22
  - `visual_router_v2_2`：65.30

### 从指标读到的主要模式

- `v2_2` 的 `ce_loss_mean` 最低，说明它不仅评测分高，训练期生成目标也学得更好，不像只是“指标好看但没转成任务性能”。
- `v2_2` 的 `delta_pool_common_mode_ratio_mean` 约 `0.9165 -> 0.9227`，是 6 组里偏低的；说明它的视觉增量没有那么强烈地塌到“公共方向”。
- `v2_2` 的 `delta_pool_specificity_ratio_mean` 约 `0.3522 -> 0.3349`，是 6 组里最高的；说明它的 residual delta 更保留图像间差异性。
- `v2_2` 的 `adapter_route_entropy_norm_latest` 在第一个窗口一度掉到 `0.3889`，但窗口均值仍在 `0.72-0.74` 左右，说明它不是全程高度平均路由，而是训练后期出现更果断的 adapter 分配。
- `v1` 组整体 `delta_pool_common_mode_ratio` 更高、`specificity_ratio` 更低，且 `adapter_route_entropy_norm` 常偏高，说明更接近“大家都走得差不多、区别不够开”的状态。

## Proposed Changes

### 分析输出结构

最终给用户的回答按以下结构组织：

1. 解释 `v2_2` 高分的直接原因
   - 不是单看 `route_entropy` 高低；
   - 核心是 `common_mode_ratio` 更低、`specificity_ratio` 更高、`ce_loss_mean` 更低；
   - 即 router 没有塌，但也没有被过强正则强制成“始终平均分流”。
2. 解释 `v2` 其余版本为什么只有 59-60
   - `v2` 的正则方向是对的，但存在显著训练随机性；
   - 同参下不同 run 落在“较分化且有效”与“分化不够/后期塌缩”两种 basin；
   - `v2_2` 更像撞到了优质 basin，而非参数完全不同。
3. 解释 `v1` 为什么偏低且飘
   - `sample_entropy_loss_weight` 偏小、`common_mode_loss_weight` 偏小、`common_mode_target` 偏宽松；
   - 结果是 router 更容易保持高熵平均混合，同时视觉残差更容易被公共模态主导；
   - 这种设定学习更“稳态保守”，但不容易把 adapter 专长拉开，因此上限和稳定性都差。
4. 给出 `v3` 参数
   - 保留 `v2` 抑制 common mode 的思路；
   - 比 `v2` 略微降低 sample entropy 正则，避免过强“强迫平均分流”；
   - balance weight 维持较低但不为零；
   - target 保持中间值，避免既过松又过硬。

### 推荐的 v3 参数方向

建议给出如下思路的 `v3` 参数：

- `adapter_usage_balance_loss_weight`: 维持低权重，小幅约束全局 usage，不要太强；
- `adapter_sample_entropy_loss_weight`: 比 `v2` 低一点，比 `v1` 高一点；
- `adapter_common_mode_loss_weight`: 保持比 `v1` 高，但略低于 `v2`；
- `adapter_sample_entropy_target`: 维持 `0.55`；
- `adapter_common_mode_target`: 取 `0.82` 左右，介于 `v1=0.85` 和 `v2=0.80` 中间。

计划中的推荐具体值：

```python
"adapter_usage_balance_loss_weight": 0.0035,
"adapter_sample_entropy_loss_weight": 0.035,
"adapter_common_mode_loss_weight": 0.055,
"adapter_sample_entropy_target": 0.55,
"adapter_common_mode_target": 0.82,
```

推荐理由：

- 相比 `v1`：更强地压制 common-mode collapse，也更鼓励 router 不要过早塌到单路；
- 相比 `v2`：不过度推动“高熵平均路由”，给 adapter 专门化留出一点空间，更接近 `v2_2` 表现出来的状态；
- 目标是追求“均值更稳”，而不是押注单次最好结果。

## Assumptions & Decisions

- 假设用户当前需要的是“参数解释 + 下一版建议”，而不是要求做统计显著性验证或代码改动。
- 决定将 `visual_router_v1` 解释为 `BASE_VISUAL_ROUTER_MODEL`，因为当前代码中不存在单独的 `visual_router_v1` 实验定义。
- 决定把“同参不同分”的主因解释为：参数作用方向 + 训练随机性共同决定，而不是简单归因为日志噪声。
- 决定给出一版保守型 `v3`，优先提高稳定性，不直接继续把 `common_mode_loss_weight` 往上推。

## Verification Steps

本轮只读计划不执行实验，但若后续进入执行阶段，建议验证步骤为：

1. 使用 `visual_router_v3` 跑至少 3 次独立重复实验；
2. 对比 `test.log` 最终分数均值与标准差；
3. 对比 `stage4/metrics/mmrl_diagnostics.jsonl` 中：
   - `ce_loss_mean`
   - `delta_pool_common_mode_ratio_mean`
   - `delta_pool_specificity_ratio_mean`
   - `adapter_route_entropy_norm_mean`
4. 若 `v3` 仍出现“高熵但低 specificty”，则继续下调 `adapter_sample_entropy_loss_weight`；
5. 若 `v3` 出现“路由过早塌缩到单 adapter”，则小幅回调 `adapter_sample_entropy_loss_weight` 或 `adapter_usage_balance_loss_weight`。
