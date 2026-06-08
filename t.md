# train/train_stages.py：阶段配置与 loss 配置（精简内部底稿）

> 范围限制：这里只整理 `train/train_stages.py` 中与**阶段配置**、**loss 配置**直接相关的内容；不展开监控、日志、诊断、回调等杂项。

---

## 1. 阶段配置总览

这个文件把训练切成 **4 个 stage**，本质上是一个“先轻量训练门控/判别，再训练完整 MMRL 分支”的分阶段策略。

### Stage 1
- 入口：`run_stage12_light(stage_id=1, ...)`
- 数据视图：`("expert-mm", "expert-text", "general-mm", "general-text")`
- `ce_enabled=False`
- 训练目标：主要训练**视觉/文本 pooling + Task_classifier + adapter_router**，先把样本级判别能力学出来
- 可训练模块：
  - `v.hidden_state_pooling`
  - `v.embedding_pooling`
  - `v.Task_classifier`
  - `v.adapter_router`
- `visionGating` 此时**不训练**
- 优化循环：手写 light loop，不走 HuggingFace Trainer

### Stage 2
- 入口：`run_stage12_light(stage_id=2, ...)`
- 数据视图同 Stage 1
- `ce_enabled=False`
- 相比 Stage 1，多训练一个模块：
  - `v.visionGating`
- 目标：在已有 `alpha_logits` 分类能力基础上，让 gate 输出也向标签靠拢

### Stage 3
- 入口：`run_stage34_full(stage_id=3, ...)`
- 数据视图：只保留 `("expert-mm",)`
- `ce_enabled=True`
- 使用完整 `Trainer`
- 可训练模块：
  - `model.model.MMRL`
  - `v.hidden_state_pooling`
  - `v.embedding_pooling`
  - `v.adapter_router`
  - `v.residual_adapters`
  - `v.visionGating`
  - `model.model.text_adapter_router`
  - `model.model.text_residual_adapters`
- 目标：开始训练完整的 MMRL 主体、视觉 residual adapter、文本 adapter 路由与注入分支

### Stage 4
- 入口：`run_stage34_full(stage_id=4, ...)`
- 数据视图同 Stage 3，仅 `("expert-mm",)`
- 可训练模块与 Stage 3 基本一致
- 主要差别不在“训哪些模块”，而在于**允许额外启用更复杂的 loss/约束配置**（虽然代码默认很多仍可为 0 或关闭）


---

## 2. 阶段切换的核心伪代码

```text
如果 stage == 1:
    训练 pooling + Task_classifier + adapter_router
    目标 = alpha 分类

如果 stage == 2:
    在 stage1 基础上额外训练 visionGating
    目标 = alpha 分类 + gate 拟合

如果 stage == 3:
    训练 MMRL 主体 + 视觉 adapter + 文本 adapter + gate/router
    目标 = 生成 CE + 若干 adapter/text 正则

如果 stage == 4:
    与 stage3 类似
    但允许启用更强的 expert/collapse 类约束
```

---

## 3. loss 配置总览

`Qwen3VLMMRLForStages.forward(...)` 里定义了训练时的 loss 组装逻辑。可以分成三层：

1. **主任务 loss**
2. **alpha / gate 相关 loss**
3. **visual/text adapter 正则 loss**

### 3.1 Stage 1 / 2 的 loss

这两个阶段不走完整生成损失，而是轻量监督：

#### `cls_loss`
- 来源：`alpha_logits = v.Task_classifier(vision_pooled, text_pooled)`
- Stage 1：
  - 使用 `focal_bce_with_logits`
- Stage 2：
  - 使用普通 `binary_cross_entropy_with_logits`

含义：监督 `Task_classifier` 输出的 `alpha_logits` 去拟合 `alpha_labels`。

#### `gate_loss`（仅 Stage 2 可启用）
- 条件：`stage_id == 2 and enable_gate_loss_stage2`
- 公式实现：
  - `g = v.visionGating(alpha_logits, temperature_override)`
  - `gate_loss = mse_loss(g, alpha_labels)`

含义：不仅要求分类 logit 对，还要求经过 gate 后的连续门值也接近标签。

#### Stage 1/2 总 loss

```text
loss = cls_loss + gate_loss
```

其中 Stage 1 通常 `gate_loss = 0`。

---

### 3.2 Stage 3 / 4 的主损失

#### `ce_loss`
- 来源：`super().forward(...)` 返回的生成 loss
- 若有 `labels`，则直接用模型输出的 `outputs.loss`
- 权重：`ce_loss_weight`，当前代码里 Stage 3/4 都设为 `1.0`

这是完整多模态生成训练的主损失。

#### `alpha_guide_loss`
- 只有当 `enable_alpha_guide_loss=True` 时才启用
- 形式：`binary_cross_entropy_with_logits(alpha_logits, expanded_labels)`
- 再乘 `alpha_loss_weight`

但从本文件当前 stage3/4 配置看：
- `enable_alpha_guide_loss = False`
- `alpha_loss_weight = 0.0`

所以它目前更像保留接口。

---

### 3.3 Visual adapter 正则项

这些 loss 直接从 `self.model.visual` 读取：

#### `adapter_usage_balance_loss`
- 作用：抑制视觉 adapter 路由塌缩到单一专家
- 最终乘：`adapter_usage_balance_loss_weight`

#### `adapter_sample_entropy_loss`
- 作用：约束单样本 adapter 路由熵，不让路由过尖
- 最终乘：`adapter_sample_entropy_loss_weight`

#### `adapter_common_mode_loss`
- 作用：抑制不同样本的视觉增强残差过于共模
- 最终乘：`adapter_common_mode_loss_weight`

---

### 3.4 Text 注入分支正则项

这些 loss 直接从 `self.model` 读取：

#### `text_common_mode_loss`
- 作用：抑制不同样本的文本注入向量过于相似
- 最终乘：`text_common_mode_loss_weight`

#### `text_adapter_balance_loss`
- 作用：平衡文本 residual adapters 的使用率
- 最终乘：`text_adapter_balance_loss_weight`

#### `text_adapter_sample_entropy_loss`
- 作用：控制文本 adapter 路由熵
- 最终乘：`text_adapter_sample_entropy_loss_weight`

---

## 4. 最终并入总 loss 的项

当前代码真正写进 `outputs.loss` 的是：

```text
total_loss =
    ce_loss_weight * ce_loss
    + alpha_guide_loss
    + adapter_usage_balance_loss * adapter_usage_balance_loss_weight
    + adapter_sample_entropy_loss * adapter_sample_entropy_loss_weight
    + adapter_common_mode_loss * adapter_common_mode_loss_weight
    + text_common_mode_loss * text_common_mode_loss_weight
    + text_adapter_balance_loss * text_adapter_balance_loss_weight
    + text_adapter_sample_entropy_loss * text_adapter_sample_entropy_loss_weight
```

也就是说，**Stage 3/4 的核心是“生成 CE + visual/text adapter 正则”**。

---

## 5. 没有真正并入总 loss、但保留接口的项

代码中还计算/维护了下面这些量：

- `expert_floor_loss`
- `anti_collapse_loss`
- `k_general_loss`
- `k_expert_loss`（这里只是把 `expert_floor_loss` 记到指标里）

但在当前 `outputs.loss = ...` 处，**它们并没有被真正加进去**。

因此从实现上讲：

> 当前版本真正生效的训练目标，主要还是 `CE + adapter/text regularization`；expert floor / anti-collapse 目前更多是预留机制，而不是已正式接入总优化目标的主路径。

---

## 6. 配置项与生效位置的最短总结

### 阶段相关
- `epochs[stage_id]`：每阶段 epoch 数
- `learning_rate[stage_id]`：每阶段学习率
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `enable_gate_loss_stage2`

### Visual 正则相关
- `adapter_usage_balance_loss_weight`
- `adapter_sample_entropy_loss_weight`
- `adapter_common_mode_loss_weight`

### Text 正则相关
- `text_common_mode_loss_weight`
- `text_adapter_balance_loss_weight`
- `text_adapter_sample_entropy_loss_weight`

### 保留但当前常默认关闭/不并入总 loss 的项
- `enable_alpha_guide_loss_s4`
- `alpha_loss_weight_s4`
- `enable_expert_floor_loss_s4`
- `expert_floor_loss_weight`
- `enable_collapse_loss_s4`
- `anti_collapse_weight`

---

## 7. 一段最短结论

这个文件的训练策略可以浓缩成一句话：

> Stage 1/2 先用轻量二分类目标把 `alpha / gate` 学稳；Stage 3/4 再切到完整生成训练，并把视觉 adapter 与文本注入 adapter 的若干均衡性/熵/共模正则一起并入总 loss。当前代码里，真正主用的 loss 是 `CE + adapter/text regularization`，而 expert-floor、collapse、alpha-guide 更多属于可选保留接口。
