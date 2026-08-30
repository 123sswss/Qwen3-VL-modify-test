# MMRL Experiment Ledger

Last updated: 2026-08-30

This file is the persistent source of truth for completed experiments. Results are recorded from official SLAKE evaluation output or diagnostics supplied during development. Unless noted otherwise, SLAKE evaluation contains 2,094 test questions, uses all languages, and reports percentages.

## Recording Rules

- Record every completed experiment, including failed runs and negative results.
- Record the exact experiment name, seed, architecture/loss change, Overall score, and available breakdowns.
- Do not overwrite historical rows. Add corrections with an explanation.
- Multi-seed summaries use the sample standard deviation.
- A result without an exact log or official score must be marked approximate or unresolved.
- Update this file in the same change that adds or completes an experiment, then commit and push it.

## Current Snapshot

- Current PathVQA matched comparison: frozen Base **34.77**, minimal last8-layer MMRL + Relation0.05 seed44 **49.41**, and last8-layer Visual Attention LoRA-r128 seed44 **53.85**. The scope-matched LoRA lead is **4.44** points; all24-layer LoRA remains a broader upper bound at **55.45**.
- Strongest overall result: static Prompt Tuning, seed44, **74.40** with only 51,200 trainable parameters.
- Strongest MMRL single run: 128-slot Attention Pooling + shared CA + same-initialized layer MLPs + Relation 0.05, seed44, **73.93**.
- Final Mean Pooling MMRL across seeds44-47: **72.98 +/- 0.49**.
- 128-slot Attention Pooling MMRL across seeds44-47: **72.94 +/- 0.66**.
- Mean Pooling and 128-slot Attention Pooling are statistically tied on SLAKE; Mean Pooling is simpler and less variable.
- The current naive parameter-matched Concat-MLP replacement scores **69.25**, but its joint normalization is confounded by Query/Memory scale mismatch. A separately normalized Concat-MLP is pending before claiming CA is irreplaceable.
- A second dataset is required before making general claims about Pooling, Relation, or Prompt Tuning.

## Current MMRL Definition

The current clean MMRL candidate is:

```text
Shared low-dimensional latent table S
  -> 8 same-initialized but independently optimized layer MLPs
  -> layer-specific base prompts Q

Visual Mean Pooling + Text Mean Pooling
  -> K/V memory

Shared residual Cross-Attention(Q, K, V)
  -> 40 sample-conditioned prompts per selected ViT layer
  -> replacement/refresh injection into ViT layers 17-24
```

Default training configuration: seed varies, data seed42, 3 Stage-3 epochs, effective batch32, Relation weight0.05, open/full CE training gate.

## Formal Multi-Seed Results

### Pooling Comparison

| Method | Seed44 | Seed45 | Seed46 | Seed47 | Mean +/- SD |
|---|---:|---:|---:|---:|---:|
| 128-slot Attention Pooling + Relation | 73.93 | 72.64 | 72.54 | 72.64 | 72.94 +/- 0.66 |
| Mean Pooling + Relation | 72.97 | 73.26 | 73.40 | 72.30 | 72.98 +/- 0.49 |

Conclusion: the mean difference is only +0.04 for Mean Pooling. The learned 128-slot pooling does not provide a stable SLAKE gain.

### Full-CE Relation Weighting

| Relation weighting | Seed44 | Seed45 | Seed46 | Seed47 | Mean +/- SD |
|---|---:|---:|---:|---:|---:|
| Uniform | 73.93 | 72.64 | 72.54 | 72.64 | 72.94 +/- 0.66 |
| Alpha-weighted | - | 72.73 | 72.11 | 73.02 | 72.62 +/- 0.46 over 3 seeds |

Conclusion: alpha-weighting did not improve the multi-seed mean.

### Gate Strategy

| Gate strategy | Seed44 | Seed45 | Seed46 | Seed47 | Mean +/- SD |
|---|---:|---:|---:|---:|---:|
| Hard Concrete | 72.59 | 71.30 | 73.50 | 73.45 | 72.71 +/- 1.03 |
| Alpha probability | 73.59 | 72.92 | 73.02 | 72.16 | 72.92 +/- 0.59 |

Conclusion: alpha probability reduced variance but did not materially improve the mean.

## Cross-Attention Delta Scale

### Seed44

| Scale | Overall | Delta | KVQA | VQA | CLOSED | OPEN | EN | ZH |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 45.32 | -28.08 | 34.83 | 46.85 | 54.90 | 38.95 | 53.44 | 36.98 |
| 0.5 | 72.68 | -0.72 | 55.81 | 75.15 | 80.74 | 67.33 | 75.21 | 70.09 |
| 1.0 | 73.40 | +0.00 | 56.55 | 75.86 | 80.26 | 68.84 | 76.06 | 70.67 |

### Seed45

| Scale | Overall | Delta | KVQA | VQA | CLOSED | OPEN | EN | ZH |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 45.46 | -26.65 | 35.58 | 46.91 | 55.14 | 39.03 | 53.44 | 37.27 |
| 0.5 | 71.20 | -0.91 | 53.56 | 73.78 | 81.22 | 64.55 | 75.78 | 66.51 |
| 1.0 | 72.11 | +0.00 | 55.43 | 74.55 | 82.66 | 65.10 | 76.44 | 67.67 |

Conclusion: the learned CA delta is essential; removing it loses about 27 points. Half scale retains most but not all performance.

## Query-Generator Swap

| Composition | Overall | Recipient delta | KVQA | VQA | CLOSED | OPEN | EN | ZH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q44 + Rest44 | 73.40 | +0.00 | 56.55 | 75.86 | 80.26 | 68.84 | 76.06 | 70.67 |
| Q45 + Rest45 | 72.11 | +0.00 | 55.43 | 74.55 | 82.66 | 65.10 | 76.44 | 67.67 |
| Q44 + Rest45 | 53.39 | -18.72 | 41.20 | 55.17 | 58.37 | 50.08 | 62.58 | 43.95 |
| Q45 + Rest44 | 53.77 | -19.63 | 41.57 | 55.56 | 56.82 | 51.75 | 64.75 | 42.50 |

Conclusion: the query generator and recipient parameters co-adapt strongly and cannot be swapped independently.

## Architecture Experiments

| Experiment | Seed | Overall | Status / conclusion |
|---|---:|---:|---|
| `slake_mmrl_layer_mlp_full_ca_relation0050_repro3_seed44_20260817_1` | 44 | 73.40 | Historical Layer-MLP full-CA baseline |
| Layer-MLP full-CA counterpart | 45 | 72.11 | Historical seed45 counterpart |
| `slake_mmrl_layer_mlp_same_init_relation0050_seed44_20260819` | 44 | 73.93 | Same-origin MLP initialization; strongest MMRL single run |
| `slake_mmrl_layer_mlp_deepstack_relation0050_seed44_20260819` | 44 | 68.53 | DeepStack residual failed; delta/original ratio reached about0.22 late |
| `slake_mmrl_layer_mlp_cross_relation0010_relation0050_seed44_20260819` | 44 | 68.29 | Mixed old Relation + Cross-Relation 0.01 failed |
| `slake_mmrl_layer_mlp_current_control_relation0050_seed44_20260819` | 44 | 71.87 | Code-state control; exposed training trajectory instability |
| `slake_mmrl_layer_mlp_same_init_cross_relation_only0050_seed44_20260819` | 44 | 72.68 | Cross-Relation only; below 73.93 reference |
| `slake_mmrl_layer_mlp_same_init_reverse_assignment_relation0050_seed44_20260819` | 44 | 70.96 | Reverse assignment attention failed |
| `slake_mmrl_layer_mlp_same_init_dynamic_query_static_kv_relation0050_seed44_20260820` | 44 | 61.70 | Dynamic serial query; initialization scale was invalid |
| `slake_mmrl_layer_mlp_same_init_dynamic_query_zerogate_static_kv_relation0050_seed44_20260820` | 44 | 67.38 | Zero-gated repair improved but remained poor |
| `slake_mmrl_layer_mlp_same_init_pooled_query_zerogate_static_kv_relation0050_seed44_20260820` | 44 | 65.66 | Pooled query attention was nearly uniform |
| `slake_mmrl_layer_mlp_same_init_dynamic_query_competitive_visual_zerogate_static_kv_relation0050_seed44_20260820` | 44 | 67.19 | Competitive visual assignment remained nearly uniform |
| `slake_mmrl_layer_mlp_same_init_dynamic_query_dual_softmax_visual_zerogate_static_kv_relation0050_seed44_20260820` | 44 | 67.57 | Dual-softmax repair did not recover baseline |

### Detailed Relation / Assignment Results

| Experiment | Overall | CLOSED | OPEN | KVQA | VQA | EN | ZH |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current control | 71.87 | 79.55 | 66.77 | 52.43 | 74.71 | 74.27 | 69.41 |
| Cross-Relation only | 72.68 | 78.59 | 68.76 | 54.68 | 75.31 | 75.31 | 69.99 |
| Reverse assignment | 70.96 | 77.87 | 66.38 | 52.06 | 73.73 | 74.27 | 67.57 |

Reverse-assignment diagnostics: normalized entropy about0.990, indicating diffuse assignment; slot mass became imbalanced despite high entropy.

## Gate and Reproducibility Experiments

| Experiment | Seed | Overall | Notes |
|---|---:|---:|---|
| Same-init original run | 44 | 73.93 | Stage-1 hidden pooling delta0.7061 |
| Same-init repro4 | 44 | 69.29 | Same nominal seed/config; Stage-1 hidden pooling delta0.2922 |
| No gate (`G=1`) | 44 | 71.97 | Gate ablated; changed optimization trajectory |
| Gated rerun | 44 | 73.07 | `G_mean` about0.988; confirms rerun variability |
| Mislaunched ordinary Layer-MLP run | 44 | 73.69 | Output name lacked `alpha_prob`; not valid as alpha experiment |

Conclusion: nominally decoupled gate and Relation components alter shared optimization trajectories. Same seed/config was not perfectly reproducible in older code states, so later formal results use explicit open/full-CE configuration and multi-seed reporting.

## Final SLAKE Ablations

### Attention-Pooling Configuration

| Experiment | Seed | Overall | CLOSED | OPEN | KVQA | VQA | EN | ZH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No Relation | 44 | 72.83 | 79.07 | 68.68 | 54.31 | 75.53 | 75.40 | 70.18 |
| No Relation | 45 | 72.92 | 81.58 | 67.17 | 53.56 | 75.75 | 75.21 | 70.57 |
| Independent MLP initialization | 44 | 72.68 | 80.98 | 67.17 | 54.31 | 75.37 | 75.59 | 69.70 |
| Independent MLP initialization | 45 | 72.83 | 82.06 | 66.69 | 53.56 | 75.64 | 75.02 | 70.57 |
| Static query / no dynamic fusion | 45 | 56.69 | 63.76 | 51.99 | 40.82 | 59.00 | 64.66 | 48.50 |

Two-seed summaries: No Relation **72.88 +/- 0.06**; independent initialization **72.76 +/- 0.11**. The static-query collapse proves that sample-conditioned fusion is necessary.

### Mean-Pooling Configuration

| Experiment | Seed | Overall | CLOSED | OPEN | KVQA | VQA | EN | ZH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean Pooling + Relation | 44 | 72.97 | - | - | - | - | - | - |
| Mean Pooling + Relation | 45 | 73.26 | 82.78 | 66.93 | 52.81 | 76.25 | 75.12 | 71.35 |
| Mean Pooling + Relation | 46 | 73.40 | 82.42 | 67.41 | 53.56 | 76.30 | 75.49 | 71.25 |
| Mean Pooling + Relation | 47 | 72.30 | 79.67 | 67.41 | 55.81 | 74.71 | 75.21 | 69.31 |
| Mean Pooling, no Relation | 45 | 73.30 | 81.46 | 67.89 | 52.43 | 76.35 | 76.34 | 70.18 |
| Mean Pooling, independent initialization | 45 | 72.59 | 81.58 | 66.61 | 55.81 | 75.04 | 75.49 | 69.60 |

Conclusion: Relation has no visible Overall gain on Mean Pooling at seed45 (+0.04 when removed). Same initialization improves Overall by0.67 at seed45, although KVQA moves in the opposite direction.

## Pooling Redesign Attempts

| Experiment | Seed | Overall | Key diagnostics / conclusion |
|---|---:|---:|---|
| Text-guided visual slots8, selection-only | 45 | 73.16 | Attention entropy0.964; slot pair cosine0.985; soft slot collapse |
| Text-guided standard residual CA slots8 + Relation | 45 | 71.11 | Entropy0.898; slot cosine0.476; slots became diverse but accuracy fell |
| Text-guided standard residual CA slots8, no Relation | 45 | 71.49 | Relation removal recovered only0.38 |
| Balanced text/visual fusion slots8 + Relation | 45 | 70.49 | Entropy0.982; slot cosine0.996; equal fusion collapsed slots again |

Conclusion: under current CE supervision, more elaborate learned pooling either remains diffuse or becomes diverse without improving task accuracy.

## Fusion Replacement

| Fusion | Seed | Overall | CLOSED | OPEN | KVQA | VQA | EN | ZH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Shared CA + Mean Pooling | 45 | 73.26 | 82.78 | 66.93 | 52.81 | 76.25 | 75.12 | 71.35 |
| Naive equal-parameter Concat-MLP + Mean Pooling | 45 | 69.25 | 76.79 | 64.23 | 52.81 | 71.65 | 73.52 | 64.86 |

Concat-MLP audit:

- Fusion parameters: 4,202,496, exactly matching CA.
- Total MMRL parameters: 20,506,624.
- CE decreased normally and gradients were active; this was not a dead module.
- Query norm was about0.67 while visual/text memory norms were about32/26.
- The implementation used one joint `LayerNorm(3D)`, whereas CA separately normalizes Q and memory.
- Therefore the 4.01-point gap currently proves naive concatenation is poor, but does not yet isolate attention from normalization. The fair follow-up is `Concat([LN(Q), LN(V), LN(T)]) -> MLP`, which preserves the exact parameter budget.

## Static Prompt Tuning Baseline

Experiment: `slake_prompt_tuning_len20_seed44_20260825_1`

| Seed | Overall | CLOSED | OPEN | KVQA | VQA | EN | ZH |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | 74.40 | 80.62 | 70.27 | 62.55 | 76.14 | 75.40 | 73.38 |

Training audit:

- Prompt length20, LLM hidden size2,560.
- Exactly 51,200 trainable parameters.
- 3 epochs, seed44, data seed42, learning rate0.3.
- Runtime4,320 seconds; train loss3.723.
- About924 optimization steps, matching the MMRL three-epoch schedule.
- No known train/test leakage; treat the result as valid unless a later audit contradicts it.

Interpretation: Prompt Tuning matches MMRL on VQA but gains heavily on KVQA and OPEN questions. It directly conditions the LLM's knowledge and answer space, while MMRL primarily specializes the visual path. This is now the strongest SLAKE baseline and must be included in future comparisons.

## PathVQA Results

### 2026-08-26 - pathvqa_mmrl_layer_mlp_same_init_full_ce_uniform_relation0050_seed44_20260826

- Commit/config: PathVQA MMRL pipeline at commit `bc84ee3`; 128-slot Attention Pooling, shared Cross-Attention, same-initialized layer MLPs, 40 Rep tokens in visual layers17-24, `open_full_ce`, uniform Relation weight0.05.
- Dataset and split: PathVQA official test, all6,719 questions, 858 image clusters; train contains19,654 samples.
- Seed / data seed: 44 / 42.
- Controlled change: first formal transfer of the frozen-LLM MMRL method from SLAKE to PathVQA.
- Overall: **47.30**; image-clustered 95% bootstrap CI **[46.08, 48.49]**.
- Yes/No / Free-form: **82.57 / 11.97**.
- Per question type: how10.79, other0.00, what6.82, when0.00, where46.40, why0.00, yes/no82.57; Free-form question-type macro10.67.
- Stage-3 trainable MMRL parameters:20,998,400; compact checkpoint stores24,288,513 parameters including the trained Stage-1 Gate path.
- Evaluation timing: TTFT mean0.0577s, TPOT0.02095s/token,47.74 decode tokens/s, request mean0.1103s.
- Diagnostics: 1,845 steps, 8 windows, no malformed rows; CE mean0.8668 and first/last windows1.5247/0.7966; Cross-Attention grad norm mean0.7225; `delta_to_org_ratio`0.6176; Cross-Attention delta/base ratio6.781; text/visual pooling grad norms0.4438/0.1285; scaled Relation loss0.001577; `G=1` throughout Stage3 as required by `open_full_ce`.
- Conclusion: training and gradient flow were active, but performance is sharply split between Yes/No and open answers. The result is poor as an absolute score, yet attribution is deferred until the matched Visual LoRA-r128 and frozen Base evaluations finish; if all three have low Free-form accuracy, the bottleneck is likely the frozen LLM/answer-space interface rather than an MMRL-only failure.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/mmrl/pathvqa_mmrl_layer_mlp_same_init_full_ce_uniform_relation0050_seed44_20260826`.

### 2026-08-26 - pathvqa_lora_visual_all_attention_r128_seed44_20260826

- Commit/config: commit `bc84ee3`; LoRA on `qkv/proj` Attention linears in all24 visual layers, rank128, alpha256, dropout0.05; visual MLP and full LLM frozen.
- Dataset and split: PathVQA official train19,654; checkpoint selection on validation6,259; one final evaluation on test6,719 with858 image clusters.
- Seed / data seed: 44 / 42.
- Controlled change: standard visual-only Attention LoRA baseline against frozen-LLM MMRL; no language-layer LoRA.
- Validation Overall by epoch: **48.94, 54.91, 55.70**; earliest-best selection chose epoch3. Validation Yes/No by epoch:81.98,86.69,85.82; Free-form:15.99,23.23,25.65.
- Test Overall: **55.45**; image-clustered 95% bootstrap CI **[53.99, 56.99]**.
- Test Yes/No / Free-form: **86.29 / 24.58**.
- Per question type: how5.76, other5.56, what20.50, when0.00, where58.93, why0.00, yes/no86.29; Free-form question-type macro15.12.
- Trainable parameters:18,874,368 of4,456,690,176 total, ratio0.4235%.
- Training: 3 epochs, LR1e-4, micro-batch1, gradient accumulation32; runtime9,882.8s, loss0.7361.
- Evaluation timing: TTFT mean0.0523s, TPOT0.01991s/token,50.23 decode tokens/s, request mean0.1025s.
- Conclusion: LoRA exceeds MMRL by8.16 Overall,3.72 Yes/No, and12.60 Free-form points, so visual specialization is learnable and MMRL transfers poorly relative to this standard baseline. Absolute performance remains low and epoch3 was still improving, but no rank/MLP/epoch expansion is scheduled before the frozen Base result establishes the real gain over the original model.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/lora/pathvqa_lora_visual_all_attention_r128_seed44_20260826`; selected checkpoint `checkpoints/epoch_3`.

Correction (2026-08-26): the intended scope-matched LoRA baseline is Attention LoRA-r128 on visual layers17-24, because MMRL also modifies only those last8 visual layers. This completed LoRA run targets all24 visual layers and must therefore be treated as a broader-scope upper-bound baseline, not as evidence that LoRA is intrinsically8.16 points better under matched adaptation scope. A last8-layer PathVQA LoRA-r128 run is required for the formal comparison.

### 2026-08-26 - pathvqa_base_20260826

- Commit/config: commit `bc84ee3`; frozen original `/root/autodl-tmp/model`, no checkpoint and no training; generation/evaluation settings exactly match MMRL and LoRA.
- Dataset and split: PathVQA official test, all6,719 questions and858 image clusters.
- Seed: not applicable; deterministic greedy decoding with temperature0.0.
- Controlled change: unadapted Base control for the first PathVQA method comparison.
- Overall: **34.77**; image-clustered 95% bootstrap CI **[33.79, 35.81]**.
- Yes/No / Free-form: **67.34 / 2.14**.
- Per question type: how2.16, other0.00, what1.35, when0.00, where7.42, why0.00, yes/no67.34; Free-form question-type macro1.82.
- Evaluation timing: TTFT mean0.0484s, TPOT0.02166s/token,46.18 decode tokens/s, request mean0.1171s.
- Three-way deltas: MMRL versus Base is **+12.53 Overall, +15.23 Yes/No, +9.83 Free-form**; LoRA versus Base is **+20.69 Overall, +18.95 Yes/No, +22.43 Free-form**; LoRA versus MMRL is **+8.16 Overall**.
- Conclusion: PathVQA is genuinely difficult for the original model, especially under normalized exact-match for open answers. MMRL is not a failed or dead adaptation: it delivers a large gain over Base. However, standard visual-only LoRA learns the task substantially better, especially on Free-form answers, so the current MMRL cannot claim superior effectiveness on the primary dataset. Qualitative Base outputs also show plausible pathology phrases that may be semantically related but fail the single-reference exact match; semantic scoring or a blinded error sample should supplement, not replace, the official metric.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/base/pathvqa_base_20260826`.

### 2026-08-27 - pathvqa_lora_visual_last8_attention_r128_seed44_20260826

- Commit/config: commit `96f0550`; LoRA-r128 on visual Attention `qkv/proj` in layers17-24 only (0-based16-23), alpha256, dropout0.05; visual MLP and LLM frozen.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed: 44 / 42.
- Controlled change: scope-matched visual-only LoRA baseline using exactly the same last8 visual layers affected by MMRL.
- Validation Overall by epoch: **49.5606, 52.8998, 53.8904**; epoch3 selected.
- Test Overall: **53.8473**; image-clustered 95% bootstrap CI **[52.3630, 55.4189]**.
- Yes/No / Free-form: **84.2356 / 23.4138**.
- Per question type: how5.7554, other5.5556, what18.7523, when0.0000, where61.0209, why0.0000, yes/no84.2356; Free-form question-type macro15.1807.
- Trainable parameters: **6,291,456**; target audit selected0-based layers16-23 and exactly16 `qkv/proj` modules.
- Training: 3 epochs, LR1e-4, micro-batch1, gradient accumulation32; runtime8,772.4s; train loss0.7768.
- Conclusion: last8 LoRA retains most of the all24-layer LoRA result:53.85 versus55.45, only1.60 points lower while using one-third as many LoRA parameters. It is the valid scope-matched visual-only baseline.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/lora/pathvqa_lora_visual_last8_attention_r128_seed44_20260826`; selected checkpoint `checkpoints/epoch_3`.

### 2026-08-27 - pathvqa_mmrl_minimal_shared_s_mean_relation0_seed44_20260826

- Commit/config: commit `96f0550`; 40x1024 shared full-dimensional `S` directly enters one shared8-head residual Cross-Attention; Mean Pooling supplies one visual and one text memory token; no layer MLP projectors; layer embeddings retained; insertion into visual layers17-24; `open_full_ce`; Relation0.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed: 44 / 42.
- Controlled change: minimal MMRL architecture and Relation-off half of an exact shared-Stage1 comparison.
- Validation Overall by epoch: **46.2853, 46.8605, 46.6049**; epoch2 selected.
- Test Overall: **46.8373**; image-clustered 95% bootstrap CI **[45.5662, 48.2917]**.
- Yes/No / Free-form: **79.1196 / 14.5070**.
- Yes/No class audit:1,816 `yes` questions at69.1079% and1,546 `no` questions at90.8797%; without Relation the model remains strongly biased toward `no`.
- Per question type: how8.6331, other0.0000, what10.4706, when0.0000, where43.6195, why0.0000, yes/no79.1196; Free-form question-type macro10.4539.
- Trainable MMRL parameters: **7,927,808** exactly: shared S40,960; layer embeddings8,192; projectors0; visual Mean Pooling1,051,648; text Mean Pooling2,624,512; shared CA4,202,496.
- Stage3 runtime/loss:6,616s /0.8948.
- Diagnostics:1,845 steps,8 windows, no malformed rows; CE mean0.8919; CA grad norm0.7160; `delta_to_org_ratio`0.6680; CA delta/base ratio5.5979; text/visual pooling grad norms0.8021/0.2468; Relation loss exactly0.
- Conclusion: the minimal architecture remains a real specialist at+12.07 points over Base, but without Relation it does not exceed the earlier20.998M-parameter MMRL result.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/mmrl/pathvqa_mmrl_minimal_shared_s_mean_relation0_seed44_20260826`; selected checkpoint `checkpoints/stage3_epoch_2`.

### 2026-08-27 - pathvqa_mmrl_minimal_shared_s_mean_relation0050_seed44_20260826

- Commit/config: identical to the preceding minimal MMRL except uniform Relation weight0.05. It strictly reloads the same Stage1 checkpoint and untouched initial MMRL state with SHA-256 `a2cef33f65e3e837e99a0d188fd4186612f331d491b782f31314a5aa1bc1a093`.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed: 44 / 42.
- Controlled change: Relation0.05 versus0 under an exact shared-Stage1, matched-initialization comparison.
- Validation Overall by epoch: **45.4545, 49.5447, 48.7618**; epoch2 selected.
- Test Overall: **49.4121**; image-clustered 95% bootstrap CI **[48.1175, 50.7317]**.
- Yes/No / Free-form: **82.5996 / 16.1752**.
- Yes/No class audit:1,816 `yes` questions at79.6806% and1,546 `no` questions at86.0285%. Relative to the paired Relation0 run, Relation0.05 changes `yes` by+10.5727 and `no` by-4.8512 points, trading a modest loss on `no` for a much larger recovery on `yes`.
- Per question type: how9.3525, other0.0000, what11.2003, when0.0000, where51.7401, why0.0000, yes/no82.5996; Free-form question-type macro12.0488.
- Trainable MMRL parameters: **7,927,808**; parameter audit passed exactly.
- Stage3 runtime/loss:6,617s /0.8953.
- Diagnostics:1,845 steps,8 windows, no malformed rows; CE mean0.8915; CA grad norm0.6972; `delta_to_org_ratio`0.6873; CA delta/base ratio5.4629; text/visual pooling grad norms0.7683/0.2474; scaled Relation loss0.001819.
- Paired Relation effect: **+2.5748 Overall**, +3.4801 Yes/No, +1.6682 Free-form. Discordant correctness counts are284 Relation0-only versus457 Relation0.05-only (`McNemar p=2.20e-10`); image-clustered bootstrap 95% CI for the Overall difference is **[+1.7217, +3.4544]**.
- Matched LoRA comparison: last8 LoRA is+4.4352 Overall, +1.6359 Yes/No, and+7.2386 Free-form; clustered bootstrap 95% CI for its Overall lead is **[+3.3462, +5.5093]**.
- Conclusion: Relation now has a clear positive PathVQA effect under a clean controlled comparison, not a Gate/Stage1 trajectory confound. The minimal MMRL also improves over the earlier47.30 MMRL while cutting Stage3 MMRL parameters from20.998M to7.928M, although epoch-selection and architecture both changed, so that cross-run gain is descriptive rather than a pure ablation. Last8 LoRA remains stronger, primarily on Free-form answers.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/mmrl/pathvqa_mmrl_minimal_shared_s_mean_relation0050_seed44_20260826`; selected checkpoint `checkpoints/stage3_epoch_2`.

### 2026-08-27 - pathvqa_mmrl_minimal_shared_s_mean_grid5x8_relation0050_seed44_20260827

- Commit/config: commit `7b7bc44`; identical to the minimal shared-S Mean-Pooling MMRL + Relation0.05 run except that the 40 Rep Tokens use fixed cell-center RoPE coordinates on a 5x8 image grid instead of sharing the origin.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed: 44 / 42.
- Controlled change: Rep Token RoPE positions only; fixed coordinates, no learnable position parameters, and the same Stage1 checkpoint SHA-256 `a2cef33f65e3e837e99a0d188fd4186612f331d491b782f31314a5aa1bc1a093`.
- Validation Overall by epoch: **46.52, 45.90, 47.26**; epoch3 selected.
- Test Overall: **47.67**.
- Yes/No / Free-form: **81.02 / 14.27**.
- Trainable MMRL parameters: **7,927,808** exactly; unchanged from the origin-position control.
- Stage3 runtime/loss:6,608s /0.9042.
- Position audit: `grid_5x8`,40 slots and40 unique positions per sequence; the sampled first sequence spans `[2,1]` to `[18,28]`.
- Diagnostics:1,845 steps,8 windows, no malformed rows; CE mean0.9012; CA grad norm0.7757; `delta_to_org_ratio`0.6597; CA delta/base ratio5.8809; text/visual pooling grad norms0.8550/0.2431; scaled Relation loss0.001710.
- Matched effect versus origin-position Relation0.05: **-1.74 Overall, -1.58 Yes/No, and -1.91 Free-form**.
- Conclusion: fixed spatial RoPE is active and optimization remains healthy, but it consistently harms both answer categories. The shared Rep Tokens are semantic prompts rather than tokens aligned to fixed image patches; forcing distinct spatial phases introduces a false correspondence and breaks their useful shared-origin symmetry. Reject this direction and do not add learnable coordinates without a new token-to-region alignment mechanism.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/mmrl/pathvqa_mmrl_minimal_shared_s_mean_grid5x8_relation0050_seed44_20260827`; selected checkpoint `checkpoints/stage3_epoch_3`.

### 2026-08-27 - pathvqa_prompt_tuning_len20_seed44_20260827

- Commit/config: commit `722a65f`; classic static Prompt Tuning with20 learned prefix embeddings of width2,560; the complete Qwen3-VL backbone is frozen.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed: 44 / 42.
- Controlled change: formal frozen-backbone static Prompt Tuning baseline using the same PathVQA templates, evaluator, generation settings, and validation-selected test protocol as MMRL and LoRA.
- Validation Overall by epoch: **52.71, 52.76, 54.87**; epoch3 selected.
- Test Overall: **55.8268**; image-clustered 95% bootstrap CI **[54.4524, 57.1496]**.
- Yes/No / Free-form: **90.3331 / 21.2690**.
- Per question type: how7.9137, other22.2222, what16.4903, when0.0000, where57.3086, why0.0000, yes/no90.3331; Free-form question-type macro17.3225.
- Yes/No class audit:1,816 `yes` questions at91.2996% and1,546 `no` questions at89.1979%; the aggregate gain is not a one-class collapse.
- Trainable parameters: **51,200** exactly (`20 x 2,560`), versus7.928M for minimal MMRL and6.291M for last8 Attention LoRA-r128.
- Training:3 epochs, LR0.3, micro-batch2, gradient accumulation16; runtime5,969.6s; reported train loss12.0592.
- Evaluation timing: TTFT mean0.0474s, weighted TPOT0.01959s/token,51.05 decode tokens/s, request mean0.0989s.
- Deltas versus Base: **+21.0568 Overall, +22.9931 Yes/No, +19.1290 Free-form**.
- Deltas versus minimal MMRL+Relation0.05: **+6.4147 Overall, +7.7335 Yes/No, +5.0938 Free-form**.
- Deltas versus last8 Attention LoRA-r128: **+1.9795 Overall, +6.0975 Yes/No, but -2.1448 Free-form**.
- Conclusion: static Prompt Tuning is the current best PathVQA Overall result and demonstrates that the dominant adaptation bottleneck is the frozen LLM context/interface, not only internal ViT specialization. Contrary to the initial prediction, its largest advantage over LoRA is binary-answer calibration; LoRA remains stronger on Free-form. A dynamic multimodal Prompt should therefore be judged on whether it preserves the static prompt's balanced Yes/No performance while recovering the Free-form gap, not merely on aggregate Overall.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/prompt_tuning/pathvqa_prompt_tuning_len20_seed44_20260827`; selected checkpoint `checkpoints/epoch_3`.

### 2026-08-27 - pathvqa_mmrl_minimal_shared_s_mean_prompt20_relation0050_seed44_20260827

- Commit/config: commits `d057bf3` through `acf809a`; jointly trains the exact7,927,808-parameter minimal shared-S Mean-Pooling MMRL with Relation0.05 and a20-token static Soft Prompt. MMRL LR6e-5 retains fixed warmup plus epoch decay; Prompt LR0.3 uses3% warmup plus linear decay. The Prompt prefix is excluded from `mmrl_gating_mask`, so MMRL text pooling sees only the original question context.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42. The MMRL branch reloads the same Stage1 checkpoint used by the minimal controls; the Prompt uses seed44 token-embedding initialization.
- Controlled change: decisive complementarity test adding the independently strong static Prompt20 LLM-context interface to minimal MMRL+Relation0.05 while freezing the vision backbone and LLM weights.
- Validation-selected epoch: epoch3, validation Overall **55.3763**.
- Test Overall: **55.7523**; image-clustered95% bootstrap CI **[54.4270, 57.1364]**.
- Yes/No / Free-form: **88.9352 / 22.5201**.
- Per question type: how12.2302, other27.7778, what17.5848, when0.0000, where58.4687, why0.0000, yes/no88.9352; Free-form question-type macro19.3436.
- Trainable parameters: MMRL7,927,808 + Prompt51,200 = **7,979,008**. The compact checkpoint contains11,269,121 parameters because it additionally serializes frozen Stage1 delta modules.
- Training:3 epochs,1,845 optimizer steps, runtime6,747s, reported train loss0.7651.
- Diagnostics:8 windows, no malformed lines; CE mean0.7547; total MMRL grad norm0.1592; Prompt grad norm0.00894; CA grad norm0.0967; scaled Relation0.000200; `delta_to_org_ratio`0.1634; CA delta/base ratio20.121; Prompt norm grows from214.02 to663.34. Both branches are active, so the tie is not caused by a dead branch.
- Paired behavior versus Static Prompt: predictions are textually identical on4,460/6,719 questions (66.38%), but correctness changes on709 questions. The combination preserves3,394 jointly correct answers, breaks357 Prompt-correct answers, and fixes352 Prompt-wrong answers. By type, Yes/No has173 broken versus126 fixed, while Free-form has184 broken versus226 fixed. A Prompt-or-combination oracle reaches61.07 Overall, showing latent sample-level complementarity that the always-on joint model cannot arbitrate.
- Versus Static Prompt20: **-0.0745 Overall, -1.3979 Yes/No, +1.2511 Free-form**. Versus minimal MMRL+Relation0.05: **+6.3402 Overall, +6.3356 Yes/No, +6.3449 Free-form**. Versus last8 Visual LoRA: **+1.9050 Overall, +4.6996 Yes/No, -0.8937 Free-form**.
- Evaluation timing: TTFT mean0.06061s, weighted TPOT0.02364s/token,42.30 decode tokens/s, request mean0.11123s.
- Conclusion: the combination is statistically and practically tied with Static Prompt alone and fails the pre-registered success condition because Free-form22.52 remains below last8 LoRA23.41. This is not branch death or a trivial no-op: MMRL shifts capacity from binary calibration toward a different set of Free-form answers, but the shared CE objective provides no sample-wise authority/routing mechanism, so fixes and regressions cancel. The dominant PathVQA bottleneck is the LLM context interface; do not spend more budget on an always-on combination of the current internal-ViT MMRL with static Prompt Tuning.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/mmrl/pathvqa_mmrl_minimal_shared_s_mean_prompt20_relation0050_seed44_20260827`; selected checkpoint `checkpoints/stage3_epoch_3`.

### 2026-08-28 - pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827

- Commit/config: commit `619787d`; frozen Qwen3-VL backbone with 20 jointly trained 2,560-dimensional Static Prompt tokens. The same tokens act as 20 Cross-Attention queries over two sample-conditioned memory tokens: Mean-Pooled image tokens and Mean-Pooled question-only text tokens. The 256-dimensional 8-head CA returns a zero-initialized residual to the same 20-token LLM prefix. No pretrained Prompt checkpoint, Stage1, Gate, or Relation is used.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed: 44 / 42.
- Controlled change: replace Static Prompt20 with an equally long jointly trained dynamic multimodal Prompt. The backbone, question template, answer exclusion, generation settings, validation selection, and official normalized exact-match evaluator remain matched to the Static Prompt baseline.
- Validation Overall by epoch: **52.8040, 54.9609, 55.7278**; epoch3 selected.
- Test Overall: **56.5412**; image-clustered95% bootstrap CI for the model score **[55.2460, 57.8331]**.
- Yes/No / Free-form: **89.8275 / 23.2052**.
- Per question type: how9.3525, other38.8889, what17.5848, when0.0000, where64.0371, why4.5455, yes/no89.8275; Free-form question-type macro22.4015.
- Yes/No class audit:1,816 `yes` questions at93.7225% and1,546 `no` questions at85.2523%. Relative to Static Prompt, `yes` changes by+2.4229 and `no` by-3.9457 points, so the small aggregate Yes/No loss hides a calibration shift rather than uniform degradation.
- Trainable parameters: Static Prompt51,200 + Dynamic CA2,634,240 = **2,685,440**. The checkpoint file is10,745,909 bytes.
- Training:3 epochs, Prompt LR0.3, Dynamic CA LR3e-4, shared3% warmup and linear decay; runtime6,051.7s; reported train loss11.4899.
- Initialization audits: dynamic residual maximum absolute value is exactly0 at step0; memory has shape `(batch,2,2560)`; question-only text Memory has zero answer-token overlap.
- Diagnostics:93 rows over steps0-1,840. In epoch1/2/3 windows, normalized two-token attention entropy averages0.08931/0.000506/0.0000995, while visual attention mass averages0.46042/0.45076/0.443754. Over the final30 records, visual mass is0.443754 with std0.0000111 and entropy is0.0001008, indicating an almost fixed hard allocation of the160 query-head pairs between visual and text Memory. The Dynamic residual remains large and sample-varying in norm: epoch3 `delta/base` mean0.4748, range0.3420-0.6459. Dynamic grad norm remains approximately1.0 while Static Prompt grad norm declines to an epoch3 mean0.01849.
- Paired effect versus Static Prompt20: **+0.7144 Overall**, -0.5057 Yes/No, and **+1.9363 Free-form**. Dynamic-only/Static-only correct counts are375/327 Overall (`McNemar p=0.0760`),127/144 Yes/No (`p=0.3311`), and248/183 Free-form (`p=0.00202`). The image-clustered paired95% bootstrap CI for the Overall difference is **[-0.1021,+1.5520]**. Predictions are textually identical on4,393/6,719 questions (65.38%).
- Comparison to last8 Visual LoRA-r128: Dynamic Prompt is+2.6939 Overall and-0.2086 Free-form; it nearly closes LoRA's Free-form advantage while retaining much stronger binary accuracy.
- Evaluation timing: TTFT mean0.049734s, weighted TPOT0.019937s/token,50.157 decode tokens/s, request mean0.095868s.
- Conclusion: this is the strongest PathVQA seed44 point estimate and the first method to improve Static Prompt specifically on Free-form with a significant paired correctness shift. It does not yet prove a statistically stable Overall improvement because the clustered paired CI crosses zero. The near-zero entropy and exactly stable44.375% visual mass show that CA learned a fixed head/slot modality partition, not visibly sample-dependent attention routing. The residual values can still carry sample-conditioned image/text content, so the correct next test is checkpoint-level causal intervention (`delta=0`, shuffled Memory, fixed/mean residual) before adding seeds or claiming dynamic multimodal routing.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827`; selected checkpoint `checkpoints/epoch_3`; test log `eval_test_epoch_3.log`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

### 2026-08-28 - pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827__intervention_zero

- Commit/config: commit `26ef735`; inference-only causal intervention on the selected epoch3 Dynamic Prompt checkpoint. The jointly trained20-token Soft Prompt remains unchanged, but the complete sample-conditioned CA residual is forced to exact zero for every test sample.
- Dataset and split: PathVQA test6,719 with858 image clusters; source training seed/data seed44/42.
- Controlled change: Dynamic Prompt residual only; backbone, checkpoint, prefix length, test order, templates, generation, and official evaluator exactly match the normal56.5412 run.
- Overall: **49.2930**; image-clustered95% CI **[48.1936,50.3944]**.
- Yes/No / Free-form: **87.1802 / 11.3494**.
- Yes/No class audit: `yes`90.5837% and `no`83.1824%.
- Per question type: how7.1942, other11.1111, what7.8803, when0.0000, where35.4988, why0.0000, yes/no87.1802; Free-form macro10.2807.
- Intervention audit:6,719/6,719 samples changed; zero warmup samples.
- Paired normal-minus-zero effect: **+7.2481 Overall**, +2.6472 Yes/No, and **+11.8558 Free-form**. Normal-only/zero-only correct counts are604/117 Overall (`McNemar p=6.36e-80`),157/68 Yes/No (`p=2.79e-9`), and447/49 Free-form (`p=1.86e-81`). The image-clustered paired95% CI for the Overall difference is **[+6.3431,+8.1816]**.
- Inference timing:742.5s for6,719 samples; TTFT mean0.048596s, weighted TPOT0.019972s/token, request mean0.101003s.
- Conclusion: the Dynamic residual is indispensable inside the jointly optimized checkpoint, especially for Free-form. This is not an independently trained Static Prompt baseline: the Soft Prompt co-adapted with CA, so the result proves branch necessity and co-adaptation rather than claiming that any Dynamic model must beat a separately optimized Static Prompt by7.25 points.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827/eval_interventions/zero`; log `eval_intervention_zero.log`.

### 2026-08-28 - pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827__intervention_mean_residual_lag32

- Commit/config: commit `26ef735`; inference-only intervention using normal sample residuals for the first32 calibration samples, then replacing every subsequent sample residual with their fixed average.
- Dataset and split: PathVQA test6,719 with858 image clusters; source training seed/data seed44/42.
- Controlled change: residual sample dependence only; all weights and evaluation settings match the normal run. The first32 samples are explicitly retained as calibration and excluded in the supplementary clean-effect calculation.
- Overall: **43.1910**; image-clustered95% CI **[42.1834,44.2155]**.
- Yes/No / Free-form: **79.8037 / 6.5237**.
- Yes/No class audit: `yes`78.5793% and `no`81.2419%.
- Per question type: how7.9137, other33.3333, what2.9186, when0.0000, where28.0742, why4.5455, yes/no79.8037; Free-form macro12.7976.
- Intervention audit:6,719 samples seen,6,687 changed,32 calibration samples.
- Paired normal-minus-mean effect: **+13.3502 Overall**, +10.0238 Yes/No, and **+16.6816 Free-form**. Normal-only/mean-only correct counts are1,014/117 Overall (`McNemar p=7.01e-179`),423/86 Yes/No (`p=1.81e-54`), and591/31 Free-form (`p=2.79e-135`). The image-clustered paired95% CI for Overall is **[+12.2733,+14.3957]**; after excluding the first32 samples, the effect remains+13.4141 with CI[+12.3604,+14.4930].
- Inference timing:802.9s; TTFT mean0.048430s, weighted TPOT0.019850s/token, request mean0.110921s.
- Conclusion: a constant domain-level residual is not merely insufficient; it is substantially worse than removing the residual entirely. Dynamic residual vectors are not interchangeable static offsets, and averaging them destroys information needed by both binary and Free-form predictions.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827/eval_interventions/mean-residual`; log `eval_intervention_mean-residual.log`.

### 2026-08-28 - pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827__intervention_lagged_memory32

- Commit/config: commit `26ef735`; inference-only intervention preserving the trained CA but replacing each sample's visual and question Memory with Memory from32 samples earlier. Dataset-order audit confirms zero same-image donor pairs after the first32 samples.
- Dataset and split: PathVQA test6,719 with858 image clusters; source training seed/data seed44/42.
- Controlled change: current-sample Memory alignment only; the first32 samples remain normal while6,687 use mismatched Memory. Main VLM image/question inputs remain untouched, so the intervention changes only the Dynamic Prompt branch.
- Overall: **48.5787**; image-clustered95% CI **[47.4625,49.6968]**.
- Yes/No / Free-form: **84.1761 / 12.9282**.
- Yes/No class audit: `yes`84.0859% and `no`84.2820%.
- Per question type: how7.1942, other16.6667, what8.6830, when0.0000, where42.4594, why0.0000, yes/no84.1761; Free-form macro12.5005.
- Intervention audit:6,719 samples seen,6,687 changed,32 warmup samples; no same-image pair among changed samples.
- Paired normal-minus-lagged effect: **+7.9625 Overall**, +5.6514 Yes/No, and **+10.2770 Free-form**. Normal-only/lagged-only correct counts are705/170 Overall (`McNemar p=4.56e-78`),284/94 Yes/No (`p=2.73e-23`), and421/76 Free-form (`p=6.25e-59`). The image-clustered paired95% CI for Overall is **[+6.9923,+8.8855]**. Excluding the first32 samples gives+8.0006 Overall with CI[+7.0470,+8.9920].
- Relative to zero residual: lagged Memory is-0.7144 Overall with clustered CI[-1.5219,+0.0754], trading-3.0042 Yes/No (`p=1.63e-7`) for+1.5788 Free-form (`p=0.00375`). Mismatched domain Memory therefore retains limited generic open-answer signal but loses the much larger benefit of correct sample alignment.
- Inference timing:748.4s; TTFT mean0.049211s, weighted TPOT0.019497s/token, request mean0.102236s.
- Conclusion: current-sample visual/question Memory is causally necessary. Although attention weights harden into an almost fixed modality allocation, the selected Value content remains strongly sample-dependent; Dynamic Prompt conditioning operates through values and residual direction, not through visibly dynamic attention weights.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_mean_ca256_len20_seed44_20260827/eval_interventions/lagged-memory`; log `eval_intervention_lagged-memory.log`.

### 2026-08-28 - pathvqa_dynamic_prompt_sparse_visual_layers5_11_17_slots8_ca128_init0050_relation0050_seed44_20260828

- Commit/config: commit `5cb805f`; the Dynamic Prompt20+CA256 method is retained unchanged and jointly trained with a 1,201,155-parameter Sparse Visual MMRL branch. Eight shared 1,024-dimensional Rep tokens use CA128/4 heads over image-level Mean-Pooled visual states and question-only text states, then enter native visual layers5/11/17 (0-based) through independent insert/strip paths. The actual `tanh(gamma_l)` residual scale starts at0.05 and local Relation0.05 is averaged over the three anchors.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42.
- Controlled change: add the Sparse Visual branch to the exact seed44 Dynamic Prompt protocol. Relative to the aborted zero-scale attempt, only the initial visual residual scale changes from0 to0.05; Prompt LR0.3, Dynamic CA LR3e-4, Sparse Visual LR3e-4, architecture, Relation and all data/evaluation settings remain fixed.
- Validation Overall / Yes-No / Free-form by epoch: epoch1 **54.8171 / 88.5760 / 21.1551**; epoch2 **51.5897 / 87.4240 / 15.8583**; epoch3 **54.9129 / 88.2560 / 21.6656**. Epoch3 selected by Overall.
- Test Overall: **56.0202**; image-clustered95% bootstrap CI **[54.7117,57.3767]**.
- Yes/No / Free-form: **88.8757 / 23.1159**.
- Per question type: how13.6691, other55.5556, what17.0741, when16.6667, where64.2691, why4.5455, yes/no88.8757; Free-form question-type macro28.6300.
- Trainable parameters: Soft Prompt51,200 + Dynamic Prompt CA2,634,240 + Sparse Visual1,201,155 = **3,886,595**.
- Training:3 epochs,1,845 optimizer steps, runtime7,122.0s, reported train loss11.2830.
- Late diagnostics at step1,840: Sparse Visual grad norm0.1529 versus Dynamic Prompt0.9881; scales are0.04492/0.07520/0.05029 at layers5/11/17. Applied residual ratios are0.000660/0.034100/0.000321, so the branch is active but overwhelmingly dominated by layer11. Attention entropy is nearly zero at all anchors and visual masses settle near0.25/0.28125/0.50. Mean Sparse residual ratio is0.011694. Scaled Relation is only1.13e-7 and therefore does not materially control optimization.
- Versus Dynamic Prompt20+CA256: **-0.5210 Overall, -0.9518 Yes/No, -0.0893 Free-form**, while Free-form question-type macro changes by+6.2285. The macro gain is concentrated in small how/other/when categories and does not establish a primary-metric improvement.
- Evaluation timing: TTFT mean0.05545s; weighted TPOT0.019804s/token;50.496 decode tokens/s; request mean0.10601s.
- Conclusion: the0.05 initialization successfully fixes branch death, but the active Sparse Visual addition does not improve the main method. It perturbs binary calibration, leaves aggregate Free-form essentially unchanged and lowers both validation and test Overall. Do not run seed45 or tune this sparse architecture further under the current one-week budget; retain it as evidence that stronger internal visual adaptation is not complementary to the Dynamic Prompt interface under shared CE.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_sparse_visual_layers5_11_17_slots8_ca128_init0050_relation0050_seed44_20260828`; selected checkpoint `checkpoints/epoch_3`.

Correction (2026-08-28): the immediate recommendation to stop all tuning was too strong. Relative to the original Dynamic Prompt at the same epoch, Sparse Visual epoch1 changes Overall/Yes-No/Free-form by **+2.0131/+0.7360/+3.2866**, before collapsing in epoch2 and recovering in epoch3. This is a positive early signal followed by unstable co-adaptation, not evidence that the visual branch is intrinsically useless. One strict diagnostic is therefore permitted: reduce only Sparse Visual LR from3e-4 to3e-5. This can support an update-timescale/CE-competition explanation if it removes the epoch2 collapse, but cannot by itself prove or disprove gradient conflict.

### 2026-08-28 - pathvqa_dynamic_prompt_sparse_visual_layers5_11_17_slots8_ca128_init0050_relation0050_sparse_lr3e5_seed44_20260828

- Commit/config: commit `89deb94`; exact same Dynamic Prompt20+CA256 and Sparse Visual layers5/11/17 architecture as the3e-4 control, with Soft Prompt LR0.3, Dynamic CA LR3e-4, initial residual scale0.05 and Relation0.05 unchanged. The only controlled change is Sparse Visual LR **3e-4 -> 3e-5**.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42.
- Validation Overall / Yes-No / Free-form by epoch: epoch1 **54.5295 / 88.2880 / 20.8679**; epoch2 **55.1366 / 88.1920 / 22.1761**; epoch3 **56.8461 / 88.8640 / 24.9202**. Epoch3 selected. Unlike the3e-4 control's54.8171 -> 51.5897 -> 54.9129 Overall trajectory, the lower-LR run improves monotonically and removes the epoch2 collapse.
- Test Overall: **57.5086**; image-clustered95% bootstrap CI **[56.1883,58.9046]**.
- Yes/No / Free-form: **89.6193 / 25.3500**.
- Yes/No class audit:1,816 `yes` questions at93.7225% and1,546 `no` questions at84.7995%. Relative to the original Dynamic Prompt, `yes` is unchanged and `no` changes by-0.4528 points; the aggregate binary difference is not significant.
- Per question type: how10.0719, other38.8889, what19.7738, when0.0000, where66.5893, why4.5455, yes/no89.6193; Free-form question-type macro23.3116.
- Trainable parameters: Soft Prompt51,200 + Dynamic Prompt CA2,634,240 + Sparse Visual1,201,155 = **3,886,595**.
- Training:3 epochs,1,845 optimizer steps, runtime7,122.0s, reported train loss11.3922.
- Final diagnostics at step1,840: Soft Prompt norm915.972; Dynamic/Sparse grad norms0.99976/0.01116. Sparse scales remain close to initialization at0.04980/0.05225/0.05151 for layers5/11/17. Applied residual ratios are only0.000464/0.000710/0.001016, with mean0.000730; unlike the3e-4 control, layer11 no longer dominates. Sparse attention entropy remains0.416/0.574/0.565 and visual mass0.306/0.318/0.844. Dynamic attention remains hard with entropy3.98e-5 and visual mass0.44375. Scaled Relation is only1.41e-8 and cannot explain the gain.
- Paired versus the original Dynamic Prompt20+CA256: **+0.9674 Overall, -0.2082 Yes/No, +2.1448 Free-form**. New-only/old-only correct counts are338/273 Overall (`McNemar p=0.00956`),128/135 Yes/No (`p=0.7115`), and210/138 Free-form (`p=0.000134`). Image-clustered paired95% CIs are **[+0.1985,+1.7252] Overall**, [-1.1138,+0.7405] Yes/No, and **[+0.8408,+3.4390] Free-form**. Predictions are textually identical on4,661/6,719 questions (69.37%).
- Paired versus the exact3e-4 Sparse Visual control: **+1.4883 Overall, +0.7436 Yes/No, +2.2341 Free-form**. New-only/old-only counts are397/297 Overall and237/162 Free-form; clustered paired95% CIs are **[+0.6213,+2.3085] Overall** and **[+0.8736,+3.6055] Free-form**.
- Conclusion: the pre-registered LR diagnostic succeeds. Sparse LR3e-4 over-adapts and destabilizes the jointly trained LLM-entry Prompt; reducing only its LR by10x removes the epoch2 collapse and produces statistically positive paired Overall and Free-form gains. This supports an update-timescale mismatch / unstable shared-CE co-adaptation explanation, not the stronger claim that LR alone proves gradient conflict. Because the final Sparse residual averages only0.073%, the result does not yet distinguish a causally useful inference-time visual correction from a training-time regularization/trajectory effect. A same-checkpoint Sparse-off intervention is required before claiming direct visual complementarity.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_sparse_visual_layers5_11_17_slots8_ca128_init0050_relation0050_sparse_lr3e5_seed44_20260828`; selected checkpoint `checkpoints/epoch_3`; test log `eval_test_epoch_3.log`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

Storage correction (2026-08-28): by explicit user direction, the `checkpoints/` and `final/` weight directories for both completed dual-path Sparse Visual runs (LR3e-4 and LR3e-5) were deleted after their results were fully recorded. Evaluation logs, summaries, predictions, comparisons and diagnostics remain at the recorded output roots. The57.5086 result remains valid historical evidence, but its checkpoint can no longer support the planned Sparse-off intervention or direct reuse.

### 2026-08-29 - pathvqa_dynamic_prompt_sparse_visual_single_pass_layers5_11_17_slots8_ca128_lr3e5_seed44_20260828

- Commit/config: commit `e3715c0`; corrected single-pass Sparse Visual baseline using `Strip(Block([Rep; h]))` at visual layers5/11/17. It retains the independently trainable20-token Prompt, Dynamic CA256/8 heads over Mean-Pooled visual/question Memory, shared8x1024 visual Rep base, shared Visual CA128/4 heads and Sparse LR3e-5. Residual Scale, Relation and a direct shared-S text bridge are absent.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42.
- Controlled change: establish the corrected single-pass reference after removing the historical dual execution path; all three experiments in this suite use the same data order,3 epochs, Prompt LR0.3, Dynamic CA LR3e-4 and Sparse LR3e-5.
- Validation Overall / Yes-No / Free-form by epoch: epoch1 **55.0567 / 87.9360 / 22.2719**; epoch2 **56.2071 / 88.0640 / 24.4416**; epoch3 **57.5172 / 89.7280 / 25.3989**. Epoch3 selected.
- Test Overall: **57.7913**; image-clustered95% bootstrap CI **[56.4283,59.1711]**.
- Yes/No / Free-form: **89.7680 / 25.7671**.
- Per question type: how9.3525, other38.8889, what20.0292, when0.0000, where68.4455, why4.5455, yes/no89.7680; Free-form question-type macro23.5436.
- Trainable parameters: Prompt51,200 + Dynamic CA2,634,240 + Sparse Visual1,201,152 = **3,886,592**.
- Training:3 epochs,1,845 optimizer steps, runtime6,914.1s, reported train loss11.3922.
- Diagnostics:93 rows through step1,840. Over the last30 rows, Prompt norm875.14 and grad norm0.01340; Dynamic/Sparse grad norms0.7821/0.5845. Dynamic attention again hardens to normalized entropy5.23e-5 with44.375% visual mass. Dynamic delta/base averages0.4781. Sparse Rep/input and output/input norm ratios average0.7834 and1.1062, confirming a strongly active single-pass branch rather than the historical0.073% residual-scale regime.
- Suite comparison: this is the winner. It exceeds the shared-S-Memory keep-P variant by **+1.6520 Overall**, +0.7436 Yes/No and **+2.5618 Free-form**. Baseline-only/bridge-only correct counts are381/270 Overall and232/146 Free-form; McNemar p-values are1.55e-5 and1.14e-5. The image-clustered paired95% CI for Overall is **[+0.9157,+2.4331]**.
- Historical comparison: its point estimate is+0.2827 Overall and+0.4171 Free-form above the deleted-checkpoint dual-path57.5086 run, but that is an architecture change rather than a same-checkpoint intervention and is not used as causal evidence.
- Evaluation timing: TTFT mean0.079574s, weighted TPOT0.021946s/token,45.566 decode tokens/s, request mean0.137107s.
- Conclusion: the corrected single-pass Sparse Visual method not only preserves the low-LR gain but establishes the new PathVQA seed44 best point estimate. The expensive historical dual block execution, explicit residual scale and Relation are not required for this result. This checkpoint is the supported reference for any additional seed or causal Sparse-off analysis.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_sparse_visual_single_pass_layers5_11_17_slots8_ca128_lr3e5_seed44_20260828`; selected checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

### 2026-08-29 - pathvqa_dynamic_prompt_shared_s_memory_main_merger_keep_p_layers5_11_17_slots8_ca128_lr3e5_seed44_20260828

- Commit/config: commit `e3715c0`; exact single-pass baseline plus a direct shared-S text bridge. The8 last-anchor Rep outputs are mapped by the frozen input-aligned main `visual.merger`, reduced to one sample-level S-Memory token and appended as the third Text CA Memory. The independently trainable20-token Prompt remains Query and residual base.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42.
- Controlled change versus the winning baseline: enable only the parameter-free S-Memory bridge; all trainable tensors, learning rates, initialization, data order and evaluation settings remain identical.
- Validation Overall / Yes-No / Free-form by epoch: epoch1 **52.1649 / 86.7200 / 17.7090**; epoch2 **54.3378 / 88.0320 / 20.7403**; epoch3 **56.3189 / 89.3440 / 23.3886**. Epoch3 selected.
- Test Overall: **56.1393**; image-clustered95% bootstrap CI **[54.8226,57.5498]**.
- Yes/No / Free-form: **89.0244 / 23.2052**.
- Per question type: how11.5108, other27.7778, what17.3294, when0.0000, where65.4292, why4.5455, yes/no89.0244; Free-form question-type macro21.0988.
- Trainable parameters: unchanged at **3,886,592**; the reused main Visual Merger remains frozen and is held by weak reference rather than registered again.
- Training:3 epochs,1,845 optimizer steps, runtime6,939.8s, reported train loss11.3500.
- Diagnostics:93 rows through step1,840. The bridge is not dead: last30 mean shared-S parameter grad is0.02356, S-Memory interface grad is0.00905 and total Sparse grad is0.6260. Text CA assigns a stable23.752% attention mass to S-Memory,43.166% to ordinary visual Memory and33.083% to text; normalized entropy is1.91e-4. S-Memory norm averages54.38, while Dynamic delta/base remains a normal0.4417.
- Paired effect versus single-pass baseline: **-1.6520 Overall**, -0.7436 Yes/No and **-2.5618 Free-form**. Bridge-only/baseline-only correct counts are270/381 Overall and146/232 Free-form. The image-clustered paired95% CI for bridge-minus-baseline Overall is **[-2.4331,-0.9157]**; predictions are identical on4,589/6,719 questions.
- Evaluation timing: TTFT mean0.079963s, weighted TPOT0.020459s/token,48.878 decode tokens/s, request mean0.136093s.
- Conclusion: reject the direct shared-S bridge. It receives healthy CE gradients and substantial stable attention yet degrades every primary aggregate metric throughout training. The failure is therefore not branch death; the extra shortcut supplies a competing representation that displaces the already effective native visual/question Memories and worsens joint optimization.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_shared_s_memory_main_merger_keep_p_layers5_11_17_slots8_ca128_lr3e5_seed44_20260828`; selected checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

### 2026-08-29 - pathvqa_dynamic_prompt_shared_s_memory_main_merger_no_trainable_p_layers5_11_17_slots8_ca128_lr3e5_seed44_20260828

- Commit/config: commit `e3715c0`; identical to the keep-P shared-S bridge except the same seed44-initialized20-token P0 is frozen. P0 retains Query positions and prefix length, while Dynamic CA and Sparse Visual remain trainable.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42.
- Controlled change versus the keep-P bridge: remove only the51,200 trainable Prompt parameters; no sequence length, Query initialization, Memory, backbone or scheduling change.
- Validation Overall / Yes-No / Free-form by epoch: epoch1 **51.4140 / 86.3360 / 16.5922**; epoch2 **53.7945 / 87.0400 / 20.6445**; epoch3 **53.8105 / 87.0080 / 20.7084**. Epoch3 narrowly selected; learning essentially plateaus after epoch2.
- Test Overall: **54.7105**; image-clustered95% bootstrap CI **[53.3957,56.0586]**.
- Yes/No / Free-form: **87.4479 / 21.9243**.
- Per question type: how12.2302, other16.6667, what17.3294, when0.0000, where55.9165, why0.0000, yes/no87.4479; Free-form question-type macro17.0238.
- Trainable parameters: Dynamic CA2,634,240 + Sparse Visual1,201,152 = **3,835,392**; Prompt trainable parameters are exactly0.
- Training:3 epochs,1,845 optimizer steps, runtime6,929.2s, reported train loss12.3435.
- Diagnostics: P0 norm remains4.91334 with exactly zero gradient. The bridge remains active: last30 shared-S parameter/interface grad norms are0.01944/0.01780 and S-Memory receives35.000% attention. However Dynamic delta/base explodes to a last30 mean **76.77x** because the frozen token-embedding P0 is tiny relative to the learned residual. Dynamic CA is forced to reconstruct nearly the entire domain Prompt instead of refining a learned base.
- Paired effect versus keep-P bridge: **-1.4288 Overall**, -1.5764 Yes/No and-1.2809 Free-form. No-P-only/keep-P-only counts are297/393 Overall,141/194 Yes/No and156/199 Free-form; McNemar p-values are0.000292,0.00443 and0.0257. The image-clustered paired95% CI for no-P-minus-keep-P Overall is **[-2.2509,-0.6522]**.
- Paired effect versus winning baseline: **-3.0808 Overall**, -2.3200 Yes/No and-3.8427 Free-form; clustered paired95% CI **[-4.0072,-2.2300]**.
- Evaluation timing: TTFT mean0.081610s, weighted TPOT0.019485s/token,51.321 decode tokens/s, request mean0.137366s.
- Conclusion: reject replacing the independent trainable Prompt with frozen P0 plus shared S. The learned P is not redundant scaffolding; it supplies the high-capacity domain anchor around which sample-conditioned CA can operate as a residual. Freezing it harms both binary calibration and open answers and produces a pathological residual/base scale ratio.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_shared_s_memory_main_merger_no_trainable_p_layers5_11_17_slots8_ca128_lr3e5_seed44_20260828`; selected checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

### 2026-08-29 - pathvqa_dynamic_prompt_raw_shared_s_separate_residual_keep_p_layers5_11_17_slots8_ca128_lr3e5_seed44_20260829

- Commit/config: commit `26f95ed`; corrected raw Shared-S keep-P experiment. It preserves the57.79 single-pass baseline's independently trainable P20, original CA256 over Mean-Pooled visual/question Memory, and Sparse Visual injections at layers5/11/17. The unconditioned8x1024 shared parameter bank S is separately mapped by the frozen main `visual.merger` into2 LLM-space tokens; a new zero-output-initialized CA128/4-head branch reads those tokens into20 extra Prompt residuals. No conditioned visual Rep is exported.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42.
- Controlled change versus the57.79 baseline: add only the raw-S-to-text CA128 residual branch. P20, original visual/question Dynamic CA, Sparse Visual architecture, initialization, data order,3 epochs, Prompt LR0.3, Dynamic/Shared-S CA LR3e-4, Sparse LR3e-5 and evaluator remain matched.
- Validation Overall / Yes-No / Free-form by epoch: epoch1 **54.53 / 88.16 / 21.00**; epoch2 **53.49 / 88.58 / 18.51**; epoch3 **54.75 / 88.61 / 21.00**. Epoch3 selected.
- Test Overall: **54.9487**; image-clustered95% bootstrap CI **[53.65,56.26]**.
- Yes/No / Free-form: **88.3105 / 21.5371**; Free-form question-type macro20.6440.
- Per question type: how12.2302, other27.7778, what15.5053, when0.0000, where63.8051, why4.5455, yes/no88.3105.
- Trainable parameters: P51,200 + original Dynamic CA2,634,240 + raw-S CA1,323,520 + Sparse Visual1,201,152 = **5,210,112**.
- Training:3 epochs, runtime6,960s, reported train loss11.44.
- Late diagnostics over the last30 rows: P norm829.44; original Dynamic delta/base0.4598; raw-S residual/base0.03949; raw-S CA grad0.12697; shared-S parameter grad0.03170; mapped-S interface grad0.001309; Sparse total grad0.40464. The branch is active but small. More importantly, visual attention mass at layers5/11/17 becomes47.86%/96.50%/91.16%, versus the baseline's approximately67%/50%/62% final pattern, showing that CE coupling through raw S drives the late visual anchors toward near visual-only routing.
- Paired effect versus the exact57.7913 baseline: **-2.8427 Overall**, -1.4575 Yes/No and **-4.2300 Free-form**. Variant-only/baseline-only correct counts are235/426 Overall,97/146 Yes/No and138/280 Free-form; McNemar p-values are9.94e-14,0.00201 and3.32e-12. The image-clustered paired95% CI for Overall is **[-3.6682,-2.0402]**.
- Evaluation timing: TTFT mean0.053676s, weighted TPOT0.019891s/token,50.273 decode tokens/s, request mean0.106494s.
- Conclusion: reject the separate raw-S residual architecture. This is not branch death, excessive residual magnitude or a mere test fluctuation. The controlled branch addition significantly harms both answer categories and coincides with a major reorganization of visual CA at layers11/17. This is strong evidence for cross-branch optimization interference, but the current run changes both the forward residual and the backward path; a text-side stop-gradient control would be required to distinguish harmful shared gradients from harmful raw-S residual content.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_raw_shared_s_separate_residual_keep_p_layers5_11_17_slots8_ca128_lr3e5_seed44_20260829`; selected checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

### 2026-08-29 - pathvqa_dynamic_prompt_raw_shared_s_direct_prompt_layers5_11_17_slots8_ca128_lr3e5_seed44_20260829

- Commit/config: commit `26f95ed`; corrected pure raw Shared-S experiment. Independent P and frozen P0 are both absent. The unconditioned8x1024 S bank is mapped by the frozen main `visual.merger` into its native2x2560 outputs, which directly form the LLM prefix base and the original CA256 queries over Mean-Pooled visual/question Memory. The same raw S remains the visual Rep basis for layers5/11/17; no conditioned Rep is exported and no token expander is added.
- Dataset and split: PathVQA train19,654; validation6,259 for epoch selection; test6,719 with858 image clusters.
- Seed / data seed:44 /42.
- Controlled change versus the57.79 baseline: replace independent P20 with the2 native `visual.merger(raw S)` tokens; all visual architecture, Dynamic CA256, Sparse LR3e-5, Dynamic LR3e-4, data order, training length and evaluator remain matched.
- Validation Overall / Yes-No / Free-form by epoch: epoch1 **49.93 / 83.78 / 16.18**; epoch2 **52.12 / 88.06 / 16.27**; epoch3 **51.80 / 87.84 / 15.86**. Epoch2 selected.
- Test Overall: **51.6446**; image-clustered95% bootstrap CI **[50.32,53.02]**.
- Yes/No / Free-form: **87.3290 / 15.9071**; Free-form question-type macro12.5477.
- Per question type: how5.7554, other11.1111, what11.7840, when0.0000, where46.6357, why0.0000, yes/no87.3290.
- Trainable parameters: original Dynamic CA2,634,240 + Sparse Visual1,201,152 = **3,835,392**; independent Prompt parameters are exactly0 and effective prefix length is2.
- Training:3 epochs, runtime6,812s, reported train loss13.16.
- Late diagnostics over the last30 rows: mapped raw-S Prompt norm70.33; Dynamic delta/base **2.8340**; shared-S parameter grad0.67544; mapped-S interface grad0.05325; Sparse total grad0.69522. Dynamic attention hardens to81.25% visual mass at the final step. The high gradients prove raw S receives CE, but its3e-5 optimizer group and2-token capacity leave a nearly static, low-capacity base while CA256 must generate a residual almost3x larger than that base.
- Paired effect versus the exact57.7913 baseline: **-6.1467 Overall**, -2.4390 Yes/No and **-9.8600 Free-form**. Variant-only/baseline-only correct counts are268/681 Overall,158/240 Yes/No and110/441 Free-form; McNemar p-values are4.08e-42,4.63e-5 and6.45e-48. The image-clustered paired95% CI for Overall is **[-7.0982,-5.2028]**.
- Evaluation timing: TTFT mean0.053031s, weighted TPOT0.019733s/token,50.675 decode tokens/s, request mean0.102321s.
- Conclusion: reject direct replacement of P20 by native2-token raw S. This run diagnoses a different failure from the keep-P version: the frozen Merger output is trainable through S, but a shared parameter constrained to Sparse LR3e-5 cannot simultaneously behave like the successful LR0.3 textual domain anchor, and two prefix positions are insufficient for PathVQA Free-form adaptation. Raising S to Prompt LR would destroy the controlled visual optimization regime rather than cleanly solve the conflict.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_raw_shared_s_direct_prompt_layers5_11_17_slots8_ca128_lr3e5_seed44_20260829`; selected checkpoint `checkpoints/epoch_2`; diagnostics `dynamic_prompt_diagnostics.jsonl`.

### 2026-08-29 - pathvqa_dynamic_prompt_asymmetric_shared_s_text_owned_visual_readonly_layers5_11_17_slots8_adapter128_seed44_20260829

- Commit/config: commit `8da2390`; the20x2560 trainable LLM Prompt is the sole shared S and remains in the Prompt LR0.3 group. The visual branch can only read `detach(LayerNorm(S))`, maps20 Prompt tokens to8 slots with a learned Token Mixer and2560->128->1024 adapter, and then uses the existing CA128/4-head single-pass injections at visual layers5/11/17. Dynamic Text CA256 remains at3e-4, while the visual adapter, Visual CA and layer embeddings use3e-5. There is no independent visual S, second P, Relation or `visual.merger`.
- Dataset and split: PathVQA train19,654; fixed full validation6,259 with832 image clusters. Per the current development protocol, only epoch3 Validation was evaluated; Test was not run.
- Seed / data seed:44 /42.
- Controlled change versus the single-pass baseline: replace the independent8x1024 visual Rep bank with a detached low-rank projection of the existing P20/S, while preserving Prompt length/init/LR, Dynamic CA, visual anchor layers, Visual CA, data order and3-epoch schedule.
- Fixed epoch3 Validation Overall: **56.8142**; image-clustered standalone95% CI **[55.3244,58.2287]**.
- Yes/No / Free-form: **88.2240 /25.4946**; Free-form question-type macro19.6369.
- Binary class audit: among1,712 `yes` references, baseline/new accuracies are95.3271/95.6776 (+0.3505); among1,413 `no` references they are82.9441/79.1932 (**-3.7509**). The aggregate binary loss is therefore an amplified affirmative bias, not a uniform recognition decline.
- Per question type: how9.3023, other14.2857, what19.5447, when0.0000, where69.9267, why4.7619, yes/no88.2240.
- Trainable parameters: S/P51,200 + Dynamic CA2,634,240 + Sparse Visual1,651,872 = **4,337,312**.
- Training:3 epochs,1,845 optimizer steps, runtime6,893.1s, reported train loss11.4963.
- Late diagnostics over the last30 rows: S/P grad0.01487; Dynamic/Sparse/visual-adapter grad norms0.7926/0.5937/0.2124; the visual-to-S gradient-block audit is1.0 for every diagnostic row. Token-Mixer normalized entropy0.4435 and adapter Rep norm3.399 confirm a live, non-collapsed adapter. Dynamic delta/base0.4689, visual mass46.876% and entropy0.000197 remain close to the baseline text-side branch.
- Training trajectory: visual text-dominance exists from the first0-280-step window, where layers5/11/17 visual mass is35.58%/29.13%/29.35% versus baseline59.48%/54.76%/62.49%; it is therefore an architectural/query-source bias rather than a late-training accident. Adapter Rep norm rises0.623 ->3.366 and its gradient0.0427 ->0.1974, while Token-Mixer entropy remains effectively fixed0.4443 ->0.4435. The feature projections amplify the branch, but the arbitrary20-to-8 token routing does not discover meaningful specialization.
- Visual-side mechanism shift: layers5/11/17 Visual CA mass becomes28.75%/27.31%/28.65%, versus baseline68.43%/50.26%/61.09%. Rep/input ratios increase to1.588/1.486/0.678 versus baseline1.173/0.831/0.346. The detached text-owned anchor therefore makes the visual branch stronger but markedly text-memory dominated.
- Paired effect versus the exact single-pass baseline epoch3 Validation57.5172: **-0.7030 Overall**, -1.5040 Yes/No and +0.0957 Free-form. New-only/baseline-only correct counts are229/273 Overall,82/129 Yes/No and147/144 Free-form; McNemar p-values are0.0549,0.00148 and0.9067. The image-clustered paired95% CI for Overall is **[-1.4045,+0.0161]**.
- Evaluation timing: TTFT mean0.053565s, weighted TPOT0.019811s/token,50.478 decode tokens/s, request mean0.105524s.
- Conclusion: gradient decoupling works as designed and rescues raw hard sharing from a6.15-point failure to a near-baseline result, but it does not improve the supported method. Free-form is statistically tied while binary calibration degrades significantly. The remaining failure is forward over-coupling, not backward gradient pollution: replacing the independent visual Rep basis with text-derived S causes Visual CA to abandon visual Memory and amplifies Rep strength. Do not add seed45 for this pure replacement architecture. Any final rescue must preserve the independent visual basis and add only a bounded detached-S residual; that would be a new hypothesis rather than a retune of this run.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_asymmetric_shared_s_text_owned_visual_readonly_layers5_11_17_slots8_adapter128_seed44_20260829`; checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`; evaluation `eval_validation/epoch_3`.

### 2026-08-30 - pathvqa_dynamic_prompt_full_workspace_z32_d1024_blocks3_private_p20_private_s8_seed44_20260829

- Commit/config: commit `deebda4`; preserve the exact supported private baseline components (P20 at LR0.3, Text CA256/8 heads at3e-4, independent visual S8, shared private Visual CA128/4 heads at3e-5 and single-pass insert/strip at visual layers5/11/17). Add a32x1024 sample-conditioned workspace Z that is sequentially updated at the three anchors by independent full-token Cross-Attention, Self-Attention and FFN4096 blocks. Full current-layer visual tokens and full question-token embeddings are the workspace Memory. Separate Text CA1024/16-head and three Visual CA1024/16-head residual readers consume Z; their output projections start at exact zero, and Z is never concatenated into the original private CA Memories.
- Dataset and split: PathVQA train19,654; fixed full validation6,259 with832 image clusters. Only epoch3 Validation was evaluated under the current protocol; Test was not run.
- Seed / data seed:44 /42.
- Controlled change versus the single-pass baseline: add only the full-capacity shared workspace trunk and its four separate residual readers. The private P, private visual S, original Text/Visual CAs, anchor layers, injection rule, data order and their learning rates remain present. Workspace LR is1e-4.
- Fixed epoch3 Validation Overall: **59.4504**; image-clustered standalone95% CI **[58.0231,60.8857]**.
- Yes/No / Free-form: **90.6240 /28.3663**; Free-form question-type macro19.4184.
- Per question type: how8.5271, other14.2857, what23.0377, when0.0000, where70.6601, why0.0000, yes/no90.6240.
- Trainable parameters: Prompt51,200 + Dynamic CA2,634,240 + private Sparse Visual1,201,152 + Shared Workspace73,009,664 = **76,896,256**.
- Training:3 epochs,1,845 optimizer steps, runtime7,322s, reported train loss10.93. This is408s /5.9% slower than the exact baseline's6,914.1s despite19.8x trainable parameters.
- Paired effect versus the exact single-pass baseline epoch3 Validation57.5172/89.7280/25.3989: **+1.9332 Overall, +0.8960 Yes/No and +2.9675 Free-form**. Workspace-only/baseline-only correct counts are352/231 Overall,126/98 Yes/No and226/133 Free-form; exact McNemar p-values are6.13e-7,0.0710 and1.05e-6. Image-clustered paired95% CIs are **[+1.1747,+2.7132] Overall**, [0.0000,+1.7948] Yes/No and **[+1.8602,+4.1721] Free-form**. Predictions are textually identical on4,478/6,259 questions (71.55%).
- Binary class audit: `yes` improves by+1.9276 points (45 new-only /12 baseline-only, p=1.31e-5, paired clustered CI[+1.0782,+2.8324]); `no` changes by-0.3539 (81/86, p=0.7570, CI[-2.1161,+1.5626]). The aggregate binary gain is affirmative-side, but the primary Overall improvement is not a binary-calibration artifact because the larger Free-form gain is independently significant.
- Mechanism trajectory: over the first30 versus last30 diagnostic rows, Workspace grad remains healthy0.6160->0.5745 while private Sparse Visual grad falls0.1244->0.0615. Workspace visual delta/private-Rep rises2.836x->5.246x and total Rep/input rises2.479x->5.472x, yet frozen visual Block output/input stays stable1.112x->1.107x. The shared visual path therefore becomes functionally dominant without causing a numerical backbone-output explosion.
- Workspace behavior: full-token attention becomes strongly selective (layer5/11/17 normalized entropy first30 0.888/0.906/0.821 -> last30 0.377/0.365/0.226), while final text and visual readers remain nearly uniform over32 Z slots (about0.9998 normalized entropy). Current evidence supports a useful deep full-token workspace with global/mean-like downstream readout; it does not yet establish semantic specialization among the32 slots.
- Final-step diagnostics: Dynamic and Workspace-text delta/base are0.3477 and0.2219; private Sparse grad0.0438 versus Workspace0.5790; layer5/11/17 Workspace visual delta/private-Rep are7.310/5.305/2.671; aggregate Rep/input4.977 and Block output/input1.085. The high score coexists with shared-path takeover, so retaining private branches is structurally true but cooperative contribution from both paths remains unproven.
- Evaluation timing: TTFT mean0.058198s, weighted TPOT0.019724s/token,50.698 decode tokens/s, request mean0.104538s. Timing is recorded but not used for a strict cross-run claim because the baseline was evaluated under a different runtime session.
- Conclusion: this is the new PathVQA seed44 development best and validates the high-capacity-first strategy. It clears the pre-registered success rule with a significant Overall gain driven primarily by Free-form improvements. Before parameter reduction, run seed45 for replication and perform same-checkpoint inference interventions that separately disable/scale the Workspace text and visual residual readers; these are required to distinguish useful shared coordination from a high-capacity visual takeover and to choose a principled slimming target.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_dynamic_prompt_full_workspace_z32_d1024_blocks3_private_p20_private_s8_seed44_20260829`; checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`; evaluation `eval_validation/epoch_3`.

### 2026-08-30 - pathvqa_lora_full_model_attention_r8_seed44_20260830

- Dataset and split: PathVQA train19,654; fixed full validation6,259 with832 image clusters. Only epoch3 Validation was evaluated under the current protocol; Test was not run.
- Seed / data seed:44 /42.
- Controlled method: full-model Attention LoRA-r8 with alpha16, dropout0.05 and LR1e-4. It targets qkv/proj in all24 visual Attention layers and q/k/v/o in all36 LLM Self-Attention layers, for48 visual +144 language =192 selected Linear modules. Micro-batch1 with gradient accumulation32 gives effective batch32; training lasts3 epochs.
- Fixed epoch3 Validation Overall: **59.3386**; image-clustered standalone95% CI **[57.8839,60.7231]**.
- Yes/No / Free-form: **92.1920 /26.5795**; Free-form question-type macro23.5619.
- Per question type: how12.4031, other21.4286, what19.7802, when7.6923, where75.3056, why4.7619, yes/no92.1920.
- Trainable parameters: expected and audited actual **7,077,888**, or0.159236% of4,444,893,696 total parameters.
- Training:3 epochs,1,845 optimizer steps, runtime14,827.47s, reported train loss0.588059.
- Paired effect of Full Workspace minus LoRA on the identical Validation questions: **+0.1118 Overall**, **-1.5680 Yes/No** and **+1.7869 Free-form**. Workspace-only/LoRA-only correct counts are329/322 Overall,103/152 Yes/No and226/170 Free-form; exact McNemar p-values are0.8141,0.00258 and0.00564. Image-clustered paired95% CIs are **[-0.7669,+0.9717] Overall**, **[-2.5633,-0.5853] Yes/No** and **[+0.2844,+3.2611] Free-form**. Their predictions are textually identical on4,137/6,259 questions (66.10%).
- Capability decomposition: Full Workspace exceeds LoRA by+3.2575 on `what` but trails by-4.6455 on `where`. Within Yes/No, Workspace is+2.9206 on `yes` references and-7.0064 on `no` references, so LoRA's aggregate binary advantage mainly reflects markedly better negative-class calibration rather than a uniform gain over both labels.
- Evaluation timing: TTFT mean0.057489s, weighted TPOT0.030869s/token,32.395 decode tokens/s, request mean0.131781s. Timing is descriptive only because the two methods were evaluated in different runtime sessions.
- Conclusion: Full Workspace and full-model LoRA-r8 are statistically tied on Overall at seed44, but not functionally equivalent. LoRA is10.87x smaller and significantly stronger on Yes/No, while Full Workspace is significantly stronger on Free-form and `what`, which is the more relevant evidence for sample-conditioned multimodal coordination. This prevents an Overall-only superiority claim but establishes LoRA as a strong, genuinely matched performance baseline rather than evidence that the Workspace idea failed. Cross-dataset SLAKE validation now has higher priority than PathVQA seed45.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/lora/pathvqa_lora_full_model_attention_r8_seed44_20260830`; checkpoint `checkpoints/epoch_3`; training log `train.log`; evaluation `eval_validation/epoch_3` and `eval_validation_epoch_3.log`.

### 2026-08-30 - slake_dynamic_prompt_full_workspace_z32_d1024_blocks3_private_p20_private_s8_seed44_20260830

- Commit/config: commit `4a68586`; exact cross-dataset transfer of the PathVQA Full Workspace architecture. It keeps P20, private Text CA256/8 heads, independent visual S8, private Visual CA128/4 heads and single-pass visual injections at layers5/11/17. The added Workspace uses Z32x1024, three independent full-token Cross-Attention/Self-Attention/FFN4096 blocks, one Text reader CA1024/16 heads and three Visual reader CAs1024/16 heads. All architecture and optimizer hyperparameters were fixed before seeing SLAKE results.
- Dataset and split: SLAKE train.json for3 training epochs; official test.json with2,094 questions for the single fixed epoch3 evaluation. Test contains836 CLOSED /1,258 OPEN,267 KVQA /1,827 VQA,1,061 EN /1,033 ZH samples.
- Seed / data seed:44 /42.
- Controlled change: transfer the complete PathVQA Workspace method to SLAKE without per-dataset architecture or learning-rate tuning. Prompt/Dynamic/private-Sparse/Workspace learning rates remain0.3/3e-4/3e-5/1e-4; effective batch is32.
- Official Test Overall: **78.37** (1,641/2,094 correct).
- CLOSED / OPEN: **84.09 /74.56**.
- KVQA / VQA: **63.30 /80.57**.
- EN / ZH: **78.79 /77.93**.
- Trainable parameters: **76,896,256**. Exact decomposition is P20 51,200; private Text CA2,634,240; private Sparse Visual1,201,152; three Workspace update blocks50,396,160; three Workspace Visual readers12,598,272; Workspace Text reader7,349,760; text-to-Workspace projection/norm2,627,584; Workspace seeds/type/layer embeddings37,888.
- Training:3 epochs,924 optimizer steps, runtime5,195s, reported train loss3.335.
- Paired effect versus SLAKE Static Prompt Tuning seed44 74.40: **+3.9637 Overall** with145 Workspace-only and62 Static-only correct questions, exact McNemar `p=7.65e-9`. Effects are +3.4689 CLOSED (`p=0.00169`), +4.2925 OPEN (`p=1.64e-6`), +0.7491 KVQA (`p=0.856`), +4.4335 VQA (`p=9.30e-10`), +3.3930 EN (`p=0.000223`) and +4.5499 ZH (`p=1.38e-5`). The gain is robust but is concentrated in visual VQA rather than knowledge questions.
- Mechanism diagnostics over47 rows, steps0..920: Workspace grad remains active but declines0.6864 ->0.3625, while private Sparse grad rises0.1711 ->0.2005. Dynamic Prompt attention hardens and remains text-heavy in aggregate (last15 visual/text mass30.63%/69.38%). In contrast, Workspace full-token updates become almost visual-only: last15 visual mass at layers5/11/17 is95.22%/99.48%/99.85%. The Workspace Text-reader residual/base falls22.49% ->9.00%, and all Text/Visual readers remain nearly uniform over Z32 (normalized entropy about0.9997-0.9999). Workspace visual residual/private-Rep ends at0.558x/2.080x/1.038x by layer.
- Evaluation timing: TTFT mean0.073514s, weighted TPOT0.019453s/token,51.407 decode tokens/s, request mean0.121174s.
- Conclusion: cross-dataset performance generalization is established: the fixed Workspace significantly improves both PathVQA and SLAKE, especially OPEN/VQA questions. The current76.90M implementation is nevertheless not an acceptable final efficiency point, and the intended multimodal-workspace interpretation remains unsupported on SLAKE because full-token Workspace updates nearly ignore text and downstream slot readers are uniform. Preserve this checkpoint as the high-capacity teacher/reference. Before any width or slot sweep, use same-checkpoint reader interventions to identify indispensable branches; then prioritize recurrent sharing of the three update blocks and sharing of the three Visual readers. Modality-balanced visual/text attention is a separate performance/mechanism correction and must not be conflated with the first parameter-reduction control.
- Strategy correction (2026-08-30): parameter reduction is intentionally postponed. With approximately475 visual versus15 question tokens per batch sample, a joint softmax assigns about96.9% aggregate mass to vision even under equal logits, closely matching the observed95-99.8% visual mass. The next controlled experiment therefore keeps full capacity and replaces joint visual/text competition with separate modality softmax branches plus learned fusion. Weight sharing, reader removal and FFN reduction resume only after the highest-performing architecture is fixed.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_dynamic_prompt_full_workspace_z32_d1024_blocks3_private_p20_private_s8_seed44_20260830`; checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`; evaluation `eval_test/epoch_3` and `eval_test_epoch_3.log`.

### 2026-08-30 - slake_dynamic_prompt_full_workspace_17only_z32_d1024_blocks1_private_p20_private_s8_seed44_20260830

- Commit/config: commit `d2641a3`; retrained layer ablation of the exact SLAKE Full Workspace method. Private visual injection, Workspace update and Workspace visual readout are all retained only at visual layer17; layers5/11 and their independent Workspace blocks/readers are absent. P20, private Text CA256/8 heads, independent visual S8, private Visual CA128/4 heads, Z32x1024, FFN4096, Workspace Text reader and all learning rates remain unchanged.
- Dataset and split: SLAKE train.json for3 epochs; official test.json with2,094 questions evaluated once at epoch3.
- Seed / data seed:44 /42.
- Controlled change: visual anchor layers `[5,11,17] -> [17]`, which also changes Workspace blocks/readers `3 -> 1`; no other architecture, optimizer, data-order or evaluation change.
- Official Test Overall: **78.37** (1,641/2,094 correct), exactly equal to the three-layer reference.
- CLOSED / OPEN: **84.57 /74.24**.
- KVQA / VQA: **62.92 /80.62**.
- EN / ZH: **79.17 /77.54**.
- Trainable parameters: **34,895,872**, down42,000,384 / **54.62%** from76,896,256.
- Training:3 epochs,924 optimizer steps, runtime4,705.84s, reported train loss3.2140. Runtime is489.16s /9.42% lower than the three-layer run's5,195s.
- Paired effect versus the exact three-layer Workspace: Overall difference **0.0000**, with70 layer17-only-only and70 three-layer-only correct questions, exact McNemar `p=1.0`. Normalized predictions are identical on1,834/2,094 questions. Subgroup differences are all nonsignificant: CLOSED+0.4785 (`p=0.683`), OPEN-0.3180 (`p=0.747`), KVQA-0.3745 (`p=1.0`), VQA+0.0547 (`p=1.0`), EN+0.3770 (`p=0.712`) and ZH-0.3872 (`p=0.728`).
- Diagnostics:47 rows through step920. First15 -> last15 Workspace grad remains healthy0.750 ->0.711. The single layer17 Workspace update becomes less visual-dominated over training but remains biased: visual/text mass97.02%/2.98% ->93.64%/6.36%, while normalized update entropy falls0.884 ->0.693. Workspace norm grows80.46 ->151.02 rather than reaching the three-layer final depth's approximately420. The layer17 Workspace visual residual/private-Rep ratio rises0.537 ->2.434. Its Z32 visual-reader entropy remains nearly uniform0.99979 ->0.99971. Crucially, Workspace Text residual/base rises0.194 ->0.293 instead of decaying to the three-layer run's approximately0.09.
- Conclusion: layers5/11 are **removable under retraining** at seed44, but this experiment does not establish that they are inactive inside the trained three-layer model. It jointly removes their Workspace updates and direct visual writes, while allowing layer17 to compensate: the last-window layer17 Workspace residual/private-Rep ratio rises from1.038 in the three-layer model to2.434 in layer17-only; the original layer11 ratio2.080 also proves that branch was not dead. The result therefore establishes functional redundancy/non-identifiability, not the causal reason for it. Candidate mechanisms are sequential overwrite of a single Z bank, nearly uniform Z32 readout, repeated visual-dominated updates from the same static question embeddings, and attenuation/compensation of early insert-strip writes. Same-checkpoint path interventions are required before rejecting multi-stage coordination.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_dynamic_prompt_full_workspace_17only_z32_d1024_blocks1_private_p20_private_s8_seed44_20260830`; checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`; evaluation `eval_test/epoch_3` and `eval_test_epoch_3.log`.

### 2026-08-30 - slake_full_workspace_path_interventions_seed44_20260830

- Commit/config: commit `0b3d5ec`; six inference-only interventions reuse the exact trained three-layer Full Workspace checkpoint and its fixed official SLAKE Test predictions. No weight, input, decoding setting or checkpoint changes. The normal reference is **78.3668** Overall from `slake_dynamic_prompt_full_workspace_z32_d1024_blocks3_private_p20_private_s8_seed44_20260830`.
- Dataset and split: official SLAKE test.json, 2,094 questions: 836 CLOSED / 1,258 OPEN, 267 KVQA / 1,827 VQA, 1,061 EN / 1,033 ZH.
- Seed / data seed: 44 / 42; inference interventions only, with no retraining.
- `visual_write_off_l5`: disable only the layer5 Workspace visual residual write after preserving its Z update and all later paths. Overall **78.5578**; CLOSED / OPEN **84.3301 / 74.7218**; KVQA / VQA **63.6704 / 80.7334**; EN / ZH **78.8878 / 78.2188**. Effect versus normal: **+0.1910**, intervention-only/reference-only correct **10/6**, exact McNemar `p=0.4545`.
- `visual_write_off_l11`: disable only the layer11 Workspace visual residual write. Overall **78.3190**; CLOSED / OPEN **84.3301 / 74.3243**; KVQA / VQA **63.6704 / 80.4598**; EN / ZH **78.7936 / 77.8316**. Effect **-0.0478**, exclusive correct **19/20**, `p=1.0`.
- `visual_write_off_l5_l11`: disable both early direct Workspace visual writes while preserving both Z updates and the layer17 write. Overall **78.0802**; CLOSED / OPEN **84.3301 / 73.9269**; KVQA / VQA **63.2959 / 80.2408**; EN / ZH **78.5108 / 77.6379**. Effect **-0.2865**, exclusive correct **16/22**, `p=0.4177`.
- `workspace_update_off_l5`: bypass only the layer5 Workspace update while retaining the layer5 visual reader/write over the incoming Z and all later paths. Overall **78.5578**; CLOSED / OPEN **84.4498 / 74.6423**; KVQA / VQA **64.0449 / 80.6787**; EN / ZH **78.8878 / 78.2188**. Effect **+0.1910**, exclusive correct **7/3**, `p=0.3438`.
- `workspace_update_off_l11`: bypass only the layer11 Workspace update. Overall **78.5100**; CLOSED / OPEN **84.0909 / 74.8013**; KVQA / VQA **64.0449 / 80.6240**; EN / ZH **78.6993 / 78.3156**. Effect **+0.1433**, exclusive correct **8/5**, `p=0.5811`.
- `workspace_update_off_l5_l11`: bypass both early Z updates while preserving the direct visual readers/writes and the complete layer17 Workspace path. Overall **78.4145**; CLOSED / OPEN **84.8086 / 74.1653**; KVQA / VQA **63.2959 / 80.6240**; EN / ZH **78.5108 / 78.3156**. Effect **+0.0478**, exclusive correct **20/19**, `p=1.0`.
- Diagnostic scale intervention: normal-like visual-write interventions retain Workspace text-memory norm near **493**. Bypassing layer11, layer5 or both early updates changes that norm to **328.26 / 349.81 / 180.78**, respectively, while Overall remains statistically unchanged. Workspace Text-reader residual/base is **0.0892** for normal-like runs and **0.0896 / 0.0788 / 0.0711** for the three update-bypass runs. Its Z32 attention remains effectively uniform (`entropy_norm` approximately 1.0).
- Diagnostic limitation: `workspace_debug_means` captured only final Text-reader metrics. The intended per-layer `workspace_transition_*` LayerNorm-cosine and visual-delta cosine metrics are absent from the saved comparison, so this suite cannot directly distinguish repeated information from overwritten or cancelling layer updates. This logging gap does not affect the intervention configurations, predictions or paired scores.
- Conclusion: the layer5 and layer11 **Z updates are causally dispensable in the already-trained three-layer checkpoint**, not merely removable after retraining. They strongly alter Workspace magnitude/history but add no identifiable accuracy, consistent with layer17 overwrite plus scale-insensitive LayerNorm and near-uniform final Z readout. The two early direct visual writes are also individually dispensable; disabling both produces only a small nonsignificant `-0.2865`, leaving weak joint contribution/co-adaptation possible but unproven. Do not claim the modules are numerically dead: claim that their learned updates/writes contain no indispensable task information under these controlled interventions.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_full_workspace_path_interventions_seed44_20260830`; manifest `intervention_manifest.tsv`; per-variant predictions/summaries under the six named subdirectories; aggregate paired results `workspace_path_comparison.tsv` and `workspace_path_comparison.json`.

### 2026-08-30 - slake_dynamic_prompt_full_workspace_17only_z32_d1024_blocks1_private_p20_private_s20_seed44_20260830

- Commit/config: commit `7b3f00c`; exact layer17-only Full Workspace configuration with the sole controlled change `private visual S / inserted Rep tokens 8 -> 20`. Workspace remains Z32x1024 with one full-token CA/Self-Attention/FFN4096 Block, P20 and both text readers are unchanged, and all learning rates/data order remain matched.
- Dataset and split: SLAKE train.json for3 epochs; official test.json with2,094 questions evaluated once at epoch3.
- Seed / data seed:44 /42.
- Official Test Overall: **78.22** (1,638/2,094 correct).
- CLOSED / OPEN: **82.78 /75.20**.
- KVQA / VQA: **62.92 /80.46**.
- EN / ZH: **79.17 /77.25**.
- Trainable parameters: **34,908,160**, exactly12,288 above S8 because only12 additional1024-dimensional private Rep parameters are introduced.
- Training:3 epochs,924 optimizer steps, runtime4,696s, reported train loss3.193.
- Paired effect versus exact S8+Z32 layer17-only reference: **-0.1433 Overall**, with77 S20-only and80 S8-only correct questions, exact McNemar `p=0.8732`. CLOSED changes84.57 ->82.78 (**-1.7943**), exclusive correct21/36, `p=0.0627`; OPEN changes74.24 ->75.20 (**+0.9539**), exclusive correct56/44, `p=0.2713`. KVQA has identical168 correct with12/12 churn; VQA is-0.1642 with65/68, `p=0.8624`; EN has identical840 correct with31/31 churn; ZH is-0.2904 with46/49, `p=0.8376`.
- Mechanism comparison, first15 -> last15 diagnostic means. S20 private Rep/input grows0.727 ->1.829 versus S8 0.407 ->1.418, but frozen Visual Block output/input remains effectively identical at1.163 versus1.164. Private Visual CA entropy hardens much further0.497 ->0.165 versus S8 0.775 ->0.361, while its visual-memory mass falls31.65% ->27.00% versus S8 43.15% ->42.67%; the extra queries become more selective but more text-dominated. Workspace visual residual/private-Rep grows1.073 ->2.931 versus S8 0.537 ->2.434, while its Z32 read entropy remains uniform at0.9997. Sparse late gradient remains healthy0.0631 versus S8 0.0668, but Workspace late gradient falls0.711 ->0.442 and Workspace Text residual/base falls from the S8 last-window0.2928 to S20 0.1402.
- Evaluation timing: TTFT mean0.067927s, weighted TPOT0.020225s/token,49.444 decode tokens/s, request mean0.117485s. Timing is descriptive because runs were evaluated in separate runtime sessions.
- Conclusion: reject the hypothesis that8 private visual Rep positions are the current expression-capacity bottleneck. Expanding to20 makes the visual path stronger and more selective and changes157 question outcomes, so the branch is neither gradient-dead nor functionally inert. It nevertheless produces no net accuracy and shifts capability from CLOSED toward OPEN while suppressing the Workspace Text contribution. The supported diagnosis is **active but non-complementary visual writing / cross-branch compensation**, not “the visual side does nothing.” Per the pre-registered stop rule, do not run the combined S20+Z8 training experiment. Test final visual and text path necessity with same-checkpoint inference interventions before another retraining change.
- Output/log path: `/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/dynamic_prompt/slake_dynamic_prompt_full_workspace_17only_z32_d1024_blocks1_private_p20_private_s20_seed44_20260830`; checkpoint `checkpoints/epoch_3`; diagnostics `dynamic_prompt_diagnostics.jsonl`; evaluation `eval_test/epoch_3`; paired comparison `eval_test/workspace_path_comparison.json` and `.tsv`.

### PathVQA Yes/No Class-Balance Audit

Correction (2026-08-29): the two `shared_s_memory_main_merger` experiments above did not test the intended raw shared-S architecture. Their implementation captured the sample-conditioned Rep tokens after visual block17, mapped those outputs through `visual.merger`, and fed the result back as a third Text-CA Memory. They therefore reject only a final-anchor Rep feedback shortcut. They do not reject mapping the original, unconditioned shared parameter bank `S` into the LLM prompt space. Two corrected seed44 experiments are pending: (1) retain independent P20 and add a separate zero-initialized CA128 residual over `visual.merger(raw S)` while leaving the winning visual/question Dynamic CA unchanged; (2) remove independent P/P0 entirely and use the two native `visual.merger(raw S)` outputs directly as the prompt and visual/question CA queries. Neither corrected method reads any Rep output from layers5/11/17.

| Method | Yes count | Yes accuracy | No count | No accuracy | Class gap |
|---|---:|---:|---:|---:|---:|
| Frozen Base | 1,816 | 53.30 | 1,546 | 83.83 | 30.53 |
| Earlier 128-slot MMRL seed44 | 1,816 | 80.62 | 1,546 | 84.86 | 4.24 |
| Minimal MMRL Relation0 seed44 | 1,816 | 69.11 | 1,546 | 90.88 | 21.77 |
| Minimal MMRL Relation0.05 seed44 | 1,816 | 79.68 | 1,546 | 86.03 | 6.35 |
| Visual LoRA-r128 seed44 | 1,816 | 83.65 | 1,546 | 89.39 | 5.74 |
| Static Prompt Tuning seed44 | 1,816 | 91.30 | 1,546 | 89.20 | 2.10 |
| Dynamic Multimodal Prompt seed44 | 1,816 | 93.72 | 1,546 | 85.25 | 8.47 |

The aggregate Yes/No score hides a severe Base bias toward `no`. MMRL removes most of this imbalance and therefore does not obtain82.57 by collapsing to one class. It nevertheless trails the all24-layer LoRA run by3.03 points on `yes` and4.53 points on `no`. Same-question paired exact McNemar tests reject a tie for this unmatched comparison: on `yes`, MMRL-only/LoRA-only correct counts are127/182 (`p=0.0021`); on `no`,64/134 (`p=7.29e-7`). These statistics remain valid for the completed models but cannot establish a scope-matched architectural advantage because MMRL acts on visual layers17-24 while this LoRA acts on all24 layers.

Correction (2026-08-27): the original `MMRL seed44` class row above refers to the earlier 128-slot47.30 run. The two minimal-MMRL rows were added from their own stored comparison files and are the correct class breakdowns for the clean Relation experiment. They show that Relation's binary gain is primarily calibration of affirmative answers rather than a uniform visual-accuracy increase.

## Rejected / Superseded Directions

- DeepStack residual injection: large regression to68.53.
- Mixed Cross-Relation and Cross-Relation-only: no improvement over the clean reference.
- Reverse assignment and all dynamic-query reversal variants: 61.70-70.96, all below reference.
- Competitive and dual-softmax visual assignment: attention remained nearly uniform.
- Balanced text-guided fusion: slot collapse and70.49.
- Alpha gate/Relation weighting: no stable multi-seed gain.
- Learned 128-slot Attention Pooling: no mean gain over Mean Pooling on SLAKE.

These results remain useful negative evidence and should not be rerun unless a new hypothesis specifically addresses the observed failure mechanism.

## Pending Experiments

1. Fair CA replacement: separately normalize Q, visual memory, and text memory before the equal-parameter Concat-MLP; run seed45 only.
2. Add a supplementary semantic metric or blinded manual sample for PathVQA Free-form answers while retaining normalized exact-match as the official primary metric.
3. Confirm the clean PathVQA Relation gain on one additional seed before making a cross-seed claim; the seed44 paired result is already statistically clear.
4. Final comparison table: Base VLM, Visual LoRA, Static Prompt Tuning, nearest visual prompt/adapter baseline, and MMRL.

## Update Template

```markdown
### YYYY-MM-DD - experiment_name

- Commit/config:
- Dataset and split:
- Seed / data seed:
- Controlled change:
- Overall:
- CLOSED / OPEN:
- KVQA / VQA:
- EN / ZH:
- Trainable parameters:
- Runtime:
- Diagnostics:
- Conclusion:
- Output/log path:
```
