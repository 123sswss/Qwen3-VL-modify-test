# MMRL Experiment Ledger

Last updated: 2026-08-27

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

### PathVQA Yes/No Class-Balance Audit

| Method | Yes count | Yes accuracy | No count | No accuracy | Class gap |
|---|---:|---:|---:|---:|---:|
| Frozen Base | 1,816 | 53.30 | 1,546 | 83.83 | 30.53 |
| Earlier 128-slot MMRL seed44 | 1,816 | 80.62 | 1,546 | 84.86 | 4.24 |
| Minimal MMRL Relation0 seed44 | 1,816 | 69.11 | 1,546 | 90.88 | 21.77 |
| Minimal MMRL Relation0.05 seed44 | 1,816 | 79.68 | 1,546 | 86.03 | 6.35 |
| Visual LoRA-r128 seed44 | 1,816 | 83.65 | 1,546 | 89.39 | 5.74 |
| Static Prompt Tuning seed44 | 1,816 | 91.30 | 1,546 | 89.20 | 2.10 |

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
