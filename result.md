# Experiment Result Summary

Last updated: 2026-08-28

This file is the concise experiment memory shared by the user and Codex. The complete append-only record remains in `EXPERIMENT_RESULTS.md`.

## Recording Contract

- Add every completed experiment here in summarized form, including failed and negative results.
- Keep exact experiment names, full breakdowns, diagnostics, and output paths in `EXPERIMENT_RESULTS.md`.
- Do not silently remove historical conclusions. Add an explicit correction when later evidence changes an interpretation.
- Experiment scheduling and additions belong in `plan.md`; completed outcomes belong here and in `EXPERIMENT_RESULTS.md`.

## SLAKE Snapshot

| Method | Seeds | Overall | Summary |
|---|---|---:|---|
| Static Prompt Tuning, length 20 | 44 | **74.40** | Best SLAKE result; 51,200 trainable parameters; strongest gains are on KVQA and OPEN questions. |
| 128-slot Attention Pooling MMRL + Relation 0.05 | 44/45/46/47 | **72.94 +/- 0.66** | Scores: 73.93, 72.64, 72.54, 72.64. Strongest MMRL single run is 73.93. |
| Mean Pooling MMRL + Relation 0.05 | 44/45/46/47 | **72.98 +/- 0.49** | Scores: 72.97, 73.26, 73.40, 72.30. Statistically tied with 128-slot pooling and slightly less variable. |
| Alpha-probability Gate | 44/45/46/47 | **72.92 +/- 0.59** | Scores: 73.59, 72.92, 73.02, 72.16. Lower variance than Hard Concrete but no stable mean gain. |
| Hard Concrete Gate | 44/45/46/47 | **72.71 +/- 1.03** | Scores: 72.59, 71.30, 73.50, 73.45. Higher variance than continuous alpha. |
| Alpha-weighted Relation | 45/46/47 | **72.62 +/- 0.46** | Scores: 72.73, 72.11, 73.02. Did not improve uniform Relation weighting. |

## PathVQA Snapshot

| Method | Seed | Overall | Yes/No | Free-form | Summary |
|---|---:|---:|---:|---:|---|
| Dynamic Multimodal Prompt20, Mean Memory, CA256 | 44 | **56.54** | 89.83 | 23.21 | Strongest PathVQA point estimate; versus Static Prompt: +0.71 Overall and +1.94 Free-form, but the paired clustered Overall CI crosses zero. Near-zero attention entropy requires causal intervention before claiming sample-dependent routing. |
| Minimal MMRL + Relation0.05 + Static Prompt20 | 44 | **55.75** | 88.94 | 22.52 | Tied with Static Prompt alone (-0.08 Overall); gains1.25 Free-form but loses1.40 Yes/No, so MMRL changes the tradeoff without a net gain. |
| Static Prompt Tuning, length20 | 44 | **55.83** | 90.33 | 21.27 | Parameter-efficiency baseline with only51,200 parameters; balanced yes/no classes, but last8 LoRA remains2.14 points stronger on Free-form. |
| Visual Attention LoRA-r128, all24 layers | 44 | **55.45** | 86.29 | 24.58 | +20.69 Overall over Base, but broader in scope than last8-layer MMRL; treat as an upper bound rather than the formal matched baseline. |
| Visual Attention LoRA-r128, last8 layers | 44 | **53.85** | 84.24 | 23.41 | Scope-matched baseline; 6.291M parameters and only1.60 points below all24-layer LoRA. |
| Minimal shared-S Mean-Pooling MMRL + Relation0.05 | 44 | **49.41** | 82.60 | 16.18 | 7.928M parameters; clean shared-Stage1 comparison shows Relation adds+2.57 Overall. |
| Minimal MMRL + Relation0.05, fixed5x8 Rep RoPE | 44 | **47.67** | 81.02 | 14.27 | Exact position-only control; fixed spatial coordinates reduce all three metrics versus the shared-origin version. |
| Minimal shared-S Mean-Pooling MMRL, no Relation | 44 | **46.84** | 79.12 | 14.51 | Exact Relation-off control using the same Stage1 and initial MMRL state. |
| 128-slot MMRL + Relation0.05 | 44 | **47.30** | 82.57 | 11.97 | +12.53 Overall over Base, proving the method adapts, but Free-form learning is much weaker than LoRA. |
| Frozen Base | - | **34.77** | 67.34 | 2.14 | PathVQA open exact-match is extremely difficult for the original model; no training or checkpoint. |

## Core Ablations

| Controlled change | Result | Conclusion |
|---|---:|---|
| No Relation, seeds 44/45 | 72.83 / 72.92 | Relation did not show a decisive SLAKE gain under this comparison. |
| Independent layer-projector initialization, seeds 44/45 | 72.68 / 72.83 | Same initialization is not strongly supported by the available mean difference. |
| Static Query | 56.69 | Input-conditioned query generation is essential in the tested configuration. |
| Naive equal-parameter Concat-MLP | 69.25 | Clearly below shared CA, but the comparison is confounded by joint normalization and scale mismatch. |
| Cross-Relation only | 72.68 | Did not exceed the clean 73.93 reference. |
| Reverse Assignment | 70.96 | Diffuse and imbalanced assignment; below reference. |
| DeepStack residual injection | 68.53 | Large regression. |
| Dynamic-query reversal family | 61.70-67.57 | Initialization fixes and competitive/dual-softmax variants did not recover the reference. |
| Text-guided slots | 70.49-73.16 | Either remained diffuse or became diverse without improving task accuracy. |

## Gate and Reproducibility

- Original same-init seed44: 73.93.
- Same nominal seed/config reproduction: 69.29, confirming historical trajectory instability.
- Removing Gate/Stage 1 and forcing `G=1`: 71.97.
- Retaining Stage 1 and the complete Gate path with `open_full_ce`: seed44 73.93; four-seed mean 72.94 +/- 0.66.
- Therefore `No gate (G=1)` is not equivalent to keeping the full training path while opening the Stage 3 data flow.
- Continuous alpha is a more stable bottleneck than Hard Concrete, but it has not demonstrated a higher multi-seed mean.

## Current Evidence Boundary

- SLAKE mixes visual VQA, knowledge questions, and two languages. Prompt Tuning outperforming MMRL suggests that SLAKE is not a pure visual-specialization benchmark.
- Learned 128-slot pooling has no stable SLAKE advantage over Mean Pooling.
- Relation, pooling, and routing claims require confirmation on another public dataset before being generalized.
- PathVQA is the primary public dataset and SLAKE is the secondary robustness dataset. The matched last8 comparison is LoRA53.85 versus minimal MMRL+Relation49.41; frozen Base is34.77. LoRA leads by4.44 points, mainly through Free-form answers.
- The all24-layer LoRA score55.45 is only1.60 points above last8 LoRA53.85, showing that most visual-only adaptation value lies in the last8 visual layers under this setup.
- Relation0.05 beats Relation0 by2.57 Overall with an image-clustered paired 95% CI of[+1.72,+3.45]. Because both runs share the exact Stage1 checkpoint and initial MMRL state, this is the first clean evidence that Relation helps on PathVQA.
- On binary questions, minimal MMRL Relation0 is strongly `no`-biased (`yes`69.11 / `no`90.88). Relation0.05 shifts this to79.68 /86.03: `yes` gains10.57 while `no` loses4.85. Relation therefore acts partly as answer calibration, not as a uniform visual-recognition gain.
- Giving the40 shared Rep Tokens fixed5x8 spatial RoPE positions lowers49.41 to47.67 despite healthy gradients and the exact same Stage1 checkpoint. Rep Tokens are not aligned to fixed image regions, so shared-origin positioning remains the supported design.
- Static Prompt Tuning reaches55.83 with only51,200 parameters, beating minimal MMRL by6.42 and last8 LoRA by1.98 Overall. Its Yes/No score90.33 is balanced across `yes`91.30 and `no`89.20, while Free-form21.27 still trails last8 LoRA23.41. This shifts the main bottleneck and next architecture toward the LLM context interface.
- The decisive MMRL+Static-Prompt combination reaches55.75, effectively identical to Static Prompt55.83. Both branches have healthy nonzero gradients, but the combination trades1.40 Yes/No points for1.25 Free-form points and still trails last8 LoRA on Free-form by0.89. This rejects practical complementarity for the current internal-ViT MMRL and supports moving directly to a dynamic multimodal LLM Prompt interface.
- Dynamic Multimodal Prompt reaches56.54/89.83/23.21 and is the strongest PathVQA seed44 point estimate. Against Static Prompt it gains0.71 Overall and1.94 Free-form while losing0.51 Yes/No; the Free-form paired shift is significant (`p=0.0020`), but the image-clustered95% CI for the Overall difference is[-0.10,+1.55], so a stable aggregate gain is not yet established.
- Dynamic Prompt attention over the two pooled Memory tokens hardens almost completely: late normalized entropy is0.00010 and visual mass is fixed at44.375%, consistent with a fixed assignment of71/160 prompt-head pairs to vision and89/160 to text. Values remain sample-conditioned and the residual/base norm ratio averages0.475 in epoch3, so the branch is active; zero-residual, shuffled-Memory, and fixed-residual inference interventions are required to distinguish genuine conditional information from extra static capacity.
- The Overall tie hides substantial sample churn: versus Static Prompt, the combination fixes352 questions and breaks357, with a61.07 Prompt-or-combination oracle. MMRL therefore contributes distinct information, but an always-on shared-CE composition lacks sample-wise arbitration and lets binary calibration losses cancel Free-form gains.
- Separate binary classes confirm that MMRL is balanced rather than exploiting label bias: Yes/No accuracies are80.62/84.86, versus all24-layer LoRA83.65/89.39 and Base53.30/83.83. The all24-layer LoRA wins both classes, but this remains a broader-scope diagnostic rather than the formal matched comparison.
- PathVQA Free-form normalized exact-match is unusually harsh: Base reaches only2.14, and qualitative outputs can contain plausible pathology phrases that miss the single reference. Add semantic/error analysis as a supplement, not a replacement for the official metric.
- The in-house dataset will be reported as a single-seed internal application case and will not support the main statistical claims.
