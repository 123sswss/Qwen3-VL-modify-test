# Experiment Result Summary

Last updated: 2026-08-31

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
| Full Workspace Dynamic Prompt | 44 | **78.37** | 76.90M parameters; 84.09 CLOSED /74.56 OPEN,63.30 KVQA /80.57 VQA,78.79 EN /77.93 ZH. Significant +3.96 over Static Prompt, but Workspace attention is nearly visual-only and parameter efficiency is not yet competitive. |
| Layer17-only Workspace | 44 | **78.37** | 34.90M parameters; exactly ties the three-layer model with70/70 paired exclusive correct (`p=1.0`) while removing54.62% of trainable parameters. This proves retrained removability, not that layers5/11 were inactive; layer17 grows stronger to compensate. |

## PathVQA Snapshot

| Method | Seed | Overall | Yes/No | Free-form | Summary |
|---|---:|---:|---:|---:|---|
| Dynamic Prompt + Sparse Visual, single-pass LR3e-5 | 44 | **57.79** | **89.77** | **25.77** | New best PathVQA point estimate. Standard one-pass `Strip(Block([Rep;h]))` exceeds the shared-S bridge by1.65 Overall and2.56 Free-form with clustered paired CI excluding zero. |
| Asymmetric Shared-S, text-write/visual-readonly [Validation] | 44 | **56.81** | 88.22 | 25.49 | Fixed epoch3 Validation only. Versus the exact57.52 Validation baseline: -0.70 Overall, -1.50 Yes/No, +0.10 Free-form; detached mapping works but makes all three Visual CA anchors text-dominated. |
| Raw Shared-S separate CA residual + P20 | 44 | **54.95** | 88.31 | 21.54 | Correct raw-S keep-P test. Active4% residual significantly loses2.84 Overall and4.23 Free-form; shared CE drives visual CA layers11/17 to96.5%/91.2% visual mass. |
| Raw Shared-S direct2-token Prompt, no P/P0 | 44 | **51.64** | 87.33 | 15.91 | Correct pure-sharing test. Dynamic residual grows to2.83x the low-LR2-token base and loses6.15 Overall; text-anchor capacity/LR is insufficient. |
| Dynamic Prompt + Sparse Visual, historical dual-path LR3e-5 | 44 | **57.51** | 89.62 | 25.35 | Previous numerical reference; paired gains over Dynamic Prompt were significant, but its checkpoint was deleted and its dual execution path is now superseded by the cleaner single-pass result. |
| Dynamic Multimodal Prompt20, Mean Memory, CA256 | 44 | **56.54** | 89.83 | 23.21 | Previous best; versus Static Prompt: +0.71 Overall and +1.94 Free-form. Causal interventions prove that matched sample Memory is necessary despite near-zero attention entropy. |
| Single-pass + direct shared-S Memory, trainable P | 44 | **56.14** | 89.02 | 23.21 | Active bridge receives healthy gradients and23.75% attention but loses1.65 Overall versus the exact baseline; reject the direct shortcut. |
| Dynamic Prompt + Sparse Visual, LR3e-4 | 44 | **56.02** | 88.88 | 23.12 | High-LR control: the active branch becomes dominated by layer11, suffers an epoch2 validation collapse and trails Dynamic Prompt. Superseded by the controlled LR3e-5 result. |
| Minimal MMRL + Relation0.05 + Static Prompt20 | 44 | **55.75** | 88.94 | 22.52 | Tied with Static Prompt alone (-0.08 Overall); gains1.25 Free-form but loses1.40 Yes/No, so MMRL changes the tradeoff without a net gain. |
| Static Prompt Tuning, length20 | 44 | **55.83** | 90.33 | 21.27 | Parameter-efficiency baseline with only51,200 parameters; balanced yes/no classes, but last8 LoRA remains2.14 points stronger on Free-form. |
| Shared-S Memory with frozen P0 | 44 | **54.71** | 87.45 | 21.92 | Freezing only P loses1.43 Overall versus keep-P and makes Dynamic delta/base explode to76.8x; independent trainable P is necessary. |
| Visual Attention LoRA-r128, all24 layers | 44 | **55.45** | 86.29 | 24.58 | +20.69 Overall over Base, but broader in scope than last8-layer MMRL; treat as an upper bound rather than the formal matched baseline. |
| Visual Attention LoRA-r128, last8 layers | 44 | **53.85** | 84.24 | 23.41 | Scope-matched baseline; 6.291M parameters and only1.60 points below all24-layer LoRA. |
| Minimal shared-S Mean-Pooling MMRL + Relation0.05 | 44 | **49.41** | 82.60 | 16.18 | 7.928M parameters; clean shared-Stage1 comparison shows Relation adds+2.57 Overall. |
| Minimal MMRL + Relation0.05, fixed5x8 Rep RoPE | 44 | **47.67** | 81.02 | 14.27 | Exact position-only control; fixed spatial coordinates reduce all three metrics versus the shared-origin version. |
| Minimal shared-S Mean-Pooling MMRL, no Relation | 44 | **46.84** | 79.12 | 14.51 | Exact Relation-off control using the same Stage1 and initial MMRL state. |
| 128-slot MMRL + Relation0.05 | 44 | **47.30** | 82.57 | 11.97 | +12.53 Overall over Base, proving the method adapts, but Free-form learning is much weaker than LoRA. |
| Frozen Base | - | **34.77** | 67.34 | 2.14 | PathVQA open exact-match is extremely difficult for the original model; no training or checkpoint. |

## Dynamic Prompt Causal Interventions

| Same checkpoint inference | Overall | Yes/No | Free-form | Effect versus normal |
|---|---:|---:|---:|---|
| Normal matched Memory | **56.54** | **89.83** | **23.21** | Reference |
| Zero Dynamic residual | 49.29 | 87.18 | 11.35 | -7.25 Overall; residual branch is necessary |
| Mean residual after32-sample calibration | 43.19 | 79.80 | 6.52 | -13.35 Overall; a fixed static offset is actively harmful |
| Lag32 mismatched Memory | 48.58 | 84.18 | 12.93 | -7.96 Overall; matched sample Memory is causally necessary |

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
- Dynamic Multimodal Prompt reaches56.54/89.83/23.21 and was the strongest PathVQA seed44 point estimate before Sparse-LR stabilization. Against Static Prompt it gains0.71 Overall and1.94 Free-form while losing0.51 Yes/No; the Free-form paired shift is significant (`p=0.0020`), but the image-clustered95% CI for the Overall difference is[-0.10,+1.55].
- Pre-intervention concern: Dynamic Prompt attention over the two pooled Memory tokens hardens almost completely, with late normalized entropy0.00010 and visual mass fixed at44.375%, consistent with71/160 prompt-head pairs assigned to vision and89/160 to text.
- Causal interventions resolve that concern. Normal matched Memory beats zero residual by+7.25 Overall and lag32 mismatched Memory by+7.96, with clustered paired95% CIs[+6.34,+8.18] and[+6.99,+8.89]. A fixed mean residual falls to43.19. The method is genuinely sample-conditioned even though attention weights are nearly fixed: dynamic information is carried by current-sample Value vectors and residual direction rather than variable attention routing.
- Sparse visual layers5/11/17 with8 Rep tokens, CA128, initial residual scale0.05 and Relation0.05 reaches56.02/88.88/23.12. The branch is active and layer11 alone reaches a3.41% residual ratio, but it lowers Dynamic Prompt by0.52 Overall and0.95 Yes/No while leaving Free-form nearly unchanged. Its Free-form type macro rises to28.63 from22.40 through small how/other/when categories, which is insufficient to support the extension.
- Correction (2026-08-28): “stop this extension” was too categorical. The same run beats the original Dynamic Prompt at epoch1 by+2.01 Overall and+3.29 Free-form, then collapses in epoch2 and recovers in epoch3. Treat it as a positive architectural signal with unstable joint optimization; permit one Sparse-LR-only diagnostic at3e-5 before deciding whether to stop.
- The Sparse-LR diagnostic succeeds: validation becomes monotonic54.53 ->55.14 ->56.85 and test reaches57.51/89.62/25.35. Versus the original Dynamic Prompt, paired clustered CIs are[+0.20,+1.73] Overall and[+0.84,+3.44] Free-form; versus the exact3e-4 control they are[+0.62,+2.31] and[+0.87,+3.61]. An over-fast visual branch destabilizes joint CE optimization, but the final0.073% Sparse residual requires a same-checkpoint Sparse-off intervention before attributing the gain to inference-time visual correction rather than training-time trajectory regularization.
- Storage note: the two completed dual-path Sparse Visual checkpoints were explicitly deleted on2026-08-28; logs and prediction-level evidence remain. Its57.51 score was the numerical reference at deletion but is now superseded by the reproducible single-pass57.79 checkpoint.
- The corrected single-pass Sparse Visual baseline reaches57.79/89.77/25.77 with3.887M trainable parameters and monotonic validation55.06 ->56.21 ->57.52. It beats the direct shared-S bridge by+1.65 Overall and+2.56 Free-form; the paired clustered Overall CI[+0.92,+2.43] excludes zero. Standard one-pass Rep injection is therefore the supported PathVQA architecture and supersedes the deleted dual-path57.51 numerical reference.
- Mapping the final-anchor Rep through the frozen main Visual Merger into a third Text CA S-Memory is actively harmful despite healthy gradients, normal residual scale and23.75% stable attention mass:56.14/89.02/23.21. The direct cross-branch shortcut competes with stronger native visual/question Memories rather than solving branch conflict.
- Freezing independent P0 in the shared-S model falls further to54.71/87.45/21.92. Relative to keep-P, the clustered paired Overall CI is[-2.25,-0.65], and Dynamic delta/base grows to76.8x. The20 trainable Prompt tokens are a necessary domain anchor, not redundant Query scaffolding; do not pursue pure shared-S replacement under this design.
- Correction (2026-08-29): the preceding two shared-S conclusions apply only to the mistakenly implemented post-layer17 Rep feedback shortcut. That path had already passed through Visual CA and block17 before returning to Text CA; it was not `raw S -> frozen visual.merger -> text prompt`. The raw shared-S hypothesis remains untested. Two corrected seed44 runs are scheduled, and neither will export a conditioned Rep from any visual layer.
- The corrected raw-S tests are now complete. Keeping P20 and adding a separate raw-S CA reaches54.95/88.31/21.54, significantly below the57.79 baseline with clustered paired Overall CI[-3.67,-2.04]. The added residual is only3.95% of P, while layers11/17 move toward visual-only attention. This rules out branch death and strongly suggests cross-branch optimization interference, but only a text-side stop-gradient control can separate backward-gradient conflict from harmful forward residual content.
- Replacing P entirely with the two native `visual.merger(raw S)` tokens reaches51.64/87.33/15.91. Its Dynamic residual is2.83x the base while S remains in the3e-5 visual optimizer group; this confirms that a low-LR2-token shared bank cannot replace the independently high-LR textual domain anchor. Do not spend another full run on raw-S hard sharing.
- The asymmetric text-owned Shared-S replacement reaches56.81/88.22/25.49 on fixed epoch3 Validation, versus the exact single-pass baseline57.52/89.73/25.40. Gradient isolation and the adapter are both healthy, and Free-form is tied, but Yes/No loses1.50 points (`p=0.00148`): `yes` gains0.35 while `no` loses3.75. Visual CA is text-dominated from the first training window and ends at29%/27%/29% visual mass versus baseline68%/50%/61%, while Rep strength rises sharply and Token-Mixer entropy barely moves. This is forward semantic over-coupling and affirmative-prior amplification rather than backward gradient pollution. Stop this pure replacement form; only a bounded detached-S residual on top of an independent visual basis remains technically justified.
- The Overall tie hides substantial sample churn: versus Static Prompt, the combination fixes352 questions and breaks357, with a61.07 Prompt-or-combination oracle. MMRL therefore contributes distinct information, but an always-on shared-CE composition lacks sample-wise arbitration and lets binary calibration losses cancel Free-form gains.
- Separate binary classes confirm that MMRL is balanced rather than exploiting label bias: Yes/No accuracies are80.62/84.86, versus all24-layer LoRA83.65/89.39 and Base53.30/83.83. The all24-layer LoRA wins both classes, but this remains a broader-scope diagnostic rather than the formal matched comparison.
- PathVQA Free-form normalized exact-match is unusually harsh: Base reaches only2.14, and qualitative outputs can contain plausible pathology phrases that miss the single reference. Add semantic/error analysis as a supplement, not a replacement for the official metric.
- The in-house dataset will be reported as a single-seed internal application case and will not support the main statistical claims.
- 2026-08-30 PathVQA Full Workspace seed44：`Z32x1024 + 3个全Token CA/Self-Attention/FFN4096 Block + 独立文本/视觉共享残差头`，保留P20、Text CA256、私有视觉S8/CA128和层5/11/17单路径注入；76,896,256参数。固定epoch3 Validation **59.4504/90.6240/28.3663**，相对57.5172基线为**+1.9332 Overall、+0.8960 Yes/No、+2.9675 Free-form**；Overall聚类配对95% CI[+1.1747,+2.7132]，Free-form CI[+1.8602,+4.1721]。共享视觉残差末段约为私有Rep的5.25倍且32槽下游读取近似均匀，因此先跑seed45复现与同checkpoint分支缩放干预，再决定如何降参。
- 2026-08-30 PathVQA全模型Attention LoRA-r8 seed44：7,077,888参数，固定epoch3 Validation **59.3386/92.1920/26.5795**。与Full Workspace的Overall差仅+0.1118且配对p=0.814、CI[-0.7669,+0.9717]，统计上完全打平；但Workspace显著领先Free-form **+1.7869**，LoRA显著领先Yes/No **+1.5680**。LoRA参数仅为Workspace的9.20%，因此Workspace不能主张Overall或参数效率胜出，但其开放题优势为跨模态共享工作区保留了明确价值；下一优先级改为SLAKE seed44泛化验证，而非先跑PathVQA seed45。
- 2026-08-30 SLAKE Full Workspace seed44：固定PathVQA架构直接迁移后官方Test为**78.37**，相对Static Prompt74.40显著提升**+3.96**（p=7.65e-9），OPEN +4.29、VQA +4.43、中文+4.55，证明跨数据集性能泛化成立。但76.90M参数中50.40M来自三套Workspace Block、12.60M来自三套视觉读头、7.35M来自文本读头；Workspace在层5/11/17末段的视觉注意力达95.2%/99.5%/99.8%，下游读取Z32仍近乎均匀。当前应视为高容量教师/上界，不是最终方法；先做同checkpoint分支干预，再共享三层Block和视觉读头，最后才缩FFN/宽度或修正模态失衡。
- SLAKE三层Workspace同checkpoint路径分解已完成。单独关闭层5/11视觉写回为+0.19/-0.05，同时关闭为-0.29且均不显著；旁路层5/11 Z更新为+0.19/+0.14，同时旁路为+0.05。最关键的是，双更新旁路令Workspace Memory范数约493降至181而总分仍为78.41，证明早期Z更新只改变数值轨迹、未提供不可替代任务信息。当前证据支持以layer17-only为主结构起点，但两次早期视觉写回仍可能存在很弱的联合协作，不能声称模块完全死亡。预定跨层余弦诊断未被保存，只得到Text-reader诊断；在修复捕获链路前不得为补日志盲目重跑。
- SLAKE layer17-only `S20+Z32` seed44为**78.22**，相对`S8+Z32` 78.37是-0.14，逐题77/80互换、`p=0.873`。CLOSED下降1.79而OPEN上升0.95。视觉支路不是没工作：Rep/input由S8末窗1.42升至S20 1.83，视觉CA熵由0.36降至0.16，Workspace视觉残差/私有Rep由2.43升至2.93，并改写157道题；但Visual Block输出范数保持不变，额外查询更偏文本且Workspace Text残差被压低。结论是8个Rep不是容量瓶颈，视觉写入活跃但与文本支路不互补；取消S20+Z8组合训练，先做同checkpoint最终视觉/文本路径关闭。
- 2026-08-30 SLAKE Layer17-only Workspace seed44：仅保留第17层的私有视觉注入、Workspace更新Block和视觉读头，其他配置不变；参数由76.90M降至**34.90M**，官方Test仍为**78.37**。相对三层版本逐题互有70题独占答对，McNemar `p=1.0`，全部子组差异均不显著；训练时间减少9.42%。但第17层共享残差/私有Rep由三层版末窗1.04x升至2.43x，而原第11层为2.08x，说明17-only发生了明显补偿。当前结论是多层参数化可替代，不是层5/11无效。
- SLAKE layer17-only最终路径干预：关闭Workspace视觉写回为**78.08**（-0.29，`p=0.471`），关闭全部视觉Rep写回为**77.84**（-0.53，`p=0.108`）；关闭Workspace文本写回则显著降至**76.36**（-2.01，`p=0.000894`），同时关闭Workspace视觉/文本写回为**76.22**（-2.15，`p=0.000410`）。当前不可替代的输出路径是`Z -> Workspace Text Reader -> LLM Prompt`，不是视觉Block内的Rep写回；但Z仍读取视觉Token，因此尚不能说视觉信息无效，下一步应只切断或错配Workspace更新的视觉Memory。
- SLAKE Directional Concat Workspace seed44：单层text-Q/visual-KV CA得到Z10，分别经零初始化MLP生成显式concat的视觉10-token与LLM10-token块，总参数**12.73M**。官方Test **77.17**，相对34.90M layer17-only参考-1.19（77/102独占，`p=0.0725`），但相对Static Prompt显著+2.77（139/81，`p=0.000112`）。参数减少63.5%却无训练加速；Z槽余弦仍0.941、视觉更新/query为3.35x、文本delta/anchor为0.50，说明简化有效但共享槽仍趋同且两个写入过强。它是有价值的紧凑Pareto点，不是新的最高分主结构。
- SLAKE Directional Concat同checkpoint缩放：文本`MLP(Z)`降至0.5后为**76.03**（-1.15，26/50，`p=0.0079`），归零后为**74.26**（-2.91，38/99，`p=1.88e-7`），且OPEN/KVQA/中文分别较正常值下降4.29/3.37/3.48，证明完整文本动态写入是主要有效路径而非过强干扰。视觉`MLP(Z)`降至0.5为**77.22**、归零为**77.13**，相对77.17均仅正负0.05且`p=1.0`；即使视觉delta/anchor从28.17x归零也无净影响。结论：动态视觉写回可旁路，剩余性能差距在Z构造/槽多样性或文本变换能力，不在输出缩放。
- SLAKE Directional Concat视觉Memory错配控制：保持原图完整VLM输入不变，只将Directional CA的视觉K/V换成上一张不同图像，2,094/2,094正式样本全部审计通过。总分由**77.17降至76.41**，错配/正常独占正确10/26，McNemar `p=0.0113`；CLOSED/OPEN分别下降0.72/0.80。即便错配视觉Memory与原Memory平均余弦仍高达0.916，净损失仍显著，证明`当前图像视觉K/V -> Z -> LLM动态Prompt`存在真实但幅度有限的图像条件贡献；不能再把该方法解释为纯问题条件Prompt，也不能据此恢复已被否定的视觉动态写回。
- PathVQA Directional Text-Dynamic-Only seed44：删除无效的`Z -> visual dynamic MLP`但保留静态`S8+A_v10`视觉插入、P20/A_t10、问题Q10、层17完整视觉K/V及`Z -> LLM Prompt`，参数降至**10.63M**。固定epoch3 Validation为**59.5782/91.5520/27.6962**，相对7.08M全模型Attention LoRA-r8仅+0.2397 Overall且不显著（336/321，`p=0.585`，聚类配对CI[-0.600,+1.138]），因此只能主张统计持平；但它以少86.18%参数打平76.90M Full Workspace，并显著超过57.5172单路径基线+2.0610（382/253，`p=3.46e-7`，CI[+1.286,+2.891]），Yes/No和Free-form均显著提升。结论：数值止损线通过，新路线保留为强Pareto候选，但停止继续堆结构，下一步只做复现与问题/图像条件机制控制。
- PathVQA同checkpoint条件错配完成并通过全量审计。仅错配Directional CA的问题`Q10`时，59.5782降至**49.9441**（-9.6341，111/714，`p=1.37e-108`，图像聚类配对CI[-10.5451,-8.7203]），Free-form下降16.9751、where下降42.5428；仅错配视觉K/V时降至**52.5483**（-7.0299，151/591，`p=2.70e-62`，CI[-7.9508,-6.1191]），Yes/No下降5.28、Free-form下降8.7747。两项均保持原图、原问题和原生VLM路径不变，使用1条异源不计分prime后实现6,259/6,259正式样本错配。结论：当前方法确实依赖`当前问题Q -> 当前图像冻结特征K/V -> 动态LLM Prompt`的正确对齐，不是Static Prompt或单模态捷径；但错配比关闭更具破坏性，掉分不能直接当作各输入的独立贡献值。结合动态视觉写回归零无损，主结构应继续朝read-only vision / write-only language收缩。
- PathVQA Directional D512压缩 seed44：只将锁定结构的Directional宽度1024降为512，参数由10.63M降至**4.59M**（-56.79%，比LoRA-r8少35.13%），Validation为**58.6835/90.8480/26.6114**。相对D1024显著-0.8947（240/296，`p=0.0174`，图像聚类配对CI[-1.665,-0.127]），说明不是无损压缩；但相对LoRA-r8仅-0.6551且不显著（319/360，`p=0.1247`，CI[-1.542,+0.223]），Free-form几乎完全相同并显著领先`what`+1.84。诊断显示D512槽更趋同、视觉注意力更分散、CA更新和文本动态残差均变弱，属于容量损失而非训练坍缩。D1024保留为精度主模型，D512作为低于LoRA参数量的效率Pareto点，不立即追加D256/D768扫描。
- PathVQA Directional宽度消融已闭环。D768为**7.805M、59.5622/91.0400/28.1749**，相对D1024仅-0.016（265/266，`p=1.0`，CI[-0.763,+0.739]），以少26.54%参数实现完全等价Overall，并将`where`显著提高6.85；相对LoRA-r8 Overall统计持平，但Free-form显著+1.60、`what`显著+2.51。D256为**2.033M、57.1497/89.4720/24.9202**，相对D512显著-1.53、相对D1024显著-2.43，二分类与开放题同时下降。D256槽并未坍缩，真正失败信号是视觉注意力熵0.974近乎均匀、CA更新/query仅0.270和文本残差仅0.267。最终容量曲线为：D768-D1024平台、D512效率折中、D256性能悬崖；D768作为高性能主配置，D512作为低于LoRA参数量的效率配置，停止继续扫宽度。
- 2026-09-02 训练资源观测：PathVQA QDPT-D768 单卡训练的 AutoDL 面板峰值显存为 **25,349 MiB（24.75 GiB）**，截图时间10:58:46。该值是设备级外部监控峰值，不等同于PyTorch allocator峰值；等待用相同面板口径补充Full-Attention LoRA-r8。
- 2026-09-02 PathVQA QDPT-D768多seed复现：seed45为**58.6835/90.3360/27.1219**，seed46为**58.7634/92.2560/25.3669**；连同seed44的59.5622/91.0400/28.1749，三seed Overall为**59.0030 +/- 0.4859**，Yes/No为91.2107+/-0.9709，Free-form为26.8879+/-1.4190。seed44是偏高端点而非稳定中心；seed46体现二分类与开放题之间明显的随机种子取舍。复现并非白跑，它把主张修正为“稳定约59分、开放题方差较大”，后续必须与LoRA做同seed或多seed比较。
- 2026-09-02 PathVQA QDPT-D768 no-static-visual seed44：删除全部18个Layer17静态视觉Prompt（`S_v8+A_v10`）但保留问题Q10读取完整视觉K/V及Z到LLM动态Prompt，参数仅由7.805M降至7.787M；Validation为**58.4438/90.9440/26.0370**。相对同seed D768基线显著-1.1184（261/331，McNemar `p=0.00453`，图像聚类配对CI[-1.8938,-0.3484]）；Yes/No仅-0.096，但Free-form-2.138、where-12.469。结论：18个静态视觉Prompt整体有必要，主要补充开放/空间视觉证据；保留时应统一命名为一个18-token模块，不能再把S8和A_v10包装成两个功能不同的模块。
- 2026-09-03 PathVQA D768 direct-visual-Z concat三seed：用`[A_v10; Proj(Z10)]`取代`[S_v8; A_v10]`，seed44/45/46分别为**59.6741/57.9006/58.8273**，三seed Overall **58.8007 +/- 0.8870**，Yes/No **90.4747 +/- 0.4164**，Free-form **27.2176 +/- 1.3564**。相对原D768同seed均值Overall **-0.2023**，Yes/No -0.7360，Free-form +0.3297；参数由7.805M增至8.586M，Overall方差反而增大。诊断确认Z的视觉写入很强且Visual Block输出发生实质改变，但没有转化为稳定收益；最终结构保留轻量静态视觉校准，共享Z主要用于LLM侧动态Prompt，放弃直接视觉Z拼接。
- 论文机理结论：在当前冻结生成式MLLM中，问题条件的主要价值位于视觉编码后的证据检索与语言解释，而非视觉编码器内的特征重写。错配Q或视觉K/V会大幅掉分，证明当前问题与当前图像的条件化对齐必要；去掉静态视觉Prompt会显著损失开放/空间题，证明稳定的领域视觉校准有价值；但将同一条件Z强力写回Layer17只增加参数和seed波动而无稳定收益。因此最终采用“冻结视觉特征保留广泛证据，Q对视觉K/V做后置定向检索，Z只写入LLM Prompt”的read-only-vision/write-only-language原则；不外推为所有模型和任务都无需视觉条件写回。
- 2026-09-03 unified-V20 Layer17 seed44得到**56.43/89.44/23.52**，但该结果存在明确初始化混杂，不能解释为V20结构失败。V20范数仅0.6430->0.6498，旧S8/A10同样几乎不变，train loss也基本一致；真正差异在step0已经出现：纯问题query norm由5.464降至5.151，说明视觉Prompt从18行改成20行推进了全局RNG并重置了后续问题投影/Directional CA。层位实验暂停，必须先隔离模块初始化随机数，再重跑Layer17统一前缀控制。
- 2026-09-03 unified-V20 RNG控制复跑得到**58.5237/90.6880/26.4518**，较未隔离初始化版本回升2.0946 Overall，确认初始化漂移解释了原始跌幅的大部分；但相对严格同初始化的旧`S8@3e-5+A_v10@1e-4` seed44仍低**1.0385 Overall**（251/316，McNemar `p=0.00714`，聚类配对CI[-1.7844,-0.2075]），差距集中于Free-form -1.7230（`p=0.00502`），Yes/No -0.352不显著。结论修正为：初始化是主要混杂但不是全部原因；统一V20仍存在真实优化劣势，按停止规则最终恢复旧8+10双速率参数化，V20只保留为负消融。
- 2026-09-03 Layer18-only与共享Layer17+18+19均已完成3-epoch训练并保存完整checkpoint，但评估加载器误取`static_visual_prompt_tokens=0`而忽略正确的`private_visual_prompt_tokens=8`，导致推理前报错；这是统一V20支持引入的反序列化回归，不是模型失败。两个checkpoint原地补评估，禁止重训；当前尚无分数。
- 2026-09-04 PathVQA层位敏感性闭环：Layer18-only为**58.3959/91.3280/25.5584**，相对Layer17基线Overall -1.1663（239/312，`p=0.00213`，聚类配对CI[-1.9252,-0.4212]），损失集中于Free-form -2.6165。共享Layer17+18+19为**58.3799/91.4560/25.3989**，相对Layer17 Overall -1.1823（246/320，`p=0.00212`，CI[-1.9729,-0.3846]），Free-form -2.7760、where -8.5575。多层与Layer18总体完全打平（-0.016，290/291，`p=1.0`），且where再低5.379。最终固定Layer17-only，不再扫描层数；结论是当前架构存在显著经验层位敏感性，不宣称Layer17具有普遍理论唯一性。
- 2026-09-04 PathVQA Full-Attention LoRA-r8多seed复现：seed45为**59.2427/91.4560/27.1219**，seed46为**59.2107/91.5200/26.9943**；连同seed44后，LoRA三seed Overall为**59.2640 +/- 0.0666**，Yes/No为91.7227+/-0.4077，Free-form为26.8986+/-0.2836。QDPT-D768对应均值为59.0030/91.2107/26.8879，因此QDPT减LoRA仅为**-0.2610/-0.5120/-0.0107**，应表述为性能持平而非胜出；LoRA更稳定且参数少10.27%，QDPT则在已有同seed运行中训练约快2.29倍，并保留显式问题引导视觉证据路径。seed44的QDPT开放题优势没有稳定复现为多seed均值优势。
- 2026-09-04 PathVQA D768 learned-static-query控制：把当前问题生成的Q10替换为等量可学习静态Query后，Validation降至**57.1817/89.6000/24.8564**，相对同seed问题引导版Overall **-2.3806**（205/354，McNemar `p=3.05e-10`，图像簇配对CI[-3.1377,-1.6077]），Free-form -3.3184、where -12.2249。分支梯度和CA更新均正常，但末段槽余弦升至0.9336，表明静态Query学成强但同质的通用视觉摘要。结论：当前问题条件化Q是QDPT不可替代的核心机制，不是普通learned-query增加参数即可复现；该消融无需追加seed。
- 2026-09-04 SLAKE最终QDPT-D768三seed：seed44/45/46 Overall为**77.65/76.70/76.74**，三seed **77.03 +/- 0.54**；CLOSED 83.53+/-0.80、OPEN 72.71+/-0.36、KVQA 61.30+/-2.49、VQA 79.33+/-0.28。固定PathVQA架构与超参数直接迁移，三次均完整训练并只在epoch3评估官方Test，证明约77分的跨数据集泛化成立；波动主要来自KVQA。相对Static Prompt seed44高3.25，较76.90M Full Workspace seed44低0.72，但仅用7.805M参数。2026-09-05误启动的同配置seed44副本已人工中止，无有效结果且禁止续跑。
