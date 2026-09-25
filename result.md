# Experiment Result Summary

## 2026-09-13 电气评估口径更正

用户确认原评价将相同的18条缺图样本自动计对，并以972为分母。按954条有效样本重新计分，Static Prompt为663/954＝69.50%，CoCoOp-style为671/954＝70.34%，QDPT为681/954＝71.38%；QDPT分别高1.89、1.05个百分点。上述三个整数总数由原两位百分比与已确认的972分母重建，本轮未重新运行模型。LoRA以评估器输出`rank8: score=70.96 evaluated=954`为准，即677/954＝70.96%；旧70.69%及候选687/972属于抄录或统计口径混用。QDPT比LoRA多答对4题，高0.42个百分点。本次单种子结果的排序为Static Prompt < CoCoOp-style < LoRA < QDPT。以下历史记录保留，涉及原电气分数的结论以上述更正为准。

Last updated: 2026-09-06

This file is the concise experiment memory shared by the user and Codex. The complete append-only record remains in `EXPERIMENT_RESULTS.md`.

## Recording Contract

- Add every completed experiment here in summarized form, including failed and negative results.
- Keep exact experiment names, full breakdowns, diagnostics, and output paths in `EXPERIMENT_RESULTS.md`.
- Do not silently remove historical conclusions. Add an explicit correction when later evidence changes an interpretation.
- Experiment scheduling and additions belong in `plan.md`; completed outcomes belong here and in `EXPERIMENT_RESULTS.md`.

## Publication Scope Decision

- 2026-09-06：论文主问题固定为**冻结生成式MLLM中的问题引导动态Prompt适配**，主文用Static LLM Prompt、Static Visual Prompt、Dual Static Prompt、Image-conditioned Prompt和Learned-query Prompt等同范式方法建立对比，不再反复以LoRA为叙事中心。LoRA属于不同适配范式的强参考：正文只简要说明一次并指向附录，附录完整保留PathVQA/SLAKE多seed结果与效率数据；不得删除或模糊SLAKE上的明显差距。取消未运行的LoRA-r4/r16，有限算力优先补同范式Prompt基线和同领域文献表。

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
- 2026-09-05 SLAKE Full-Attention LoRA-r8三seed：seed44/45/46 Overall为**81.95/81.57/81.95**，均值**81.82 +/- 0.22**，以7.078M参数稳定超过QDPT-D768的77.03+/-0.54。QDPT同seed低4.30/4.87/5.21分，逐题McNemar均`p<2e-10`；均值差在CLOSED/OPEN/KVQA/VQA分别为-3.47/-5.67/-9.11/-4.16，最大短板是知识型与开放生成，不是单纯二分类校准。SLAKE明确否定“QDPT跨数据集打平LoRA”：QDPT参数还多10.27%，仅保留训练约快1.53x、TTFT较低和TPOT约快1.51x的实现效率优势，论文必须改写为机制与准确率-效率权衡，不能宣称普遍替代LoRA。
- 2026-09-07 DRAPE文献碰撞结论：DRAPE（arXiv:2605.10765）是当前最接近且完成度最高的Related Work。其“指令来源Query -> 当前视觉K/V -> Cross-Attention -> 实例级LLM Soft Prompt”与QDPT核心生成器高度重合，因此QDPT不得宣称首次提出跨模态动态Prompt、文本Query/视觉K-V或模态不对称思想。两者仍有实质边界：DRAPE面向多模态持续指令微调，训练共享视觉projector、每任务保存生成器并依赖CLIP路由和null-space保护；QDPT面向单领域医学VQA，冻结视觉编码器、visual merger/projector和LLM，读取Layer17内部视觉证据，以静态领域锚点和动态样本证据共同适配，并提供逐样本错配与视觉写回干预。正文将DRAPE作为首要技术近邻先肯定后区分，不把其CoIN/UCIT分数与PathVQA/SLAKE直接排名。
- DRAPE带来的实验修正：视觉K/V错配只能证明错误证据有害，不能严格证明视觉Cross-Attention优于纯问题条件Prompt，因此`question-only / w/o visual CA`由审稿后候补提升为必做重训。现有Static Prompt、learned static query、D256-D1024宽度曲线可直接对应DRAPE的核心消融；另使用现有checkpoint补充Prompt t-SNE、同图不同问题的Prompt-to-image注意力图和成功/失败案例。路由、null-space、BWT和持续学习数据集不适用于QDPT，不照搬。
- QDPT用途的最终表述：方法不是通过修改骨干补充新的医学知识，而是在完全冻结MLLM已编码的视觉信息中，根据当前问题检索相关证据，并把证据转换成LLM可消费的动态Prompt。Static Prompt只表达领域级先验，QDPT增加样本级证据；其价值由相对Static Prompt的增益、learned-query退化、图文错配、未来question-only控制和视觉写回关闭共同支撑，而不是由是否普遍击败LoRA定义。
- 2026-09-07 CoTBox-TTT文献结论：CoTBox-TTT（arXiv:2511.12446v1）是目前最贴近QDPT医学VQA任务与证据选择叙事的工作。它冻结VisCoT和回答VLM，仅以24-token Evidence Prompt和32-token Answer Prompt在每个测试样本上优化20轮，通过框定位、裁剪重编码与EMA Teacher改善VQA-RAD、SLAKE和PathVQA。它支持“冻结VLM可能已保留证据，适配重点在问题相关证据选择”的研究动机，也可反衬QDPT无需外部定位器、无需测试时反向传播和重复视觉编码的单次前向优势。
- CoTBox-TTT比较边界：其方法属于无标签Test-Time Training而非训练期条件Prompt生成；使用不同骨干，PathVQA/SLAKE开放题报告关键词Recall，SLAKE设置与当前全语言官方评估不一致，并且没有Overall、多seed、显著性或公开代码。因此它可以进入核心Related Work和独立原协议文献表，但不能进入QDPT同协议主结果表，也不计入3至4个可公平分数对比基线的数量。其cross-view公式实际是两个同视图Teacher-Student损失之和，论文叙事和公开涨幅在复现前只作参考。
- 2026-09-07 PathVQA question-only / w/o visual CA seed44：保持同初始化、Q10、P20/A_t10、Layer17静态视觉Prompt和文本写入头，只将`Z=Q+CA(Q,V,V)`改为`Z=Q`；Validation为**57.6290/89.7280/25.6222**。相对完整QDPT-D768 seed44显著下降**1.9332 Overall**（237/358，McNemar `p=7.98e-7`，图像簇配对95% CI[-2.7582,-1.1009]），Yes/No -1.3120、Free-form -2.5526，`where`-9.7800。剩余问题Prompt支路训练健康，视觉CA相关诊断严格为0。结论：问题条件Prompt本身有效，但正确视觉K/V读取额外提供稳定且尤其重要的空间证据；结合视觉K/V错配-7.03，现有证据形成“关闭会降、错配大降、正确对齐最佳”的机制闭环，保留Directional Visual CA且不追加该消融seed。
- 2026-09-07 Electrical QDPT-D768首次启动失败：CPU测试通过后，私有训练数据的8个JSON路径以tuple传给只识别list的公共数据管线，`normalize_json_paths()`在首个batch前报`os.fspath(tuple)`类型错误；无训练、checkpoint或分数，不能解释为模型失败。修复同时令`normalize_json_paths()`和`load_jsons()`支持list/tuple多路径输入并增加双JSON来源回归测试；保持原实验名与配置原样重跑。
- Electrical QDPT第二次启动仍在训练前失败：JSON tuple修复后，4个图片根目录tuple进入同样只识别list的`build_image_mapping()`，最终触发`os.path.join(tuple, image_file)`。仍无训练或有效结果。完整修复将图片根目录也统一支持list/tuple，并要求在再次启动前使用真实8个JSON和4个目录完成无模型路径解析预检。
## GRASP Reproduction Audit

- 2026-09-08：初版PathVQA GRASP近似复现为**45.0871 Overall /82.9120 Yes-No /7.3708 Free-form**，2.633M参数。分支有梯度但Entmax末段零权重率为0、归一化熵0.9798，退化为近均匀区域平均；3轮后损失仍下降且Prompt范数仅2.03增至2.08。事后代码审计确认Prompt被放在完整chat开头而非视觉Token段，查询也混入chat模板而非纯问题Token，因此该结果只记录为初版实现失败，不能作为GRASP公平基线或架构否定。当前容器没有SLAKE GRASP结果产物，不得误记为已完成。
## 2026-09-08 PathVQA QDPT-D768 10-epoch marathon

- 10-epoch线性调度下，epoch3-10 Overall依次为56.4467、57.2456、58.6675、58.7794、57.4852、57.1817、57.3894、57.1657；峰值出现在epoch6。
- epoch6峰值58.7794仍比原3-epoch seed44的59.5622低0.7828，epoch10低2.3965。训练loss继续下降而Validation不再改善，说明追加训练预算没有转化为泛化收益。
- 结论：保留原3-epoch线性调度，不继续QDPT长程训练，也不在当前收尾阶段做学习率扫参。曲线见`figures/qdpt_d768_marathon_curve.svg`。

## 2026-09-09 GRASP Prompt顺序错误记录

- 修正版查询但错误Prompt顺序的PathVQA GRASP为**40.7254 Overall /74.91 Yes-No /6.64 Free-form**。纯问题Token编码符合论文，但实现成了`[Visual, Prompt, Question]`；论文Eq.(6-7)明确要求`[Prompt, Visual, Question]`。因果LLM中两者不等价，本结果只作为工程负记录，不进入主表，也不能否定GRASP。下一次只修正Prompt到视觉段之前，其余配置不动。

## 2026-09-09 GRASP正式顺序近似复现

- `pathvqa_grasp_reimpl_paper_order_n4_h512_seed44`按论文顺序实现`[Prompt, Visual, Question]`后，PathVQA Validation为**39.7508 Overall /75.3600 Yes-No /4.2438 Free-form**，图像簇95% CI[38.5286,40.9129]。Entmax已经形成稀疏区域选择且原型/投影梯度正常，因此低分不是分支死亡或残留位置错误；单个空间原型混合Prompt在当前生成式医学VQA协议下表达能力不足，尤其无法支持开放回答。该结果可作为明确标注的独立统一协议复现，但不能外推否定原论文20-epoch遥感设置。当前项目停止GRASP调参和SLAKE迁移，转向CoCoOp-style与Q-Former-style经典动态Prompt基线。

## 2026-09-09 CoCoOp-style经典动态Prompt基线

- `pathvqa_cocoop_style_p20_h160_seed44_20260909`以冻结post-merger视觉Token均值、`2560->160->2560` Meta-Net和P20实现图像条件Prompt，共873,120参数。PathVQA Validation为**57.4053 Overall /89.8560 Yes-No /25.0479 Free-form**，图像簇95% CI[55.9473,58.7765]。它比Static Prompt epoch3约高2.54分，但比同seed QDPT-D768低2.16、比Sandwich低3.37；`where`仅57.70，对应QDPT72.62和Sandwich76.04。Meta-Net末段梯度稳定约0.999，动态偏置达到静态Prompt范数的18%-23%，排除分支死亡或被P20压制。结论是图像条件化本身有效且极具参数效率，但全局图像均值无法替代问题Q对视觉K/V的定向证据检索。

## 2026-09-10 QDPT因果位置与静态视觉基线

- Sandwich `[P20; Visual; Z10; Question]`为**60.7765**，显著优于全放视觉后`[Visual; P20; Z10; Question]`的**59.2587**和反向Sandwich `[Z10; Visual; P20; Question]`的**58.1243**。最有效的分工是静态领域先验在视觉前、问题条件化证据在视觉后且靠近问题；收益不是把任意Prompt放到视觉后即可获得。
- Static Visual Layer17 V20仅得**35.9482/68.7040/3.2865**，20,480参数；Dual Static V20+P20为**54.6253/88.3840/20.9636**，71,680参数，与Static LLM Prompt约54.87基本持平。视觉Prompt单独几乎不能完成生成式任务，双侧静态Prompt也不能复现QDPT增益，进一步确认关键贡献来自问题引导的视觉证据检索和动态LLM Prompt，而不是额外Prompt容量。

## 2026-09-10 QDPT-Lite R256输出头负结果

- `pathvqa_qdpt_lite_d768_r256_question_q10_l17_p20_s8_av10_sandwich_seed44_20260910`只把Dense Sandwich的文本输出头从`768->768->2560`压缩为`768->256->2560`，参数由7.805M降至6.101M（-21.84%），但Validation降至**57.4213/90.7840/24.1544**，较60.7765 Dense基线下降3.3552，远超1分止损线；`where`下降17.36。CA和动态残差保持活跃，失败是LLM空间表达瓶颈而非分支死亡。拒绝R256，不跑R160或更多seed，永久保留Dense D768 Sandwich为最终方法。
## 2026-09-10 Final Dense Sandwich seed sensitivity

- PathVQA Dense D768 Sandwich seed45 Validation is **57.2935/90.6880/23.9949**, versus seed44 **60.7765/92.7360/28.9087**, despite identical architecture,7,805,184 parameters, data seed42 and training protocol.
- The3.4830-point Overall gap is accompanied by an early and persistent lower-scale optimization path: final P20, text-anchor and Workspace norms are574.94/471.84/35.50 for seed45 versus767.82/544.87/46.51 for seed44. Seed45 also has more diffuse final visual attention entropy0.6467 versus0.5179 and roughly4.2x the final sparse-visual gradient, while mismatch audits remain zero.
- Treat this as initialization sensitivity of the soft-Prompt/Workspace optimization, not configuration drift or a dead branch. Use the completed three-seed mean +/- standard deviation as the primary paper result; label60.7765 only as the best seed and discuss sensitivity explicitly in Analysis/Limitations. Seed46 is pending in the final suite.

## 2026-09-10 Final Dense Sandwich suite complete

- PathVQA已全部完成。Validation seeds44/45/46为60.7765/57.2935/59.3865，三seed **59.1522 +/- 1.7528**；Test为60.4554/56.8983/59.2945，三seed **58.8827 +/- 1.8141**。Test与Validation保持相同seed排序，初始化敏感性主要体现在Free-form和`where`。
- SLAKE最终Sandwich seeds44/45/46 Test为**76.74/76.65/77.46**，三seed **76.95 +/- 0.44**；CLOSED/OPEN均值84.01/72.26，KVQA/VQA均值62.17/79.11，EN/ZH均值77.51/76.38。相对旧非Sandwich均值77.03几乎不变(-0.08)，说明因果位置收益集中在PathVQA，未迁移为SLAKE准确率增益。
- 最终套件8/8全部完成。PathVQA mean Validation与LoRA-r8约持平，但SLAKE仍低LoRA-r8均值4.87分且参数更多；论文必须表述为任务相关的准确率-适配范式权衡，不宣称普遍击败LoRA。
## 2026-09-11 Electrical Dense Sandwich seed47 interrupted

- 电气最终Dense D768 Sandwich seed47只运行到121/1875 steps（约6.5%），无epoch3 checkpoint、train report或私有holdout summary，因此没有可报告分数。已有loss与各分支诊断正常且无Python/CUDA traceback，暂记为外部中断，不解释为模型失败；服务器稳定后原配置重跑。
## 2026-09-11 Static Prompt P20多seed稳定性

- PathVQA Static Prompt P20 seeds44/45/46 Validation为54.8650/55.0567/55.3124，Overall **55.0780 +/- 0.2244**，range0.4474；Yes/No **88.9493 +/- 0.5509**，Free-form **21.3040 +/- 0.9117**。Overall未达到预注册的std0.5或range1.0触发线，因此不补CoCoOp seeds45/46。
- 结果不支持“Prompt方法普遍高度seed敏感”的强主张。Static Prompt aggregate稳定，QDPT Sandwich的1.7528 Overall std应诚实描述为当前复杂条件检索/Prompt路径的稳定性代价；稳定性表仍可用于对比LoRA、Static Prompt和QDPT，但不能替QDPT消解责任。
## 2026-09-11 CoCoOp-style多seed稳定性

- PathVQA CoCoOp-style P20/H160 seeds44/45/46 Validation为57.4053/56.3988/55.1366，Overall **56.3136 +/- 1.1367**，range2.2687；Yes/No **89.3547 +/- 0.5903**，Free-form **23.3674 +/- 1.6913**。最佳seed比均值高1.0917分。
- 同协议Overall std呈LoRA-r8 0.0666、Static Prompt0.2244、CoCoOp-style1.1367、QDPT Sandwich1.7528的递增梯度。结果支持“当前冻结生成式VLM中的条件动态Prompt比静态Prompt和LoRA更具初始化敏感性”，但不推广为所有Prompt方法的普遍定律。论文加入专门稳定性表，并继续以多seed均值而非最佳seed作为主结论。

## 2026-09-12 电气数据集最终对比

- 同一私有固定holdout、seed47/data seed42和三epoch协议下，Static Prompt P20、CoCoOp-style P20/H160、QDPT Dense D768 Sandwich分别为 **70.06/70.88/71.91**，均评估954条并跳过18条。QDPT比Static、CoCoOp和已有LoRA-r8 70.69分别高 **1.85/1.03/1.22**，但目前仅有总分，不能宣称统计显著。
- 该数据集以看图判别和封闭选择为主，恰好匹配QDPT“问题Q从冻结视觉K/V中检索证据，再生成LLM侧动态Prompt”的归纳偏置。应将结果解释为视觉证据选择型任务上的适配优势，而非QDPT普遍优于LoRA；封闭答案空间、答案先验和较低的语言生成/推理负担都可能放大该优势。结合PathVQA近似持平和SLAKE明显落后LoRA，论文结论固定为Prompt空间与权重空间适配具有不同任务偏好。

## 2026-09-13 PathVQA效率补全与统一V20 Sandwich

- PathVQA三seed Validation计时：最终QDPT Sandwich TTFT/TPOT均值为 **0.052451s/0.018487s-token**，Full-Attention LoRA-r8为 **0.057551s/0.030364s-token**；QDPT观测值分别低8.86%和39.12%。仅在确认硬件和软件条件匹配后作受控速度主张。
- RNG控制的统一V20 Sandwich seed44显示约 **55.70 Overall**，相对最终`S8+A_v10` Sandwich60.7765约低5.08，也比统一V20非-Sandwich58.5237低约2.82。Sandwich对双速率8+10有益、对统一V20反而有害，说明视觉Prompt参数化与LLM因果位置存在强优化交互；保留8+10。当前只掌握显示精度总分，精确分项和诊断待从服务器summary补齐。

## 2026-09-13 GRASP-Qwen第一次救援

- `pathvqa_grasp_qwen_adapted_embedding_init_dual_lr_n4_h512_seed44`将四个Prompt原型改为Qwen词嵌入行初始化，并使用Prompt LR0.3、路由投影LR1e-4；其余GRASP结构与三轮协议不变。用户报告PathVQA Validation约 **42.40 Overall**，比正式顺序复现39.7508高约2.65，但仍比Static Prompt54.8650低约12.47。由此确认Prompt尺度失配存在但不是主因；下一步固定该适配配置，优先隔离范数约35.8的二维位置编码是否压过真实视觉块语义。精确分项、诊断和路径待补。
- 第二步移除固定二维位置编码后约 **43.92 Overall**，再提高约1.52，累计比39.7508高约4.17。位置编码确有负面影响但不是主因；剩余主要疑点转向“四区域经静态原型凸组合后仅输出一个Prompt Token”的信息与容量瓶颈。
- 精确结果为42.4029/76.2240/8.6790和43.9208/79.0400/8.9024。去位置编码的+1.5179几乎全来自Yes/No +2.8160，Free-form仅+0.2234；它降低了晚期路由稀疏度但没有修复开放生成。无位置版最终Prompt范数91.17、原型范数174.45，尺度不足已排除；训练loss仍按epoch下降16.54->13.25->12.46，下一步应先评估现有epoch1/2检查点确认Validation趋势，再决定长程训练或多Prompt容量改造。
- 无位置版现有epoch1/2/3 Validation为 **42.1793/43.4414/43.9208 Overall**、77.6320/78.5280/79.0400 Yes-No、6.8283/8.4556/8.9024 Free-form，确认三轮内仍在改善，但Overall单轮增益已从+1.2621缩至+0.4794。应先从头执行更长scheduler-horizon的收敛实验，不可直接续训已衰减到零学习率的epoch3；趋势支持欠训练存在，却不足以证明长训能追回与Static Prompt约10.95分差距。
- 10-epoch长程控制在epoch2发生数值发散并于约epoch5.82人工停止：epoch2 loss日志出现122,678量级，随后loss被记录为0，Prompt/原型/路由及梯度诊断全部NaN。原因是10轮scheduler让Prompt LR0.3维持高位远长于三轮版；该运行没有产生有效长训性能结论，epoch2以后检查点不得评估或使用。

## 2026-09-14 Sandwich Learned Query容量匹配对照

- `pathvqa_qdpt_d768_learned_q10_l17_p20_s8_av10_sandwich_seed44`得到约 **59.05 Overall /90.88 Yes-No /27.31 Free-form**。它与最终问题引导Sandwich严格同为7,805,184参数、同seed44、同视觉K/V、同Prompt位置和训练协议，只把问题生成的Q10替换为等量可学习静态Q10。
- 相对问题引导Sandwich的60.7765/92.7360/28.9087，约下降 **1.73/1.86/1.60**。因此最终收益不能解释为单纯增加Query参数或通用learned-query视觉汇聚；当前问题条件确实提供了额外价值。该结果仍是单seed机制消融，正文不得写成跨seed稳定优势。精确未舍入值、配对统计、诊断和唯一输出路径待从服务器summary补齐。

## 2026-09-14 Prompt位置控制多seed更正

- 全放视觉后 `[Visual; P20; Z10; Question]` seeds44/45/46为59.2587/58.4119/59.2267，均值 **58.9658 +/- 0.4799**；反向Sandwich `[Z10; Visual; P20; Question]`为58.1243/58.4119/59.0989，均值 **58.5450 +/- 0.5008**；最终Sandwich沿用未舍入summary统计为 **59.1522 +/- 1.7528**。
- 相对最终Sandwich，同seed差值分别为全放视觉后-1.5178/+1.1184/-0.1598，反向-2.6522/+1.1184/-0.2876。seed44的位置优势没有稳定复现，三种顺序均值接近，且Sandwich方差最大。旧有“位置机制已经闭环、Sandwich稳定最优”的表述作废；只能说位置会影响优化轨迹，其方向依赖初始化。
- 不据此更换最终结构：Sandwich是在补充多seed结果前按seed44 Validation选定并冻结，正式PathVQA Test与SLAKE均已完成。论文保留其主模型身份，但位置消融必须报告三seed并降级为稳定性限制，不再用因果可见性解释承担核心创新证据。

## 2026-09-14 PathVQA基线Test

- Static Prompt P20 seeds44/45/46 Test Overall为 **55.8268/55.9161/56.4965**，均值 **56.0798 +/- 0.3636**；CoCoOp-style为 **57.7467/57.0323/56.0798**，均值 **56.9529 +/- 0.8363**。两组均已完整生成summary。
- Full-Attention LoRA-r8 seed44/45为 **59.6815/59.6369**，两seed暂均值 **59.6592 +/- 0.0315**；seed46只有评估日志而没有summary，当前记为未完成，禁止用两seed均值冒充最终三seed结果。QDPT Sandwich既有Test均值为 **58.8827 +/- 1.8141**，最终与LoRA的比较待补seed46。
- 补评更正：LoRA-r8 seed46 PathVQA Test已完成，显示结果 **59.67 Overall /91.6716 Yes-No /27.61 Free-form**，聚类95% CI[58.29,61.09]；TTFT0.055708s、TPOT0.030822s/token。三seed Overall为59.6815/59.6369/59.67，按显示精度约 **59.6628 +/-0.0232**，比QDPT Sandwich Test均值58.8827高约0.78分。上一条“seed46未完成”状态作废，最终精确统计应从summary读取未舍入值。

## 2026-09-14 Learned Query多seed与受控吞吐

- Learned Query容量匹配对照seeds44/45/46 Validation为 **59.0510/57.7249/58.1882**，均值 **58.3214 +/-0.6730**。相对问题引导QDPT同seed分别-1.7255/+0.4314/-1.1983，平均低 **0.8308**；问题条件化平均有益且两seed胜出，但seed45反转，不能宣称逐seed稳定机制优势。
- RTX5090公平短测中，QDPT和LoRA均用microbatch1/累积32、相同3,200样本和1,202,879视觉Token，20步预热后计时100个optimizer steps。QDPT为438.07s、7.3048 samples/s、峰值allocated15.805GiB；LoRA为824.08s、3.8831 samples/s、19.317GiB。QDPT训练吞吐为 **1.881x**，峰值allocated显存低 **3.512GiB/18.18%**。三轮2.245h/4.223h只能标注为纯训练线性外推，不是完整实测时间。

## 2026-09-21 QDPT seed44/45无训练模块交换

- PathVQA Validation上，`P20_45+rest_44`与`P20_44+rest_45`分别为**54.2898/55.0727 Overall**，相对各自receiver下降**6.4867/2.2208**；`A_t10_45+rest_44`与`A_t10_44+rest_45`分别为**53.0276/49.3849**，下降**7.7488/7.9086**。四项均为配对显著负效应，Anchor双向交换尤其严重，where最多下降32.27分。
- 较强seed44的P20或Anchor移入seed45都不能提升其性能，说明终点参数不是可独立替换的“好模块”；当前seed方差主要表现为P20、文本Anchor与动态生成器之间的seed特异强共适应。该结果不能将方差归咎于单个Prompt张量，也不能把交换模型作为性能方案。`P20_44+rest_45`的where仍提高3.18，提示seed44 P20含部分可迁移空间偏置，但总体兼容性损失更大。
- 本轮尚未包含后来加入的`Visual18`双向交换；它仍需单独运行，以判断Layer17静态视觉Prompt是否也参与这种跨seed共适应。

## 2026-09-22 Visual18双向交换续补

- 本条明确续补上一条记录，不修改旧四项结果。`Visual18_45+rest_44`为**60.4729 Overall**，相对receiver44仅-0.3036，Yes/No +0.0640、Free-form -0.6701，配对95% CI[-0.5578,-0.0474]、McNemar p=0.0271；`Visual18_44+rest_45`为**57.0858**，相对receiver45仅-0.2077，Yes/No -0.0320、Free-form -0.3829，CI[-0.5212,+0.0960]、p=0.2082。
- Visual18双向平均绝对损失仅**0.2556**，而P20与文本Anchor分别为4.3537和7.8287；交换后seed间差距仍保留原差距约97%。因此Layer17完整`S8+A_v10`视觉Prompt具有很强跨seed可移植性，可排除为当前方差的主要来源。后续优先检查LLM侧P20、文本Anchor、动态生成器的初始化与联合优化轨迹，不优先重构轻量视觉Prompt。
- 边界：该实验只排除了保存的Visual18终点张量是主因，不能推出所有视觉适配均无关；P20/Anchor的大幅交换损失证明文本侧强共适应，但仍不能把方差唯一归因于其中某一张量。

## 2026-09-22 冻结同seed Static P20稳定化控制（负结果，路线关闭）

- seed45在加载同seed独立Static P20 epoch3并冻结P20、仅重新训练Visual18/A_t10/完整动态分支后，PathVQA Validation为 **55.0248 Overall /89.8880 Yes-No /20.2616 Free-form**，较原QDPT seed45下降 **2.2687/0.8000/3.7333**；Overall又与独立Static P20 seed45的55.0567几乎相同。说明当前冻结P20方案使动态分支未能恢复QDPT增益，不满足最多下降0.30分的止损线。
- seed44因旧Static Prompt checkpoint没有保存seed元数据，在训练前被严格加载检查拒绝，报`checkpoint=None model=44`，无分数。这是旧checkpoint兼容问题而非训练发散。鉴于seed45已构成充分的性能否决证据，本路线直接关闭：不修兼容逻辑，不补跑seed44/46。单seed不能回答方差是否下降，但足以否定“预训练并冻结P20可在基本不扣分下稳定QDPT”；当前增益依赖P20、A_t与条件分支的联合适配。

## 2026-09-22 动态分支晚启动10%控制（方差降低但均值崩塌，路线关闭）

- 晚启动seed45/44 PathVQA Validation Overall为 **57.8048/57.2136**。相对原同seed QDPT，seed45 **+0.5113**，但seed44 **-3.5629**；两seed均值由59.0350降至 **57.5092 (-1.5258)**。seed差距由3.4830缩至 **0.5912**（缩小83.03%），主要来自压低强seed，不是合格的稳定性改进。
- Free-form两seed均值由26.4518降至 **25.1595 (-1.2923)**，未达到预注册25.9518下限；seed44为24.5692，较原值下降4.3395。seed45确实回升且通过首轮门槛，但最终两seed均值与Free-form门槛均失败，因此不补seed46、不扫启动比例、不评估Test。
- 机制上，这说明静态与动态分支同时起跑确实会影响seed依赖的优化轨迹：延迟动态分支能救弱seed45，却会破坏强seed44。它是稳定性来源之一但不是可直接采用的解决方案；固定10%硬切换被否决。边界审计的精确loss、梯度、残差比与注意力熵仍待从服务器`dynamic_late_start_audit.jsonl`提取。

## 2026-09-23 V0 三层视觉选择＋问题偏移 seed44

- `pathvqa_v0_visual_selection_offset_seed44` 在完整 PathVQA Validation（6,259题/832图像簇）、data seed42、3 epochs、固定epoch3取得 **55.6479 Overall /89.6640 Yes-No /显示21.73 Free-form**，聚类95% CI **[54.17,57.08]**。可训练参数 **2,356,675**，约为旧Dense Sandwich QDPT的30.19%；没有旧动态生成器/Z10，也未跑Test或其他seed。输出根目录：`pathvqa/outputs/visual_selection_offset/pathvqa_v0_visual_selection_offset_seed44_20260923`。
- 同seed同Validation协议，V0较Static P20 **+0.7829**，但较CoCoOp-style P20/H160 **-1.7574**、LoRA-r8 **-3.6907**、Dense Sandwich QDPT **-5.1286**。说明首版可运行且较纯静态P20有小幅收益，尚不足以主张相对更强轻量条件方法的参数/性能优势。当前只有单seed，不能判断稳定性；首批偏移比例、分支梯度、三层地图及层权重的数值待从服务器审计文件提取后再解释机制。

### 2026-09-23 V0诊断续补（用户提供日志，无新实验）

- 用户补充首批审计、93条训练诊断的摘要及训练报告；未独立读取服务器原始文件。训练6,164.61s、3 epochs、loss12.6862；各分支有有限非零梯度，深层地图明显偏离均匀，不支持“数值炸了/分支全死”的判断。成绩仍为55.6479 Overall，不新增评估结果。
- 末10条均值condition_rms0.594963、down_rms0.037482，二者均值比约15.87；offset/question RMS比0.681297。地图5/11/17熵均值0.980851/0.907977/0.811040，层权重0.167730/0.549122/0.283148。非均匀不等于正确定位，范数差不等于因果证明。
- 待证假设：视觉条件在加式ReLU瓶颈中主导激活，可能使问题词得到近似公共偏移，削弱token条件交互。先看词间激活/偏移差异和已训练checkpoint的配对推理干预，再决定是否值得单因素补救；不据此直接加层、扩宽、改学习率或续训。首批与轨迹梯度的聚合/裁剪口径未核实，不能直接比较量级。

### 2026-09-23 V0 seed44 epoch3只读诊断首次启动失败

- 目标`pathvqa_v0_seed44_epoch3_diagnostic`在脚本导入阶段因`from train.data_pipeline`报`ModuleNotFoundError`，尚未加载模型、运行128条前向诊断或`offset_off`/`condition_off`完整PathVQA Validation；无新Overall、分项分数或配对CI。未训练、未改checkpoint、未跑Test或其他seed。失败日志位于V0的`diagnostics/`独立输出目录，首次脚本未打印精确目录名，待用户确认；原V0成绩55.6479不变。已修正导入，待同范围重跑；这次失败不构成结构优劣证据。

### 2026-09-23 V0只读诊断在问题mask一致性审计处停止

- 修复导入后，目标成功加载原seed44 epoch3 checkpoint并重新核对2,356,675参数契约，但在首个命中样本`pathvqa:validation:756`发现训练问题token为`[12555,1558,419,2168,1473,30]`、推理prefill为`[12555,1558,419,2168,1473]`，推理少了末尾问号token。原因是现有推理fallback只保留完整字符offset落在原问题内的token，边界token可能跨越问题后的换行而被整体排除。
- 按预注册停止条件，没有继续128条统计，也没有运行`offset_off`或`condition_off`完整Validation；因此无新分数、独占正确数或配对CI，不能据此判断公共偏移退化。原55.6479应标记为旧推理mask实现下的历史结果，不能再视为训练/推理语义完全一致的干净V0基线。
- checkpoint未被修改，完整图文仍进入冻结基座；错误只影响V0用于生成条件地图/摘要的真实问题范围及偏移注入位置。当前不重训、不加Norm/gate、不改学习率。若继续，最小正确顺序是先修边界mask，再用同一checkpoint只重跑正常Validation基线，之后才决定是否恢复两项干预。输出：`pathvqa/outputs/visual_selection_offset/diagnostics/pathvqa_v0_seed44_epoch3_diagnostic_20260923_1`。
- 已实现边界修复：推理选择所有与原问题字符范围相交的token，并强制最终token ID序列与训练问题token完全一致；新增`pathvqa_v0_seed44_mask_fixed_validation`，只重跑同checkpoint正常Validation并与旧预测配对。新分数尚未产生，这不是干预结果或重训授权。

### 2026-09-23 V0问题mask修复v1在warmup停止

- v1恢复了末尾边界位置，但错误要求完整prompt中的边界token ID等于独立问题ID。实际warmup中训练末尾为问号ID30，推理把问号与后续换行上下文化为ID5267，因此在计分前失败；无新Validation分数或配对结果，也未运行干预、训练、Test或其他seed。
- v2改为分离“条件来源”和“写入位置”：条件分支读取独立分词的真实问题ID以匹配训练，偏移写入完整prefill中与问题重叠的位置；两侧数量必须一致。完整prompt和解码不变。下一步仍只是同checkpoint的修复后正常Validation，不据此重训。

### 2026-09-23 V0 mask修复v2正常Validation完成

- 同一seed44 epoch3 checkpoint、不重训，采用`standalone_training_source_with_prefill_overlap_targets_v2`得到 **56.7503 Overall /89.5040 Yes-No /24.0906 Free-form**，where59.4132。相对旧mask结果55.6479，Overall **+1.1024**，配对独占正确202/133，按图像簇配对95%差值CI[+0.4775,+1.7333]；Free-form **+2.3612**，Yes/No-0.16。来源为用户提供JSON，本任务未独立拉取服务器产物。
- 当前正常V0基线更新为56.7503，旧值保留作实现历史。相对Static P20 seed44高1.8853、相对CoCoOp-style seed44低0.6550，尚无针对这两个基线的新配对统计。已确认推理接口曾造成实质性损失，尚未验证公共偏移退化、视觉定位机制或稳定性；不能据此直接加Norm或重训。输出：`pathvqa/outputs/visual_selection_offset/diagnostics/pathvqa_v0_seed44_mask_fixed_validation_20260923_1`；两项关闭干预仍无结果。
- 用户已授权恢复128条前向诊断及两项关闭干预，配对基线固定为上述修复后预测，统一v2协议，跳过正常基线重跑。代码准备完成，服务器结果仍待回传；不能提前判断公共偏移、条件净收益或是否值得补救训练。

### 2026-09-23 V0修复mask后两项推理关闭干预

- 同一seed44 epoch3、完整Validation、修复后正常基线56.7503：`offset_off`为49.2251（-7.5252，配对95%差值CI[-8.4364,-6.5907]），Free-form13.9438（-10.1468）、where37.8973（-21.5159）；`condition_off`为51.9412（-4.8091，CI[-5.5663,-4.0373]），Free-form17.3261（-6.7645）、where42.7873（-16.6259）。完整分项和独占正确数见账本。
- 当前checkpoint明显依赖偏移和视觉条件，但这不是重训消融，不能把分差当作可加的模块贡献，也不能证明问题引导定位正确。公共偏移即使存在也可能有用，关闭结果不排除其表达不足。暂不直接加Norm或重训，先读取已安排的小样本探针。结果来自用户消息，实际干预输出目录、提交号与探针统计尚未提供，本轮未运行实验。

### 2026-09-24 V1 seed44首次训练在3%停止

- `pathvqa_v1_visual_selection_prefix_p20_seed44`（PathVQA、model seed44/data seed42、提交`807cc9b`）在49/1845步、约4分49秒触发监控断言，未保存checkpoint，未评估Validation/Test，Overall等分数不存在。原因是把不同P20基底经BF16舍入后的“反算增量”强行要求几乎相同；实际共享偏移在算术上仍由同一个shift广播。只移除这个无效阈值、保留诊断值与有效检查；配置不变，需从头重跑。输出：`/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/visual_selection_prefix/pathvqa_v1_visual_selection_prefix_p20_seed44_20260924`。

## 2026-09-24 V1前置条件P20完成（梯度累积归一化偏差版本）

- `pathvqa_v1_visual_selection_prefix_p20_seed44`，PathVQA seed44/data seed42、3 epochs固定epoch3 Validation：**57.5491 Overall /89.4080 Yes-No /25.78 Free-form**，where62.8362，图像簇95%CI[56.03,58.96]；参数1,864,963。用户回传结果，未独立拉取服务器产物。
- 对修复V0 +0.7988，配对CI[-0.0652,1.6414]；对CoCoOp +0.1438，CI[-0.6944,0.9793]；对Static P20 +2.6841，CI[1.7788,3.5726]。静态基线收益明确；尚不能宣称优于V0或CoCoOp，也不能以CI跨0宣称等效。where相对V0/CoCoOp分别+3.4230/+5.1345，仅为分项点估计。参数约为CoCoOp的2.14倍，未建立参数效率优势。
- 本次保持已发现的归一化偏差跑完：TF5.0.0/Accelerate1.12.0，直接forward loss3.8914；累积16步缺少平均，日志及裁剪前梯度约放大16倍，非参数更新16倍。手工CE、全窗口/尾窗口梯度数值复核及旧基线实际版本审计仍待完成，不提前归因于架构或宣称稳定性。暂不启动修正重训、Test或其他seed。
- 输出：`pathvqa/outputs/visual_selection_prefix/pathvqa_v1_visual_selection_prefix_p20_seed44_20260924_1`，checkpoint `checkpoints/epoch_3`，预测/summary在`eval_validation/epoch_3`。TTFT0.083398s，TPOT0.035026s/token，28.55token/s；训练耗时/峰值显存待补，不将历史耗时差异当作受控性能比较。

## 2026-09-24 V1梯度累积归一化只读数值复核通过

- `pathvqa_v1_loss_scaling_audit`用原seed44 epoch3 checkpoint与PathVQA训练样本，不重训、不评估、不更新参数。加P20后的同一logits/labels手工因果CE与模型loss均为**1.68652809**（12有效token）；传入窗口`num_items_in_batch=228`后loss不变。
- 完整16步窗口裁剪前原Trainer梯度为手工等权microbatch参照的**15.9949倍**；尾部实际3步为**2.9981倍**。只设`model_accepts_loss_kwargs=False`分别为**1.0001/0.9999倍**，方向近乎一致，数值门槛全部通过。原V1 57.5491分保留“梯度累积归一化偏差版本”标记；不等于参数更新被放大16倍。
- 独立诊断目录：`pathvqa/outputs/visual_selection_prefix/diagnostics/pathvqa_v1_loss_scaling_audit_20260924`。历史V0、CoCoOp、Static P20、QDPT、LoRA的实际运行版本/损失路径表未包含在用户贴出的日志中，仍待读取JSON；不能推定旧成绩全部受影响。受审计结果保护的单独修正入口已经准备，但未启动，匹配seed44修正复跑须另行授权。
- 用户随后提供`audit_summary.md`：审计实际提交`e42ae4608bcbb93137b3c9e410a565f03800237e`、运行环境PyTorch2.8.0+cu128/Transformers5.0.0/Accelerate1.12.0；历史五项seed44产物均未提取到当时提交和库版本，因此损失归一化协议均为**证据不足**，不能追认全体受影响。表中“PEFT wrapper or source unavailable”对四种Prompt方法只是缺失源码证据的占位文字；只有LoRA训练入口明确使用PEFT。

## 2026-09-24 V1归一化修正复跑完成

- `pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed44`，PathVQA seed44/data seed42、3 epochs固定epoch3 Validation：**58.57 Overall /90.2080 Yes-No /27.03 Free-form**，where66.7482；图像簇95%CI[57.07,60.05]。Overall/Free-form为终端两位精度，完整summary待提取。配置参数1,864,963。
- 唯一计划训练改动为`model_accepts_loss_kwargs=False`。对原V1偏差版本Overall约+1.02、Yes/No+0.80、Free-form约+1.25、where+3.9120；对CoCoOp Overall约+1.16，对修复V0约+1.82。本次配对CI未提供，不能沿用旧V1的CI；旧基线归一化协议未知，不能据此宣称结构显著胜出或多seed稳定。旧57.5491保留偏差标记。
- 输出：`pathvqa/outputs/visual_selection_prefix/pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed44_20260924`，预测和summary在`eval_validation/epoch_3`。TTFT0.084144s，TPOT0.035596s/token；最终训练耗时、显存、提交及诊断待补。下一步仅从已有产物提取精确分数、配对比较和训练记录，不自动增加训练或Test。

## 2026-09-25 V1修正版本只读诊断预检失败

- `pathvqa_v1_norm_fixed_seed44_diagnostic`在均匀地图小样本预检读取`p["grid"]`时因诊断元数据未写入而停止（`KeyError`）。可能已经完成正常预测核对、基线配对和128题探针，但用户未提供其数值；三项完整Validation干预均未完成，没有新分数。原V1修正版本约58.57分不变。失败输出位于`pathvqa/outputs/visual_selection_prefix/diagnostics/pathvqa_v1_norm_fixed_seed44_diagnostic_*`，具体后缀未从回传信息确认。现仅补齐探针网格记录，保留原预检门槛后重跑；不重训、不跑Test或其他seed。

## 2026-09-25 V1归一化修正版配对与机制诊断完成

- 正常分数精确为58.5717 Overall、90.2080 Yes/No、27.0262 Free-form、what21.9780、where66.7482。固定seed44 epoch3，6259题/832图。对原V1/修复V0/CoCoOp/Static配对差分别+1.0225/+1.8214/+1.1663/+3.7067；图像簇95%CI分别[0.2861,1.7806]/[0.9927,2.6473]/[0.4047,1.9510]/[2.7645,4.6560]，均排除0。仅适用于这些checkpoint，不证明多seed稳定或排除历史训练协议差异。
- 同类型条件问题错配55.6958（-2.8759，CI[-3.4472,-2.2992]），跨类型51.8933（-6.6784，CI[-7.4929,-5.8937]），均匀地图57.3574（-1.2143，CI[-1.7321,-0.7077]）；预测改变比例13.96%/23.15%/12.94%。支持当前checkpoint利用问题内容和非均匀视觉选择，但不是正确定位证明、重训消融或三层必要性证据。
- 128题探针同图换问题覆盖128/128、同问题换条件图像99/128；原始地图/摘要/偏移统计未随报告提供，不能判断三层冗余。干预分项及分歧图像人工核查同样待提取。输出`pathvqa/outputs/visual_selection_prefix/diagnostics/pathvqa_v1_norm_fixed_seed44_diagnostic_20260925_1`；本次用户回传report，未独立拉取JSON。不新增训练或Test。

## 2026-09-25 V1已完成诊断的原始产物补读（只读）

- 独立读取同一诊断目录的JSON及已导出的60张分歧图，无新增模型前向/评估。三项干预的分项正常→干预：同类型错配Yes/No 90.2080→88.9600、Free-form27.0262→22.5271、what21.9780→17.5039、where66.7482→60.1467；跨类型分别87.3920/16.4965/14.5604/32.2738；均匀地图分别89.1520/25.6541/21.1146/61.6137。同类型错配实际覆盖6259/6259，无保持原状题目。各项独占正确数、图像簇配对CI与题/图像数见`EXPERIMENT_RESULTS.md`；均匀地图的what子组CI跨0，子组分析均为探索性。
- 128题共同网格地图JS中位数：5↔11 0.303、5↔17 0.357、11↔17 0.221；对应投影前共同Value摘要余弦0.951/0.931/0.965，摘要相对L2 0.343/0.389/0.290。地图差异在摘要处有所压缩，却未缩成相同摘要。偏移/P20 RMS中位数0.376，去探针均值后的偏移范数中位数22.50（原始40.63），存在公共分量也存在差异。
- 同图换问题128/128，偏移原始/去均值相对变化中位数0.189/0.400；同问题换条件图像99/128，中位数0.489/0.794。在共同有效99题上，换问题中位数0.241/0.676。两种替换取样不同，不能把中位数差解释成图像因果贡献更大。图像错配缺29题：匹配必须使用此前存入、不同图且完全相同视觉网格的特征；其中5题可确认是库锚点，另外24题的具体原因因未保存网格/库状态而无法区分。无零范数。四组共60张预选分歧图以器官/系统辨认为主，另有明显答案粒度和措辞评分差异；复杂病理切片无法仅凭目视定论。
- 结论：当前checkpoint依赖真实问题内容与非均匀空间权重，应暂保留两条路径。三层11/17虽最相近，差异仍不小，缺少单层性能归因，不指定删层或宣称参数可安全减少。未证实定位正确、多seed稳定或历史基线协议等价；本次不执行简化训练。
