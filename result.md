# Experiment Result Summary

Last updated: 2026-08-26

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
| 128-slot MMRL + Relation0.05 | 44 | **47.30** | 82.57 | 11.97 | First completed PathVQA result. Gradient flow was active, but open answers were very weak; wait for matched Visual LoRA-r128 and Base before assigning the failure to MMRL. |

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
- PathVQA is the primary public dataset and SLAKE is the secondary robustness dataset. The first PathVQA MMRL run completed at47.30 Overall, but its meaning remains unresolved until Visual LoRA-r128 and Base controls finish.
- The in-house dataset will be reported as a single-seed internal application case and will not support the main statistical claims.
