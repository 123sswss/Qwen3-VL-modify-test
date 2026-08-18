# Checkpoint Diagnostics

`dissect_dynamic_rep.py` performs an offline, read-only dissection of trained
dynamic-Rep compact checkpoints. It localizes same-layer slot collapse across:

- projected Q and visual/text K/V memory;
- pre-Softmax logits and Cross-Attention maps;
- retrieved context, CA delta, and final dynamic Rep;
- per-sample CE, dataset categories, and optional prediction correctness;
- cross-sample variation of the common Q/attention/context/delta direction.

The default dataset is a deterministic 256-sample subset of SLAKE validation.
The script forces `G=1` in memory only so every sample reaches the Rep branch;
pass `--respect-gate` to retain the checkpoint's binary gate.

## One checkpoint

```bash
python checkpoint_diagnostics/dissect_dynamic_rep.py \
  /root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/mmrl/EXPERIMENT/final
```

An experiment directory can also be passed; the script resolves its unique
`mmrl_manifest.json` automatically.

## Compare seed 44 and seed 45

```bash
python checkpoint_diagnostics/dissect_dynamic_rep.py \
  /root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/mmrl/slake_mmrl_dynamic_rep_cross_attention_seed44_20260807/final \
  /root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/mmrl/slake_mmrl_full_geometry_repro_seed45_20260808/final \
  --limit 512
```

## Join existing test predictions

This mode adds `correct` to each sample and reports correct/incorrect geometry.
It currently accepts one checkpoint at a time.

```bash
python checkpoint_diagnostics/dissect_dynamic_rep.py \
  /path/to/EXPERIMENT/final \
  --questions /root/autodl-tmp/dataset/slake/test.json \
  --predictions /path/to/EXPERIMENT/slake_predictions.json \
  --limit 2094
```

Outputs are written under `checkpoint_diagnostics/outputs/TIMESTAMP/`:

- `samples.jsonl`: one record per SLAKE sample;
- `layer_metrics.jsonl`: one record per sample and insertion layer;
- `summary.json`: aggregate geometry, correlations, and category groups;
- `report.md`: compact human-readable diagnosis;
- `comparison.md`: checkpoint comparison table.

## Matched-seed architecture comparison

Compare two architectures across seeds 44/45/46 without loading a model. The
script consumes the existing `eval/slake_comparisons.json` files and reports
paired wins/losses, exact McNemar tests, category deltas, and cross-seed answer
and correctness instability:

```bash
python checkpoint_diagnostics/compare_seeded_architectures.py
```

The defaults compare the shared-direct and layer-MLP three-seed reproduction
runs under `slake/outputs/mmrl`. Use `--root`, `--left-prefix`, and
`--right-prefix` to compare another pair of experiment families. If reruns add
date or `_1` suffixes, the script selects the candidate with the newest completed
`slake_comparisons.json` and prints all six selected paths before analysis.

## Cross-Attention delta scale ablation

Keep the trained static Rep base queries fixed and evaluate
`Rep = BaseQuery + scale * CrossAttentionDelta` at scales 0, 0.5, and 1. The
runner reuses an existing full `scale=1` evaluation when possible, so a normal
run adds only two evaluations:

```bash
bash checkpoint_diagnostics/run_cross_delta_scale_ablation.sh \
  /root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/mmrl/EXPERIMENT
```

The output table separates KVQA/VQA, answer type, and language. Set
`SLAKE_LIMIT` for a smoke test; partial runs evaluate all three scales rather
than mixing a partial result with the existing full baseline. Reusing an
existing `OUTPUT_ROOT` automatically resumes `slake_progress.jsonl` and skips
scales that already have a final summary.

## Query-generator checkpoint swap

Swap the complete static query generator (`shared S`, eight layer MLPs, and
layer embeddings) between two otherwise untouched checkpoints. This separates
query-side seed quality from memory-pooling/Cross-Attention seed quality without
training:

```bash
bash checkpoint_diagnostics/run_query_generator_swap_ablation.sh \
  /path/to/seed44/final /path/to/seed45/final
```

The two original evaluations are reused. Only `Q44 + Rest45` and
`Q45 + Rest44` are newly evaluated. Donor keys and tensor shapes are checked
strictly before inference, and interrupted runs resume automatically.

## Memory-count inference ablation

Evaluate one trained checkpoint four times without retraining: original memory,
text `128->1`, visual `128->1`, and both modalities `128->1`.
Collapsed tokens receive a `log(original_count)` attention-logit correction, so
the ablation preserves modality mass exactly when the original tokens are
identical instead of introducing a token-count bias.

```bash
bash checkpoint_diagnostics/run_memory_collapse_ablation.sh \
  /root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/mmrl/EXPERIMENT/final
```

Set `SLAKE_LIMIT=200` for a quick pipeline check. Failures in one mode do not
prevent later modes from running. The final table is saved as
`memory_collapse_summary.md` under the printed output directory.

This inference ablation still computes the trained 128-query pooling before
collapsing its outputs. Accuracy directly tests whether slot diversity matters;
its TTFT reduction is only a lower bound for a model retrained with one pooling
query per modality.

## Matched vs shuffled memory causality

`test_memory_causality.py` keeps each sample's visual backbone and Rep queries
fixed, then swaps visual memory, text memory, or both at the Cross-Attention
boundary. It performs no optimization and reports whether a real memory change
causes changes in attention, retrieved context, CA delta, dynamic Rep, supervised
CE, and teacher-forced token decisions.

```bash
python checkpoint_diagnostics/test_memory_causality.py \
  /root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/mmrl/EXPERIMENT/final \
  --limit 256 --batch-size 2
```

Pass multiple checkpoints to produce a direct comparison table. The default
forces `G=1`; use `--respect-gate` only when gate behavior is part of the test.
The current SLAKE diagnostic requires one image per sample and drops an
incomplete final batch so every sample has a mismatch partner. Pairing is
randomized reproducibly with `--pair-seed` instead of using adjacent records,
which may contain different questions about the same image.

To separate Q/K routing from V content transfer, shuffle projected K and V
independently while leaving the other projection matched:

```bash
python checkpoint_diagnostics/test_memory_causality.py \
  /root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/mmrl/EXPERIMENT/final \
  --limit 256 --batch-size 2 \
  --modes key_visual key_text key_both value_visual value_text value_both
```

`key_*` changes attention routing but reads the original sample's V;
`value_*` preserves the original attention routing but reads another sample's
V. The report includes projected K/V change ratios to verify each intervention.

Two K-structure modes isolate content addressing without changing V or memory
token counts:

```bash
python checkpoint_diagnostics/test_memory_causality.py \
  /root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/mmrl/EXPERIMENT/final \
  --limit 256 --batch-size 2 \
  --modes key_modality_mean key_zero key_both
```

`key_modality_mean` repeats one projected mean K inside each modality, retaining
only visual-vs-text routing. `key_zero` makes attention uniform across all
memory tokens. Both retain every original V token.

Calibrated static routing removes Q/K from the effective forward. It estimates
one visual mass per attention head, insertion layer, and Rep slot on a disjoint
calibration subset, then reads all V tokens uniformly inside each modality:

```bash
python checkpoint_diagnostics/test_memory_causality.py \
  /root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/mmrl/EXPERIMENT/final \
  --limit 256 --batch-size 2 \
  --static-routing-calibration-samples 64 \
  --modes attention_static_modality key_modality_mean key_zero
```

The calibration split uses no labels and is excluded from the reported
evaluation samples. The resulting fixed routing has at most
`heads * layers * Rep_slots` independent visual-mass parameters.
