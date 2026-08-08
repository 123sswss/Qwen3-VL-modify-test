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
