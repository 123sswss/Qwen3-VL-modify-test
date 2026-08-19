# MMRL Checkpoint Diagnostics

This directory keeps the reproducible cross-seed comparison used for the two
retained MMRL architectures:

- `layer_mlp_post_cross`
- `shared_direct_post_cross`

Run from the repository root after each experiment has produced
`eval/slake_comparisons.json`:

```bash
python checkpoint_diagnostics/compare_seeded_architectures.py
```

The script selects the latest matching seed 44, 45, and 46 runs, reports paired
accuracy and McNemar statistics, and writes Markdown and JSON reports under
`checkpoint_diagnostics/outputs/`.

Run its unit test with:

```bash
python -m unittest checkpoint_diagnostics.test_compare_seeded_architectures
```
