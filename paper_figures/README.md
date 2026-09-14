# QDPT Paper Figures

These scripts keep raw values separate from rendered figures. Every plotting command writes SVG, PDF, and 300-DPI PNG files with the same prefix.

## Final one-folder bundle

Generate Figures 1-4, attention source data, selection metadata, a machine-readable
status report, and a tar archive inside one directory:

```bash
RUN_TARGET=pathvqa_qdpt_paper_figures_final_bundle bash run_experiment.sh
```

The default destination is `paper_figures/output/final_bundle_<RUN_DATE>/`.
This target performs no training. Figure 1 loads the frozen seed44 checkpoint for
two no-gradient inference calls; the remaining figures only read existing files.
The attention case defaults to deterministic candidate rank2 so that the original
rank1 case is not reused. Override it with `ATTENTION_CANDIDATE_RANK=3` if needed.

## Dependencies

```bash
python -m pip install -r pathvqa/requirements.txt
```

## Figure 2: Three-seed training dynamics

```bash
python -m paper_figures.qdpt_figures dynamics \
  --run seed44=/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed44_20260909 \
  --run seed45=/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed45_20260910_1 \
  --run seed46=/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed46_20260910 \
  --score 60.7765 --score 57.2935 --score 59.3865 \
  --output paper_figures/output/figure2_training_dynamics
```

The x-axis is each run's logged step divided by its final diagnostic step. Loss is read from `trainer/trainer_state.json`; all other curves come from `dynamic_prompt_diagnostics.jsonl`.

## Figure 3: Seed stability

The audited PathVQA Validation values are stored in `pathvqa_seed_scores.json` rather than hidden in plotting code.

```bash
python -m paper_figures.qdpt_figures stability \
  --output paper_figures/output/figure3_seed_stability
```

## Figure 4: Module activity

```bash
python -m paper_figures.qdpt_figures activity \
  --run seed44=/root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed44_20260909 \
  --output paper_figures/output/figure4_module_activity
```

## Figure 1: Question-guided visual attention

First rank same-image cases by a deterministic rule. This step only reads the dataset and does not load the model.

```bash
python -m paper_figures.select_attention_cases \
  --data-root /root/autodl-tmp/dataset/pathVQA \
  --split validation \
  --top 20 \
  --output paper_figures/output/attention_candidates.json
```

Choose two row indices from one candidate. Record the rank and selection rationale in the paper artifact. Exporting attention loads the checkpoint and performs inference without gradients; it does not train or modify the checkpoint.

```bash
python -m paper_figures.export_qdpt_attention \
  --checkpoint /root/autodl-tmp/Qwen3-VL-modify-test/pathvqa/outputs/dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed44_20260909/checkpoints/epoch_3 \
  --data-root /root/autodl-tmp/dataset/pathVQA \
  --split validation \
  --row-index ROW_A \
  --row-index ROW_B \
  --output-dir paper_figures/output/attention_case_01
```

The bundle contains the source image, generated and reference answers, grid dimensions, question-pooling weights, Workspace tokens, and the full `16 x 10 x L` Cross-Attention matrix for every question. Render it with:

```bash
python -m paper_figures.qdpt_figures attention \
  --bundle-dir paper_figures/output/attention_case_01 \
  --output paper_figures/output/figure1_question_guided_attention
```

The aggregate map averages 16 heads and then 10 queries. The three additional maps are selected by the lowest normalized visual-attention entropy. The temporal grid dimension is averaged before overlaying the spatial map on the image.
