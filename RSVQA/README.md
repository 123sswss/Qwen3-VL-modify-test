# RSVQA-LR interface

This package is the shared local interface for future RSVQA-LR inference and
training.  `data.py` owns official split discovery, independent `active=true`
filtering, ID joins, image resolution, and integrity checks.  Training code
should consume `load_rsvqa_lr_split()` rather than reimplementing the joins.

The evaluator reports official question-type accuracy, Overall Accuracy (OA),
and Average Accuracy (AA).  For LR count questions, both references and integer
model outputs use the original release's `VocabEncoder(range_numbers=True)`
ranges: `0`, `1-10`, `11-100`, `101-1000`, and `>1000`.

CPU-only dataset audit (does not load a model):

```bash
python -m RSVQA.rsvqa_lr_official_eval \
  --data-root /root/autodl-tmp/dataset/RSVQA/6344334 \
  --split test --backend base --base-model /root/autodl-tmp/model \
  --output-dir /tmp/rsvqa-audit --audit-only
```

The full base-model Test run is exposed through
`bash run_experiment.sh rsvqa_lr_base_qwen3vl_test`.  It evaluates all 10,004
official Test questions; `--limit` results are explicitly marked partial and
must not be reported as official scores.
