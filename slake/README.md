# SLAKE data pipeline

`SLAKEDataset` scans a directory containing `xmlabN` folders. Each object in
each `question.json` is treated as one training sample, while all questions in
the same folder reuse that folder's `source.jpg`.

```text
SLAKE_ROOT/
  xmlab0/
    source.jpg
    question.json
  xmlab1/
    source.jpg
    question.json
```

Basic usage:

```python
from slake import SLAKEDataCollator, SLAKEDataset

dataset = SLAKEDataset(
    processor=processor,
    image_root="/path/to/imgs",
    languages=("en",),       # None keeps English and Chinese
    base_types=("vqa",),     # None also keeps knowledge-VQA records
    ce_enabled=True,
    seed=42,
)
collator = SLAKEDataCollator(processor)
```

The dataset item and collated batch use the same keys as
`train/data_pipeline.py`: token tensors, visual tensors, labels, expert label,
task metadata, and multimodal sample metadata.
