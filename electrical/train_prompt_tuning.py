#!/usr/bin/env python3
"""Train static Prompt Tuning on the private electrical dataset."""

from pathvqa.train_prompt_tuning import main


if __name__ == "__main__":
    raise SystemExit(main(dataset_name="electrical"))
