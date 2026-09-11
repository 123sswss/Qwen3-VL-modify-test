#!/usr/bin/env python3
"""Train CoCoOp-style Prompt Tuning on the private electrical dataset."""

from pathvqa.train_cocoop import main


if __name__ == "__main__":
    raise SystemExit(main(dataset_name="electrical"))
