#!/usr/bin/env python3
"""Train the final QDPT configuration on the private electrical dataset."""

from pathvqa.train_dynamic_prompt import main


if __name__ == "__main__":
    raise SystemExit(main(dataset_name="electrical"))
