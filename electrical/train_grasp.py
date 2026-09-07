#!/usr/bin/env python3
"""Train GRASP through the existing private electrical data interface."""

from pathvqa.train_grasp import main


if __name__ == "__main__":
    raise SystemExit(main(dataset_name="electrical"))
