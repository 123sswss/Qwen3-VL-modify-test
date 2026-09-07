#!/usr/bin/env python3
"""Train GRASP on SLAKE through the existing data interface."""

from pathvqa.train_grasp import main


if __name__ == "__main__":
    raise SystemExit(main(dataset_name="slake"))
