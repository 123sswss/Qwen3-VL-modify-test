#!/usr/bin/env python3
"""Train the shared multimodal Dynamic Prompt implementation on SLAKE."""

from pathvqa.train_dynamic_prompt import main


if __name__ == "__main__":
    raise SystemExit(main(dataset_name="slake"))
