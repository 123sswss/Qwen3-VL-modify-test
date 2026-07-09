"""Run each experiment in a fresh Python process, then analyze all JSON evidence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ACTIVE_RUNS = [
    ("eval_base.py", "base_run_1"),
    ("eval_base.py", "base_run_2"),
    ("eval_vision_lora_rank8.py", "vision_lora_rank8"),
    ("eval_vision_lora_rank16.py", "vision_lora_rank16"),
    ("eval_vision_lora_rank32.py", "vision_lora_rank32"),
    ("eval_vision_dora_rank8.py", "vision_dora_rank8"),
    ("eval_vision_dora_rank16.py", "vision_dora_rank16"),
    (
        "eval_last8_lora_full_linear_rank32.py",
        "last8_lora_full_linear_rank32",
    ),
    (
        "eval_last8_lora_attention_rank32.py",
        "last8_lora_attention_rank32",
    ),
    # ("eval_ours.py", "ours"),  # Enable after our checkpoint finishes training.
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="./eval_consistency/results")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = here.parent
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    for script, run_id in ACTIVE_RUNS:
        result_path = output_root / run_id / "raw_results.json"
        if args.skip_existing and result_path.exists():
            print(f"[skip] {run_id}: {result_path}")
            continue
        command = [
            sys.executable,
            str(here / script),
            "--run-id",
            run_id,
            "--output-root",
            str(output_root),
            "--num-samples",
            str(args.num_samples),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        print(f"[run] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=repo_root, check=True)

    subprocess.run(
        [
            sys.executable,
            str(here / "analyze.py"),
            "--output-root",
            str(output_root),
        ],
        cwd=repo_root,
        check=True,
    )


if __name__ == "__main__":
    main()
