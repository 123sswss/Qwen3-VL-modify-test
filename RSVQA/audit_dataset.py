#!/usr/bin/env python3
"""CPU-only full RSVQA-LR release audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import audit_rsvqa_lr_splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_rsvqa_lr_splits(args.data_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("[RSVQA_DATASET_AUDIT] " + json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
