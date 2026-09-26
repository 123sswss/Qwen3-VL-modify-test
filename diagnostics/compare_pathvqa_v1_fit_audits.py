#!/usr/bin/env python3
"""Compare the exact fixed V1 fitting sample after 3 and 5 epochs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GROUPS = ("yes/no", "what", "where", "other", "sample_total")


def load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--three-epoch-audit", type=Path, required=True)
    parser.add_argument("--five-epoch-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old, new = load(args.three_epoch_audit), load(args.five_epoch_audit)
    results = {}
    for split in ("train", "validation"):
        old_rows = load(args.three_epoch_audit.parent / f"{split}_per_question.json")
        new_rows = load(args.five_epoch_audit.parent / f"{split}_per_question.json")
        old_by_id = {str(row["question_id"]): row for row in old_rows}
        new_by_id = {str(row["question_id"]): row for row in new_rows}
        if set(old_by_id) != set(new_by_id) or len(old_by_id) != 256:
            raise ValueError(f"{split} fixed sample differs between audits")
        for qid, before in old_by_id.items():
            after = new_by_id[qid]
            for key in ("question", "reference", "image_id", "stratum"):
                if str(before[key]) != str(after[key]):
                    raise ValueError(f"{split} metadata differs at {qid}: {key}")
        results[split] = {}
        for group in GROUPS:
            before = old["summary"][split]["by_stratum"][group]
            after = new["summary"][split]["by_stratum"][group]
            row = {
                "count": after["count"],
                "three_epoch_accuracy": before["accuracy"],
                "five_epoch_accuracy": after["accuracy"],
                "accuracy_delta": after["accuracy"] - before["accuracy"],
                "three_epoch_body_ce_question_equal": before["body_ce_question_equal"],
                "five_epoch_body_ce_question_equal": after["body_ce_question_equal"],
                "body_ce_question_equal_delta": (
                    after["body_ce_question_equal"] - before["body_ce_question_equal"]
                ),
                "three_epoch_body_ce_token_equal": before["body_ce_token_equal"],
                "five_epoch_body_ce_token_equal": after["body_ce_token_equal"],
                "body_ce_token_equal_delta": (
                    after["body_ce_token_equal"] - before["body_ce_token_equal"]
                ),
            }
            results[split][group] = row
            print(
                f"[V1_FIT_5EP_VS_3EP] split={split} group={group} n={row['count']} "
                f"accuracy={row['three_epoch_accuracy']:.4f}->{row['five_epoch_accuracy']:.4f} "
                f"delta={row['accuracy_delta']:+.4f} "
                f"body_ce_q={row['three_epoch_body_ce_question_equal']:.6f}->"
                f"{row['five_epoch_body_ce_question_equal']:.6f} "
                f"delta={row['body_ce_question_equal_delta']:+.6f}", flush=True,
            )
    payload = {
        "experiment": "pathvqa_v1_norm_fixed_5ep_seed44",
        "sample_identity": "exact_question_ids_from_epoch3_sample_manifest",
        "splits": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
