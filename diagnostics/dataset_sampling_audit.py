#!/usr/bin/env python3
"""Audit historical and current Stage 3 sampling without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EXPERT_JSON = [
    "/root/autodl-tmp/dataset/1json.json",
    "/root/autodl-tmp/dataset/2conv_c.json",
    "/root/autodl-tmp/dataset/1conv_c.json",
    "/root/autodl-tmp/dataset/4conv_c.json",
    "/root/autodl-tmp/dataset/14json.json",
    "/root/autodl-tmp/dataset/prof_test.json",
    "/root/autodl-tmp/dataset/test2_train.json",
    "/root/autodl-tmp/dataset/test7_train.json",
]

DEFAULT_EXPERT_IMAGE_DIRS = [
    "/root/autodl-tmp/dataset/1/train",
    "/root/autodl-tmp/dataset/2/train",
    "/root/autodl-tmp/dataset/4/train",
    "/root/autodl-tmp/dataset/14",
]


def parse_args() -> argparse.Namespace:
    script_repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact FourViewMMRLDataset._build implementations from an "
            "old git commit and the current worktree, then compare their raw "
            "20,000-image Stage 3 selections."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=script_repo_root)
    parser.add_argument("--old-commit", default="88a3c30")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=20_000)
    parser.add_argument(
        "--expert-json",
        nargs="+",
        default=DEFAULT_EXPERT_JSON,
        metavar="PATH",
    )
    parser.add_argument(
        "--expert-image-dir",
        nargs="+",
        default=DEFAULT_EXPERT_IMAGE_DIRS,
        metavar="DIR",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_repo_root / "diagnostics" / "dataset_sampling_audit_output",
    )
    return parser.parse_args()


def require_inputs(paths: Iterable[str], kind: str) -> None:
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing {kind} paths:\n{rendered}")


def load_module_from_source(
    module_name: str,
    source: str,
    source_label: str,
) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__file__ = source_label
    module.__audit_source_sha256__ = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()
    sys.modules[module_name] = module
    try:
        exec(compile(source, source_label, "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_old_module(repo_root: Path, commit: str) -> types.ModuleType:
    result = subprocess.run(
        ["git", "show", f"{commit}:train/data_pipeline.py"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return load_module_from_source(
        f"_dataset_pipeline_old_{commit.replace('-', '_')}",
        result.stdout,
        f"{commit}:train/data_pipeline.py",
    )


def load_current_module(repo_root: Path) -> types.ModuleType:
    source_path = repo_root / "train" / "data_pipeline.py"
    source = source_path.read_text(encoding="utf-8")
    return load_module_from_source(
        "_dataset_pipeline_current_audit",
        source,
        str(source_path),
    )


def git_revision(repo_root: Path, revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", revision],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def normalize_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "bot"}:
        return "assistant"
    return role


def first_turn(conversations: Any, wanted_role: str) -> tuple[int | None, str | None]:
    if not isinstance(conversations, list):
        return None, None
    for index, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            continue
        if normalize_role(turn.get("from")) == wanted_role:
            value = turn.get("value")
            return index, value if isinstance(value, str) else str(value or "")
    return None, None


def json_safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def prepare_raw_dataset(
    module: types.ModuleType,
    expert_json: list[str],
    expert_image_dirs: list[str],
    limit: int,
    seed: int,
) -> Any:
    """Construct enough dataset state to execute the module's exact _build()."""
    dataset_class = module.FourViewMMRLDataset
    dataset = object.__new__(dataset_class)
    dataset.processor = None
    dataset.total_limit = limit
    dataset.enable_views = {"expert-mm"}
    dataset.mode = "stage3_sampling_audit"
    dataset.seed = seed
    dataset.deterministic_sampling = True
    dataset.resample_round = 0
    dataset.ce_enabled = False
    dataset.assistant_turn_policy = "first"

    dataset.expert_raw = module.load_jsons(expert_json)
    dataset.general_raw = []
    dataset.expert_map, dataset.expert_dir = module.build_image_mapping(
        expert_image_dirs
    )
    dataset.general_map, dataset.general_dir = None, None
    dataset.data = []

    source_offsets: Counter[str] = Counter()
    item_metadata: dict[int, tuple[int, int]] = {}
    for pool_index, item in enumerate(dataset.expert_raw):
        source_path = str(item.get("__source_json_path") or "")
        source_index = source_offsets[source_path]
        source_offsets[source_path] += 1
        item_metadata[id(item)] = (pool_index, source_index)

    def build_audit_row(self: Any, item: dict[str, Any], task_type: str) -> list[dict]:
        is_expert = task_type == "expert"
        conversations = item.get("conversations", [])
        user_index, first_user = first_turn(conversations, "user")
        assistant_index, first_assistant = first_turn(conversations, "assistant")
        source_path = str(item.get("__source_json_path") or "")
        pool_index, source_item_index = item_metadata[id(item)]
        raw_image = str(item.get("image", ""))
        resolved_image = self._resolve_img(item, is_expert)
        return [
            {
                "source_pool_index": pool_index,
                "source_item_index": source_item_index,
                "source_json_path": source_path,
                "source_name": os.path.basename(source_path) or "unknown",
                "item_id": json_safe_scalar(item.get("id")),
                "raw_image": raw_image,
                "resolved_image_path": str(resolved_image or ""),
                "raw_turn_count": len(conversations)
                if isinstance(conversations, list)
                else 0,
                "first_user_turn_index": user_index,
                "first_user": first_user,
                "first_assistant_turn_index": assistant_index,
                "first_assistant": first_assistant,
            }
        ]

    dataset._build_views_from_item = types.MethodType(build_audit_row, dataset)
    return dataset


def finalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finalized = []
    for final_index, row in enumerate(rows):
        copied = dict(row)
        copied["final_index"] = final_index
        finalized.append(copied)
    return finalized


def build_initial_rows(dataset: Any) -> list[dict[str, Any]]:
    dataset.data = []
    dataset._build()
    return finalize_rows(dataset.data)


def build_resampled_rows(dataset: Any) -> list[dict[str, Any]]:
    dataset.resample_data()
    return finalize_rows(dataset.data)


def image_identity(row: dict[str, Any]) -> tuple[str, str]:
    return row["source_name"], row["raw_image"]


def turn_identity(row: dict[str, Any]) -> tuple[str, str, str | None, str | None]:
    source_name, raw_image = image_identity(row)
    return source_name, raw_image, row["first_user"], row["first_assistant"]


def stable_sha256(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_sort(values: Iterable[Any]) -> list[Any]:
    return sorted(
        values,
        key=lambda value: json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def counter_difference_examples(
    left: Counter,
    right: Counter,
    limit: int = 10,
) -> list[dict[str, Any]]:
    examples = []
    for key, count in (left - right).most_common(limit):
        examples.append({"value": list(key), "count": count})
    return examples


def duplicate_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    counts = Counter(row[field] for row in rows)
    duplicates = [(value, count) for value, count in counts.items() if count > 1]
    duplicates.sort(key=lambda pair: (-pair[1], pair[0]))
    return {
        "duplicated_value_count": len(duplicates),
        "extra_occurrence_count": sum(count - 1 for _, count in duplicates),
        "examples": [
            {"value": value, "count": count}
            for value, count in duplicates[:20]
        ],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered_images = [image_identity(row) for row in rows]
    ordered_turns = [turn_identity(row) for row in rows]
    return {
        "count": len(rows),
        "ordered_image_sha256": stable_sha256(ordered_images),
        "unordered_image_multiset_sha256": stable_sha256(
            canonical_sort(ordered_images)
        ),
        "ordered_first_turn_sha256": stable_sha256(ordered_turns),
        "unordered_first_turn_multiset_sha256": stable_sha256(
            canonical_sort(ordered_turns)
        ),
        "per_source_counts": dict(sorted(Counter(
            row["source_name"] for row in rows
        ).items())),
        "duplicate_raw_images": duplicate_summary(rows, "raw_image"),
        "duplicate_resolved_image_paths": duplicate_summary(
            rows,
            "resolved_image_path",
        ),
        "missing_first_user_count": sum(
            row["first_user"] is None for row in rows
        ),
        "missing_first_assistant_count": sum(
            row["first_assistant"] is None for row in rows
        ),
    }


def compare_rows(
    left_name: str,
    left_rows: list[dict[str, Any]],
    right_name: str,
    right_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    left_images = [image_identity(row) for row in left_rows]
    right_images = [image_identity(row) for row in right_rows]
    left_turns = [turn_identity(row) for row in left_rows]
    right_turns = [turn_identity(row) for row in right_rows]

    left_image_counts = Counter(left_images)
    right_image_counts = Counter(right_images)
    left_turn_counts = Counter(left_turns)
    right_turn_counts = Counter(right_turns)

    paired_count = min(len(left_rows), len(right_rows))
    same_image_positions = [
        index
        for index in range(paired_count)
        if left_images[index] == right_images[index]
    ]
    same_user_at_same_image = sum(
        left_rows[index]["first_user"] == right_rows[index]["first_user"]
        for index in same_image_positions
    )
    same_assistant_at_same_image = sum(
        left_rows[index]["first_assistant"]
        == right_rows[index]["first_assistant"]
        for index in same_image_positions
    )

    first_order_mismatch = None
    for index in range(paired_count):
        if left_turns[index] != right_turns[index]:
            first_order_mismatch = {
                "index": index,
                left_name: left_rows[index],
                right_name: right_rows[index],
            }
            break
    if first_order_mismatch is None and len(left_rows) != len(right_rows):
        first_order_mismatch = {
            "index": paired_count,
            "reason": "list lengths differ",
        }

    return {
        "left": left_name,
        "right": right_name,
        "left_count": len(left_rows),
        "right_count": len(right_rows),
        "image_membership": {
            "exact_multiset_match": left_image_counts == right_image_counts,
            "shared_occurrences": sum(
                (left_image_counts & right_image_counts).values()
            ),
            "left_only_occurrences": sum(
                (left_image_counts - right_image_counts).values()
            ),
            "right_only_occurrences": sum(
                (right_image_counts - left_image_counts).values()
            ),
            "left_only_examples": counter_difference_examples(
                left_image_counts,
                right_image_counts,
            ),
            "right_only_examples": counter_difference_examples(
                right_image_counts,
                left_image_counts,
            ),
        },
        "image_order": {
            "exact_match": left_images == right_images,
            "matching_position_count": len(same_image_positions),
            "paired_position_count": paired_count,
        },
        "first_turns": {
            "exact_ordered_match": left_turns == right_turns,
            "exact_multiset_match": left_turn_counts == right_turn_counts,
            "matching_first_user_at_same_image_position": same_user_at_same_image,
            "matching_first_assistant_at_same_image_position": (
                same_assistant_at_same_image
            ),
            "left_only_occurrences": sum(
                (left_turn_counts - right_turn_counts).values()
            ),
            "right_only_occurrences": sum(
                (right_turn_counts - left_turn_counts).values()
            ),
            "left_only_examples": counter_difference_examples(
                left_turn_counts,
                right_turn_counts,
            ),
            "right_only_examples": counter_difference_examples(
                right_turn_counts,
                left_turn_counts,
            ),
        },
        "first_order_or_content_mismatch": first_order_mismatch,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_image_list(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "final_index\tsource_name\traw_image\tresolved_image_path\n"
        )
        for row in rows:
            values = (
                row["final_index"],
                row["source_name"],
                row["raw_image"],
                row["resolved_image_path"],
            )
            handle.write("\t".join(str(value) for value in values) + "\n")


def assert_expected_count(name: str, rows: list[dict[str, Any]], limit: int) -> None:
    if len(rows) != limit:
        raise RuntimeError(
            f"{name} produced {len(rows)} rows instead of the required {limit}. "
            "This usually means fewer than limit valid image/conversation items "
            "were available."
        )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    expert_json = [str(Path(path).resolve()) for path in args.expert_json]
    expert_image_dirs = [
        str(Path(path).resolve()) for path in args.expert_image_dir
    ]

    require_inputs(expert_json, "expert JSON")
    require_inputs(expert_image_dirs, "expert image directory")
    if not (repo_root / ".git").exists():
        raise FileNotFoundError(f"Not a git worktree: {repo_root}")

    print("========== Dataset Sampling Audit ==========")
    print(f"repo_root={repo_root}")
    print(f"old_commit={args.old_commit}")
    print(f"base_seed={args.base_seed}")
    print(f"limit={args.limit}")
    print("Loading historical and current data_pipeline.py...")

    old_module = load_old_module(repo_root, args.old_commit)
    current_module = load_current_module(repo_root)
    old_revision = git_revision(repo_root, args.old_commit)
    current_revision = git_revision(repo_root, "HEAD")

    print("Building old seed42 initial selection...")
    old_dataset = prepare_raw_dataset(
        old_module,
        expert_json,
        expert_image_dirs,
        args.limit,
        args.base_seed,
    )
    old_initial = build_initial_rows(old_dataset)

    print("Building old seed43 callback-resampled selection...")
    old_resampled = build_resampled_rows(old_dataset)

    print("Building current seed42 same-seed selection...")
    current_seed42_dataset = prepare_raw_dataset(
        current_module,
        expert_json,
        expert_image_dirs,
        args.limit,
        args.base_seed,
    )
    current_seed42 = build_initial_rows(current_seed42_dataset)

    print("Building current seed43 Stage 3 selection...")
    current_seed43_dataset = prepare_raw_dataset(
        current_module,
        expert_json,
        expert_image_dirs,
        args.limit,
        args.base_seed + 1,
    )
    current_seed43 = build_initial_rows(current_seed43_dataset)

    datasets = {
        f"old_seed{args.base_seed}_initial": old_initial,
        f"old_seed{args.base_seed + 1}_resampled": old_resampled,
        f"current_seed{args.base_seed}": current_seed42,
        f"current_seed{args.base_seed + 1}_stage3": current_seed43,
    }
    for name, rows in datasets.items():
        assert_expected_count(name, rows, args.limit)

    comparisons = {
        "same_seed_initial_implementation": compare_rows(
            f"old_seed{args.base_seed}_initial",
            old_initial,
            f"current_seed{args.base_seed}",
            current_seed42,
        ),
        "same_seed_resampled_implementation": compare_rows(
            f"old_seed{args.base_seed + 1}_resampled",
            old_resampled,
            f"current_seed{args.base_seed + 1}_stage3",
            current_seed43,
        ),
        "historical_initial_vs_current_stage3": compare_rows(
            f"old_seed{args.base_seed}_initial",
            old_initial,
            f"current_seed{args.base_seed + 1}_stage3",
            current_seed43,
        ),
    }
    verdict = {
        "old_and_current_sampling_match_exactly_at_base_seed": (
            comparisons["same_seed_initial_implementation"]["first_turns"][
                "exact_ordered_match"
            ]
        ),
        "old_resampled_and_current_stage3_match_exactly": (
            comparisons["same_seed_resampled_implementation"]["first_turns"][
                "exact_ordered_match"
            ]
        ),
        "old_initial_and_current_stage3_match_exactly": (
            comparisons["historical_initial_vs_current_stage3"]["first_turns"][
                "exact_ordered_match"
            ]
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_summaries = {}
    for name, rows in datasets.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
        write_image_list(output_dir / f"{name}.images.tsv", rows)
        dataset_summaries[name] = summarize_rows(rows)

    summary = {
        "audit_contract": {
            "old_source": f"{args.old_commit}:train/data_pipeline.py",
            "old_revision": old_revision,
            "old_source_sha256": old_module.__audit_source_sha256__,
            "current_source": str(repo_root / "train" / "data_pipeline.py"),
            "current_head_revision": current_revision,
            "current_source_sha256": current_module.__audit_source_sha256__,
            "sampling_method": (
                "Each source module's unmodified FourViewMMRLDataset._build() "
                "and resample_data() execute directly. Only item-to-view "
                "conversion is replaced with a raw audit record, so no "
                "processor, model, CUDA, turn expansion, or length filtering "
                "can alter the 20,000-item selection."
            ),
            "expert_json": expert_json,
            "expert_image_dirs": expert_image_dirs,
            "limit": args.limit,
            "base_seed": args.base_seed,
        },
        "verdict": verdict,
        "datasets": dataset_summaries,
        "comparisons": comparisons,
    }
    write_json(output_dir / "summary.json", summary)

    print("========== Audit Result ==========")
    for name, comparison in comparisons.items():
        membership = comparison["image_membership"]
        order = comparison["image_order"]
        turns = comparison["first_turns"]
        print(
            f"{name}: "
            f"images_multiset_equal={membership['exact_multiset_match']} "
            f"images_order_equal={order['exact_match']} "
            f"first_turns_order_equal={turns['exact_ordered_match']} "
            f"shared_images={membership['shared_occurrences']}/{args.limit}"
        )
    print(f"summary={output_dir / 'summary.json'}")
    print("No model was loaded and no CUDA operation was requested.")


if __name__ == "__main__":
    main()
