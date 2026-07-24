import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set


IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def resolve_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def collect_files(root: Path, all_files: bool) -> List[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and (all_files or path.suffix.lower() in IMAGE_SUFFIXES)
        ),
        key=lambda path: str(path.relative_to(root)),
    )


def build_report(
    reference_root: Path,
    training_root: Path,
    reference_files: List[Path],
    matched_files: List[Path],
    deleted_files: List[Path],
    duplicate_reference_names: Dict[str, int],
    delete_requested: bool,
    all_files: bool,
) -> Dict:
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": "delete" if delete_requested else "dry-run",
        "file_filter": "all" if all_files else "images",
        "reference_root": str(reference_root),
        "training_root": str(training_root),
        "reference_file_count": len(reference_files),
        "reference_unique_name_count": len({path.name for path in reference_files}),
        "duplicate_reference_names": duplicate_reference_names,
        "matched_training_file_count": len(matched_files),
        "deleted_training_file_count": len(deleted_files),
        "reference_names": sorted({path.name for path in reference_files}),
        "matched_training_files": [
            str(path.relative_to(training_root))
            for path in matched_files
        ],
        "deleted_training_files": [
            str(path.relative_to(training_root))
            for path in deleted_files
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively collect exact filenames from a reference/test directory, "
            "then find and optionally delete files with the same basename from a "
            "training directory."
        )
    )
    parser.add_argument(
        "reference_root",
        type=Path,
        help="Test-image directory whose filenames must be excluded from training.",
    )
    parser.add_argument(
        "training_root",
        type=Path,
        help="Training-data directory to scan recursively.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete matched training files. Without this flag, only report.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Match every file type. The safe default only considers image files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "training_image_overlap_report.json",
        help="UTF-8 JSON report path.",
    )
    args = parser.parse_args()

    try:
        reference_root = resolve_directory(args.reference_root, "reference_root")
        training_root = resolve_directory(args.training_root, "training_root")
        if paths_overlap(reference_root, training_root):
            raise ValueError(
                "reference_root and training_root must not be equal or nested; "
                "this prevents accidental deletion of test files"
            )
    except ValueError as error:
        parser.error(str(error))

    reference_files = collect_files(reference_root, all_files=args.all_files)
    reference_name_counts = Counter(path.name for path in reference_files)
    reference_names: Set[str] = set(reference_name_counts)
    duplicate_reference_names = {
        name: count
        for name, count in sorted(reference_name_counts.items())
        if count > 1
    }

    training_files = collect_files(training_root, all_files=args.all_files)
    matched_files = [
        path
        for path in training_files
        if path.name in reference_names
    ]

    deleted_files: List[Path] = []
    if args.delete:
        for path in matched_files:
            path.unlink()
            deleted_files.append(path)

    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(
        reference_root=reference_root,
        training_root=training_root,
        reference_files=reference_files,
        matched_files=matched_files,
        deleted_files=deleted_files,
        duplicate_reference_names=duplicate_reference_names,
        delete_requested=args.delete,
        all_files=args.all_files,
    )
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    mode = "DELETE" if args.delete else "DRY-RUN"
    print(
        f"[{mode}] reference_files={len(reference_files)} "
        f"unique_names={len(reference_names)} "
        f"training_files={len(training_files)} "
        f"matches={len(matched_files)} "
        f"deleted={len(deleted_files)}"
    )
    if duplicate_reference_names:
        print(
            "[WARNING] duplicate basenames inside reference directory: "
            f"{len(duplicate_reference_names)}"
        )
    for path in matched_files[:20]:
        print(f"  MATCH {path.relative_to(training_root)}")
    if len(matched_files) > 20:
        print(f"  ... and {len(matched_files) - 20} more")
    print(f"Report: {report_path}")
    if matched_files and not args.delete:
        print("No files were deleted. Re-run with --delete after reviewing the report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
