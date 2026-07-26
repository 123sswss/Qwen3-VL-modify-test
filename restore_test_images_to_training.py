import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set


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


def collect_images(root: Path) -> List[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: str(path.relative_to(root)),
    )


def index_unique_basenames(paths: Iterable[Path], label: str) -> Dict[str, Path]:
    paths_by_name = defaultdict(list)
    for path in paths:
        paths_by_name[path.name].append(path)
    duplicates = {
        name: values
        for name, values in paths_by_name.items()
        if len(values) > 1
    }
    if duplicates:
        examples = {
            name: [str(path) for path in values[:3]]
            for name, values in list(sorted(duplicates.items()))[:5]
        }
        raise ValueError(f"Duplicate basenames in {label}: {examples}")
    return {name: values[0] for name, values in paths_by_name.items()}


def load_training_image_references(json_paths: Iterable[Path]) -> Dict[str, List[str]]:
    references = defaultdict(list)
    for raw_path in json_paths:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"training JSON does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"training JSON is not a list: {path}")

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"training JSON item is not an object: {path} index={index}"
                )
            image_value = record.get("image")
            if not isinstance(image_value, str) or not image_value.strip():
                continue
            references[Path(image_value).name].append(str(path))
    return dict(references)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "DIAGNOSTIC DATA-LEAKAGE TOOL. Copy test images back into a training "
            "image directory so existing training JSON records become valid again."
        )
    )
    parser.add_argument(
        "test_root",
        type=Path,
        help="Test-image directory to scan recursively.",
    )
    parser.add_argument(
        "training_root",
        type=Path,
        help=(
            "Training image directory. Restored files are placed directly in this "
            "directory because the current dataset image mapping is non-recursive."
        ),
    )
    parser.add_argument(
        "--training-json",
        action="append",
        type=Path,
        default=[],
        help=(
            "Training JSON that must reference restored images. Repeat this option "
            "for multiple JSON files. When supplied, unreferenced test images are "
            "not copied."
        ),
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Actually copy files. Without this flag, only perform a dry-run.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent
        / "training_image_restore_report.json",
        help="UTF-8 JSON report path.",
    )
    args = parser.parse_args()

    try:
        test_root = resolve_directory(args.test_root, "test_root")
        training_root = resolve_directory(args.training_root, "training_root")
        if paths_overlap(test_root, training_root):
            raise ValueError("test_root and training_root must not overlap")

        source_images = collect_images(test_root)
        source_by_name = index_unique_basenames(source_images, "test_root")
        training_references = load_training_image_references(args.training_json)
    except ValueError as error:
        parser.error(str(error))

    source_names: Set[str] = set(source_by_name)
    if args.training_json:
        restorable_names = source_names & set(training_references)
        unreferenced_names = source_names - set(training_references)
    else:
        restorable_names = set(source_names)
        unreferenced_names = set()

    already_present = []
    pending = []
    collisions = []
    for name in sorted(restorable_names):
        source = source_by_name[name]
        destination = training_root / name
        source_hash = sha256_file(source)
        if destination.exists():
            if not destination.is_file():
                collisions.append(
                    {
                        "name": name,
                        "reason": "destination is not a file",
                        "destination": str(destination),
                    }
                )
                continue
            destination_hash = sha256_file(destination)
            if source_hash == destination_hash:
                already_present.append(
                    {
                        "name": name,
                        "source": str(source),
                        "destination": str(destination),
                        "sha256": source_hash,
                    }
                )
            else:
                collisions.append(
                    {
                        "name": name,
                        "reason": "destination has different content",
                        "source": str(source),
                        "destination": str(destination),
                        "source_sha256": source_hash,
                        "destination_sha256": destination_hash,
                    }
                )
            continue
        pending.append(
            {
                "name": name,
                "source": str(source),
                "destination": str(destination),
                "sha256": source_hash,
            }
        )

    if collisions and args.copy:
        parser.error(
            f"refusing partial copy because {len(collisions)} destination "
            "collisions were found; review the report in dry-run mode"
        )

    copied = []
    if args.copy:
        for item in pending:
            shutil.copy2(item["source"], item["destination"])
            if sha256_file(Path(item["destination"])) != item["sha256"]:
                raise RuntimeError(
                    f"copy verification failed: {item['destination']}"
                )
            copied.append(item)

    referenced_test_record_count = (
        sum(
            len(training_references[name])
            for name in restorable_names
        )
        if args.training_json
        else 0
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "diagnostic_data_leakage_only": True,
        "mode": "copy" if args.copy else "dry-run",
        "test_root": str(test_root),
        "training_root": str(training_root),
        "training_jsons": [
            str(path.expanduser().resolve())
            for path in args.training_json
        ],
        "test_image_count": len(source_images),
        "test_unique_name_count": len(source_names),
        "restorable_unique_image_count": len(restorable_names),
        "referenced_training_record_count": referenced_test_record_count,
        "unreferenced_test_image_count": len(unreferenced_names),
        "already_present_count": len(already_present),
        "pending_copy_count": len(pending),
        "copied_count": len(copied),
        "collision_count": len(collisions),
        "unreferenced_test_images": sorted(unreferenced_names),
        "already_present": already_present,
        "pending": pending,
        "copied": copied,
        "collisions": collisions,
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    mode = "COPY" if args.copy else "DRY-RUN"
    print("=" * 72)
    print("WARNING: DIAGNOSTIC TEST-SET LEAKAGE; NEVER REPORT THIS SCORE.")
    print("=" * 72)
    print(
        f"[{mode}] test_images={len(source_images)} "
        f"restorable={len(restorable_names)} "
        f"training_records={referenced_test_record_count} "
        f"already_present={len(already_present)} "
        f"pending={len(pending)} copied={len(copied)} "
        f"unreferenced={len(unreferenced_names)} "
        f"collisions={len(collisions)}"
    )
    for item in pending[:20]:
        print(f"  {'COPIED' if args.copy else 'WOULD_COPY'} {item['name']}")
    if len(pending) > 20:
        print(f"  ... and {len(pending) - 20} more")
    print(f"Report: {report_path}")
    if not args.training_json:
        print(
            "[WARNING] No --training-json was supplied, so dataset re-entry "
            "could not be verified."
        )
    if pending and not args.copy:
        print("No files were copied. Re-run with --copy after reviewing the report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
