import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


SEED = 20260727
DEFAULT_MIN_AREA_RATIO = 0.001
DEFAULT_MIN_SIDE_PIXELS = 14.0

CATEGORY_TRANSLATIONS = {
    "1": {
        "tower id plate": "输电塔上的塔号牌",
        "spacer": "导线间隔棒",
        "polymer insulator upper shackle": "复合绝缘子顶部挂环",
        "polymer insulator": "复合绝缘子",
        "polymer insulator lower shackle": "复合绝缘子底部挂环",
        "yoke suspension": "悬垂式均压环",
        "glass insulator big shackle": "玻璃绝缘子串顶部大挂环",
        "stockbridge damper": "防振锤",
        "lightning rod shackle": "避雷针挂环",
        "nest": "输电塔上的鸟巢",
        "kite": "缠绕在线路上的风筝",
        "trash": "悬挂在设备上的垃圾",
        "balloon": "飘挂在线路上的气球",
        "lightning rod suspension": "避雷针的悬挂装置",
        "spiral damper": "螺旋防振鞭",
        "polymer insulator tower shackle": "复合绝缘子塔侧挂环",
        "glass insulator": "玻璃绝缘子",
        "vari-grip": "预绞丝护线条",
        "yoke": "均压联板",
        "defect": "设备上发现的明显缺陷",
        "insulator": "绝缘子",
        "burst": "发生自爆或破损的绝缘子",
        "glass insulator small shackle": "玻璃绝缘子串底部小挂环",
        "glass insulator tower shackle": "玻璃绝缘子塔侧挂环",
        "sphere": "航空警示球",
    },
    "14": {
        "Bend Part": "部件弯曲变形",
        "Vegetation": "植被或异物遮挡",
        "Missing Plate": "牌号或标识牌缺失",
        "Barbed Wire Missing": "防鸟刺/刺线缺失",
        "Bolt Missing": "螺栓缺失",
        "Foreign Material": "异物缠绕",
        "Split Pin Missing": "开口销缺失",
        "Loose Part": "部件松动",
        "Position andCondition": "位置或状态异常",
        "Missing Part": "部件缺失",
    },
}

X_ZONES = (
    "紧靠左边缘",
    "左侧",
    "中央偏左",
    "中央",
    "中央偏右",
    "右侧",
    "紧靠右边缘",
)
Y_ZONES = (
    "紧靠上边缘",
    "上部",
    "中央偏上",
    "中央",
    "中央偏下",
    "下部",
    "紧靠下边缘",
)
ZONE_BOUNDARIES = (0.08, 0.24, 0.40, 0.60, 0.76, 0.92)


def zone_index(value: float) -> int:
    for index, boundary in enumerate(ZONE_BOUNDARIES):
        if value < boundary:
            return index
    return len(ZONE_BOUNDARIES)


def center_location(cx: float, cy: float) -> str:
    x_zone = X_ZONES[zone_index(cx)]
    y_zone = Y_ZONES[zone_index(cy)]
    if x_zone == "中央" and y_zone == "中央":
        return "画面中央"
    if x_zone == "中央":
        if y_zone.startswith("紧靠"):
            return f"画面{y_zone}的中央区域"
        return f"画面{y_zone}区域"
    if y_zone == "中央":
        if x_zone.startswith("紧靠"):
            return f"画面{x_zone}的中部区域"
        return f"画面{x_zone}区域"
    return f"画面{x_zone}、{y_zone}区域"


def coarse_point(x_value: float, y_value: float) -> str:
    if x_value < 1 / 3:
        x_name = "左"
    elif x_value < 2 / 3:
        x_name = "中"
    else:
        x_name = "右"
    if y_value < 1 / 3:
        y_name = "上"
    elif y_value < 2 / 3:
        y_name = "中"
    else:
        y_name = "下"
    if x_name == "中" and y_name == "中":
        return "画面中央"
    if x_name == "中":
        return f"画面{y_name}部中央"
    if y_name == "中":
        return f"画面{x_name}侧中央"
    return f"画面{x_name}{y_name}方"


def rounded_percent(value: float) -> int:
    return max(0, min(100, int(round(value * 20.0) * 5)))


def size_description(area_ratio: float) -> str:
    if area_ratio < 0.0003:
        return "范围极小"
    if area_ratio < 0.002:
        return "范围很小"
    if area_ratio < 0.01:
        return "范围较小"
    if area_ratio < 0.08:
        return "大小中等"
    if area_ratio < 0.25:
        return "范围较大"
    return "占据画面较大范围"


def geometry_description(target: Dict, force_precise: bool = False) -> Tuple[str, str]:
    left = target["left"]
    top = target["top"]
    right = target["right"]
    bottom = target["bottom"]
    width = target["width"]
    height = target["height"]
    cx = target["center_x"]
    cy = target["center_y"]
    area = target["area_ratio"]
    aspect = max(width / max(height, 1e-9), height / max(width, 1e-9))

    if aspect >= 4.0 and max(width, height) >= 0.12:
        if width >= height:
            location = (
                f"沿{center_location(cx, cy)}，"
                f"从{coarse_point(left, cy)}向{coarse_point(right, cy)}横向延伸"
            )
            strategy = "horizontal_extent"
        else:
            location = (
                f"沿{center_location(cx, cy)}，"
                f"从{coarse_point(cx, top)}向{coarse_point(cx, bottom)}纵向延伸"
            )
            strategy = "vertical_extent"
    elif area >= 0.12 or width >= 0.45 or height >= 0.45:
        start = coarse_point(left, top)
        end = coarse_point(right, bottom)
        if start == end:
            location = f"主要位于{center_location(cx, cy)}"
        else:
            location = f"主要覆盖从{start}到{end}的区域"
        strategy = "box_extent"
    else:
        location = f"位于{center_location(cx, cy)}"
        strategy = "center_zone"

    tiny = area < 0.002 or min(width, height) < 0.025
    if tiny or force_precise:
        location += (
            f"，中心约在画面横向{rounded_percent(cx)}%、"
            f"纵向{rounded_percent(cy)}%处"
        )
        strategy += "_precise"
    if force_precise:
        location += (
            f"，框宽约占画面{rounded_percent(width)}%、"
            f"框高约占画面{rounded_percent(height)}%"
        )
    return location, strategy


def build_answer(dataset_name: str, location: str, category_name_zh: str) -> str:
    if location.startswith("位于"):
        natural_location = location[len("位于"):]
        if not natural_location.endswith("处"):
            natural_location += "处"
        if dataset_name == "14":
            return f"{natural_location}存在{category_name_zh}。"
        return f"{natural_location}有一个{category_name_zh}。"
    if dataset_name == "14":
        return f"{location}所对应的异常是{category_name_zh}。"
    if location.startswith("主要覆盖") and location.endswith("的区域"):
        location = location[:-3]
    return f"{location}的目标是{category_name_zh}。"


def normalize_bbox(annotation: Dict, image: Dict) -> Dict:
    image_width = float(image["width"])
    image_height = float(image["height"])
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size: {image}")

    x_value, y_value, width, height = [
        float(value)
        for value in annotation["bbox"]
    ]
    left = max(0.0, min(1.0, x_value / image_width))
    top = max(0.0, min(1.0, y_value / image_height))
    right = max(left, min(1.0, (x_value + width) / image_width))
    bottom = max(top, min(1.0, (y_value + height) / image_height))
    normalized_width = right - left
    normalized_height = bottom - top
    if normalized_width <= 0 or normalized_height <= 0:
        raise ValueError(
            f"Invalid bbox after clipping: image={image['file_name']} "
            f"bbox={annotation['bbox']}"
        )
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": normalized_width,
        "height": normalized_height,
        "center_x": (left + right) / 2,
        "center_y": (top + bottom) / 2,
        "area_ratio": normalized_width * normalized_height,
    }


def collect_excluded_names(roots: Iterable[Path]) -> Set[str]:
    names = set()
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"exclude root is not a directory: {root}")
        names.update(path.name for path in root.rglob("*") if path.is_file())
    return names


def load_targets(
    coco_paths: Sequence[Path],
    dataset_name: str,
    excluded_names: Set[str],
    min_area_ratio: float,
    min_side_pixels: float,
) -> Tuple[List[Dict], Dict[int, str], Counter, Counter]:
    translations = CATEGORY_TRANSLATIONS[dataset_name]
    all_targets = []
    category_names = None
    source_counts = Counter()
    too_small_counts = Counter()
    seen_image_names = set()

    for raw_path in coco_paths:
        path = raw_path.expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
        current_names = {
            int(category["id"]): str(category["name"])
            for category in coco["categories"]
        }
        if set(current_names.values()) != set(translations):
            raise ValueError(
                f"Category/translation mismatch in {path}: "
                f"missing={sorted(set(current_names.values()) - set(translations))} "
                f"extra={sorted(set(translations) - set(current_names.values()))}"
            )
        if category_names is None:
            category_names = current_names
        elif current_names != category_names:
            raise ValueError(f"COCO category mappings differ: {path}")

        images = {int(image["id"]): image for image in coco["images"]}
        annotations_by_image = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        for image_id, image in images.items():
            image_name = Path(image["file_name"]).name
            if image_name in excluded_names:
                continue
            if image_name in seen_image_names:
                raise ValueError(f"Duplicate image basename across COCO files: {image_name}")
            seen_image_names.add(image_name)
            source_counts[path.name] += 1
            for annotation in annotations_by_image.get(image_id, []):
                category_id = int(annotation["category_id"])
                geometry = normalize_bbox(annotation, image)
                _, _, bbox_width, bbox_height = [
                    float(value)
                    for value in annotation["bbox"]
                ]
                if (
                    geometry["area_ratio"] < min_area_ratio
                    or min(bbox_width, bbox_height) < min_side_pixels
                ):
                    too_small_counts[category_names[category_id]] += 1
                    continue
                all_targets.append(
                    {
                        "dataset": dataset_name,
                        "source_coco": str(path),
                        "source_split": path.stem.replace("_coco", ""),
                        "image": image_name,
                        "image_width": int(image["width"]),
                        "image_height": int(image["height"]),
                        "category_id": category_id,
                        "category_name": category_names[category_id],
                        "category_name_zh": translations[category_names[category_id]],
                        "bbox": [float(value) for value in annotation["bbox"]],
                        **geometry,
                    }
                )
    return all_targets, category_names or {}, source_counts, too_small_counts


def build_candidates(targets: List[Dict], dataset_name: str) -> Tuple[List[Dict], int]:
    targets_by_image = defaultdict(list)
    for target in targets:
        targets_by_image[target["image"]].append(target)

    candidates = []
    ambiguous_skipped = 0
    for image_name, image_targets in targets_by_image.items():
        deduplicated = {}
        for target in image_targets:
            key = (
                target["category_id"],
                rounded_percent(target["center_x"]),
                rounded_percent(target["center_y"]),
                rounded_percent(target["width"]),
                rounded_percent(target["height"]),
            )
            existing = deduplicated.get(key)
            if existing is None or target["area_ratio"] > existing["area_ratio"]:
                deduplicated[key] = target
        image_targets = list(deduplicated.values())

        base_locations = Counter(
            geometry_description(target, force_precise=False)[0]
            for target in image_targets
        )
        described = []
        for target in image_targets:
            base_location, _ = geometry_description(target, force_precise=False)
            nearby_different_class = any(
                other["category_id"] != target["category_id"]
                and math.dist(
                    (other["center_x"], other["center_y"]),
                    (target["center_x"], target["center_y"]),
                ) < 0.08
                for other in image_targets
            )
            force_precise = base_locations[base_location] > 1 or nearby_different_class
            location, strategy = geometry_description(target, force_precise=force_precise)
            described.append((target, location, strategy))

        descriptor_categories = defaultdict(set)
        for target, location, _ in described:
            descriptor_categories[location].add(target["category_id"])

        for target, location, strategy in described:
            if len(descriptor_categories[location]) > 1:
                ambiguous_skipped += 1
                continue
            size_text = size_description(target["area_ratio"])
            if dataset_name == "14":
                prompt = (
                    "<image>\n"
                    f"请识别{location}、{size_text}的异常区域。"
                    "只用一句简短中文回答其位置和异常类别。"
                )
            else:
                prompt = (
                    "<image>\n"
                    f"请识别{location}、{size_text}的目标。"
                    "只用一句简短中文回答其位置和目标类别。"
                )
            answer = build_answer(
                dataset_name,
                location,
                target["category_name_zh"],
            )
            if target["category_name_zh"] in prompt:
                raise ValueError(f"Answer leakage in generated prompt: {image_name}")
            candidates.append(
                {
                    **target,
                    "location": location,
                    "location_strategy": strategy,
                    "size_description": size_text,
                    "prompt": prompt,
                    "answer": answer,
                }
            )
    return candidates, ambiguous_skipped


def select_balanced_candidates(
    candidates: List[Dict],
    max_per_image: int,
    max_per_category: int,
    rng: random.Random,
) -> List[Dict]:
    availability = Counter(candidate["category_id"] for candidate in candidates)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda item: availability[item["category_id"]])

    image_counts = Counter()
    category_counts = Counter()
    selected = []
    for candidate in shuffled:
        image_name = candidate["image"]
        category_id = candidate["category_id"]
        if image_counts[image_name] >= max_per_image:
            continue
        if category_counts[category_id] >= max_per_category:
            continue
        selected.append(candidate)
        image_counts[image_name] += 1
        category_counts[category_id] += 1
    rng.shuffle(selected)
    return selected


def build_output_records(selected: List[Dict]) -> List[Dict]:
    records = []
    for index, candidate in enumerate(selected, start=1):
        records.append(
            {
                "id": (
                    f"spatial_{candidate['dataset']}_{index:06d}_"
                    f"{Path(candidate['image']).stem}"
                ),
                "image": candidate["image"],
                "conversations": [
                    {"from": "human", "value": candidate["prompt"]},
                    {"from": "gpt", "value": candidate["answer"]},
                ],
                "metadata": {
                    "source_dataset": candidate["dataset"],
                    "source_coco": candidate["source_coco"],
                    "source_split": candidate["source_split"],
                    "category_id": candidate["category_id"],
                    "category_name": candidate["category_name"],
                    "category_name_zh": candidate["category_name_zh"],
                    "bbox": candidate["bbox"],
                    "normalized_bbox": [
                        candidate["left"],
                        candidate["top"],
                        candidate["right"],
                        candidate["bottom"],
                    ],
                    "area_ratio": candidate["area_ratio"],
                    "location": candidate["location"],
                    "location_strategy": candidate["location_strategy"],
                    "size_description": candidate["size_description"],
                },
            }
        )
    return records


def audit_records(
    records: List[Dict],
    excluded_names: Set[str],
    max_per_image: int,
    max_per_category: int,
) -> Dict:
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Generated record IDs are not unique")
    leaked_images = sorted(
        {record["image"] for record in records}
        & excluded_names
    )
    if leaked_images:
        raise ValueError(f"Excluded test images leaked into output: {leaked_images[:5]}")

    image_counts = Counter(record["image"] for record in records)
    category_counts = Counter(
        record["metadata"]["category_name_zh"]
        for record in records
    )
    if image_counts and max(image_counts.values()) > max_per_image:
        raise ValueError(f"Per-image cap failed: {max(image_counts.values())}")
    if category_counts and max(category_counts.values()) > max_per_category:
        raise ValueError(f"Per-category cap failed: {max(category_counts.values())}")
    for record in records:
        prompt = record["conversations"][0]["value"]
        answer = record["conversations"][1]["value"]
        category_name = record["metadata"]["category_name_zh"]
        if category_name in prompt or category_name not in answer:
            raise ValueError(f"Prompt/answer semantic audit failed: {record['id']}")

    return {
        "record_count": len(records),
        "unique_image_count": len(image_counts),
        "max_records_per_image": max(image_counts.values(), default=0),
        "category_counts": dict(sorted(category_counts.items())),
        "location_strategy_counts": dict(
            sorted(
                Counter(
                    record["metadata"]["location_strategy"]
                    for record in records
                ).items()
            )
        ),
        "size_counts": dict(
            sorted(
                Counter(
                    record["metadata"]["size_description"]
                    for record in records
                ).items()
            )
        ),
        "excluded_image_overlap": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic spatial-grounding conversations directly "
            "from COCO boxes and Chinese category labels."
        )
    )
    parser.add_argument("--dataset", choices=sorted(CATEGORY_TRANSLATIONS), required=True)
    parser.add_argument(
        "--coco",
        action="append",
        type=Path,
        required=True,
        help="COCO annotation JSON. Repeat for multiple clean training splits.",
    )
    parser.add_argument(
        "--exclude-root",
        action="append",
        type=Path,
        default=[],
        help="Image directory whose basenames must never enter the output.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-image", type=int, default=2)
    parser.add_argument("--max-per-category", type=int, default=1000)
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=DEFAULT_MIN_AREA_RATIO,
    )
    parser.add_argument(
        "--min-side-pixels",
        type=float,
        default=DEFAULT_MIN_SIDE_PIXELS,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if (
        args.max_per_image < 1
        or args.max_per_category < 1
        or args.min_area_ratio < 0
        or args.min_side_pixels < 0
    ):
        parser.error("sample caps and minimum box thresholds must be non-negative")

    excluded_names = collect_excluded_names(args.exclude_root)
    targets, category_names, source_counts, too_small_counts = load_targets(
        args.coco,
        args.dataset,
        excluded_names,
        min_area_ratio=args.min_area_ratio,
        min_side_pixels=args.min_side_pixels,
    )
    candidates, ambiguous_skipped = build_candidates(targets, args.dataset)
    selected = select_balanced_candidates(
        candidates,
        max_per_image=args.max_per_image,
        max_per_category=args.max_per_category,
        rng=random.Random(args.seed + int(args.dataset)),
    )
    records = build_output_records(selected)
    audit = audit_records(
        records,
        excluded_names=excluded_names,
        max_per_image=args.max_per_image,
        max_per_category=args.max_per_category,
    )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": args.dataset,
        "seed": args.seed,
        "coco_paths": [str(path.expanduser().resolve()) for path in args.coco],
        "excluded_roots": [
            str(path.expanduser().resolve())
            for path in args.exclude_root
        ],
        "excluded_name_count": len(excluded_names),
        "source_image_counts": dict(sorted(source_counts.items())),
        "min_area_ratio": args.min_area_ratio,
        "min_side_pixels": args.min_side_pixels,
        "too_small_target_count": sum(too_small_counts.values()),
        "too_small_category_counts": dict(sorted(too_small_counts.items())),
        "raw_target_count": len(targets),
        "candidate_count": len(candidates),
        "ambiguous_target_count": ambiguous_skipped,
        "max_per_image": args.max_per_image,
        "max_per_category": args.max_per_category,
        "category_id_to_name": category_names,
        **audit,
    }
    report_path = output_path.with_suffix(".report.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(
        "[SPATIAL_DATASET] "
        f"dataset={args.dataset} kept_targets={len(targets)} "
        f"too_small_skipped={sum(too_small_counts.values())} "
        f"candidates={len(candidates)} ambiguous_skipped={ambiguous_skipped} "
        f"records={len(records)} unique_images={audit['unique_image_count']} "
        f"excluded_overlap={audit['excluded_image_overlap']}"
    )
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
