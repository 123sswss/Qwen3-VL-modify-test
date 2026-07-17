from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT_DIR = Path("experiment_outputs")
OUTPUT_DIR = ROOT_DIR / "output"
TARGET_NAME = "test.log"
STEP_MARKER = "[330/972]"
ACC_PATTERN = re.compile(r"Acc=\d+\.\d%")
SCORE_KEYWORD = "百分制分数"
SCORE_PATTERN = re.compile(r"百分制分数\s*[:：]\s*(-?\d+(?:\.\d+)?)")
DIAGNOSTICS_FILENAME = "mmrl_diagnostics.jsonl"
DIAGNOSTIC_STAGES = (3, 4)
SEED_LOG_PATTERN = re.compile(r"seed=(\d+)")


def should_skip(log_path: Path) -> bool:
    try:
        relative_parts = log_path.relative_to(OUTPUT_DIR).parts
    except ValueError:
        relative_parts = log_path.parts

    return bool(relative_parts) and relative_parts[0] == "trash"


def diagnostics_relative_path(stage_id: int) -> Path:
    return Path(f"stage{stage_id}") / "metrics" / DIAGNOSTICS_FILENAME


def read_stage_metrics(experiment_dir: Path, stage_id: int) -> tuple[str, str]:
    relative_path = diagnostics_relative_path(stage_id)
    diagnostics_path = experiment_dir / relative_path

    if not diagnostics_path.is_file():
        return str(diagnostics_path), f"未找到 {relative_path.as_posix()}"

    try:
        diagnostics_content = diagnostics_path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError as exc:
        return str(diagnostics_path), f"读取失败: {exc}"

    if not diagnostics_content:
        return str(diagnostics_path), f"{relative_path.as_posix()} 文件为空"

    return str(diagnostics_path), diagnostics_content


def read_experiment_seed(experiment_dir: Path) -> int | None:
    for manifest_path in sorted(experiment_dir.glob("*_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            seed = manifest.get("train_cfg", {}).get("seed")
            if seed is not None:
                return int(seed)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue

    train_log = experiment_dir / "train.log"
    if train_log.is_file():
        try:
            for line in train_log.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "[Reproducibility]" not in line:
                    continue
                match = SEED_LOG_PATTERN.search(line)
                if match:
                    return int(match.group(1))
        except OSError:
            pass
    return None


def read_epoch_scores(experiment_dir: Path) -> list[tuple[str, float | None]]:
    results = []
    eval_root = experiment_dir / "eval_epochs"
    if not eval_root.is_dir():
        return results
    for log_path in sorted(eval_root.glob("stage*_epoch*/test.log")):
        results.append((log_path.parent.name, read_score_value(log_path)))
    return results


def extract_info(log_path: Path) -> dict[str, object]:
    experiment_dir = log_path.parents[1] if len(log_path.parents) >= 2 else None
    experiment_name = experiment_dir.name if experiment_dir is not None else "路径层级不足"
    diagnostics_by_stage = {}
    for stage_id in DIAGNOSTIC_STAGES:
        diagnostics_by_stage[stage_id] = (
            read_stage_metrics(experiment_dir, stage_id)
            if experiment_dir is not None
            else ("路径层级不足", f"路径层级不足，无法定位 {diagnostics_relative_path(stage_id).as_posix()}")
        )

    step_line = "未找到"
    acc_value = "未找到"
    score_line = "未找到"
    score_value = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if STEP_MARKER in line:
                step_line = line.strip()
                acc_match = ACC_PATTERN.search(line)
                if acc_match:
                    acc_value = acc_match.group(0)

            if SCORE_KEYWORD in line:
                score_line = line.strip()
                score_match = SCORE_PATTERN.search(line)
                if score_match:
                    score_value = float(score_match.group(1))

    return {
        "folder_name": experiment_name,
        "seed": read_experiment_seed(experiment_dir) if experiment_dir is not None else None,
        "epoch_scores": read_epoch_scores(experiment_dir) if experiment_dir is not None else [],
        "acc": acc_value,
        "score_line": score_line,
        "score_value": score_value,
        "step_line": step_line,
        "log_path": str(log_path),
        "diagnostics_by_stage": diagnostics_by_stage,
    }


def find_log_files() -> list[Path]:
    return sorted(
        log_path
        for log_path in OUTPUT_DIR.glob(f"*/eval/{TARGET_NAME}")
        if not should_skip(log_path)
    )


def read_score_value(log_path: Path) -> float | None:
    score_value = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if SCORE_KEYWORD not in line:
                continue
            score_match = SCORE_PATTERN.search(line)
            if score_match:
                score_value = float(score_match.group(1))
    return score_value


def find_scored_checkpoints() -> list[tuple[str, float]]:
    if not OUTPUT_DIR.is_dir():
        return []

    scored_checkpoints = []
    for experiment_dir in sorted(OUTPUT_DIR.iterdir()):
        if not experiment_dir.is_dir() or experiment_dir.name == "trash":
            continue
        if not (experiment_dir / "final").is_dir():
            continue

        log_path = experiment_dir / "eval" / TARGET_NAME
        if not log_path.is_file():
            continue
        score_value = read_score_value(log_path)
        if score_value is not None:
            scored_checkpoints.append((experiment_dir.name, score_value))

    return scored_checkpoints


def checkpoint_extrema_names() -> list[str]:
    scored_checkpoints = find_scored_checkpoints()

    if not scored_checkpoints:
        return []

    scores = [score for _, score in scored_checkpoints]
    min_score = min(scores)
    max_score = max(scores)
    return sorted(
        name
        for name, score in scored_checkpoints
        if score == min_score or score == max_score
    )


def checkpoint_retention_plan() -> list[tuple[str, str, float]]:
    scored_checkpoints = find_scored_checkpoints()
    keep_names = set(checkpoint_extrema_names())
    return [
        ("KEEP" if name in keep_names else "PRUNE", name, score)
        for name, score in scored_checkpoints
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-extrema-names",
        action="store_true",
        help="仅输出当前仍有 final 目录的最高/最低分实验名",
    )
    parser.add_argument(
        "--checkpoint-retention-plan",
        action="store_true",
        help="输出 KEEP/PRUNE、实验名和分数，供实验脚本安全清理 final",
    )
    args = parser.parse_args()
    log_files = find_log_files()

    if args.checkpoint_extrema_names:
        for name in checkpoint_extrema_names():
            print(name)
        return

    if args.checkpoint_retention_plan:
        for action, name, score in checkpoint_retention_plan():
            print(f"{action}\t{name}\t{score:.12g}")
        return

    if not log_files:
        print(f"在 {OUTPUT_DIR} 下没有找到任何匹配 */eval/{TARGET_NAME} 的日志")
        return

    print(f"共找到 {len(log_files)} 个 {TARGET_NAME}\n")

    for index, log_path in enumerate(log_files, start=1):
        info = extract_info(log_path)
        print(f"===== 结果 {index} =====")
        print(f"seed: {info['seed'] if info['seed'] is not None else '未找到'}")
        print(f"上2级文件夹名: {info['folder_name']}")
        print(f"{STEP_MARKER} 对应Acc: {info['acc']}")
        print(f"{SCORE_KEYWORD}行: {info['score_line']}")
        print(f"日志路径: {info['log_path']}")
        if info["epoch_scores"]:
            epoch_curve = ", ".join(
                f"{name}={score if score is not None else '未找到'}"
                for name, score in info["epoch_scores"]
            )
            print(f"逐epoch分数: {epoch_curve}")
        for stage_id in DIAGNOSTIC_STAGES:
            diagnostics_path, diagnostics_content = info["diagnostics_by_stage"][stage_id]
            print(f"stage{stage_id} 指标路径: {diagnostics_path}")
            print(f"stage{stage_id}/metrics/{DIAGNOSTICS_FILENAME} 内容:")
            print(diagnostics_content)
        print()


if __name__ == "__main__":
    main()
