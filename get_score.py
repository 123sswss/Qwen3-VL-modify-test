from __future__ import annotations

import argparse
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


def checkpoint_extrema_names(log_files: list[Path]) -> list[str]:
    scored_checkpoints = []
    for log_path in log_files:
        experiment_dir = log_path.parents[1]
        if not (experiment_dir / "final").is_dir():
            continue
        score_value = read_score_value(log_path)
        if score_value is not None:
            scored_checkpoints.append((experiment_dir.name, score_value))

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-extrema-names",
        action="store_true",
        help="仅输出当前仍有 final 目录的最高/最低分实验名",
    )
    args = parser.parse_args()
    log_files = find_log_files()

    if args.checkpoint_extrema_names:
        for name in checkpoint_extrema_names(log_files):
            print(name)
        return

    if not log_files:
        print(f"在 {OUTPUT_DIR} 下没有找到任何匹配 */eval/{TARGET_NAME} 的日志")
        return

    print(f"共找到 {len(log_files)} 个 {TARGET_NAME}\n")

    for index, log_path in enumerate(log_files, start=1):
        info = extract_info(log_path)
        print(f"===== 结果 {index} =====")
        print(f"上2级文件夹名: {info['folder_name']}")
        print(f"{STEP_MARKER} 对应Acc: {info['acc']}")
        print(f"{SCORE_KEYWORD}行: {info['score_line']}")
        print(f"日志路径: {info['log_path']}")
        for stage_id in DIAGNOSTIC_STAGES:
            diagnostics_path, diagnostics_content = info["diagnostics_by_stage"][stage_id]
            print(f"stage{stage_id} 指标路径: {diagnostics_path}")
            print(f"stage{stage_id}/metrics/{DIAGNOSTICS_FILENAME} 内容:")
            print(diagnostics_content)
        print()


if __name__ == "__main__":
    main()
