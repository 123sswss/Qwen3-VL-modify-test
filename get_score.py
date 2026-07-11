from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


ROOT_DIR = Path("experiment_outputs")
OUTPUT_DIR = ROOT_DIR / "output"
TARGET_NAME = "test.log"
STEP_MARKER = "[330/972]"
ACC_PATTERN = re.compile(r"Acc=\d+\.\d%")
SCORE_KEYWORD = "百分制分数"
SCORE_VALUE_PATTERN = re.compile(r"百分制分数[^\d+-]*([+-]?\d+(?:\.\d+)?)")
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


def extract_info(log_path: Path) -> dict[str, str]:
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

    return {
        "folder_name": experiment_name,
        "acc": acc_value,
        "score_line": score_line,
        "step_line": step_line,
        "log_path": str(log_path),
        "diagnostics_by_stage": diagnostics_by_stage,
    }


def extract_score_value(log_path: Path) -> float | None:
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                if SCORE_KEYWORD not in raw_line:
                    continue
                match = SCORE_VALUE_PATTERN.search(raw_line)
                if match:
                    return float(match.group(1))
    except OSError:
        return None
    return None


def manage_best_checkpoint(checkpoint_root: Path, current_experiment_dir: Path) -> None:
    checkpoint_root = checkpoint_root.resolve()
    current_experiment_dir = current_experiment_dir.resolve()
    current_final_dir = current_experiment_dir / "final"
    current_log = current_experiment_dir / "eval" / TARGET_NAME

    if current_experiment_dir.parent != checkpoint_root:
        print(f"[KEEP][WARN] 当前实验目录不直属于 checkpoint 根目录，跳过清理: {current_experiment_dir}")
        return

    experiment_dirs = sorted(
        path.resolve()
        for path in checkpoint_root.iterdir()
        if path.is_dir() and path.name != "trash"
    )
    previous_dirs = [path for path in experiment_dirs if path != current_experiment_dir]

    # 第一次启用策略时，当前目录之外没有历史实验，直接保留第一个模型。
    if not previous_dirs:
        print(f"[KEEP] 首次使用保留策略，无条件保留: {current_final_dir}")
        return

    current_score = extract_score_value(current_log)
    if current_score is None:
        print(f"[KEEP][WARN] 无法解析当前百分制分数，为防止误删而保留: {current_final_dir}")
        return

    historical_scores: list[tuple[float, Path]] = []
    for experiment_dir in previous_dirs:
        score = extract_score_value(experiment_dir / "eval" / TARGET_NAME)
        if score is not None:
            historical_scores.append((score, experiment_dir))

    if not historical_scores:
        print(f"[KEEP] 未找到可解析的历史分数，将当前模型作为首个基准保留: score={current_score:.4f}")
        return

    historical_best = max(score for score, _ in historical_scores)
    if current_score < historical_best:
        print(
            f"[KEEP] 当前分数 {current_score:.4f} 低于历史最高分 "
            f"{historical_best:.4f}，删除当前 final。"
        )
        if current_final_dir.is_dir():
            shutil.rmtree(current_final_dir)
        return

    print(
        f"[KEEP] 当前分数 {current_score:.4f} 达到或刷新历史最高分 "
        f"{historical_best:.4f}，保留当前 final。"
    )
    if current_score > historical_best:
        for score, experiment_dir in historical_scores:
            old_final_dir = experiment_dir / "final"
            if score < current_score and old_final_dir.is_dir():
                shutil.rmtree(old_final_dir)
                print(f"[KEEP] 已删除旧低分 final: score={score:.4f} path={old_final_dir}")


def print_score_report() -> None:
    log_files = sorted(
        log_path
        for log_path in OUTPUT_DIR.glob(f"*/eval/{TARGET_NAME}")
        if not should_skip(log_path)
    )

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manage-best", type=Path, metavar="CURRENT_EXPERIMENT_DIR")
    parser.add_argument("--checkpoint-root", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.manage_best is not None:
        manage_best_checkpoint(args.checkpoint_root, args.manage_best)
    else:
        print_score_report()


if __name__ == "__main__":
    main()
