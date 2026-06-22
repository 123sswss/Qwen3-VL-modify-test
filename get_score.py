from __future__ import annotations

import re
from pathlib import Path


ROOT_DIR = Path("experiment_outputs")
OUTPUT_DIR = ROOT_DIR / "output"
TARGET_NAME = "test.log"
STEP_MARKER = "[330/972]"
ACC_PATTERN = re.compile(r"Acc=\d+\.\d%")
SCORE_KEYWORD = "百分制分数"
DIAGNOSTICS_RELATIVE_PATH = Path("stage4") / "metrics" / "mmrl_diagnostics.jsonl"
MISSING_DIAGNOSTICS_MESSAGE = "未找到 stage4/metrics/mmrl_diagnostics.jsonl"
EMPTY_DIAGNOSTICS_MESSAGE = "stage4/metrics/mmrl_diagnostics.jsonl 文件为空"


def should_skip(log_path: Path) -> bool:
    try:
        relative_parts = log_path.relative_to(OUTPUT_DIR).parts
    except ValueError:
        relative_parts = log_path.parts

    return bool(relative_parts) and relative_parts[0] == "trash"


def read_stage4_metrics(experiment_dir: Path) -> tuple[str, str]:
    diagnostics_path = experiment_dir / DIAGNOSTICS_RELATIVE_PATH

    if not diagnostics_path.is_file():
        return str(diagnostics_path), MISSING_DIAGNOSTICS_MESSAGE

    try:
        diagnostics_content = diagnostics_path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError as exc:
        return str(diagnostics_path), f"读取失败: {exc}"

    if not diagnostics_content:
        return str(diagnostics_path), EMPTY_DIAGNOSTICS_MESSAGE

    return str(diagnostics_path), diagnostics_content


def extract_info(log_path: Path) -> dict[str, str]:
    experiment_dir = log_path.parents[1] if len(log_path.parents) >= 2 else None
    experiment_name = experiment_dir.name if experiment_dir is not None else "路径层级不足"
    diagnostics_path, diagnostics_content = (
        read_stage4_metrics(experiment_dir)
        if experiment_dir is not None
        else ("路径层级不足", "路径层级不足，无法定位 stage4/metrics/mmrl_diagnostics.jsonl")
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
        "diagnostics_path": diagnostics_path,
        "diagnostics_content": diagnostics_content,
    }


def main() -> None:
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
        print(f"stage4 指标路径: {info['diagnostics_path']}")
        print("stage4/metrics/mmrl_diagnostics.jsonl 内容:")
        print(info["diagnostics_content"])
        print()


if __name__ == "__main__":
    main()
