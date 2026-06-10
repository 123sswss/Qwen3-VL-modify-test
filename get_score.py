from __future__ import annotations

import re
from pathlib import Path


ROOT_DIR = Path("experiment_outputs")
TARGET_NAME = "test.log"
STEP_MARKER = "[330/972]"
ACC_PATTERN = re.compile(r"Acc=\d+\.\d%")
SCORE_KEYWORD = "百分制分数"
SKIP_DIRS = {"trash"}


def should_skip(log_path: Path) -> bool:
    try:
        relative_parts = log_path.relative_to(ROOT_DIR).parts
    except ValueError:
        relative_parts = log_path.parts

    return any(part in SKIP_DIRS for part in relative_parts)


def extract_info(log_path: Path) -> dict[str, str]:
    upper_two_level_name = log_path.parents[1].name if len(log_path.parents) >= 2 else "路径层级不足"

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
        "folder_name": upper_two_level_name,
        "acc": acc_value,
        "score_line": score_line,
        "step_line": step_line,
        "log_path": str(log_path),
    }


def main() -> None:
    log_files = sorted(
        log_path
        for log_path in ROOT_DIR.rglob(TARGET_NAME)
        if not should_skip(log_path)
    )

    if not log_files:
        print(f"在 {ROOT_DIR} 下没有找到任何 {TARGET_NAME}")
        return

    print(f"共找到 {len(log_files)} 个 {TARGET_NAME}\n")

    for index, log_path in enumerate(log_files, start=1):
        info = extract_info(log_path)
        print(f"===== 结果 {index} =====")
        print(f"上2级文件夹名: {info['folder_name']}")
        print(f"{STEP_MARKER} 对应Acc: {info['acc']}")
        print(f"{SCORE_KEYWORD}行: {info['score_line']}")
        print(f"日志路径: {info['log_path']}")
        print()


if __name__ == "__main__":
    main()