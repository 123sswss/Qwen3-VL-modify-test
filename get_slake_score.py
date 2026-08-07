from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "slake" / "outputs" / "mmrl"
EVAL_LOG_NAME = "eval.log"
DIAGNOSTICS_RELATIVE_PATH = Path("stage3/metrics/mmrl_diagnostics.jsonl")

METRIC_PATTERNS = (
    (
        "Overall Accuracy",
        re.compile(r"^Overall Accuracy:\s*[-+]?\d+(?:\.\d+)?\s*$", re.MULTILINE),
    ),
    (
        "Per Answer Type",
        re.compile(r"^Per Answer Type:\s*\{[^\r\n]*\}\s*$", re.MULTILINE),
    ),
    (
        "Per Question Type",
        re.compile(r"^Per Question Type:\s*\{[^\r\n]*\}\s*$", re.MULTILINE),
    ),
    (
        "Per Language",
        re.compile(r"^Per Language:\s*\{[^\r\n]*\}\s*$", re.MULTILINE),
    ),
    (
        "TTFT",
        re.compile(r"^TTFT:\s*\{[^\r\n]*\}\s*$", re.MULTILINE),
    ),
    (
        "TPOT",
        re.compile(
            r"^TPOT:\s*[-+]?\d+(?:\.\d+)?\s+s/token,\s*"
            r"[-+]?\d+(?:\.\d+)?\s+token/s\s*$",
            re.MULTILINE,
        ),
    ),
    (
        "Request Latency",
        re.compile(r"^Request Latency:\s*\{[^\r\n]*\}\s*$", re.MULTILINE),
    ),
)


def find_eval_logs() -> list[Path]:
    if not OUTPUT_DIR.is_dir():
        return []
    logs = [path for path in OUTPUT_DIR.glob(f"*/{EVAL_LOG_NAME}") if path.is_file()]
    return sorted(logs, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def extract_metrics(content: str) -> list[str]:
    extracted = []
    for label, pattern in METRIC_PATTERNS:
        matches = list(pattern.finditer(content))
        extracted.append(matches[-1].group(0).strip() if matches else f"{label}: 未找到")
    return extracted


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def format_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
        sep=" ", timespec="seconds"
    )


def print_diagnostics(experiment_dir: Path) -> None:
    diagnostics_path = experiment_dir / DIAGNOSTICS_RELATIVE_PATH
    print(f"诊断日志路径: {diagnostics_path}")
    if not diagnostics_path.is_file():
        print(f"诊断日志: 未找到 {DIAGNOSTICS_RELATIVE_PATH.as_posix()}")
        return

    content = read_text(diagnostics_path).strip()
    print(f"{DIAGNOSTICS_RELATIVE_PATH.as_posix()} 内容:")
    print(content if content else "文件为空")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 eval.log 修改时间快速汇总 SLAKE 最终测评结果。"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="额外显示每个实验的 Stage 3 MMRL 诊断日志。",
    )
    args = parser.parse_args()

    eval_logs = find_eval_logs()
    if not eval_logs:
        print(f"在 {OUTPUT_DIR} 下没有找到 */{EVAL_LOG_NAME}")
        return

    print(f"共找到 {len(eval_logs)} 个 {EVAL_LOG_NAME}（按修改时间从旧到新）\n")
    for index, eval_log in enumerate(eval_logs, start=1):
        experiment_dir = eval_log.parent
        print(f"===== 结果 {index} =====")
        print(f"实验名: {experiment_dir.name}")
        print(f"修改时间: {format_mtime(eval_log)}")
        print(f"日志路径: {eval_log}")
        for metric_line in extract_metrics(read_text(eval_log)):
            print(metric_line)
        if args.log:
            print_diagnostics(experiment_dir)
        print()


if __name__ == "__main__":
    main()
