## Summary

- 目标：扩展 `get_score.py`，在输出每个实验的 `test.log` 提取结果时，同步打印对应实验目录下 `stage4/metrics/mmrl_diagnostics.jsonl` 的内容。
- 目标：将搜索范围收紧为 `experiment_outputs/output/*/eval/test.log` 这一真实目录模式，并明确跳过 `output/trash` 树下的内容。
- 结果：脚本按实验目录遍历，输出实验名、Acc、百分制分数行、日志路径，以及同实验目录下 `stage4/metrics/mmrl_diagnostics.jsonl` 的文件内容；若诊断文件不存在则给出清晰占位提示。

## Current State Analysis

- 当前脚本位于 `get_score.py`，根目录常量为 `ROOT_DIR = Path("experiment_outputs")`，并通过 `ROOT_DIR.rglob("test.log")` 递归搜索日志。
- 当前过滤逻辑由 `should_skip(log_path)` 完成，按路径片段是否包含 `trash` 做排除；这能挡住现有 `experiment_outputs/output/trash/**`，但语义上仍然是“全树递归后再过滤”，不是按用户要求的 `output/*/eval/test.log` 模式定向搜索。
- 当前 `extract_info(log_path)` 只读取 `test.log`，返回上 2 级目录名、匹配到的 `Acc=...%`、`百分制分数` 行、`[330/972]` 对应行和日志路径。
- 只读勘察确认：真实日志路径形如 `experiment_outputs/output/visual_router_v2_2/eval/test.log`，对应诊断文件形如 `experiment_outputs/output/visual_router_v2_2/stage4/metrics/mmrl_diagnostics.jsonl`。
- 只读勘察确认：`mmrl_diagnostics.jsonl` 是逐行 JSON 文本文件，不需要额外解析也可以直接按文本打印。

## Proposed Changes

- 修改 `get_score.py` 的日志发现方式。
  - 将“从 `experiment_outputs` 全量 `rglob` 搜索 `test.log`”改为“从 `experiment_outputs/output` 下只匹配直接实验目录的 `*/eval/test.log`”。
  - 这样会天然只覆盖单层实验目录，避免把 `output/trash/<experiment>/eval/test.log` 误纳入结果。
  - 保留或重写 `should_skip()` 时，逻辑要聚焦为“仅跳过 `output` 下名为 `trash` 的目录”，避免未来把别处合法路径里恰好叫 `trash` 的片段也一并排除。

- 扩展 `extract_info(log_path)` 的返回信息。
  - 以 `log_path.parents[1]` 作为实验目录，即 `experiment_outputs/output/<experiment>`。
  - 在该实验目录下拼出 `stage4/metrics/mmrl_diagnostics.jsonl` 的绝对/相对路径。
  - 新增一个读取诊断文件内容的辅助函数，例如 `read_stage4_metrics(experiment_dir: Path) -> str`，按 UTF-8、`errors="ignore"` 读取全文。
  - 若文件不存在，返回固定提示，如“未找到 stage4/metrics/mmrl_diagnostics.jsonl”；若文件为空，返回“文件为空”。

- 调整命令行输出格式。
  - 在每个实验结果块中，继续打印实验名、Acc、百分制分数行、日志路径。
  - 新增一行打印诊断文件路径，便于定位来源。
  - 新增一个段落打印 `mmrl_diagnostics.jsonl` 的内容，采用“标签 + 原文内容”的形式输出，便于肉眼对照。
  - 若内容是多行，按原始换行保留；若缺失，则打印占位提示而不是抛异常。

- 保持兼容性与最小改动。
  - 不修改现有正则提取 `Acc=\d+\.\d%` 与 `百分制分数` 的行为。
  - 不引入新依赖，仅使用标准库 `pathlib` 与现有文件读取方式。
  - 若现有“上 2 级文件夹名”字段实质对应实验目录名，则保留该用户可见输出，避免破坏已有使用习惯。

## Assumptions & Decisions

- 假设用户所说“打印 `stage4/metrics/mmrl_diagnostics.jsonl` 文件夹中的内容”实际指“打印该文件的内容”；仓库中的真实对象也是文件而非目录。
- 决定不对 `mmrl_diagnostics.jsonl` 做 JSON 结构化解析，只按原始文本输出，因为样本已是标准 JSONL，用户需求是“同步打印输出内容”而不是“抽取字段”。
- 决定以“定向搜索 `experiment_outputs/output/*/eval/test.log`”作为主策略，而不是继续全树递归后依赖过滤函数补救；这与用户描述完全一致，也最直接排除了 `output/trash/**`。
- 决定对缺失文件、空文件、读取失败使用显式文本兜底，避免单个实验异常影响整个脚本遍历。

## Verification Steps

- 运行脚本并确认只统计 `experiment_outputs/output/*/eval/test.log` 下的实验，不包含 `experiment_outputs/output/trash/**`。
- 对任一已知实验目录检查输出中是否同时出现：
  - 实验目录名；
  - `[330/972]` 对应的 `Acc`；
  - `百分制分数` 对应行；
  - `stage4/metrics/mmrl_diagnostics.jsonl` 的路径；
  - 该 JSONL 文件的完整文本内容。
- 选取一个不存在 `stage4/metrics/mmrl_diagnostics.jsonl` 的实验目录或临时构造场景，确认脚本输出缺失提示且不会中断遍历。
- 若仓库内有可用静态检查，再对 `get_score.py` 做一次语法/诊断检查，确保未引入类型或语法错误。
