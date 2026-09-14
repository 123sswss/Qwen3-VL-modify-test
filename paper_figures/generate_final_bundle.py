#!/usr/bin/env python3
"""Generate every prepared QDPT paper figure into one self-contained folder."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "paper_figures" / "output" / "final_bundle",
    )
    parser.add_argument("--model-path", default="/root/autodl-tmp/model")
    parser.add_argument(
        "--data-root", default="/root/autodl-tmp/dataset/pathVQA"
    )
    parser.add_argument(
        "--cache-dir", default="/root/autodl-tmp/dataset/pathVQA/.hf_cache"
    )
    return parser.parse_args()


def select_candidate_rows(candidate_data: dict[str, Any]) -> tuple[int, int]:
    candidates = candidate_data.get("candidates", [])
    if not candidates:
        raise ValueError("No same-image, different-question-type candidate exists")
    first = candidates[0]["records"][0]
    second = next(
        (
            row
            for row in candidates[0]["records"]
            if row["question_type"] != first["question_type"]
        ),
        None,
    )
    if second is None:
        raise ValueError("Top attention candidate lacks distinct question types")
    return int(first["row_index"]), int(second["row_index"])


def run_step(
    name: str,
    command: Sequence[str],
    report: dict[str, Any],
) -> bool:
    print(f"\n{'=' * 100}\nFIGURE STEP: {name}", flush=True)
    completed = subprocess.run(list(command), cwd=PROJECT_ROOT, check=False)
    report["steps"][name] = {
        "status": "complete" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": list(command),
    }
    return completed.returncode == 0


def create_archive(output_dir: Path) -> Path:
    archive = output_dir / "qdpt_paper_figures.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path != archive:
                handle.add(path, arcname=path.relative_to(output_dir))
    return archive


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    dynamic_root = PROJECT_ROOT / "pathvqa" / "outputs" / "dynamic_prompt"
    checkpoint_root = dynamic_root / (
        "pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_"
        "sandwich_seed44_20260909"
    )
    seed_runs = {
        "seed44": checkpoint_root,
        "seed45": dynamic_root
        / "pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed45_20260910_1",
        "seed46": dynamic_root
        / "pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed46_20260910",
    }
    report: dict[str, Any] = {
        "output_dir": str(output_dir),
        "steps": {},
        "attention_selection": None,
    }

    dynamics_command = [
        python,
        "-m",
        "paper_figures.qdpt_figures",
        "dynamics",
    ]
    for label, path in seed_runs.items():
        dynamics_command.extend(("--run", f"{label}={path}"))
    for score in (60.7765, 57.2935, 59.3865):
        dynamics_command.extend(("--score", str(score)))
    dynamics_command.extend(
        ("--output", str(output_dir / "figure2_training_dynamics"))
    )
    run_step("figure2_training_dynamics", dynamics_command, report)

    run_step(
        "figure3_seed_stability",
        [
            python,
            "-m",
            "paper_figures.qdpt_figures",
            "stability",
            "--output",
            str(output_dir / "figure3_seed_stability"),
        ],
        report,
    )
    run_step(
        "figure4_module_activity",
        [
            python,
            "-m",
            "paper_figures.qdpt_figures",
            "activity",
            "--run",
            f"seed44={checkpoint_root}",
            "--output",
            str(output_dir / "figure4_module_activity"),
        ],
        report,
    )

    candidates_path = output_dir / "attention_candidates.json"
    candidates_ok = run_step(
        "attention_candidate_selection",
        [
            python,
            "-m",
            "paper_figures.select_attention_cases",
            "--data-root",
            args.data_root,
            "--cache-dir",
            args.cache_dir,
            "--split",
            "validation",
            "--top",
            "20",
            "--output",
            str(candidates_path),
        ],
        report,
    )
    if candidates_ok:
        candidate_data = json.loads(candidates_path.read_text(encoding="utf-8"))
        row_a, row_b = select_candidate_rows(candidate_data)
        report["attention_selection"] = {
            "candidate_rank": 1,
            "row_indices": [row_a, row_b],
            "rule": candidate_data["selection_rule"],
            "rationale": "top-ranked same-image pair with distinct question types",
        }
        selection_path = output_dir / "attention_selection.json"
        selection_path.write_text(
            json.dumps(report["attention_selection"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        cuda_ok = run_step(
            "cuda_preflight",
            [
                python,
                "-c",
                (
                    "import torch; assert torch.cuda.is_available(); "
                    "x=torch.zeros(1,device='cuda'); "
                    "print('[GPU OK]',torch.cuda.get_device_name(0),x.device)"
                ),
            ],
            report,
        )
        attention_dir = output_dir / "attention_case_01"
        if cuda_ok:
            export_ok = run_step(
                "figure1_attention_export",
                [
                    python,
                    "-m",
                    "paper_figures.export_qdpt_attention",
                    "--base-model",
                    args.model_path,
                    "--checkpoint",
                    str(checkpoint_root / "checkpoints" / "epoch_3"),
                    "--data-root",
                    args.data_root,
                    "--cache-dir",
                    args.cache_dir,
                    "--split",
                    "validation",
                    "--row-index",
                    str(row_a),
                    "--row-index",
                    str(row_b),
                    "--output-dir",
                    str(attention_dir),
                ],
                report,
            )
            if export_ok:
                run_step(
                    "figure1_question_guided_attention",
                    [
                        python,
                        "-m",
                        "paper_figures.qdpt_figures",
                        "attention",
                        "--bundle-dir",
                        str(attention_dir),
                        "--output",
                        str(output_dir / "figure1_question_guided_attention"),
                    ],
                    report,
                )

    failed = [
        name
        for name, status in report["steps"].items()
        if status["status"] != "complete"
    ]
    report["failed_steps"] = failed
    report_path = output_dir / "bundle_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive = create_archive(output_dir)
    print(f"\n[BUNDLE] directory={output_dir}")
    print(f"[BUNDLE] archive={archive}")
    print(f"[BUNDLE] failed_steps={failed}")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(path)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
