"""Offline tests for the isolated SLAKE evaluation pipeline."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from PIL import Image

from slake_model_interfaces import BACKEND_SPECS
from slake_official_eval import (
    ImageResolver,
    audit_samples,
    select_records,
    summarize_generation_timings,
)
from slake_vqa_metric import evaluate_slake_predictions, normalize_vqa_answer
from summarize_checkpoint_timing import load_rows, write_outputs


class SlakeMetricTest(unittest.TestCase):
    def test_last8_vision_lora_backend_is_explicit(self):
        self.assertEqual(
            BACKEND_SPECS["lora-vision-last8"],
            (
                "loraTest/loraLast8VisionExperiments.py",
                "LoraModelInterface",
            ),
        )

    def test_vqa_normalization(self):
        self.assertEqual(normalize_vqa_answer("The LEFT lung."), "left lung")
        self.assertEqual(normalize_vqa_answer("Two"), "2")
        self.assertEqual(normalize_vqa_answer("No, it isn't!"), "no it isn't")

    def test_exact_match_and_group_metrics(self):
        references = [
            {
                "question_id": 1,
                "question": "Which lung?",
                "answer": "left lung",
                "answer_type": "OPEN",
                "question_type": "organ",
                "q_lang": "en",
            },
            {
                "question_id": 2,
                "question": "Is this normal?",
                "answer": "yes",
                "answer_type": "CLOSED",
                "question_type": "abnormality",
                "q_lang": "en",
            },
        ]
        predictions = [
            {"question_id": 1, "answer": "The left lung."},
            {"question_id": 2, "answer": "no"},
        ]
        summary, comparisons = evaluate_slake_predictions(references, predictions)
        self.assertEqual(summary["overall_accuracy"], 50.0)
        self.assertEqual(summary["per_answer_type_accuracy"]["OPEN"], 100.0)
        self.assertEqual(summary["per_answer_type_accuracy"]["CLOSED"], 0.0)
        self.assertEqual([row["correct"] for row in comparisons], [1, 0])


class SlakeDatasetAuditTest(unittest.TestCase):
    def test_split_filter_and_nested_image_resolution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "xmlab0"
            image_dir.mkdir()
            Image.new("RGB", (8, 8), "white").save(image_dir / "source.jpg")

            records = [
                {
                    "qid": 10,
                    "img_name": "xmlab0/source.jpg",
                    "question": "What is shown?",
                    "answer": "lung",
                    "q_lang": "en",
                    "split": "test",
                    "answer_type": "OPEN",
                },
                {
                    "qid": 11,
                    "img_name": "xmlab0/source.jpg",
                    "question": "What is shown?",
                    "answer": "lung",
                    "q_lang": "en",
                    "split": "train",
                },
                {
                    "qid": 12,
                    "img_name": "xmlab0/source.jpg",
                    "question": "图中是什么？",
                    "answer": "肺",
                    "q_lang": "zh",
                    "split": "test",
                },
            ]
            selected = select_records(
                records,
                language="en",
                base_types=[],
                expected_split="test",
            )
            self.assertEqual([record["question_id"] for record in selected], [10])

            audited = audit_samples(
                selected,
                ImageResolver(root),
                allow_missing_images=False,
            )
            self.assertEqual(len(audited), 1)
            self.assertTrue(Path(audited[0]["_slake_image_path"]).is_file())


class SlakeTimingSummaryTest(unittest.TestCase):
    def test_tpot_is_weighted_by_measured_decode_tokens(self):
        rows = [
            {
                "status": "ok",
                "timing": {
                    "request_total_seconds": 1.5,
                    "first_token_latency_seconds": 0.8,
                    "subsequent_token_mean_seconds": 0.1,
                    "generation_total_seconds": 1.2,
                    "generated_token_count": 3,
                    "measured_token_steps": 3,
                    "timing_method": "test",
                },
            },
            {
                "status": "ok",
                "timing": {
                    "request_total_seconds": 2.5,
                    "first_token_latency_seconds": 1.0,
                    "subsequent_token_mean_seconds": 0.2,
                    "generation_total_seconds": 2.0,
                    "generated_token_count": 2,
                    "measured_token_steps": 2,
                    "timing_method": "test",
                },
            },
        ]

        summary = summarize_generation_timings(rows, warmup_runs=3)

        self.assertEqual(summary["generated_tokens"], 5)
        self.assertEqual(summary["subsequent_tokens"], 3)
        self.assertAlmostEqual(summary["tpot_weighted_seconds"], 0.133333, places=6)
        self.assertAlmostEqual(summary["decode_tokens_per_second"], 7.5, places=3)
        self.assertEqual(summary["methodology"]["warmup_runs_excluded"], 3)

    def test_checkpoint_comparison_outputs_are_generated(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            method_dir = root / "frost_vl"
            method_dir.mkdir()
            summary = {
                "overall_accuracy": 66.95,
                "timing": {
                    "ttft_seconds": {"mean": 0.8, "p50": 0.7, "p95": 1.1},
                    "tpot_weighted_seconds": 0.02,
                    "decode_tokens_per_second": 50.0,
                    "request_seconds": {"mean": 1.2},
                    "model_generated_tokens_per_second": 20.0,
                    "end_to_end_generated_tokens_per_second": 18.0,
                    "model_timed_requests": 10,
                    "generated_tokens": 100,
                },
            }
            (method_dir / "slake_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

            rows = load_rows(root)
            write_outputs(root, rows)

            self.assertEqual(rows[0]["method"], "frost_vl")
            self.assertTrue((root / "comparison_summary.json").is_file())
            self.assertTrue((root / "comparison_summary.csv").is_file())
            self.assertTrue((root / "comparison_summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
