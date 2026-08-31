import sys
import unittest
from unittest.mock import patch

from PIL import Image

from pathvqa.pathvqa_official_eval import (
    audit_directional_intervention,
    parse_args,
    prime_directional_intervention,
)


class _FakeStore:
    def __init__(self):
        self.samples = [
            {"question_id": "q0", "question": "first?", "image_value": 0},
            {"question_id": "q1", "question": "second?", "image_value": 1},
        ]

    def load_image(self, record):
        value = int(record["image_value"])
        return Image.new("RGB", (2, 2), color=(value, value, value))


class _FakeModel:
    def __init__(self):
        self.calls = []
        self.measurements_reset = 0

    def infer(self, image, prompt, **kwargs):
        self.calls.append((image.getpixel((0, 0)), prompt, kwargs))
        return "ignored"

    def reset_inference_measurements(self):
        self.measurements_reset += 1


class PathVQADirectionalInterventionTest(unittest.TestCase):
    def test_cli_defaults_preserve_normal_behavior(self):
        argv = [
            "pathvqa_official_eval.py",
            "--data-root",
            "data",
            "--base-model",
            "model",
            "--output-dir",
            "output",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.directional_question_query_mode, "normal")
        self.assertEqual(args.directional_visual_memory_mode, "normal")

    def test_cli_accepts_both_directional_mismatches(self):
        argv = [
            "pathvqa_official_eval.py",
            "--data-root",
            "data",
            "--base-model",
            "model",
            "--output-dir",
            "output",
            "--directional-question-query-mode",
            "previous-distinct-question",
            "--directional-visual-memory-mode",
            "previous-distinct-image",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(
            args.directional_question_query_mode,
            "previous-distinct-question",
        )
        self.assertEqual(
            args.directional_visual_memory_mode,
            "previous-distinct-image",
        )

    def test_priming_is_distinct_and_excluded(self):
        store = _FakeStore()
        model = _FakeModel()
        metadata = prime_directional_intervention(
            store,
            store.samples,
            model,
            max_new_tokens=4,
            temperature=0.0,
            instruction=None,
            question_query_mode="previous-distinct-question",
            visual_memory_mode="normal",
        )
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.measurements_reset, 1)
        self.assertTrue(metadata["excluded_from_scoring"])
        self.assertTrue(metadata["question_distinct_from_first_scored"])
        self.assertTrue(metadata["image_distinct_from_first_scored"])

    def test_strict_audit_requires_all_scored_samples_mismatched(self):
        intervention = {
            "workspace": {
                "question_query_units_seen": 6,
                "question_query_units_mismatched": 5,
                "question_query_units_natural": 1,
                "visual_memory_images_seen": 6,
                "visual_memory_images_mismatched": 5,
                "visual_memory_images_natural": 1,
                "question_query_mode": "previous-distinct-question",
                "visual_memory_mode": "previous-distinct-image",
                "original_image_input_unchanged": True,
                "original_question_input_unchanged": True,
                "anchors_preserved": True,
                "token_count_preserved": True,
            }
        }
        priming = {
            "excluded_from_scoring": True,
            "question_distinct_from_first_scored": True,
            "image_distinct_from_first_scored": True,
        }
        audit = audit_directional_intervention(
            intervention,
            scored_count=5,
            priming=priming,
            question_query_mode="previous-distinct-question",
            visual_memory_mode="previous-distinct-image",
        )
        self.assertTrue(audit["question_query_mismatch_audit_pass"])
        self.assertTrue(audit["visual_memory_mismatch_audit_pass"])
        self.assertEqual(audit["excluded_priming_samples"], 1)

        intervention["workspace"]["question_query_units_mismatched"] = 4
        intervention["workspace"]["visual_memory_mode"] = "normal"
        with self.assertRaisesRegex(RuntimeError, "question-Q mismatch"):
            audit_directional_intervention(
                intervention,
                scored_count=5,
                priming=priming,
                question_query_mode="previous-distinct-question",
                visual_memory_mode="normal",
            )


if __name__ == "__main__":
    unittest.main()
