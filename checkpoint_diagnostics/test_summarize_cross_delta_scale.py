import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("summarize_cross_delta_scale.py")
SPEC = importlib.util.spec_from_file_location("summarize_cross_delta_scale", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def summary(overall, kvqa):
    return {
        "overall_accuracy": overall,
        "per_question_type_accuracy": {"kvqa": kvqa, "vqa": 70.0},
        "per_answer_type_accuracy": {"CLOSED": 80.0, "OPEN": 60.0},
        "per_language_accuracy": {"en": 75.0, "zh": 65.0},
    }


class CrossDeltaScaleSummaryTest(unittest.TestCase):
    def test_writes_grouped_scale_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for tag, payload in (
                ("scale_0", summary(69.0, 50.0)),
                ("scale_0p5", summary(72.0, 55.0)),
                ("scale_1", summary(73.0, 56.0)),
            ):
                directory = root / tag
                directory.mkdir()
                (directory / "slake_summary.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            with mock.patch.object(sys, "argv", ["summarize", str(root)]):
                self.assertEqual(MODULE.main(), 0)

            result = json.loads(
                (root / "cross_delta_scale_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["baseline_accuracy"], 73.0)
            self.assertEqual(result["results"][0]["delta"], -4.0)
            markdown = (root / "cross_delta_scale_summary.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("| 0.5 | 72.00 | -1.00 | 55.00 |", markdown)


if __name__ == "__main__":
    unittest.main()
