import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("summarize_query_generator_swap.py")
SPEC = importlib.util.spec_from_file_location(
    "summarize_query_generator_swap",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def summary(overall, kvqa):
    return {
        "overall_accuracy": overall,
        "per_question_type_accuracy": {"kvqa": kvqa, "vqa": 75.0},
        "per_answer_type_accuracy": {"CLOSED": 80.0, "OPEN": 65.0},
        "per_language_accuracy": {"en": 76.0, "zh": 68.0},
    }


class QueryGeneratorSwapSummaryTest(unittest.TestCase):
    def test_recipient_relative_deltas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = {
                "baseline_44": summary(73.4, 56.55),
                "baseline_45": summary(72.11, 55.43),
                "query44_rest45": summary(73.0, 56.0),
                "query45_rest44": summary(72.5, 55.0),
            }
            for tag, payload in payloads.items():
                directory = root / tag
                directory.mkdir()
                (directory / "slake_summary.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )

            with mock.patch.object(sys, "argv", ["summarize", str(root)]):
                self.assertEqual(MODULE.main(), 0)

            result = json.loads(
                (root / "query_generator_swap_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            rows = {row["tag"]: row for row in result["results"]}
            self.assertAlmostEqual(
                rows["query44_rest45"]["recipient_delta"],
                0.89,
            )
            self.assertAlmostEqual(
                rows["query45_rest44"]["recipient_delta"],
                -0.9,
            )
            report = (root / "query_generator_swap_summary.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("| Q44 + Rest45 | 73.00 | +0.89 |", report)


if __name__ == "__main__":
    unittest.main()
