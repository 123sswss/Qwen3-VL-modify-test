from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from RSVQA.data import load_rsvqa_lr_split
from RSVQA.metric import evaluate_rsvqa_predictions, normalize_rsvqa_answer


class RSVQAInterfaceTest(unittest.TestCase):
    def _write(self, path: Path, payload) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_active_join_and_official_count_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "Images_LR"
            images.mkdir()
            (images / "0.tif").write_bytes(b"not-opened-by-loader")
            self._write(root / "USGS_split_test_images.json", {"images": [
                {"id": 0, "questions_ids": [0, 1], "active": True},
                {"id": 1, "questions_ids": [2], "active": False},
            ]})
            self._write(root / "USGS_split_test_questions.json", {"questions": [
                {"id": 0, "img_id": 0, "type": "count", "question": "How many?", "answers_ids": [0], "active": True},
                {"id": 1, "img_id": 0, "type": "presence", "question": "Is there water?", "answers_ids": [1], "active": True},
                {"id": 2, "img_id": 1, "type": "count", "question": "Inactive?", "answers_ids": [2], "active": False},
            ]})
            self._write(root / "USGS_split_test_answers.json", {"answers": [
                {"id": 0, "question_id": 0, "answer": "57", "active": True},
                {"id": 1, "question_id": 1, "answer": "yes", "active": True},
                {"id": 2, "question_id": 2, "answer": "0", "active": False},
            ]})
            records, manifest = load_rsvqa_lr_split(
                root, "test", enforce_official_counts=False
            )
            self.assertEqual(len(records), 2)
            self.assertEqual(manifest["active_images"], 1)
            summary, rows = evaluate_rsvqa_predictions(
                records,
                [{"question_id": 0, "answer": "42"}, {"question_id": 1, "answer": "Yes."}],
                bootstrap_iterations=10,
            )
            self.assertEqual(summary["overall_accuracy"], 100.0)
            self.assertTrue(all(row["correct"] for row in rows))

    def test_count_boundaries(self) -> None:
        expected = {
            "0": "0", "1": "between 0 and 10", "10": "between 0 and 10",
            "11": "between 10 and 100", "100": "between 10 and 100",
            "101": "between 100 and 1000", "1000": "between 100 and 1000",
            "1001": "more than 1000",
        }
        for value, target in expected.items():
            self.assertEqual(normalize_rsvqa_answer(value, "count"), target)


if __name__ == "__main__":
    unittest.main()
