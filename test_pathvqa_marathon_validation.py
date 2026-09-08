import csv
import tempfile
import unittest
from pathlib import Path

from pathvqa.marathon_progress import (
    MARATHON_PROGRESS_COLUMNS,
    MarathonProgressLog,
    scheduled_validation_epochs,
)


class PathVQAMarathonValidationTest(unittest.TestCase):
    def test_validation_runs_from_epoch_three_through_ten(self):
        self.assertEqual(
            scheduled_validation_epochs(total_epochs=10, start_epoch=3),
            tuple(range(3, 11)),
        )

    def test_invalid_validation_window_is_rejected(self):
        with self.assertRaises(ValueError):
            scheduled_validation_epochs(total_epochs=10, start_epoch=11)

    def test_progress_log_is_compact_and_rejects_duplicate_epochs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "marathon_progress.tsv"
            progress = MarathonProgressLog(path)
            progress.append(
                {
                    "timestamp": "2026-09-08T12:00:00+0800",
                    "epoch": 3,
                    "global_step": 1845,
                    "overall": "59.5622",
                    "yes_no": "91.0400",
                    "free_form": "28.1748",
                    "latest_train_loss": "11.2",
                    "elapsed_hours": "2.0",
                    "evaluation_minutes": "20.0",
                    "checkpoint": "/tmp/epoch_3",
                    "status": "complete",
                }
            )
            with self.assertRaises(RuntimeError):
                progress.append({"epoch": 3, "status": "complete"})

            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(tuple(rows[0]), MARATHON_PROGRESS_COLUMNS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["epoch"], "3")
            self.assertEqual(rows[0]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
