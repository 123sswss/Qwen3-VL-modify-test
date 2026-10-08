"""Small CPU-only checks; never start a model, timer, or shutdown command."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diagnostics.run_pathvqa_v10_seeds import aggregate, shutdown_worker, SEEDS


class QueueTest(unittest.TestCase):
    def test_mean_and_sample_std(self):
        rows = [{"overall_accuracy":x,"yes_no_accuracy":x,"free_form_accuracy":x,
                 "per_question_type_accuracy":{"what":x,"where":x}} for x in (57.,58.,59.)]
        self.assertEqual(SEEDS,(44,45,46))
        for item in aggregate(rows).values():
            self.assertEqual(item["mean"],58.)
            self.assertEqual(item["sample_std"],1.)
            self.assertEqual(item["ddof"],1)

    def test_wait_before_shutdown_and_record_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output/"shutdown_status.json").write_text(json.dumps({"scheduled_unix":1600}),encoding="utf-8")
            calls = []
            with patch("diagnostics.run_pathvqa_v10_seeds.time.time",return_value=1000), \
                 patch("diagnostics.run_pathvqa_v10_seeds.time.sleep",side_effect=lambda seconds:calls.append(seconds)), \
                 patch("diagnostics.run_pathvqa_v10_seeds.subprocess.run") as run:
                run.side_effect = lambda *a,**k: (calls.append(a[0]),type("Exit",(),{
                    "returncode":7,"stdout":"","stderr":"denied"})())[1]
                shutdown_worker(output)
            self.assertEqual(calls,[600.,["/usr/bin/shutdown"]])
            self.assertEqual(json.loads((output/"shutdown_status.json").read_text(encoding="utf-8"))["exit_code"],7)


if __name__ == "__main__":
    unittest.main()
