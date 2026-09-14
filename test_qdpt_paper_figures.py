import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper_figures.qdpt_figures import (
    attention_maps,
    finite_series,
    load_losses,
    load_jsonl,
    moving_average,
    parse_run_argument,
    progress_percent,
)
from paper_figures.select_attention_cases import question_type


class QDPTPaperFiguresTest(unittest.TestCase):
    def test_attention_maps_preserve_grid_and_rank_concentrated_query(self):
        attention = np.full((2, 3, 8), 1.0 / 8.0)
        attention[:, 1, :] = 0.0
        attention[:, 1, 3] = 1.0
        aggregate, per_query, top_queries = attention_maps(attention, (1, 2, 4))
        self.assertEqual(aggregate.shape, (2, 4))
        self.assertEqual(per_query.shape, (3, 2, 4))
        self.assertEqual(int(top_queries[0]), 1)

    def test_metric_parser_ignores_non_finite_values(self):
        rows = [
            {"step": 1, "metric": 2.0},
            {"step": 2, "metric": float("nan")},
            {"step": 3, "metric": 4.0},
        ]
        steps, values, key = finite_series(rows, ("missing", "metric"))
        self.assertEqual(key, "metric")
        np.testing.assert_array_equal(steps, (1.0, 3.0))
        np.testing.assert_array_equal(values, (2.0, 4.0))

    def test_jsonl_and_training_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                json.dumps({"step": 1, "value": 2}) + "\n"
                + json.dumps({"step": 2, "value": 3}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(len(load_jsonl(path)), 2)
        np.testing.assert_allclose(progress_percent(np.asarray([1, 2, 4])), (25, 50, 100))
        self.assertEqual(moving_average(np.asarray([1.0, 2.0, 3.0]), 1).tolist(), [1, 2, 3])

    def test_run_argument_and_question_type(self):
        label, path = parse_run_argument("seed44=/tmp/run")
        self.assertEqual(label, "seed44")
        self.assertEqual(path, Path("/tmp/run"))
        self.assertEqual(question_type("Where is the lesion?"), "where")
        self.assertEqual(question_type("Is this malignant?"), "yes/no")

    def test_load_losses_falls_back_to_trainer_stdout_log(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "train.log").write_text(
                "loading\n"
                "{'loss': 12.5, 'learning_rate': 0.1, 'epoch': 0.5}\n"
                "progress {'loss': 10.25, 'epoch': 1.0} trailing\n",
                encoding="utf-8",
            )
            steps, losses = load_losses(run_dir)
        np.testing.assert_allclose(steps, (0.5, 1.0))
        np.testing.assert_allclose(losses, (12.5, 10.25))


if __name__ == "__main__":
    unittest.main()
