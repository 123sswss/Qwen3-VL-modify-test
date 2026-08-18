import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("compare_seeded_architectures.py")
SPEC = importlib.util.spec_from_file_location("compare_seeded_architectures", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(qid, correct, answer, question_type="vqa"):
    return {
        "question_id": qid,
        "question": f"question-{qid}",
        "ground_truth_answers": ["yes"],
        "normalized_predicted_answer": answer,
        "correct": int(correct),
        "answer_type": "CLOSED",
        "question_type": question_type,
        "language": "en",
    }


class SeededArchitectureComparisonTest(unittest.TestCase):
    def test_paired_counts_and_mcnemar(self):
        left = {
            "1": row(1, True, "yes"),
            "2": row(2, True, "yes"),
            "3": row(3, False, "no"),
            "4": row(4, False, "no"),
        }
        right = {
            "1": row(1, True, "yes"),
            "2": row(2, False, "no"),
            "3": row(3, True, "yes"),
            "4": row(4, True, "yes"),
        }
        qids = MODULE.validate_alignment(left, right)
        result = MODULE.paired_counts(left, right, qids)
        self.assertEqual(result["both_correct"], 1)
        self.assertEqual(result["left_only"], 1)
        self.assertEqual(result["right_only"], 2)
        self.assertEqual(result["both_wrong"], 0)
        self.assertEqual(result["right_minus_left_pp"], 25.0)
        self.assertEqual(result["mcnemar_exact_p"], 1.0)

    def test_cross_seed_stability(self):
        runs = {
            44: {"1": row(1, True, "yes"), "2": row(2, False, "a")},
            45: {"1": row(1, True, "yes"), "2": row(2, True, "yes")},
            46: {"1": row(1, True, "yes"), "2": row(2, False, "b")},
        }
        result = MODULE.architecture_stability(runs)
        self.assertEqual(result["unanimous_correct"], 1)
        self.assertEqual(result["correctness_unstable"], 1)
        self.assertEqual(result["unanimous_wrong"], 0)
        self.assertEqual(result["answer_unstable"], 1)

    def test_resolves_latest_experiment_and_eval_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "arch_seed44_old"
            new = root / "arch_seed44_new"
            for experiment in (old, new):
                eval_dir = experiment / "eval"
                eval_dir.mkdir(parents=True)
                with (eval_dir / "slake_comparisons.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump([row(1, True, "yes")], handle)
            old.touch()
            new.touch()
            latest = MODULE.resolve_latest_experiment(root, "arch", 44)
            self.assertEqual(latest, new)
            loaded = MODULE.load_rows(latest)
            self.assertEqual(set(loaded), {"1"})

    def test_selects_latest_completed_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            older_complete = root / "arch_seed44_20260817"
            newer_complete = root / "arch_seed44_20260817_1"
            newest_incomplete = root / "arch_seed44_20260817_2"
            for experiment in (older_complete, newer_complete, newest_incomplete):
                experiment.mkdir(parents=True)
            for timestamp, experiment in (
                (100.0, older_complete),
                (200.0, newer_complete),
            ):
                eval_dir = experiment / "eval"
                eval_dir.mkdir()
                comparisons = eval_dir / "slake_comparisons.json"
                with comparisons.open("w", encoding="utf-8") as handle:
                    json.dump([row(1, True, "yes")], handle)
                os.utime(comparisons, (timestamp, timestamp))
            os.utime(newest_incomplete, (300.0, 300.0))

            experiment, comparisons = MODULE.resolve_latest_evaluated_experiment(
                root, "arch", 44
            )
            self.assertEqual(experiment, newer_complete)
            self.assertEqual(comparisons, newer_complete / "eval" / "slake_comparisons.json")


if __name__ == "__main__":
    unittest.main()
