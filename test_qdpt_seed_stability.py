import unittest

from diagnostics.analyze_qdpt_seed_stability import analyze_runs


def _row(question_id, prediction, correct, answer_type="free-form"):
    return {
        "question_id": question_id,
        "image_id": f"image-{question_id}",
        "question": f"question {question_id}",
        "reference": "target",
        "prediction": prediction,
        "answer_type": answer_type,
        "question_type": "what" if answer_type == "free-form" else "yes/no",
        "correct": correct,
    }


class QDPTSeedStabilityTest(unittest.TestCase):
    def test_three_seed_categories_and_pairwise_counts(self):
        runs = {
            "seed44": [
                _row("all", "target", True),
                _row("none", "a", False),
                _row("one", "target", True),
                _row("two", "target", True, "yes/no"),
            ],
            "seed45": [
                _row("all", "target", True),
                _row("none", "b", False),
                _row("one", "x", False),
                _row("two", "target", True, "yes/no"),
            ],
            "seed46": [
                _row("all", "target", True),
                _row("none", "c", False),
                _row("one", "y", False),
                _row("two", "no", False, "yes/no"),
            ],
        }
        summary, samples, pairwise = analyze_runs(runs)

        overall = summary["overall"]
        self.assertEqual(overall["all_correct"], 1)
        self.assertEqual(overall["all_wrong"], 1)
        self.assertEqual(overall["correct_count_histogram"]["1"], 1)
        self.assertEqual(overall["correct_count_histogram"]["2"], 1)
        categories = {row["question_id"]: row["category"] for row in samples}
        self.assertEqual(categories["one"], "exactly_1_correct")
        self.assertEqual(categories["two"], "exactly_2_correct")

        seed44_seed45 = next(
            row
            for row in pairwise
            if row["left"] == "seed44" and row["right"] == "seed45"
        )
        self.assertEqual(seed44_seed45["left_only_correct"], 1)
        self.assertEqual(seed44_seed45["right_only_correct"], 0)
        self.assertEqual(seed44_seed45["correctness_agreement"], 3)

    def test_metadata_mismatch_is_rejected(self):
        left = [_row("q0", "target", True)]
        right = [_row("q0", "target", True)]
        right[0]["reference"] = "different"
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            analyze_runs({"seed44": left, "seed45": right})


if __name__ == "__main__":
    unittest.main()
