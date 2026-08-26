import unittest

from pathvqa.pathvqa_vqa_metric import (
    answer_type,
    evaluate_pathvqa_predictions,
    normalize_pathvqa_answer,
    question_type,
)


class PathVQAMetricTest(unittest.TestCase):
    def test_normalization_and_types(self):
        self.assertEqual(normalize_pathvqa_answer("  YES. "), "yes")
        self.assertEqual(answer_type("No"), "yes/no")
        self.assertEqual(answer_type("liver"), "free-form")
        self.assertEqual(question_type("What is shown?", "liver"), "what")
        self.assertEqual(question_type("Is necrosis present?", "yes"), "yes/no")

    def test_evaluation_and_clustered_bootstrap(self):
        records = [
            {
                "question_id": "q1",
                "image_id": "image-a",
                "question": "Is this malignant?",
                "answer": "yes",
            },
            {
                "question_id": "q2",
                "image_id": "image-a",
                "question": "What is visible?",
                "answer": "tumor",
            },
            {
                "question_id": "q3",
                "image_id": "image-b",
                "question": "Where is the lesion?",
                "answer": "liver",
            },
        ]
        predictions = [
            {"question_id": "q1", "image_id": "image-a", "answer": "Yes."},
            {"question_id": "q2", "image_id": "image-a", "answer": "tumor"},
            {"question_id": "q3", "image_id": "image-b", "answer": "kidney"},
        ]
        summary, comparisons = evaluate_pathvqa_predictions(
            records,
            predictions,
            bootstrap_iterations=100,
            bootstrap_seed=7,
        )
        self.assertAlmostEqual(summary["overall_accuracy"], 66.6667)
        self.assertEqual(summary["yes_no_accuracy"], 100.0)
        self.assertEqual(summary["free_form_accuracy"], 50.0)
        self.assertEqual(summary["clustered_bootstrap"]["image_clusters"], 2)
        self.assertEqual(sum(row["correct"] for row in comparisons), 2)


if __name__ == "__main__":
    unittest.main()
