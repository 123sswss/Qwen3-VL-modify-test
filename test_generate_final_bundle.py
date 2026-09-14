import unittest

from paper_figures.generate_final_bundle import select_candidate_rows


class GenerateFinalBundleTest(unittest.TestCase):
    def test_selects_first_distinct_question_type_pair(self):
        data = {
            "candidates": [
                {
                    "records": [
                        {"row_index": 10, "question_type": "what"},
                        {"row_index": 11, "question_type": "what"},
                        {"row_index": 12, "question_type": "where"},
                    ]
                }
            ]
        }
        self.assertEqual(select_candidate_rows(data), (10, 12))

    def test_rejects_missing_candidates(self):
        with self.assertRaises(ValueError):
            select_candidate_rows({"candidates": []})


if __name__ == "__main__":
    unittest.main()
