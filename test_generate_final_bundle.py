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

    def test_selects_requested_candidate_rank(self):
        data = {
            "candidates": [
                {"records": [
                    {"row_index": 1, "question_type": "what"},
                    {"row_index": 2, "question_type": "where"},
                ]},
                {"records": [
                    {"row_index": 20, "question_type": "yes/no"},
                    {"row_index": 21, "question_type": "how"},
                ]},
            ]
        }
        self.assertEqual(select_candidate_rows(data, 2), (20, 21))

    def test_rejects_missing_candidates(self):
        with self.assertRaises(ValueError):
            select_candidate_rows({"candidates": []})

    def test_rejects_out_of_range_candidate_rank(self):
        with self.assertRaises(ValueError):
            select_candidate_rows({"candidates": []}, 2)


if __name__ == "__main__":
    unittest.main()
