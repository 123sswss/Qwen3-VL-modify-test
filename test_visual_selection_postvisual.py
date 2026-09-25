"""Small CPU-only tests of V3's per-sample image-boundary insertion."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from pathvqa.train_visual_selection_postvisual import _check_sequence_reorder
from processingWithMMRL import reserve_v3_postvisual_slots
from slake.visual_selection_postvisual import PROMPT_LENGTH, VisualSelectionPostvisualModel


class VisualSelectionPostvisualTest(unittest.TestCase):
    def setUp(self):
        self.model = VisualSelectionPostvisualModel.__new__(VisualSelectionPostvisualModel)
        nn.Module.__init__(self.model)
        self.model.config = SimpleNamespace(
            vision_start_token_id=91, vision_end_token_id=92,
            image_token_id=93, pad_token_id=0,
        )

    def _batch(self):
        return {
            "input_ids": torch.tensor([
                [1, 91, 93, 93, 92, 10, 11, 12, 13, 0],
                [1, 91, 93, 92, 10, 14, 15, 16, 0, 0],
            ]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            ]),
            "labels": torch.tensor([
                [-100, -100, -100, -100, -100, -100, -100, -100, 13, -100],
                [-100, -100, -100, -100, -100, -100, -100, 16, -100, -100],
            ]),
            "image_grid_thw": torch.tensor([[1, 2, 4], [1, 2, 2]]),
            "pixel_values": torch.ones(12, 1024),
            "question_source_ids": torch.tensor([[11, 12], [14, 15]]),
            "question_source_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.bool),
        }

    def test_insertion_after_complete_image_and_before_question_with_padding(self):
        batch = self._batch()
        expanded, _ = reserve_v3_postvisual_slots(
            {key: value for key, value in batch.items() if key not in
             ("labels", "question_source_ids", "question_source_mask")},
            vision_start_id=91, vision_end_id=92, image_token_id=93,
            pad_token_id=0, merge_size=2,
        )
        expanded["labels"] = torch.stack([
            torch.cat((batch["labels"][row, :pos],
                       torch.full((PROMPT_LENGTH,), -100),
                       batch["labels"][row, pos:]))
            for row, pos in enumerate((5, 4))
        ])
        expanded["question_source_ids"] = batch["question_source_ids"]
        expanded["question_source_mask"] = batch["question_source_mask"]
        _, source_ids, source_mask, positions, prompt_mask = self.model._prepare(expanded)
        self.assertEqual(positions[:, 0].tolist(), [5, 4])
        self.assertEqual(expanded["input_ids"].shape, (2, 10 + PROMPT_LENGTH))
        self.assertTrue(_check_sequence_reorder(batch, expanded))
        self.assertEqual(prompt_mask.sum(dim=1).tolist(), [PROMPT_LENGTH, PROMPT_LENGTH])
        self.assertTrue(torch.equal(source_ids, batch["question_source_ids"]))
        self.assertTrue(torch.equal(source_mask, batch["question_source_mask"]))
        self.assertEqual(expanded["input_ids"][0, 4].item(), 92)
        self.assertEqual(expanded["input_ids"][0, 25].item(), 10)

    def test_bad_visual_boundary_and_grid_fail_closed(self):
        missing_end = self._batch()
        missing_end["input_ids"][0, 4] = 10
        with self.assertRaises(ValueError):
            reserve_v3_postvisual_slots(missing_end, vision_start_id=91, vision_end_id=92,
                                        image_token_id=93, pad_token_id=0, merge_size=2)
        wrong_grid = self._batch()
        wrong_grid["image_grid_thw"][0, 2] = 2
        with self.assertRaises(ValueError):
            reserve_v3_postvisual_slots(wrong_grid, vision_start_id=91, vision_end_id=92,
                                        image_token_id=93, pad_token_id=0, merge_size=2)
        split_image = self._batch()
        split_image["input_ids"][0, 3] = 10
        split_image["input_ids"][0, 5] = 93
        with self.assertRaises(ValueError):
            reserve_v3_postvisual_slots(split_image, vision_start_id=91, vision_end_id=92,
                                        image_token_id=93, pad_token_id=0, merge_size=2)


if __name__ == "__main__":
    unittest.main()
