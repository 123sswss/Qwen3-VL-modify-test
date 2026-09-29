"""CPU contracts for V1B's 20-static-plus-one-evidence prefix layout."""

from __future__ import annotations

import unittest

import torch

from slake.visual_selection_prefix import EXPECTED_TRAINABLE, VisualSelectionPrefixModel
from slake.visual_selection_prefix_evidence import (
    TOTAL_PREFIX_TOKENS,
    VisualSelectionPrefixEvidenceModel,
)
from test_visual_selection_offset import _Base


class VisualSelectionPrefixEvidenceTest(unittest.TestCase):
    def test_initialization_budget_and_expansion(self):
        torch.manual_seed(44)
        reference = VisualSelectionPrefixModel(_Base(), init_seed=44)
        expected = {
            name: parameter.detach().clone()
            for name, parameter in reference.named_parameters() if parameter.requires_grad
        }
        torch.manual_seed(44)
        model = VisualSelectionPrefixEvidenceModel(_Base(), init_seed=44)
        actual = {
            name: parameter.detach()
            for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(set(expected), set(actual))
        for name in expected:
            self.assertTrue(torch.equal(expected[name], actual[name]), name)
        self.assertEqual(sum(model._audit_parameters().values()), EXPECTED_TRAINABLE)

        ids = torch.tensor([[11, 12, 13]])
        labels = torch.tensor([[-100, -100, 13]])
        expanded, source_ids, source_mask = model._expand({
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": labels,
            "question_source_ids": torch.tensor([[12]]),
            "question_source_mask": torch.ones(1, 1, dtype=torch.bool),
        })
        self.assertEqual(expanded["input_ids"].shape[1], ids.shape[1] + TOTAL_PREFIX_TOKENS)
        self.assertTrue(expanded["attention_mask"][:, :TOTAL_PREFIX_TOKENS].eq(1).all())
        self.assertTrue(expanded["labels"][:, :TOTAL_PREFIX_TOKENS].eq(-100).all())
        self.assertTrue(torch.equal(expanded["input_ids"][:, TOTAL_PREFIX_TOKENS:], ids))
        self.assertTrue(torch.equal(source_ids, torch.tensor([[12]])))
        self.assertTrue(source_mask.all())


if __name__ == "__main__":
    unittest.main()
