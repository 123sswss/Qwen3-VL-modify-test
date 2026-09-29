"""CPU contracts for the direct-summary V1 ablation."""

from __future__ import annotations

import unittest

import torch

from slake.visual_selection_prefix import VisualSelectionPrefixModel
from slake.visual_selection_prefix_direct import (
    EXPECTED_TRAINABLE_DIRECT,
    VisualSelectionPrefixDirectModel,
)
from test_visual_selection_offset import _Base


def _batch():
    ids = torch.tensor([[11, 12, 13, 14, 15]])
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "pixel_values": torch.randn(8, 1024),
        "image_grid_thw": torch.tensor([[1, 2, 4]]),
        "labels": torch.tensor([[-100, -100, -100, -100, 15]]),
        "question_source_ids": torch.tensor([[12, 13]]),
        "question_source_mask": torch.ones(1, 2, dtype=torch.bool),
    }


class VisualSelectionPrefixDirectTest(unittest.TestCase):
    def test_retained_initialization_budget_calibration_and_backward(self):
        torch.manual_seed(44)
        reference = VisualSelectionPrefixModel(_Base(), init_seed=44)
        expected = {
            name: parameter.detach().clone()
            for name, parameter in reference.named_parameters()
            if parameter.requires_grad
            and not name.startswith(("value_blocks.", "prefix_output."))
        }
        torch.manual_seed(44)
        model = VisualSelectionPrefixDirectModel(_Base(), init_seed=44)
        actual = {
            name: parameter.detach()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name != "alpha"
        }
        self.assertEqual(set(expected), set(actual))
        for name in expected:
            self.assertTrue(torch.equal(expected[name], actual[name]), name)
        self.assertEqual(sum(model._audit_parameters().values()), EXPECTED_TRAINABLE_DIRECT)
        self.assertFalse(hasattr(model, "value_blocks"))
        self.assertFalse(hasattr(model, "prefix_output"))

        batch = _batch()
        cpu_rng = torch.random.get_rng_state().clone()
        audit = model.calibrate_alpha_from_first_batch(batch)
        self.assertTrue(torch.equal(cpu_rng, torch.random.get_rng_state()))
        self.assertTrue(model.alpha_calibrated)
        self.assertAlmostEqual(audit["calibrated_to_old_ratio"], 1.0, delta=1e-5)
        self.assertFalse(hasattr(model, "_calibration_output_weight"))

        output = model(**batch)
        output.loss.backward()
        groups = model.trainable_parameter_groups()
        self.assertEqual(set(groups), {
            "p20", "visual_s8", "visual_av10", "question_context",
            "maps", "layer_condition", "alpha",
        })
        for name, parameters in groups.items():
            self.assertTrue(all(parameter.grad is not None for parameter in parameters), name)
            self.assertGreater(
                sum(float(parameter.grad.float().norm()) for parameter in parameters), 0.0, name,
            )


if __name__ == "__main__":
    unittest.main()
