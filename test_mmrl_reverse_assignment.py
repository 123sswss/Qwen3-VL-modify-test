import unittest

import torch

from MMRL import ReverseAssignmentAttention


class ReverseAssignmentAttentionTest(unittest.TestCase):
    def test_assignment_axes_are_normalized(self):
        scores = torch.randn(2, 3, 4, 7, 5)

        assignments, slot_weights = (
            ReverseAssignmentAttention._normalize_assignments(scores)
        )

        torch.testing.assert_close(
            assignments.sum(dim=-1),
            torch.ones_like(assignments.sum(dim=-1)),
        )
        torch.testing.assert_close(
            slot_weights.sum(dim=-2),
            torch.ones_like(slot_weights.sum(dim=-2)),
        )

    def test_zero_output_projection_preserves_slots(self):
        module = ReverseAssignmentAttention(hidden_dim=16, num_heads=4)
        module.train()
        slots = torch.randn(2, 3, 5, 16)
        memory = torch.randn(2, 7, 16)

        output, delta = module(slots, memory)

        torch.testing.assert_close(delta, torch.zeros_like(delta))
        torch.testing.assert_close(output, slots)
        self.assertAlmostEqual(
            float(module.debug_context["reverse_assignment_slot_mass_mean"]),
            7.0 / 5.0,
            places=5,
        )
        self.assertEqual(tuple(output.shape), (2, 3, 5, 16))


if __name__ == "__main__":
    unittest.main()
