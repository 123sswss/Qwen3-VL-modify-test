import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from MMRL import MMRL, MaskedMeanPooling
from utils import attention_pooling


class AttentionPoolingTest(unittest.TestCase):
    def test_segmented_pooling_matches_dense_pooling(self):
        torch.manual_seed(7)
        pooling = attention_pooling(input_dim=6, proj_dim=4).double()
        tokens = torch.randn(8, 6, dtype=torch.double)
        batch_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
        valid_mask = torch.tensor(
            [True, False, True, True, True, False, True, True]
        )

        actual = pooling.forward_vectorized(
            tokens,
            batch_indices,
            batch_size=3,
            valid_mask=valid_mask,
        )
        expected = torch.stack([
            pooling(tokens[(batch_indices == index) & valid_mask])
            for index in range(3)
        ])

        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_empty_segment_uses_layer_norm_zero_state(self):
        pooling = attention_pooling(input_dim=4, proj_dim=3).double()
        with torch.no_grad():
            pooling.ln.bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        tokens = torch.randn(2, 4, dtype=torch.double)
        batch_indices = torch.tensor([0, 0])

        actual = pooling.forward_vectorized(tokens, batch_indices, batch_size=2)
        expected_empty = pooling.ln(torch.zeros(4, dtype=torch.double))

        torch.testing.assert_close(actual[1], expected_empty)


class MMRLAblationTest(unittest.TestCase):
    @staticmethod
    def _config(*, dynamic_cross_attention=True, pooling_mode="multi_query"):
        return SimpleNamespace(mmrl_config={
            "INSERT_LAYER": [1, 2],
            "RP_SPACE_LENGTH": 3,
            "RP_SPACE_DIM": 2,
            "vision_token_dim": 4,
            "text_token_dim": 6,
            "MMRL_MEMORY_QUERY_COUNT": 2,
            "MMRL_MEMORY_ATTENTION_DIM": 3,
            "MMRL_PROJECTOR_HIDDEN_DIM": 5,
            "MMRL_CROSS_ATTENTION_HEADS": 2,
            "MMRL_SAME_INIT_LAYER_PROJECTORS": True,
            "MMRL_USE_DYNAMIC_CROSS_ATTENTION": dynamic_cross_attention,
            "MMRL_MEMORY_POOLING_MODE": pooling_mode,
        })

    @staticmethod
    def _inputs():
        return {
            "visual_states": torch.randn(5, 4),
            "cu_seqlens": torch.tensor([0, 2, 5]),
            "text_states": torch.randn(2, 4, 6),
            "text_mask": torch.tensor([
                [True, True, False, False],
                [True, True, True, False],
            ]),
            "images_per_sample": [1, 1],
        }

    def test_masked_mean_pooling_ignores_invalid_tokens(self):
        pooling = MaskedMeanPooling(2, 2).double()
        pooling.output_norm = nn.Identity()
        with torch.no_grad():
            pooling.value_projection.weight.copy_(torch.eye(2, dtype=torch.double))
            pooling.value_projection.bias.zero_()

        states = torch.tensor([
            [[1.0, 3.0], [5.0, 7.0], [1000.0, 1000.0]],
            [[2.0, 4.0], [1000.0, 1000.0], [1000.0, 1000.0]],
        ], dtype=torch.double)
        mask = torch.tensor([
            [True, True, False],
            [True, False, False],
        ])

        actual = pooling(states, mask)
        expected = torch.tensor([[[3.0, 5.0]], [[2.0, 4.0]]], dtype=torch.double)
        torch.testing.assert_close(actual, expected)

        with self.assertRaisesRegex(RuntimeError, "without valid tokens"):
            pooling(states, torch.zeros_like(mask))

    def test_static_query_bypasses_memory_and_cross_attention(self):
        mmrl = MMRL(self._config(dynamic_cross_attention=False))
        outputs = mmrl(**self._inputs())

        self.assertEqual(len(outputs), 2)
        self.assertEqual([tuple(output.shape) for output in outputs], [
            (2, 3, 4),
            (2, 3, 4),
        ])
        self.assertEqual(mmrl.last_rep_shape, (2, 2, 3, 4))
        self.assertIsNone(mmrl.last_memory_shape)
        torch.testing.assert_close(outputs[0][0], outputs[0][1])

    def test_mean_pooling_uses_two_memory_tokens_per_image(self):
        mmrl = MMRL(self._config(pooling_mode="mean"))
        outputs = mmrl(**self._inputs())

        self.assertEqual(len(outputs), 2)
        self.assertEqual(mmrl.last_rep_shape, (2, 2, 3, 4))
        self.assertEqual(mmrl.last_memory_shape, (2, 2, 4))

    def test_text_guided_pooling_uses_dynamic_visual_slots(self):
        mmrl = MMRL(self._config(pooling_mode="text_guided"))
        inputs = self._inputs()
        outputs = mmrl(**inputs)

        self.assertEqual(len(outputs), 2)
        self.assertEqual(mmrl.last_rep_shape, (2, 2, 3, 4))
        self.assertEqual(mmrl.last_memory_shape, (2, 3, 4))
        self.assertIn(
            "text_guided_visual_attention_entropy_norm",
            mmrl.debug_context,
        )

        visual_padded, visual_mask = mmrl._pack_visual_states(
            inputs["visual_states"],
            inputs["cu_seqlens"],
        )
        first = mmrl.visual_memory_pooling(
            visual_padded,
            visual_mask,
            inputs["text_states"],
            inputs["text_mask"],
            inputs["images_per_sample"],
        )
        second = mmrl.visual_memory_pooling(
            visual_padded,
            visual_mask,
            inputs["text_states"].flip(-1),
            inputs["text_mask"],
            inputs["images_per_sample"],
        )
        self.assertFalse(torch.allclose(first, second))


if __name__ == "__main__":
    unittest.main()
