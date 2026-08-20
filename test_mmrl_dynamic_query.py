import unittest
from types import SimpleNamespace

import torch

from MMRL import DynamicSourceAttentionPooling, MMRL, ResidualCrossAttention


class DynamicQueryPoolingTest(unittest.TestCase):
    def test_masked_source_tokens_do_not_change_output(self):
        torch.manual_seed(7)
        pooling = DynamicSourceAttentionPooling(
            input_dim=6,
            output_dim=8,
            attention_dim=4,
        )
        pooling.train()
        query_states = torch.randn(2, 3, 8)
        projected_queries = torch.randn(2, 3, 4)
        source_states = torch.randn(2, 5, 6)
        valid_mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, True, True, False],
        ])

        expected = pooling(
            query_states, projected_queries, source_states, valid_mask
        )
        changed_source = source_states.clone()
        changed_source[~valid_mask] = 1000.0
        actual = pooling(
            query_states, projected_queries, changed_source, valid_mask
        )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (2, 3, 8))
        entropy = pooling.debug_context["attention_entropy_norm"]
        self.assertTrue(bool(torch.isfinite(entropy)))
        self.assertGreaterEqual(float(entropy), 0.0)
        self.assertLessEqual(float(entropy), 1.0 + 1e-6)

    def test_dynamic_queries_feed_standard_cross_attention(self):
        torch.manual_seed(11)
        hidden_dim = 8
        attention_dim = 4
        query_count = 3
        layer_count = 2
        batch_size = 2

        query_projection = torch.nn.Linear(hidden_dim, attention_dim)
        text_pooling = DynamicSourceAttentionPooling(6, hidden_dim, attention_dim)
        visual_pooling = DynamicSourceAttentionPooling(
            hidden_dim, hidden_dim, attention_dim
        )
        cross_attention = ResidualCrossAttention(hidden_dim, num_heads=2)

        seeds = torch.randn(batch_size, query_count, hidden_dim)
        text_states = torch.randn(batch_size, 5, 6)
        visual_states = torch.randn(batch_size, 7, hidden_dim)
        text_mask = torch.ones(batch_size, 5, dtype=torch.bool)
        visual_mask = torch.ones(batch_size, 7, dtype=torch.bool)
        text_queries = text_pooling(
            seeds,
            query_projection(seeds),
            text_states,
            text_mask,
        )
        multimodal_queries = visual_pooling(
            text_queries,
            query_projection(text_queries),
            visual_states,
            visual_mask,
        )
        layer_queries = multimodal_queries.unsqueeze(1).expand(
            -1, layer_count, -1, -1
        )
        static_slots = torch.randn(
            batch_size, layer_count, query_count, hidden_dim
        )

        output, delta = cross_attention(
            layer_queries.reshape(-1, query_count, hidden_dim),
            static_slots.reshape(-1, query_count, hidden_dim),
        )

        self.assertEqual(
            tuple(output.shape),
            (batch_size * layer_count, query_count, hidden_dim),
        )
        torch.testing.assert_close(delta, torch.zeros_like(delta))
        torch.testing.assert_close(
            output,
            layer_queries.reshape(-1, query_count, hidden_dim),
        )

    def test_full_dynamic_mmrl_path_handles_multiple_images(self):
        config = SimpleNamespace(mmrl_config={
            "DIRECT_SHARED_REP": False,
            "MMRL_QUERY_ARCHITECTURE": "layer_mlp_dynamic_query_static_kv",
            "MMRL_SAME_INIT_LAYER_PROJECTORS": True,
            "INSERT_LAYER": [0, 1],
            "RP_SPACE_LENGTH": 3,
            "RP_SPACE_DIM": 4,
            "vision_token_dim": 8,
            "text_token_dim": 6,
            "MMRL_MEMORY_QUERY_COUNT": 3,
            "MMRL_MEMORY_ATTENTION_DIM": 4,
            "MMRL_PROJECTOR_HIDDEN_DIM": 8,
            "MMRL_CROSS_ATTENTION_HEADS": 2,
        })
        mmrl = MMRL(config)
        mmrl.synchronize_layer_projector_initialization()
        mmrl.train()
        visual_states = torch.randn(12, 8)
        cu_seqlens = torch.tensor([0, 4, 7, 12])
        text_states = torch.randn(2, 6, 6)
        text_mask = torch.tensor([
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ])

        layer_reps = mmrl(
            visual_states=visual_states,
            cu_seqlens=cu_seqlens,
            text_states=text_states,
            text_mask=text_mask,
            images_per_sample=[2, 1],
        )

        self.assertEqual(len(layer_reps), 2)
        self.assertEqual(tuple(layer_reps[0].shape), (3, 3, 8))
        self.assertEqual(mmrl.last_memory_shape, (3, 3, 8))
        loss = sum(rep.square().mean() for rep in layer_reps)
        loss.backward()
        self.assertIsNotNone(mmrl.dynamic_query_seeds.grad)
        self.assertTrue(bool(torch.isfinite(mmrl.dynamic_query_seeds.grad).all()))
        self.assertIsNotNone(mmrl.cross_attention.output_projection.weight.grad)


if __name__ == "__main__":
    unittest.main()
