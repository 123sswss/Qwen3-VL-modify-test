import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from slake.sparse_visual_mmrl import LatentResidualAttention, SparseVisualMMRL


class _SegmentMixingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None,
    ):
        self.calls += 1
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outputs = []
        for segment in torch.split(hidden_states, lengths, dim=0):
            outputs.append(segment + segment.mean(dim=0, keepdim=True))
        return torch.cat(outputs, dim=0)


class _FakeVisual(nn.Module):
    def __init__(self, depth=3, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.blocks = nn.ModuleList(_SegmentMixingBlock() for _ in range(depth))


class _FakeMerger(nn.Module):
    def __init__(self, visual_dim=8, text_dim=12, merge_unit=4):
        super().__init__()
        self.linear = nn.Linear(visual_dim * merge_unit, text_dim, bias=False)

    def forward(self, hidden_states):
        return self.linear(hidden_states.view(-1, self.linear.in_features))


class _FakeMergedVisual(_FakeVisual):
    def __init__(self):
        super().__init__()
        self.merger = _FakeMerger()
        self.spatial_merge_unit = 4


class SparseVisualMMRLTest(unittest.TestCase):
    def _adapter(self):
        return SparseVisualMMRL(
            visual_dim=8,
            text_dim=12,
            anchor_layers=(1,),
            rep_token_count=2,
            attention_dim=4,
            num_heads=2,
        )

    def test_inactive_wrapper_executes_original_block_once(self):
        torch.manual_seed(7)
        visual = _FakeVisual()
        adapter = self._adapter()
        adapter.install(visual)
        hidden = torch.randn(7, 8)
        cu_seqlens = torch.tensor([0, 2, 4, 7], dtype=torch.int32)

        expected = _SegmentMixingBlock()(hidden, cu_seqlens)
        actual = visual.blocks[1](hidden, cu_seqlens=cu_seqlens)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(visual.blocks[1].block.calls, 1)

    def test_active_injection_executes_block_once_and_trains_rep_generator(self):
        torch.manual_seed(11)
        visual = _FakeVisual()
        adapter = self._adapter()
        adapter.install(visual)
        hidden = torch.randn(7, 8)
        cu_seqlens = torch.tensor([0, 2, 4, 7], dtype=torch.int32)
        with adapter.activate(torch.randn(2, 12), torch.tensor([1, 1])):
            adapter.prepare_visual(torch.tensor([[1, 1, 4], [1, 1, 3]]))
            output = visual.blocks[1](hidden, cu_seqlens=cu_seqlens)
            loss = output.square().mean()
        self.assertEqual(visual.blocks[1].block.calls, 1)
        self.assertFalse(torch.equal(output, _SegmentMixingBlock()(hidden, cu_seqlens)))
        loss.backward()
        self.assertGreater(float(adapter.shared_rep.grad.abs().sum()), 0.0)
        self.assertIn(
            "sparse_visual_layer1_rep_to_input_norm_ratio",
            adapter.debug_context,
        )

    def test_production_parameter_count(self):
        adapter = SparseVisualMMRL(
            visual_dim=1024,
            text_dim=2560,
            anchor_layers=(5, 11, 17),
            rep_token_count=8,
            attention_dim=128,
            num_heads=4,
        )
        parameter_count = sum(parameter.numel() for parameter in adapter.parameters())
        self.assertEqual(parameter_count, 1_201_152)

    def test_raw_shared_s_maps_to_text_without_visual_conditioning(self):
        torch.manual_seed(13)
        visual = _FakeMergedVisual()
        for parameter in visual.parameters():
            parameter.requires_grad = False
        adapter = SparseVisualMMRL(
            visual_dim=8,
            text_dim=12,
            anchor_layers=(1,),
            rep_token_count=4,
            attention_dim=4,
            num_heads=2,
            map_shared_s_to_text=True,
        )
        adapter.install(visual)
        self.assertNotIn(
            "merger",
            dict(adapter.named_modules()),
        )
        shared_prompt = adapter.shared_s_text_prompt()
        expected = visual.merger(adapter.shared_rep)
        self.assertEqual(adapter.shared_s_text_prompt_length, 1)
        self.assertEqual(tuple(shared_prompt.shape), (1, 12))
        torch.testing.assert_close(shared_prompt, expected)
        loss = shared_prompt.square().mean()
        loss.backward()
        self.assertGreater(float(adapter.shared_rep.grad.abs().sum()), 0.0)
        self.assertGreater(float(adapter.shared_s_text_prompt_grad_norm()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in visual.parameters()))

    def test_text_owned_s_is_detached_while_visual_adapter_trains(self):
        adapter = SparseVisualMMRL(
            visual_dim=8,
            text_dim=12,
            anchor_layers=(1,),
            rep_token_count=4,
            attention_dim=4,
            num_heads=2,
            text_anchor_tokens=6,
            text_anchor_bottleneck_dim=4,
        )
        text_anchor = torch.randn(6, 12, requires_grad=True)
        visual_rep = adapter.adapt_read_only_text_anchor(text_anchor)
        self.assertEqual(tuple(visual_rep.shape), (4, 8))
        visual_rep.square().mean().backward()
        self.assertIsNone(text_anchor.grad)
        self.assertIsNone(adapter.shared_rep)
        self.assertGreater(
            float(adapter.text_anchor_up_projection.weight.grad.norm()),
            0.0,
        )
        self.assertIn(
            "shared_s_visual_adapter_token_entropy_norm",
            adapter._text_anchor_debug,
        )

    def test_full_workspace_uses_tokens_and_preserves_private_visual_path_at_init(self):
        torch.manual_seed(17)
        visual = _FakeVisual()
        adapter = SparseVisualMMRL(
            visual_dim=8,
            text_dim=12,
            anchor_layers=(1,),
            rep_token_count=2,
            attention_dim=4,
            num_heads=2,
            shared_workspace=True,
            workspace_tokens=3,
            workspace_dim=8,
            workspace_heads=2,
            workspace_ffn_dim=16,
            workspace_visual_attention_dim=4,
            workspace_visual_heads=2,
        )
        adapter.install(visual)
        hidden = torch.randn(7, 8)
        text_tokens = torch.randn(2, 4, 12)
        text_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        cu_seqlens = torch.tensor([0, 2, 4, 7], dtype=torch.int32)
        with adapter.activate(
            torch.randn(2, 12),
            torch.tensor([1, 1]),
            text_tokens=text_tokens,
            text_token_mask=text_mask,
        ):
            adapter.prepare_visual(torch.tensor([[1, 1, 4], [1, 1, 3]]))
            output = visual.blocks[1](hidden, cu_seqlens=cu_seqlens)
            workspace = adapter.shared_workspace_text_memory()
            self.assertEqual(tuple(workspace.shape), (2, 3, 8))
            self.assertEqual(
                float(
                    adapter.debug_context[
                        "workspace_layer1_visual_delta_to_base_ratio"
                    ]
                ),
                0.0,
            )
            loss = output.square().mean() + workspace.square().mean()
        loss.backward()
        self.assertGreater(float(adapter.shared_rep.grad.norm()), 0.0)
        self.assertGreater(float(adapter.workspace_seed.grad.norm()), 0.0)
        private_ids = {id(parameter) for parameter in adapter.private_parameters()}
        workspace_ids = {
            id(parameter) for parameter in adapter.shared_workspace_parameters()
        }
        self.assertTrue(private_ids.isdisjoint(workspace_ids))
        self.assertEqual(
            private_ids | workspace_ids,
            {id(parameter) for parameter in adapter.parameters()},
        )

    def test_full_workspace_production_parameter_count(self):
        adapter = SparseVisualMMRL(
            visual_dim=1024,
            text_dim=2560,
            anchor_layers=(5, 11, 17),
            rep_token_count=8,
            attention_dim=128,
            num_heads=4,
            shared_workspace=True,
            workspace_tokens=32,
            workspace_dim=1024,
            workspace_heads=16,
            workspace_ffn_dim=4096,
            workspace_visual_attention_dim=1024,
            workspace_visual_heads=16,
        )
        private_count = sum(
            parameter.numel() for parameter in adapter.private_parameters()
        )
        workspace_count = sum(
            parameter.numel()
            for parameter in adapter.shared_workspace_parameters()
        )
        self.assertEqual(private_count, 1_201_152)
        self.assertEqual(workspace_count, 65_659_904)
        text_head = LatentResidualAttention(
            query_dim=2560,
            memory_dim=1024,
            attention_dim=1024,
            num_heads=16,
        )
        text_head_count = sum(
            parameter.numel() for parameter in text_head.parameters()
        )
        self.assertEqual(text_head_count, 7_349_760)
        self.assertEqual(
            51_200 + 2_634_240 + private_count + workspace_count + text_head_count,
            76_896_256,
        )


if __name__ == "__main__":
    unittest.main()
