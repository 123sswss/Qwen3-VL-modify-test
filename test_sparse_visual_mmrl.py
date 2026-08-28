import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from slake.sparse_visual_mmrl import SparseVisualMMRL


class _SegmentMixingBlock(nn.Module):
    def forward(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None,
    ):
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


class SparseVisualMMRLTest(unittest.TestCase):
    def _adapter(self, initial_residual_scale=0.0):
        return SparseVisualMMRL(
            visual_dim=8,
            text_dim=12,
            anchor_layers=(1,),
            rep_token_count=2,
            attention_dim=4,
            num_heads=2,
            relation_weight=0.05,
            relation_max_tokens=4,
            initial_residual_scale=initial_residual_scale,
        )

    def test_zero_scale_is_exact_and_scale_receives_gradient(self):
        torch.manual_seed(7)
        visual = _FakeVisual()
        adapter = self._adapter()
        adapter.install(visual)
        hidden = torch.randn(7, 8)
        cu_seqlens = torch.tensor([0, 2, 4, 7], dtype=torch.int32)
        text_memory = torch.randn(2, 12)
        images_per_sample = torch.tensor([1, 1])
        grid_thw = torch.tensor([[1, 1, 4], [1, 1, 3]])

        expected = _SegmentMixingBlock()(hidden, cu_seqlens)
        with adapter.activate(text_memory, images_per_sample):
            adapter.prepare_visual(grid_thw)
            actual = visual.blocks[1](hidden, cu_seqlens=cu_seqlens)
            loss = actual.square().mean() + adapter.relation_loss * 0.05
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        loss.backward()
        self.assertIsNotNone(adapter.residual_scales.grad)
        self.assertGreater(float(adapter.residual_scales.grad.abs().sum()), 0.0)
        self.assertEqual(float(adapter.relation_loss), 0.0)

    def test_nonzero_scale_changes_tokens_and_trains_rep_generator(self):
        torch.manual_seed(11)
        visual = _FakeVisual()
        adapter = self._adapter()
        adapter.install(visual)
        adapter.residual_scales.data.fill_(0.2)
        adapter._forward_audited = True
        hidden = torch.randn(7, 8)
        cu_seqlens = torch.tensor([0, 2, 4, 7], dtype=torch.int32)
        with adapter.activate(torch.randn(2, 12), torch.tensor([1, 1])):
            adapter.prepare_visual(torch.tensor([[1, 1, 4], [1, 1, 3]]))
            output = visual.blocks[1](hidden, cu_seqlens=cu_seqlens)
            loss = output.square().mean() + adapter.relation_loss * 0.05
        self.assertFalse(torch.equal(output, _SegmentMixingBlock()(hidden, cu_seqlens)))
        loss.backward()
        self.assertGreater(float(adapter.shared_rep.grad.abs().sum()), 0.0)
        self.assertIn("sparse_visual_layer1_delta_ratio", adapter.debug_context)

    def test_production_parameter_count(self):
        adapter = SparseVisualMMRL(
            visual_dim=1024,
            text_dim=2560,
            anchor_layers=(5, 11, 17),
            rep_token_count=8,
            attention_dim=128,
            num_heads=4,
            relation_weight=0.05,
            initial_residual_scale=0.05,
        )
        parameter_count = sum(parameter.numel() for parameter in adapter.parameters())
        self.assertEqual(parameter_count, 1_201_155)
        torch.testing.assert_close(
            torch.tanh(adapter.residual_scales),
            torch.full((3,), 0.05),
        )


if __name__ == "__main__":
    unittest.main()
