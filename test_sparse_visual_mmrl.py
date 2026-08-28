import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from slake.sparse_visual_mmrl import SparseVisualMMRL


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


if __name__ == "__main__":
    unittest.main()
