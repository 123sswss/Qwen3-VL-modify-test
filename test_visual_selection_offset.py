"""Small CPU contracts for the independent V0 method (requires torch)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from slake.visual_selection_offset import (
    EXPECTED_TRAINABLE, VisualSelectionOffsetModel, locate_question_mask,
)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class _Block(nn.Module):
    def forward(self, hidden_states, cu_seqlens=None, position_embeddings=None, **kwargs):
        del cu_seqlens, position_embeddings, kwargs
        # A cheap cross-token interaction makes Visual18 affect native tokens.
        return hidden_states + hidden_states.mean(dim=0, keepdim=True) * 0.01


class _Merger(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1024, 2560, bias=False)

    def forward(self, hidden):
        return self.projection(hidden.reshape(-1, 4, 1024).mean(dim=1))


class _Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=1024)
        self.spatial_merge_size = 2
        self.blocks = nn.ModuleList(_Block() for _ in range(18))
        self.merger = _Merger()
        self.calls = 0

    def forward(self, hidden, grid_thw):
        self.calls += 1
        self.last_grid = grid_thw
        cu = torch.tensor([0, hidden.shape[0]], dtype=torch.int32)
        pos = (torch.ones_like(hidden), torch.zeros_like(hidden))
        for block in self.blocks:
            hidden = block(hidden, cu_seqlens=cu, position_embeddings=pos)
        return self.merger(hidden)


class _Language(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, inputs_embeds):
        self.calls += 1
        return SimpleNamespace(loss=inputs_embeds.square().mean())


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 2560)
        self.model = nn.Module()
        self.model.visual = _Vision()
        self.model.language_model = _Language()
        self.config = SimpleNamespace(pad_token_id=0)
        self.generation_config = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw,
                labels=None):
        del attention_mask, labels
        inputs = self.embedding(input_ids)
        value = self.model.visual(pixel_values, image_grid_thw)
        inputs = inputs.clone()
        inputs[:, 20 : 20 + value.shape[0]] = value
        return self.model.language_model(inputs_embeds=inputs)


class VisualSelectionOffsetTest(unittest.TestCase):
    def test_exact_question_span_rejects_duplicate_and_answer(self):
        ids = torch.tensor([1, ord("q"), ord("?"), 2, ord("q"), ord("?")])
        context = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool)
        mask = locate_question_mask(ids, "q?", _Tokenizer(), context_mask=context)
        self.assertEqual(mask.nonzero().flatten().tolist(), [1, 2])
        with self.assertRaises(ValueError):
            locate_question_mask(ids, "q?", _Tokenizer())

    def test_parameter_budget_and_first_real_backward(self):
        torch.manual_seed(44)
        base = _Base()
        model = VisualSelectionOffsetModel(base, init_seed=44)
        groups = model.trainable_parameter_groups()
        self.assertEqual(sum(sum(p.numel() for p in v) for v in groups.values()), EXPECTED_TRAINABLE)
        self.assertTrue(all(not p.requires_grad for p in model.base_model.parameters()))
        self.assertAlmostEqual(float(model.offset_up.weight.std()), 1e-4, delta=1e-5)
        self.assertTrue(torch.equal(model.offset_up.bias, torch.zeros_like(model.offset_up.bias)))
        self.assertAlmostEqual(float(model.visual_s8.std()), 0.02, delta=0.002)
        self.assertAlmostEqual(float(model.visual_av10.std()), 0.02, delta=0.002)
        ids = torch.tensor([[11, 12, 13, 14, 15]])
        question_mask = torch.tensor([[False, False, True, True, False]])
        labels = torch.tensor([[-100, -100, -100, -100, 15]])
        # Two post-merger visual tokens are necessary: with only one, all
        # spatial maps collapse to probability 1 and Q/K gradients are zero.
        pixels = torch.randn(8, 1024)
        output = model(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            pixel_values=pixels,
            image_grid_thw=torch.tensor([[1, 2, 4]]),
            labels=labels, question_mask=question_mask,
        )
        output.loss.backward()
        self.assertEqual(base.model.visual.calls, 1)
        self.assertEqual(base.model.language_model.calls, 1)
        self.assertLess(model.first_batch_diagnostics["offset_to_question_rms"], 0.1)
        by_id = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
        for group, params in groups.items():
            self.assertGreater(
                sum(model.first_backward_gradients.get(by_id[id(p)], 0) for p in params),
                0, group,
            )
        first_condition = {name: value.clone() for name, value in model.debug_context.items()}
        changed_answer = ids.clone()
        changed_answer[0, -1] = 16
        model(
            input_ids=changed_answer, attention_mask=torch.ones_like(ids),
            pixel_values=pixels, image_grid_thw=torch.tensor([[1, 2, 4]]),
            labels=labels, question_mask=question_mask,
        )
        for name, value in first_condition.items():
            self.assertTrue(torch.equal(value, model.debug_context[name]), name)

    def test_maps_keep_images_separate(self):
        model = VisualSelectionOffsetModel(_Base(), init_seed=44)
        question = torch.randn(2, 3, 2560)
        valid = torch.ones(2, 3, dtype=torch.bool)
        grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
        model._features = {
            5: torch.randn(8, 1024),
            11: torch.randn(8, 1024),
            17: torch.randn(8, 1024),
            "value": torch.randn(2, 2560),
        }
        first, _ = model._condition(question, valid, grid)
        changed = model._features["value"].clone()
        changed[1, 0] += 10
        model._features["value"] = changed
        second, _ = model._condition(question, valid, grid)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertFalse(torch.equal(first[1], second[1]))

if __name__ == "__main__":
    unittest.main()
