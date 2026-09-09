import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from slake.prompt_tuning import StaticPromptTuningModel


class _FakeMultimodalModel(nn.Module):
    def __init__(self, vocab_size=16, hidden_size=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.config = SimpleNamespace(pad_token_id=0)
        self.generation_config = None
        self.last_input_ids = None
        self.last_embeddings = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        self.last_input_ids = input_ids.detach().clone()
        self.last_embeddings = self.embedding(input_ids)
        return SimpleNamespace(loss=self.last_embeddings.float().sum())

    def generate(self, input_ids, attention_mask, **kwargs):
        self.last_input_ids = input_ids.detach().clone()
        self.last_embeddings = self.embedding(input_ids)
        suffix = torch.ones(
            input_ids.shape[0],
            1,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, suffix), dim=1)


class _FakeVisualBlock(nn.Module):
    def forward(self, hidden_states, cu_seqlens=None, **kwargs):
        return hidden_states + hidden_states.mean(dim=0, keepdim=True)


class _FakeVisual(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.blocks = nn.ModuleList([_FakeVisualBlock(), _FakeVisualBlock()])
        self.anchor = nn.Parameter(torch.zeros(1, hidden_size))

    def forward(self, hidden_states, cu_seqlens):
        return self.blocks[1](hidden_states, cu_seqlens=cu_seqlens)


class _FakeVisualMultimodalModel(_FakeMultimodalModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = _FakeVisual()
        self.last_visual = None

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        self.last_input_ids = input_ids.detach().clone()
        self.last_embeddings = self.embedding(input_ids)
        visual_input = torch.randn(5, 8)
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
        self.last_visual = self.model.visual(visual_input, cu_seqlens)
        return SimpleNamespace(
            loss=self.last_embeddings.float().sum() + self.last_visual.float().sum()
        )


class StaticPromptTuningTest(unittest.TestCase):
    @staticmethod
    def _batch():
        return {
            "input_ids": torch.tensor([[2, 3, 4]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "labels": torch.tensor([[2, 3, 4]]),
        }

    def test_forward_replaces_prefix_embeddings_and_preserves_gradient(self):
        base = _FakeMultimodalModel()
        model = StaticPromptTuningModel(base, prompt_length=2, init_seed=5)
        output = model(**self._batch())

        self.assertEqual(tuple(base.last_input_ids.shape), (1, 5))
        torch.testing.assert_close(
            base.last_embeddings[:, :2].float(),
            model.soft_prompt.unsqueeze(0),
        )
        output.loss.backward()
        self.assertIsNotNone(model.soft_prompt.grad)
        self.assertGreater(float(model.soft_prompt.grad.norm()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))

    def test_generate_prepends_prompt_ids_without_passing_inputs_embeds(self):
        base = _FakeMultimodalModel()
        model = StaticPromptTuningModel(base, prompt_length=2, init_seed=5)
        generated = model.generate(**self._batch())

        self.assertEqual(tuple(base.last_input_ids.shape), (1, 5))
        self.assertEqual(tuple(generated.shape), (1, 6))
        torch.testing.assert_close(
            base.last_embeddings[:, :2].float(),
            model.soft_prompt.unsqueeze(0),
        )

    def test_static_visual_only_has_exact_parameter_budget_and_gradient(self):
        base = _FakeVisualMultimodalModel()
        model = StaticPromptTuningModel(
            base,
            prompt_length=0,
            init_seed=5,
            visual_prompt_length=3,
            visual_anchor_layers=(1,),
        )
        output = model(**self._batch())
        self.assertEqual(tuple(base.last_input_ids.shape), (1, 3))
        self.assertEqual(tuple(base.last_visual.shape), (5, 8))
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            24,
        )
        output.loss.backward()
        self.assertGreater(
            float(model.static_visual_prompt.visual_prompt.grad.norm()), 0.0
        )
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))

    def test_dual_static_checkpoint_round_trip(self):
        model = StaticPromptTuningModel(
            _FakeVisualMultimodalModel(),
            prompt_length=2,
            init_seed=5,
            visual_prompt_length=3,
            visual_anchor_layers=(1,),
        )
        with torch.no_grad():
            model.soft_prompt.add_(0.25)
            model.static_visual_prompt.visual_prompt.add_(0.5)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_prompt(Path(tmp))
            restored = StaticPromptTuningModel(
                _FakeVisualMultimodalModel(),
                prompt_length=2,
                init_seed=9,
                visual_prompt_length=3,
                visual_anchor_layers=(1,),
            )
            restored.load_prompt(Path(tmp))
        torch.testing.assert_close(restored.soft_prompt, model.soft_prompt)
        torch.testing.assert_close(
            restored.static_visual_prompt.visual_prompt,
            model.static_visual_prompt.visual_prompt,
        )


if __name__ == "__main__":
    unittest.main()
