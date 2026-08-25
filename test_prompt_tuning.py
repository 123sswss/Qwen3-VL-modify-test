import unittest
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


if __name__ == "__main__":
    unittest.main()
