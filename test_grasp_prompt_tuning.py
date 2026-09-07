from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from slake.grasp_prompt_tuning import (
    GRASPPromptTuningModel,
    entmax15,
    fixed_2d_sincos_encoding,
    pool_spatial_blocks,
)


class FakeTokenizer:
    ids = {"<|image_pad|>": 10, "<|vision_start|>": 11, "<|vision_end|>": 12}

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, -1)


class FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.last_inputs = None

    def forward(self, inputs_embeds, **kwargs):
        self.calls += 1
        self.last_inputs = inputs_embeds
        return SimpleNamespace(last_hidden_state=inputs_embeds * 1.25)


class FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(spatial_merge_size=2)


class FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = FakeVisual()
        self.language_model = FakeLanguageModel()


class FakeBaseModel(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(64, hidden_size)
        self.model = FakeCore()
        self.config = SimpleNamespace(pad_token_id=0, use_cache=False)
        self.generation_config = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, **kwargs):
        embeddings = self.embedding(input_ids)
        output = self.model.language_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            visual_pos_masks=input_ids.eq(10),
            return_dict=True,
        )
        return SimpleNamespace(loss=output.last_hidden_state.float().sum())


class GRASPTest(unittest.TestCase):
    def test_entmax15_is_normalized_and_sparse(self):
        probabilities = entmax15(torch.tensor([[10.0, 0.0, -1.0, -2.0]]))
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))
        self.assertGreater(int(probabilities.eq(0).sum()), 0)

    def test_spatial_pooling_is_lossless_for_uneven_grid(self):
        tokens = torch.arange(30, dtype=torch.float32).reshape(15, 2)
        pooled = pool_spatial_blocks(tokens, 3, 5, 4)
        self.assertEqual(tuple(pooled.shape), (4, 2))
        expected = tokens.reshape(3, 5, 2)[:1, :2].reshape(-1, 2).mean(0)
        torch.testing.assert_close(pooled[0], expected)

    def test_position_encoding_is_fixed_and_shaped(self):
        first = fixed_2d_sincos_encoding(4, 10)
        second = fixed_2d_sincos_encoding(4, 10)
        self.assertEqual(tuple(first.shape), (4, 10))
        torch.testing.assert_close(first, second)

    def test_prompt_prototypes_remain_registered_trainable_parameters(self):
        model = GRASPPromptTuningModel(
            FakeBaseModel(), FakeTokenizer(), block_count=4, bottleneck_dim=4
        )
        self.assertIsInstance(model.prompt_prototypes, nn.Parameter)
        self.assertTrue(model.prompt_prototypes.requires_grad)
        self.assertIn("prompt_prototypes", dict(model.named_parameters()))

    def test_forward_uses_two_llm_passes_and_only_grasp_gets_gradients(self):
        base = FakeBaseModel()
        model = GRASPPromptTuningModel(
            base, FakeTokenizer(), block_count=4, bottleneck_dim=4, init_seed=44
        )
        input_ids = torch.tensor([[11, 10, 10, 10, 10, 12, 20, 21, 22]])
        labels = torch.tensor([[-100, -100, -100, -100, -100, -100, -100, -100, 22]])
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
            mmrl_gating_mask=labels.eq(-100),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            pixel_values=torch.zeros(1, 2),
        )
        output.loss.backward()
        self.assertEqual(base.model.language_model.calls, 2)
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        for parameters in model.trainable_parameter_groups().values():
            self.assertTrue(all(parameter.grad is not None for parameter in parameters))
        self.assertEqual(tuple(base.model.language_model.last_inputs.shape), (1, 10, 8))
        self.assertIn("grasp_zero_weight_fraction", model.debug_context)

    def test_answer_overlap_is_rejected(self):
        model = GRASPPromptTuningModel(
            FakeBaseModel(), FakeTokenizer(), block_count=4, bottleneck_dim=4
        )
        with self.assertRaisesRegex(RuntimeError, "overlaps answer"):
            model(
                input_ids=torch.tensor([[10, 10, 10, 10, 20]]),
                attention_mask=torch.ones(1, 5, dtype=torch.long),
                labels=torch.tensor([[-100, -100, -100, -100, 20]]),
                mmrl_gating_mask=torch.ones(1, 5, dtype=torch.bool),
                image_grid_thw=torch.tensor([[1, 4, 4]]),
            )

    def test_checkpoint_round_trip(self):
        model = GRASPPromptTuningModel(
            FakeBaseModel(), FakeTokenizer(), block_count=4, bottleneck_dim=4
        )
        with tempfile.TemporaryDirectory() as directory:
            model.save_grasp(directory)
            restored = GRASPPromptTuningModel(
                FakeBaseModel(), FakeTokenizer(), block_count=4, bottleneck_dim=4
            )
            restored.load_grasp(Path(directory))
            torch.testing.assert_close(restored.prompt_prototypes, model.prompt_prototypes)
            torch.testing.assert_close(
                restored.visual_key_projection.weight,
                model.visual_key_projection.weight,
            )


if __name__ == "__main__":
    unittest.main()
