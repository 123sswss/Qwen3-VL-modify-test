import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from slake.cocoop_prompt_tuning import (
    COCOOP_CONFIG_NAME,
    CoCoOpStylePromptTuningModel,
)


class _FakeTokenizer:
    token_ids = {
        "<|vision_start|>": 8,
        "<|image_pad|>": 9,
        "<|vision_end|>": 10,
    }

    def convert_tokens_to_ids(self, token):
        return self.token_ids[token]


class _FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_embeddings = None

    def forward(self, inputs_embeds=None, visual_pos_masks=None, **kwargs):
        self.last_embeddings = inputs_embeds
        return SimpleNamespace(loss=inputs_embeds.float().sum())


class _FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _FakeLanguageModel()


class _FakeMultimodalModel(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(16, hidden_size)
        self.model = _FakeCore()
        self.config = SimpleNamespace(pad_token_id=0)
        self.generation_config = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        pixel_values=None,
        **kwargs,
    ):
        embeddings = self.embedding(input_ids).clone()
        visual_mask = input_ids.eq(9)
        image_features = pixel_values.to(
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        embeddings[visual_mask] = image_features
        return self.model.language_model(
            inputs_embeds=embeddings,
            visual_pos_masks=visual_mask,
        )


class CoCoOpStylePromptTuningTest(unittest.TestCase):
    @staticmethod
    def _batch(image_value=1.0, question_token=2):
        return {
            "input_ids": torch.tensor([[8, 9, 10, question_token, 3]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "labels": torch.tensor([[-100, -100, -100, -100, 3]]),
            "pixel_values": torch.full((1, 8), float(image_value)),
        }

    @staticmethod
    def _model():
        return CoCoOpStylePromptTuningModel(
            _FakeMultimodalModel(),
            tokenizer=_FakeTokenizer(),
            prompt_length=2,
            bottleneck_dim=4,
            init_seed=5,
        )

    def test_prompt_depends_on_image_but_not_question(self):
        model = self._model()
        model(**self._batch(image_value=1.0, question_token=2))
        first = model.base_model.model.language_model.last_embeddings[:, :2].clone()
        model(**self._batch(image_value=2.0, question_token=2))
        second = model.base_model.model.language_model.last_embeddings[:, :2].clone()
        model(**self._batch(image_value=1.0, question_token=4))
        changed_question = (
            model.base_model.model.language_model.last_embeddings[:, :2].clone()
        )

        self.assertFalse(torch.allclose(first, second))
        torch.testing.assert_close(first, changed_question)
        self.assertEqual(
            float(model.debug_context["cocoop_visual_tokens_mean"]), 1.0
        )

    def test_gradients_are_limited_to_prompt_and_meta_net(self):
        model = self._model()
        output = model(**self._batch())
        output.loss.backward()
        self.assertGreater(float(model.soft_prompt.grad.norm()), 0.0)
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.meta_net.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.base_model.parameters())
        )
        grouped = {
            id(parameter)
            for parameters in model.trainable_parameter_groups().values()
            for parameter in parameters
        }
        active = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(grouped, active)

    def test_checkpoint_round_trip_records_approximation_boundary(self):
        model = self._model()
        with torch.no_grad():
            model.soft_prompt.add_(0.25)
            model.meta_net[-1].weight.add_(0.5)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            model.save_cocoop(checkpoint)
            config = json.loads(
                (checkpoint / COCOOP_CONFIG_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(config["source"], "CoCoOp_CVPR_2022")
            self.assertFalse(config["question_access"])
            restored = self._model()
            restored.load_cocoop(checkpoint)
            torch.testing.assert_close(restored.soft_prompt, model.soft_prompt)
            for key, value in model.meta_net.state_dict().items():
                torch.testing.assert_close(restored.meta_net.state_dict()[key], value)

    def test_p20_h160_parameter_budget_is_exact(self):
        model = CoCoOpStylePromptTuningModel(
            _FakeMultimodalModel(hidden_size=2560),
            tokenizer=_FakeTokenizer(),
            prompt_length=20,
            bottleneck_dim=160,
            init_seed=44,
        )
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(trainable, 873120)

    def test_missing_image_is_rejected(self):
        model = self._model()
        batch = self._batch()
        batch["input_ids"] = torch.tensor([[8, 7, 10, 2, 3]])
        with self.assertRaisesRegex(RuntimeError, "requires an image"):
            model(**batch)


if __name__ == "__main__":
    unittest.main()
