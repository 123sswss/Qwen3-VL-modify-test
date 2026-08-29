import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from slake.dynamic_prompt_tuning import (
    DynamicPromptCrossAttention,
    DynamicPromptTuningModel,
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


class _FakeVisualBlock(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _FakeVisualMerger(nn.Module):
    def __init__(self, visual_size=8, text_size=8, merge_unit=2):
        super().__init__()
        self.linear = nn.Linear(visual_size * merge_unit, text_size, bias=False)

    def forward(self, hidden_states):
        return self.linear(hidden_states.view(-1, self.linear.in_features))


class _FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.blocks = nn.ModuleList(_FakeVisualBlock() for _ in range(2))
        self.merger = _FakeVisualMerger()
        self.spatial_merge_unit = 2


class _FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _FakeLanguageModel()
        self.visual = _FakeVisual()


class _FakeMultimodalModel(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(16, hidden_size)
        self.model = _FakeCore()
        self.config = SimpleNamespace(pad_token_id=0)
        self.generation_config = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        embeddings = self.embedding(input_ids)
        visual_mask = input_ids.eq(9)
        return self.model.language_model(
            inputs_embeds=embeddings,
            visual_pos_masks=visual_mask,
        )


class DynamicPromptTuningTest(unittest.TestCase):
    @staticmethod
    def _batch(answer_in_context=False, text_token=2):
        context = [1, 1, 1, 1, int(answer_in_context)]
        return {
            "input_ids": torch.tensor([[8, 9, 10, text_token, 3]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "labels": torch.tensor([[-100, -100, -100, -100, 3]]),
            "mmrl_gating_mask": torch.tensor([context], dtype=torch.bool),
        }

    def _model(self):
        return DynamicPromptTuningModel(
            _FakeMultimodalModel(),
            tokenizer=_FakeTokenizer(),
            prompt_length=2,
            init_seed=5,
            attention_dim=4,
            num_heads=2,
        )

    def test_zero_init_matches_static_prompt_and_receives_gradients(self):
        model = self._model()
        output = model(**self._batch())
        prefix = model.base_model.model.language_model.last_embeddings[:, :2]
        torch.testing.assert_close(
            prefix.float(),
            model.soft_prompt.unsqueeze(0),
        )
        self.assertEqual(
            float(model.debug_context["dynamic_prompt_delta_norm_mean"]),
            0.0,
        )
        self.assertEqual(
            float(model.debug_context["dynamic_prompt_visual_tokens_mean"]),
            1.0,
        )
        self.assertEqual(
            float(model.debug_context["dynamic_prompt_text_tokens_mean"]),
            1.0,
        )

        output.loss.backward()
        self.assertGreater(float(model.soft_prompt.grad.norm()), 0.0)
        self.assertGreater(
            float(model.dynamic_prompt.output_projection.weight.grad.norm()),
            0.0,
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.base_model.parameters())
        )

    def test_supervised_answer_tokens_cannot_enter_text_memory(self):
        model = self._model()
        with self.assertRaisesRegex(RuntimeError, "must not include supervised"):
            model(**self._batch(answer_in_context=True))

    def test_checkpoint_round_trip_preserves_all_trainable_state(self):
        model = self._model()
        with torch.no_grad():
            model.soft_prompt.add_(0.25)
            model.dynamic_prompt.output_projection.weight.add_(0.5)
        with tempfile.TemporaryDirectory() as directory:
            model.save_dynamic_prompt(Path(directory))
            restored = self._model()
            restored.load_dynamic_prompt(Path(directory))
            torch.testing.assert_close(restored.soft_prompt, model.soft_prompt)
            for key, value in model.dynamic_prompt.state_dict().items():
                torch.testing.assert_close(
                    restored.dynamic_prompt.state_dict()[key], value
                )
            restored(**self._batch())
            self.assertGreater(
                float(restored.debug_context["dynamic_prompt_delta_norm_mean"]),
                0.0,
            )

    def test_zero_intervention_removes_trained_dynamic_residual(self):
        model = self._model().eval()
        with torch.no_grad():
            model.dynamic_prompt.output_projection.weight.fill_(0.25)
            model.dynamic_prompt._forward_audited = True
            model.configure_inference_intervention("zero")
            model(**self._batch())
        prefix = model.base_model.model.language_model.last_embeddings[:, :2]
        torch.testing.assert_close(prefix.float(), model.soft_prompt.unsqueeze(0))
        self.assertEqual(
            model.inference_intervention_summary()["samples_changed"],
            1,
        )

    def test_mean_residual_reuses_calibration_delta(self):
        model = self._model().eval()
        with torch.no_grad():
            model.dynamic_prompt.output_projection.weight.fill_(0.25)
            model.dynamic_prompt._forward_audited = True
            model.configure_inference_intervention("mean-residual", memory_lag=1)
            model(**self._batch(text_token=2))
            first = model.base_model.model.language_model.last_embeddings[:, :2].clone()
            model(**self._batch(text_token=4))
            second = model.base_model.model.language_model.last_embeddings[:, :2]
        torch.testing.assert_close(second, first)
        summary = model.inference_intervention_summary()
        self.assertEqual(summary["warmup_samples"], 1)
        self.assertEqual(summary["samples_changed"], 1)

    def test_lagged_memory_uses_prior_sample_memory(self):
        model = self._model().eval()
        with torch.no_grad():
            model.dynamic_prompt.output_projection.weight.fill_(0.25)
            model.dynamic_prompt._forward_audited = True
            model.configure_inference_intervention("lagged-memory", memory_lag=1)
            model(**self._batch(text_token=2))
            first = model.base_model.model.language_model.last_embeddings[:, :2].clone()
            model(**self._batch(text_token=4))
            second = model.base_model.model.language_model.last_embeddings[:, :2]
        torch.testing.assert_close(second, first)
        summary = model.inference_intervention_summary()
        self.assertEqual(summary["warmup_samples"], 1)
        self.assertEqual(summary["samples_changed"], 1)

    def test_cross_attention_accepts_two_memory_tokens(self):
        attention = DynamicPromptCrossAttention(
            hidden_size=8,
            attention_dim=4,
            num_heads=2,
        )
        delta = attention(torch.randn(2, 3, 8), torch.randn(2, 2, 8))
        self.assertEqual(tuple(delta.shape), (2, 3, 8))
        self.assertNotIn("dynamic_prompt_shared_s_attention_mean", attention.debug_context)

    def test_direct_shared_s_prompt_has_no_independent_prompt_or_p0(self):
        model = DynamicPromptTuningModel(
            _FakeMultimodalModel(),
            tokenizer=_FakeTokenizer(),
            prompt_length=20,
            init_seed=5,
            attention_dim=4,
            num_heads=2,
            sparse_visual_anchor_layers=(1,),
            sparse_visual_rep_tokens=4,
            sparse_visual_attention_dim=4,
            sparse_visual_heads=2,
            shared_s_text_mode="direct_prompt",
        )
        self.assertIsNone(model.soft_prompt)
        self.assertEqual(model.prompt_length, 2)
        self.assertEqual(model.trainable_parameter_groups()["soft_prompt"], [])
        output = model(**self._batch())
        prefix = model.base_model.model.language_model.last_embeddings[:, :2]
        expected = model.sparse_visual.shared_s_text_prompt().unsqueeze(0)
        torch.testing.assert_close(prefix.float(), expected.float())
        output.loss.backward()
        self.assertGreater(float(model.sparse_visual.shared_rep.grad.norm()), 0.0)
        self.assertGreater(
            float(model.dynamic_prompt.output_projection.weight.grad.norm()),
            0.0,
        )

    def test_separate_shared_s_residual_preserves_independent_prompt_at_step_zero(self):
        model = DynamicPromptTuningModel(
            _FakeMultimodalModel(),
            tokenizer=_FakeTokenizer(),
            prompt_length=2,
            init_seed=5,
            attention_dim=4,
            num_heads=2,
            sparse_visual_anchor_layers=(1,),
            sparse_visual_rep_tokens=4,
            sparse_visual_attention_dim=4,
            sparse_visual_heads=2,
            shared_s_text_mode="separate_residual",
            shared_s_attention_dim=4,
            shared_s_heads=2,
        )
        output = model(**self._batch())
        prefix = model.base_model.model.language_model.last_embeddings[:, :2]
        torch.testing.assert_close(prefix.float(), model.soft_prompt.unsqueeze(0))
        output.loss.backward()
        self.assertIsNotNone(model.shared_s_prompt_attention)
        self.assertGreater(
            float(
                model.shared_s_prompt_attention.output_projection.weight.grad.norm()
            ),
            0.0,
        )

    def test_text_owned_s_remains_the_prompt_and_visual_side_has_no_shared_rep(self):
        model = DynamicPromptTuningModel(
            _FakeMultimodalModel(),
            tokenizer=_FakeTokenizer(),
            prompt_length=2,
            init_seed=5,
            attention_dim=4,
            num_heads=2,
            sparse_visual_anchor_layers=(1,),
            sparse_visual_rep_tokens=4,
            sparse_visual_attention_dim=4,
            sparse_visual_heads=2,
            shared_s_text_mode="text_owned_visual_readonly",
            shared_s_visual_bottleneck_dim=4,
        )
        self.assertIsNotNone(model.soft_prompt)
        self.assertEqual(model.prompt_length, 2)
        self.assertIsNone(model.sparse_visual.shared_rep)
        self.assertEqual(
            model.sparse_visual.text_anchor_tokens,
            model.prompt_length,
        )
        groups = model.trainable_parameter_groups()
        self.assertEqual(groups["soft_prompt"], [model.soft_prompt])
        self.assertTrue(
            any(
                parameter is model.sparse_visual.text_anchor_token_mixer
                for parameter in groups["sparse_visual"]
            )
        )

        output = model(**self._batch())
        prefix = model.base_model.model.language_model.last_embeddings[:, :2]
        torch.testing.assert_close(prefix.float(), model.soft_prompt.unsqueeze(0))
        output.loss.backward()
        self.assertGreater(float(model.soft_prompt.grad.norm()), 0.0)

    def test_text_owned_s_checkpoint_round_trip(self):
        kwargs = dict(
            tokenizer=_FakeTokenizer(),
            prompt_length=2,
            init_seed=5,
            attention_dim=4,
            num_heads=2,
            sparse_visual_anchor_layers=(1,),
            sparse_visual_rep_tokens=4,
            sparse_visual_attention_dim=4,
            sparse_visual_heads=2,
            shared_s_text_mode="text_owned_visual_readonly",
            shared_s_visual_bottleneck_dim=4,
        )
        model = DynamicPromptTuningModel(_FakeMultimodalModel(), **kwargs)
        with torch.no_grad():
            model.soft_prompt.add_(0.25)
            model.sparse_visual.text_anchor_token_mixer.add_(0.5)
        with tempfile.TemporaryDirectory() as directory:
            model.save_dynamic_prompt(Path(directory))
            restored = DynamicPromptTuningModel(_FakeMultimodalModel(), **kwargs)
            restored.load_dynamic_prompt(Path(directory))
            torch.testing.assert_close(restored.soft_prompt, model.soft_prompt)
            for key, value in model.sparse_visual.state_dict().items():
                torch.testing.assert_close(
                    restored.sparse_visual.state_dict()[key],
                    value,
                )

    def test_full_workspace_checkpoint_and_optimizer_groups_round_trip(self):
        kwargs = dict(
            tokenizer=_FakeTokenizer(),
            prompt_length=2,
            init_seed=5,
            attention_dim=4,
            num_heads=2,
            sparse_visual_anchor_layers=(1,),
            sparse_visual_rep_tokens=2,
            sparse_visual_attention_dim=4,
            sparse_visual_heads=2,
            shared_workspace=True,
            workspace_tokens=3,
            workspace_dim=8,
            workspace_heads=2,
            workspace_ffn_dim=16,
            workspace_text_attention_dim=4,
            workspace_text_heads=2,
            workspace_visual_attention_dim=4,
            workspace_visual_heads=2,
        )
        model = DynamicPromptTuningModel(_FakeMultimodalModel(), **kwargs)
        groups = model.trainable_parameter_groups()
        self.assertIn("shared_workspace", groups)
        grouped_ids = {
            id(parameter)
            for parameters in groups.values()
            for parameter in parameters
        }
        self.assertEqual(
            grouped_ids,
            {
                id(parameter)
                for parameter in model.parameters()
                if parameter.requires_grad
            },
        )
        with torch.no_grad():
            model.sparse_visual.workspace_seed.add_(0.25)
            model.workspace_text_attention.output_projection.weight.add_(0.5)
        with tempfile.TemporaryDirectory() as directory:
            model.save_dynamic_prompt(Path(directory))
            restored = DynamicPromptTuningModel(_FakeMultimodalModel(), **kwargs)
            restored.load_dynamic_prompt(Path(directory))
            torch.testing.assert_close(
                restored.sparse_visual.workspace_seed,
                model.sparse_visual.workspace_seed,
            )
            for key, value in model.workspace_text_attention.state_dict().items():
                torch.testing.assert_close(
                    restored.workspace_text_attention.state_dict()[key],
                    value,
                )


if __name__ == "__main__":
    unittest.main()
