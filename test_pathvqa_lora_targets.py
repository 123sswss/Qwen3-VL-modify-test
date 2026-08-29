import unittest

from torch import nn

from pathvqa.train_visual_lora import (
    expected_lora_parameter_count,
    find_language_attention_targets,
    find_visual_attention_targets,
)


class _VisualAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 24, bias=False)
        self.proj = nn.Linear(8, 8, bias=False)


class _VisualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _VisualAttention()
        self.mlp = nn.Linear(8, 8, bias=False)


class _VisualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_VisualBlock() for _ in range(24))


class _LanguageAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)


class _LanguageLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _LanguageAttention()
        self.mlp = nn.Linear(8, 8, bias=False)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(_LanguageLayer() for _ in range(36))


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _VisualModel()
        self.language_model = _LanguageModel()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Core()


class PathVQALoRATargetTest(unittest.TestCase):
    def test_full_model_attention_targets_are_exact(self):
        model = _Model()
        visual, visual_layers = find_visual_attention_targets(model, 24)
        language, language_layers = find_language_attention_targets(model)
        targets = visual + language

        self.assertEqual(len(visual), 48)
        self.assertEqual(visual_layers, list(range(24)))
        self.assertEqual(len(language), 144)
        self.assertEqual(language_layers, list(range(36)))
        self.assertEqual(len(targets), 192)
        self.assertFalse(any("mlp" in name for name in targets))
        self.assertEqual(len(targets), len(set(targets)))

    def test_parameter_prediction_uses_linear_input_and_output_widths(self):
        model = _Model()
        visual, _ = find_visual_attention_targets(model, 24)
        language, _ = find_language_attention_targets(model)
        predicted = expected_lora_parameter_count(
            model,
            visual + language,
            rank=8,
        )
        visual_per_layer = 8 * ((8 + 24) + (8 + 8))
        language_per_layer = 8 * ((8 + 8) + (8 + 4) + (8 + 4) + (8 + 8))
        self.assertEqual(
            predicted,
            24 * visual_per_layer + 36 * language_per_layer,
        )


if __name__ == "__main__":
    unittest.main()
