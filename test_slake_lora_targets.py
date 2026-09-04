import unittest

from torch import nn

from slake.train_visual_peft import (
    EXPERIMENTS,
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


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList(_VisualBlock() for _ in range(24))
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList(
            _LanguageLayer() for _ in range(36)
        )


class SLAKELoRATargetTest(unittest.TestCase):
    def test_r8_config_matches_final_protocol(self):
        config = EXPERIMENTS["lora_full_model_attention_r8"]
        self.assertEqual(config["target_scope"], "full_model")
        self.assertEqual(config["rank"], 8)
        self.assertFalse(config["use_dora"])
        self.assertEqual(config["expected_trainable_parameters"], 7_077_888)

    def test_full_attention_targets_and_parameter_formula(self):
        model = _Model()
        visual, visual_layers = find_visual_attention_targets(model, None)
        language, language_layers = find_language_attention_targets(model)
        targets = visual + language

        self.assertEqual(visual_layers, list(range(24)))
        self.assertEqual(language_layers, list(range(36)))
        self.assertEqual(len(visual), 48)
        self.assertEqual(len(language), 144)
        self.assertEqual(len(targets), 192)
        self.assertEqual(len(targets), len(set(targets)))

        predicted = expected_lora_parameter_count(model, targets, rank=8)
        visual_per_layer = 8 * ((8 + 24) + (8 + 8))
        language_per_layer = 8 * (
            (8 + 8) + (8 + 4) + (8 + 4) + (8 + 8)
        )
        self.assertEqual(
            predicted,
            24 * visual_per_layer + 36 * language_per_layer,
        )


if __name__ == "__main__":
    unittest.main()
