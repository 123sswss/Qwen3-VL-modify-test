import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from train.train_stages import (
    Qwen3VLMMRLForStages,
    _build_stage3_epoch_decay_scheduler,
    _build_stage3_grouped_optimizer,
)


class _PromptInputHarness:
    _expand_soft_prompt_inputs = Qwen3VLMMRLForStages._expand_soft_prompt_inputs

    def __init__(self):
        self.soft_prompt_length = 2
        self.soft_prompt = nn.Parameter(torch.zeros(2, 4))
        self.config = SimpleNamespace(pad_token_id=0)


class _OptimizerHarness(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.MMRL = nn.Linear(2, 2, bias=False)
        self.soft_prompt = nn.Parameter(torch.zeros(2, 4))


class MMRLPromptTests(unittest.TestCase):
    def test_prompt_prefix_is_supervised_off_and_excluded_from_mmrl_pooling(self):
        harness = _PromptInputHarness()
        expanded = harness._expand_soft_prompt_inputs({
            "input_ids": torch.tensor([[5, 6, 7]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[-100, 6, 7]]),
            "mmrl_gating_mask": torch.tensor([[1, 1, 0]]),
        })

        self.assertEqual(expanded["input_ids"].tolist(), [[0, 0, 5, 6, 7]])
        self.assertEqual(expanded["attention_mask"].tolist(), [[1, 1, 1, 1, 1]])
        self.assertEqual(expanded["labels"].tolist(), [[-100, -100, -100, 6, 7]])
        self.assertEqual(expanded["mmrl_gating_mask"].tolist(), [[0, 0, 1, 1, 0]])

    def test_optimizer_has_disjoint_mmrl_and_prompt_groups(self):
        model = _OptimizerHarness()
        optimizer = _build_stage3_grouped_optimizer(model, {
            "learning_rate": {3: 6e-5},
            "experiment_cfg": {
                "stage3_mmrl_learning_rate": 6e-5,
                "stage3_prompt_learning_rate": 0.3,
                "stage3_prompt_warmup_ratio": 0.03,
                "expected_total_trainable_parameters": 12,
            },
        })

        self.assertEqual(
            [group["group_name"] for group in optimizer.param_groups],
            ["mmrl", "soft_prompt"],
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [6e-5, 0.3],
        )
        mmrl_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
        prompt_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
        self.assertTrue(mmrl_ids.isdisjoint(prompt_ids))

    def test_prompt_and_mmrl_use_independent_epoch_schedules(self):
        model = _OptimizerHarness()
        optimizer = _build_stage3_grouped_optimizer(model, {
            "learning_rate": {3: 6e-5},
            "experiment_cfg": {
                "stage3_mmrl_learning_rate": 6e-5,
                "stage3_prompt_learning_rate": 0.3,
                "expected_total_trainable_parameters": 12,
            },
        })
        scheduler = _build_stage3_epoch_decay_scheduler(
            optimizer,
            updates_per_epoch=100,
            total_epochs=3,
            warmup_ratio=0.10,
            epoch_decay=0.5,
        )

        mmrl_lambda, prompt_lambda = scheduler.lr_lambdas
        self.assertEqual(mmrl_lambda(100), 0.5)
        self.assertEqual(mmrl_lambda(200), 0.25)
        self.assertAlmostEqual(prompt_lambda(9), 1.0)
        self.assertAlmostEqual(prompt_lambda(300), 0.0)


if __name__ == "__main__":
    unittest.main()
