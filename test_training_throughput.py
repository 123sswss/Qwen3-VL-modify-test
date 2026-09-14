import unittest
import sys
import types

try:
    import torch
except ModuleNotFoundError:
    torch = None
    torch_stub = types.ModuleType("torch")
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.Trainer = object
    transformers_stub.TrainerCallback = object
    sys.modules["torch"] = torch_stub
    sys.modules["transformers"] = transformers_stub

from pathvqa.throughput_benchmark import (
    TrainingWorkTracker,
    estimated_optimizer_steps,
)


class TrainingThroughputTest(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is unavailable in the local test runtime")
    def test_work_tracker_counts_samples_and_post_merge_visual_tokens(self):
        tracker = TrainingWorkTracker(spatial_merge_size=2)
        tracker.observe(
            {
                "input_ids": torch.zeros((2, 4), dtype=torch.long),
                "image_grid_thw": torch.tensor([[1, 8, 8], [2, 4, 4]]),
            }
        )
        self.assertEqual(tracker.snapshot(), {"samples": 2, "visual_tokens": 24})

    def test_estimated_optimizer_steps_uses_complete_epochs(self):
        self.assertEqual(estimated_optimizer_steps(19654, 32, 3), 1845)

    def test_invalid_work_tracker_merge_size_is_rejected(self):
        with self.assertRaises(ValueError):
            TrainingWorkTracker(spatial_merge_size=0)


if __name__ == "__main__":
    unittest.main()
