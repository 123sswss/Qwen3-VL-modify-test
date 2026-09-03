import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from loraTest.data_protocol import (
    TRAIN_EXPERT_IMAGE_DIRS,
    TRAIN_EXPERT_JSONS,
)
from pathvqa.train_dynamic_prompt import (
    _build_train_dataset,
    _dataset_display_name,
    _normalize_dataset_name,
)


class _NonEmptyDataset:
    def __len__(self):
        return 7


class ElectricalQDPTTest(unittest.TestCase):
    def test_electrical_dataset_uses_only_private_multimodal_expert_data(self):
        args = SimpleNamespace(data_seed=42)
        dataset_class = Mock(return_value=_NonEmptyDataset())
        collator_class = Mock()
        with patch(
            "pathvqa.train_dynamic_prompt._load_electrical_data_pipeline",
            return_value=(dataset_class, collator_class),
        ):
            dataset = _build_train_dataset("electrical", args, object())

        self.assertEqual(len(dataset), 7)
        kwargs = dataset_class.call_args.kwargs
        self.assertEqual(kwargs["expert_json"], TRAIN_EXPERT_JSONS)
        self.assertEqual(kwargs["expert_img_dir"], TRAIN_EXPERT_IMAGE_DIRS)
        self.assertEqual(kwargs["general_json"], ())
        self.assertEqual(kwargs["general_img_dir"], ())
        self.assertEqual(kwargs["enable_views"], ("expert-mm",))
        self.assertTrue(kwargs["ce_enabled"])
        self.assertTrue(kwargs["deterministic_sampling"])
        self.assertEqual(kwargs["assistant_turn_policy"], "joint")

    def test_electrical_dataset_name_is_registered(self):
        self.assertEqual(_normalize_dataset_name("Electrical"), "electrical")
        self.assertEqual(_dataset_display_name("electrical"), "Electrical")


if __name__ == "__main__":
    unittest.main()
