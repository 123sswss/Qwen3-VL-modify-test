import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from loraTest.data_protocol import (
    TRAIN_EXPERT_IMAGE_DIRS,
    TRAIN_EXPERT_JSONS,
)
from pathvqa.train_dynamic_prompt import (
    PROJECT_ROOT,
    _build_train_dataset,
    _dataset_display_name,
    _normalize_dataset_name,
)


def _load_data_pipeline_module():
    source = PROJECT_ROOT / "train" / "data_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "test_electrical_data_pipeline",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load data pipeline: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _NonEmptyDataset:
    def __len__(self):
        return 7


class ElectricalQDPTTest(unittest.TestCase):
    def test_tuple_json_inputs_are_loaded_as_multiple_sources(self):
        data_pipeline = _load_data_pipeline_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(json.dumps([{"id": 1}]), encoding="utf-8")
            second.write_text(json.dumps([{"id": 2}]), encoding="utf-8")

            paths = (first, second)
            self.assertEqual(
                data_pipeline.normalize_json_paths(paths),
                [str(first), str(second)],
            )
            loaded = data_pipeline.load_jsons(paths)

        self.assertEqual([item["id"] for item in loaded], [1, 2])
        self.assertEqual(
            [item["__source_json_path"] for item in loaded],
            [first, second],
        )

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
