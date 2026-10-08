"""Local numpy/AST CPU checks; no torch, model import, CUDA or dataset generation."""
import ast
from pathlib import Path
import unittest
import numpy as np

from diagnostics.v10_protocol import (
    METHOD, CONFIG_NAME, WEIGHTS_NAME, EXPECTED_TRAINABLE, EXPECTED_GROUP_COUNTS, GROUP_LRS,
)

ROOT = Path(__file__).resolve().parents[1]


class V10ProtocolTest(unittest.TestCase):
    def test_parameter_budget_from_dimensions_and_lrs(self):
        counts = {"p20":20*2560,"visual_s8":8*1024,"visual_av10":10*1024,
            "question_context":2560*128+128*3+128+128*128+128+2*128+128,
            "maps":3*(128*128+128+2*1024+1024*128),"layer_condition":128*3+3,
            "meta_net":2560*160+160+160*2560+2560}
        self.assertEqual(counts,EXPECTED_GROUP_COUNTS)
        self.assertEqual(sum(counts.values()),EXPECTED_TRAINABLE)
        self.assertEqual(GROUP_LRS["p20"],.3)
        self.assertEqual(GROUP_LRS["visual_s8"],3e-5)
        self.assertTrue(all(rate == 1e-4 for name,rate in GROUP_LRS.items() if name not in ("p20","visual_s8")))

    def test_actual_fusion_function_preserves_mass_and_value_formula(self):
        tree = ast.parse((ROOT/"slake/visual_selection_v10.py").read_text(encoding="utf-8"))
        node = next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name == "fuse_merged_maps")
        scope = {}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"actual_v10_fusion","exec"),scope)
        fuse = scope["fuse_merged_maps"]
        rng = np.random.default_rng(42)
        for count in (1,2,6,18):
            maps = rng.random((3,count)); maps /= maps.sum(axis=1,keepdims=True)
            beta = np.array([.1,.2,.7])
            values = rng.normal(size=(count,2560))
            fused = fuse(maps,beta)
            self.assertAlmostEqual(float(fused.sum()),1.)
            np.testing.assert_allclose(fused @ values, beta @ (maps @ values),rtol=1e-12,atol=1e-12)
        maps = np.array([[1.,0.],[0.,1.],[.5,.5]])
        np.testing.assert_allclose(fuse(maps,np.array([.1,.2,.7])),[.45,.55])
        # Dividing by3 or re-softmax would fail this intentionally nonuniform reference.
        np.testing.assert_allclose(fuse(maps,np.array([1.,0.,0.])),[1.,0.])

    def test_independent_storage_and_removed_heads(self):
        self.assertNotEqual(CONFIG_NAME,"visual_selection_prefix_config.json")
        self.assertNotEqual(WEIGHTS_NAME,"visual_selection_prefix.pt")
        self.assertIn("v10",METHOD)
        tree = ast.parse((ROOT/"slake/visual_selection_v10.py").read_text(encoding="utf-8"))
        deleted = {
            target.attr for node in ast.walk(tree) if isinstance(node,ast.Delete)
            for target in node.targets if isinstance(target,ast.Attribute)}
        self.assertTrue({"value_norms","value_blocks","prefix_output"}.issubset(deleted))
        src = (ROOT/"slake/visual_selection_v10.py").read_text(encoding="utf-8")
        self.assertNotIn("torch.manual_seed(",src)  # CPU head initialization must not reseed CUDA.
        self.assertIn("probability.reshape(-1,4).sum(dim=1)",src)

    def test_fixed_training_and_norm(self):
        source = (ROOT/"slake/train_v10.py").read_text(encoding="utf-8")
        self.assertIn("num_train_epochs=5",source)
        self.assertIn("per_device_train_batch_size=2",source)
        self.assertIn("gradient_accumulation_steps=16",source)
        self.assertIn("trainer.model_accepts_loss_kwargs = False",source)
        self.assertIn("epoch in (3,4,5)",source)
        self.assertNotIn("eval_strategy=",source)


if __name__ == "__main__":
    unittest.main()
