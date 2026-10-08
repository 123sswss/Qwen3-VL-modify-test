"""CPU tensor checks for V10; requires torch (never uses CUDA/mock real backbone)."""
import copy
from pathlib import Path
import tempfile
import unittest
import torch

from slake.visual_selection_prefix import VisualSelectionPrefixModel
from slake.visual_selection_v10 import VisualSelectionV10Model, fuse_merged_maps
from diagnostics.v10_protocol import EXPECTED_TRAINABLE, WEIGHTS_NAME
from test_visual_selection_offset import _Base


class VisualSelectionV10Test(unittest.TestCase):
    def test_initialization_budget_forward_backward_and_roundtrip(self):
        torch.random.default_generator.manual_seed(44)
        base = _Base()
        common_rng = torch.random.get_rng_state()
        v1 = VisualSelectionPrefixModel(copy.deepcopy(base), init_seed=44)
        after_v1 = torch.random.get_rng_state()
        expected = {name:p.detach().clone() for name,p in v1.named_parameters() if p.requires_grad
                    and not name.startswith(("value_norms.","value_blocks.","prefix_output."))}
        torch.random.set_rng_state(common_rng)
        model = VisualSelectionV10Model(base,init_seed=44)
        self.assertTrue(torch.equal(torch.random.get_rng_state(),after_v1))
        named = dict(model.named_parameters())
        self.assertTrue(all(torch.equal(value,named[name]) for name,value in expected.items()))
        self.assertEqual(sum(model._audit_parameters().values()),EXPECTED_TRAINABLE)
        self.assertTrue(torch.equal(model.meta_net[2].bias,torch.zeros(2560)))
        self.assertAlmostEqual(float(model.meta_net[2].weight.detach().std()),1e-4,delta=1e-5)
        batch = {"input_ids":torch.tensor([[11,12,13,14,15]]),"attention_mask":torch.ones(1,5,dtype=torch.long),
                 "pixel_values":torch.randn(8,1024),"image_grid_thw":torch.tensor([[1,2,4]]),
                 "labels":torch.tensor([[-100,-100,-100,-100,15]]),
                 "question_source_ids":torch.tensor([[13,14]]),"question_source_mask":torch.ones(1,2,dtype=torch.bool)}
        model(**batch).loss.backward()
        self.assertEqual(base.model.visual.calls,1)
        self.assertEqual(base.model.language_model.calls,1)
        self.assertTrue(model.last_injection_audit["native_embeddings_unchanged"])
        for name,params in model.trainable_parameter_groups().items():
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),name)
        with tempfile.TemporaryDirectory() as temp:
            model.save_v10(temp)
            state = torch.load(Path(temp)/WEIGHTS_NAME,weights_only=True)
            self.assertFalse(any(key.startswith(("value_norms.","value_blocks.","prefix_output.")) for key in state))
            original = model.p20.detach().clone()
            with torch.no_grad():model.p20.add_(1)
            model.load_v10(temp)
            self.assertTrue(torch.equal(original,model.p20))

    def test_per_image_mixed_grids_and_fusion_gradients(self):
        torch.random.default_generator.manual_seed(44)
        model = VisualSelectionV10Model(_Base(),init_seed=44)
        model._features = {layer:torch.randn(24,1024) for layer in (5,11,17)}
        model._features["value"] = torch.randn(6,2560)
        probe = {}
        z,_ = model._condition(torch.randn(2,4,2560),torch.tensor([[1,1,1,1],[1,1,0,0]],dtype=torch.bool),
                               torch.tensor([[1,2,4],[1,4,4]]),probe)
        self.assertEqual(z.shape,(2,2560))
        for i,count in enumerate((2,4)):
            self.assertEqual(probe["fused_map"][i].numel(),count)
            self.assertAlmostEqual(float(probe["fused_map"][i].sum()),1.,places=5)
        model._prefix_shift(z).square().sum().backward()
        self.assertGreater(float(model.layer_gate.weight.grad.norm()),0)
        self.assertTrue(all(float(head.weight.grad.norm()) > 0 for head in model.query_heads))
        maps = [torch.tensor([1.,0.],requires_grad=True),torch.tensor([0.,1.],requires_grad=True),
                torch.tensor([.5,.5],requires_grad=True)]
        beta = torch.tensor([.1,.2,.7],requires_grad=True)
        fused = fuse_merged_maps(maps,beta)
        self.assertTrue(torch.allclose(fused,torch.tensor([.45,.55])))
        fused[0].backward()
        self.assertTrue(all(m.grad is not None for m in maps))


if __name__ == "__main__":
    unittest.main()
