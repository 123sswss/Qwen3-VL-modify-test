"""V10: question-weighted three maps -> one native Value -> H160 Meta-Net."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from diagnostics.v10_protocol import (
    METHOD, CONFIG_NAME, WEIGHTS_NAME, EXPECTED_GROUP_COUNTS, EXPECTED_TRAINABLE,
)
from slake.visual_selection_prefix import VisualSelectionPrefixModel
from slake.visual_selection_offset import LAYERS, QUESTION_WIDTH, OUTPUT_INIT_STD


def fuse_merged_maps(merged_maps, beta):
    """Convex combination ONLY; no second softmax, mean, temperature or scale."""
    return sum(beta[index] * merged_maps[index] for index in range(3))


class VisualSelectionV10Model(VisualSelectionPrefixModel):
    method_name = METHOD

    def __init__(self, base_model, init_seed=44):
        # Build the exact original V1 first; retained tensors and its RNG draw
        # sequence are unchanged. Deleted heads never survive into training.
        print("[V10_PARENT_INITIALIZATION_ONLY] temporary_V1_heads=True never_optimized_or_saved=True", flush=True)
        super().__init__(base_model, init_seed=init_seed)
        retained = {name: p.detach().cpu().clone() for name, p in self.named_parameters()
                    if p.requires_grad and not name.startswith(("value_norms.", "value_blocks.", "prefix_output."))}
        del self.value_norms
        del self.value_blocks
        del self.prefix_output
        # Private CPU initialization stream: new head does not alter training RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(torch.Generator(device="cpu").manual_seed(self.init_seed).get_state())
            self.meta_net = nn.Sequential(nn.Linear(2560, 160), nn.ReLU(), nn.Linear(160, 2560))
            nn.init.normal_(self.meta_net[2].weight, mean=0.0, std=OUTPUT_INIT_STD)
            nn.init.zeros_(self.meta_net[2].bias)
        self.meta_net.to(device=self.p20.device, dtype=torch.float32)
        final = dict(self.named_parameters())
        if any(not torch.equal(value, final[name].detach().cpu()) for name, value in retained.items()):
            raise RuntimeError("V10 retained initialization differs from same-seed parent V1")
        self.initialization_audit = {
            "reference": "exact_parent_original_V1_construction_same_seed",
            "shared_tensor_count": len(retained), "all_shared_initial_values_equal": True,
            "visual18_order": "S8_then_Av10", "visual18_init": "Normal(0,0.02)",
            "head_rng": "private_CPU_seed_equals_model_seed_global_state_restored",
            "meta_first_init": "nn.Linear_default_Kaiming_uniform_and_bias",
            "meta_output_init": "Normal(0,1e-4)", "meta_output_bias_zero": True,
            "meta_output_actual_std": float(self.meta_net[2].weight.detach().std()),
        }
        for handle in self._first_grad_handles:
            handle.remove()
        self._first_grad_handles = []
        self.first_backward_gradients = {}
        self._install_first_backward_hooks()
        self._audit_parameters()
        print("[V10_INITIALIZATION_AUDIT] " + json.dumps(self.initialization_audit), flush=True)

    def trainable_parameter_groups(self):
        if not hasattr(self, "meta_net"):
            return VisualSelectionPrefixModel.trainable_parameter_groups(self)
        return {
            "p20": [self.p20], "visual_s8": [self.visual_s8], "visual_av10": [self.visual_av10],
            "question_context": list(self.text_projection.parameters()) + list(self.question_depthwise.parameters())
                + list(self.question_pointwise.parameters()) + list(self.question_norm.parameters())
                + list(self.question_pool.parameters()),
            "maps": list(self.query_heads.parameters()) + list(self.key_norms.parameters()) + list(self.key_heads.parameters()),
            "layer_condition": list(self.layer_gate.parameters()),
            "meta_net": list(self.meta_net.parameters()),
        }

    def _audit_parameters(self):
        if not hasattr(self, "meta_net"):
            return VisualSelectionPrefixModel._audit_parameters(self)
        groups = self.trainable_parameter_groups()
        grouped = [p for values in groups.values() for p in values]
        active = [p for p in self.parameters() if p.requires_grad]
        if len(grouped) != len({id(p) for p in grouped}) or {id(p) for p in active} != {id(p) for p in grouped}:
            raise RuntimeError("V10 optimizer groups are not a complete unique partition")
        counts = {name: sum(p.numel() for p in values) for name, values in groups.items()}
        if counts != EXPECTED_GROUP_COUNTS or sum(counts.values()) != EXPECTED_TRAINABLE:
            raise RuntimeError(f"V10 parameter budget mismatch: {counts}")
        if any(hasattr(self, name) for name in ("value_norms", "value_blocks", "prefix_output", "offset_down", "offset_up")):
            raise RuntimeError("Removed V1/V0 heads remain in V10")
        if any(p.requires_grad for p in self.base_model.parameters()):
            raise RuntimeError("V10 backbone must remain frozen")
        print("[V10_PARAMETER_AUDIT] " + json.dumps(counts) + f" total={sum(counts.values())}", flush=True)
        return counts

    def _condition(self, question, valid, grid, probe=None):
        features = self._features
        if set(features) != {*LAYERS, "value"}:
            raise RuntimeError("V10 requires all three real-token layers and native merger Value")
        x = self.text_projection(question) * valid.unsqueeze(-1)
        conv = self.question_pointwise(torch.nn.functional.gelu(self.question_depthwise(x.transpose(1,2))))
        x = (x + conv.transpose(1,2)) * valid.unsqueeze(-1)
        logits = self.question_pool(self.question_norm(x)).squeeze(-1).masked_fill(~valid, -torch.inf)
        u = (logits.softmax(dim=-1).unsqueeze(-1) * x).sum(dim=1)
        beta = self.layer_gate(u).float().softmax(dim=-1)
        if grid.ndim != 2 or grid.shape != (question.shape[0], 3):
            raise RuntimeError("V10 requires exactly one image grid per sample")
        geometry = grid.detach().cpu().tolist()
        if any(int(t) < 1 or int(h) < 2 or int(w) < 2 or int(h)%2 or int(w)%2 for t,h,w in geometry):
            raise RuntimeError("V10 invalid 2x2 block-major image geometry")
        patches = [int(t*h*w) for t,h,w in geometry]
        counts = [n//4 for n in patches]
        if features["value"].shape != (sum(counts), 2560):
            raise RuntimeError("V10 merger Value/image grid mismatch")
        values = torch.split(features["value"].to(question.device, torch.float32), counts, dim=0)
        layer_maps, diagnostics = [], {}
        for li, layer in enumerate(LAYERS):
            hidden = features[layer]
            if hidden.shape != (sum(patches), 1024):
                raise RuntimeError(f"V10 layer{layer} includes temporary Prompt/padding or wrong grid")
            segments = torch.split(hidden.to(question.device, torch.float32), patches, dim=0)
            queries = self.query_heads[li](u)
            maps, entropies = [], []
            for b, segment in enumerate(segments):
                keys = self.key_heads[li](self.key_norms[li](segment))
                probability = ((keys * queries[b]).sum(dim=-1) / math.sqrt(QUESTION_WIDTH)).float().softmax(dim=0)
                merged = probability.reshape(-1,4).sum(dim=1)
                if merged.numel() != counts[b] or not torch.allclose(merged.sum(), merged.new_tensor(1.), atol=1e-4):
                    raise RuntimeError("V10 map-to-native-Value alignment/mass mismatch")
                maps.append(merged)
                entropies.append(-(probability * probability.clamp_min(1e-12).log()).sum()/math.log(probability.numel()))
            layer_maps.append(maps)
            diagnostics[f"map{layer}_entropy_norm"] = torch.stack(entropies).mean().detach()
            diagnostics[f"map{layer}_top_mass"] = torch.stack([m.max() for m in maps]).mean().detach()
            diagnostics[f"layer{layer}_weight"] = beta[:,li].mean().detach()
            if probe is not None:
                probe[f"map{layer}"] = [m.detach().float().cpu().clone() for m in maps]
        summaries, fused_maps = [], []
        for b, value in enumerate(values):
            fused = fuse_merged_maps([layer_maps[li][b] for li in range(3)], beta[b])
            if not bool(torch.isfinite(fused).all()) or bool((fused < 0).any()) or not torch.allclose(
                    fused.sum(), fused.new_tensor(1.), atol=1e-4):
                raise RuntimeError("V10 weighted fused map is not a finite per-image probability")
            summaries.append((fused[:,None] * value).sum(dim=0))
            fused_maps.append(fused)
        z = torch.stack(summaries)
        diagnostics["fused_map_mass_error"] = torch.stack([(m.sum()-1).abs() for m in fused_maps]).max().detach()
        diagnostics["fused_map_entropy_norm"] = torch.stack([
            -(m*m.clamp_min(1e-12).log()).sum()/max(math.log(m.numel()), 1e-12) for m in fused_maps]).mean().detach()
        diagnostics["summary_rms"] = z.square().mean().sqrt().detach()
        if probe is not None:
            probe["layer_weights"] = beta.detach().float().cpu().clone()
            probe["fused_map"] = [m.detach().float().cpu().clone() for m in fused_maps]
            probe["summary"] = z.detach().float().cpu().clone()
        return z, diagnostics

    def _prefix_shift(self, condition):
        return self.meta_net(condition)

    def save_v10(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        config = {"method": METHOD, "init_seed": self.init_seed, "layers": list(LAYERS),
                  "visual_tokens": [8,10], "prefix_tokens": 20, "hidden_size": 2560,
                  "bottleneck_dim": 160, "map_fusion": "beta_weighted_sum_no_resoftmax_no_divide",
                  "output_init_std": OUTPUT_INIT_STD, "trainable_parameters": self._audit_parameters()}
        (path/CONFIG_NAME).write_text(json.dumps(config, indent=2), encoding="utf-8")
        torch.save({name: p.detach().cpu() for name,p in self.named_parameters() if p.requires_grad}, path/WEIGHTS_NAME)

    def load_v10(self, directory):
        path = Path(directory)
        config = json.loads((path/CONFIG_NAME).read_text(encoding="utf-8"))
        for key, expected in {"method": METHOD, "init_seed": self.init_seed, "layers": list(LAYERS),
                "visual_tokens": [8,10], "prefix_tokens": 20, "hidden_size": 2560, "bottleneck_dim": 160,
                "map_fusion": "beta_weighted_sum_no_resoftmax_no_divide", "trainable_parameters": EXPECTED_GROUP_COUNTS}.items():
            if config.get(key) != expected:
                raise ValueError(f"V10 checkpoint architecture mismatch: {key}")
        state = torch.load(path/WEIGHTS_NAME, map_location="cpu", weights_only=True)
        parameters = {name:p for name,p in self.named_parameters() if p.requires_grad}
        if set(state) != set(parameters):
            raise ValueError("V10 checkpoint parameter set mismatch")
        for name,p in parameters.items():
            if state[name].shape != p.shape or not bool(torch.isfinite(state[name]).all()):
                raise ValueError(f"V10 checkpoint invalid tensor: {name}")
        with torch.no_grad():
            for name,p in parameters.items():
                p.copy_(state[name].to(p.device))
