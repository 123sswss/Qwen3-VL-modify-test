"""Matched five-epoch V1 training ablations, with exact parent initialization."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from slake.visual_selection_prefix import (
    CONFIG_NAME, WEIGHTS_NAME, EXPECTED_GROUP_COUNTS, VisualSelectionPrefixModel,
)

EXPECTED_TOTALS = {"c_static": 69_632, "c_qmap": 1_815_811, "c_no_visual": 1_846_531}
DYNAMIC_MODULES = (
    "text_projection", "question_depthwise", "question_pointwise", "question_norm",
    "question_pool", "query_heads", "key_norms", "key_heads", "layer_gate",
    "value_norms", "value_blocks", "prefix_output",
)


class SharedMapQuery(nn.Module):
    """A single FP32 vector per layer; batch size is the only input dependency."""

    def __init__(self, original_bias):
        super().__init__()
        self.query = nn.Parameter(original_bias.detach().float().clone())

    def forward(self, u):
        return self.query.unsqueeze(0).expand(u.shape[0], -1)


class V1AblationModel(VisualSelectionPrefixModel):
    ablation_mode = ""

    def __init__(self, base_model, init_seed=44):
        self._ablation_ready = False
        super().__init__(base_model, init_seed=init_seed)
        reference = {n: p.detach().cpu().clone() for n, p in self.named_parameters()
                     if p.requires_grad}
        rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if self.p20.is_cuda else []
        if self.ablation_mode == "c_static":
            for name in DYNAMIC_MODULES:
                delattr(self, name)
        elif self.ablation_mode == "c_qmap":
            self.query_heads = nn.ModuleList(SharedMapQuery(head.bias) for head in self.query_heads)
        elif self.ablation_mode == "c_no_visual":
            del self.visual_s8
            del self.visual_av10
            self.skip_visual_prompt = True
        else:
            raise ValueError(f"Unknown V1 ablation: {self.ablation_mode}")
        if not torch.equal(rng, torch.random.get_rng_state()):
            raise RuntimeError("Ablation transformation consumed global CPU RNG")
        if cuda_rng and any(not torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())):
            raise RuntimeError("Ablation transformation consumed global CUDA RNG")
        actual = {n: p for n, p in self.named_parameters() if p.requires_grad}
        shared = set(reference) & set(actual)
        mismatches = [n for n in shared if not torch.equal(reference[n], actual[n].detach().cpu())]
        if mismatches:
            raise RuntimeError(f"Shared V1 initial tensors changed: {mismatches}")
        self.initialization_audit = {
            "reference": "exact_same_seed_parent_V1_before_deletion_or_replacement",
            "equal": True, "shared_tensor_count": len(shared),
            "removed_tensors": sorted(set(reference) - set(actual)),
            "added_tensors": sorted(set(actual) - set(reference)),
            "global_rng_unchanged": True,
            "query_initialization": (
                "original_Linear128_bias_clone_Uniform(-1/sqrt(128),1/sqrt(128)); original_query_at_u_zero"
                if self.ablation_mode == "c_qmap" else None
            ),
            "query_vectors": (
                [head.query.detach().cpu().tolist() for head in self.query_heads]
                if self.ablation_mode == "c_qmap" else None
            ),
        }
        for handle in self._first_grad_handles:
            handle.remove()
        self._first_grad_handles = []
        self.first_backward_gradients = {}
        self._install_first_backward_hooks()
        self._ablation_ready = True
        self._audit_parameters()

    @property
    def method_name(self):
        return "visual_selection_prefix_" + self.ablation_mode

    def trainable_parameter_groups(self):
        if not self._ablation_ready:
            return VisualSelectionPrefixModel.trainable_parameter_groups(self)
        if self.ablation_mode == "c_static":
            return {"p20": [self.p20], "visual_s8": [self.visual_s8],
                    "visual_av10": [self.visual_av10]}
        # Parent grouping refers to Visual18 explicitly; build the retained
        # dynamic partition without placeholder visual parameters.
        groups = {
            "p20": [self.p20],
            "question_context": list(self.text_projection.parameters())
            + list(self.question_depthwise.parameters()) + list(self.question_pointwise.parameters())
            + list(self.question_norm.parameters()) + list(self.question_pool.parameters()),
            "maps": list(self.query_heads.parameters()) + list(self.key_norms.parameters())
            + list(self.key_heads.parameters()),
            "layer_condition": list(self.layer_gate.parameters()) + list(self.value_norms.parameters())
            + list(self.value_blocks.parameters()),
            "prefix_output": list(self.prefix_output.parameters()),
        }
        if self.ablation_mode != "c_no_visual":
            groups.update(visual_s8=[self.visual_s8], visual_av10=[self.visual_av10])
        return groups

    def _audit_parameters(self):
        if not self._ablation_ready:
            return VisualSelectionPrefixModel._audit_parameters(self)
        groups = self.trainable_parameter_groups()
        grouped = [p for values in groups.values() for p in values]
        active = [p for p in self.parameters() if p.requires_grad]
        if len(grouped) != len({id(p) for p in grouped}) or {id(p) for p in grouped} != {id(p) for p in active}:
            raise RuntimeError("Ablation optimizer partition is not complete and unique")
        expected = dict(EXPECTED_GROUP_COUNTS)
        if self.ablation_mode == "c_static":
            expected = {k: expected[k] for k in ("p20", "visual_s8", "visual_av10")}
        elif self.ablation_mode == "c_no_visual":
            expected.pop("visual_s8")
            expected.pop("visual_av10")
        else:
            expected["maps"] = 399_744
        counts = {k: sum(p.numel() for p in values) for k, values in groups.items()}
        if counts != expected or sum(counts.values()) != EXPECTED_TOTALS[self.ablation_mode]:
            raise RuntimeError(f"Ablation parameter mismatch: {counts} expected={expected}")
        if any(p.requires_grad for p in self.base_model.parameters()):
            raise RuntimeError("Ablation backbone must be frozen")
        print(f"[V1_ABLATION_PARAMETERS] mode={self.ablation_mode} counts={counts} total={sum(counts.values())}")
        return counts

    def _condition(self, question, valid, grid, probe=None):
        if self.ablation_mode == "c_static" and self._ablation_ready:
            return question.new_zeros((question.shape[0], 192)), {}
        return super()._condition(question, valid, grid, probe=probe)

    def _prefix_shift(self, condition):
        if self.ablation_mode == "c_static" and self._ablation_ready:
            return condition.new_zeros((condition.shape[0], 2560))
        return super()._prefix_shift(condition)

    def save_v1(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        config = {"method": self.method_name, "init_seed": self.init_seed,
                  "ablation_mode": self.ablation_mode, "layers": [5, 11, 17],
                  "prefix_tokens": 20, "visual_tokens": [] if self.ablation_mode == "c_no_visual" else [8, 10],
                  "trainable_parameters": self._audit_parameters(),
                  "initialization_audit": self.initialization_audit}
        (path / CONFIG_NAME).write_text(json.dumps(config, indent=2), encoding="utf-8")
        torch.save({n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}, path / WEIGHTS_NAME)

    def load_v1(self, checkpoint_dir):
        path = Path(checkpoint_dir)
        config = json.loads((path / CONFIG_NAME).read_text(encoding="utf-8"))
        if config["method"] != self.method_name or config["init_seed"] != self.init_seed or config["layers"] != [5, 11, 17]:
            raise ValueError("V1 ablation checkpoint identity mismatch")
        state = torch.load(path / WEIGHTS_NAME, map_location="cpu", weights_only=True)
        parameters = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if set(state) != set(parameters):
            raise ValueError("Ablation checkpoint parameter set mismatch")
        for name, parameter in parameters.items():
            if state[name].shape != parameter.shape:
                raise ValueError(f"Ablation checkpoint shape mismatch: {name}")
            with torch.no_grad():
                parameter.copy_(state[name].to(parameter.device))


class V1StaticModel(V1AblationModel):
    ablation_mode = "c_static"


class V1QMapModel(V1AblationModel):
    ablation_mode = "c_qmap"


class V1NoVisualModel(V1AblationModel):
    ablation_mode = "c_no_visual"


ABLATION_CLASSES = {cls.ablation_mode: cls for cls in (V1StaticModel, V1QMapModel, V1NoVisualModel)}
