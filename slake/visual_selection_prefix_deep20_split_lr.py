"""Final exploration: per-layer 10-low + 10-high transient visual prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from slake.visual_selection_offset import LAYERS, OUTPUT_INIT_STD, _Visual18Block
from slake.visual_selection_prefix import (
    BLOCK_WIDTH,
    CONFIG_NAME,
    QUESTION_WIDTH,
    WEIGHTS_NAME,
    VisualSelectionPrefixModel,
)
from slake.visual_selection_prefix_deep5 import DEEP_VISUAL_LAYERS


DEEP_VISUAL_TOKENS_PER_GROUP = 10
EXPECTED_TRAINABLE_DEEP20_SPLIT_LR = 2_010_371
EXPECTED_GROUP_COUNTS_DEEP20_SPLIT_LR = {
    "p20": 51_200,
    "visual_deep_low_prompts": 81_920,
    "visual_deep_high_prompts": 81_920,
    "question_context": 345_088,
    "maps": 448_896,
    "layer_condition": 507_267,
    "prefix_output": 494_080,
}


class VisualSelectionPrefixDeep20SplitLRModel(VisualSelectionPrefixModel):
    """Normalized V1 with independent transient 10+10 prompts at layers 16..23."""

    method_name = "visual_selection_prefix_p20_v1_deep20_split_lr_l16_23"

    def __init__(self, base_model: nn.Module, init_seed: int = 44) -> None:
        object.__setattr__(self, "_deep20_split_lr_ready", False)
        visual = base_model.model.visual
        if len(visual.blocks) != 24:
            raise ValueError(
                "V1 Deep20 split-LR requires exactly 24 visual blocks; "
                f"found {len(visual.blocks)} and will not adjust indexes 16..23"
            )
        super().__init__(base_model, init_seed=init_seed)

        visual = self.base_model.model.visual
        cpu_before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        shared_before = {
            name: parameter.detach().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and name not in {"visual_s8", "visual_av10"}
        }

        layer17 = visual.blocks[17]
        if not isinstance(layer17, _Visual18Block):
            raise RuntimeError(
                "V1 Deep20 split-LR expected the audited layer-17 Visual18 wrapper"
            )
        visual.blocks[17] = layer17.block
        del self.visual_s8
        del self.visual_av10

        generator = torch.Generator(device="cpu").manual_seed(
            (self.init_seed + 20_260_929) % (2**63 - 1)
        )

        def make_tables() -> nn.ParameterList:
            tables = []
            for _layer in DEEP_VISUAL_LAYERS:
                table = torch.empty(
                    DEEP_VISUAL_TOKENS_PER_GROUP, 1024, dtype=torch.float32,
                )
                table.normal_(mean=0.0, std=0.02, generator=generator)
                tables.append(nn.Parameter(table.to(device=self.p20.device)))
            return nn.ParameterList(tables)

        # One private stream, deterministic low-then-high construction. These
        # remain distinct Parameters and therefore distinct optimizer groups.
        self.visual_deep_low_prompts = make_tables()
        self.visual_deep_high_prompts = make_tables()

        for slot, layer in enumerate(DEEP_VISUAL_LAYERS):
            visual.blocks[layer] = _Visual18Block(
                visual.blocks[layer], self, prompt_slot=slot,
            )

        def capture_layer17(_module, _args, output):
            if self._active:
                if not torch.is_tensor(output):
                    raise TypeError("ViT layer 17 did not return a tensor")
                self._features[17] = output

        visual.blocks[17].register_forward_hook(capture_layer17)

        if not torch.equal(cpu_before, torch.random.get_rng_state()):
            raise RuntimeError("Deep20 split-LR initialization changed global CPU RNG")
        if cuda_before:
            cuda_after = torch.cuda.get_rng_state_all()
            if len(cuda_before) != len(cuda_after) or any(
                not torch.equal(before, after)
                for before, after in zip(cuda_before, cuda_after)
            ):
                raise RuntimeError("Deep20 split-LR initialization changed global CUDA RNG")
        shared_after = {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and not name.startswith("visual_deep_low_prompts.")
            and not name.startswith("visual_deep_high_prompts.")
        }
        if set(shared_before) != set(shared_after):
            raise RuntimeError("V1 Deep20 split-LR changed the shared parameter set")
        mismatches = [
            name for name, before in shared_before.items()
            if not torch.equal(before, shared_after[name].detach())
        ]
        if mismatches:
            raise RuntimeError(
                f"V1 Deep20 split-LR changed shared initial values: {mismatches}"
            )

        for group_name, parameters in (
            ("visual_deep_low_prompts", self.visual_deep_low_prompts),
            ("visual_deep_high_prompts", self.visual_deep_high_prompts),
        ):
            for slot, parameter in enumerate(parameters):
                def capture(gradient: torch.Tensor, key=f"{group_name}.{slot}") -> torch.Tensor:
                    if key not in self.first_backward_gradients:
                        self.first_backward_gradients[key] = float(
                            gradient.detach().float().norm()
                        )
                    return gradient
                self._first_grad_handles.append(parameter.register_hook(capture))

        self._deep20_split_lr_ready = True
        self._audit_parameters()
        print(
            "[V1_DEEP20_SPLIT_LR_INIT_AUDIT] backbone_blocks=24 "
            "layers=16,17,18,19,20,21,22,23 tokens_per_layer=10_low_then_10_high "
            "transient_insert_remove=True shared=False distribution=Normal(0,0.02) "
            "rng=private_cpu optimizer_groups=distinct "
            f"shared_tensors_unchanged={len(shared_before)} global_rng_unchanged=True "
            f"low_stds={[float(p.detach().std()) for p in self.visual_deep_low_prompts]} "
            f"high_stds={[float(p.detach().std()) for p in self.visual_deep_high_prompts]}"
        )

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        if not getattr(self, "_deep20_split_lr_ready", False):
            return super().trainable_parameter_groups()
        return {
            "p20": [self.p20],
            "visual_deep_low_prompts": list(self.visual_deep_low_prompts),
            "visual_deep_high_prompts": list(self.visual_deep_high_prompts),
            "question_context": list(self.text_projection.parameters())
            + list(self.question_depthwise.parameters())
            + list(self.question_pointwise.parameters())
            + list(self.question_norm.parameters())
            + list(self.question_pool.parameters()),
            "maps": list(self.query_heads.parameters())
            + list(self.key_norms.parameters())
            + list(self.key_heads.parameters()),
            "layer_condition": list(self.layer_gate.parameters())
            + list(self.value_norms.parameters())
            + list(self.value_blocks.parameters()),
            "prefix_output": list(self.prefix_output.parameters()),
        }

    def _audit_parameters(self) -> dict[str, int]:
        if not getattr(self, "_deep20_split_lr_ready", False):
            return super()._audit_parameters()
        visual = self.base_model.model.visual
        if len(visual.blocks) != 24:
            raise RuntimeError("V1 Deep20 split-LR backbone depth changed")
        if any(
            not isinstance(visual.blocks[layer], _Visual18Block)
            or visual.blocks[layer].prompt_slot != slot
            for slot, layer in enumerate(DEEP_VISUAL_LAYERS)
        ):
            raise RuntimeError("V1 Deep20 split-LR wrappers do not match layers 16..23")
        groups = self.trainable_parameter_groups()
        grouped = [parameter for values in groups.values() for parameter in values]
        active = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if (
            len(grouped) != len({id(parameter) for parameter in grouped})
            or {id(parameter) for parameter in grouped} != {id(parameter) for parameter in active}
        ):
            raise RuntimeError("V1 Deep20 split-LR groups are not a unique complete partition")
        counts = {
            name: sum(parameter.numel() for parameter in values)
            for name, values in groups.items()
        }
        if (
            counts != EXPECTED_GROUP_COUNTS_DEEP20_SPLIT_LR
            or sum(counts.values()) != EXPECTED_TRAINABLE_DEEP20_SPLIT_LR
        ):
            raise RuntimeError(f"V1 Deep20 split-LR parameter budget mismatch: {counts}")
        if any(parameter.requires_grad for parameter in self.base_model.parameters()):
            raise RuntimeError("V1 Deep20 split-LR backbone is not frozen")
        if any(
            hasattr(self, name)
            for name in ("visual_s8", "visual_av10", "visual_prompt20", "visual_deep_prompts")
        ):
            raise RuntimeError("another visual Prompt layout leaked into V1 Deep20 split-LR")
        print(
            f"[V1_DEEP20_SPLIT_LR_PARAMETER_AUDIT] {json.dumps(counts, sort_keys=True)} "
            f"total={sum(counts.values())}"
        )
        return counts

    def save_v1(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        config: dict[str, Any] = {
            "method": self.method_name,
            "init_seed": self.init_seed,
            "layers": list(LAYERS),
            "question_width": QUESTION_WIDTH,
            "block_width": BLOCK_WIDTH,
            "condition_width": 192,
            "visual_prompt_layers": list(DEEP_VISUAL_LAYERS),
            "visual_tokens_per_group_per_layer": DEEP_VISUAL_TOKENS_PER_GROUP,
            "visual_prompt_order": "low10_then_high10",
            "visual_prompt_init_distribution": "normal",
            "visual_prompt_init_std": 0.02,
            "visual_prompt_learning_rates": {"low10": 3e-5, "high10": 1e-4},
            "visual_prompt_lifetime": "single_block_insert_then_remove",
            "prefix_tokens": 20,
            "output_init_distribution": "normal",
            "output_init_std": OUTPUT_INIT_STD,
            "trainable_parameters": self._audit_parameters(),
        }
        with (path / CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        torch.save(
            {
                name: parameter.detach().cpu()
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
            },
            path / WEIGHTS_NAME,
        )

    def load_v1(self, checkpoint_dir: str | Path) -> None:
        path = Path(checkpoint_dir)
        with (path / CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if (
            config.get("method") != self.method_name
            or tuple(config.get("layers", ())) != LAYERS
            or tuple(config.get("visual_prompt_layers", ())) != DEEP_VISUAL_LAYERS
            or int(config.get("visual_tokens_per_group_per_layer", -1))
            != DEEP_VISUAL_TOKENS_PER_GROUP
            or config.get("visual_prompt_order") != "low10_then_high10"
        ):
            raise ValueError("V1 Deep20 split-LR checkpoint architecture mismatch")
        if int(config["init_seed"]) != self.init_seed:
            raise ValueError("V1 Deep20 split-LR checkpoint seed mismatch")
        state = torch.load(path / WEIGHTS_NAME, map_location="cpu", weights_only=True)
        parameters = {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if set(state) != set(parameters):
            raise ValueError("V1 Deep20 split-LR checkpoint tensor set mismatch")
        for name, parameter in parameters.items():
            if tuple(state[name].shape) != tuple(parameter.shape):
                raise ValueError(f"V1 Deep20 split-LR tensor shape mismatch: {name}")
            parameter.data.copy_(state[name].to(parameter.device))
