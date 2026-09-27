"""V1 experiment: replace split S8+Av10 with one static Visual20 table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from slake.visual_selection_offset import LAYERS, OUTPUT_INIT_STD
from slake.visual_selection_prefix import (
    BLOCK_WIDTH,
    CONFIG_NAME,
    QUESTION_WIDTH,
    WEIGHTS_NAME,
    VisualSelectionPrefixModel,
)


EXPECTED_TRAINABLE_VISUAL20 = 1_867_011
EXPECTED_GROUP_COUNTS_VISUAL20 = {
    "p20": 51_200,
    "visual_prompt20": 20_480,
    "question_context": 345_088,
    "maps": 448_896,
    "layer_condition": 507_267,
    "prefix_output": 494_080,
}


class VisualSelectionPrefixVisual20Model(VisualSelectionPrefixModel):
    """Normalized V1 with one layer-17 static Visual20 parameter group."""

    method_name = "visual_selection_prefix_p20_v1_visual20_lr1e4"

    def __init__(self, base_model: nn.Module, init_seed: int = 44) -> None:
        object.__setattr__(self, "_visual20_ready", False)
        super().__init__(base_model, init_seed=init_seed)

        cpu_before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        shared_before = {
            name: parameter.detach().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and name not in {"visual_s8", "visual_av10"}
        }
        legacy = torch.cat(
            (self.visual_s8.detach().clone(), self.visual_av10.detach().clone()), dim=0,
        )
        del self.visual_s8
        del self.visual_av10

        # The two extra rows must not perturb any global initialization stream.
        extra_generator = torch.Generator(device="cpu").manual_seed(
            (self.init_seed + 20_260_927) % (2**63 - 1)
        )
        extra = torch.empty(2, 1024, dtype=torch.float32, device="cpu")
        extra.normal_(mean=0.0, std=0.02, generator=extra_generator)
        self.visual_prompt20 = nn.Parameter(
            torch.cat((legacy, extra.to(device=legacy.device, dtype=legacy.dtype)), dim=0)
        )
        if not torch.equal(cpu_before, torch.random.get_rng_state()):
            raise RuntimeError("Visual20 extra-row initialization changed global CPU RNG")
        if cuda_before:
            cuda_after = torch.cuda.get_rng_state_all()
            if len(cuda_before) != len(cuda_after) or any(
                not torch.equal(before, after) for before, after in zip(cuda_before, cuda_after)
            ):
                raise RuntimeError("Visual20 extra-row initialization changed global CUDA RNG")
        shared_after = {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and name != "visual_prompt20"
        }
        if set(shared_before) != set(shared_after):
            raise RuntimeError("V1 Visual20 conversion changed the shared parameter set")
        shared_mismatches = [
            name for name, before in shared_before.items()
            if not torch.equal(before, shared_after[name].detach())
        ]
        if shared_mismatches:
            raise RuntimeError(
                f"V1 Visual20 conversion changed shared initial values: {shared_mismatches}"
            )

        def capture_visual20(gradient: torch.Tensor) -> torch.Tensor:
            if "visual_prompt20" not in self.first_backward_gradients:
                self.first_backward_gradients["visual_prompt20"] = float(
                    gradient.detach().float().norm()
                )
            return gradient

        self._first_grad_handles.append(self.visual_prompt20.register_hook(capture_visual20))
        self._visual20_ready = True
        self._audit_parameters()
        print(
            "[V1_VISUAL20_INIT_AUDIT] layer=17 tokens=20 distribution=Normal(0,0.02) "
            "first18_source=exact_S8_then_Av10 extra2_rng=private_cpu "
            f"shared_tensors_unchanged={len(shared_before)} "
            f"first18_std={float(self.visual_prompt20[:18].detach().std()):.8f} "
            f"extra2_std={float(self.visual_prompt20[18:].detach().std()):.8f} "
            "global_rng_unchanged=True"
        )

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        if not getattr(self, "_visual20_ready", False):
            return super().trainable_parameter_groups()
        groups = super().trainable_parameter_groups()
        groups.pop("visual_s8", None)
        groups.pop("visual_av10", None)
        groups["visual_prompt20"] = [self.visual_prompt20]
        return groups

    def _audit_parameters(self) -> dict[str, int]:
        if not getattr(self, "_visual20_ready", False):
            return super()._audit_parameters()
        groups = self.trainable_parameter_groups()
        grouped = [parameter for values in groups.values() for parameter in values]
        active = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if (
            len(grouped) != len({id(parameter) for parameter in grouped})
            or {id(parameter) for parameter in grouped} != {id(parameter) for parameter in active}
        ):
            raise RuntimeError("V1 Visual20 groups are not a unique complete partition")
        counts = {
            name: sum(parameter.numel() for parameter in values)
            for name, values in groups.items()
        }
        if (
            counts != EXPECTED_GROUP_COUNTS_VISUAL20
            or sum(counts.values()) != EXPECTED_TRAINABLE_VISUAL20
        ):
            raise RuntimeError(f"V1 Visual20 parameter budget mismatch: {counts}")
        if any(parameter.requires_grad for parameter in self.base_model.parameters()):
            raise RuntimeError("V1 Visual20 backbone is not frozen")
        if hasattr(self, "visual_s8") or hasattr(self, "visual_av10"):
            raise RuntimeError("split Visual18 parameters leaked into V1 Visual20")
        print(
            f"[V1_VISUAL20_PARAMETER_AUDIT] {json.dumps(counts, sort_keys=True)} "
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
            "visual_tokens": [20],
            "visual_prompt_init_distribution": "normal",
            "visual_prompt_init_std": 0.02,
            "visual_prompt_learning_rate": 1e-4,
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
            or config.get("visual_tokens") != [20]
        ):
            raise ValueError("V1 Visual20 checkpoint architecture mismatch")
        if int(config["init_seed"]) != self.init_seed:
            raise ValueError("V1 Visual20 checkpoint seed mismatch")
        state = torch.load(path / WEIGHTS_NAME, map_location="cpu", weights_only=True)
        parameters = {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if set(state) != set(parameters):
            raise ValueError("V1 Visual20 checkpoint trainable tensor set mismatch")
        for name, parameter in parameters.items():
            if tuple(state[name].shape) != tuple(parameter.shape):
                raise ValueError(f"V1 Visual20 checkpoint tensor shape mismatch: {name}")
            parameter.data.copy_(state[name].to(parameter.device))
