"""V2: position-specific three-layer mixtures on V1's existing P20."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from slake.visual_selection_offset import BLOCK_WIDTH, LAYERS
from slake.visual_selection_prefix import (
    EXPECTED_GROUP_COUNTS as V1_GROUP_COUNTS,
    VisualSelectionPrefixModel,
)


CONFIG_NAME = "visual_selection_layer_mix_config.json"
WEIGHTS_NAME = "visual_selection_layer_mix.pt"
EXPECTED_GROUP_COUNTS = {**V1_GROUP_COUNTS, "alpha": 60}
EXPECTED_TRAINABLE = 1_865_023


class VisualSelectionLayerMixModel(VisualSelectionPrefixModel):
    """Keep every V1 component, adding only a free FP32 alpha[20,3]."""

    def __init__(self, base_model: nn.Module, init_seed: int = 44) -> None:
        # V1 audits itself during construction. The V2 tensor is registered
        # only afterward, so common initialization and RNG consumption match V1.
        super().__init__(base_model, init_seed=init_seed)
        self.alpha = nn.Parameter(torch.ones(20, len(LAYERS), device=self.p20.device, dtype=torch.float32))
        self.alpha.register_hook(self._capture_first_alpha_gradient)
        self._v2_ready = True
        self._audit_parameters()
        print("[V2_ALPHA_INIT] shape=[20,3] dtype=float32 all_one=True random_draws=0")

    def _capture_first_alpha_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        if "alpha" not in self.first_backward_gradients:
            self.first_backward_gradients["alpha"] = float(gradient.detach().float().norm())
        return gradient

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = super().trainable_parameter_groups()
        if getattr(self, "_v2_ready", False):
            groups["alpha"] = [self.alpha]
        return groups

    def _audit_parameters(self) -> dict[str, int]:
        if not getattr(self, "_v2_ready", False):
            return super()._audit_parameters()
        groups = self.trainable_parameter_groups()
        grouped = [parameter for values in groups.values() for parameter in values]
        active = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if len(grouped) != len({id(parameter) for parameter in grouped}):
            raise RuntimeError("V2 optimizer groups contain duplicate parameters")
        if {id(parameter) for parameter in grouped} != {id(parameter) for parameter in active}:
            raise RuntimeError("V2 optimizer groups do not cover exactly the trainable parameters")
        counts = {name: sum(parameter.numel() for parameter in values) for name, values in groups.items()}
        if counts != EXPECTED_GROUP_COUNTS or sum(counts.values()) != EXPECTED_TRAINABLE:
            raise RuntimeError(f"V2 parameter budget mismatch: {counts}")
        if self.alpha.dtype != torch.float32 or tuple(self.alpha.shape) != (20, 3):
            raise RuntimeError("V2 alpha must be FP32 with shape [20,3]")
        if any(parameter.requires_grad for parameter in self.base_model.parameters()):
            raise RuntimeError("V2 backbone must remain frozen")
        print(f"[V2_PARAMETER_AUDIT] {json.dumps(counts, sort_keys=True)} total={sum(counts.values())}")
        return counts

    def layer_components(self, condition: torch.Tensor) -> torch.Tensor:
        """Return b_5,b_11,b_17 using slices of the one existing output head."""
        if condition.ndim != 2 or condition.shape[-1] != len(LAYERS) * BLOCK_WIDTH:
            raise ValueError("V2 condition must contain three ordered 64-wide blocks")
        activated = torch.relu(condition)
        pieces = [
            F.linear(
                activated[:, index * BLOCK_WIDTH:(index + 1) * BLOCK_WIDTH],
                self.prefix_output.weight[:, index * BLOCK_WIDTH:(index + 1) * BLOCK_WIDTH],
                bias=None,
            )
            for index in range(len(LAYERS))
        ]
        return torch.stack(pieces, dim=1)

    def _prefix_shift(self, condition: torch.Tensor) -> torch.Tensor:
        components = self.layer_components(condition)  # batch x 3 x 2560
        # Algebraically sum_l alpha[i,l]*b_l + a. Computing the all-one
        # reference with the original fused Linear preserves V1's numerical
        # behavior at initialization while retaining exact alpha gradients.
        v1_shift = self.prefix_output(torch.relu(condition))
        correction = torch.einsum("il,bld->bid", self.alpha - 1.0, components)
        shift = v1_shift[:, None, :] + correction
        self._prefix_shift_debug = {
            f"layer{layer}_component_rms": components[:, index].square().mean().sqrt().detach()
            for index, layer in enumerate(LAYERS)
        }
        self._prefix_shift_debug["alpha_abs_max"] = self.alpha.detach().abs().max()
        return shift

    def save_v2(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "method": "visual_selection_layer_mix_prefix_p20_v2",
            "init_seed": self.init_seed,
            "layers": list(LAYERS),
            "prefix_tokens": 20,
            "alpha_shape": [20, 3],
            "alpha_init": 1.0,
            "parameter_groups": self._audit_parameters(),
        }
        with (path / CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        torch.save(
            {name: parameter.detach().cpu() for name, parameter in self.named_parameters()
             if parameter.requires_grad},
            path / WEIGHTS_NAME,
        )

    def load_v2(self, checkpoint_dir: str | Path) -> None:
        path = Path(checkpoint_dir)
        with (path / CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if (config.get("method") != "visual_selection_layer_mix_prefix_p20_v2"
            or tuple(config.get("layers", ())) != LAYERS
            or config.get("alpha_shape") != [20, 3]
            or int(config["init_seed"]) != self.init_seed):
            raise ValueError("V2 checkpoint architecture or seed mismatch")
        state = torch.load(path / WEIGHTS_NAME, map_location="cpu", weights_only=True)
        parameters = {name: parameter for name, parameter in self.named_parameters()
                      if parameter.requires_grad}
        if set(state) != set(parameters):
            raise ValueError("V2 checkpoint trainable tensor set mismatch")
        for name, parameter in parameters.items():
            if tuple(state[name].shape) != tuple(parameter.shape):
                raise ValueError(f"V2 checkpoint tensor shape mismatch: {name}")
            parameter.data.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
        self._audit_parameters()
