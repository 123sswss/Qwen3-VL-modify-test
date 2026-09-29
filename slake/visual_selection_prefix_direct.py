"""V1 ablation: inject the gated native visual summaries without bottleneck projections."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from slake.visual_selection_offset import LAYERS, QUESTION_WIDTH
from slake.visual_selection_prefix import (
    CONFIG_NAME, WEIGHTS_NAME, VisualSelectionPrefixModel,
)


EXPECTED_TRAINABLE_DIRECT = 879_364
EXPECTED_GROUP_COUNTS_DIRECT = {
    "p20": 51_200,
    "visual_s8": 8_192,
    "visual_av10": 10_240,
    "question_context": 345_088,
    "maps": 448_896,
    "layer_condition": 15_747,
    "alpha": 1,
}


class VisualSelectionPrefixDirectModel(VisualSelectionPrefixModel):
    """Use ``alpha * sum(beta_l * LN_l(z_l))`` as the shared P20 shift."""

    method_name = "visual_selection_prefix_direct_summary_v1"

    def __init__(self, base_model: nn.Module, init_seed: int = 44) -> None:
        # Construct the exact V1 graph first. This preserves every retained
        # tensor and the global RNG consumption order. The removed tensors are
        # copied as plain CPU calibration data, then unregistered and deleted.
        super().__init__(base_model, init_seed=init_seed)
        object.__setattr__(self, "_calibration_value_weights", tuple(
            block.weight.detach().float().cpu().clone() for block in self.value_blocks
        ))
        object.__setattr__(
            self, "_calibration_output_weight",
            self.prefix_output.weight.detach().float().cpu().clone(),
        )
        object.__setattr__(
            self, "_calibration_output_bias",
            self.prefix_output.bias.detach().float().cpu().clone(),
        )
        del self.value_blocks
        del self.prefix_output
        device = self.p20.device
        self.alpha = nn.Parameter(torch.ones((), dtype=torch.float32, device=device))
        self.alpha_calibrated = False
        self.alpha_calibration_audit: dict[str, float | str | bool] | None = None
        self._calibration_active = False
        self._calibration_old_condition: torch.Tensor | None = None

        # The parent installed hooks before the removed heads were deleted and
        # before alpha existed. Rebuild them over the final trainable set.
        for handle in self._first_grad_handles:
            handle.remove()
        self._first_grad_handles = []
        self.first_backward_gradients = {}
        self._install_first_backward_hooks()
        self._audit_parameters()
        print(
            "[V1_DIRECT_INIT_AUDIT] retained_initialization=exact_parent_v1 "
            "removed=value_blocks,prefix_output alpha_placeholder=1.0 "
            "alpha_calibration=pending_first_train_batch"
        )

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        # During the parent constructor alpha does not exist yet.
        if not hasattr(self, "alpha"):
            return VisualSelectionPrefixModel.trainable_parameter_groups(self)
        return {
            "p20": [self.p20],
            "visual_s8": [self.visual_s8],
            "visual_av10": [self.visual_av10],
            "question_context": list(self.text_projection.parameters())
            + list(self.question_depthwise.parameters())
            + list(self.question_pointwise.parameters())
            + list(self.question_norm.parameters())
            + list(self.question_pool.parameters()),
            "maps": list(self.query_heads.parameters())
            + list(self.key_norms.parameters())
            + list(self.key_heads.parameters()),
            "layer_condition": list(self.layer_gate.parameters())
            + list(self.value_norms.parameters()),
            "alpha": [self.alpha],
        }

    def _audit_parameters(self) -> dict[str, int]:
        if not hasattr(self, "alpha"):
            return VisualSelectionPrefixModel._audit_parameters(self)
        groups = self.trainable_parameter_groups()
        grouped = [parameter for values in groups.values() for parameter in values]
        active = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if (
            len(grouped) != len({id(parameter) for parameter in grouped})
            or {id(parameter) for parameter in grouped} != {id(parameter) for parameter in active}
        ):
            raise RuntimeError("V1 direct-summary groups are not a unique complete partition")
        counts = {
            name: sum(parameter.numel() for parameter in values)
            for name, values in groups.items()
        }
        if counts != EXPECTED_GROUP_COUNTS_DIRECT or sum(counts.values()) != EXPECTED_TRAINABLE_DIRECT:
            raise RuntimeError(f"V1 direct-summary parameter budget mismatch: {counts}")
        if hasattr(self, "value_blocks") or hasattr(self, "prefix_output"):
            raise RuntimeError("Removed V1 bottleneck/output modules remain registered")
        if any(parameter.requires_grad for parameter in self.base_model.parameters()):
            raise RuntimeError("V1 direct-summary backbone is not frozen")
        print(
            "[V1_DIRECT_PARAMETER_AUDIT] "
            f"{json.dumps(counts, sort_keys=True)} total={sum(counts.values())}"
        )
        return counts

    def _condition(
        self, question: torch.Tensor, valid: torch.Tensor, grid: torch.Tensor,
        probe: dict[str, Any] | None = None,
    ):
        features = getattr(self, "diagnostic_condition_features", None) or self._features
        if set(features) != {*LAYERS, "value"}:
            raise RuntimeError(f"V1 direct-summary expected layers {LAYERS} and Value, got {set(features)}")
        x = self.text_projection(question)
        x = x * valid.unsqueeze(-1)
        conv = self.question_pointwise(F.gelu(self.question_depthwise(x.transpose(1, 2))))
        x = (x + conv.transpose(1, 2)) * valid.unsqueeze(-1)
        pool_logits = self.question_pool(self.question_norm(x)).squeeze(-1)
        pool_logits = pool_logits.masked_fill(~valid, -torch.inf)
        u = (pool_logits.softmax(dim=-1).unsqueeze(-1) * x).sum(dim=1)
        beta = self.layer_gate(u).softmax(dim=-1)
        if probe is not None:
            probe["layer_weights"] = beta.detach().float().cpu().clone()
        grid = grid.to(device=question.device)
        if grid.ndim != 2 or grid.shape[0] != question.shape[0]:
            raise RuntimeError("V1 direct-summary requires exactly one image per sample")
        patches = [int(t * h * w) for t, h, w in grid.tolist()]
        if any(int(h) % 2 or int(w) % 2 for _, h, w in grid.tolist()):
            raise RuntimeError("visual height and width must be divisible by the merger size")
        value_counts = [count // 4 for count in patches]
        if features["value"].shape[0] != sum(value_counts):
            raise RuntimeError("native Value length does not match post-merger geometry")
        values = torch.split(
            features["value"].to(device=question.device, dtype=torch.float32),
            value_counts, dim=0,
        )
        gated_summaries = []
        old_blocks = []
        diagnostics: dict[str, torch.Tensor] = {}
        for index, layer in enumerate(LAYERS):
            h_all = features[layer]
            if h_all.shape[0] != sum(patches):
                raise RuntimeError(f"ViT layer {layer} contains temporary prompt/padding tokens")
            segments = torch.split(
                h_all.to(device=question.device, dtype=torch.float32), patches, dim=0,
            )
            q = self.query_heads[index](u)
            summaries, entropies, top_mass, merged_maps = [], [], [], []
            for batch_index, (hidden, value) in enumerate(zip(segments, values)):
                keys = self.key_heads[index](self.key_norms[index](hidden))
                logits = (keys * q[batch_index]).sum(dim=-1) / math.sqrt(QUESTION_WIDTH)
                patch_prob = logits.softmax(dim=0)
                merged_prob = patch_prob.reshape(-1, 4).sum(dim=1)
                if getattr(self, "diagnostic_uniform_maps", False):
                    merged_prob = torch.full_like(merged_prob, 1.0 / merged_prob.numel())
                if merged_prob.shape[0] != value.shape[0] or not torch.allclose(
                    merged_prob.sum(), merged_prob.new_tensor(1.0), atol=1e-4,
                ):
                    raise RuntimeError("V1 direct-summary lost map probability/image alignment")
                summaries.append((merged_prob[:, None] * value).sum(dim=0))
                if probe is not None:
                    merged_maps.append(merged_prob)
                entropies.append(
                    -(patch_prob * patch_prob.clamp_min(1e-12).log()).sum()
                    / math.log(patch_prob.numel())
                )
                top_mass.append(merged_prob.max())
            z = torch.stack(summaries)
            normalized = self.value_norms[index](z)
            gated_summaries.append(normalized * beta[:, index, None])
            if self._calibration_active:
                weight = self._calibration_value_weights[index].to(
                    device=normalized.device, dtype=normalized.dtype,
                )
                old_blocks.append(F.linear(normalized, weight) * beta[:, index, None])
            if probe is not None:
                probe[f"map{layer}"] = [item.detach().float().cpu().clone() for item in merged_maps]
                probe[f"summary{layer}"] = z.detach().float().cpu().clone()
            diagnostics[f"map{layer}_entropy_norm"] = torch.stack(entropies).mean().detach()
            diagnostics[f"map{layer}_top_mass"] = torch.stack(top_mass).mean().detach()
            diagnostics[f"layer{layer}_weight"] = beta[:, index].mean().detach()
            diagnostics[f"layer{layer}_summary_rms"] = z.square().mean().sqrt().detach()
        if self._calibration_active:
            self._calibration_old_condition = torch.cat(old_blocks, dim=-1)
        summary = torch.stack(gated_summaries, dim=0).sum(dim=0)
        diagnostics["direct_summary_rms"] = summary.square().mean().sqrt().detach()
        return summary, diagnostics

    def _prefix_shift(self, summary: torch.Tensor) -> torch.Tensor:
        if self._calibration_active:
            if self._calibration_old_condition is None:
                raise RuntimeError("Old V1 calibration condition was not captured")
            weight = self._calibration_output_weight.to(summary.device)
            bias = self._calibration_output_bias.to(summary.device)
            old_shift = F.linear(F.relu(self._calibration_old_condition), weight, bias)
            old_rms = old_shift.float().square().mean().sqrt()
            summary_rms = summary.float().square().mean().sqrt()
            if (
                not bool(torch.isfinite(old_rms))
                or not bool(torch.isfinite(summary_rms))
                or float(old_rms) <= 0.0
                or float(summary_rms) <= 0.0
            ):
                raise RuntimeError(
                    f"Invalid alpha calibration RMS: old={float(old_rms)} summary={float(summary_rms)}"
                )
            alpha0 = old_rms / (summary_rms + 1e-8)
            if not bool(torch.isfinite(alpha0)) or float(alpha0) <= 0.0:
                raise RuntimeError(f"Invalid calibrated alpha0={float(alpha0)}")
            self.alpha.data.copy_(alpha0.to(self.alpha.device, self.alpha.dtype))
            calibrated_rms = (self.alpha.float() * summary.float()).square().mean().sqrt()
            self.alpha_calibration_audit = {
                "batch": "fixed_first_train_batch",
                "mode": "eval_no_grad_dropout_disabled",
                "old_shift_rms": float(old_rms),
                "direct_summary_rms": float(summary_rms),
                "alpha0": float(self.alpha.detach()),
                "calibrated_shift_rms": float(calibrated_rms),
                "calibrated_to_old_ratio": float(calibrated_rms / old_rms),
                "rng_restored_before_training_preflight": True,
            }
            self.alpha_calibrated = True
            self._calibration_old_condition = None
            for name in (
                "_calibration_value_weights", "_calibration_output_weight",
                "_calibration_output_bias",
            ):
                object.__delattr__(self, name)
            print("[V1_DIRECT_ALPHA_CALIBRATION] " + json.dumps(self.alpha_calibration_audit))
        if not self.alpha_calibrated:
            raise RuntimeError("V1 direct-summary alpha must be calibrated before use")
        shift = self.alpha.float() * summary.float()
        self._prefix_shift_debug = {
            "alpha": self.alpha.detach(),
            "direct_summary_rms": summary.square().mean().sqrt().detach(),
        }
        return shift

    def calibrate_alpha_from_first_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.alpha_calibrated:
            raise RuntimeError("V1 direct-summary alpha calibration may run only once")
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        was_training = self.training
        self.eval()
        self._calibration_active = True
        try:
            with torch.no_grad():
                self(**batch)
        finally:
            self._calibration_active = False
            self.train(was_training)
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
        if not self.alpha_calibrated or self.alpha_calibration_audit is None:
            raise RuntimeError("V1 direct-summary alpha calibration did not complete")
        return dict(self.alpha_calibration_audit)

    def save_v1(self, output_dir: str | Path) -> None:
        if not self.alpha_calibrated or self.alpha_calibration_audit is None:
            raise RuntimeError("Cannot save an uncalibrated direct-summary checkpoint")
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "method": self.method_name,
            "init_seed": self.init_seed,
            "layers": list(LAYERS),
            "prefix_tokens": 20,
            "visual_tokens": [8, 10],
            "shift": "alpha_times_sum_beta_layernorm_native_summary",
            "alpha_calibration": self.alpha_calibration_audit,
            "trainable_parameters": self._audit_parameters(),
        }
        with (path / CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        torch.save(
            {name: parameter.detach().cpu() for name, parameter in self.named_parameters()
             if parameter.requires_grad},
            path / WEIGHTS_NAME,
        )

    def load_v1(self, checkpoint_dir: str | Path) -> None:
        path = Path(checkpoint_dir)
        with (path / CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("method") != self.method_name or tuple(config.get("layers", ())) != LAYERS:
            raise ValueError("V1 direct-summary checkpoint architecture mismatch")
        if int(config["init_seed"]) != self.init_seed:
            raise ValueError("V1 direct-summary checkpoint seed mismatch")
        state = torch.load(path / WEIGHTS_NAME, map_location="cpu", weights_only=True)
        parameters = {name: parameter for name, parameter in self.named_parameters()
                      if parameter.requires_grad}
        if set(state) != set(parameters):
            raise ValueError("V1 direct-summary checkpoint trainable tensor set mismatch")
        for name, parameter in parameters.items():
            if tuple(state[name].shape) != tuple(parameter.shape):
                raise ValueError(f"V1 direct-summary tensor shape mismatch: {name}")
            parameter.data.copy_(state[name].to(parameter.device))
        audit = config.get("alpha_calibration")
        if not isinstance(audit, dict):
            raise ValueError("V1 direct-summary checkpoint lacks alpha calibration audit")
        self.alpha_calibration_audit = audit
        self.alpha_calibrated = True
        self._calibration_old_condition = None
        for name in (
            "_calibration_value_weights", "_calibration_output_weight",
            "_calibration_output_bias",
        ):
            if hasattr(self, name):
                object.__delattr__(self, name)
        self._audit_parameters()
