"""Sample-conditioned soft prompts for frozen Qwen3-VL models."""

from __future__ import annotations

import json
import math
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, Sequence

import torch
from torch import nn

from slake.directional_concat_workspace import (
    DirectionalConcatWorkspaceVisual,
    ZeroInitWorkspaceProjection,
)
from slake.sparse_visual_mmrl import LatentResidualAttention, SparseVisualMMRL

DYNAMIC_PROMPT_CONFIG_NAME = "dynamic_prompt_config.json"
DYNAMIC_PROMPT_WEIGHTS_NAME = "dynamic_prompt.pt"
DYNAMIC_PROMPT_INTERVENTIONS = (
    "normal",
    "zero",
    "mean-residual",
    "lagged-memory",
)
SHARED_S_TEXT_MODES = (
    "none",
    "separate_residual",
    "direct_prompt",
    "text_owned_visual_readonly",
)


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    name: str,
) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        indices = (counts == 0).nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(
            f"Dynamic Prompt {name} memory is empty for samples={indices}"
        )
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / counts.to(values.dtype).unsqueeze(-1)


class DynamicPromptCrossAttention(nn.Module):
    """Read multimodal memory tokens into prompt slots."""

    def __init__(
        self,
        hidden_size: int,
        attention_dim: int = 256,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or attention_dim < 1 or num_heads < 1:
            raise ValueError("Dynamic Prompt dimensions must be positive")
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")

        self.hidden_size = int(hidden_size)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.memory_norm = nn.LayerNorm(self.hidden_size)
        self.query_projection = nn.Linear(
            self.hidden_size, self.attention_dim, bias=False
        )
        self.key_projection = nn.Linear(
            self.hidden_size, self.attention_dim, bias=False
        )
        self.value_projection = nn.Linear(
            self.hidden_size, self.attention_dim, bias=False
        )
        self.output_projection = nn.Linear(
            self.attention_dim, self.hidden_size, bias=True
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.debug_context: Dict[str, torch.Tensor] = {}
        self._forward_audited = False

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        prompt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        if prompt.ndim != 3 or prompt.shape[-1] != self.hidden_size:
            raise ValueError(
                f"prompt must have [B, P, {self.hidden_size}], got {tuple(prompt.shape)}"
            )
        if (
            memory.ndim != 3
            or memory.shape[1] not in (2, 3)
            or memory.shape[-1] != self.hidden_size
        ):
            raise ValueError(
                f"memory must have [B, 2|3, {self.hidden_size}], got "
                f"{tuple(memory.shape)}"
            )

        parameter_dtype = self.query_projection.weight.dtype
        prompt_f = prompt.to(dtype=parameter_dtype)
        memory_f = memory.to(dtype=parameter_dtype)
        query = self._split_heads(self.query_projection(self.query_norm(prompt_f)))
        key = self._split_heads(self.key_projection(self.memory_norm(memory_f)))
        value = self._split_heads(self.value_projection(self.memory_norm(memory_f)))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=value.dtype)
        context = torch.matmul(attention, value)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(prompt.shape[0], prompt.shape[1], self.attention_dim)
        )
        delta = self.output_projection(context).to(dtype=prompt.dtype)

        with torch.no_grad():
            attention_f = attention.detach().float().clamp_min(1e-8)
            memory_count = int(memory.shape[1])
            entropy = -(attention_f * attention_f.log()).sum(dim=-1) / math.log(
                float(memory_count)
            )
            prompt_norm = prompt.detach().float().norm(dim=-1).mean()
            delta_norm = delta.detach().float().norm(dim=-1).mean()
            self.debug_context = {
                "dynamic_prompt_attention_entropy_norm": entropy.mean(),
                "dynamic_prompt_visual_attention_mean": attention_f[..., 0].mean(),
                "dynamic_prompt_text_attention_mean": attention_f[..., 1].mean(),
                "dynamic_prompt_delta_norm_mean": delta_norm,
                "dynamic_prompt_base_norm_mean": prompt_norm,
                "dynamic_prompt_delta_to_base_ratio": delta_norm
                / prompt_norm.clamp_min(1e-8),
            }
            if memory_count == 3:
                self.debug_context["dynamic_prompt_shared_s_attention_mean"] = (
                    attention_f[..., 2].mean()
                )
            if not self._forward_audited:
                max_abs = float(delta.detach().float().abs().max().item())
                if max_abs != 0.0:
                    raise RuntimeError(
                        "Dynamic Prompt output projection must start at exact zero; "
                        f"max_abs_delta={max_abs}"
                    )
                print(
                    "[DYNAMIC_PROMPT_ZERO_INIT_AUDIT] "
                    f"prompt_shape={tuple(prompt.shape)} memory_shape={tuple(memory.shape)} "
                    f"attention_dim={self.attention_dim} heads={self.num_heads} "
                    "max_abs_delta=0.0 pass=True"
                )
                self._forward_audited = True
        return delta


class DynamicPromptTuningModel(nn.Module):
    """Jointly train a static prompt and its multimodal residual generator."""

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,
        prompt_length: int = 20,
        init_seed: int = 44,
        attention_dim: int = 256,
        num_heads: int = 8,
        sparse_visual_anchor_layers: Sequence[int] | None = None,
        sparse_visual_rep_tokens: int = 8,
        sparse_visual_attention_dim: int = 128,
        sparse_visual_heads: int = 4,
        shared_s_text_mode: str = "none",
        shared_s_attention_dim: int = 128,
        shared_s_heads: int = 4,
        shared_s_visual_bottleneck_dim: int = 128,
        shared_workspace: bool = False,
        workspace_tokens: int = 32,
        workspace_dim: int = 1024,
        workspace_heads: int = 16,
        workspace_ffn_dim: int = 4096,
        workspace_text_attention_dim: int = 1024,
        workspace_text_heads: int = 16,
        workspace_visual_attention_dim: int = 1024,
        workspace_visual_heads: int = 16,
        directional_concat_workspace: bool = False,
        directional_visual_dynamic_write: bool = True,
    ) -> None:
        super().__init__()
        if prompt_length < 1:
            raise ValueError("prompt_length must be positive")
        if shared_s_text_mode not in SHARED_S_TEXT_MODES:
            raise ValueError(
                f"Unsupported shared_s_text_mode={shared_s_text_mode!r}; "
                f"choices={SHARED_S_TEXT_MODES}"
            )
        if shared_s_attention_dim < 1 or shared_s_heads < 1:
            raise ValueError("Shared-S attention dimensions must be positive")
        if shared_s_attention_dim % shared_s_heads != 0:
            raise ValueError("shared_s_attention_dim must be divisible by shared_s_heads")
        if shared_workspace and shared_s_text_mode != "none":
            raise ValueError(
                "The full workspace requires independent private P and visual S"
            )
        if shared_workspace and directional_concat_workspace:
            raise ValueError(
                "Shared Workspace and Directional Concat Workspace are exclusive"
            )
        if directional_concat_workspace and shared_s_text_mode != "none":
            raise ValueError(
                "Directional Concat Workspace does not use Shared-S text modes"
            )
        if directional_concat_workspace and (
            not sparse_visual_anchor_layers
            or len(tuple(sparse_visual_anchor_layers)) != 1
        ):
            raise ValueError(
                "Directional Concat Workspace requires exactly one visual anchor"
            )
        if workspace_text_attention_dim % workspace_text_heads != 0:
            raise ValueError(
                "workspace_text_attention_dim must be divisible by heads"
            )
        self.base_model = base_model
        self.requested_prompt_length = int(prompt_length)
        self.init_seed = int(init_seed)
        self.shared_s_text_mode = shared_s_text_mode
        self.shared_workspace_enabled = bool(shared_workspace)
        self.directional_concat_workspace_enabled = bool(
            directional_concat_workspace
        )
        self.directional_visual_dynamic_write = bool(
            directional_visual_dynamic_write
        )
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

        embeddings = self.base_model.get_input_embeddings().weight.detach()
        self.hidden_size = int(embeddings.shape[-1])
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.sparse_visual = None
        if sparse_visual_anchor_layers:
            model_core = getattr(self.base_model, "model", None)
            visual_model = getattr(model_core, "visual", None)
            visual_config = getattr(visual_model, "config", None)
            visual_dim = getattr(visual_config, "hidden_size", None)
            if visual_model is None or visual_dim is None:
                raise RuntimeError(
                    "Sparse Visual MMRL requires base_model.model.visual.config.hidden_size"
                )
            visual_device = next(visual_model.parameters()).device
            if self.directional_concat_workspace_enabled:
                self.sparse_visual = DirectionalConcatWorkspaceVisual(
                    visual_dim=int(visual_dim),
                    text_dim=self.hidden_size,
                    anchor_layer=int(tuple(sparse_visual_anchor_layers)[0]),
                    private_prompt_tokens=sparse_visual_rep_tokens,
                    workspace_tokens=workspace_tokens,
                    workspace_dim=workspace_dim,
                    workspace_heads=workspace_heads,
                    visual_dynamic_write=self.directional_visual_dynamic_write,
                ).to(device=visual_device)
            else:
                self.sparse_visual = SparseVisualMMRL(
                    visual_dim=int(visual_dim),
                    text_dim=self.hidden_size,
                    anchor_layers=sparse_visual_anchor_layers,
                    rep_token_count=sparse_visual_rep_tokens,
                    attention_dim=sparse_visual_attention_dim,
                    num_heads=sparse_visual_heads,
                    map_shared_s_to_text=self.shared_s_text_mode
                    in ("separate_residual", "direct_prompt"),
                    text_anchor_tokens=(
                        self.requested_prompt_length
                        if self.shared_s_text_mode == "text_owned_visual_readonly"
                        else None
                    ),
                    text_anchor_bottleneck_dim=shared_s_visual_bottleneck_dim,
                    shared_workspace=self.shared_workspace_enabled,
                    workspace_tokens=workspace_tokens,
                    workspace_dim=workspace_dim,
                    workspace_heads=workspace_heads,
                    workspace_ffn_dim=workspace_ffn_dim,
                    workspace_visual_attention_dim=workspace_visual_attention_dim,
                    workspace_visual_heads=workspace_visual_heads,
                ).to(device=visual_device)
            self.sparse_visual.install(visual_model)
        elif self.shared_s_text_mode != "none":
            raise ValueError("Shared-S text modes require Sparse Visual MMRL")

        self.private_prompt_length = self.requested_prompt_length
        self.workspace_prompt_length = (
            int(workspace_tokens)
            if self.directional_concat_workspace_enabled
            else 0
        )
        if self.shared_s_text_mode == "direct_prompt":
            self.prompt_length = self.sparse_visual.shared_s_text_prompt_length
            self.register_parameter("soft_prompt", None)
            self.register_parameter("workspace_text_anchor", None)
            self.workspace_text_projection = None
        else:
            self.prompt_length = (
                self.private_prompt_length + self.workspace_prompt_length
            )
            generator = torch.Generator(device="cpu").manual_seed(self.init_seed)
            sampled_rows = torch.randint(
                embeddings.shape[0],
                (self.prompt_length,),
                generator=generator,
                device="cpu",
            )
            initial_prompt = embeddings[sampled_rows.to(embeddings.device)].clone()
            self.soft_prompt = nn.Parameter(
                initial_prompt[: self.private_prompt_length].float()
            )
            if self.directional_concat_workspace_enabled:
                self.workspace_text_anchor = nn.Parameter(
                    initial_prompt[self.private_prompt_length :].float()
                )
                self.workspace_text_projection = ZeroInitWorkspaceProjection(
                    int(workspace_dim),
                    self.hidden_size,
                ).to(device=embeddings.device)
            else:
                self.register_parameter("workspace_text_anchor", None)
                self.workspace_text_projection = None

        self.dynamic_prompt = None
        if not self.directional_concat_workspace_enabled:
            self.dynamic_prompt = DynamicPromptCrossAttention(
                hidden_size=self.hidden_size,
                attention_dim=attention_dim,
                num_heads=num_heads,
            ).to(device=embeddings.device)
        self.shared_s_prompt_attention = None
        if self.shared_s_text_mode == "separate_residual":
            self.shared_s_prompt_attention = DynamicPromptCrossAttention(
                hidden_size=int(embeddings.shape[-1]),
                attention_dim=shared_s_attention_dim,
                num_heads=shared_s_heads,
            ).to(device=embeddings.device)
        self.workspace_text_attention = None
        if self.shared_workspace_enabled:
            if self.sparse_visual is None:
                raise ValueError("Shared workspace requires Sparse Visual MMRL")
            self.workspace_text_attention = LatentResidualAttention(
                query_dim=int(embeddings.shape[-1]),
                memory_dim=workspace_dim,
                attention_dim=workspace_text_attention_dim,
                num_heads=workspace_text_heads,
                zero_init_output=True,
            ).to(device=embeddings.device)

        self.visual_template_token_ids = self._resolve_visual_token_ids(tokenizer)
        self.config = self.base_model.config
        self.generation_config = getattr(self.base_model, "generation_config", None)
        self.debug_context: Dict[str, torch.Tensor] = {}
        self._memory_audited = False
        self._intervention_mode = "normal"
        self._intervention_memory_lag = 32
        self._intervention_memory_queue: Deque[torch.Tensor] = deque()
        self._intervention_delta_sum = None
        self._intervention_delta_count = 0
        self._intervention_mean_delta = None
        self._intervention_samples_seen = 0
        self._intervention_samples_changed = 0
        self._intervention_audited = False
        self._workspace_text_write_disabled = False
        self._directional_text_delta_scale = 1.0

    @staticmethod
    def _resolve_visual_token_ids(tokenizer: Any) -> Sequence[int]:
        token_ids = []
        for token in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id < 0:
                raise RuntimeError(f"Failed to resolve Dynamic Prompt token {token!r}")
            token_ids.append(int(token_id))
        return tuple(token_ids)

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    def trainable_parameter_groups(self) -> Dict[str, list[nn.Parameter]]:
        dynamic_parameters = (
            list(self.dynamic_prompt.parameters())
            if self.dynamic_prompt is not None
            else []
        )
        if self.shared_s_prompt_attention is not None:
            dynamic_parameters.extend(self.shared_s_prompt_attention.parameters())
        prompt_parameters = [
            parameter
            for parameter in (self.soft_prompt, self.workspace_text_anchor)
            if parameter is not None
        ]
        groups = {
            "soft_prompt": prompt_parameters,
            "dynamic_prompt": dynamic_parameters,
        }
        if self.sparse_visual is not None:
            groups["sparse_visual"] = self.sparse_visual.private_parameters()
        if self.directional_concat_workspace_enabled:
            if (
                self.sparse_visual is None
                or self.workspace_text_projection is None
                or self.workspace_text_anchor is None
            ):
                raise RuntimeError("Directional Concat Workspace modules are incomplete")
            groups["shared_workspace"] = (
                self.sparse_visual.workspace_parameters()
                + list(self.workspace_text_projection.parameters())
            )
        if self.shared_workspace_enabled:
            if self.sparse_visual is None or self.workspace_text_attention is None:
                raise RuntimeError("Shared workspace modules are incomplete")
            groups["shared_workspace"] = (
                self.sparse_visual.shared_workspace_parameters()
                + list(self.workspace_text_attention.parameters())
            )
        return groups

    def configure_inference_intervention(
        self,
        mode: str = "normal",
        memory_lag: int = 32,
    ) -> None:
        if self.directional_concat_workspace_enabled and mode != "normal":
            raise ValueError(
                "Legacy Dynamic Prompt interventions do not apply to "
                "Directional Concat Workspace"
            )
        if mode not in DYNAMIC_PROMPT_INTERVENTIONS:
            raise ValueError(
                f"Unsupported Dynamic Prompt intervention={mode!r}; "
                f"choices={DYNAMIC_PROMPT_INTERVENTIONS}"
            )
        if memory_lag < 1:
            raise ValueError("Dynamic Prompt memory_lag must be positive")
        self._intervention_mode = mode
        self._intervention_memory_lag = int(memory_lag)
        self.reset_inference_intervention_state()

    def configure_workspace_inference_intervention(
        self,
        visual_write_disabled_layers: Sequence[int] = (),
        visual_rep_write_disabled_layers: Sequence[int] = (),
        update_bypassed_layers: Sequence[int] = (),
        text_write_disabled: bool = False,
        directional_text_delta_scale: float = 1.0,
        directional_visual_delta_scale: float = 1.0,
        directional_visual_memory_mode: str = "normal",
        directional_question_query_mode: str = "normal",
    ) -> None:
        text_scale = float(directional_text_delta_scale)
        visual_scale = float(directional_visual_delta_scale)
        visual_memory_mode = str(directional_visual_memory_mode)
        question_query_mode = str(directional_question_query_mode)
        for name, scale in (
            ("text", text_scale),
            ("visual", visual_scale),
        ):
            if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
                raise ValueError(
                    f"Directional {name} delta scale must be finite and in [0, 1]"
                )
        if self.training and (
            text_scale != 1.0
            or visual_scale != 1.0
            or visual_memory_mode != "normal"
            or question_query_mode != "normal"
        ):
            raise RuntimeError("Directional interventions are inference-only")
        if self.directional_concat_workspace_enabled:
            if (
                visual_write_disabled_layers
                or visual_rep_write_disabled_layers
                or update_bypassed_layers
                or text_write_disabled
            ):
                raise ValueError(
                    "Legacy Workspace interventions do not apply to "
                    "Directional Concat Workspace"
                )
            if self.sparse_visual is None:
                raise RuntimeError("Directional Concat Workspace is unavailable")
            self._directional_text_delta_scale = text_scale
            self.sparse_visual.configure_inference_intervention(
                visual_delta_scale=visual_scale,
                visual_memory_mode=visual_memory_mode,
                question_query_mode=question_query_mode,
            )
            print(
                "[DIRECTIONAL_CONCAT_WORKSPACE_TEXT_INTERVENTION] "
                f"text_delta_scale={text_scale} anchors_preserved=True "
                "token_count_preserved=True"
            )
            return
        if (
            text_scale != 1.0
            or visual_scale != 1.0
            or visual_memory_mode != "normal"
            or question_query_mode != "normal"
        ):
            raise ValueError(
                "Directional interventions require Directional Concat Workspace"
            )
        if self.sparse_visual is None:
            if (
                visual_write_disabled_layers
                or visual_rep_write_disabled_layers
                or update_bypassed_layers
                or text_write_disabled
            ):
                raise ValueError("Workspace interventions require Sparse Visual MMRL")
            return
        self.sparse_visual.configure_workspace_inference_intervention(
            visual_write_disabled_layers=visual_write_disabled_layers,
            visual_rep_write_disabled_layers=visual_rep_write_disabled_layers,
            update_bypassed_layers=update_bypassed_layers,
        )
        if text_write_disabled and not self.shared_workspace_enabled:
            raise ValueError("Workspace Text intervention requires shared_workspace")
        self._workspace_text_write_disabled = bool(text_write_disabled)
        print(
            "[SHARED_WORKSPACE_TEXT_INTERVENTION] "
            f"text_write_disabled={self._workspace_text_write_disabled}"
        )

    def reset_inference_intervention_state(self) -> None:
        self._intervention_memory_queue.clear()
        self._intervention_delta_sum = None
        self._intervention_delta_count = 0
        self._intervention_mean_delta = None
        self._intervention_samples_seen = 0
        self._intervention_samples_changed = 0
        self._intervention_audited = False
        if (
            self.directional_concat_workspace_enabled
            and self.sparse_visual is not None
        ):
            self.sparse_visual.reset_inference_intervention_state()

    def inference_intervention_summary(self) -> Dict[str, Any]:
        summary = {
            "mode": self._intervention_mode,
            "memory_lag": self._intervention_memory_lag,
            "samples_seen": self._intervention_samples_seen,
            "samples_changed": self._intervention_samples_changed,
            "warmup_samples": (
                min(
                    self._intervention_samples_seen,
                    self._intervention_memory_lag,
                )
                if self._intervention_mode == "lagged-memory"
                else (
                    min(
                        self._intervention_samples_seen,
                        self._intervention_memory_lag,
                    )
                    if self._intervention_mode == "mean-residual"
                    else 0
                )
            ),
        }
        if self.sparse_visual is not None:
            workspace_summary = self.sparse_visual.workspace_inference_intervention_summary()
            if self.directional_concat_workspace_enabled:
                workspace_summary["text_delta_scale"] = (
                    self._directional_text_delta_scale
                )
            else:
                workspace_summary["text_write_disabled"] = (
                    self._workspace_text_write_disabled
                )
            summary["workspace"] = workspace_summary
        return summary

    def _apply_memory_intervention(self, memory: torch.Tensor) -> torch.Tensor:
        if self._intervention_mode != "lagged-memory":
            return memory
        selected = []
        for sample_memory in memory:
            if len(self._intervention_memory_queue) >= self._intervention_memory_lag:
                selected.append(self._intervention_memory_queue.popleft())
                self._intervention_samples_changed += 1
            else:
                selected.append(sample_memory)
            self._intervention_memory_queue.append(sample_memory.detach().clone())
        return torch.stack(selected, dim=0).to(
            device=memory.device,
            dtype=memory.dtype,
        )

    def _apply_delta_intervention(
        self,
        prompt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(prompt.shape[0])
        if self._intervention_mode == "zero":
            self._intervention_samples_changed += batch_size
            return torch.zeros_like(prompt)
        if (
            self._intervention_mode == "mean-residual"
            and self._intervention_mean_delta is not None
        ):
            self._intervention_samples_changed += batch_size
            return self._intervention_mean_delta.to(
                device=prompt.device,
                dtype=prompt.dtype,
            ).expand(batch_size, -1, -1)

        delta = self.dynamic_prompt(prompt, memory)
        if self._intervention_mode == "mean-residual":
            batch_sum = delta.detach().sum(dim=0, keepdim=True)
            self._intervention_delta_sum = (
                batch_sum.clone()
                if self._intervention_delta_sum is None
                else self._intervention_delta_sum + batch_sum
            )
            self._intervention_delta_count += batch_size
            if self._intervention_delta_count >= self._intervention_memory_lag:
                self._intervention_mean_delta = (
                    self._intervention_delta_sum / self._intervention_delta_count
                )
        return delta

    def _expand_inputs(
        self,
        batch: Dict[str, Any],
    ) -> tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        batch = dict(batch)
        input_ids = batch.pop("input_ids")
        attention_mask = batch.pop("attention_mask")
        labels = batch.get("labels")
        context_mask = batch.pop("mmrl_gating_mask", None)
        if context_mask is None:
            context_mask = (
                (labels == -100) & attention_mask.bool()
                if labels is not None
                else attention_mask.bool()
            )
        elif context_mask.shape != input_ids.shape:
            raise ValueError("mmrl_gating_mask must match input_ids")
        if labels is not None and bool((context_mask.bool() & (labels != -100)).any()):
            raise RuntimeError(
                "Dynamic Prompt text memory must not include supervised answer tokens"
            )

        batch_size = input_ids.shape[0]
        pad_id = int(getattr(self.config, "pad_token_id", 0) or 0)
        prompt_ids = torch.full(
            (batch_size, self.prompt_length),
            pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        prompt_attention = torch.ones(
            (batch_size, self.prompt_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        prompt_context = torch.zeros(
            (batch_size, self.prompt_length),
            dtype=torch.bool,
            device=context_mask.device,
        )
        expanded_ids = torch.cat((prompt_ids, input_ids), dim=1)
        expanded_attention = torch.cat((prompt_attention, attention_mask), dim=1)
        expanded_context = torch.cat((prompt_context, context_mask.bool()), dim=1)
        expanded = {
            **batch,
            "input_ids": expanded_ids,
            "attention_mask": expanded_attention,
        }
        if labels is not None:
            ignored = torch.full(
                (batch_size, self.prompt_length),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            expanded["labels"] = torch.cat((ignored, labels), dim=1)
        return expanded, expanded_ids, expanded_context

    @contextmanager
    def _inject_prompt_embeddings(self) -> Iterator[None]:
        embeddings = self.get_input_embeddings()

        def replace_prompt(_module: nn.Module, _inputs: Any, output: torch.Tensor):
            if output.ndim != 3 or output.shape[1] < self.prompt_length:
                return output
            if self.soft_prompt is None:
                prompt = self._expanded_shared_s_prompt(
                    output.shape[0], output.device, output.dtype
                )
            else:
                private_prompt = (
                    self.soft_prompt.to(
                        device=output.device,
                        dtype=output.dtype,
                    )
                    .unsqueeze(0)
                    .expand(output.shape[0], -1, -1)
                )
                if self.directional_concat_workspace_enabled:
                    if self.workspace_text_anchor is None:
                        raise RuntimeError(
                            "Directional Workspace text anchor is unavailable"
                        )
                    workspace_anchor = self.workspace_text_anchor.to(
                        device=output.device,
                        dtype=output.dtype,
                    ).unsqueeze(0).expand(output.shape[0], -1, -1)
                    prompt = torch.cat((private_prompt, workspace_anchor), dim=1)
                else:
                    prompt = private_prompt
            return torch.cat((prompt, output[:, self.prompt_length :]), dim=1)

        handle = embeddings.register_forward_hook(replace_prompt)
        try:
            yield
        finally:
            handle.remove()

    def _expanded_shared_s_prompt(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.sparse_visual is None or self.shared_s_text_mode == "none":
            raise RuntimeError("Raw Shared-S prompt was requested while disabled")
        prompt = self.sparse_visual.shared_s_text_prompt().to(
            device=device,
            dtype=dtype,
        )
        return prompt.unsqueeze(0).expand(batch_size, -1, -1)

    @contextmanager
    def _inject_dynamic_prompt(
        self,
        expanded_ids: torch.Tensor,
        expanded_context: torch.Tensor,
    ) -> Iterator[None]:
        model_core = getattr(self.base_model, "model", None)
        language_model = getattr(model_core, "language_model", None)
        if language_model is None:
            raise RuntimeError(
                "Dynamic Prompt requires base_model.model.language_model"
            )

        def replace_language_inputs(
            _module: nn.Module, args: Any, kwargs: Dict[str, Any]
        ):
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is None or inputs_embeds.ndim != 3:
                return args, kwargs
            if inputs_embeds.shape[1] != expanded_context.shape[1]:
                # Decode steps reuse the dynamic prompt already stored in KV cache.
                return args, kwargs

            if self.directional_concat_workspace_enabled:
                if (
                    self.sparse_visual is None
                    or self.workspace_text_projection is None
                    or self.workspace_text_anchor is None
                    or self.soft_prompt is None
                ):
                    raise RuntimeError(
                        "Directional Concat Workspace modules are incomplete"
                    )
                workspace = self.sparse_visual.shared_workspace_text_memory()
                workspace_prompt = self.workspace_text_projection(
                    workspace,
                    self.workspace_text_anchor,
                    delta_scale=self._directional_text_delta_scale,
                ).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                private_prompt = inputs_embeds[:, : self.private_prompt_length]
                dynamic_prompt = torch.cat(
                    (private_prompt, workspace_prompt),
                    dim=1,
                )
                kwargs["inputs_embeds"] = torch.cat(
                    (dynamic_prompt, inputs_embeds[:, self.prompt_length :]),
                    dim=1,
                )
                self.debug_context = {
                    **{
                        f"workspace_text_{key}": value
                        for key, value in self.workspace_text_projection.debug_context.items()
                    },
                    "workspace_text_memory_norm_mean": workspace.detach()
                    .float()
                    .norm(dim=-1)
                    .mean(),
                    "workspace_text_memory_tokens": inputs_embeds.new_tensor(
                        float(workspace.shape[1])
                    ),
                    "workspace_private_prompt_norm_mean": private_prompt.detach()
                    .float()
                    .norm(dim=-1)
                    .mean(),
                }
                if not self._memory_audited:
                    print(
                        "[DIRECTIONAL_CONCAT_WORKSPACE_MEMORY_AUDIT] "
                        f"private_prompt_tokens={self.private_prompt_length} "
                        f"workspace_prompt_tokens={self.workspace_prompt_length} "
                        f"workspace_shape={tuple(workspace.shape)} "
                        f"total_prompt_tokens={self.prompt_length} pass=True"
                    )
                    self._memory_audited = True
                return args, kwargs

            visual_mask = kwargs.get("visual_pos_masks")
            if visual_mask is None:
                visual_mask = expanded_ids.eq(self.visual_template_token_ids[0])
            visual_mask = visual_mask.to(
                device=inputs_embeds.device,
                dtype=torch.bool,
            )
            text_mask = (
                expanded_context.to(
                    device=inputs_embeds.device,
                    dtype=torch.bool,
                )
                & ~visual_mask
            )
            ids = expanded_ids.to(inputs_embeds.device)
            for token_id in self.visual_template_token_ids:
                text_mask = text_mask & ids.ne(token_id)
            text_mask[:, : self.prompt_length] = False

            visual_memory = _masked_mean(inputs_embeds, visual_mask, "visual")
            text_memory = _masked_mean(inputs_embeds, text_mask, "text")
            if self._intervention_mode != "normal" and self.training:
                raise RuntimeError(
                    "Dynamic Prompt causal interventions are inference-only"
                )
            natural_memory = torch.stack((visual_memory, text_memory), dim=1)
            memory = self._apply_memory_intervention(natural_memory)
            prompt = inputs_embeds[:, : self.prompt_length]
            if self.dynamic_prompt is None:
                raise RuntimeError("Dynamic Prompt attention is unavailable")
            delta = self._apply_delta_intervention(prompt, memory)
            shared_delta = torch.zeros_like(prompt)
            shared_debug = {}
            if self.shared_s_prompt_attention is not None:
                shared_prompt = self._expanded_shared_s_prompt(
                    prompt.shape[0], prompt.device, prompt.dtype
                )
                shared_delta = self.shared_s_prompt_attention(prompt, shared_prompt)
                shared_debug = {
                    "shared_s_prompt_attention_entropy_norm": self.shared_s_prompt_attention.debug_context[
                        "dynamic_prompt_attention_entropy_norm"
                    ],
                    "shared_s_prompt_token0_attention_mean": self.shared_s_prompt_attention.debug_context[
                        "dynamic_prompt_visual_attention_mean"
                    ],
                    "shared_s_prompt_token1_attention_mean": self.shared_s_prompt_attention.debug_context[
                        "dynamic_prompt_text_attention_mean"
                    ],
                    "shared_s_prompt_delta_norm_mean": self.shared_s_prompt_attention.debug_context[
                        "dynamic_prompt_delta_norm_mean"
                    ],
                    "shared_s_prompt_base_norm_mean": self.shared_s_prompt_attention.debug_context[
                        "dynamic_prompt_base_norm_mean"
                    ],
                    "shared_s_prompt_delta_to_base_ratio": self.shared_s_prompt_attention.debug_context[
                        "dynamic_prompt_delta_to_base_ratio"
                    ],
                    "shared_s_text_prompt_norm_mean": shared_prompt.detach()
                    .float()
                    .norm(dim=-1)
                    .mean(),
                    "shared_s_text_prompt_tokens": prompt.new_tensor(
                        float(shared_prompt.shape[1])
                    ),
                }
            workspace_delta = torch.zeros_like(prompt)
            workspace_debug = {}
            if self.workspace_text_attention is not None:
                if self.sparse_visual is None:
                    raise RuntimeError("Shared workspace visual trunk is unavailable")
                workspace_memory = self.sparse_visual.shared_workspace_text_memory()
                workspace_delta = self.workspace_text_attention(
                    prompt,
                    workspace_memory,
                )
                text_workspace_debug = self.workspace_text_attention.debug_context
                workspace_debug = {
                    "workspace_text_attention_entropy_norm": text_workspace_debug[
                        "attention_entropy_norm"
                    ],
                    "workspace_text_delta_norm_mean": text_workspace_debug[
                        "delta_norm_mean"
                    ],
                    "workspace_text_base_norm_mean": text_workspace_debug[
                        "base_norm_mean"
                    ],
                    "workspace_text_delta_to_base_ratio": text_workspace_debug[
                        "delta_to_base_ratio"
                    ],
                    "workspace_text_memory_norm_mean": workspace_memory.detach()
                    .float()
                    .norm(dim=-1)
                    .mean(),
                    "workspace_text_memory_tokens": prompt.new_tensor(
                        float(workspace_memory.shape[1])
                    ),
                    "workspace_text_write_disabled": prompt.new_tensor(
                        float(self._workspace_text_write_disabled)
                    ),
                }
                if self._workspace_text_write_disabled:
                    if self.training:
                        raise RuntimeError(
                            "Workspace Text write disable is inference-only"
                        )
                    workspace_delta = torch.zeros_like(workspace_delta)
            if not self.training:
                self._intervention_samples_seen += int(prompt.shape[0])
            dynamic = prompt + delta + shared_delta + workspace_delta
            kwargs["inputs_embeds"] = torch.cat(
                (dynamic, inputs_embeds[:, self.prompt_length :]), dim=1
            )
            self.debug_context = {
                **self.dynamic_prompt.debug_context,
                **shared_debug,
                **workspace_debug,
                "dynamic_prompt_visual_tokens_mean": visual_mask.sum(dim=1)
                .float()
                .mean(),
                "dynamic_prompt_text_tokens_mean": text_mask.sum(dim=1).float().mean(),
            }
            if not self._memory_audited:
                print(
                    "[DYNAMIC_PROMPT_MEMORY_AUDIT] "
                    f"visual_tokens={visual_mask.sum(dim=1).tolist()} "
                    f"text_tokens={text_mask.sum(dim=1).tolist()} "
                    f"memory_tokens={natural_memory.shape[1]} "
                    f"shared_s_text_mode={self.shared_s_text_mode} "
                    f"shared_workspace={self.shared_workspace_enabled} "
                    f"prompt_tokens={self.prompt_length} "
                    "answer_overlap=0 pass=True"
                )
                self._memory_audited = True
            if not self.training and not self._intervention_audited:
                print(
                    "[DYNAMIC_PROMPT_INTERVENTION_AUDIT] "
                    f"mode={self._intervention_mode} "
                    f"memory_lag={self._intervention_memory_lag}"
                )
                self._intervention_audited = True
            return args, kwargs

        handle = language_model.register_forward_pre_hook(
            replace_language_inputs,
            with_kwargs=True,
        )
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def _inject_sparse_visual(
        self,
        expanded_ids: torch.Tensor,
        expanded_context: torch.Tensor,
    ) -> Iterator[None]:
        if self.sparse_visual is None:
            yield
            return
        ids = expanded_ids.to(self.get_input_embeddings().weight.device)
        text_mask = expanded_context.to(device=ids.device, dtype=torch.bool)
        for token_id in self.visual_template_token_ids:
            text_mask = text_mask & ids.ne(token_id)
        text_mask[:, : self.prompt_length] = False
        token_embeddings = self.get_input_embeddings()(ids)
        text_memory = _masked_mean(token_embeddings, text_mask, "sparse visual text")
        vision_end_id = self.visual_template_token_ids[2]
        images_per_sample = ids.eq(vision_end_id).sum(dim=1)
        if bool((images_per_sample == 0).any()):
            indices = (images_per_sample == 0).nonzero(as_tuple=True)[0].tolist()
            raise RuntimeError(
                f"Sparse Visual MMRL requires an image for every sample: {indices}"
            )

        visual_model = self.base_model.model.visual

        def prepare_visual(_module: nn.Module, args: Any, kwargs: Dict[str, Any]):
            grid_thw = kwargs.get("grid_thw")
            if grid_thw is None and len(args) > 1:
                grid_thw = args[1]
            self.sparse_visual.prepare_visual(grid_thw)
            return args, kwargs

        text_anchor = (
            self.soft_prompt
            if self.shared_s_text_mode == "text_owned_visual_readonly"
            else None
        )
        with self.sparse_visual.activate(
            text_memory,
            images_per_sample,
            text_anchor=text_anchor,
            text_tokens=(
                token_embeddings
                if (
                    self.shared_workspace_enabled
                    or self.directional_concat_workspace_enabled
                )
                else None
            ),
            text_token_mask=(
                text_mask
                if (
                    self.shared_workspace_enabled
                    or self.directional_concat_workspace_enabled
                )
                else None
            ),
        ):
            handle = visual_model.register_forward_pre_hook(
                prepare_visual,
                with_kwargs=True,
            )
            try:
                yield
            finally:
                handle.remove()

    @contextmanager
    def _injection_context(
        self,
        expanded_ids: torch.Tensor,
        expanded_context: torch.Tensor,
    ) -> Iterator[None]:
        with self._inject_prompt_embeddings():
            with self._inject_sparse_visual(expanded_ids, expanded_context):
                with self._inject_dynamic_prompt(expanded_ids, expanded_context):
                    yield

    def _finalize_forward_output(self, output: Any) -> Any:
        if self.sparse_visual is None:
            return output
        self.debug_context = {
            **self.debug_context,
            **self.sparse_visual.debug_context,
        }
        return output

    def forward(self, **kwargs: Any) -> Any:
        expanded, expanded_ids, expanded_context = self._expand_inputs(kwargs)
        with self._injection_context(expanded_ids, expanded_context):
            output = self.base_model(**expanded)
        return self._finalize_forward_output(output)

    def generate(self, **kwargs: Any) -> Any:
        expanded, expanded_ids, expanded_context = self._expand_inputs(kwargs)
        with self._injection_context(expanded_ids, expanded_context):
            output = self.base_model.generate(**expanded)
        return self._finalize_forward_output(output)

    def save_dynamic_prompt(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config = {
            "method": (
                "directional_concat_workspace_prompt_tuning"
                if self.directional_concat_workspace_enabled
                else "dynamic_multimodal_prompt_tuning"
            ),
            "prompt_length": self.prompt_length,
            "requested_prompt_length": self.requested_prompt_length,
            "hidden_size": self.hidden_size,
            "attention_dim": self.attention_dim,
            "num_heads": self.num_heads,
            "init_seed": self.init_seed,
            "memory": (
                "text_attention_pool_queries_plus_full_visual_kv"
                if self.directional_concat_workspace_enabled
                else "mean_visual_plus_mean_question"
            ),
            "shared_s_text_mode": self.shared_s_text_mode,
            "shared_s_text_merger": (
                "main_visual_merger"
                if self.shared_s_text_mode in ("separate_residual", "direct_prompt")
                else None
            ),
            "shared_s_ownership": (
                {
                    "owner": "text_prompt",
                    "visual_access": "detach_layer_norm_adapter",
                    "visual_adapter_bottleneck_dim": self.sparse_visual.text_anchor_bottleneck_dim,
                }
                if self.shared_s_text_mode == "text_owned_visual_readonly"
                else None
            ),
            "shared_s_prompt_attention": (
                {
                    "attention_dim": self.shared_s_prompt_attention.attention_dim,
                    "num_heads": self.shared_s_prompt_attention.num_heads,
                    "zero_init_output": True,
                }
                if self.shared_s_prompt_attention is not None
                else None
            ),
            "shared_workspace": (
                {
                    "tokens": self.sparse_visual.workspace_tokens,
                    "dim": self.sparse_visual.workspace_dim,
                    "heads": self.sparse_visual.workspace_heads,
                    "ffn_dim": self.sparse_visual.workspace_ffn_dim,
                    "text_attention_dim": self.workspace_text_attention.attention_dim,
                    "text_attention_heads": self.workspace_text_attention.num_heads,
                    "visual_attention_dim": self.sparse_visual.workspace_visual_attentions[
                        0
                    ].attention_dim,
                    "visual_attention_heads": self.sparse_visual.workspace_visual_attentions[
                        0
                    ].num_heads,
                    "memory": "full_visual_plus_full_question_tokens",
                    "private_text_prompt": True,
                    "private_visual_rep": True,
                    "zero_init_residual_heads": True,
                }
                if self.shared_workspace_enabled
                else None
            ),
            "directional_concat_workspace": (
                {
                    "tokens": self.sparse_visual.workspace_tokens,
                    "dim": self.sparse_visual.workspace_dim,
                    "heads": self.sparse_visual.workspace_heads,
                    "anchor_layer": self.sparse_visual.anchor_layers[0],
                    "private_visual_prompt_tokens": self.sparse_visual.private_prompt_tokens,
                    "private_text_prompt_tokens": self.private_prompt_length,
                    "text_workspace_anchor_tokens": self.workspace_prompt_length,
                    "query": "attention_pool_full_question_tokens",
                    "memory": "full_layer17_visual_tokens",
                    "fusion": "text_query_visual_kv_cross_attention",
                    "visual_output": "anchored_token_concat",
                    "text_output": "anchored_token_concat",
                    "visual_dynamic_write": self.sparse_visual.visual_dynamic_write,
                    "text_dynamic_projection_zero_initialized": True,
                    "visual_dynamic_projection_zero_initialized": (
                        self.sparse_visual.visual_dynamic_write
                    ),
                    "zero_init_dynamic_projections": True,
                    "private_dynamic_text_attention": False,
                    "private_visual_attention": False,
                }
                if self.directional_concat_workspace_enabled
                else None
            ),
            "train_soft_prompt": self.soft_prompt is not None,
            "sparse_visual": (
                {
                    "anchor_layers": list(self.sparse_visual.anchor_layers),
                    "rep_token_count": (
                        self.sparse_visual.private_prompt_tokens
                        if self.directional_concat_workspace_enabled
                        else self.sparse_visual.rep_token_count
                    ),
                    "attention_dim": (
                        0
                        if self.directional_concat_workspace_enabled
                        else self.sparse_visual.rep_attention.attention_dim
                    ),
                    "num_heads": (
                        0
                        if self.directional_concat_workspace_enabled
                        else self.sparse_visual.rep_attention.num_heads
                    ),
                    "injection_mode": (
                        "directional_concat_single_pass_insert_strip"
                        if self.directional_concat_workspace_enabled
                        else "single_pass_insert_strip"
                    ),
                    "map_raw_shared_s_to_text": self.shared_s_text_mode
                    in ("separate_residual", "direct_prompt"),
                    "text_owned_s_visual_readonly": (
                        self.shared_s_text_mode == "text_owned_visual_readonly"
                    ),
                    "text_anchor_tokens": (
                        0
                        if self.directional_concat_workspace_enabled
                        else self.sparse_visual.text_anchor_tokens
                    ),
                    "text_anchor_bottleneck_dim": (
                        0
                        if self.directional_concat_workspace_enabled
                        else self.sparse_visual.text_anchor_bottleneck_dim
                    ),
                    "shared_s_text_merger": (
                        "main_visual_merger"
                        if self.shared_s_text_mode
                        in ("separate_residual", "direct_prompt")
                        else None
                    ),
                }
                if self.sparse_visual is not None
                else None
            ),
        }
        with (output_path / DYNAMIC_PROMPT_CONFIG_NAME).open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        torch.save(
            {
                "soft_prompt": (
                    self.soft_prompt.detach().cpu()
                    if self.soft_prompt is not None
                    else None
                ),
                "dynamic_prompt": {
                    key: value.detach().cpu()
                    for key, value in self.dynamic_prompt.state_dict().items()
                } if self.dynamic_prompt is not None else None,
                "workspace_text_anchor": (
                    self.workspace_text_anchor.detach().cpu()
                    if self.workspace_text_anchor is not None
                    else None
                ),
                "workspace_text_projection": (
                    {
                        key: value.detach().cpu()
                        for key, value in self.workspace_text_projection.state_dict().items()
                    }
                    if self.workspace_text_projection is not None
                    else None
                ),
                "shared_s_prompt_attention": (
                    {
                        key: value.detach().cpu()
                        for key, value in self.shared_s_prompt_attention.state_dict().items()
                    }
                    if self.shared_s_prompt_attention is not None
                    else None
                ),
                "workspace_text_attention": (
                    {
                        key: value.detach().cpu()
                        for key, value in self.workspace_text_attention.state_dict().items()
                    }
                    if self.workspace_text_attention is not None
                    else None
                ),
                "sparse_visual": (
                    {
                        key: value.detach().cpu()
                        for key, value in self.sparse_visual.state_dict().items()
                    }
                    if self.sparse_visual is not None
                    else None
                ),
            },
            output_path / DYNAMIC_PROMPT_WEIGHTS_NAME,
        )

    def load_dynamic_prompt(self, checkpoint_dir: str | Path) -> None:
        checkpoint_path = Path(checkpoint_dir)
        state = torch.load(
            checkpoint_path / DYNAMIC_PROMPT_WEIGHTS_NAME,
            map_location="cpu",
            weights_only=True,
        )
        prompt = state["soft_prompt"]
        if (prompt is None) != (self.soft_prompt is None):
            raise ValueError(
                "Soft prompt checkpoint/model architecture mismatch: "
                f"checkpoint={prompt is not None} model={self.soft_prompt is not None}"
            )
        if self.soft_prompt is not None:
            if tuple(prompt.shape) != tuple(self.soft_prompt.shape):
                raise ValueError(
                    f"Soft prompt shape mismatch: {tuple(prompt.shape)} vs "
                    f"{tuple(self.soft_prompt.shape)}"
                )
            self.soft_prompt.data.copy_(prompt.to(self.soft_prompt.device))
        dynamic_state = state.get("dynamic_prompt")
        if (dynamic_state is None) != (self.dynamic_prompt is None):
            raise ValueError(
                "Dynamic Prompt checkpoint/model architecture mismatch: "
                f"checkpoint={dynamic_state is not None} "
                f"model={self.dynamic_prompt is not None}"
            )
        if self.dynamic_prompt is not None:
            self.dynamic_prompt.load_state_dict(dynamic_state, strict=True)
        workspace_anchor = state.get("workspace_text_anchor")
        if (workspace_anchor is None) != (self.workspace_text_anchor is None):
            raise ValueError(
                "Workspace text anchor checkpoint/model architecture mismatch"
            )
        if self.workspace_text_anchor is not None:
            if tuple(workspace_anchor.shape) != tuple(self.workspace_text_anchor.shape):
                raise ValueError("Workspace text anchor shape mismatch")
            self.workspace_text_anchor.data.copy_(
                workspace_anchor.to(self.workspace_text_anchor.device)
            )
        workspace_projection_state = state.get("workspace_text_projection")
        if (workspace_projection_state is None) != (
            self.workspace_text_projection is None
        ):
            raise ValueError(
                "Workspace text projection checkpoint/model architecture mismatch"
            )
        if self.workspace_text_projection is not None:
            self.workspace_text_projection.load_state_dict(
                workspace_projection_state,
                strict=True,
            )
            self.workspace_text_projection._forward_audited = True
        shared_attention_state = state.get("shared_s_prompt_attention")
        if (shared_attention_state is None) != (self.shared_s_prompt_attention is None):
            raise ValueError(
                "Shared-S prompt attention checkpoint/model architecture mismatch: "
                f"checkpoint={shared_attention_state is not None} "
                f"model={self.shared_s_prompt_attention is not None}"
            )
        if self.shared_s_prompt_attention is not None:
            self.shared_s_prompt_attention.load_state_dict(
                shared_attention_state, strict=True
            )
            self.shared_s_prompt_attention._forward_audited = True
        workspace_text_state = state.get("workspace_text_attention")
        if (workspace_text_state is None) != (self.workspace_text_attention is None):
            raise ValueError(
                "Workspace text attention checkpoint/model architecture mismatch: "
                f"checkpoint={workspace_text_state is not None} "
                f"model={self.workspace_text_attention is not None}"
            )
        if self.workspace_text_attention is not None:
            self.workspace_text_attention.load_state_dict(
                workspace_text_state,
                strict=True,
            )
            self.workspace_text_attention._forward_audited = True
        sparse_state = state.get("sparse_visual")
        if (sparse_state is None) != (self.sparse_visual is None):
            raise ValueError(
                "Sparse Visual checkpoint/model architecture mismatch: "
                f"checkpoint={sparse_state is not None} model={self.sparse_visual is not None}"
            )
        if self.sparse_visual is not None:
            self.sparse_visual.load_state_dict(sparse_state, strict=True)
            self.sparse_visual._forward_audited = True
            if self.directional_concat_workspace_enabled:
                if self.sparse_visual.workspace_visual_delta is not None:
                    self.sparse_visual.workspace_visual_delta._forward_audited = True
            else:
                for attention in self.sparse_visual.workspace_visual_attentions:
                    attention._forward_audited = True
        # A restored checkpoint is no longer expected to have a zero residual.
        if self.dynamic_prompt is not None:
            self.dynamic_prompt._forward_audited = True
        print(
            "[DYNAMIC_PROMPT_CHECKPOINT_AUDIT] "
            f"loaded={checkpoint_path} zero_init_check=skipped_for_trained_state"
        )
