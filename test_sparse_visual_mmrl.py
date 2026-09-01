import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from slake.directional_concat_workspace import (
    DirectionalConcatWorkspaceVisual,
    ZeroInitWorkspaceProjection,
)
from slake.sparse_visual_mmrl import LatentResidualAttention, SparseVisualMMRL


class _SegmentMixingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None,
    ):
        self.calls += 1
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outputs = []
        for segment in torch.split(hidden_states, lengths, dim=0):
            outputs.append(segment + segment.mean(dim=0, keepdim=True))
        return torch.cat(outputs, dim=0)


class _FakeVisual(nn.Module):
    def __init__(self, depth=3, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.blocks = nn.ModuleList(_SegmentMixingBlock() for _ in range(depth))


class _FakeMerger(nn.Module):
    def __init__(self, visual_dim=8, text_dim=12, merge_unit=4):
        super().__init__()
        self.linear = nn.Linear(visual_dim * merge_unit, text_dim, bias=False)

    def forward(self, hidden_states):
        return self.linear(hidden_states.view(-1, self.linear.in_features))


class _FakeMergedVisual(_FakeVisual):
    def __init__(self):
        super().__init__()
        self.merger = _FakeMerger()
        self.spatial_merge_unit = 4


class SparseVisualMMRLTest(unittest.TestCase):
    def _adapter(self):
        return SparseVisualMMRL(
            visual_dim=8,
            text_dim=12,
            anchor_layers=(1,),
            rep_token_count=2,
            attention_dim=4,
            num_heads=2,
        )

    def test_inactive_wrapper_executes_original_block_once(self):
        torch.manual_seed(7)
        visual = _FakeVisual()
        adapter = self._adapter()
        adapter.install(visual)
        hidden = torch.randn(7, 8)
        cu_seqlens = torch.tensor([0, 2, 4, 7], dtype=torch.int32)

        expected = _SegmentMixingBlock()(hidden, cu_seqlens)
        actual = visual.blocks[1](hidden, cu_seqlens=cu_seqlens)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(visual.blocks[1].block.calls, 1)

    def test_active_injection_executes_block_once_and_trains_rep_generator(self):
        torch.manual_seed(11)
        visual = _FakeVisual()
        adapter = self._adapter()
        adapter.install(visual)
        hidden = torch.randn(7, 8)
        cu_seqlens = torch.tensor([0, 2, 4, 7], dtype=torch.int32)
        with adapter.activate(torch.randn(2, 12), torch.tensor([1, 1])):
            adapter.prepare_visual(torch.tensor([[1, 1, 4], [1, 1, 3]]))
            output = visual.blocks[1](hidden, cu_seqlens=cu_seqlens)
            loss = output.square().mean()
        self.assertEqual(visual.blocks[1].block.calls, 1)
        self.assertFalse(torch.equal(output, _SegmentMixingBlock()(hidden, cu_seqlens)))
        loss.backward()
        self.assertGreater(float(adapter.shared_rep.grad.abs().sum()), 0.0)
        self.assertIn(
            "sparse_visual_layer1_rep_to_input_norm_ratio",
            adapter.debug_context,
        )

    def test_production_parameter_count(self):
        adapter = SparseVisualMMRL(
            visual_dim=1024,
            text_dim=2560,
            anchor_layers=(5, 11, 17),
            rep_token_count=8,
            attention_dim=128,
            num_heads=4,
        )
        parameter_count = sum(parameter.numel() for parameter in adapter.parameters())
        self.assertEqual(parameter_count, 1_201_152)

    def test_raw_shared_s_maps_to_text_without_visual_conditioning(self):
        torch.manual_seed(13)
        visual = _FakeMergedVisual()
        for parameter in visual.parameters():
            parameter.requires_grad = False
        adapter = SparseVisualMMRL(
            visual_dim=8,
            text_dim=12,
            anchor_layers=(1,),
            rep_token_count=4,
            attention_dim=4,
            num_heads=2,
            map_shared_s_to_text=True,
        )
        adapter.install(visual)
        self.assertNotIn(
            "merger",
            dict(adapter.named_modules()),
        )
        shared_prompt = adapter.shared_s_text_prompt()
        expected = visual.merger(adapter.shared_rep)
        self.assertEqual(adapter.shared_s_text_prompt_length, 1)
        self.assertEqual(tuple(shared_prompt.shape), (1, 12))
        torch.testing.assert_close(shared_prompt, expected)
        loss = shared_prompt.square().mean()
        loss.backward()
        self.assertGreater(float(adapter.shared_rep.grad.abs().sum()), 0.0)
        self.assertGreater(float(adapter.shared_s_text_prompt_grad_norm()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in visual.parameters()))

    def test_text_owned_s_is_detached_while_visual_adapter_trains(self):
        adapter = SparseVisualMMRL(
            visual_dim=8,
            text_dim=12,
            anchor_layers=(1,),
            rep_token_count=4,
            attention_dim=4,
            num_heads=2,
            text_anchor_tokens=6,
            text_anchor_bottleneck_dim=4,
        )
        text_anchor = torch.randn(6, 12, requires_grad=True)
        visual_rep = adapter.adapt_read_only_text_anchor(text_anchor)
        self.assertEqual(tuple(visual_rep.shape), (4, 8))
        visual_rep.square().mean().backward()
        self.assertIsNone(text_anchor.grad)
        self.assertIsNone(adapter.shared_rep)
        self.assertGreater(
            float(adapter.text_anchor_up_projection.weight.grad.norm()),
            0.0,
        )
        self.assertIn(
            "shared_s_visual_adapter_token_entropy_norm",
            adapter._text_anchor_debug,
        )

    def test_full_workspace_uses_tokens_and_preserves_private_visual_path_at_init(self):
        torch.manual_seed(17)
        visual = _FakeVisual()
        adapter = SparseVisualMMRL(
            visual_dim=8,
            text_dim=12,
            anchor_layers=(1,),
            rep_token_count=2,
            attention_dim=4,
            num_heads=2,
            shared_workspace=True,
            workspace_tokens=3,
            workspace_dim=8,
            workspace_heads=2,
            workspace_ffn_dim=16,
            workspace_visual_attention_dim=4,
            workspace_visual_heads=2,
        )
        adapter.install(visual)
        hidden = torch.randn(7, 8)
        text_tokens = torch.randn(2, 4, 12)
        text_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        cu_seqlens = torch.tensor([0, 2, 4, 7], dtype=torch.int32)
        with adapter.activate(
            torch.randn(2, 12),
            torch.tensor([1, 1]),
            text_tokens=text_tokens,
            text_token_mask=text_mask,
        ):
            adapter.prepare_visual(torch.tensor([[1, 1, 4], [1, 1, 3]]))
            output = visual.blocks[1](hidden, cu_seqlens=cu_seqlens)
            workspace = adapter.shared_workspace_text_memory()
            self.assertEqual(tuple(workspace.shape), (2, 3, 8))
            self.assertEqual(
                float(
                    adapter.debug_context[
                        "workspace_layer1_visual_delta_to_base_ratio"
                    ]
                ),
                0.0,
            )
            loss = output.square().mean() + workspace.square().mean()
        loss.backward()
        self.assertGreater(float(adapter.shared_rep.grad.norm()), 0.0)
        self.assertGreater(float(adapter.workspace_seed.grad.norm()), 0.0)
        private_ids = {id(parameter) for parameter in adapter.private_parameters()}
        workspace_ids = {
            id(parameter) for parameter in adapter.shared_workspace_parameters()
        }
        self.assertTrue(private_ids.isdisjoint(workspace_ids))
        self.assertEqual(
            private_ids | workspace_ids,
            {id(parameter) for parameter in adapter.parameters()},
        )

    def test_workspace_inference_interventions_separate_write_from_update(self):
        torch.manual_seed(19)
        visual = _FakeVisual()
        adapter = SparseVisualMMRL(
            visual_dim=8,
            text_dim=12,
            anchor_layers=(1,),
            rep_token_count=2,
            attention_dim=4,
            num_heads=2,
            shared_workspace=True,
            workspace_tokens=3,
            workspace_dim=8,
            workspace_heads=2,
            workspace_ffn_dim=16,
            workspace_visual_attention_dim=4,
            workspace_visual_heads=2,
        )
        adapter.install(visual)
        adapter.eval()
        hidden = torch.randn(7, 8)
        text_memory = torch.randn(2, 12)
        text_tokens = torch.randn(2, 4, 12)
        text_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        image_counts = torch.tensor([1, 1])
        grid = torch.tensor([[1, 1, 4], [1, 1, 3]])
        cu_seqlens = torch.tensor([0, 2, 4, 7], dtype=torch.int32)

        def run_once():
            with adapter.activate(
                text_memory,
                image_counts,
                text_tokens=text_tokens,
                text_token_mask=text_mask,
            ):
                adapter.prepare_visual(grid)
                output = visual.blocks[1](hidden, cu_seqlens=cu_seqlens)
                workspace = adapter.shared_workspace_text_memory().clone()
                debug = dict(adapter.debug_context)
            return output, workspace, debug

        run_once()  # Complete the exact-zero initialization audit first.
        with torch.no_grad():
            reader = adapter.workspace_visual_attentions[0]
            reader.output_projection.weight.fill_(0.1)
            reader.output_projection.bias.zero_()

        adapter.configure_workspace_inference_intervention()
        normal_output, normal_workspace, _ = run_once()
        adapter.configure_workspace_inference_intervention(
            visual_write_disabled_layers=(1,)
        )
        disabled_output, disabled_workspace, disabled_debug = run_once()
        torch.testing.assert_close(disabled_workspace, normal_workspace)
        self.assertFalse(torch.allclose(disabled_output, normal_output))
        self.assertEqual(
            float(disabled_debug["workspace_layer1_visual_write_disabled"]),
            1.0,
        )

        adapter.configure_workspace_inference_intervention(
            update_bypassed_layers=(1,)
        )
        _, bypassed_workspace, bypassed_debug = run_once()
        self.assertFalse(torch.allclose(bypassed_workspace, normal_workspace))
        self.assertEqual(
            float(bypassed_debug["workspace_layer1_update_bypassed"]),
            1.0,
        )
        adapter.configure_workspace_inference_intervention(
            visual_rep_write_disabled_layers=(1,)
        )
        (
            rep_disabled_output,
            rep_disabled_workspace,
            rep_disabled_debug,
        ) = run_once()
        torch.testing.assert_close(rep_disabled_workspace, normal_workspace)
        torch.testing.assert_close(
            rep_disabled_output,
            _SegmentMixingBlock()(hidden, cu_seqlens),
        )
        self.assertEqual(
            float(rep_disabled_debug["workspace_layer1_visual_rep_write_disabled"]),
            1.0,
        )
        self.assertEqual(
            adapter.workspace_inference_intervention_summary()[
                "visual_rep_write_disabled_layers"
            ],
            [1],
        )
        with self.assertRaisesRegex(ValueError, "configured anchors"):
            adapter.configure_workspace_inference_intervention(
                visual_write_disabled_layers=(0,)
            )

    def test_workspace_multilayer_similarity_diagnostics_are_finite(self):
        torch.manual_seed(23)
        visual = _FakeVisual()
        adapter = SparseVisualMMRL(
            visual_dim=8,
            text_dim=12,
            anchor_layers=(0, 1, 2),
            rep_token_count=2,
            attention_dim=4,
            num_heads=2,
            shared_workspace=True,
            workspace_tokens=3,
            workspace_dim=8,
            workspace_heads=2,
            workspace_ffn_dim=16,
            workspace_visual_attention_dim=4,
            workspace_visual_heads=2,
        )
        adapter.install(visual)
        adapter.eval()
        hidden = torch.randn(7, 8)
        cu_seqlens = torch.tensor([0, 2, 4, 7], dtype=torch.int32)
        with adapter.activate(
            torch.randn(2, 12),
            torch.tensor([1, 1]),
            text_tokens=torch.randn(2, 4, 12),
            text_token_mask=torch.tensor(
                [[True, True, False, False], [True, True, True, False]]
            ),
        ):
            adapter.prepare_visual(torch.tensor([[1, 1, 4], [1, 1, 3]]))
            for block in visual.blocks:
                hidden = block(hidden, cu_seqlens=cu_seqlens)
            debug = dict(adapter.debug_context)

        for transition in ("0_to_1", "1_to_2"):
            for suffix in (
                "state_cosine_mean",
                "update_cosine_mean",
                "visual_delta_cosine_mean",
            ):
                key = f"workspace_transition_{transition}_{suffix}"
                self.assertIn(key, debug)
                self.assertTrue(torch.isfinite(debug[key]))

    def test_full_workspace_production_parameter_count(self):
        adapter = SparseVisualMMRL(
            visual_dim=1024,
            text_dim=2560,
            anchor_layers=(5, 11, 17),
            rep_token_count=8,
            attention_dim=128,
            num_heads=4,
            shared_workspace=True,
            workspace_tokens=32,
            workspace_dim=1024,
            workspace_heads=16,
            workspace_ffn_dim=4096,
            workspace_visual_attention_dim=1024,
            workspace_visual_heads=16,
        )
        private_count = sum(
            parameter.numel() for parameter in adapter.private_parameters()
        )
        workspace_count = sum(
            parameter.numel()
            for parameter in adapter.shared_workspace_parameters()
        )
        self.assertEqual(private_count, 1_201_152)
        self.assertEqual(workspace_count, 65_659_904)
        text_head = LatentResidualAttention(
            query_dim=2560,
            memory_dim=1024,
            attention_dim=1024,
            num_heads=16,
        )
        text_head_count = sum(
            parameter.numel() for parameter in text_head.parameters()
        )
        self.assertEqual(text_head_count, 7_349_760)
        self.assertEqual(
            51_200 + 2_634_240 + private_count + workspace_count + text_head_count,
            76_896_256,
        )

    def test_directional_concat_workspace_uses_text_queries_and_token_concat(self):
        torch.manual_seed(29)
        visual = _FakeVisual()
        adapter = DirectionalConcatWorkspaceVisual(
            visual_dim=8,
            text_dim=12,
            anchor_layer=1,
            private_prompt_tokens=2,
            workspace_tokens=3,
            workspace_dim=8,
            workspace_heads=2,
        )
        adapter.eval()
        adapter.configure_inference_intervention(visual_delta_scale=0.5)
        adapter.install(visual)
        hidden = torch.randn(7, 8)
        text_tokens = torch.randn(2, 4, 12)
        text_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        cu_seqlens = torch.tensor([0, 2, 4, 7], dtype=torch.int32)
        with adapter.activate(
            torch.randn(2, 12),
            torch.tensor([1, 1]),
            text_tokens=text_tokens,
            text_token_mask=text_mask,
        ):
            adapter.prepare_visual(torch.tensor([[1, 1, 4], [1, 1, 3]]))
            output = visual.blocks[1](hidden, cu_seqlens=cu_seqlens)
            workspace = adapter.shared_workspace_text_memory()
            self.assertEqual(tuple(workspace.shape), (2, 3, 8))
            self.assertEqual(
                float(
                    adapter.debug_context[
                        "workspace_visual_delta_norm_mean"
                    ]
                ),
                0.0,
            )
            self.assertEqual(
                float(adapter.debug_context["workspace_visual_delta_scale"]),
                0.5,
            )
            self.assertTrue(
                bool(
                    torch.isfinite(
                        adapter.debug_context[
                            "workspace_visual_attention_entropy_norm"
                        ]
                    )
                )
            )
            loss = output.square().mean() + workspace.square().mean()
        self.assertEqual(visual.blocks[1].block.calls, 1)
        self.assertEqual(tuple(output.shape), tuple(hidden.shape))
        loss.backward()
        self.assertGreater(float(adapter.private_visual_prompt.grad.norm()), 0.0)
        self.assertGreater(
            float(adapter.workspace_visual_delta.output_projection.weight.grad.norm()),
            0.0,
        )
        self.assertGreater(
            float(adapter.workspace_text_value_projection.weight.grad.norm()),
            0.0,
        )
        private_ids = {id(parameter) for parameter in adapter.private_parameters()}
        workspace_ids = {id(parameter) for parameter in adapter.workspace_parameters()}
        self.assertTrue(private_ids.isdisjoint(workspace_ids))
        self.assertEqual(
            private_ids | workspace_ids,
            {id(parameter) for parameter in adapter.parameters()},
        )
        self.assertEqual(
            adapter.workspace_inference_intervention_summary()[
                "visual_delta_scale"
            ],
            0.5,
        )

    def test_directional_concat_without_visual_dynamic_write_uses_static_anchor(self):
        torch.manual_seed(30)
        visual = _FakeVisual()
        adapter = DirectionalConcatWorkspaceVisual(
            visual_dim=8,
            text_dim=12,
            anchor_layer=1,
            private_prompt_tokens=2,
            workspace_tokens=3,
            workspace_dim=4,
            workspace_heads=2,
            visual_dynamic_write=False,
        )
        self.assertIsNone(adapter.workspace_visual_delta)
        self.assertIsInstance(adapter.workspace_visual_memory_projection, nn.Linear)
        self.assertEqual(
            tuple(adapter.workspace_visual_memory_projection.weight.shape),
            (4, 8),
        )
        adapter.install(visual)
        hidden = torch.randn(7, 8)
        text_tokens = torch.randn(2, 4, 12)
        text_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        cu_seqlens = torch.tensor([0, 2, 4, 7], dtype=torch.int32)
        with adapter.activate(
            torch.randn(2, 12),
            torch.tensor([1, 1]),
            text_tokens=text_tokens,
            text_token_mask=text_mask,
        ):
            adapter.prepare_visual(torch.tensor([[1, 1, 4], [1, 1, 3]]))
            output = visual.blocks[1](hidden, cu_seqlens=cu_seqlens)
            workspace = adapter.shared_workspace_text_memory()
            self.assertEqual(
                float(
                    adapter.debug_context[
                        "workspace_visual_dynamic_write_enabled"
                    ]
                ),
                0.0,
            )
            self.assertEqual(
                float(adapter.debug_context["workspace_visual_delta_norm_mean"]),
                0.0,
            )
            self.assertEqual(
                float(
                    adapter.debug_context[
                        "workspace_visual_projection_enabled"
                    ]
                ),
                1.0,
            )
            loss = output.square().mean() + workspace.square().mean()
        self.assertEqual(visual.blocks[1].block.calls, 1)
        self.assertEqual(tuple(output.shape), tuple(hidden.shape))
        loss.backward()
        self.assertGreater(float(adapter.private_visual_prompt.grad.norm()), 0.0)
        self.assertGreater(float(adapter.workspace_visual_anchor.grad.norm()), 0.0)
        self.assertGreater(
            float(adapter.workspace_cross_attention.in_proj_weight.grad.norm()),
            0.0,
        )
        self.assertFalse(
            adapter.workspace_inference_intervention_summary()[
                "visual_dynamic_write"
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires visual_dynamic_write"):
            adapter.configure_inference_intervention(visual_delta_scale=0.5)

    def test_workspace_projection_scales_only_dynamic_delta(self):
        torch.manual_seed(31)
        projection = ZeroInitWorkspaceProjection(8, 12)
        workspace = torch.randn(2, 3, 8)
        anchor = torch.randn(3, 12)
        projection(workspace, anchor)
        with torch.no_grad():
            projection.output_projection.weight.fill_(0.1)
            projection.output_projection.bias.fill_(0.05)
        full = projection(workspace, anchor, delta_scale=1.0)
        half = projection(workspace, anchor, delta_scale=0.5)
        off = projection(workspace, anchor, delta_scale=0.0)
        expanded_anchor = anchor.unsqueeze(0).expand_as(off)
        torch.testing.assert_close(off, expanded_anchor)
        torch.testing.assert_close(
            half - expanded_anchor,
            (full - expanded_anchor) * 0.5,
        )
        self.assertEqual(float(projection.debug_context["delta_scale"]), 0.0)
        with self.assertRaisesRegex(ValueError, "in \\[0, 1\\]"):
            projection(workspace, anchor, delta_scale=1.1)

    def test_directional_visual_memory_uses_previous_distinct_image(self):
        adapter = DirectionalConcatWorkspaceVisual(
            visual_dim=8,
            text_dim=12,
            anchor_layer=1,
            private_prompt_tokens=2,
            workspace_tokens=3,
            workspace_dim=8,
            workspace_heads=2,
        ).eval()
        adapter.configure_inference_intervention(
            visual_memory_mode="previous-distinct-image"
        )

        def apply(memory):
            mask = torch.ones(
                memory.shape[:2],
                dtype=torch.bool,
                device=memory.device,
            )
            return adapter._apply_visual_memory_intervention(memory, mask)

        image_a = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
        image_b = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8) + 100
        image_c = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) + 200

        selected, selected_mask = apply(image_a)
        torch.testing.assert_close(selected[selected_mask], image_a.reshape(-1, 8))
        selected, selected_mask = apply(image_a.clone())
        torch.testing.assert_close(selected[selected_mask], image_a.reshape(-1, 8))
        selected, selected_mask = apply(image_b)
        torch.testing.assert_close(selected[selected_mask], image_a.reshape(-1, 8))
        selected, selected_mask = apply(image_b.clone())
        torch.testing.assert_close(selected[selected_mask], image_a.reshape(-1, 8))
        selected, selected_mask = apply(image_c)
        torch.testing.assert_close(selected[selected_mask], image_b.reshape(-1, 8))

        summary = adapter.workspace_inference_intervention_summary()
        self.assertEqual(summary["visual_memory_mode"], "previous-distinct-image")
        self.assertEqual(summary["visual_memory_images_seen"], 5)
        self.assertEqual(summary["visual_memory_images_mismatched"], 3)
        self.assertEqual(summary["visual_memory_images_natural"], 2)
        self.assertTrue(summary["original_image_input_unchanged"])
        self.assertEqual(
            summary["intervention_scope"],
            "directional_ca_visual_kv_only",
        )
        adapter.reset_inference_intervention_state()
        reset_summary = adapter.workspace_inference_intervention_summary()
        self.assertEqual(reset_summary["visual_memory_images_seen"], 0)
        self.assertEqual(reset_summary["visual_memory_images_mismatched"], 0)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            adapter.configure_inference_intervention(
                visual_memory_mode="not-a-mode"
            )
        adapter.train()
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            adapter.configure_inference_intervention(
                visual_memory_mode="previous-distinct-image"
            )

    def test_directional_concat_workspace_production_parameter_count(self):
        adapter = DirectionalConcatWorkspaceVisual(
            visual_dim=1024,
            text_dim=2560,
            anchor_layer=17,
            private_prompt_tokens=8,
            workspace_tokens=10,
            workspace_dim=1024,
            workspace_heads=16,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in adapter.private_parameters()),
            8_192,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in adapter.workspace_parameters()),
            8_966_144,
        )
        self.assertIsInstance(adapter.workspace_visual_memory_projection, nn.Identity)
        text_projection = ZeroInitWorkspaceProjection(1024, 2560)
        private_text_prompt = nn.Parameter(torch.empty(20, 2560))
        workspace_text_anchor = nn.Parameter(torch.empty(10, 2560))
        total = (
            sum(parameter.numel() for parameter in adapter.parameters())
            + sum(parameter.numel() for parameter in text_projection.parameters())
            + private_text_prompt.numel()
            + workspace_text_anchor.numel()
        )
        self.assertEqual(total, 12_726_784)

        static_visual_adapter = DirectionalConcatWorkspaceVisual(
            visual_dim=1024,
            text_dim=2560,
            anchor_layer=17,
            private_prompt_tokens=8,
            workspace_tokens=10,
            workspace_dim=1024,
            workspace_heads=16,
            visual_dynamic_write=False,
        )
        self.assertIsNone(static_visual_adapter.workspace_visual_delta)
        static_total = (
            sum(parameter.numel() for parameter in static_visual_adapter.parameters())
            + sum(parameter.numel() for parameter in text_projection.parameters())
            + private_text_prompt.numel()
            + workspace_text_anchor.numel()
        )
        self.assertEqual(static_total, 10_625_536)

        compressed_expectations = {
            256: (1_224_192, 2_033_408),
            512: (2_929_664, 4_591_616),
            768: (5_159_424, 7_805_184),
        }
        for workspace_dim, (
            expected_workspace_parameters,
            expected_total,
        ) in compressed_expectations.items():
            with self.subTest(workspace_dim=workspace_dim):
                compressed_adapter = DirectionalConcatWorkspaceVisual(
                    visual_dim=1024,
                    text_dim=2560,
                    anchor_layer=17,
                    private_prompt_tokens=8,
                    workspace_tokens=10,
                    workspace_dim=workspace_dim,
                    workspace_heads=16,
                    visual_dynamic_write=False,
                )
                compressed_text_projection = ZeroInitWorkspaceProjection(
                    workspace_dim,
                    2560,
                )
                compressed_total = (
                    sum(
                        parameter.numel()
                        for parameter in compressed_adapter.parameters()
                    )
                    + sum(
                        parameter.numel()
                        for parameter in compressed_text_projection.parameters()
                    )
                    + private_text_prompt.numel()
                    + workspace_text_anchor.numel()
                )
                self.assertIsInstance(
                    compressed_adapter.workspace_visual_memory_projection,
                    nn.Linear,
                )
                self.assertEqual(
                    sum(
                        parameter.numel()
                        for parameter in compressed_adapter.workspace_parameters()
                    ),
                    expected_workspace_parameters,
                )
                self.assertEqual(compressed_total, expected_total)


if __name__ == "__main__":
    unittest.main()
