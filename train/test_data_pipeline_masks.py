import unittest

import torch

from data_pipeline import (
    FourViewMMRLDataset,
    build_compact_target_window,
    build_joint_assistant_supervision_masks,
    build_target_supervision_masks,
    expand_conversation_by_assistant_turn,
)


class TargetSupervisionMaskTests(unittest.TestCase):
    HEADER = [100, 101, 102]
    LABEL_PREFIX = [100, 101]
    IM_END = 103

    def test_single_turn_supervises_only_target_tail(self):
        input_ids = torch.tensor([1, 2, 100, 101, 102, 10, 103, 11, 0, 0])
        attention_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])

        labels, pooling_mask = build_target_supervision_masks(
            input_ids,
            attention_mask,
            self.HEADER,
            self.LABEL_PREFIX,
            self.IM_END,
        )

        self.assertEqual(
            labels.tolist(),
            [-100, -100, -100, -100, 102, 10, 103, 11, -100, -100],
        )
        self.assertEqual(
            pooling_mask.tolist(),
            [True, True, True, True, False, False, False, False, False, False],
        )

    def test_multi_turn_prefix_supervises_only_last_assistant(self):
        input_ids = torch.tensor([
            1, 100, 101, 102, 20, 103, 2, 3,
            100, 101, 102, 30, 31, 103, 11, 0,
        ])
        attention_mask = torch.tensor([1] * 15 + [0])

        labels, pooling_mask = build_target_supervision_masks(
            input_ids,
            attention_mask,
            self.HEADER,
            self.LABEL_PREFIX,
            self.IM_END,
            target_assistant_ordinal=2,
        )

        self.assertTrue(torch.all(labels[:10] == -100))
        self.assertEqual(labels[10:15].tolist(), [102, 30, 31, 103, 11])
        self.assertTrue(torch.all(pooling_mask[:10]))
        self.assertTrue(torch.all(~pooling_mask[10:]))
        self.assertFalse(bool(((labels != -100) & pooling_mask).any()))

    def test_joint_multiturn_supervises_assistants_but_not_users(self):
        input_ids = torch.tensor([
            1, 2, 100, 101, 102, 20, 21, 103, 3, 4,
            100, 101, 102, 30, 103, 11, 0,
        ])
        attention_mask = torch.tensor([1] * 16 + [0])

        labels, pooling_mask = build_joint_assistant_supervision_masks(
            input_ids,
            attention_mask,
            self.HEADER,
            self.LABEL_PREFIX,
            self.IM_END,
        )

        expected = [-100] * 17
        expected[4:8] = [102, 20, 21, 103]
        expected[12:15] = [102, 30, 103]
        self.assertEqual(labels.tolist(), expected)
        self.assertEqual(
            pooling_mask.tolist(),
            [True, True, True, True] + [False] * 13,
        )
        self.assertFalse(bool(((labels != -100) & pooling_mask).any()))

    def test_joint_multiturn_keeps_visible_truncated_assistant_prefix(self):
        input_ids = torch.tensor([1, 100, 101, 102, 20, 21])
        attention_mask = torch.tensor([1] * 6)

        labels, pooling_mask = build_joint_assistant_supervision_masks(
            input_ids,
            attention_mask,
            self.HEADER,
            self.LABEL_PREFIX,
            self.IM_END,
        )

        self.assertEqual(
            labels.tolist(),
            [-100, -100, -100, 102, 20, 21],
        )
        self.assertEqual(
            pooling_mask.tolist(),
            [True, True, True, False, False, False],
        )

    def test_joint_multiturn_does_not_label_truncated_user_tail(self):
        input_ids = torch.tensor([
            1, 100, 101, 102, 20, 103, 2, 3, 4,
        ])
        attention_mask = torch.tensor([1] * 9)

        labels, _ = build_joint_assistant_supervision_masks(
            input_ids,
            attention_mask,
            self.HEADER,
            self.LABEL_PREFIX,
            self.IM_END,
        )

        self.assertEqual(
            labels.tolist(),
            [-100, -100, -100, 102, 20, 103, -100, -100, -100],
        )

    def test_joint_policy_keeps_one_full_conversation_sample(self):
        conversations = [
            {"from": "human", "value": "<image>\nQ1"},
            {"from": "gpt", "value": "A1"},
            {"from": "human", "value": "Q2"},
            {"from": "gpt", "value": "A2"},
        ]
        dataset = object.__new__(FourViewMMRLDataset)
        dataset.ce_enabled = True
        dataset.assistant_turn_policy = "joint"
        dataset.enable_views = {"expert-mm"}
        dataset._resolve_img = lambda item, is_expert: "/tmp/image.jpg"

        samples = dataset._build_views_from_item(
            {
                "image": "image.jpg",
                "conversations": conversations,
                "__source_json_path": "/tmp/1conv_c.json",
            },
            "expert",
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["conversations"], conversations)
        self.assertEqual(samples[0]["raw_turn_count"], 4)

    def test_missing_target_header_fails(self):
        with self.assertRaisesRegex(ValueError, "header is missing"):
            build_target_supervision_masks(
                torch.tensor([1, 2, 3, 0]),
                torch.tensor([1, 1, 1, 0]),
                self.HEADER,
                self.LABEL_PREFIX,
                self.IM_END,
            )

    def test_truncated_target_response_fails(self):
        with self.assertRaisesRegex(ValueError, "response is incomplete"):
            build_target_supervision_masks(
                torch.tensor([1, 100, 101, 102, 30, 31]),
                torch.tensor([1, 1, 1, 1, 1, 1]),
                self.HEADER,
                self.LABEL_PREFIX,
                self.IM_END,
            )

    def test_truncated_later_assistant_header_does_not_reuse_prior_turn(self):
        with self.assertRaisesRegex(
            ValueError,
            "expected_assistant_headers=2, found=1",
        ):
            build_target_supervision_masks(
                torch.tensor([1, 100, 101, 102, 20, 103, 2, 3, 4]),
                torch.tensor([1] * 9),
                self.HEADER,
                self.LABEL_PREFIX,
                self.IM_END,
                target_assistant_ordinal=2,
            )

    def test_expands_one_prefix_per_assistant_turn(self):
        conversations = [
            {"from": "human", "value": "Q1"},
            {"from": "gpt", "value": "A1"},
            {"from": "human", "value": "Q2"},
            {"from": "assistant", "value": "A2"},
        ]

        prefixes = expand_conversation_by_assistant_turn(conversations)

        self.assertEqual([turn_idx for turn_idx, _ in prefixes], [1, 3])
        self.assertEqual([len(prefix) for _, prefix in prefixes], [2, 4])
        self.assertEqual(prefixes[0][1][-1]["value"], "A1")
        self.assertEqual(prefixes[1][1][-1]["value"], "A2")

    def test_first_policy_keeps_only_first_assistant_prefix(self):
        conversations = [
            {"from": "human", "value": "Q1"},
            {"from": "gpt", "value": "A1"},
            {"from": "human", "value": "Q2"},
            {"from": "assistant", "value": "A2"},
        ]

        prefixes = expand_conversation_by_assistant_turn(
            conversations,
            policy="first",
        )

        self.assertEqual(len(prefixes), 1)
        self.assertEqual(prefixes[0][0], 1)
        self.assertEqual([turn["value"] for turn in prefixes[0][1]], ["Q1", "A1"])

    def test_rejects_unknown_assistant_turn_policy(self):
        with self.assertRaisesRegex(ValueError, "Unsupported assistant turn policy"):
            expand_conversation_by_assistant_turn([], policy="random")

    def test_compact_target_window_keeps_latest_exchange_and_rehomes_image(self):
        conversations = [
            {"from": "human", "value": "<image>\nQ1"},
            {"from": "gpt", "value": "A1"},
            {"from": "human", "value": "Q2"},
            {"from": "gpt", "value": "A2"},
            {"from": "human", "value": "Q3"},
            {"from": "gpt", "value": "A3"},
        ]

        compact = build_compact_target_window(conversations, require_image=True)

        self.assertEqual(len(compact), 2)
        self.assertEqual(compact[0]["from"], "human")
        self.assertEqual(compact[0]["value"], "<image>\nQ3")
        self.assertEqual(compact[1]["value"], "A3")
        self.assertEqual(conversations[0]["value"], "<image>\nQ1")

    def test_compact_target_window_requires_preceding_user(self):
        with self.assertRaisesRegex(ValueError, "no preceding user"):
            build_compact_target_window(
                [{"from": "gpt", "value": "A1"}],
                require_image=True,
            )


if __name__ == "__main__":
    unittest.main()
