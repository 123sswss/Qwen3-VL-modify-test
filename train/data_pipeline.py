# data_pipeline.py
import os
import re
import json
import random
from collections import Counter
from typing import List, Dict, Any, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset


VISUAL_HINT_PATTERNS = [
    r"根据图片",
    r"图中所示",
    r"如图",
    r"从图中",
    r"看图",
    r"观察图片",
    r"这张图片",
    r"图里",
    r"图片中",
]


def clean_text_visual_hints(text: str) -> str:
    t = text
    for p in VISUAL_HINT_PATTERNS:
        t = re.sub(p, "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_semantic_collapsed(text: str, min_len: int = 4) -> bool:
    if text is None:
        return True
    txt = text.strip()
    if len(txt) < min_len:
        return True
    return False


def load_jsons(json_input):
    data = []
    if isinstance(json_input, list):
        for path in json_input:
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)
                for item in items:
                    if isinstance(item, dict):
                        item = dict(item)
                        item["__source_json_path"] = path
                    data.append(item)
    else:
        with open(json_input, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                if isinstance(item, dict):
                    item = dict(item)
                    item["__source_json_path"] = json_input
                data.append(item)
    return data


TASK_TYPE_TO_ID = {
    "report": 0,
    "vqa": 1,
    "test": 2,
    "unknown": 3,
}


def normalize_conversation_role(raw_role: Any) -> str:
    role = str(raw_role or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "bot"}:
        return "assistant"
    if role == "system":
        return "system"
    raise ValueError(f"Unsupported conversation role: {raw_role!r}")


def expand_conversation_by_assistant_turn(conversations):
    """Create one causal prefix per assistant response."""
    prefixes = []
    for turn_idx, turn in enumerate(conversations):
        if normalize_conversation_role(turn.get("from")) == "assistant":
            prefixes.append((turn_idx, conversations[:turn_idx + 1]))
    return prefixes


def build_compact_target_window(conversations, require_image=False):
    """Keep the current user/assistant exchange when earlier context overflows."""
    if not conversations:
        raise ValueError("cannot compact an empty conversation")

    target_idx = len(conversations) - 1
    if normalize_conversation_role(conversations[target_idx].get("from")) != "assistant":
        raise ValueError("target conversation must end with an assistant turn")

    user_idx = None
    for turn_idx in range(target_idx - 1, -1, -1):
        if normalize_conversation_role(conversations[turn_idx].get("from")) == "user":
            user_idx = turn_idx
            break
    if user_idx is None:
        raise ValueError("target assistant turn has no preceding user turn")

    compact = [dict(turn) for turn in conversations[user_idx:target_idx + 1]]
    if require_image:
        for turn in compact:
            turn["value"] = str(turn.get("value", "")).replace("<image>", "")
        compact[0]["value"] = "<image>\n" + str(compact[0].get("value", "")).lstrip()
    return compact


def _find_subsequence_positions(input_ids, pattern_ids, valid_length):
    if not pattern_ids or valid_length < len(pattern_ids):
        return []

    first_id = int(pattern_ids[0])
    candidates = (input_ids[:valid_length] == first_id).nonzero(as_tuple=True)[0].tolist()
    positions = []
    for start in candidates:
        end = start + len(pattern_ids)
        if end <= valid_length and input_ids[start:end].tolist() == list(pattern_ids):
            positions.append(start)
    return positions


def build_target_supervision_masks(
    input_ids,
    attention_mask,
    assistant_header_ids,
    assistant_label_prefix_ids,
    im_end_token_id,
    target_assistant_ordinal=1,
):
    """Build current-response labels and its strictly causal text-pooling mask."""
    active_positions = (attention_mask != 0).nonzero(as_tuple=True)[0]
    if active_positions.numel() == 0:
        raise ValueError("tokenized sample has no active tokens")

    valid_length = int(active_positions[-1].item()) + 1
    expected_active = torch.arange(valid_length, device=active_positions.device)
    if not torch.equal(active_positions, expected_active):
        raise ValueError("only right-padded attention masks are supported")

    header_positions = _find_subsequence_positions(
        input_ids,
        assistant_header_ids,
        valid_length,
    )
    expected_count = int(target_assistant_ordinal)
    if expected_count < 1:
        raise ValueError(f"invalid target assistant ordinal: {expected_count}")
    if len(header_positions) != expected_count:
        raise ValueError(
            "target assistant header is missing after truncation "
            f"(expected_assistant_headers={expected_count}, "
            f"found={len(header_positions)}, valid_tokens={valid_length})"
        )

    header_start = header_positions[-1]
    if (
        not assistant_label_prefix_ids
        or assistant_header_ids[:len(assistant_label_prefix_ids)]
        != list(assistant_label_prefix_ids)
    ):
        raise ValueError("assistant label prefix must match the assistant header")
    target_start = header_start + len(assistant_label_prefix_ids)
    target_tokens = input_ids[target_start:valid_length]
    if not bool((target_tokens == int(im_end_token_id)).any()):
        raise ValueError(
            "target assistant response is incomplete after truncation "
            f"(target_start={target_start}, valid_tokens={valid_length})"
        )

    labels = torch.full_like(input_ids, -100)
    # The target assistant is the final turn in this prefix, so the remaining
    # active tail contains only its response, terminator, and template separator.
    labels[target_start:valid_length] = input_ids[target_start:valid_length]

    mmrl_gating_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    mmrl_gating_mask[:target_start] = attention_mask[:target_start].bool()
    return labels, mmrl_gating_mask


def infer_task_type_from_source(source_json_path: Optional[str]) -> str:
    if not source_json_path:
        return "unknown"
    base = os.path.basename(source_json_path).lower()
    stem = os.path.splitext(base)[0]

    if "conversation" in stem or "llava" in stem:
        return "vqa"
    if "conv_c" in stem:
        return "vqa"
    if "test" in stem:
        return "test"
    if stem.endswith("json") or re.search(r"\d+json($|[._-])", stem):
        return "report"
    return "unknown"


def build_image_mapping(img_input):
    if isinstance(img_input, list):
        mapping = {}
        for d in img_input:
            if not os.path.exists(d):
                continue
            for fn in os.listdir(d):
                fp = os.path.join(d, fn)
                if os.path.isfile(fp):
                    mapping[fn] = fp
        return mapping, None
    else:
        return None, img_input


class FourViewMMRLDataset(Dataset):
    """
    每条样本可构造四视图：
    expert-mm / expert-text / general-mm / general-text

    total_limit 语义：
    - 当同时启用 expert/general 视图时，total_limit 表示“每一侧”的 raw sample 配额；
      例如 total_limit=20000，会采 20000 expert + 20000 general。
    - 当只启用 expert 或只启用 general 时，total_limit 表示该单侧的 raw sample 配额。
    """

    def __init__(
        self,
        processor,
        expert_json,
        expert_img_dir,
        general_json,
        general_img_dir,
        total_limit=50000,
        enable_views=("expert-mm", "expert-text", "general-mm", "general-text"),
        mode="stage1_cls",  # stage1_cls / stage3_joint
        ce_enabled=False,
        seed=42,
        deterministic_sampling=False,
    ):
        self.processor = processor
        self.total_limit = total_limit
        self.enable_views = set(enable_views)
        self.mode = mode
        self.seed = seed
        self.deterministic_sampling = deterministic_sampling
        self.resample_round = 0
        self.ce_enabled = ce_enabled
        self.max_length = 1024

        tokenizer = self.processor.tokenizer
        self.assistant_header_ids = tokenizer.encode(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        self.assistant_label_prefix_ids = tokenizer.encode(
            "<|im_start|>assistant",
            add_special_tokens=False,
        )
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if (
            not self.assistant_header_ids
            or not self.assistant_label_prefix_ids
            or self.im_end_token_id is None
        ):
            raise RuntimeError("Failed to resolve Qwen assistant boundary tokens")

        self.expert_raw = load_jsons(expert_json) if expert_json else []
        self.general_raw = load_jsons(general_json) if general_json else []

        self.expert_map, self.expert_dir = build_image_mapping(expert_img_dir)
        self.general_map, self.general_dir = build_image_mapping(general_img_dir)

        self.data = []
        self._build()

    def _encode_conversation_for_length(self, conversations, view_type, image_path):
        image = None
        if view_type == "mm" and image_path is not None:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        qwen_conv = self._build_qwen_conv(conversations, image)
        text_inputs = self.processor.apply_chat_template(
            qwen_conv,
            tokenize=False,
            add_generation_prompt=False,
        )
        processor_kwargs = {
            "text": text_inputs,
            "padding": False,
            "truncation": False,
            "return_tensors": "pt",
        }
        if view_type == "mm":
            processor_kwargs["images"] = image
        encoded = self.processor(**processor_kwargs)
        return (
            encoded["input_ids"].squeeze(0),
            encoded["attention_mask"].squeeze(0),
        )

    def _target_fits_after_compaction(self, sample, view_type, image_path):
        if not self.ce_enabled:
            return True, ""

        compact_conv = build_compact_target_window(
            sample["conversations"],
            require_image=(view_type == "mm" and image_path is not None),
        )
        compact_target_ordinal = sum(
            normalize_conversation_role(turn.get("from")) == "assistant"
            for turn in compact_conv
        )
        compact_ids, compact_mask = self._encode_conversation_for_length(
            compact_conv,
            view_type=view_type,
            image_path=image_path,
        )
        valid_length = int((compact_mask != 0).sum().item())
        if valid_length > self.max_length:
            return False, f"compact_valid_tokens={valid_length}"

        try:
            build_target_supervision_masks(
                input_ids=compact_ids,
                attention_mask=compact_mask,
                assistant_header_ids=self.assistant_header_ids,
                assistant_label_prefix_ids=self.assistant_label_prefix_ids,
                im_end_token_id=self.im_end_token_id,
                target_assistant_ordinal=compact_target_ordinal,
            )
        except ValueError as exc:
            return False, str(exc)
        return True, ""

    def _resolve_img(self, item, is_expert: bool) -> Optional[str]:
        image_file = item.get("image", "")
        if is_expert:
            if self.expert_map is not None:
                return self.expert_map.get(image_file, None)
            return os.path.join(self.expert_dir, image_file) if self.expert_dir else None
        else:
            if self.general_map is not None:
                return self.general_map.get(image_file, None)
            return os.path.join(self.general_dir, image_file) if self.general_dir else None

    def _valid_item(self, item, is_expert: bool) -> bool:
        p = self._resolve_img(item, is_expert)
        return (p is not None) and os.path.exists(p) and (item.get("conversations") is not None)

    def _build_views_from_item(self, item, task_type: str):
        is_expert = task_type == "expert"
        img_path = self._resolve_img(item, is_expert)
        conv = item.get("conversations", [])
        source_json_path = item.get("__source_json_path")
        source_name = os.path.basename(source_json_path) if source_json_path else "unknown"
        dataset_task_type = infer_task_type_from_source(source_json_path)
        dataset_task_id = TASK_TYPE_TO_ID.get(dataset_task_type, TASK_TYPE_TO_ID["unknown"])

        mm_key = f"{task_type}-mm"
        text_key = f"{task_type}-text"

        if self.ce_enabled:
            conversation_variants = expand_conversation_by_assistant_turn(conv)
            if not conversation_variants:
                raise ValueError(
                    "CE sample contains no assistant turn: "
                    f"source={source_name} image={item.get('image', '')!r}"
                )
        else:
            conversation_variants = [(None, conv)]

        out = []
        for target_assistant_ordinal, (target_turn_index, target_conv) in enumerate(
            conversation_variants,
            start=1,
        ):
            common = {
                "task_type": task_type,
                "dataset_task_type": dataset_task_type,
                "dataset_task_id": dataset_task_id,
                "source_name": source_name,
                "target_turn_index": target_turn_index,
                "target_assistant_ordinal": target_assistant_ordinal,
                "raw_turn_count": len(conv),
                "alpha_label": 1.0 if is_expert else 0.0,
            }

            if mm_key in self.enable_views:
                sample = {
                    **common,
                    "view_type": "mm",
                    "image_path": img_path,
                    "conversations": target_conv,
                }
                fits, reason = self._target_fits_after_compaction(sample, "mm", img_path)
                if fits:
                    out.append(sample)
                else:
                    self.skipped_overlong_targets += 1
                    self.skipped_overlong_target_sources[source_name] += 1
                    self.skipped_overlong_target_examples.append({
                        "source": source_name,
                        "image": img_path,
                        "target_turn_index": target_turn_index,
                        "target_assistant_ordinal": target_assistant_ordinal,
                        "raw_turn_count": len(conv),
                        "reason": reason,
                    })

            if text_key in self.enable_views:
                cleaned_conv = []
                for t in target_conv:
                    v = t.get("value", "")
                    v = v.replace("<image>", "")
                    v = clean_text_visual_hints(v)
                    cleaned_conv.append({"from": t.get("from", "human"), "value": v})

                merged_text = " ".join([x["value"] for x in cleaned_conv])
                if not is_semantic_collapsed(merged_text):
                    sample = {
                        **common,
                        "view_type": "text",
                        "image_path": None,
                        "conversations": cleaned_conv,
                    }
                    fits, reason = self._target_fits_after_compaction(sample, "text", None)
                    if fits:
                        out.append(sample)
                    else:
                        self.skipped_overlong_targets += 1
                        self.skipped_overlong_target_sources[source_name] += 1
                        self.skipped_overlong_target_examples.append({
                            "source": source_name,
                            "image": None,
                            "target_turn_index": target_turn_index,
                            "target_assistant_ordinal": target_assistant_ordinal,
                            "raw_turn_count": len(conv),
                            "reason": reason,
                        })
        return out

    def _build(self):
        self.skipped_overlong_targets = 0
        self.skipped_overlong_target_sources = Counter()
        self.skipped_overlong_target_examples = []
        if self.deterministic_sampling:
            rng = random.Random(self.seed + self.resample_round)
        else:
            rng = random
        expert_pool = [x for x in self.expert_raw if self._valid_item(x, True)]
        general_pool = [x for x in self.general_raw if self._valid_item(x, False)]

        use_expert = any(v.startswith("expert-") for v in self.enable_views)
        use_general = any(v.startswith("general-") for v in self.enable_views)

        if not use_expert and not use_general:
            raise ValueError("enable_views must contain at least one expert-* or general-* view")

        expert_quota = self.total_limit if use_expert else 0
        general_quota = self.total_limit if use_general else 0

        e_samples = rng.sample(expert_pool, min(expert_quota, len(expert_pool))) if expert_quota > 0 else []
        g_samples = rng.sample(general_pool, min(general_quota, len(general_pool))) if general_quota > 0 else []

        for item in e_samples:
            self.data.extend(self._build_views_from_item(item, "expert"))
        for item in g_samples:
            self.data.extend(self._build_views_from_item(item, "general"))

        rng.shuffle(self.data)
        print(
            "[FourViewMMRLDataset] "
            f"mode={self.mode} expert_selected={len(e_samples)} general_selected={len(g_samples)} "
            f"total view samples={len(self.data)}"
        )
        if self.ce_enabled:
            print(
                "[TARGET_LENGTH_FILTER] "
                f"max_length={self.max_length} "
                f"skipped_overlong_targets={self.skipped_overlong_targets} "
                f"sources={dict(self.skipped_overlong_target_sources)}"
            )
            for example in self.skipped_overlong_target_examples[:5]:
                print(f"[TARGET_LENGTH_FILTER_EXAMPLE] {example}")

    def __len__(self):
        return len(self.data)
    
    def resample_data(self):
        self.resample_round += 1
        self.data = []
        self._build()

    def _build_qwen_conv(self, conversations, image_obj_or_none):
        qwen_conv = []
        for turn in conversations:
            role = normalize_conversation_role(turn.get("from"))
            content = turn.get("value", "")
            if role == "user" and "检测到：" in content:
                has_image_tag = "<image>" in content
                content = content.split("检测到：")[0]
                if has_image_tag and "<image>" not in content:
                    content = "<image>\n" + content

            if "<image>" in content:
                content = content.replace("<image>", "")
                if image_obj_or_none is not None:
                    msg_content = [{"type": "image", "image": image_obj_or_none},
                                   {"type": "text", "text": content.strip()}]
                else:
                    msg_content = [{"type": "text", "text": content.strip()}]
            else:
                msg_content = [{"type": "text", "text": content.strip()}]

            qwen_conv.append({"role": role, "content": msg_content})
        return qwen_conv

    def __getitem__(self, idx):
        s = self.data[idx]
        alpha_label = s["alpha_label"]
        view_type = s["view_type"]
        image = None
        if s["image_path"] is not None:
            image = Image.open(s["image_path"]).convert("RGB")

        def encode_conversation(conversations):
            qwen_conv = self._build_qwen_conv(conversations, image)
            text_inputs = self.processor.apply_chat_template(
                qwen_conv, tokenize=False, add_generation_prompt=False
            )
            processor_kwargs = {
                "text": text_inputs,
                "padding": "max_length",
                "max_length": 1024,
                "truncation": True,
                "return_tensors": "pt",
            }
            if view_type == "mm":
                processor_kwargs["images"] = image
            encoded = self.processor(**processor_kwargs)
            return (
                encoded,
                encoded["input_ids"].squeeze(0),
                encoded["attention_mask"].squeeze(0),
            )

        target_assistant_ordinal = s.get("target_assistant_ordinal", 1)
        inputs, input_ids, attention_mask = encode_conversation(s["conversations"])

        if self.ce_enabled:
            try:
                labels, mmrl_gating_mask = build_target_supervision_masks(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    assistant_header_ids=self.assistant_header_ids,
                    assistant_label_prefix_ids=self.assistant_label_prefix_ids,
                    im_end_token_id=self.im_end_token_id,
                    target_assistant_ordinal=target_assistant_ordinal,
                )
            except ValueError as initial_exc:
                try:
                    compact_conv = build_compact_target_window(
                        s["conversations"],
                        require_image=(view_type == "mm" and image is not None),
                    )
                    if len(compact_conv) >= len(s["conversations"]):
                        raise initial_exc
                    inputs, input_ids, attention_mask = encode_conversation(compact_conv)
                    compact_target_ordinal = sum(
                        normalize_conversation_role(turn.get("from")) == "assistant"
                        for turn in compact_conv
                    )
                    labels, mmrl_gating_mask = build_target_supervision_masks(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        assistant_header_ids=self.assistant_header_ids,
                        assistant_label_prefix_ids=self.assistant_label_prefix_ids,
                        im_end_token_id=self.im_end_token_id,
                        target_assistant_ordinal=compact_target_ordinal,
                    )
                except ValueError as compact_exc:
                    raise RuntimeError(
                        "[TARGET_MASK_ERROR] "
                        f"idx={idx} source={s.get('source_name')} "
                        f"image={s.get('image_path')} "
                        f"target_turn_index={s.get('target_turn_index')} "
                        f"target_assistant_ordinal={target_assistant_ordinal} "
                        f"raw_turn_count={s.get('raw_turn_count')}: "
                        f"initial={initial_exc}; compact={compact_exc}"
                    ) from compact_exc
        else:
            labels = input_ids.clone()
            labels[:] = -100
            mmrl_gating_mask = attention_mask.bool()

        if view_type == "mm":
            # pixel_values: [num_patches, patch_dim], image_grid_thw: [num_images, 3]
            pixel_values = inputs["pixel_values"].squeeze(0) if inputs["pixel_values"].dim() == 3 else inputs["pixel_values"]
            image_grid_thw = inputs["image_grid_thw"]
            if image_grid_thw.dim() == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)
            images_per_sample = 1
            is_mm = 1
        else:
            pixel_values = None
            image_grid_thw = None
            images_per_sample = 0
            is_mm = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
            "mmrl_gating_mask": mmrl_gating_mask,
            "alpha_labels": float(alpha_label),
            "task_type_id": int(s.get("dataset_task_id", TASK_TYPE_TO_ID["unknown"])),
            "task_type_name": s.get("dataset_task_type", "unknown"),
            "source_name": s.get("source_name", "unknown"),
            "images_per_sample": images_per_sample,
            "is_mm": is_mm,
        }


class MMRLDataCollator:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        alpha_labels = [f.pop("alpha_labels") for f in features]
        task_type_ids = [f.pop("task_type_id") for f in features]
        task_type_names = [f.pop("task_type_name") for f in features]
        source_names = [f.pop("source_name") for f in features]
        images_per_sample = [f.pop("images_per_sample") for f in features]
        is_mm = [f.pop("is_mm") for f in features]

        # 单独处理 pixel_values 和 image_grid_thw（变长，仅 mm 样本有值）
        pixel_values_list = [f.pop("pixel_values") for f in features]
        image_grid_thw_list = [f.pop("image_grid_thw") for f in features]

        mm_pixels = [pv for pv in pixel_values_list if pv is not None]
        mm_grids = [g for g in image_grid_thw_list if g is not None]

        batch = {}
        for key in features[0].keys():
            if torch.is_tensor(features[0][key]):
                batch[key] = torch.stack([f[key] for f in features], dim=0)
            else:
                batch[key] = [f[key] for f in features]

        # pixel_values: cat all mm patches -> [total_patches, patch_dim]
        # image_grid_thw: cat all mm grids -> [num_mm_images, 3]
        if mm_pixels:
            batch["pixel_values"] = torch.cat(mm_pixels, dim=0)
            batch["image_grid_thw"] = torch.cat(mm_grids, dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        batch["alpha_labels"] = torch.tensor(alpha_labels, dtype=torch.float32)
        batch["task_type_ids"] = torch.tensor(task_type_ids, dtype=torch.long)
        batch["task_type_names"] = task_type_names
        batch["source_names"] = source_names
        batch["images_per_sample"] = images_per_sample
        batch["is_mm"] = torch.tensor(is_mm, dtype=torch.long)
        return batch
