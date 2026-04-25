# data_pipeline.py
import os
import re
import json
import random
from typing import List, Dict, Any, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset

# ===== 新增：数据集分组 =====
DATASET_GROUP_TO_ID = {
    "general": 0,  # 通用
    "report": 1,   # *json（固定格式报告）
    "vqa": 2,      # *conv
    "test": 3,     # *test（做题适配）
}
ID_TO_DATASET_GROUP = {v: k for k, v in DATASET_GROUP_TO_ID.items()}


def _normalize_dataset_name(path: str) -> str:
    name = os.path.basename(path or "").lower()
    name = name.replace(".translated.json", ".json")
    name = name.replace(".translated", "")
    return name


def infer_dataset_group(source_json_path: str, is_expert: bool) -> str:
    if not is_expert:
        return "general"
    n = _normalize_dataset_name(source_json_path)
    # 优先 test，再 conv，剩余归 report
    if "test" in n:
        return "test"
    if "conv" in n:
        return "vqa"
    return "report"


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


def load_jsons(json_input, attach_source=False):
    data = []

    def _append_records(arr, src_path):
        if not isinstance(arr, list):
            return
        if not attach_source:
            data.extend(arr)
            return
        for x in arr:
            if isinstance(x, dict):
                y = dict(x)
                y["_source_json"] = src_path
                data.append(y)

    if isinstance(json_input, list):
        for path in json_input:
            with open(path, "r", encoding="utf-8") as f:
                arr = json.load(f)
            _append_records(arr, path)
    else:
        with open(json_input, "r", encoding="utf-8") as f:
            arr = json.load(f)
        _append_records(arr, json_input)

    return data


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
        mode="stage1_cls",  # stage1_cls / stage2_gate / stage3_joint / stage4_sparse
        ce_enabled=False,
        seed=42,
    ):
        self.processor = processor
        self.total_limit = total_limit
        self.enable_views = set(enable_views)
        self.mode = mode
        random.seed(seed)

        self.expert_raw = load_jsons(expert_json, attach_source=True) if expert_json else []
        self.general_raw = load_jsons(general_json, attach_source=True) if general_json else []

        self.expert_map, self.expert_dir = build_image_mapping(expert_img_dir)
        self.general_map, self.general_dir = build_image_mapping(general_img_dir)

        self.data = []
        self._build()

        self.ce_enabled = ce_enabled

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
        dataset_group = infer_dataset_group(item.get("_source_json", ""), is_expert=is_expert)
        dataset_group_id = DATASET_GROUP_TO_ID.get(dataset_group, 0 if not is_expert else 1)
        conv = item.get("conversations", [])

        mm_key = f"{task_type}-mm"
        text_key = f"{task_type}-text"

        out = []
        if mm_key in self.enable_views:
            out.append({
                "task_type": task_type,
                "view_type": "mm",
                "image_path": img_path,
                "conversations": conv,
                "alpha_label": 1.0 if is_expert else 0.0,
                "dataset_group": dataset_group,
                "dataset_group_id": dataset_group_id,
            })

        if text_key in self.enable_views:
            # text-only 清洗
            cleaned_conv = []
            for t in conv:
                v = t.get("value", "")
                v = v.replace("<image>", "")
                v = clean_text_visual_hints(v)
                cleaned_conv.append({"from": t.get("from", "human"), "value": v})

            merged_text = " ".join([x["value"] for x in cleaned_conv])
            if not is_semantic_collapsed(merged_text):
                out.append({
                    "task_type": task_type,
                    "view_type": "text",
                    "image_path": None,  # text-only
                    "conversations": cleaned_conv,
                    "alpha_label": 1.0 if is_expert else 0.0,
                    "dataset_group": dataset_group,
                    "dataset_group_id": dataset_group_id,
                })
        return out

    def _build(self):
        expert_pool = [x for x in self.expert_raw if self._valid_item(x, True)]
        general_pool = [x for x in self.general_raw if self._valid_item(x, False)]

        half = self.total_limit // 2
        e_samples = random.sample(expert_pool, min(half, len(expert_pool)))
        g_samples = random.sample(general_pool, min(self.total_limit - len(e_samples), len(general_pool)))

        for item in e_samples:
            self.data.extend(self._build_views_from_item(item, "expert"))
        for item in g_samples:
            self.data.extend(self._build_views_from_item(item, "general"))

        random.shuffle(self.data)
        print(f"[FourViewMMRLDataset] total view samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)
    
    def resample_general_data(self):
        self.data = []
        self._build()

    def _build_qwen_conv(self, conversations, image_obj_or_none):
        qwen_conv = []
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
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

        qwen_conv = self._build_qwen_conv(s["conversations"], image)
        text_inputs = self.processor.apply_chat_template(
            qwen_conv, tokenize=False, add_generation_prompt=False
        )

        if view_type == "mm":
            inputs = self.processor(
                images=image,
                text=text_inputs,
                padding="max_length",
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            )
            # pixel_values: [num_patches, patch_dim],  image_grid_thw: [num_images, 3]
            pixel_values = inputs["pixel_values"].squeeze(0) if inputs["pixel_values"].dim() == 3 else inputs["pixel_values"]
            image_grid_thw = inputs["image_grid_thw"]
            if image_grid_thw.dim() == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)
            images_per_sample = 1
            is_mm = 1
        else:
            inputs = self.processor(
                text=text_inputs,
                padding="max_length",
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            )
            pixel_values = None
            image_grid_thw = None
            images_per_sample = 0
            is_mm = 0

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_positions = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0]
        if len(assistant_positions) >= 2:
            assistant_start = assistant_positions[1].item() + 2
            labels[:assistant_start] = -100
        labels[attention_mask == 0] = -100

        if not self.ce_enabled:
            labels[:] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
            "alpha_labels": float(alpha_label),
            "images_per_sample": images_per_sample,
            "is_mm": is_mm,
            "dataset_group_id": s.get("dataset_group_id", 0),
        }


class MMRLDataCollator:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        alpha_labels = [f.pop("alpha_labels") for f in features]
        images_per_sample = [f.pop("images_per_sample") for f in features]
        is_mm = [f.pop("is_mm") for f in features]

        # 单独处理 pixel_values 和 image_grid_thw（变长，仅 mm 样本有值）
        pixel_values_list = [f.pop("pixel_values") for f in features]
        image_grid_thw_list = [f.pop("image_grid_thw") for f in features]
        dataset_group_ids = [f.pop("dataset_group_id", 0) for f in features]

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
        batch["images_per_sample"] = images_per_sample
        batch["is_mm"] = torch.tensor(is_mm, dtype=torch.long)
        batch["dataset_group_ids"] = torch.tensor(dataset_group_ids, dtype=torch.long)
        return batch
