from operator import is_
import os
import json
import random
from torch.utils.data import Dataset
from PIL import Image

class MixedMMRLDataset(Dataset):
    def __init__(self, processor,
                 expert_json, expert_img_dir,
                 general_json, general_img_dir,
                 general_ratio_limit=1.0,
                 expert_limit=None):
        self.processor = processor
        self.general_ratio_limit = general_ratio_limit

        # 1. 加载专家数据 JSON
        self.expert_data_raw = self._load_json(expert_json, "专业")
        if expert_limit is not None and expert_limit < len(self.expert_data_raw):
            self.expert_data_raw = random.sample(self.expert_data_raw, expert_limit)
            print(f"[Dataset] 专业数据集限制为 {expert_limit} 条（随机采样）")

        # 2. 处理专家图片目录
        self.expert_img_mapping, self.expert_img_dir = self._process_img_dirs(expert_img_dir, "专业")
        self.use_expert_mapping = self.expert_img_mapping is not None

        # 3. 加载通用数据 JSON
        self.general_data_raw = self._load_json(general_json, "通用")

        # 4. 处理通用图片目录 (新增逻辑)
        self.general_img_mapping, self.general_img_dir = self._process_img_dirs(general_img_dir, "通用")
        self.use_general_mapping = self.general_img_mapping is not None

        # 5. 初始化数据列表
        self._build_data_list()

    def _load_json(self, json_input, tag):
        """通用JSON加载逻辑"""
        data = []
        if isinstance(json_input, list):
            for path in json_input:
                with open(path, 'r', encoding='utf-8') as f:
                    data.extend(json.load(f))
            print(f"[Dataset] 合并了 {len(json_input)} 个 {tag} JSON文件，共 {len(data)} 条数据")
        else:
            with open(json_input, 'r', encoding='utf-8') as f:
                data = json.load(f)
        return data

    def _process_img_dirs(self, img_input, tag):
        """通用图片目录处理逻辑：支持单路径或多路径列表"""
        if isinstance(img_input, list):
            mapping = {}
            seen_files = {}
            for img_dir in img_input:
                if not os.path.exists(img_dir):
                    print(f"[Warning] {tag}图片目录不存在: {img_dir}")
                    continue
                files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
                for filename in files:
                    if filename in seen_files:
                        raise ValueError(
                            f"{tag}文件名冲突！'{filename}' 同时存在于:\n"
                            f"  - {seen_files[filename]}\n"
                            f"  - {img_dir}"
                        )
                    seen_files[filename] = img_dir
                    mapping[filename] = os.path.join(img_dir, filename)
            print(f"[Dataset] 合并了 {len(img_input)} 个 {tag}图片文件夹，共 {len(mapping)} 张图片")
            return mapping, None
        else:
            return None, img_input

    def _build_data_list(self):
        """修正后的逻辑：先筛选有效数据池，再按比例抽样"""
        self.data_list = []
        
        # 1. 筛选出所有有效的专业数据
        valid_experts = []
        expert_skipped = 0
        for item in self.expert_data_raw:
            image_file = item.get("image", "")
            exists = False
            if self.use_expert_mapping:
                if image_file in self.expert_img_mapping:
                    exists = True
            else:
                if os.path.exists(os.path.join(self.expert_img_dir, image_file)): # 修正了变量名
                    exists = True
            
            if exists:
                valid_experts.append(item)
            else:
                expert_skipped += 1

        # 2. 筛选出所有有效的通用数据池
        valid_generals_pool = []
        general_skipped_in_pool = 0
        for item in self.general_data_raw:
            image_file = item.get("image", "")
            exists = False
            if self.use_general_mapping:
                if image_file in self.general_img_mapping:
                    exists = True
            else:
                if os.path.exists(os.path.join(self.general_img_dir, image_file)): # 修正了变量名
                    exists = True
            
            if exists:
                valid_generals_pool.append(item)
            else:
                general_skipped_in_pool += 1

        # 3. 基于有效专家数据的数量，从有效通用数据池中进行抽样
        num_expert = len(valid_experts)
        max_general_needed = int(num_expert * self.general_ratio_limit)
        
        # 执行抽样
        sampled_generals = random.sample(
            valid_generals_pool, 
            min(max_general_needed, len(valid_generals_pool))
        )

        # 4. 组装最终的 data_list
        for item in valid_experts:
            self.data_list.append({"data": item, "type": "expert", "alpha_label": 1.0})
        
        for item in sampled_generals:
            self.data_list.append({"data": item, "type": "general", "alpha_label": 0.0})

        random.shuffle(self.data_list)
        
        # 打印统计
        print(f"\n[Dataset Status] 数据过滤与平衡完成:")
        print(f"  - 有效数据总数: {len(self.data_list)}")
        print(f"  - 专业数据: {len(valid_experts)} (因图片缺失跳过 {expert_skipped})")
        print(f"  - 通用数据: {len(sampled_generals)} (从 {len(valid_generals_pool)} 条有效数据中采样，原始缺失 {general_skipped_in_pool})")
        if len(sampled_generals) < max_general_needed:
            print(f"  [Warning] 通用数据储备不足，当前实际比例为 1 : {len(sampled_generals)/len(valid_experts):.2f}")

    def resample_general_data(self):
        self._build_data_list()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item_wrapper = self.data_list[idx]
        item = item_wrapper["data"]
        data_type = item_wrapper["type"]
        alpha_label = item_wrapper["alpha_label"]
        image_file = item.get("image")

        # 核心修改：动态获取图片路径
        if data_type == "expert":
            if self.use_expert_mapping:
                image_path = self.expert_img_mapping.get(image_file)
            else:
                image_path = os.path.join(self.expert_img_dir, image_file)
        else:  # general
            if self.use_general_mapping:
                image_path = self.general_img_mapping.get(image_file)
            else:
                image_path = os.path.join(self.general_img_dir, image_file)

        if image_path is None or not os.path.exists(image_path):
            raise FileNotFoundError(f"无法在{data_type}路径中找到图片: {image_file}")

        image = Image.open(image_path).convert("RGB")

        # 后续 Qwen 格式处理逻辑保持一致
        conversations = item.get("conversations")

        if not conversations:
            print(f"警告：索引 {idx} 的对话内容为空！")
            return self.__getitem__((idx + 1) % len(self.data_list))

        # 检查是否每一条 value 都是空字符串
        all_empty = True
        for turn in conversations:
            if turn.get("value", "").strip():
                all_empty = False
                break
        if all_empty:
            print(f"警告：索引 {idx} 的文本内容全是空格或为空！")
            return self.__getitem__((idx + 1) % len(self.data_list))

        qwen_conv = []
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            content = turn["value"]
            if role == "user" and "检测到：" in content:
                # 建议先保留 <image> 标签，再处理文字
                has_image_tag = "<image>" in content
                content = content.split("检测到：")[0]
                if has_image_tag and "<image>" not in content:
                    content = "<image>\n" + content # 补回来

            if "<image>" in content:
                content = content.replace("<image>", "")
                msg_content = [{"type": "image", "image": image}, {"type": "text", "text": content.strip()}]
            else:
                msg_content = [{"type": "text", "text": content}]
            qwen_conv.append({"role": role, "content": msg_content})

        text_inputs = self.processor.apply_chat_template(qwen_conv, tokenize=False, add_generation_prompt=False)

        
        inputs = self.processor(
            images=image,
            text=text_inputs,
            padding="max_length",
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        image_grid_thw = inputs["image_grid_thw"].squeeze(0)
        
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)

        labels = input_ids.clone()
        assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_positions = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0]

        if len(assistant_positions) >= 2:
            assistant_start = assistant_positions[1].item() + 2
            labels[:assistant_start] = -100
        
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
            "alpha_labels": float(alpha_label),
            "images_per_sample": 1
        }