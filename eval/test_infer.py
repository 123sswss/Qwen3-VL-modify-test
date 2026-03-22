# infer.py
import os
import json
import random
from typing import Optional, List, Dict, Any

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoConfig,
    AutoTokenizer,
    AutoImageProcessor,
    GenerationConfig,
)

import config as cfg
import QWen3WithMMRL
import processingWithMMRL
import MMRLGating


# =============================================================================
# 1. 模型包装器
# =============================================================================
class Qwen3VLMMRLForGen(Qwen3VLForConditionalGeneration):
    def __init__(self, config, tokenizer):
        import torch.nn as nn
        nn.Module.__init__(self)
        self.config = config

        current_vocab_size = len(tokenizer)
        self.model = QWen3WithMMRL.QWen3WithMMRL(config, tokenizer=tokenizer)

        hidden_size = config.text_config.hidden_size
        self.lm_head = nn.Linear(hidden_size, current_vocab_size, bias=False)

        self.generation_config = GenerationConfig.from_model_config(config)
        if tokenizer.pad_token_id is not None:
            self.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            self.generation_config.eos_token_id = tokenizer.eos_token_id

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


# =============================================================================
# 2. 基础工具
# =============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pretty(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2)


def to_list_safe(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.item()
        return x.tolist()
    return x


# =============================================================================
# 3. 加载模型
# =============================================================================
def load_model_and_processor(
    trained_model_path: str,
    base_model_path: str,
    device: torch.device,
):
    print("[1/4] 加载 tokenizer / config / image_processor ...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.add_special_tokens(cfg.SPECIAL_TOKENS)
    print(f"    -> tokenizer vocab size: {len(tokenizer)}")

    try:
        config = AutoConfig.from_pretrained(trained_model_path, trust_remote_code=True)
        print("    -> 使用训练目录 config")
    except Exception:
        config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        print("    -> 训练目录 config 缺失，回退到基座 config")

    image_processor = AutoImageProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    print("[2/4] 构建模型架构 ...")
    model = Qwen3VLMMRLForGen(config, tokenizer)

    print(f"[3/4] 加载权重: {trained_model_path}")
    safetensors_path = os.path.join(trained_model_path, "model.safetensors")
    bin_path = os.path.join(trained_model_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"在 {trained_model_path} 下未找到权重文件")

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"    -> Missing keys: {len(msg.missing_keys)}")
    print(f"    -> Unexpected keys: {len(msg.unexpected_keys)}")
    if len(msg.missing_keys) > 0:
        print(f"       sample missing keys: {msg.missing_keys[:8]}")
    if len(msg.unexpected_keys) > 0:
        print(f"       sample unexpected keys: {msg.unexpected_keys[:8]}")

    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device, dtype=torch.float32)

    model.eval()

    processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
        image_processor=image_processor,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    return model, processor, tokenizer


# =============================================================================
# 4. 消融 Hook
# =============================================================================
def attach_force_alpha_hook(model, force_alpha_prob: Optional[float] = None):
    """
    强制 Task_classifier 输出 alpha logits
    force_alpha_prob:
        None / 0.0 / 1.0
    """
    handles = []
    if force_alpha_prob is None:
        return handles

    assert force_alpha_prob in [0.0, 1.0]
    forced_logit = -20.0 if force_alpha_prob == 0.0 else 20.0

    def _hook(module, inputs, output):
        return torch.full_like(output, forced_logit)

    count = 0
    for m in model.modules():
        if isinstance(m, MMRLGating.Task_classifier):
            handles.append(m.register_forward_hook(_hook))
            count += 1

    print(f"    -> alpha hook 已启用: prob={force_alpha_prob}, logit={forced_logit}, modules={count}")
    return handles


def attach_force_k_hook(model, force_k: Optional[int] = None):
    """
    正确强制 K：
    直接覆盖 text_gating 的输出 mask [B, total_rep_num]
    这样后续 k_hard / k_sums / gate_soft_mask / attention_mask / inputs_embeds
    都会按真实逻辑执行。
    """
    handles = []
    if force_k is None:
        return handles

    def _hook(module, inputs, output):
        # output:
        #   eval: hard_k_logits [B, T]
        #   train: (hard_k_logits, tax_loss)
        if isinstance(output, tuple):
            gate, tax_loss = output
        else:
            gate, tax_loss = output, None

        if not isinstance(gate, torch.Tensor) or gate.dim() != 2:
            return output

        B, T = gate.shape
        k = max(0, min(int(force_k), T))

        forced_gate = torch.zeros_like(gate)
        if k > 0:
            forced_gate[:, :k] = 1.0

        if tax_loss is None:
            return forced_gate
        else:
            return forced_gate, tax_loss * 0.0

    count = 0
    for m in model.modules():
        if isinstance(m, MMRLGating.textGating):
            handles.append(m.register_forward_hook(_hook))
            count += 1

    print(f"    -> K hook 已启用: force_k={force_k}, modules={count}")
    return handles


# =============================================================================
# 5. LLM 入口调试器
# =============================================================================
class LLMInputDebugger:
    """
    目标：
    1) 记录 QWen3WithMMRL.forward 进入时的 input_ids
    2) 记录 language_model.forward 前真正收到的 inputs_embeds / attention_mask
    3) 统计 placeholder 中实际有多少个被激活
    """
    def __init__(self, qwen_mmrl_model, rep_placeholder_ids: List[int]):
        self.qwen_mmrl_model = qwen_mmrl_model
        self.rep_placeholder_ids = torch.tensor(rep_placeholder_ids, dtype=torch.long)
        self.handles = []
        self.latest_input_ids = None
        self.latest_debug = {}

    def clear(self):
        self.latest_input_ids = None
        self.latest_debug = {}

    def attach(self):
        # 1) 在 QWen3WithMMRL.forward 入口抓 input_ids
        def _top_prehook(module, args, kwargs):
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            self.latest_input_ids = input_ids.detach().clone() if isinstance(input_ids, torch.Tensor) else None

        self.handles.append(
            self.qwen_mmrl_model.register_forward_pre_hook(_top_prehook, with_kwargs=True)
        )

        # 2) 在 language_model.forward 前抓最终送入 LLM 的 inputs_embeds / attention_mask
        def _lm_prehook(module, args, kwargs):
            inputs_embeds = kwargs.get("inputs_embeds", None)
            attention_mask = kwargs.get("attention_mask", None)
            position_ids = kwargs.get("position_ids", None)

            debug = {}

            if isinstance(inputs_embeds, torch.Tensor):
                debug["llm_inputs_embeds_shape"] = list(inputs_embeds.shape)
                debug["llm_inputs_embeds_dtype"] = str(inputs_embeds.dtype)

            if isinstance(attention_mask, torch.Tensor):
                debug["llm_attention_mask_shape"] = list(attention_mask.shape)
                debug["llm_attention_mask_dtype"] = str(attention_mask.dtype)

            if isinstance(position_ids, torch.Tensor):
                debug["llm_position_ids_shape"] = list(position_ids.shape)

            input_ids = self.latest_input_ids
            if input_ids is not None and isinstance(inputs_embeds, torch.Tensor):
                rep_ids = self.rep_placeholder_ids.to(input_ids.device)
                is_placeholder = (input_ids.unsqueeze(-1) == rep_ids).any(dim=-1)  # [B, L]

                debug["placeholder_total_per_sample"] = is_placeholder.sum(dim=-1).detach().cpu().tolist()

                if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2:
                    active_placeholder = is_placeholder & attention_mask.bool()
                    inactive_placeholder = is_placeholder & (~attention_mask.bool())

                    debug["placeholder_active_per_sample"] = active_placeholder.sum(dim=-1).detach().cpu().tolist()
                    debug["placeholder_inactive_per_sample"] = inactive_placeholder.sum(dim=-1).detach().cpu().tolist()

                    # 再看 active placeholder embedding 的范数，确认不是“mask开了但embed是0”
                    embed_norm = torch.norm(inputs_embeds.float(), dim=-1)  # [B, L]
                    active_norms = []
                    inactive_norms = []
                    for b in range(inputs_embeds.shape[0]):
                        act = embed_norm[b][active_placeholder[b]]
                        ina = embed_norm[b][inactive_placeholder[b]]
                        active_norms.append(act.detach().cpu().tolist())
                        inactive_norms.append(ina.detach().cpu().tolist())

                    debug["active_placeholder_embed_norms"] = active_norms
                    debug["inactive_placeholder_embed_norms"] = inactive_norms

                # 再检查所有 placeholder 位置 embedding 是否非零
                placeholder_embed_nonzero = []
                embed_norm = torch.norm(inputs_embeds.float(), dim=-1)
                for b in range(inputs_embeds.shape[0]):
                    vals = embed_norm[b][is_placeholder[b]]
                    placeholder_embed_nonzero.append((vals > 1e-8).sum().item())
                debug["placeholder_nonzero_embed_count_per_sample"] = placeholder_embed_nonzero

            self.latest_debug = debug

        self.handles.append(
            self.qwen_mmrl_model.language_model.register_forward_pre_hook(_lm_prehook, with_kwargs=True)
        )

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# =============================================================================
# 6. 构造输入
# =============================================================================
def build_inputs(processor, model, image_path: str, prompt_text: str):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None

    inputs = processor(
        text=[text_prompt],
        images=image,
        padding=False,
        max_length=False,
        truncation=False,
        return_tensors="pt",
    ).to(model.device)

    return inputs, text_prompt


# =============================================================================
# 7. 单次实验
# =============================================================================
@torch.no_grad()
def run_one_experiment(
    model,
    processor,
    tokenizer,
    debugger: LLMInputDebugger,
    image_path: str,
    prompt_text: str,
    force_alpha: Optional[float],
    force_k: Optional[int],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
):
    hook_handles = []
    hook_handles += attach_force_alpha_hook(model, force_alpha)
    hook_handles += attach_force_k_hook(model, force_k)

    debugger.clear()

    try:
        inputs, text_prompt = build_inputs(processor, model, image_path, prompt_text)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

        input_len = inputs.input_ids.shape[1]
        output_ids = generated_ids[:, input_len:]
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        result = {
            "prompt": prompt_text,
            "text_prompt": text_prompt,
            "output_text": output_text,
            "debug": {}
        }

        # alpha debug
        if hasattr(model.model.visual, "alpha_list") and model.model.visual.alpha_list is not None:
            alpha_logits = model.model.visual.alpha_list
            alpha_probs = torch.sigmoid(alpha_logits)
            result["debug"]["alpha_logits"] = to_list_safe(alpha_logits.squeeze())
            result["debug"]["alpha_probs"] = to_list_safe(alpha_probs.squeeze())
            result["debug"]["alpha_mean"] = float(alpha_probs.mean().item())

        # k debug
        if hasattr(model.model, "k_results") and model.model.k_results is not None:
            result["debug"]["k_results"] = to_list_safe(model.model.k_results)

        # LLM入口debug
        result["debug"]["llm_input_debug"] = debugger.latest_debug

        # 再补充原始输入信息
        result["debug"]["input_ids_shape"] = list(inputs.input_ids.shape)
        if hasattr(inputs, "attention_mask") and isinstance(inputs.attention_mask, torch.Tensor):
            result["debug"]["processor_attention_mask_shape"] = list(inputs.attention_mask.shape)

        return result

    finally:
        for h in hook_handles:
            h.remove()


# =============================================================================
# 8. main
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 路径
    # -------------------------------------------------------------------------
    TRAINED_MODEL_PATH = "/root/autodl-tmp/Qwen3-VL-modify-test/mmrl_output"
    BASE_MODEL_PATH = "/root/autodl-tmp/model"

    IMAGE_PATH = "/root/autodl-tmp/Qwen3-VL-modify-test/test02.jpg"
    PROMPT_TEXT = "描述一下这张图片。"

    # -------------------------------------------------------------------------
    # 生成设置
    # -------------------------------------------------------------------------
    MAX_NEW_TOKENS = 256
    DO_SAMPLE = True
    TEMPERATURE = 0.2

    # -------------------------------------------------------------------------
    # 实验列表
    # 你可以自由删减
    # -------------------------------------------------------------------------
    # EXPERIMENTS = [
    #     {"name": "baseline",      "alpha": None, "k": None},
    #     {"name": "force_alpha_0", "alpha": 0.0,  "k": None},
    #     {"name": "force_alpha_1", "alpha": 1.0,  "k": None},
    #     {"name": "force_k_0",     "alpha": None, "k": 0},
    #     {"name": "force_k_10",    "alpha": None, "k": 10},
    #     {"name": "force_k_20",    "alpha": None, "k": 20},
    #     {"name": "force_k_40",    "alpha": None, "k": 40},
    # ]
    # EXPERIMENTS = [
    #     {"name": "baseline",      "alpha": None, "k": None},
    #     {"name": "force_alpha_1", "alpha": 1.0,  "k": None},
    #     {"name": "force_k_10",    "alpha": None, "k": 10},
    #     {"name": "force_k_20",    "alpha": None, "k": 20},
    #     {"name": "force_k_40",    "alpha": None, "k": 40},
    # ]
    EXPERIMENTS = [
        {"name": "alpha_1_k_0",  "alpha": 1.0, "k": 0},
        {"name": "alpha_1_k_40",  "alpha": 1.0, "k": 40},
    ]

    # 如果你想只跑一个实验，就改成：
    # EXPERIMENTS = [{"name": "force_k_10", "alpha": None, "k": 10}]

    # -------------------------------------------------------------------------
    # 初始化
    # -------------------------------------------------------------------------
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    model, processor, tokenizer = load_model_and_processor(
        trained_model_path=TRAINED_MODEL_PATH,
        base_model_path=BASE_MODEL_PATH,
        device=device,
    )

    # debugger：真正检查 LLM 入口
    debugger = LLMInputDebugger(
        qwen_mmrl_model=model.model,
        rep_placeholder_ids=model.model.rep_placeholder_ids,
    )
    debugger.attach()

    print("[4/4] 开始实验 ...")

    all_results = []
    for exp in EXPERIMENTS:
        print("\n" + "=" * 100)
        print(f"[Experiment] {exp['name']} | alpha={exp['alpha']} | k={exp['k']}")

        result = run_one_experiment(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            debugger=debugger,
            image_path=IMAGE_PATH,
            prompt_text=PROMPT_TEXT,
            force_alpha=exp["alpha"],
            force_k=exp["k"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
        )

        all_results.append({
            "name": exp["name"],
            "alpha": exp["alpha"],
            "k": exp["k"],
            **result
        })

        print("-" * 100)
        print(f"Output:\n{result['output_text']}")
        print("-" * 100)
        print("[Debug summary]")

        dbg = result["debug"]

        if "alpha_mean" in dbg:
            print(f"  alpha_mean: {dbg['alpha_mean']:.6f}")
        if "k_results" in dbg:
            print(f"  k_results: {dbg['k_results']}")

        llm_dbg = dbg.get("llm_input_debug", {})
        if llm_dbg:
            print(f"  llm_inputs_embeds_shape: {llm_dbg.get('llm_inputs_embeds_shape')}")
            print(f"  llm_attention_mask_shape: {llm_dbg.get('llm_attention_mask_shape')}")
            print(f"  placeholder_total_per_sample: {llm_dbg.get('placeholder_total_per_sample')}")
            print(f"  placeholder_active_per_sample: {llm_dbg.get('placeholder_active_per_sample')}")
            print(f"  placeholder_inactive_per_sample: {llm_dbg.get('placeholder_inactive_per_sample')}")
            print(f"  placeholder_nonzero_embed_count_per_sample: {llm_dbg.get('placeholder_nonzero_embed_count_per_sample')}")

            active_norms = llm_dbg.get("active_placeholder_embed_norms", None)
            if active_norms is not None:
                print(f"  active_placeholder_embed_norms(sample0): {active_norms[0] if len(active_norms) > 0 else active_norms}")

        # 如果你想看完整debug，把下面打开
        print("-" * 100)
        print(pretty(dbg))

    debugger.remove()

    print("\n全部实验完成。")


if __name__ == "__main__":
    main()