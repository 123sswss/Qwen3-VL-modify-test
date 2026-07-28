import contextlib
import importlib.util
import os
from pathlib import Path
from types import MethodType

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SCRIPT = PROJECT_ROOT / "test" / "test.py"
LORATEST_DIR = PROJECT_ROOT / "loraTest"
if str(LORATEST_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(LORATEST_DIR))

from generation_timing import generate_with_timing


class MissingGateError(RuntimeError):
    fatal_evaluation_error = True


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _load_test_module():
    spec = importlib.util.spec_from_file_location("mmrl_test_evaluator", TEST_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载测评脚本: {TEST_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight_live_epoch_evaluation():
    evaluator = _load_test_module()
    counts = evaluator.validate_evaluation_data(
        evaluator.DEFAULT_JSON_PATHS,
        evaluator.DEFAULT_IMAGE_DIRS,
    )
    if counts.get("valid", 0) == 0:
        raise RuntimeError("逐 epoch 测评数据预检失败：没有任何可评测样本")
    return counts


def run_professional_gate_preflight(model, processor):
    evaluator = _load_test_module()
    dataset = evaluator.load_combined_dataset(evaluator.DEFAULT_JSON_PATHS)
    if not dataset:
        raise RuntimeError("Stage 1 gate preflight has no evaluation samples")

    was_training = model.training
    model.eval()
    visual = model.model.visual
    device = next(model.parameters()).device
    alpha_logits = []

    print(
        "[STAGE1_GATE_PREFLIGHT] begin "
        f"total={len(dataset)} expected_gate=1"
    )
    try:
        for index, item in enumerate(dataset, start=1):
            item_id = item.get("id", f"unknown_{index - 1}")
            image_file = item.get("image", "")
            source_json = item.get("_source_json")

            prompt_text = None
            for turn in item.get("conversations", []):
                if str(turn.get("from", "")).lower() in {"human", "user"}:
                    prompt_text = str(turn.get("value", ""))
            if prompt_text is None:
                raise RuntimeError(
                    f"Gate preflight sample has no user prompt: id={item_id}"
                )
            prompt_text = prompt_text.replace("<image>\n", "").replace("<image>", "")

            image_path = evaluator.resolve_image_path(
                image_file,
                evaluator.DEFAULT_IMAGE_DIRS,
                source_json_path=source_json,
            )
            if image_path is None:
                raise RuntimeError(
                    "Gate preflight image is missing: "
                    f"id={item_id} image={image_file}"
                )

            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            text_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=[text_prompt],
                images=image,
                padding=False,
                max_length=False,
                truncation=False,
                return_tensors="pt",
            ).to(device)

            pixel_values = inputs["pixel_values"]
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.squeeze(0)
            image_grid_thw = inputs["image_grid_thw"]
            if image_grid_thw.dim() == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)

            with torch.no_grad():
                text_embedding = model.model.get_input_embeddings()(
                    inputs["input_ids"]
                ).to(dtype=visual.dtype)
                text_pooling_mask = model.model.build_mmrl_text_pooling_mask(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    audit_context="stage1_preflight",
                )
                text_pooled = visual.embedding_pooling(
                    text_embedding,
                    mask=text_pooling_mask,
                )
                vision_pooled = visual.extract_gate_pooled_vision(
                    pixel_values,
                    image_grid_thw,
                )
                alpha = visual.Task_classifier(vision_pooled, text_pooled)
                raw_gate = visual.visionGating(alpha)
                hard_gate = raw_gate > 0.5

            alpha_logit = float(alpha.detach().float().reshape(-1)[0].item())
            alpha_prob = float(
                torch.sigmoid(alpha.detach().float()).reshape(-1)[0].item()
            )
            alpha_logits.append(alpha_logit)
            gate_value = int(hard_gate.detach().reshape(-1)[0].item())

            if gate_value != 1:
                print(
                    "[STAGE1_GATE_PREFLIGHT_FAIL] "
                    f"index={index}/{len(dataset)} id={item_id} "
                    f"G={gate_value} alpha_logit={alpha_logit:.6f} "
                    f"alpha_prob={alpha_prob:.6f} image={image_file}"
                )
                raise RuntimeError(
                    "Stage 1 professional gate preflight failed; "
                    "Stage 3 was not started"
                )
            if index == 1 or index % 100 == 0 or index == len(dataset):
                print(
                    "[STAGE1_GATE_PREFLIGHT] "
                    f"checked={index}/{len(dataset)} G=1 "
                    f"alpha_logit={alpha_logit:.6f} "
                    f"alpha_prob={alpha_prob:.6f} image={image_file}"
                )

        print(
            "[STAGE1_GATE_PREFLIGHT_PASS] "
            f"checked={len(alpha_logits)} gate_ones={len(alpha_logits)} "
            f"alpha_logit_min={min(alpha_logits):.6f} "
            f"alpha_logit_max={max(alpha_logits):.6f}"
        )
    finally:
        if was_training:
            model.train()
        torch.cuda.empty_cache()


class _LiveModelInterface:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.last_gate_value = None
        self.last_generation_timing = None

    def infer(self, image, prompt_text, max_new_tokens=256, temperature=0.0):
        self.last_gate_value = None
        self.last_generation_timing = None
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }]
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if hasattr(self.model.model, "rope_deltas"):
            self.model.model.rope_deltas = None

        device = next(self.model.parameters()).device
        inputs = self.processor(
            text=[text_prompt],
            images=image,
            padding=False,
            max_length=False,
            truncation=False,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids, self.last_generation_timing = generate_with_timing(
                self.model,
                inputs,
                {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "pad_token_id": self.processor.tokenizer.pad_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                    "use_cache": True,
                },
            )
        gate = getattr(self.model.model.visual, "G_list", None)
        if not torch.is_tensor(gate) or gate.numel() == 0:
            raise MissingGateError(
                "Live evaluation did not produce a valid visual G_list"
            )
        gate_values = gate.detach().float().reshape(-1).cpu().tolist()
        self.last_gate_value = (
            gate_values[0] if len(gate_values) == 1 else gate_values
        )
        input_len = inputs.input_ids.shape[1]
        return self.processor.tokenizer.decode(
            generated_ids[0, input_len:],
            skip_special_tokens=True,
        )


def run_live_epoch_evaluation(model, processor, log_path):
    evaluator = _load_test_module()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    training_forward = model.forward
    outer_temperature = getattr(model, "temperature_override", None)
    inner_temperature = getattr(model.model, "temperature_override", None)

    # 独立 inferEngine 使用原生生成 forward，而不是训练包装器的 loss/监控 forward。
    model.forward = MethodType(Qwen3VLForConditionalGeneration.forward, model)
    model.temperature_override = None
    model.model.temperature_override = None
    model.eval()

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            tee = _Tee(os.sys.stdout, log_file)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                print(f"[EPOCH-EVAL] log_path={log_path}")
                summary = evaluator.run_evaluation(
                    evaluator.DEFAULT_JSON_PATHS,
                    _LiveModelInterface(model, processor),
                    evaluator.DEFAULT_IMAGE_DIRS,
                )
                if summary.get("evaluated", 0) == 0:
                    raise RuntimeError("逐 epoch 测评没有成功完成任何题目，请检查上方 EVAL-ERROR")
                return summary
    finally:
        model.forward = training_forward
        model.temperature_override = outer_temperature
        model.model.temperature_override = inner_temperature
        if was_training:
            model.train()
        torch.cuda.empty_cache()
