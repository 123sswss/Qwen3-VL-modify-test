import contextlib
import importlib.util
import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SCRIPT = PROJECT_ROOT / "test" / "test.py"


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


class _LiveModelInterface:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def infer(self, image, prompt_text, max_new_tokens=256, temperature=0.0):
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
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=temperature,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
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
    model.eval()

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            tee = _Tee(os.sys.stdout, log_file)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                print(f"[EPOCH-EVAL] log_path={log_path}")
                return evaluator.run_evaluation(
                    evaluator.DEFAULT_JSON_PATHS,
                    _LiveModelInterface(model, processor),
                    evaluator.DEFAULT_IMAGE_DIRS,
                )
    finally:
        if was_training:
            model.train()
        torch.cuda.empty_cache()
