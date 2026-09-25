from typing import Optional, Union

import numpy as np
from transformers import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorKwargs, Qwen3VLProcessor
import torch

class Qwen3ProcessorWithMMRL(Qwen3VLProcessor):
    attributes = ["image_processor", "tokenizer"]
    video_processor_class = None

    def __init__(self, image_processor=None,
                 tokenizer=None,
                 cfg = None,
                 **kwargs):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, video_processor=None, **kwargs)
        self.chat_template = tokenizer.chat_template

    def __call__(
            self,
            images: ImageInput = None,
            text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
            videos: VideoInput = None,
            **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> BatchFeature:

        if videos is not None:
            raise ValueError("暂不支持视频输入，请移除 `videos` 参数。")

        if text is not None and isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict):
            text = self.apply_chat_template(text, tokenize=False, **kwargs)
        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        videos_inputs = {}

        if not isinstance(text, list):
            text = [text]

        text = text.copy()

        # 仅保留图像占位符替换逻辑
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size ** 2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        if video_sizes is not None:
            raise ValueError("暂不支持视频输入，请移除 `videos` 参数。")
        return super()._get_num_multimodal_tokens(image_sizes=image_sizes, video_sizes=None, **kwargs)


def reserve_v3_postvisual_slots(inputs, *, vision_start_id: int, vision_end_id: int,
                                image_token_id: int, pad_token_id: int, merge_size: int):
    """Reserve 20 embedding-only positions after Qwen's *expanded* image block.

    This happens after native image expansion and tokenization, so the chat
    template, original question tokenization, image placeholders and grid stay
    byte-for-byte unchanged. The returned prompt_mask is the model's sole
    authority for static and dynamic replacement.
    """
    ids = inputs["input_ids"]
    attention = inputs["attention_mask"]
    grid = inputs.get("image_grid_thw")
    if not torch.is_tensor(ids) or ids.ndim != 2 or attention.shape != ids.shape:
        raise ValueError("V3 processor requires return_tensors='pt' and batched IDs/mask")
    if grid is None or grid.shape != (ids.shape[0], 3):
        raise ValueError("V3 PathVQA requires exactly one image grid per sample")
    if int(merge_size) != 2:
        raise ValueError("V3 requires Qwen3-VL's audited 2x2 spatial merger")
    length = 20
    rows, masks, prompt_masks, boundaries = [], [], [], []
    for row in range(ids.shape[0]):
        valid = attention[row].bool()
        starts = torch.nonzero(ids[row].eq(vision_start_id) & valid, as_tuple=True)[0]
        ends = torch.nonzero(ids[row].eq(vision_end_id) & valid, as_tuple=True)[0]
        if starts.numel() != 1 or ends.numel() != 1:
            raise ValueError(f"V3 row {row}: expected one complete image start/end pair")
        start, end = int(starts.item()), int(ends.item())
        if end <= start + 1 or end + 1 >= int(valid.sum()):
            raise ValueError(f"V3 row {row}: image end is not before the question/chat suffix")
        image_positions = torch.nonzero(ids[row].eq(image_token_id) & valid, as_tuple=True)[0]
        expected = int(grid[row].prod().item()) // (merge_size ** 2)
        if expected <= 0 or image_positions.numel() != expected or not torch.equal(
            image_positions, torch.arange(start + 1, end, device=ids.device)
        ):
            raise ValueError(f"V3 row {row}: image placeholders/grid not one contiguous block")
        insertion = end + 1
        rows.append(torch.cat((ids[row, :insertion], ids.new_full((length,), pad_token_id), ids[row, insertion:])))
        masks.append(torch.cat((attention[row, :insertion], attention.new_ones((length,)), attention[row, insertion:])))
        prompt = torch.zeros(ids.shape[1] + length, dtype=torch.bool, device=ids.device)
        prompt[insertion:insertion + length] = True
        prompt_masks.append(prompt)
        boundaries.append({"row": row, "vision_start": start, "image_tokens": expected,
                           "vision_end": end, "prompt_start": insertion,
                           "prompt_end_exclusive": insertion + length,
                           "original_valid_tokens": int(valid.sum())})
    expanded = dict(inputs)
    expanded["input_ids"] = torch.stack(rows)
    expanded["attention_mask"] = torch.stack(masks)
    expanded["prompt_mask"] = torch.stack(prompt_masks)
    return expanded, boundaries


class Qwen3ProcessorWithV3(Qwen3ProcessorWithMMRL):
    """The existing project processor plus V3's embedding-only slot mask."""

    def __call__(self, *args, **kwargs):
        native = super().__call__(*args, **kwargs)
        expanded, _ = reserve_v3_postvisual_slots(
            native,
            vision_start_id=self.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            vision_end_id=self.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
            image_token_id=self.image_token_id,
            pad_token_id=int(self.tokenizer.pad_token_id),
            merge_size=int(self.image_processor.merge_size),
        )
        return BatchFeature(data=expanded, tensor_type="pt")
