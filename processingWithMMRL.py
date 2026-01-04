#todo:限制原始图片token生成的最大值？（可能不需要）
from typing import Optional, Union

import numpy as np
from transformers import BatchFeature
from transformers.image_utils import ImageInput
from transformers.models.qwen3_vl import processing_qwen3_vl
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput


class Qwen3VLImageOnlyProcessorKwargs(processing_qwen3_vl.ProcessingKwargs, total=False):
    images_kwargs: processing_qwen3_vl.Qwen3VLImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }

class Qwen3VLImageOnlyProcessor(processing_qwen3_vl.Qwen3VLProcessor):
    attributes = ["image_processor", "tokenizer"]
    video_processor_class = None

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer, video_processor=None, **kwargs)

    def __call__(
            self,
            images: ImageInput = None,
            text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
            videos: VideoInput = None,
            **kwargs: Unpack[Qwen3VLImageOnlyProcessorKwargs],
    ) -> BatchFeature:

        if videos is not None:
            raise ValueError("暂不支持视频输入，请移除 `videos` 参数。")

        output_kwargs = self._merge_kwargs(
            Qwen3VLImageOnlyProcessorKwargs,
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
                    # 计算需要的占位符数量
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # 文本 Tokenize
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        # 仅检查图像模态
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
