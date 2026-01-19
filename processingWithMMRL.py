from typing import Optional, Union

import numpy as np
from transformers import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorKwargs, Qwen3VLProcessor

import config as modcfg

class Qwen3ProcessorWithMMRL(Qwen3VLProcessor):
    attributes = ["image_processor", "tokenizer"]
    video_processor_class = None

    def __init__(self, image_processor=None,
                 tokenizer=None,
                 cfg = None,
                 **kwargs):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, video_processor=None, **kwargs)
        self.rep_tokens = [f"<|REP_placeholder{i}|>" for i in range(len(modcfg.INSERT_LAYER))]
        self.rep_type_id = 3
        self.rep_token_ids = tokenizer.convert_tokens_to_ids(self.rep_tokens)

    def apply_chat_template(self,
                            conversation,
                            chat_template=None,
                            tokenize=True,
                            return_tensors=None,
                            **kwargs
    ):
        if chat_template is None:
            chat_template = self.tokenizer.chat_template
        prompt = super().apply_chat_template(
            conversation,
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=kwargs.get("add_generation_prompt", False)
        )
        rep_str = "".join(self.rep_tokens)
        if isinstance(prompt, list):
            prompt = [rep_str + p for p in prompt]
        elif isinstance(prompt, str):
            prompt = rep_str + prompt
        if tokenize:
            return self.tokenizer(
                prompt,
                return_tensors=return_tensors,
                **kwargs
            )
        else:
            return prompt

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

        ######## mmrl ########
        if not isinstance(text, list):
            text = [text]
        text = text.copy()
        rep_str = "".join(self.rep_tokens)
        text = [rep_str + t for t in text] # [rep_placeholder*40, ...]
        ######## mmrl ########
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            #####
            mask = np.isin(array_ids, self.rep_token_ids)
            mm_token_type_ids[mask] = self.rep_type_id
            #####
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        if video_sizes is not None:
            raise ValueError("暂不支持视频输入，请移除 `videos` 参数。")
        return super()._get_num_multimodal_tokens(image_sizes=image_sizes, video_sizes=None, **kwargs)
