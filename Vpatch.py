# 1. 输入：User: [全局图] + "寻找框内 <|box_start|>[y1,x1,y2,x2]<|box_end|> 的细节并回答 [问题]"
# 2. 局部增强：Crop(ROI) -> ViT -> IVTP(基于问题计算相关性得分) -> 筛选 K 个动态特征向量
# 3. 占位符：构造序列 `<|detail_start|>` + (`<|v_patch|>` * K) + `<|detail_end|>` 插入原 Prompt
# 4. 嵌入：在 Embedding 层将所有 `<|v_patch|>` 文本向量替换为 IVTP 筛选出的视觉特征向量
# 5. 模型推理：LLM 在一次 Forward 中结合全局低分辨率信息与局部 K 个高价值特征完成回答
## 用处：当人指定区域让ai细看时，ai可以根据区域内的细节回答问题
from PIL import Image
import torch
from torch import nn
import config as cfg
import similarity as sim

def _get_patch_from_origin_pic(image_path, bbox):
    """
    :param image_path: 原始图片路径
    :param bbox: 区域坐标 xyxy
    :return: 裁剪出的区域图片
    """
    origin_pic = Image.open(image_path)
    roi_image  = origin_pic.crop(bbox)
    return roi_image

def split_vision_features(hidden_states, deepstack_feature_lists, grid_thw, spatial_merge_size):
    """
    拆分 Qwen3VL 视觉编码器的输出，提取第一张图特征，保留剩余 N 张图。

    Args:
        hidden_states: 形状为 (Total_Merged_Tokens, hidden_size) 的 Tensor
        deepstack_feature_lists: 包含多个中间层特征的列表，每个 Tensor 形状同上
        grid_thw: 形状为 (1+N, 3)，记录每张图的 (T, H, W)
        spatial_merge_size: 空间合并系数，默认为 2 (即 2x2 合并)

    Returns:
        first_image: (hidden_states_0, deepstack_features_0)
        remaining_images: (hidden_states_N, deepstack_features_N)
    """
    merged_lengths = (
            grid_thw[:, 0] *
            (grid_thw[:, 1] // spatial_merge_size) *
            (grid_thw[:, 2] // spatial_merge_size)
    )
    first_img_len = merged_lengths[0].item()
    hidden_states_first = hidden_states[:first_img_len]
    hidden_states_rem = hidden_states[first_img_len:]
    deepstack_first = []
    deepstack_rem = []
    for layer_feat in deepstack_feature_lists:
        deepstack_first.append(layer_feat[:first_img_len])
        deepstack_rem.append(layer_feat[first_img_len:])
    return (hidden_states_first, deepstack_first), (hidden_states_rem, deepstack_rem)


class Vpatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.similarity = sim.IVTP_similarity()

    def forward(self,
                image_hidden_states: torch.Tensor,
                deepstack_feature_lists: list[torch.Tensor],  # 修正标注
                input_embeds: torch.Tensor,
                grid_thw: torch.Tensor,
                spatial_merge_size: int):
        global_vision, partial_vision = split_vision_features(image_hidden_states,
                                                              deepstack_feature_lists,
                                                              grid_thw,
                                                              spatial_merge_size)
        hidden_states, deepstack = partial_vision
        score = self.similarity(hidden_states, input_embeds)
        topk_values, topk_indices = torch.topk(score, k, sorted=False)
        topk_indices, _ = torch.sort(topk_indices)  # 保持原始时空顺序非常重要
        filtered_hidden_states = hidden_states[topk_indices]
        filtered_deepstack = [layer_feat[topk_indices] for layer_feat in deepstack]
        filtered_grid_thw = torch.tensor([[1, 1, k]],
                                         dtype=grid_thw.dtype,
                                         device=grid_thw.device)
        new_grid_thw = torch.cat([grid_thw[:1], filtered_grid_thw], dim=0)
        gh, gd = global_vision  # Global Hidden, Global Deepstack
        merged_hidden = torch.cat((gh, filtered_hidden_states), dim=0)
        merged_deepstack = []
        for g_layer, p_layer in zip(gd, filtered_deepstack):
            merged_deepstack.append(torch.cat([g_layer, p_layer], dim=0))
        return merged_hidden, merged_deepstack, new_grid_thw







