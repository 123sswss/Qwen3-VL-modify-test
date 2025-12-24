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

def _get_patch_from_origin_pic(image_path, bbox):
    """
    :param image_path: 原始图片路径
    :param bbox: 区域坐标 xyxy
    :return: 裁剪出的区域图片
    """
    origin_pic = Image.open(image_path)
    roi_image  = origin_pic.crop(bbox)
    return roi_image

class Vpatch(nn.Module):
    def __init__(self,):
        super().__init__()
        #todo:动态K值的阈值τ,K max,相似度计算算法
        #todo:vpatch token缓存
        # ============================================================
        # 阶段 1: 前置准备 (Preprocessing)
        # ============================================================
        # 1. 接收原始输入：一张高清大图 I_orig，用户画的框 B_list，以及原始问题 P
        # 2. 图像裁切：利用 PIL 根据 B_list 从 I_orig 中截取多个高清 ROI 小图 [I_roi_1, I_roi_2, ...]
        # 3. 构造图像列表：images = [I_orig, I_roi_1, I_roi_2, ...] (多图模式)
        # 4. 调用 Processor：将 images 和 基础 Prompt 喂给 processor，得到：
        #    - pixel_values: 拼接后的像素张量
        #    - image_grid_thw: 记录每张图在序列中的长宽信息
        #    - input_ids: 基础文本的 ID (此时还不包含细节占位符)

    def split_image_hidden_states(self, image_hidden_states) -> list[torch.Tensor]:
        return roi_hidden_states_list

    def q_embedding(self, query: str) -> torch.Tensor:
        pass

    def similarity_calculation(self, roi_hidden_states: torch.Tensor,
                              input_embed: torch.Tensor) -> torch.Tensor:
        return similarity

    def forward(self, image_hidden_states: torch.Tensor,
                input_embeds: torch.Tensor):
        # ============================================================
        # 阶段 2: 视觉特征提取与 IVTP 提炼 (Visual & Pruning)
        # ============================================================
        # 5. 提取 Query 语义：
        #    - 将用户问题 P 转化为 Embedding (query_embed)，作为 IVTP 评分的“筛子”
        # 6. 运行视觉编码器 (model.visual)：
        #    - hidden_states, _ = model.visual(pixel_values, grid_thw)
        #    - 此时 hidden_states 是一个长序列，包含了 [全局图特征, ROI_1特征, ROI_2特征, ...]
        # 7. 调用 IVTPManager.split_features：
        #    - 根据 grid_thw 的索引，将 hidden_states 拆分为 global_feat 和 [roi_feat_1, roi_feat_2, ...]
        # 8. 执行 IVTP 动态剪枝：
        #    - 遍历每个 roi_feat，计算其与 query_embed 的相关性
        #    - 根据 Top-p 或 阈值，选出最关键的 K 个 Token (例如 ROI_1 选了 5 个，ROI_2 选了 3 个)
        #    - 记录下每个 ROI 最终保留的 Token 数量列表: k_list = [5, 3, ...]

        filtered_roi_token_list = []
        ########################## 在经过投影器之后 ##########################
        roi_hidden_states_list = self.split_image_hidden_states(image_hidden_states)
        for roi_hidden_states in roi_hidden_states_list:
            sim_matrix  = self.similarity_calculation(roi_hidden_states, input_embeds)
            similarity = sim_matrix.max(dim=-1)[0]
            filtered_roi_token = roi_hidden_states[similarity.topk(roi_hidden_states.shape[0]*cfg.VPATCH_COMPRESS_RATIO).indices]
            filtered_roi_token_list.append(filtered_roi_token)



    # ============================================================
    # 阶段 3: 动态 Prompt 构造与 Embedding 手术 (Injection)
    # ============================================================
    # 9. 构造最终文本 Prompt：
    #    - 动态生成带占位符的字符串：
    #      "图中区域1的细节是：<|detail_start|>" + ("<|v_patch|>" * 5) + "<|detail_end|>，"
    #      "区域2的细节是：<|detail_start|>" + ("<|v_patch|>" * 3) + "<|detail_end|>"
    #    - 拼接原始问题 P，形成最终文本：final_prompt
    # 10. 文本转 Embedding：
    #    - final_input_ids = tokenizer(final_prompt)
    #    - input_embeds = model.model.embed_tokens(final_input_ids) (获取纯文本层面的词向量)
    # 11. 执行“手术”替换：
    #    - 在 input_embeds 中定位所有 <|v_patch|> 的索引位置
    #    - 将 IVTP 提炼出的那 [5 + 3] 个视觉 Token 向量，按顺序覆盖掉对应的 <|v_patch|> 词向量
    #    - 同时，将全局图的 global_feat 插入到 <|vision_start|> 对应的位置 (Qwen 原生逻辑)
    # ============================================================
    # 阶段 4: LLM 推理 (Final Forward)
    # ============================================================
    # 12. 最终调用 model.model (Transformer 层)：
    #    - 传入 input_embeds (已经融合了全局特征和提炼后的局部细节特征)
    #    - 注意：由于 IVTP 后的 Token 是作为 1D 序列插入的，我们需要为这些 Token 构造
    #      简单的 position_ids 或修改 attention_mask，确保模型知道它们是视觉补充





