########## 这个地方用于放置以后会用到的代码 ##########
########## 这个地方用于放置以后会用到的代码 ##########
########## 这个地方用于放置以后会用到的代码 ##########

# 这个函数用于在数据集文本中添加 R token 占位符
def add_r_token_placeholders(text, image_token_end="<|vision_end|>"):
    # 定义你的特殊 token
    start_token = "<|text_R_token|>"
    end_token = "<|text_R_token_end|>"
    # 核心：生成 40 个占位符
    placeholders = "<|text_R_token_placeholder|>" * 40#替换成真实的trtoken长度

    # 构造完整的注入串： <Start> [40个坑] <End>
    injection_str = f"{start_token}{placeholders}{end_token}"

    # 强制插入到图片结束符后面
    # 注意：这里假设你的文本里已经有了 image 相关的 token
    if image_token_end in text:
        text = text.replace(image_token_end, image_token_end + injection_str)

    return text