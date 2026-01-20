import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoImageProcessor
)
import config as cfg
import QWen3WithMMRL
import processingWithMMRL
from transformers import GenerationConfig

# ==============================================================================
# 1. 定义支持 MMRL Loss 的 Generation 模型包装器
# ==============================================================================
class Qwen3VLMMRLForGen(Qwen3VLForConditionalGeneration):
    def __init__(self, config, tokenizer):
        # 手动初始化 Module，跳过官方的 super().__init__ 以免重复分配内存
        import torch.nn as nn
        nn.Module.__init__(self) 
        self.config = config
        
        # 1. 获取准确的词表大小
        # 注意：你之前加了特殊 token，所以必须用 len(tokenizer)
        current_vocab_size = len(tokenizer)
        
        # 2. 初始化你的魔改 Base Model
        # QWen3WithMMRL 内部已经处理了嵌入层的 resize
        self.model = QWen3WithMMRL.QWen3WithMMRL(config, tokenizer=tokenizer)
        
        # 3. 初始化 LM Head
        # 输出维度必须等于当前的词表大小，否则计算 CrossEntropyLoss 时会下标越界
        hidden_size = config.text_config.hidden_size
        self.lm_head = nn.Linear(hidden_size, current_vocab_size, bias=False)
        # =================【新增修复代码】=================
        # 手动初始化生成配置，否则 generate() 会报错
        self.generation_config = GenerationConfig.from_model_config(config)
        # 确保 pad_token_id 和 eos_token_id 正确
        if tokenizer.pad_token_id is not None:
            self.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            self.generation_config.eos_token_id = tokenizer.eos_token_id
        # =================================================
        # 4. 执行官方的后处理 (比如权重初始化)
        self.post_init()
        
    def get_output_embeddings(self):
        # 必须实现这个方法，否则 Trainer 在某些逻辑下会报错
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 必须实现这个方法
        self.lm_head = new_embeddings
    
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        
        # 1. 获取 Tax Loss (这是稀疏性惩罚，保持原样或者稍微调小权重)
        mmrl_tax_loss = self.model.tax_loss if self.model.tax_loss is not None else 0.0
        
        # 2. 【关键修改】处理 Alpha Loss
        # 原始 self.model.alpha_loss 是为了让 alpha->0 (稀疏激活)
        # 但这是"专家数据"，我们要强制引导 alpha->1 (激活专家模式)
        
        # 获取 alpha 的原始 logits (在 VisionModelWithMMRL 中被保存为 alpha_list)
        # QWen3WithMMRL -> visual -> alpha_list
        # 注意：forward 必须运行过一次，这个 list 才有值
        alpha_logits = self.model.visual.alpha_list
        
        if alpha_logits is not None and isinstance(alpha_logits, torch.Tensor):
            # 计算 alpha 激活后的概率
            alpha_probs = torch.sigmoid(alpha_logits)
            
            # 构造引导 Loss：我们希望概率趋近于 1.0
            # 使用 MSE Loss: mean((1 - prob)^2)
            # 给一个较大的权重 (比如 1.0 或 5.0)，确保它在一开始就迅速打开门控
            alpha_guide_loss = torch.mean((1.0 - alpha_probs) ** 2) * 5.0
        else:
            alpha_guide_loss = 0.0

        # 3. 合并 Loss
        if outputs.loss is not None:
            # 注意：这里我们抛弃了 self.model.alpha_loss，用自定义的 alpha_guide_loss 代替
            total_loss = outputs.loss + mmrl_tax_loss + alpha_guide_loss
            
            outputs.loss = total_loss
            
            # (可选) 打印调试信息，看 Alpha 是否真的变大了
            # if alpha_logits is not None:
            #     print(f"\r[Debug] CE: {outputs.loss.item():.4f} | Alpha_Prob: {alpha_probs.mean().item():.4f} | GuideLoss: {alpha_guide_loss:.4f}", end="")
            
        return outputs

# ==============================================================================
# 2. 构造过拟合数据集
# ==============================================================================
class OverfitDataset(Dataset):
    def __init__(self, processor, image_path, prompt_text, target_text):
        self.processor = processor
        self.image = Image.open(image_path).convert("RGB")
        self.prompt_text = prompt_text
        self.target_text = target_text
        
        # 构造对话
        self.conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.image},
                    {"type": "text", "text": self.prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": self.target_text}]
            }
        ]

    def __len__(self):
        # 伪装成有 1000 个样本，这样一个 epoch 可以多跑几步，方便 Trainer 记录
        return 10

    def __getitem__(self, idx):
        # 这里的处理稍微 trick 一点，我们需要手动构造 labels
        # Qwen3 的 Processor 处理对话时，默认不会生成 labels，我们需要自己处理
        # 简单起见，我们让 Processor 处理整个对话，然后手动 mask 掉 user 部分
        
        text_inputs = self.processor.apply_chat_template(
            self.conversation, tokenize=False, add_generation_prompt=False
        )
        
        # 使用 Processor 生成 inputs
        inputs = self.processor(
            images=self.image,
            text=text_inputs,
            padding="max_length", # 简单起见定长，或者在 Collator 里做
            max_length=2048,
            truncation=True,
            return_tensors="pt",
        )
        
        # 移除 batch 维度 (因为 Dataset 应该返回单个样本)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        image_grid_thw = inputs["image_grid_thw"].squeeze(0)
        
        # 构造 Labels: 
        # 这是一个简单的 mask 逻辑。在实际对话训练中，应该找到 "<|im_start|>assistant" 的位置，
        # 把之前的所有 token 的 label 设为 -100。
        # 这里为了过拟合实验，我们简单粗暴：不做精细 mask，让它全量学习（包括 User prompt），
        # 或者你可以手动找到分割点。为了确保它学会“回答”，全量学习在这个单样本实验中也是可以接受的。
        # 也就是：labels = input_ids.clone()
        labels = input_ids.clone()
        
        # 将 padding 部分设为 -100
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels
        }

# ==============================================================================
# 3. 训练脚本主体
# ==============================================================================
def train_overfit():
    print("=" * 20 + " 启动 MMRL 过拟合实验 " + "=" * 20)
    
    # 路径配置
    MODEL_PATH = "../model/qwen3vl" # 请确保这是真实的预训练模型路径
    IMAGE_PATH = "test.png"   # 请准备一张测试图片
    
    # 如果没有测试图，生成一张红色的 dummy 图
    if not os.path.exists(IMAGE_PATH):
        Image.new('RGB', (512, 512), color='red').save(IMAGE_PATH)

    # 1. 加载 Config 和 Tokenizer
    print("[1/5] 加载配置...")
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # 添加特殊 Token
    tokenizer.add_special_tokens(cfg.SPECIAL_TOKENS)
    
    # 2. 初始化模型与加载权重
    print("[2/5] 启动加速初始化流程...")
    
    # 步骤 A: 依然在 CPU 加载预训练权重 (为了节省显存，不直接用 from_pretrained 进 GPU)
    print("    -> 正在从磁盘读取预训练权重到 CPU (Standard Qwen3-VL)...")
    original_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="cpu", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    original_model.resize_token_embeddings(len(tokenizer))

    # 步骤 B: 使用 GPU 上下文，瞬间完成魔改架构初始化
    print("    -> 正在 GPU 上瞬间创建魔改架构 (GPU Accelerated Init)...")
    with torch.device("cuda"):
        # 在这个 with 块下，所有的 nn.Linear, nn.Parameter 都会直接在 GPU 创建
        # 初始化速度会快 100 倍以上
        model = Qwen3VLMMRLForGen(config, tokenizer)
        model.to(torch.bfloat16)

    # 步骤 C: 权重迁移 (CPU -> GPU)
    print("    -> 正在将权重从 CPU 迁移到 GPU 魔改模型...")
    # model.model 对应 QWen3WithMMRL
    # 这里的 load_state_dict 会非常快，因为它只是把 CPU 上的张量 copy 到 GPU 对应的位置
    msg = model.model.load_state_dict(original_model.model.state_dict(), strict=False)
    model.lm_head.load_state_dict(original_model.lm_head.state_dict())
    
    print(f"    -> 迁移完成！缺失键(MMRL相关): {len(msg.missing_keys)} 个")

    # 步骤 D: 释放内存
    del original_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    model.train()
    print("    -> 模型就绪，准备进入参数冻结环节。")
            
    # 3. 冻结参数 (精准控制)
    print("[3/5] 冻结参数，精准开启 MMRL 相关模块训练...")

    # A. 首先，冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # B. 获取模型的核心引用
    # Qwen3VLMMRLForGen -> self.model -> QWen3WithMMRL
    qwen_model = model.model 
    
    # C. 定义需要开启训练的模块列表
    modules_to_train = []
    
    # --- 1. MMRL 全局共享空间与投影器 ---
    # 定义在 QWen3WithMMRL.MMRL
    if hasattr(qwen_model, "MMRL"):
        modules_to_train.append(qwen_model.MMRL)
    else:
        print("[警告] 未找到 model.MMRL 模块")

    # --- 2. Vision Encoder 中的 MMRL 组件 ---
    # 定义在 QWen3WithMMRL.visual (VisionWithMMRL)
    vision_model = qwen_model.visual
    
    # 2.1 新增的 ViT Blocks (处理 r_token 的支路)
    if hasattr(vision_model, "blocks_with_rep"):
        modules_to_train.append(vision_model.blocks_with_rep)
        
    # 2.2 文本 Embedding 池化层
    if hasattr(vision_model, "embedding_pooling"):
        modules_to_train.append(vision_model.embedding_pooling)
        
    # 2.3 任务分类门控 (计算 Alpha)
    if hasattr(vision_model, "Task_classifier"):
        modules_to_train.append(vision_model.Task_classifier)
        
    # 2.4 视觉门控 (HardConcreteGate) - 主要是温度等参数(如有)
    if hasattr(vision_model, "visionGating"):
        modules_to_train.append(vision_model.visionGating)
        
    # 2.5 文本门控网络 (Text Gating)
    if hasattr(vision_model, "text_gating"):
        modules_to_train.append(vision_model.text_gating)
        
    # 2.6 零初始化层 (Zero Init)
    if hasattr(vision_model, "zero_init_layer"):
        modules_to_train.append(vision_model.zero_init_layer)

    # D. 遍历列表，开启梯度
    trainable_params_count = 0
    all_param_count = 0
    
    # 先计算总参数量用于统计
    for param in model.parameters():
        all_param_count += param.numel()

    print("正在开启以下模块的梯度:")
    for module in modules_to_train:
        print(f"  - {module.__class__.__name__}")
        for param in module.parameters():
            param.requires_grad = True

    # 再次统计可训练参数
    for param in model.parameters():
        if param.requires_grad:
            trainable_params_count += param.numel()

    print(f"    - 总参数量: {all_param_count}")
    print(f"    - 可训练参数: {trainable_params_count} ({100 * trainable_params_count / all_param_count:.4f}%)")
    
    # 简单校验：如果没有可训练参数，抛出异常防止空跑
    if trainable_params_count == 0:
        raise ValueError("没有参数被设置为可训练！请检查模块名称匹配逻辑。")

    # 4. 准备数据
    print("[4/5] 准备数据...")
    processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
        image_processor=image_processor, tokenizer=tokenizer, cfg=cfg
    )
    
    # 这里设置你的台词实验
    PROMPT = "如果你们容得下，这三位在这里肆意放肆，那就容我袁术告老还乡了"
    TARGET = "公路兄息怒，你走了 我们吃什么？\n是啊 吃什么?"
    
    dataset = OverfitDataset(processor, IMAGE_PATH, PROMPT, TARGET)
    
    # 5. 设置 Trainer
    print("[5/5] 启动 Trainer...")
    
    training_args = TrainingArguments(
        output_dir="./mmrl_overfit_output",
        num_train_epochs=1,             # 跑 10 个 epoch (其实 dataset 长度伪装成了 1000，所以步数很多)
        per_device_train_batch_size=1,   # 单卡 Batch 1
        gradient_accumulation_steps=1,
        learning_rate=1e-4,              # 较大的 LR 以便快速过拟合
        logging_steps=10,
        save_strategy="no",              # 实验不用存 checkpoint
        report_to="none",                # 不用 wandb
        remove_unused_columns=False,     # 必须 False，否则 'image_grid_thw' 等自定义字段会被 Trainer 丢弃
        bf16=True,                       # 开启 BF16
        dataloader_pin_memory=False,     # 有时候自定义 dataset pin_memory 会报错
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    
    # 6. 推理测试 (看看有没有学会)
    print("\n" + "=" * 20 + " 训练后推理测试 " + "=" * 20)
    model.eval()
    test_input = processor(
        text=[f"<|image_pad|>{PROMPT}"], # 简化的 prompt 构造
        images=Image.open(IMAGE_PATH),
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **test_input, 
            max_new_tokens=20
        )
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Input Image + Prompt: {PROMPT}")
        print(f"Model Output: {output_text}")

if __name__ == "__main__":
    train_overfit()