# eval_consistency.py
"""
模型一致性评估框架
用法:
    # 1. 先跑原模型
    python eval_consistency.py --model_path Qwen/Qwen3-VL --model_type original --output_dir ./eval_cache
    
    # 2. 再跑你的模型
    python eval_consistency.py --model_path /path/to/your/model --model_type modified --output_dir ./eval_cache
    
    # 3. 计算指标
    python eval_consistency.py --compare --output_dir ./eval_cache
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import argparse

from torch.utils.data import DataLoader
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch.nn.functional as F

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoConfig,
    AutoTokenizer,
    AutoImageProcessor,
    GenerationConfig
)

import sys 
sys.path.append("..") 
from infer import Qwen3VLMMRLForGen

# ==================== 数据协议定义 ====================

@dataclass
class SampleResult:
    """单条样本的推理结果"""
    sample_id: int
    input_text: str
    # logits 太大，单独存 .pt 文件，这里只存路径
    logits_file: str
    top1_token_id: int
    top10_token_ids: List[int]
    generated_text: str
    alpha_logits: Optional[List[float]] = None  # 原始 alpha 值
    alpha_probs: Optional[List[float]] = None   # sigmoid 后的概率
    k_values: Optional[List[float]] = None      # K 值


@dataclass 
class ExperimentMeta:
    """实验元数据"""
    model_type: str  # "original" or "modified"
    model_path: str
    timestamp: str
    num_samples: int
    device: str
    dtype: str
    dataset_config: Dict[str, Any]


class EvalDataProtocol:
    """
    评估数据存储协议
    
    目录结构:
    output_dir/
    ├── original/
    │   ├── meta.json
    │   ├── results.json
    │   └── logits/
    │       ├── sample_0.pt
    │       ├── sample_1.pt
    │       └── ...
    └── modified/
        ├── meta.json
        ├── results.json
        └── logits/
            └── ...
    """
    
    def __init__(self, output_dir: str, model_type: str):
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        self.model_dir = self.output_dir / model_type
        self.logits_dir = self.model_dir / "logits"
        
        # 创建目录
        self.logits_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[SampleResult] = []
        self.meta: Optional[ExperimentMeta] = None
    
    def save_sample_logits(self, sample_id: int, logits: torch.Tensor) -> str:
        """保存单个样本的 logits，返回文件路径"""
        filename = f"sample_{sample_id}.pt"
        filepath = self.logits_dir / filename
        # 只保存第一个生成 token 的 logits (用于计算 KL)
        # logits shape: [seq_len, vocab_size] -> 取最后一个位置
        torch.save(logits.cpu().half(), filepath)  # 用 fp16 节省空间
        return str(filepath)
    
    def add_result(self, result: SampleResult):
        self.results.append(result)
    
    def set_meta(self, meta: ExperimentMeta):
        self.meta = meta
    
    def save(self):
        """保存所有结果"""
        # 保存元数据
        meta_path = self.model_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.meta), f, indent=2, ensure_ascii=False)
        
        # 保存结果列表
        results_path = self.model_dir / "results.json"
        results_data = [asdict(r) for r in self.results]
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"已保存 {len(self.results)} 条结果到 {self.model_dir}")
    
    @classmethod
    def load(cls, output_dir: str, model_type: str) -> "EvalDataProtocol":
        """加载已保存的数据"""
        protocol = cls(output_dir, model_type)
        
        meta_path = protocol.model_dir / "meta.json"
        results_path = protocol.model_dir / "results.json"
        
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
            protocol.meta = ExperimentMeta(**meta_dict)
        
        with open(results_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)
            protocol.results = [SampleResult(**r) for r in results_data]
        
        return protocol
    
    def load_logits(self, sample_id: int) -> torch.Tensor:
        """加载指定样本的 logits"""
        filepath = self.logits_dir / f"sample_{sample_id}.pt"
        return torch.load(filepath, map_location="cpu").float()


# ==================== 模型加载器 ====================

class ModelLoader:
    """统一的模型加载接口"""
    
    @staticmethod
    def load_original(model_path: str, device_map="auto", dtype=torch.float16):
        """加载原始 Qwen3-VL 模型"""
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation="sdpa"
        )
        processor = AutoProcessor.from_pretrained(model_path)
        return model, processor
    
    @staticmethod
    def load_modified(model_path: str, base_model_path: str = "/root/autodl-tmp/model", 
                    device_map="auto", dtype=torch.float16):
        """加载你的 MMRL 魔改模型"""
        import config as cfg
        import QWen3WithMMRL
        import processingWithMMRL
        from safetensors.torch import load_file
        
        # 1. 从基座加载 tokenizer 并添加特殊 token
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.add_special_tokens(cfg.SPECIAL_TOKENS)
        
        # 2. 加载 config
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        except:
            config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        
        # 3. 构建模型（复用 infer.py 的 Qwen3VLMMRLForGen）
        model = Qwen3VLMMRLForGen(config, tokenizer)
        model.to(dtype)
        
        # 4. 加载权重
        safetensors_path = os.path.join(model_path, "model.safetensors")
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        
        # 5. 构建自定义 processor
        image_processor = AutoImageProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
            image_processor=image_processor, tokenizer=tokenizer, cfg=cfg
        )
        
        return model.to("cuda"), processor
    
    @classmethod
    def load(cls, model_path: str, model_type: str, **kwargs):
        """统一加载接口"""
        if model_type == "original":
            return cls.load_original(model_path, **kwargs)
        elif model_type == "modified":
            return cls.load_modified(model_path, **kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


# ==================== 评估数据集适配器 ====================

class EvalDatasetAdapter:
    """
    适配你的 MixedMMRLDataset，提取评估所需的数据
    """
    
    def __init__(
        self,
        processor,
        general_json: str,
        general_img_dir: str,
        num_samples: int = 500
    ):
        # 延迟导入，避免依赖问题
        from dataset import MixedMMRLDataset  # TODO: 改成你的实际模块路径
        
        self.dataset = MixedMMRLDataset(
            processor=processor,
            expert_json=None,  # 不需要专业数据
            expert_img_dir=None,
            general_json=general_json,
            general_img_dir=general_img_dir,
            total_limit=num_samples,
            mode="general",
            general_ratio_limit=1.0
        )
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


# ==================== 推理引擎 ====================

class ConsistencyEvaluator:
    """一致性评估器"""
    
    def __init__(
        self,
        model,
        processor,
        protocol: EvalDataProtocol,
        device: str = "cuda"
    ):
        self.model = model
        self.processor = processor
        self.protocol = protocol
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def run_inference(self, dataloader, max_new_tokens: int = 1):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            try:
                if batch_idx == 0:
                    print("\n[DEBUG] Batch keys:", list(batch.keys()))
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                # ===== 关键修复：用 labels 找 prompt 边界 =====
                labels = batch.get("labels", None)
                prompt_len = None
                if labels is not None:
                    # labels: -100 表示 prompt token，非 -100 表示 response token
                    label_mask = (labels[0] != -100)
                    if label_mask.any():
                        prompt_len = label_mask.nonzero(as_tuple=True)[0][0].item()
                    else:
                        prompt_len = labels.shape[1]
                    
                    if batch_idx == 0:
                        print(f"  [DEBUG] total_len={labels.shape[1]}, prompt_len={prompt_len}, "
                              f"response_len={labels.shape[1] - prompt_len}")
                model_inputs = {}
                if "input_ids" in batch:
                    ids = batch["input_ids"]
                    if prompt_len is not None:
                        ids = ids[:, :prompt_len]
                    model_inputs["input_ids"] = ids.to(self.device)
                if "attention_mask" in batch:
                    mask = batch["attention_mask"]
                    if prompt_len is not None:
                        mask = mask[:, :prompt_len]
                    model_inputs["attention_mask"] = mask.to(self.device)
                # 图像字段 - 不需要截断，pixel_values 是独立的
                if "pixel_values" in batch and batch["pixel_values"] is not None:
                    model_inputs["pixel_values"] = batch["pixel_values"].to(self.device, dtype=torch.float16)
                    if batch_idx == 0:
                        print(f"  [DEBUG] pixel_values: {model_inputs['pixel_values'].shape}  ← 图片已传入")
                else:
                    if batch_idx == 0:
                        print("  [DEBUG] pixel_values: None  ← 纯文本样本，无图片")
                if "image_grid_thw" in batch and batch["image_grid_thw"] is not None:
                    grid_thw = batch["image_grid_thw"]
                    if isinstance(grid_thw, torch.Tensor):
                        if grid_thw.dim() == 1:
                            grid_thw = grid_thw.unsqueeze(0)
                        elif grid_thw.dim() == 3:
                            grid_thw = grid_thw.squeeze(0)
                        model_inputs["image_grid_thw"] = grid_thw.to(self.device)
                if "pixel_values_videos" in batch and batch["pixel_values_videos"] is not None:
                    model_inputs["pixel_values_videos"] = batch["pixel_values_videos"].to(self.device, dtype=torch.float16)
                if "video_grid_thw" in batch and batch["video_grid_thw"] is not None:
                    model_inputs["video_grid_thw"] = batch["video_grid_thw"].to(self.device)
                # 获取 logits（此时 input_ids 只含 prompt，logits[-1] 是预测第一个 response token）
                outputs = self.model(**model_inputs, return_dict=True)
                last_logits = outputs.logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                top10_values, top10_indices = torch.topk(probs, k=10, dim=-1)
                top1_token_id = top10_indices[0, 0].item()
                top10_token_ids = top10_indices[0].tolist()
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                generated_ids_trimmed = generated_ids[:, model_inputs["input_ids"].shape[1]:]
                generated_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True
                )[0]
                # input_text 现在只含 prompt 部分
                input_text = self.processor.batch_decode(
                    model_inputs["input_ids"], skip_special_tokens=True
                )[0][:200]
                # ... MMRL 门控信息提取（保持不变）...
                alpha_logits_list = None
                alpha_probs_list = None
                k_values_list = None
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
                    visual = self.model.model.visual
                    if hasattr(visual, 'alpha_list') and visual.alpha_list is not None:
                        alpha_logits = visual.alpha_list
                        alpha_probs = torch.sigmoid(alpha_logits)
                        alpha_logits_list = alpha_logits.squeeze().detach().cpu().tolist()
                        alpha_probs_list = alpha_probs.squeeze().detach().cpu().tolist()
                        if not isinstance(alpha_logits_list, list):
                            alpha_logits_list = [alpha_logits_list]
                            alpha_probs_list = [alpha_probs_list]
                    if hasattr(self.model.model, 'k_results') and self.model.model.k_results is not None:
                        k = self.model.model.k_results
                        k_values_list = k.squeeze().detach().cpu().tolist()
                        if not isinstance(k_values_list, list):
                            k_values_list = [k_values_list]
                logits_file = self.protocol.save_sample_logits(batch_idx, last_logits[0])
                result = SampleResult(
                    sample_id=batch_idx,
                    input_text=input_text,
                    logits_file=logits_file,
                    top1_token_id=top1_token_id,
                    top10_token_ids=top10_token_ids,
                    generated_text=generated_text,
                    alpha_logits=alpha_logits_list,
                    alpha_probs=alpha_probs_list,
                    k_values=k_values_list
                )
                self.protocol.add_result(result)
            except Exception as e:
                print(f"\n[ERROR] Sample {batch_idx} failed: {e}")
                raise
        self.protocol.save()


# ==================== 指标计算 ====================

class MetricsCalculator:
    """计算一致性指标"""
    
    @staticmethod
    def compute_kl_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
        """计算 KL 散度: KL(P_a || P_b)"""
        log_probs_a = F.log_softmax(logits_a, dim=-1)
        probs_b = F.softmax(logits_b, dim=-1)
        # KL(P_a || P_b) = sum(P_a * (log P_a - log P_b))
        kl = F.kl_div(F.log_softmax(logits_b, dim=-1), probs_b, reduction='batchmean', log_target=False)
        # 用更标准的方式
        probs_a = F.softmax(logits_a, dim=-1)
        kl = (probs_a * (log_probs_a - F.log_softmax(logits_b, dim=-1))).sum(dim=-1)
        return kl.item()
    
    @staticmethod
    def compute_js_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
        """计算 JS 散度 (对称版 KL)"""
        probs_a = F.softmax(logits_a, dim=-1)
        probs_b = F.softmax(logits_b, dim=-1)
        m = 0.5 * (probs_a + probs_b)
        
        kl_a_m = (probs_a * (torch.log(probs_a + 1e-10) - torch.log(m + 1e-10))).sum(dim=-1)
        kl_b_m = (probs_b * (torch.log(probs_b + 1e-10) - torch.log(m + 1e-10))).sum(dim=-1)
        
        js = 0.5 * (kl_a_m + kl_b_m)
        return js.item()
    
    @classmethod
    def compare(cls, output_dir: str) -> Dict[str, float]:
        """比较两个模型的结果"""
        # 加载数据
        original = EvalDataProtocol.load(output_dir, "original")
        modified = EvalDataProtocol.load(output_dir, "modified")
        
        assert len(original.results) == len(modified.results), \
            f"样本数不匹配: {len(original.results)} vs {len(modified.results)}"
        
        kl_divs = []
        js_divs = []
        top1_matches = []
        top10_overlaps = []
        
        for orig_r, mod_r in tqdm(
            zip(original.results, modified.results), 
            total=len(original.results),
            desc="Computing metrics"
        ):
            # 加载 logits
            orig_logits = original.load_logits(orig_r.sample_id)
            mod_logits = modified.load_logits(mod_r.sample_id)

            # 取最小词表大小，双向截断（padding token 不影响语义）
            vocab_size = min(orig_logits.shape[-1], mod_logits.shape[-1])
            orig_logits = orig_logits[..., :vocab_size]
            mod_logits = mod_logits[..., :vocab_size]
            
            # KL 散度
            kl = cls.compute_kl_divergence(orig_logits, mod_logits)
            kl_divs.append(kl)
            
            # JS 散度
            js = cls.compute_js_divergence(orig_logits, mod_logits)
            js_divs.append(js)
            
            # Top-1 一致率
            top1_match = int(orig_r.top1_token_id == mod_r.top1_token_id)
            top1_matches.append(top1_match)
            
            # Top-10 重叠率
            orig_set = set(orig_r.top10_token_ids)
            mod_set = set(mod_r.top10_token_ids)
            overlap = len(orig_set & mod_set) / 10.0
            top10_overlaps.append(overlap)
        
        alpha_means = []
        k_means = []
        for mod_r in modified.results:
            if mod_r.alpha_probs is not None:
                alpha_means.append(np.mean(mod_r.alpha_probs))
            if mod_r.k_values is not None:
                k_means.append(np.mean(mod_r.k_values))
        
        metrics = {
            "mean_kl_divergence": float(np.mean(kl_divs)),
            "std_kl_divergence": float(np.std(kl_divs)),
            "mean_js_divergence": float(np.mean(js_divs)),
            "std_js_divergence": float(np.std(js_divs)),
            "top1_agreement": float(np.mean(top1_matches)),
            "top10_overlap": float(np.mean(top10_overlaps)),
            "num_samples": len(original.results),
            "mean_alpha_prob": float(np.mean(alpha_means)) if alpha_means else None,
            "mean_k_value": float(np.mean(k_means)) if k_means else None,
        }
        
        return metrics


# ==================== 主函数 ====================

def main():
    import random
    torch.manual_seed(114514)
    random.seed(114514)
    np.random.seed(114514)
    parser = argparse.ArgumentParser(description="模型一致性评估")
    parser.add_argument("--model_path", type=str, help="模型路径")
    parser.add_argument("--model_type", type=str, choices=["original", "modified"], 
                        help="模型类型")
    parser.add_argument("--output_dir", type=str, default="./eval_cache",
                        help="输出目录")
    parser.add_argument("--compare", action="store_true", 
                        help="比较模式：计算两个模型的指标")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="评估样本数")
    parser.add_argument("--base_model_path", type=str, default="/root/autodl-tmp/model",
                        help="原始基座模型路径（用于加载 tokenizer 和 config）")
    
    args = parser.parse_args()
    
    if args.compare:
        # 比较模式
        print("=" * 50)
        print("计算一致性指标...")
        print("=" * 50)
        
        metrics = MetricsCalculator.compare(args.output_dir)
        
        print("\n📊 评估结果:")
        print("-" * 40)
        print(f"KL 散度:      {metrics['mean_kl_divergence']:.6f} ± {metrics['std_kl_divergence']:.6f}")
        print(f"JS 散度:      {metrics['mean_js_divergence']:.6f} ± {metrics['std_js_divergence']:.6f}")
        print(f"Top-1 一致率: {metrics['top1_agreement']*100:.2f}%")
        print(f"Top-10 重叠:  {metrics['top10_overlap']*100:.2f}%")
        print(f"样本数:       {metrics['num_samples']}")
        print("-" * 40)
        
        # 保存指标
        metrics_path = Path(args.output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n指标已保存到: {metrics_path}")
        
        # 给出论文可用的结论
        print("\n📝 论文可用表述:")
        if metrics['mean_kl_divergence'] < 0.01 and metrics['top1_agreement'] > 0.9:
            print("✅ 'The modified model shows negligible deviation from the original, "
                  f"with {metrics['top1_agreement']*100:.1f}% top-1 token agreement and "
                  f"{metrics['mean_kl_divergence']:.4f} mean KL divergence.'")
        else:
            print("⚠️ 存在一定差异，建议检查模型修改是否符合预期")
    
    else:
        # 推理模式
        assert args.model_path, "需要指定 --model_path"
        assert args.model_type, "需要指定 --model_type"
        
        print("=" * 50)
        print(f"运行推理: {args.model_type} 模型")
        print(f"模型路径: {args.model_path}")
        print("=" * 50)
        
        # 加载模型
        print("\n加载模型...")
        model, processor = ModelLoader.load(
            args.model_path, 
            args.model_type,
            dtype=torch.bfloat16
        )
        
        # 初始化协议
        protocol = EvalDataProtocol(args.output_dir, args.model_type)
        protocol.set_meta(ExperimentMeta(
            model_type=args.model_type,
            model_path=args.model_path,
            timestamp=datetime.now().isoformat(),
            num_samples=args.num_samples,
            device=str(next(model.parameters()).device),
            dtype="float16",
            dataset_config={
                "num_samples": args.num_samples
            }
        ))
        

        general_json="/root/autodl-tmp/dataset/llava_instruct_150k.json"
        general_img_dir="/root/autodl-tmp/dataset/gen/train2017"

        # 加载数据集
        print("\n加载数据集...")
        eval_adapter = EvalDatasetAdapter(
            processor=processor,
            general_json=general_json,
            general_img_dir=general_img_dir,
            num_samples=args.num_samples
        )
        
        # 创建 DataLoader (batch_size=1 简化处理)
        dataloader = DataLoader(
            eval_adapter.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        # 运行评估
        print("\n开始推理...")
        evaluator = ConsistencyEvaluator(model, processor, protocol)
        evaluator.run_inference(dataloader)
        
        print(f"\n✅ 完成! 结果保存在: {protocol.model_dir}")


if __name__ == "__main__":
    main()

# usage:
# # 1. 跑原模型
# python gen_eval_org.py \
#     --model_path /root/autodl-tmp/model \
#     --model_type original \
#     --num_samples 500 \
#     --output_dir ./eval_cache
# # 2. 跑你的模型 (之后)
# python gen_eval_org.py \
#     --model_path /root/autodl-tmp/Qwen3-VL-modify-test/mmrl_output \
#     --model_type modified \
#     --num_samples 500 \
#     --output_dir ./eval_cache
# # 3. 计算指标
# python gen_eval_org.py --compare --output_dir ./eval_cache