import json
import re
import os
import sys
import time
from datetime import datetime
from PIL import Image


DEFAULT_JSON_PATHS = [
    "/root/autodl-tmp/dataset/test2_val.json",
    "/root/autodl-tmp/dataset/seen_simple/llava_test.json",
    # "/root/autodl-tmp/dataset/never_seen_simple/llava_test.json"
]

DEFAULT_IMAGE_DIRS = [
    "/root/autodl-tmp/dataset/2/train",
    "/root/autodl-tmp/dataset/seen_simple/image",
    # "/root/autodl-tmp/dataset/never_seen_simple/image"
]


def extract_answer(text: str):
    """从模型输出中提取答案字母，返回 (字母或None, 是否提取成功)"""
    # 优先匹配 [[X]] 格式
    m = re.search(r'\[\[([A-Da-d])\]\]', text)
    if m:
        return m.group(1).upper(), True
    # 回退：匹配 "最终答案"后面的字母
    m = re.search(r'最终答案[：:]\s*\[?\[?([A-Da-d])', text)
    if m:
        return m.group(1).upper(), True
    # 再回退：找第一个独立的选项字母
    m = re.search(r'\b([A-Da-d])\b', text)
    if m:
        return m.group(1).upper(), True
    return None, False


def extract_gt_answer(gpt_value: str):
    """从标注的gpt回复中提取正确答案"""
    m = re.search(r'\[\[([A-Da-d])\]\]', gpt_value)
    if m:
        return m.group(1).upper()
    return None


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def parse_cli_or_default(arg_value, default_values):
    """支持单个路径或 JSON 数组字符串；为空时返回默认配置。"""
    if arg_value is None:
        return list(default_values)

    raw_value = arg_value.strip()
    if not raw_value:
        return list(default_values)

    if raw_value.startswith("["):
        parsed = json.loads(raw_value)
        if not isinstance(parsed, list):
            raise ValueError(f"期望接收到列表配置，实际得到: {type(parsed).__name__}")
        return [str(v) for v in parsed]

    return [raw_value]


def load_combined_dataset(json_paths):
    combined_dataset = []

    for json_path in json_paths:
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            raise ValueError(f"数据集文件不是题目列表: {json_path}")

        for item in dataset:
            normalized_item = dict(item)
            normalized_item["_source_json"] = json_path
            combined_dataset.append(normalized_item)

    return combined_dataset


def resolve_image_path(image_file: str, image_dirs, source_json_path: str = None):
    candidate_paths = []

    for image_dir in image_dirs:
        candidate_paths.append(os.path.join(image_dir, image_file))

    if source_json_path is not None:
        json_sibling_dir = os.path.dirname(os.path.abspath(source_json_path))
        candidate_paths.append(os.path.join(json_sibling_dir, image_file))

    seen = set()
    for candidate in candidate_paths:
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.exists(normalized):
            return normalized

    return None


def run_evaluation(json_paths, model, image_dirs=None, max_new_tokens=256, temperature=0.2):
    """
    Args:
        json_paths: 数据集 json 路径列表或单个路径
        model: ModelInterface实例
        image_dirs: 图片目录列表或单个目录；若为空则额外尝试各 json 同级目录
    """
    json_paths = ensure_list(json_paths)
    image_dirs = ensure_list(image_dirs)
    dataset = load_combined_dataset(json_paths)

    total = len(dataset)
    correct = 0
    wrong = 0
    regex_fail = 0
    logs = []

    print(f"={'='*60}")
    print(f"开始评测 | 共 {total} 题 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据集文件: {json_paths}")
    print(f"图片目录池: {image_dirs if image_dirs else '[各 JSON 同级目录]'}")
    print(f"{'='*60}")

    start_time = time.time()

    for idx, item in enumerate(dataset):
        item_id = item.get("id", f"unknown_{idx}")
        image_file = item.get("image", "")
        conversations = item.get("conversations", [])
        source_json = item.get("_source_json")

        # 提取human prompt和gt answer
        human_text = ""
        gt_answer = None
        for conv in conversations:
            if conv["from"] == "human":
                human_text = conv["value"].replace("<image>\n", "").replace("<image>", "")
            elif conv["from"] == "gpt":
                gt_answer = extract_gt_answer(conv["value"])

        if gt_answer is None:
            logs.append({"id": item_id, "status": "SKIP", "reason": "无法提取标注答案"})
            continue

        # 加载图片
        img_path = resolve_image_path(image_file, image_dirs, source_json_path=source_json)
        if img_path is None:
            searched_dirs = list(image_dirs)
            if source_json is not None:
                searched_dirs.append(os.path.dirname(os.path.abspath(source_json)))
            logs.append({
                "id": item_id,
                "status": "SKIP",
                "reason": f"图片不存在: {image_file}",
                "searched_dirs": searched_dirs,
                "source_json": source_json,
            })
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logs.append({"id": item_id, "status": "SKIP", "reason": f"图片读取失败: {e}"})
            continue

        # 推理
        try:
            output = model.infer(image, human_text, max_new_tokens=max_new_tokens, temperature=temperature)
        except Exception as e:
            logs.append({"id": item_id, "status": "ERROR", "reason": str(e)})
            continue

        # 提取答案
        pred_answer, extracted = extract_answer(output)

        log_entry = {
            "id": item_id,
            "source_json": source_json,
            "image_path": img_path,
            "question": human_text[:100] + "..." if len(human_text) > 100 else human_text,
            "gt": gt_answer,
            "pred": pred_answer,
            "extracted": extracted,
            "model_output": output,
        }

        if not extracted:
            regex_fail += 1
            wrong += 1
            log_entry["status"] = "REGEX_FAIL"
        elif pred_answer == gt_answer:
            correct += 1
            log_entry["status"] = "CORRECT"
        else:
            wrong += 1
            log_entry["status"] = "WRONG"

        logs.append(log_entry)

        # 进度
        done = idx + 1
        acc_so_far = correct / done * 100 if done > 0 else 0
        status_icon = "✓" if log_entry["status"] == "CORRECT" else ("✗" if log_entry["status"] == "WRONG" else "⚠")
        print(f"[{done}/{total}] {status_icon} GT={gt_answer} Pred={pred_answer} Acc={acc_so_far:.1f}% | {item_id}")

    elapsed = time.time() - start_time
    evaluated = correct + wrong
    score = correct / evaluated * 100 if evaluated > 0 else 0

    # 汇总
    summary = {
        "json_paths": json_paths,
        "image_dirs": image_dirs,
        "total_in_dataset": total,
        "evaluated": evaluated,
        "skipped": total - evaluated,
        "correct": correct,
        "wrong": wrong,
        "regex_fail": regex_fail,
        "score": round(score, 2),
        "elapsed_seconds": round(elapsed, 1),
        "avg_seconds_per_question": round(elapsed / max(evaluated, 1), 2),
    }

    print(f"\n{'='*60}")
    print(f"评测完成")
    print(f"  总题数:       {summary['total_in_dataset']}")
    print(f"  实际评测:     {summary['evaluated']}")
    print(f"  跳过:         {summary['skipped']}")
    print(f"  正确:         {summary['correct']}")
    print(f"  错误:         {summary['wrong']}")
    print(f"  正则提取失败: {summary['regex_fail']}")
    print(f"  百分制分数:   {summary['score']}")
    print(f"  总耗时:       {summary['elapsed_seconds']}s")
    print(f"  平均每题:     {summary['avg_seconds_per_question']}s")
    print(f"{'='*60}")

    # 保存日志
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # log_filename = f"eval_log_{timestamp}.json"
    # output_data = {"summary": summary, "logs": logs}
    # with open(log_filename, 'w', encoding='utf-8') as f:
    #     json.dump(output_data, f, ensure_ascii=False, indent=2)
    # print(f"详细日志已保存至: {log_filename}")

    return summary


if __name__ == "__main__":
    #################### ours ###########################
    from inferEngine import ModelInterface
    TRAINED_MODEL_PATH = os.getenv("MMRL_TRAINED_MODEL_PATH", "/root/autodl-tmp/Qwen3-VL-modify-test/experiment_outputs/output/visual_router_v1_3/final")
    BASE_MODEL_PATH = os.getenv("MMRL_BASE_MODEL_PATH", "/root/autodl-tmp/model")

    DEFAULT_JSON_PATHS = [
        "/root/autodl-tmp/dataset/test2_val.json",
        "/root/autodl-tmp/dataset/seen_simple/llava_test.json",
        # "/root/autodl-tmp/dataset/never_seen_simple/llava_test.json"
    ]
    
    DEFAULT_IMAGE_DIRS = [
        "/root/autodl-tmp/dataset/2/train",
        "/root/autodl-tmp/dataset/seen_simple/image",
        # "/root/autodl-tmp/dataset/never_seen_simple/image"
    ]

    model = ModelInterface(TRAINED_MODEL_PATH, BASE_MODEL_PATH)
    run_evaluation(DEFAULT_JSON_PATHS, model, DEFAULT_IMAGE_DIRS)
    #################### qwen3vl 4B ###########################
    # from inferQWen3vl import BaselineModelInterface
    # MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/model"
    # JSON_PATH = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/dataset/test2_val.json"
    # IMAGE_DIR = sys.argv[3] if len(sys.argv) > 3 else "/root/autodl-tmp/dataset/2/train"

    # model = BaselineModelInterface(MODEL_PATH)
    # run_evaluation(JSON_PATH, model, image_dir=IMAGE_DIR)