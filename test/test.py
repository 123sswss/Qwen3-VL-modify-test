import json
import re
import os
import sys
import time
from datetime import datetime
from PIL import Image


DEFAULT_JSON_PATHS = [
    "/root/autodl-tmp/dataset/test/1_test.json",
    "/root/autodl-tmp/dataset/test/14_test.json",
]

DEFAULT_IMAGE_DIRS = [
    "/root/autodl-tmp/dataset/test/1",
    "/root/autodl-tmp/dataset/test/14",
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


def validate_evaluation_data(json_paths, image_dirs=None):
    json_paths = ensure_list(json_paths)
    image_dirs = ensure_list(image_dirs)
    dataset = load_combined_dataset(json_paths)
    counts = {
        "total": len(dataset),
        "valid": 0,
        "missing_gt": 0,
        "missing_image": 0,
    }
    examples = {}

    for idx, item in enumerate(dataset):
        conversations = item.get("conversations", [])
        gt_answer = None
        for conv in conversations:
            if conv.get("from") == "gpt":
                gt_answer = extract_gt_answer(conv.get("value", ""))
        if gt_answer is None:
            counts["missing_gt"] += 1
            examples.setdefault("missing_gt", item.get("id", f"unknown_{idx}"))
            continue

        image_file = item.get("image", "")
        source_json = item.get("_source_json")
        if resolve_image_path(image_file, image_dirs, source_json_path=source_json) is None:
            counts["missing_image"] += 1
            examples.setdefault("missing_image", image_file or item.get("id", f"unknown_{idx}"))
            continue
        counts["valid"] += 1

    print(
        "[EVAL-DATA-PREFLIGHT] "
        f"total={counts['total']} valid={counts['valid']} "
        f"missing_gt={counts['missing_gt']} missing_image={counts['missing_image']} "
        f"examples={examples}"
    )
    return counts


def run_evaluation(json_paths, model, image_dirs=None, max_new_tokens=64, temperature=0.2):
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
    inference_errors = 0
    first_inference_error = None
    skip_reasons = {"missing_gt": 0, "missing_image": 0, "image_read_error": 0}
    logs = []
    generation_timings = []

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
            skip_reasons["missing_gt"] += 1
            if skip_reasons["missing_gt"] <= 3:
                print(f"[EVAL-SKIP missing_gt] id={item_id}")
            logs.append({"id": item_id, "status": "SKIP", "reason": "无法提取标注答案"})
            continue

        # 加载图片
        img_path = resolve_image_path(image_file, image_dirs, source_json_path=source_json)
        if img_path is None:
            skip_reasons["missing_image"] += 1
            if skip_reasons["missing_image"] <= 3:
                print(f"[EVAL-SKIP missing_image] id={item_id} image={image_file}")
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
            skip_reasons["image_read_error"] += 1
            if skip_reasons["image_read_error"] <= 3:
                print(f"[EVAL-SKIP image_read_error] id={item_id} error={e}")
            logs.append({"id": item_id, "status": "SKIP", "reason": f"图片读取失败: {e}"})
            continue

        # 推理
        try:
            output = model.infer(image, human_text, max_new_tokens=max_new_tokens, temperature=temperature)
        except Exception as e:
            inference_errors += 1
            if first_inference_error is None:
                first_inference_error = str(e)
            if inference_errors <= 3:
                print(f"[EVAL-ERROR {inference_errors}] id={item_id} error={e}")
            logs.append({"id": item_id, "status": "ERROR", "reason": str(e)})
            continue

        generation_timing = getattr(model, "last_generation_timing", None)
        if isinstance(generation_timing, dict):
            generation_timing = dict(generation_timing)
            generation_timings.append(generation_timing)

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
        if generation_timing is not None:
            log_entry["generation_timing"] = generation_timing

        if not extracted:
            regex_fail += 1
            wrong += 1
            log_entry["status"] = "REGEX_FAIL"
            print(
                f"[REGEX_FAIL_OUTPUT] image={image_file} id={item_id}\n"
                f"{output}\n"
            )
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
        display_name = image_file or item_id
        print(
            f"[{done}/{total}] {status_icon} GT={gt_answer} "
            f"Pred={pred_answer} Acc={acc_so_far:.1f}% | {display_name}"
        )

    elapsed = time.time() - start_time
    evaluated = correct + wrong
    score = correct / evaluated * 100 if evaluated > 0 else 0

    def timing_mean(key):
        values = [
            float(row[key])
            for row in generation_timings
            if row.get(key) is not None
        ]
        return sum(values) / len(values) if values else None

    first_token_latency = timing_mean("first_token_latency_seconds")
    subsequent_token_time = timing_mean("subsequent_token_mean_seconds")

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
        "inference_errors": inference_errors,
        "first_inference_error": first_inference_error,
        "skip_reasons": skip_reasons,
        "score": round(score, 2),
        "elapsed_seconds": round(elapsed, 1),
        "avg_seconds_per_question": round(elapsed / max(evaluated, 1), 2),
        "timed_generations": len(generation_timings),
        "first_token_latency_seconds": (
            round(first_token_latency, 4) if first_token_latency is not None else None
        ),
        "subsequent_token_mean_seconds": (
            round(subsequent_token_time, 4) if subsequent_token_time is not None else None
        ),
        "subsequent_tokens_per_second": (
            round(1.0 / subsequent_token_time, 2)
            if subsequent_token_time is not None and subsequent_token_time > 0
            else None
        ),
    }

    print(f"\n{'='*60}")
    print(f"评测完成")
    print(f"  总题数:       {summary['total_in_dataset']}")
    print(f"  实际评测:     {summary['evaluated']}")
    print(f"  跳过:         {summary['skipped']}")
    print(f"  正确:         {summary['correct']}")
    print(f"  错误:         {summary['wrong']}")
    print(f"  正则提取失败: {summary['regex_fail']}")
    print(f"  推理异常:     {summary['inference_errors']}")
    if summary["first_inference_error"] is not None:
        print(f"  首个异常:     {summary['first_inference_error']}")
    print(f"  跳过原因:     {summary['skip_reasons']}")
    print(f"  百分制分数:   {summary['score']}")
    print(f"  总耗时:       {summary['elapsed_seconds']}s")
    print(f"  平均每题:     {summary['avg_seconds_per_question']}s")
    if summary["timed_generations"]:
        print(f"  计时样本数:   {summary['timed_generations']}")
        print(f"  首Token延迟:  {summary['first_token_latency_seconds']}s")
        print(f"  后续Token耗时: {summary['subsequent_token_mean_seconds']}s/token")
        print(f"  后续生成速度: {summary['subsequent_tokens_per_second']} token/s")
    else:
        print("  Token生成速度: 未采集（当前模型推理接口未接入逐Token计时）")
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
