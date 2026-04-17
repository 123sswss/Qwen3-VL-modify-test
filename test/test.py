import json
import re
import os
import sys
import time
from datetime import datetime
from PIL import Image


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


def run_evaluation(json_path: str, model, image_dir: str = None, max_new_tokens=256, temperature=0.05):
    """
    Args:
        json_path: 数据集json路径
        model: ModelInterface实例
        image_dir: 图片目录，若为None则从json同级目录找
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if image_dir is None:
        image_dir = os.path.dirname(os.path.abspath(json_path))

    total = len(dataset)
    correct = 0
    wrong = 0
    regex_fail = 0
    logs = []

    print(f"={'='*60}")
    print(f"开始评测 | 共 {total} 题 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    start_time = time.time()

    for idx, item in enumerate(dataset):
        item_id = item.get("id", f"unknown_{idx}")
        image_file = item.get("image", "")
        conversations = item.get("conversations", [])

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
        img_path = os.path.join(image_dir, image_file)
        if not os.path.exists(img_path):
            logs.append({"id": item_id, "status": "SKIP", "reason": f"图片不存在: {img_path}"})
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"eval_log_{timestamp}.json"
    output_data = {"summary": summary, "logs": logs}
    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"详细日志已保存至: {log_filename}")

    return summary


if __name__ == "__main__":
    #################### ours ###########################
    from inferEngine import ModelInterface
    TRAINED_MODEL_PATH = "/root/autodl-tmp/Qwen3-VL-modify-test/mmrl_output"
    BASE_MODEL_PATH = "/root/autodl-tmp/model"
    JSON_PATH = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/dataset/test2_val.json"
    IMAGE_DIR = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/dataset/2/train"

    model = ModelInterface(TRAINED_MODEL_PATH, BASE_MODEL_PATH)
    run_evaluation(JSON_PATH, model, image_dir=IMAGE_DIR)
    #################### qwen3vl 4B ###########################
    # from inferQWen3vl import BaselineModelInterface
    # MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/model"
    # JSON_PATH = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/dataset/test2_val.json"
    # IMAGE_DIR = sys.argv[3] if len(sys.argv) > 3 else "/root/autodl-tmp/dataset/2/train"

    # model = BaselineModelInterface(MODEL_PATH)
    # run_evaluation(JSON_PATH, model, image_dir=IMAGE_DIR)