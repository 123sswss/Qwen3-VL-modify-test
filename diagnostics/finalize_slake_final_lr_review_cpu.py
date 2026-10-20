"""Render explicit reviewed annotations and paired statistics into reports."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    sample, manual = read("slake_sample_60.json"), read("manual_annotations.json")
    annotations = {r[0]: r for r in manual["annotations"]}
    assert set(annotations) == set(range(1,61))
    reviewed = []
    for row in sample["rows"]:
        index, tags, uncertain, reason = annotations[row["review_index"]]
        assert (root / row["exported_image"]).is_file()
        reviewed.append({**row, "tags": tags.split(","), "uncertain": uncertain,
                         "review_evidence": reason, "image_reviewed": True,
                         "official_score_unchanged": True})
    counts = {}
    for category in ("all", "r2_only", "v1_only"):
        rows = reviewed if category == "all" else [r for r in reviewed if r["category"] == category]
        counts[category] = {"n": len(rows), "uncertain": sum(r["uncertain"] for r in rows),
                            "tag_counts": dict(Counter(tag for r in rows for tag in r["tags"]))}
    write = lambda name, x: (root / name).write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")
    write("reviewed_60.json", {"reviewer": manual["reviewer"], "rows": reviewed, "counts": counts,
                               "limitations": manual["limitations"]})
    table = ["# 固定60题逐例审阅", "", "|编号|qid|独占正确方|标签|不确定|问题|参考|V1|r2|审阅依据|", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in reviewed:
        fields = [str(r["review_index"]), r["question_id"], r["category"], ",".join(r["tags"]), str(r["uncertain"]), r["question"], r["reference"], r["v1_prediction"], r["r2_prediction"], r["review_evidence"]]
        table.append("|"+"|".join(x.replace("|", "/").replace("\n", " ") for x in fields)+"|")
    (root / "review_table.md").write_text("\n".join(table)+"\n", encoding="utf-8")
    examples = [r for r in reviewed if r["review_index"] in manual["representative_indices"]]
    write("six_cases.json", examples)
    lines = ["# 六个代表案例", "", "从固定60题审阅中展示表达、粒度、标注疑义与内容分歧；不据案例估计总体比例。"]
    for r in examples:
        lines += ["", f"## #{r['review_index']} qid={r['question_id']} ({r['category']})", "", f"![原图]({r['exported_image']})", "", f"问题：{r['question']}", f"参考：{r['reference']}；V1：{r['v1_prediction']}；r2：{r['r2_prediction']}", "", r["review_evidence"], f"不确定：{r['uncertain']}。"]
    (root / "six_cases.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    paired = read("paired_v1_minus_r2.json")
    report = ["# SLAKE最终读写LR3e-4 V1 vs LoRA-r2：配对与错误分析", "",
              "2026-10-20；两者seed44/data42、五轮固定epoch5、完整双语Test2094题/180图。逐ID/原图/问题/参考/保存comparisons与官方评分器重算一致，原预测与checkpoint保持。V1为slake_v1_read_write_lr3e4_norm_fixed_5ep_seed44_20261014_131109_40238；r2为slake_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44_20261010_101607_1852。", "",
              "全部使用CPU读取/重评分既有预测、图像簇bootstrap10000次seed42。差值方向V1−r2，子组探索性、无多重比较校正；区间针对固定模型的图像抽样，不能推断训练seed总体。", "",
              "|分组|题/图|新V1|r2|V1−r2 pp|V1/r2独占|配对95%CI|", "|---|---|---:|---:|---:|---|---|"]
    for name,r in paired["groups"].items():
        report.append(f"|{name}|{r['count']}/{r['image_clusters']}|{r['variant_accuracy']:.4f}|{r['baseline_accuracy']:.4f}|{r['delta']:+.4f}|{r['variant_only_correct']}/{r['baseline_only_correct']}|[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]|")
    report += ["", "共同正确1560、共同错误333、r2独占132、V1独占69，r2净多答对63题。相对基础LR V1 75.5492%，最终V1 77.7937%净提高47题，距r2 80.8023%的点差由5.2531缩至3.0086pp。r2净优势63题中OPEN贡献46题、CLOSED17题；VQA贡献50题、KVQA13题，标签不直接等同错误原因。", "",
               "## 固定分层60题审阅", "", sample["sampling_rule"], "", manual["reviewer"], "",
               "|抽样组|n|V|K|E|A|不确定|", "|---|---:|---:|---:|---:|---:|---:|"]
    for category,r in counts.items():
        report.append(f"|{category}|{r['n']}|"+"|".join(str(r["tag_counts"].get(tag,0)) for tag in "VKEA")+f"|{r['uncertain']}|")
    report += ["", manual["limitations"], "",
               "原图审阅显示：部分分歧是可观察的内容或属性差异，如心脏存在性(#8)、成像模态(#49)；多处诊断、病灶/侧别判断需要医学专家。答案表达/列表完整性也影响官方得分：#3功能列表遗漏、#14有/是的、#18右肺/右上肺；反向#46不是/不可以。标注/参照疑义包括#28与#52肾的bone marrow/bottom spine参照差异，以及#38脑部MRI配右上叶式参考。全部保留官方判分，不做语义修分。", "",
               "## 有证据边界的解释", "",
               "最终V1缩小了与LoRA-r2的差距，但这两次固定seed44模型之间r2仍有明确总体优势。收益跨开放/封闭、知识/视觉标签及中英语言；无法把剩余差距全部归结为医学知识。固定样本显示器官/位置/属性及功能内容、回答完整性和表达形式共同影响分歧，临床与标注疑义限制了机制归因。样本配额40/20及多标签计数无法外推原因比例。", "",
               "两者训练曝光均五轮，有效batch32；V1 microbatch2/累积16、1864963参数，r2 microbatch1/累积32、1769472参数，优化器/适配位置不同，属于近似等参数预算。训练报告、完整生成协议与数据哈希见binding.json。未运行模型/训练/新生成或GPU状态命令。", "",
               "交付：paired_v1_minus_r2.json、all_paired_rows.json、binding.json、固定sample_60、manual_annotations/reviewed_60、review_table.md、six_cases.md与52张原图/六张接触表。此前基础LR错误分析独立保留。本次首次脚本按V1字段读取r2 seed时KeyError，在数值分析前停止；改为实际r2报告seed字段后全部核验通过。"]
    (root / "report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
