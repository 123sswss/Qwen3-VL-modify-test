from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm


def fill_report(json_data):
    doc = DocxTemplate("C:/Users/11473/Desktop/故障分析报告表.docx")

    # 图片处理逻辑
    # 假设图片就是 VLM 刚才看的那张，我们将其插入到 fault_image 标签处
    img_obj = InlineImage(doc, json_data['fault_image_path'], width=Mm(65))

    # 构造填充数据字典
    context = {
        "device_name": json_data['device_name'],
        "fault_image": img_obj,
        "description": json_data['description'],
        "risks": json_data['risks'],
        "suggestions": json_data['suggestions']
    }

    doc.render(context)
    doc.save("Generated_Report.docx")

if __name__ == "__main__":
    json_data = {
      "device_name": "工业级离心泵",
      "fault_image_path": "C:/Users/11473/Desktop/test.png",
      "description": "泵体左侧密封圈处出现明显渗油现象，伴随轻微金属摩擦异响，压力示数不稳定。",
      "risks": "持续漏油可能导致电机短路或轴承烧毁，存在生产线非计划停机的重大风险。",
      "suggestions": "1. 立即停止泵机运行；2. 拆解密封端盖更换 O 型圈；3. 检查轴承润滑情况并添加润滑油。"
    }
    fill_report(json_data)