# -*- coding: utf-8 -*-
"""
第三步：数据标注实践
==================
目标：模拟真实的数据标注流程，理解标注标准和质量控制。

AI训练师核心技能：数据标注 + 标注规范制定 + 质量检查
"""

import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"
processed_dir = data_dir / "processed"
output_dir = Path(__file__).parent.parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("第3步：数据标注实践")
print("=" * 50)

# 读取无标签样本
df = pd.read_csv(processed_dir / "sample_unlabeled.csv", encoding="utf-8")
print(f"\n待标注样本数：{len(df)}条")
print("（实际工作中，你会拿到这样的原始数据，需要手动标注情感分类）")

# ---------- 3.1 制定标注规范 ----------
print("\n>>> 3.1 标注规范说明")
print("""
情感分类标注规范：
----------------
任务：判断每条评论的情感倾向

标签定义：
  1（正面）：好评，表达满意、喜欢、推荐等积极情绪
  0（负面）：差评，表达不满、失望、批评等消极情绪

标注原则：
  - 只看评论本身的语气，不要脑补背景信息
  - 如果评论同时包含正面和负面内容，判断整体倾向
  - 无法判断的标记为-1（但本任务中尽量避免）

质量控制：
  - 每条标注后自我检查一遍
  - 标注完成后，抽查10%的数据做一致性检查
""")

# ---------- 3.2 自动标注（模拟人工标注）----------
print("\n>>> 3.2 标注过程演示")
print("（注：这里用已有标签演示标注流程。实际工作中，你将在LabelStudio等工具中逐条标注）")
print()

# 加载带标签的数据来模拟标注结果
df_labeled = pd.read_csv(processed_dir / "sample_labeled.csv", encoding="utf-8")

# 模拟标注记录
annotations = []
for idx, row in df_labeled.iterrows():
    text = row["text_cleaned"]
    label = row["label"]
    
    # 模拟"标注员"的判断依据
    positive_words = ["不错", "喜欢", "推荐", "好", "满意", "值得", "赞", "棒", "方便", 
                      "快", "漂亮", "实用", "性价比", "舒服", "干净", "热情"]
    negative_words = ["差", "失望", "后悔", "垃圾", "不好", "太差", "慢", "贵", "破",
                      "烂", "不行", "问题", "投诉", "退货", "差评"]
    
    # 统计正面/负面词出现次数
    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)
    
    confidence = "高" if abs(pos_count - neg_count) >= 2 else "中" if abs(pos_count - neg_count) >= 1 else "低"
    
    annotations.append({
        "text": text[:60] + "...",
        "标注结果": "正面" if label == 1 else "负面",
        "正面词数": pos_count,
        "负面词数": neg_count,
        "信心度": confidence
    })

df_annotations = pd.DataFrame(annotations)

# 显示部分标注结果
print("标注示例（前10条）：")
print(df_annotations.head(10).to_string(index=False))

# ---------- 3.3 标注质量检查 ----------
print("\n\n>>> 3.3 质量检查（一致性校验）")

# 模拟第二次标注（用于一致性检查）
df_labeled["label_round2"] = df_labeled["label"]  # 假设完全一致
# 实际工作中会有不同标注员标注同一批数据

# 计算一致率
consistency = (df_labeled["label"] == df_labeled["label_round2"]).mean()
print(f"标注一致率：{consistency:.1%}")
print("（行业标准：一般要求标注一致率 >= 95%）")

if consistency >= 0.95:
    print("结果：通过！标注质量合格")
else:
    print("结果：未通过，需要重新标注不一致的样本")

# ---------- 3.4 导出标注结果 ----------
# 导出标注数据（JSON格式，模拟提交给训练环节）
import json

annotation_records = []
for _, row in df_labeled.iterrows():
    annotation_records.append({
        "text": row["text_cleaned"],
        "label": int(row["label"]),
        "sentiment": "正面" if row["label"] == 1 else "负面"
    })

output_file = output_dir / "annotation_result.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(annotation_records, f, ensure_ascii=False, indent=2)
print(f"\n标注结果已导出：{output_file}（{len(annotation_records)}条）")

# 导出标注报告
report = pd.DataFrame({
    "指标": ["总标注量", "正面数量", "负面数量", "标注一致率"],
    "数值": [len(annotation_records), 
             sum(1 for r in annotation_records if r["label"] == 1),
             sum(1 for r in annotation_records if r["label"] == 0),
             f"{consistency:.1%}"]
})
print("\n标注报告：")
print(report.to_string(index=False))

# 保存标注报告
report_file = output_dir / "annotation_report.csv"
report.to_csv(report_file, index=False, encoding="utf-8")
print(f"标注报告已保存：{report_file}")

print("\n第3步完成！你现在理解了数据标注的完整流程。")

