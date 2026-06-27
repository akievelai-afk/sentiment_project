# -*- coding: utf-8 -*-
"""
第五步：模型评估与Prompt工程实践
==========================
目标：1. 用实际案例评估模型效果
      2. 练习Prompt Engineering（提示词工程）
      3. 生成项目报告

AI训练师核心技能：模型评估 + Prompt工程 + 报告撰写
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json

data_dir = Path(__file__).parent.parent / "data"
processed_dir = data_dir / "processed"
output_dir = Path(__file__).parent.parent / "output"
model_dir = output_dir / "model"
report_dir = output_dir / "report"
report_dir.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("第5步：模型评估与Prompt工程实践")
print("=" * 50)

# ---------- 5.1 加载模型 ----------
print("\n>>> 5.1 加载训练好的模型")

with open(model_dir / "tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open(model_dir / "classifier.pkl", "rb") as f:
    model = pickle.load(f)

print("模型加载成功！")

# ---------- 5.2 用模型预测新评论 ----------
print("\n>>> 5.2 模型实战测试")

# 一些全新的评论（训练集中没出现过的）
test_reviews = [
    "这个手机质量不错，用了一个月都没问题，推荐购买",
    "物流太慢了，等了十天还没到，差评差评差评",
    "客服态度很好，耐心回答了我的所有问题",
    "质量很差，用了几天就坏了，不要买",
    "价格便宜，质量也很好，性价比超高",
    "包装破损严重，里面的东西都压变形了",
    "第二次回购了，一如既往的好，五星好评",
    "颜色和图片不一样，有色差，很失望",
]

print("正在测试模型在新评论上的表现...")
print()

test_texts = [r.replace("，", " ").replace("！", " ") for r in test_reviews]
X_test = vectorizer.transform(test_texts)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"{'评论内容':<35} {'预测结果':<8} {'信心度':<8}")
print("-" * 55)
for i, review in enumerate(test_reviews):
    pred = "正面" if predictions[i] == 1 else "负面"
    confidence = max(probabilities[i]) * 100
    print(f"{review[:33]:<35} {pred:<8} {confidence:.1f}%".format(review[:33], pred, confidence))

# ---------- 5.3 Prompt工程实践 ----------
print("\n\n" + "=" * 50)
print("第5步副任务：Prompt Engineering 实践")
print("=" * 50)

print("""
什么是Prompt Engineering（提示词工程）？
----------------------------------------
给AI（如ChatGPT、文心一言）写的"指令"就叫Prompt。
Prompt工程就是设计这些指令的技巧，让AI输出你想要的结果。
这是AI训练师的核心技能之一。

练习：设计一个情感分类的Prompt
----------------------------------------
""")

# 定义Prompt模板
prompt_template = """
你是一个专业的情感分析助手。请判断以下评论的情感倾向是"正面"还是"负面"。

评分标准：
- 正面：好评，表达满意、喜欢、推荐等积极情绪
- 负面：差评，表达不满、失望、批评等消极情绪

评论：{review}

请只回答"正面"或"负面"，不要输出其他内容。
"""

# 用几条例句测试
test_prompts = [
    "质量很好，物流很快，非常满意",
    "用了三天就坏了，质量太差了",
    "性价比很高，值得推荐",
    "客服态度很差，问问题半天不回",
]

print("Prompt示例（可直接复制到ChatGPT/文心一言中测试）：")
print("-" * 50)
for i, review in enumerate(test_prompts):
    print(f"\n--- 测试评论 {i+1} ---")
    prompt = prompt_template.format(review=review)
    print(prompt)
    print("（把上面的Prompt复制到ChatGPT或文心一言中，看看AI的回答）")

# ---------- 5.4 高级Prompt技巧 ----------
print("\n\n" + "=" * 50)
print("高级Prompt技巧（面试加分项）")
print("=" * 50)

advanced_prompt = """
### Few-shot Prompt（少样本提示）
给AI几个例子，它就会按例子来回答：

判断以下评论的情感：

例子1：
评论：这个产品质量很好，非常满意
情感：正面

例子2：
评论：物流太慢了，快递员态度很差
情感：负面

例子3：
评论：性价比不错，价格实惠
情感：正面

现在请判断：
评论：{review}
情感：
"""

print("""
技巧1：Few-shot Prompt（给例子）
  在Prompt里先给2-3个例子，AI就能理解你的要求。
  
技巧2：Chain-of-Thought（让AI解释）
  让AI先解释为什么，再给出判断。

技巧3：Role Prompting（设定角色）
  先告诉AI它是什么角色，再给任务。
  例如："你是一个专业的情感分析专家..."

这些技巧在面试中经常被问到，建议提前练习！
""")

# ---------- 5.5 生成项目报告 ----------
print("\n" + "=" * 50)
print("生成项目总结报告")
print("=" * 50)

# 加载训练摘要
with open(output_dir / "training_summary.json", "r", encoding="utf-8") as f:
    training_summary = json.load(f)

report_content = f"""
# 项目总结报告

## 1. 项目概述
- 项目名称：中文电商评论情感分析
- 项目类型：AI训练师入门实战项目
- 模型方法：TF-IDF + 逻辑回归

## 2. 数据处理
- 原始数据量：6000条
- 清洗后数据量：{training_summary["training_samples"] + training_summary["validation_samples"]}条
- 数据划分：训练集 {training_summary["training_samples"]}条 / 验证集 {training_summary["validation_samples"]}条

## 3. 模型效果
- 训练集准确率：{training_summary["train_accuracy"]}
- 验证集准确率：{training_summary["validation_accuracy"]}
- 模型类型：{training_summary["model_type"]}
- 词汇表大小：{training_summary["vocab_size"]}个词

## 4. 模型学到的关键特征
### 最代表"正面"的词：
{", ".join(training_summary["positive_words"])}

### 最代表"负面"的词：
{", ".join(training_summary["negative_words"])}

## 5. 核心技能总结
通过本项目，练习了以下AI训练师必备技能：
1. 数据清洗与预处理
2. 数据标注流程
3. 特征工程与模型训练
4. 模型评估与分析
5. Prompt Engineering

## 6. 项目文件结构
项目路径：sentiment_project/
- data/raw/         - 原始数据
- data/processed/   - 清洗后的数据
- scripts/          - Python脚本
- output/model/     - 训练好的模型
- output/report/    - 评估报告

## 7. 面试提示
面试官可能会问：
- 你是怎么清洗数据的？遇到了什么问题？
- 为什么选择这个模型？
- 模型评估指标有哪些？各代表什么意思？
- 如果数据不平衡怎么办？
- 什么是Prompt Engineering？你用过哪些技巧？

建议：把项目放到GitHub上，面试时直接给面试官看。
"""

report_file = report_dir / "project_report.md"
with open(report_file, "w", encoding="utf-8") as f:
    f.write(report_content)
print(f"项目报告已生成：{report_file}")

# 提示面试准备
print(f"\n所有步骤完成！项目文件汇总：")
print(f"  - 数据文件：{data_dir / 'raw / chnsenticorp_train.csv'}")
print(f"  - 训练模型：{model_dir}")
print(f"  - 评估报告：{report_file}")
print(f"  - 训练摘要：{output_dir / 'training_summary.json'}")
print()
print("=" * 50)
print("项目全部完成！你现在可以：")
print("1. 把这个项目放到GitHub上")
print("2. 把报告链接放进简历")
print("3. 面试时展示你做的完整项目")
print("=" * 50)
