# -*- coding: utf-8 -*-
"""
第四步：模型训练（使用传统机器学习方法）
======================
目标：用清洗好的数据训练一个文本分类模型。
      这里使用TF-IDF + 逻辑回归，效果接近简单的BERT但无需下载大模型。

知识点：
- TF-IDF：将文本转换为数学向量（词频-逆文档频率）
- 逻辑回归：简单但有效的分类算法
- 训练/验证集划分：评估模型在未见过的数据上的表现

AI训练师核心技能：特征提取 + 模型训练 + 参数调优
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import pickle
import re
# jieba not needed for char-based TF-IDF

data_dir = Path(__file__).parent.parent / "data"
processed_dir = data_dir / "processed"
output_dir = Path(__file__).parent.parent / "output"
model_dir = output_dir / "model"
model_dir.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("第4步：模型训练")
print("=" * 50)

print("""
核心概念：
----------------------------------------
1. TF-IDF：把"文字"变成"数字"
   一条评论"质量很好" -> 计算机不理解文字
   TF-IDF把它转成 [0.2, 0.5, 0.1, ...] 这样的数字向量
   每一条评论变成一个数字列表
   
2. 逻辑回归：根据数字做判断
   输入数字向量 -> 输出"是正面"或"是负面"
   就像：看到这些关键词 -> 判断情感倾向

3. 训练 vs 验证
   训练集：让模型学习的"练习题"
   验证集：测试模型学得怎么样的"考试题"
""")

# ---------- 4.1 加载数据 ----------
print("\n>>> 4.1 加载清洗后的数据")

df_train = pd.read_csv(processed_dir / "train_cleaned.csv", encoding="utf-8-sig")
df_train = df_train[["text_cleaned", "label"]].rename(columns={"text_cleaned": "text"})

print(f"总数据量：{len(df_train)}条")
print(f"正面：{(df_train['label']==1).sum()}条 | 负面：{(df_train['label']==0).sum()}条")

# ---------- 4.2 数据划分 ----------
print("\n>>> 4.2 划分训练集和验证集")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_train["text"].tolist(),
    df_train["label"].tolist(),
    test_size=0.2,       # 20%作为验证集
    random_state=42,
    stratify=df_train["label"].tolist()  # 确保正负比例一致
)

print(f"训练集：{len(train_texts)}条 | 验证集：{len(val_texts)}条")

# ---------- 4.3 文本向量化（TF-IDF） ----------
print("\n>>> 4.3 文本向量化（TF-IDF：把文字变成数字）")

# 创建TF-IDF向量化器
vectorizer = TfidfVectorizer(
    max_features=5000,     # 最多使用5000个关键词
    ngram_range=(1, 2),    # 考虑单个词和双词组合
    min_df=2,              # 至少在2条评论中出现过的词才保留
    max_df=0.8             # 在超过80%的评论中出现的词忽略（太常见没区分度）
)

# 在训练集上"学习"词汇表
print("正在学习词汇表...")
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

print(f"词汇表大小：{len(vectorizer.get_feature_names_out())}个词")
print(f"训练集向量形状：{X_train.shape}")
print(f"验证集向量形状：{X_val.shape}")

# 显示一些重要的词
print("\n重要词汇示例：")
feature_names = vectorizer.get_feature_names_out()
word_importance = np.array(X_train.sum(axis=0)).flatten()
top_indices = word_importance.argsort()[-20:][::-1]
print("出现频率最高的20个词：")
for idx in top_indices[:10]:
    print(f"  {feature_names[idx]}: {word_importance[idx]:.0f}次")

# ---------- 4.4 训练模型 ----------
print("\n>>> 4.4 训练逻辑回归模型")

model = LogisticRegression(
    C=1.0,           # 正则化强度的倒数（越小泛化能力越强）
    max_iter=1000,   # 最大迭代次数
    random_state=42,
    n_jobs=-1        # 使用所有CPU核心
)

print("正在训练模型（预计1-2分钟）...")
model.fit(X_train, train_labels)

# ---------- 4.5 查看模型学到的重要特征 ----------
print("\n>>> 4.5 特征重要性分析（模型学到了什么？）")

# 获取每个特征（词）对模型的权重
feature_importance = pd.DataFrame({
    "feature": feature_names,
    "weight": model.coef_[0]
}).sort_values("weight", ascending=False)

print("\n最能预测【正面】的10个词（权重最高）：")
print(feature_importance.head(10).to_string(index=False))

print("\n最能预测【负面】的10个词（权重最低）：")
print(feature_importance.tail(10).to_string(index=False))

# ---------- 4.6 在验证集上评估 ----------
print("\n>>> 4.6 验证集评估")

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

train_acc = accuracy_score(train_labels, train_pred)
val_acc = accuracy_score(val_labels, val_pred)

print(f"训练集准确率：{train_acc:.4f}")
print(f"验证集准确率：{val_acc:.4f}")
print(f"过拟合程度（训练集-验证集）：{train_acc - val_acc:.4f}")

# 如果训练集准确率远高于验证集，说明过拟合了（模型死记硬背）
if train_acc - val_acc > 0.1:
    print("⚠️ 存在一定过拟合，可以通过增加数据量或调整参数改善")

# ---------- 4.7 保存模型和向量化器 ----------
print("\n>>> 4.7 保存模型")

with open(model_dir / "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open(model_dir / "classifier.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"模型已保存到：{model_dir}")

# 保存训练结果摘要
training_summary = {
    "model_type": "TF-IDF + Logistic Regression",
    "vocab_size": len(feature_names),
    "training_samples": len(train_texts),
    "validation_samples": len(val_texts),
    "train_accuracy": round(float(train_acc), 4),
    "validation_accuracy": round(float(val_acc), 4),
    "positive_words": feature_importance.head(10)["feature"].tolist(),
    "negative_words": feature_importance.tail(10)["feature"].tolist()
}
with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
    json.dump(training_summary, f, ensure_ascii=False, indent=2)
print(f"训练摘要已保存：{output_dir / 'training_summary.json'}")

print("\n第4步完成！模型训练结束。")
print("下一步：python scripts/05_evaluate.py")


