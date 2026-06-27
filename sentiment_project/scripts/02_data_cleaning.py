# -*- coding: utf-8 -*-
"""
第二步：数据清洗与预处理
======================
目标：用Pandas对原始数据进行清洗，包括去重、空值处理、文本规范化。

AI训练师核心技能：数据清洗 + 数据质量把控
"""

import pandas as pd
import re
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"

print("=" * 50)
print("第2步：数据清洗与预处理")
print("=" * 50)

# 读取原始数据
df = pd.read_csv(raw_dir / "chnsenticorp_train.csv", encoding="utf-8-sig")
print(f"\n原始数据量：{len(df)}条")

# ---------- 2.1 空值检查 ----------
print("\n>>> 2.1 空值检查")
null_count = df.isnull().sum()
print(null_count)
df = df.dropna(subset=["text"])
print(f"删除空值后：{len(df)}条")

# ---------- 2.2 重复值检查 ----------
print("\n>>> 2.2 重复值检查与处理")
dup_count = df.duplicated(subset=["text"]).sum()
print(f"完全重复的评论数：{dup_count}条")
# 保留重复数据中的第一条（模拟实际工作中的去重）
df = df.drop_duplicates(subset=["text"], keep="first")
print(f"去重后数据量：{len(df)}条")

# ---------- 2.3 文本规范化 ----------
print("\n>>> 2.3 文本清洗")

def clean_text(text):
    """清洗单条文本"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+", "", text)
    text = re.sub(r"[^\u4e00-\u9fff\w\s，。！？、：；""''【】（）,.!?:;\-]", "", text)
    return text

df["text_cleaned"] = df["text"].apply(clean_text)

empty_mask = df["text_cleaned"].str.len() == 0
print(f"清洗后变为空文本的数量：{empty_mask.sum()}")
df = df[~empty_mask]

# ---------- 2.4 过滤过短文本 ----------
print("\n>>> 2.4 过滤过短文本（少于5个字）")
short_mask = df["text_cleaned"].str.len() < 5
print(f"过短文本书：{short_mask.sum()}条")
df = df[~short_mask]

# ---------- 2.5 标签分布检查 ----------
print("\n>>> 2.5 标签分布检查")
label_counts = df["label"].value_counts()
print(f"正面（1）：{label_counts.get(1, 0)}条")
print(f"负面（0）：{label_counts.get(0, 0)}条")

# 保存清洗后的完整数据
df.to_csv(processed_dir / "train_cleaned.csv", index=False, encoding="utf-8-sig")
print(f"\n清洗后数据已保存：{processed_dir / 'train_cleaned.csv'}（{len(df)}条）")

# 保存一份带标签的样本（用于标注练习）
sample_size = min(50, len(df))
df_sample = df.sample(n=sample_size, random_state=42)
df_sample.to_csv(processed_dir / "sample_labeled.csv", index=False, encoding="utf-8-sig")
print(f"标注样本已保存：{processed_dir / 'sample_labeled.csv'}（{len(df_sample)}条）")

# 保存一份不带标签的样本（供手动标注练习）
df_unlabeled = df_sample.drop(columns=["label", "sentiment"])
df_unlabeled.to_csv(processed_dir / "sample_unlabeled.csv", index=False, encoding="utf-8-sig")
print(f"无标签样本已保存：{processed_dir / 'sample_unlabeled.csv'}（{len(df_unlabeled)}条）")

print("\n第2步完成！数据清洗总结：")
print(f"  原始：2400条 -> 清洗后：{len(df)}条")
