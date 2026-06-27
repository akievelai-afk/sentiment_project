# 02_clean_data.py
# 目标：清洗数据

import pandas as pd

df = pd.read_csv("data/raw/reviews.csv")
print("原始数据量：", len(df))

# 1. 删除空值
df = df.dropna()
print("删除空值后：", len(df))

# 2. 删除重复评论
df = df.drop_duplicates(subset=["review"])
print("去重后：", len(df))

# 3. 去掉太短的评论（少于5个字）
df = df[df["review"].str.len() >= 5]
print("过滤短文本后：", len(df))

# 4. 查看正负比例
print("正面：", len(df[df["label"]==1]))
print("负面：", len(df[df["label"]==0]))

df.to_csv("data/processed/train_clean.csv", index=False)
print("清洗完成！")