# 01_load_real_data.py
# 直接从CSV加载真实评论数据

import pandas as pd

df = pd.read_csv("data/raw/reviews_real.csv")
print("总数据量：" + str(len(df)) + "条")

print("标签分布：")
print(df["label"].value_counts())

df.to_csv("data/raw/reviews.csv", index=False)
print("已保存到 data/raw/reviews.csv")