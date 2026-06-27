# 03_train_model.py
# 目标：训练情感分类模型

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. 加载清洗后的数据
df = pd.read_csv("data/processed/train_clean.csv")
texts = df["review"].tolist()
labels = df["label"].tolist()

# 2. 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
print("训练集：" + str(len(X_train)) + "条  验证集：" + str(len(X_val)) + "条")

# 3. 把文字转成数字（TF-IDF）
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
print("词汇表大小：" + str(len(vectorizer.get_feature_names_out())))

# 4. 训练模型
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 5. 评估
train_score = model.score(X_train_vec, y_train)
val_score = model.score(X_val_vec, y_val)
print("训练集准确率：" + "{:.2%}".format(train_score))
print("验证集准确率：" + "{:.2%}".format(val_score))

# 6. 查看模型学到了什么
feat_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]

print("\n最代表 正面 的词：")
for name in feat_names[coef.argsort()[-5:]]:
    print("  + " + name)

print("\n最代表 负面 的词：")
for name in feat_names[coef.argsort()[:5]]:
    print("  - " + name)
