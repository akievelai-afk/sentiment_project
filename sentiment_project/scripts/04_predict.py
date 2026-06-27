# 04_predict.py
# 目标：用训练好的模型预测新评论

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 重新加载数据训练
df = pd.read_csv("data/processed/train_clean.csv")
texts = df["review"].tolist()
labels = df["label"].tolist()

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

# 测试新评论
test_reviews = [
    "房间很干净，服务态度也很好，推荐",
    "设施太旧了，卫生间漏水，体验很差",
    "位置很好找，前台办理入住很快",
    "隔音效果太差了，隔壁说话都能听到",
]

print("模型预测结果：")
print("-" * 40)
for review in test_reviews:
    vec = vectorizer.transform([review])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    sentiment = "正面" if pred == 1 else "负面"
    print("评论：" + review)
    print("预测：" + sentiment + "（信心度：" + "{:.1%}".format(prob) + "）")
    print()
