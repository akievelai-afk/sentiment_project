# 从0到1动手做项目 -- 完整操作指南

这份指南教你亲手打出每一行代码，从零搭建一个AI训练师实战项目。

---

## 准备工作（10分钟）

### 确认环境

打开命令行（Win+R 输入 cmd），依次输入以下命令，确认都有输出：

```
python --version
pip --version
```

如果没有Python，去 https://www.python.org 下载安装 Python 3.8+。

然后安装依赖包：
```
pip install pandas numpy scikit-learn openpyxl
```

### 创建项目文件夹

建一个文件夹叫 sentiment_project，里面再建3个文件夹：

```
sentiment_project/
  data/raw/          -- 放原始数据
  data/processed/    -- 放清洗后的数据
  scripts/           -- 放你写的Python脚本
  output/            -- 放模型和报告
```

用命令行创建：
```
mkdir sentiment_project\data\raw sentiment_project\data\processed sentiment_project\scripts sentiment_project\output
```

---

## 第一步：创建数据集（30分钟）

在 scripts 文件夹里新建 01_create_data.py，手动敲入以下代码：

```
# 01_create_data.py
# 目标：生成中文电商评论数据

import pandas as pd
import random

random.seed(42)

positive_reviews = [
    "质量很好，物流也很快，好评！",
    "东西非常好，值得购买，推荐给大家。",
    "性价比很高，比实体店便宜很多。",
    "包装很严实，没有破损，满意！",
    "第二次购买了，一如既往的好。",
]

negative_reviews = [
    "质量太差了，用了几天就坏了。",
    "物流太慢了等了一个星期才到。",
    "和图片完全不一样，有色差。",
    "客服态度很差，完全解决不了问题。",
    "非常后悔购买，不值这个价。",
]

data = []
for i in range(500):
    text = random.choice(positive_reviews)
    data.append({"text": text, "label": 1})

for i in range(500):
    text = random.choice(negative_reviews)
    data.append({"text": text, "label": 0})

random.shuffle(data)

df = pd.DataFrame(data)
df.to_csv("data/raw/reviews.csv", index=False, encoding="utf-8-sig")

print(f"生成了 {len(df)} 条评论")
print(f"正面: {len(df[df['label']==1])} 条")
print(f"负面: {len(df[df['label']==0])} 条")
```

运行：
```
cd sentiment_project
python scripts/01_create_data.py
```

---

## 第二步：数据清洗（30分钟）

新建 scripts/02_clean_data.py：

```
# 02_clean_data.py
# 目标：清洗数据

import pandas as pd

df = pd.read_csv("data/raw/reviews.csv")
print("原始数据量：", len(df))

# 1. 删除空值
df = df.dropna()
print("删除空值后：", len(df))

# 2. 删除重复评论
df = df.drop_duplicates(subset=["text"])
print("去重后：", len(df))

# 3. 去掉太短的评论（少于5个字）
df = df[df["text"].str.len() >= 5]
print("过滤短文本后：", len(df))

# 4. 查看正负比例
print("正面：", len(df[df["label"]==1]))
print("负面：", len(df[df["label"]==0]))

df.to_csv("data/processed/train_clean.csv", index=False)
print("清洗完成！")
```

运行：
```
python scripts/02_clean_data.py
```

---

## 第三步：训练模型（30分钟）

新建 scripts/03_train_model.py：

```
# 03_train_model.py
# 目标：训练情感分类模型

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/train_clean.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
print(f"训练集：{len(X_train)}条  验证集：{len(X_val)}条")

vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
print(f"词汇表大小：{len(vectorizer.get_feature_names_out())}")

model = LogisticRegression()
model.fit(X_train_vec, y_train)

train_score = model.score(X_train_vec, y_train)
val_score = model.score(X_val_vec, y_val)
print(f"训练集准确率：{train_score:.2%}")
print(f"验证集准确率：{val_score:.2%}")

# 看看模型学到了哪些关键词
feat_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]

print("\n最代表正面的词：")
for name in feat_names[coef.argsort()[-5:]]:
    print(f"  + {name}")

print("\n最代表负面的词：")
for name in feat_names[coef.argsort()[:5]]:
    print(f"  - {name}")
```

运行：
```
python scripts/03_train_model.py
```

---

## 第四步：预测新评论（20分钟）

新建 scripts/04_predict.py：

```
# 04_predict.py
# 目标：用模型预测新评论

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/processed/train_clean.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

test_reviews = [
    "这个手机质量很好，用着很舒服",
    "物流太慢了，等得很着急",
    "价格便宜质量也好，推荐",
    "客服态度很差，再也不来了",
]

for review in test_reviews:
    vec = vectorizer.transform([review])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    sentiment = "正面" if pred == 1 else "负面"
    print(f"评论：{review}")
    print(f"预测：{sentiment}（信心度：{prob:.1%}）")
    print()
```

运行：
```
python scripts/04_predict.py
```

---

## 第五步：Prompt工程练习（15分钟）

新建 scripts/05_prompt_practice.py：

```
# 05_prompt_practice.py
# 目标：练习设计Prompt

print("把下面的Prompt模板复制到ChatGPT或文心一言测试\n")

prompt = """你现在是一个电商评论分析助手。
请判断以下评论的情感倾向。

评论：{review}

请按以下格式回答：
情感：【正面/负面】
理由：简要说明判断依据"""

test_cases = [
    "质量很好，值得推荐",
    "用了三天就坏了，质量堪忧",
    "客服非常耐心，问题都解决了",
    "发货太慢了等了一个星期",
]

for i, review in enumerate(test_cases, 1):
    print(f"测试 {i}：{review}")
    print(prompt.replace("{review}", review))
    print("-" * 30)

print("进阶技巧：")
print("1. Few-shot：在prompt里先给2-3个例子")
print("2. Chain-of-Thought：让AI解释推理过程")
print("3. 角色扮演：告诉AI它是情感分析专家")
```

运行：
```
python scripts/05_prompt_practice.py
```

---

## 把项目放上GitHub（10分钟）

```
git init
git add .
git commit -m "电商评论情感分析项目"
```

去 github.com 新建一个仓库，然后：
```
git remote add origin https://github.com/你的用户名/sentiment_project.git
git push -u origin master
```

---

## 常见报错解决

| 报错 | 原因 | 解决方法 |
|------|------|----------|
| ModuleNotFoundError | 没装包 | pip install 包名 |
| SyntaxError | 语法错误 | 检查括号和引号 |
| FileNotFoundError | 路径不对 | 确认在项目目录下运行 |
| IndentationError | 缩进不对 | Python用4个空格缩进 |

不要怕报错，报错信息会告诉你是哪一行出错了，仔细看就能修。

---

## 做完后简历里可以写

个人项目：电商评论情感分析
- 用Python和Scikit-learn构建中文情感分类模型
- 完成数据清洗、TF-IDF特征提取、逻辑回归模型训练全流程
- 模型在验证集上准确率达到XX%
- 掌握Prompt Engineering，能设计有效的AI对话指令

动手做一遍比看十遍教程都有用。
