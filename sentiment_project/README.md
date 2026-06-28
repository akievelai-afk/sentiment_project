# 酒店评论情感分析系统

基于 Scikit-learn 构建的中文情感分析模型，对携程酒店评论进行正面/负面情感分类。

## 项目概述

- 数据来源：7766 条携程真实酒店评论（ChnSentiCorp 数据集）
- 技术方案：TF-IDF + 逻辑回归
- 验证集准确率：74.5%

## 项目结构

`
├── scripts/
│   ├── 01_load_data.py     # 加载原始数据
│   ├── 02_clean_data.py    # 数据清洗（去重、空值处理）
│   ├── 03_train_model.py   # 模型训练与评估
│   ├── 04_predict.py       # 模型预测演示
│   └── 05_prompt_design.py # Prompt工程练习
├── data/
│   ├── raw/                # 原始数据
│   └── processed/          # 清洗后数据
└── output/                 # 训练结果与评估报告
`

## 运行方式

`ash
pip install pandas scikit-learn
python scripts/01_load_data.py
python scripts/02_clean_data.py
python scripts/03_train_model.py
`

## 项目要点

- 使用 Pandas 完成数据清洗（去重、缺失值处理、文本规范化）
- 使用 TF-IDF 进行文本特征提取
- 使用 Logistic Regression 构建分类模型
- 模型在验证集上准确率 74.5%，学习了 "值得推荐"/"太差了" 等关键情感词汇
- 设计了 Prompt 模板，对比传统模型与大语言模型的效果差异

## 技术栈

Python, Pandas, Scikit-learn, Jupyter Notebook
