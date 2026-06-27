# 05_prompt_practice.py
# 目标：练习设计Prompt

print("把下面的Prompt模板复制到ChatGPT或文心一言测试\n")

prompt = """你现在是一个酒店评论分析助手。
请判断以下评论的情感倾向。

评论：{review}

请按以下格式回答：
情感：【正面/负面】
理由：简要说明判断依据"""

test_cases = [
    "房间很干净，服务态度也很好，推荐",
    "设施太旧了，卫生间漏水，体验很差",
    "位置很好找，前台办理入住很快",
    "隔音效果太差了，隔壁说话都能听到",
]

for i, review in enumerate(test_cases, 1):
    print("测试 " + str(i) + "：" + review)
    print(prompt.replace("{review}", review))
    print("-" * 30)

print("进阶技巧：")
print("1. Few-shot：在prompt里先给2-3个例子")
print("2. Chain-of-Thought：让AI解释推理过程")
print("3. 角色扮演：告诉AI它是情感分析专家")
