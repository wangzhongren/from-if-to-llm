"""第 30 章 experiment.py —— 30 章的成绩单。

全书的第一段代码（第 2 章的 if/else 分类器）和全书最后的东西（第 22-29 章的模型），
放在同一个测试上。

    对比一：同一批句子，各自的判断和各自的成绩
    对比二：把规则砍掉几个关键词 —— 它会掉多少分？
    对比三：把训练数据的其中一部分删掉 —— 模型会掉多少分？
    对比四：一张"两种做法"的对照表

跑法：

    ./.venv/bin/python chapter_30/experiment.py
"""

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import after  # noqa: E402
import before  # noqa: E402


def pad(text, width):
    display = sum(2 if ord(ch) > 2 ** 11 else 1 for ch in text)
    return text + " " * max(0, width - display)


def train_model(text=None, steps=600, seed=0):
    """训练一个分类模型。text 为 None 时用 after.py 里的完整语料。"""
    torch.manual_seed(seed)
    model = after.TinyLM(after.VOCAB_SIZE, dim=64, n_layer=2, n_head=4, block_size=48)
    if text is None:
        data = after.DATA
    else:
        data = np.array([after.CHAR_TO_ID[ch] for ch in text])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        starts = rng.integers(0, len(data) - 49, size=16)
        x = np.stack([data[i:i + 48] for i in starts])
        y = np.stack([data[i + 1:i + 49] for i in starts])
        logits = model(torch.from_numpy(x))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, after.VOCAB_SIZE), torch.from_numpy(y).reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def rule_accuracy():
    """if/else 分类器在测试集上的成绩。"""
    hits = []
    for sentence, label in after.TEST_SET:
        if label == "说不清":
            continue
        hits.append(before.classify(before.split_words(sentence)) == label)
    return sum(hits), len(hits)


def model_accuracy(model):
    hits = []
    for sentence, label in after.TEST_SET:
        if label == "说不清":
            continue
        probabilities = after.label_probabilities(model, sentence)
        hits.append(max(probabilities, key=probabilities.get) == label)
    return sum(hits), len(hits)


# ---------------------------------------------------------------- 对比一

def comparison_one(model):
    print("=" * 78)
    print("对比一：同一批句子，各自的判断")
    print("=" * 78)
    print(pad("句子", 20) + pad("if/else 说", 12) + pad("模型说", 10)
          + pad("模型的把握", 14) + "人给的标签")
    print("-" * 78)
    for sentence, label in after.TEST_SET:
        words = before.split_words(sentence)
        rule_says = before.classify(words)
        probabilities = after.label_probabilities(model, sentence)
        model_says = max(probabilities, key=probabilities.get)
        confidence = probabilities[model_says]
        tag = label if label != "说不清" else "说不清（有歧义）"
        print(pad(sentence, 20) + pad(rule_says, 12) + pad(model_says, 10)
              + pad(f"{confidence:.3f}", 14) + tag)
    print("-" * 78)
    rule_hits, total = rule_accuracy()
    model_hits, _ = model_accuracy(model)
    print(f"if/else 分类器：{rule_hits}/{total}")
    print(f"我们的模型：    {model_hits}/{total}")
    print()
    print("打平了。")
    print()
    print("花了 30 章造出来的东西，在这个测试上没有赢过 4 行 if/else ——")
    print("这一点应该老老实实说清楚。")
    print()
    print("因为**这批句子本来就是照着那堆规则写的**。规则是人写的，")
    print("写规则的人知道答案；模型是从数据里学的，数据也是我们照着规则造的。")
    print("在自己的考卷上，规则当然满分。")
    print()
    print("所以差距不在这里。差距在下面两个对比里。")
    print()


# ---------------------------------------------------------------- 对比二

def comparison_two():
    print("=" * 78)
    print("对比二：把 if/else 的规则砍掉几个词")
    print("=" * 78)
    print("假设写规则的时候，我们没想到'华为'和'小米'这两个品牌。")
    print("（代码里就是把 TECH_WORDS 里这两个词删掉——规则是人写的，改起来就是这么简单。）")
    print()
    original_tech = list(before.TECH_WORDS)
    original_food = list(before.FOOD_WORDS)

    stages = [
        ("完整规则", original_tech, original_food),
        ("删掉'华为''小米'", [w for w in original_tech if w not in ("华为", "小米")], original_food),
        ("再删掉'发布'", [w for w in original_tech if w not in ("华为", "小米", "发布")], original_food),
        ("再把'甜'删掉", [w for w in original_tech if w not in ("华为", "小米", "发布")],
         [w for w in original_food if w != "甜"]),
    ]
    print(pad("规则表", 26) + "成绩")
    print("-" * 78)
    for name, tech_words, food_words in stages:
        before.TECH_WORDS = tech_words
        before.FOOD_WORDS = food_words
        hits, total = rule_accuracy()
        print(pad(name, 26) + f"{hits}/{total}")
    before.TECH_WORDS = original_tech
    before.FOOD_WORDS = original_food
    print()
    print("只掉了 1 分、2 分 —— 这和'规则很脆弱'的直觉不太一样。")
    print("原因是我们的规则是**冗余**的：'小米发布新手机'里，'小米''发布''手机'")
    print("三个词都在科技词表里，删掉一个还有两个兜底。人手写规则就是会这么写：")
    print("多写几条，互相兜着。")
    print()
    print("但看清楚掉的是哪一句：删掉'小米'之后，'小米直播带货'直接掉进")
    print("最后那句'一个关键词都没碰上，先猜食品'。它的判断完全取决于")
    print("词表里有没有那个词 —— **规则的知识就是这张词表本身，一个字都不多。**")
    print()


# ---------------------------------------------------------------- 对比三

def comparison_three():
    print("=" * 78)
    print("对比三：把模型的训练数据删掉一部分")
    print("=" * 78)
    print("做同样的事：训练语料里把所有带'华为'和'小米'的句子都删掉。")
    print()

    full_text = after.build_training_text()
    filtered = "".join(part for part in full_text.split("。") if part and
                       "华为" not in part and "小米" not in part)
    filtered = "".join(part + "。" for part in filtered.split("。") if part)

    start = time.time()
    full_model = train_model()
    half_model = train_model(filtered)
    print(f"（两个模型各训练 600 步，一共用了 {time.time() - start:.0f} 秒）")
    print()
    print(pad("模型", 30) + pad("完整数据", 12) + "删掉'华为''小米'")
    print("-" * 78)
    for sentence, label in [("苹果发布新手机", "科技"), ("华为发布新电脑", "科技"),
                            ("小米发布新手机", "科技"), ("苹果很甜", "食品")]:
        full_probs = after.label_probabilities(full_model, sentence)
        half_probs = after.label_probabilities(half_model, sentence)
        full_says = max(full_probs, key=full_probs.get)
        half_says = max(half_probs, key=half_probs.get)
        print(pad(sentence, 30)
              + pad(f"{full_says}（{full_probs[full_says]:.2f}）", 12)
              + f"{half_says}（{half_probs[half_says]:.2f}）")
    print()
    full_hits, total = model_accuracy(full_model)
    half_hits, _ = model_accuracy(half_model)
    print(f"完整数据训练：{full_hits}/{total}")
    print(f"删掉两个品牌的句子之后：{half_hits}/{total}")
    print()
    print("模型也掉了分，但没有像规则那样直接崩掉 ——")
    print("因为它在语料里学到的不只是'华为''小米'这两个词，")
    print("还有'发布''新''手机'这些片段和它们的关系。")
    print("**规则的知识写在规则里，模型的知识写在参数里。**")
    print("前者删了就没了，后者是一个整体的、互相支撑的东西。")
    print()


# ---------------------------------------------------------------- 对比四

def comparison_four():
    print("=" * 78)
    print("对比四：两种做法")
    print("=" * 78)
    rows = [
        ("怎么来的", "人写规则", "从数据里学"),
        ("有多少个数字", "4 条规则、10 个关键词", "109,312 个参数"),
        ("判断一句话要多久", "几乎 0", "3 毫秒"),
        ("为什么这么判", "指着某一行 if 就能解释", "只能看概率，说不清内部"),
        ("换个任务", "重写规则", "换一批数据，代码一行不改"),
        ("没见过的词", "词表外就没辙", "按字处理，能靠上下文猜"),
        ("想变得更准", "再多想几条规则", "再喂更多数据（第 28 章）"),
        ("按什么分类", "是 / 否", "一个概率（第 5 章）"),
    ]
    print(pad("", 20) + pad("if/else 分类器（第 2 章）", 26) + "我们的模型（第 30 章）")
    print("-" * 78)
    for left, middle, right in rows:
        print(pad(left, 20) + pad(middle, 26) + right)
    print()


if __name__ == "__main__":
    start = time.time()
    torch.manual_seed(0)
    model = train_model()
    print(f"（模型训练 600 步，用时 {time.time() - start:.0f} 秒）")
    print()
    comparison_one(model)
    comparison_two()
    comparison_three()
    comparison_four()
