"""第 10 章 after.py —— 词 = 一组数字。

before.py 里我们看到了两件事：

1. 用编号当输入，模型学的其实是"编号的排列顺序"，不是"词的意思"。
   词表顺序一换，刚训练好的东西全部作废。
2. 编号还给词强加了一种假的顺序：分数只能沿着编号一条直线排下去。

这个文件换一个表示法：一个词不是一个数字，而是一组数字。

    苹果 = [+0.31, -1.02, +0.77, ...]      （16 个数字）

注意：这一章**我们不训练这组数字**。它一开始是随机撒的，之后再没动过。
这一章只想回答一个问题：一组随机数字，难道比一个精心安排的编号还好用？

答案是"在很多事情上确实更好用"。至于这组数字到底该怎么来 ——
那是下一章的问题。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import SGD, Tensor, cross_entropy, randn

# 第 1 章那张词表，顺序也一样。表里的位置就是"编号"。
VOCAB = "苹果 发布 新 手机 芯片 电脑 华为 小米 好吃 很 甜 香蕉 这个 真 做成 派".split()

TECH_WORDS = "发布 新 手机 芯片 电脑 华为 小米".split()
FOOD_WORDS = "好吃 很 甜 香蕉 这个 真 做成 派".split()

PINYIN_ORDER = "电脑 发布 好吃 很 华为 派 苹果 手机 甜 香蕉 小米 新 芯片 这个 真 做成".split()

# 每个词用几个数字来表示
DIM = 16
SEED = 0
STEPS = 800
LR = 0.2


def display_width(text):
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def pad(text, width):
    return text + " " * (width - display_width(text))


def build_vector_table():
    """随机给每个词一组数字。

    randn(...) 按正态分布撒随机数。这一章我们不训练它，
    所以 requires_grad=False —— 它是一张只读的表。
    """
    return randn(len(VOCAB), DIM, requires_grad=False, seed=SEED)


def train_scores(vectors, labels, steps=STEPS, lr=LR):
    """训练 分数 = 向量 · w + b。

    向量是从表里查出来的、已经定好的；被训练的只有 w（16 个数字）和 b。
    和 before.py 的任务完全一样，只有"词用什么表示"变了。
    """
    x = Tensor(np.stack(vectors))
    w = randn(DIM, requires_grad=True, seed=123)
    b = Tensor(0.0, requires_grad=True)
    one_two = Tensor(np.array([0.0, 1.0]))
    targets = np.array(labels)
    optimizer = SGD([w, b], lr=lr)

    loss = None
    for _ in range(steps):
        optimizer.zero_grad()
        score = x @ w + b
        logits = score.reshape(len(vectors), 1) * one_two
        loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    scores = np.stack(vectors) @ w.data + b.data
    return scores, loss.item()


def cosine_similarity(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def show_vectors(vectors):
    print("=" * 62)
    print(f"每个词现在是一组 {DIM} 个数字（随机撒的，没有训练过）")
    print("=" * 62)
    for word in ["苹果", "香蕉", "手机", "很"]:
        numbers = "  ".join(f"{v:+.2f}" for v in vectors[word])
        print(f"  {pad(word, 6)} = [{numbers}]")
    print()


def show_neighbours(vectors):
    """在随机向量里"找相似的词"，看看能找出什么。"""
    print("=" * 62)
    print("用余弦相似度找「最像的词」")
    print("=" * 62)
    for word in ["手机", "苹果"]:
        sims = [(other, cosine_similarity(vectors[word], vectors[other]))
                for other in VOCAB if other != word]
        sims.sort(key=lambda pair: -pair[1])
        line = "  ".join(f"{w} {s:+.2f}" for w, s in sims[:5])
        print(f"  和 {pad(word, 6)} 最像的 5 个词： {line}")
    print()


def main():
    table = build_vector_table()
    snapshot = table.data.copy()  # 留个底，最后确认它没被动过
    vectors = {word: table.data[i] for i, word in enumerate(VOCAB)}

    show_vectors(vectors)

    words = TECH_WORDS + FOOD_WORDS
    labels = [0] * len(TECH_WORDS) + [1] * len(FOOD_WORDS)

    # 同一批向量，三种编号方案。
    # 关键：送给模型的是 vectors[word]，跟编号一点关系都没有。
    schemes = [
        ("A  第 1 章的词表顺序", VOCAB),
        ("B  按拼音排序", PINYIN_ORDER),
        ("C  随机打乱", [VOCAB[i] for i in np.random.default_rng(0).permutation(len(VOCAB))]),
    ]

    print("=" * 62)
    print("同一个任务（把科技词和食品词分开），三种编号方案")
    print("=" * 62)

    trained_w_b = None
    for name, order in schemes:
        ids = {word: i for i, word in enumerate(order)}
        scores, loss = train_scores([vectors[w] for w in words], labels)
        predictions = (scores > 0).astype(int)
        accuracy = float((predictions == np.array(labels)).mean())
        if trained_w_b is None:
            trained_w_b = scores
        print(f"  {pad(name, 20)} loss = {loss:.4f}   准确率 = {accuracy * 100:5.1f}%")
    print()
    print("三种编号方案的准确率一模一样。")
    print("因为模型从头到尾就没有见过编号 —— 它见到的是向量。")
    print()

    print("=" * 62)
    print("编号在变，每个词的分数一动不动")
    print("=" * 62)
    ids_a = {word: i for i, word in enumerate(VOCAB)}
    ids_b = {word: i for i, word in enumerate(PINYIN_ORDER)}
    print("  词     编号A   编号B    分数     应该属于   分数算出来的类别")
    print("  " + "-" * 60)
    for word, label, score in zip(words, labels, trained_w_b):
        truth = "科技" if label == 0 else "食品"
        guess = "科技" if score <= 0 else "食品"
        mark = "" if truth == guess else "   <-- 错"
        print(f"  {pad(word, 6)} {ids_a[word]:>4}   {ids_b[word]:>4}   {score:+.2f}    {truth}         {guess}{mark}")
    print()
    print("对比 before.py：那里分数是编号的一条直线，")
    print("这里每个词的分数是它自己的向量和 w 的点积 —— 想给谁高分就给谁高分。")
    print()

    show_neighbours(vectors)

    print("=" * 62)
    print("最后确认一下：表里的数字从头到尾没有变过")
    print("=" * 62)
    print(f"  表里一共 {len(VOCAB)} x {DIM} = {len(VOCAB) * DIM} 个数字")
    print(f"  训练前后最大的变化量 = {np.abs(table.data - snapshot).max():.7f}")
    print(f"  被训练的只有 w 和 b，一共 {DIM + 1} 个数，表本身是只读的。")


if __name__ == "__main__":
    main()
