"""第 10 章 before.py —— 词 = 一个编号。

第 9 章结束时我们手上有了 toygrad：一个能自己算梯度的引擎。
"让模型自己学"这件事终于没有障碍了 —— 只要把词喂进去。

可我们喂进去的到底是什么？回看第 1 章，每个词都被换成了一个整数：

    苹果 = 0，发布 = 1，新 = 2，……

这个整数叫 token id。它唯一的用处是让我们能区分"这是第几个词"。
编号是我们按词表顺序一个一个发下去的，除此之外它什么都不是。

这个文件用一个再简单不过的任务来检验：编号里到底有没有"意义"。

任务：把 15 个词分成两类。

    只在科技句里出现过的：发布 新 手机 芯片 电脑 华为 小米
    只在食品句里出现过的：好吃 很 甜 香蕉 这个 真 做成 派

（苹果 两边都出现，本章先把它放在一边，第 12 章专门讲它。）

模型只有一个权重 w 和一个偏置 b，输入是词的一个编号：

    分数 = w * 编号 + b

分数低的算科技词，分数高的算食品词。

然后我们换三种词表顺序，做同一个任务，看会发生什么。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import SGD, Tensor, cross_entropy

# 第 1 章那张词表，顺序也一样
VOCAB = "苹果 发布 新 手机 芯片 电脑 华为 小米 好吃 很 甜 香蕉 这个 真 做成 派".split()

# 只在科技句里出现过的词 / 只在食品句里出现过的词
TECH_WORDS = "发布 新 手机 芯片 电脑 华为 小米".split()
FOOD_WORDS = "好吃 很 甜 香蕉 这个 真 做成 派".split()

# 按拼音排序的同一张词表
PINYIN_ORDER = "电脑 发布 好吃 很 华为 派 苹果 手机 甜 香蕉 小米 新 芯片 这个 真 做成".split()

STEPS = 3000
LR = 0.1


def display_width(text):
    """中文在终端里占两格，英文数字占一格。"""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def pad(text, width):
    return text + " " * (width - display_width(text))


def build_schemes():
    """三种词表顺序，每种都给出 词 -> 编号 的对照表。"""
    rng = np.random.default_rng(0)
    shuffled = [VOCAB[i] for i in rng.permutation(len(VOCAB))]
    return [
        ("A  第 1 章的词表顺序（科技词恰好排在前面）", VOCAB),
        ("B  按拼音排序（同一批词，同一张表）", PINYIN_ORDER),
        ("C  随机打乱（同一批词，同一张表）", shuffled),
    ]


def train(ids, labels, steps=STEPS, lr=LR):
    """训练 分数 = w * 编号 + b。返回训练好的 (w, b)。

    labels 里 0 表示科技词，1 表示食品词。
    两个分数拼成 (词数, 2) 的 logits：第 0 列恒为 0，第 1 列是 score。
    softmax 之后，第 1 列就是"这个词属于食品词"的概率。
    """
    w = Tensor(0.0, requires_grad=True)
    b = Tensor(0.0, requires_grad=True)
    x = Tensor(np.array(ids, dtype=float))
    targets = np.array(labels)
    one_two = Tensor(np.array([0.0, 1.0]))
    optimizer = SGD([w, b], lr=lr)

    loss = None
    for _ in range(steps):
        optimizer.zero_grad()
        score = w * x + b
        logits = score.reshape(len(ids), 1) * one_two
        loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    return w, b, loss.item()


def evaluate(w, b, ids, labels):
    """用给定的 w、b 和编号，算准确率和每个词的分数。"""
    scores = w.data * np.array(ids, dtype=float) + b.data
    predictions = (scores > 0).astype(int)
    accuracy = float((predictions == np.array(labels)).mean())
    return accuracy, scores


def show_scheme(name, order, trained=None):
    """在一种编号方案下训练 + 打印结果。

    trained 不为 None 时，直接用它（用来演示"编号换了，旧模型会怎样"）。
    """
    ids = {word: i for i, word in enumerate(order)}
    words = TECH_WORDS + FOOD_WORDS
    labels = [0] * len(TECH_WORDS) + [1] * len(FOOD_WORDS)
    word_ids = [ids[w] for w in words]

    print("=" * 62)
    print(f"编号方案 {name}")
    print("=" * 62)
    print("词表：" + "  ".join(f"{w}={ids[w]}" for w in order))
    print()

    if trained is None:
        w, b, loss = train(word_ids, labels)
        print(f"训练 {STEPS} 步，最终 loss = {loss:.4f}")
    else:
        w, b = trained
        print("不重新训练，直接用方案 A 训练好的 w 和 b")
    print(f"w = {w.data:+.3f}   b = {b.data:+.3f}")
    print()

    accuracy, scores = evaluate(w, b, word_ids, labels)
    print("  编号   词     分数     应该属于   分数算出来的类别")
    print("  " + "-" * 52)
    for word, word_id, label, score in zip(words, word_ids, labels, scores):
        truth = "科技" if label == 0 else "食品"
        guess = "科技" if score <= 0 else "食品"
        mark = "" if truth == guess else "   <-- 错"
        print(f"  {word_id:>4}   {pad(word, 6)} {score:+.2f}    {truth}         {guess}{mark}")
    print()
    print(f"准确率 = {accuracy * 100:.1f}%")

    if trained is None:
        direction = "往上走" if w.data > 0 else "往下走"
        print(f"分数只能沿着编号 {direction} —— 因为 w = {w.data:+.3f} 这一个数就定死了方向。")
        print(f"也就是说：{pad(words[int(np.argmin(scores))], 6)} 和 {pad(words[int(np.argmax(scores))], 6)}"
              " 被放在了分数轴的两个极端。")
        print("这两个词凭什么站在两端？凭它们的编号。")
    print()
    return accuracy, (w, b)


def main():
    schemes = build_schemes()

    results = []
    trained_on_a = None
    for name, order in schemes:
        accuracy, params = show_scheme(name, order)
        results.append((name, accuracy))
        if trained_on_a is None:
            trained_on_a = params

    print("=" * 62)
    print("同一个模型、同一批词、同一个任务，只换了编号")
    print("=" * 62)
    for name, accuracy in results:
        print(f"  {pad(name, 44)} 准确率 {accuracy * 100:5.1f}%")
    print()

    show_scheme(
        "B  按拼音排序（把方案 A 训练好的模型直接搬过来）",
        PINYIN_ORDER,
        trained=trained_on_a,
    )


if __name__ == "__main__":
    main()
