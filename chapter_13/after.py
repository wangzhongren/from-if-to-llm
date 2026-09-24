"""第 13 章 after.py —— 不是每个词都一样重要。

第 12 章定义上下文的方式是：

    context = 其它所有词的平均（自己不算）

这个文件把"平均"换成"按相似度加权求和"：

    1. 算分数：中心词和每个词有多像
    2. 归一化成权重：像的词权重大，不像的词权重小
    3. 加权求和：把所有词按权重混成一个新向量

这三步合起来，就是 Attention（注意力）。

跑法：

    ./.venv/bin/python chapter_13/after.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from toygrad import Tensor

# ---------------------------------------------------------------- 语料

SENTENCE = "我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机".split()

# 每个词一个 8 维向量。这 8 个维度依次代表：
#   人/动物   时间   地点   动作   物品   科技与商业   虚词   程度
#
# 第 10–12 章的向量是从语料里学出来的。这里为了实验能反复重跑、
# 也为了让每个数字都有来历，我们直接手写一份语义上说得通的表。
EMBEDDINGS = {
    "我":   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0],
    "昨天": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    "在":   [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.8, 0.0],
    "商场": [0.0, 0.0, 1.0, 0.0, 0.3, 0.2, 0.0, 0.0],
    "看到": [0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0],
    "苹果": [0.0, 0.0, 0.0, 0.0, 0.7, 1.0, 0.0, 0.0],
    "刚刚": [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.2, 0.7],
    "发布": [0.0, 0.0, 0.0, 0.8, 0.3, 1.0, 0.0, 0.0],
    "的":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "新":   [0.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.0, 1.0],
    "手机": [0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.0],
}

# 这个句子里，我们关心的那个词的编号（"苹果"）
CENTER = SENTENCE.index("苹果")


def vectors_of(words):
    """把一串词变成一堆向量，形状 (词数, 8)。"""
    return Tensor(np.array([EMBEDDINGS[word] for word in words]))


# ---------------------------------------------------------------- 三步走


def similarity_scores(vectors):
    """第一步：两两算相似度。

    第 i 行第 j 列 = 第 i 个词和第 j 个词的相似度。
    这里用的是点积：两个向量逐维相乘再相加。方向越一致，分数越大。
    """
    return vectors @ vectors.T


def attention_weights(scores):
    """第二步：把分数变成权重。

    要求每一行的权重都大于 0、加起来等于 1。
    做法和第 5 章的 softmax 完全一样：先取指数，再除以总和。
    """
    return scores.softmax(axis=-1)


def attention_output(vectors, weights):
    """第三步：加权求和。

    第 i 个词的新向量 = 所有词的向量，按第 i 行的权重混合起来。
    """
    return weights @ vectors


def average_context(vectors, center):
    """第 12 章的老办法：一个词的上下文 = **其它所有词**的平均。

    权重是写死的：除了自己，其它每个词都是 1/(词数-1)。
    留在这里，专门用来和 Attention 做对比。
    """
    n = vectors.shape[0]
    weights = np.full(n, 1.0 / (n - 1))
    weights[center] = 0.0
    return Tensor(weights) @ vectors


# ---------------------------------------------------------------- 打印


def display_width(text):
    """中文字符在终端里占两列，算宽度时要按两列算。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def pad(text, width, align="left"):
    """按终端显示宽度补空格，让表对齐。"""
    space = " " * max(0, width - display_width(text))
    return text + space if align == "left" else space + text


def bar(value, full_scale, width=28):
    """把权重画成一条 # 组成的横条。full_scale 是画满时的权重。"""
    return "#" * int(round(value / full_scale * width))


def show_weights(words, weights, center, title):
    print(f"\n{title}")
    print("-" * 60)
    row = weights[center]
    biggest = float(max(row))
    for j, word in enumerate(words):
        mark = "  <- 中心词" if j == center else ""
        print(f"  {pad(word, 6)}{row[j]:.4f}  {bar(row[j], biggest)}{mark}")
    print(f"  权重和 = {float(sum(row)):.4f}")


def show_matrix(words, matrix, title):
    print(f"\n{title}")
    print("-" * 60)
    print("  " + pad("", 6) + "".join(pad(w, 7, align="right") for w in words))
    for i, word in enumerate(words):
        cells = "".join(f"{matrix[i][j]:>7.3f}" for j in range(len(words)))
        print("  " + pad(word, 6) + cells)
    print("  （行 = 谁在看，列 = 看谁。每一行的和都是 1。）")


def main():
    words = SENTENCE
    vectors = vectors_of(words)

    print("=" * 60)
    print("句子：", " ".join(words))
    print("=" * 60)

    scores = similarity_scores(vectors)
    weights = attention_weights(scores)
    output = attention_output(vectors, weights)

    print("\n第一步：相似度分数（点积）。只看'苹果'这一行：")
    for j, word in enumerate(words):
        print(f"  {pad(word, 6)}{float(scores.data[CENTER][j]):+.3f}")

    show_weights(words, weights.data, CENTER, "第二步：归一化成权重（'苹果'这一行）")

    show_matrix(words, weights.data, "完整的权重矩阵")

    print("\n第三步：加权求和。'苹果'的新向量 =")
    for dim, name in enumerate(["人/动物", "时间", "地点", "动作", "物品", "科技商业", "虚词", "程度"]):
        print(f"  {pad(name, 10)}{float(output.data[CENTER][dim]):+.4f}")
    print("  （旧向量是 ", np.round(vectors.data[CENTER], 3), "）")

    # ---------------- 和第 12 章的平均法对比 ----------------
    print("\n" + "=" * 60)
    print("和第 12 章的'平均'比一比：看'苹果'给每个词的权重")
    print("=" * 60)
    print(f"\n  {pad('词', 6)}{'平均法':>10}{'Attention':>12}{'倍数':>10}")
    for j, word in enumerate(words):
        old_w = 0.0 if j == CENTER else 1.0 / (len(words) - 1)
        new_w = float(weights.data[CENTER][j])
        ratio = "—" if old_w == 0.0 else f"{new_w / old_w:.1f}x"
        mark = "   <- 自己（平均法不看自己）" if j == CENTER else ""
        print(f"  {pad(word, 6)}{old_w:>10.4f}{new_w:>12.4f}{ratio:>10}{mark}")

    print("\n  平均法给'发布'和'昨天'的权重：都是 0.1000 —— 一模一样。")
    print(f"  Attention 给它们的权重：{float(weights.data[CENTER][7]):.4f} 和 "
          f"{float(weights.data[CENTER][1]):.4f} —— 差了 "
          f"{float(weights.data[CENTER][7]) / float(weights.data[CENTER][1]):.1f} 倍。")

    # ---------------- 换个中心词，权重就换一套 ----------------
    print("\n" + "=" * 60)
    print("换个中心词，Attention 给出的权重也跟着换")
    print("=" * 60)
    for name in ["苹果", "手机", "商场"]:
        i = words.index(name)
        row = weights.data[i]
        top = sorted(range(len(words)), key=lambda j: -row[j])[:3]
        best = "、".join(f"{words[j]}({row[j]:.3f})" for j in top)
        print(f"  {pad(name, 6)}最关注的三个词：{best}")
    print("  （平均法这里会打印出三行完全相同的 0.1000。）")


if __name__ == "__main__":
    main()
