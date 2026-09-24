"""第 14 章 after.py —— 我找什么、你有什么。

第 13 章只有一把尺子：像不像。分数是 v_i · v_j，一个数，两边用。

这一章我们把那个点积拆成三个角色：

    Query（我在找什么） = x @ W_q
    Key  （你有什么）   = x @ W_k
    Value（我能给你什么）= x @ W_v

    权重 = softmax(Q Kᵀ / √d)
    输出 = 权重 @ V

跑法：

    ./.venv/bin/python chapter_14/after.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from toygrad import Tensor, randn

# ---------------------------------------------------------------- 语料

SENTENCE = "我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机".split()

# 8 个维度依次代表：
#   人/动物   时间   地点   动作   物品   科技与商业   虚词   程度
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

DIM = 8          # 词向量维度
D_KEY = 8        # 查询和键的维度
CENTER = SENTENCE.index("苹果")

# 最后一节要手工造一个"问法"。投影出来的向量有多长，决定这个问题问得有多"用力"：
# 向量越长，分数拉得越开，权重越尖。这里固定乘 4，只是为了把差别看清楚。
QUESTION_STRENGTH = 4.0

FEATURE_NAMES = ["人/动物", "时间", "地点", "动作", "物品", "科技与商业", "虚词", "程度"]


def vectors_of(words):
    return Tensor(np.array([EMBEDDINGS[word] for word in words]))


# ---------------------------------------------------------------- 三个角色


def project(vectors, weight):
    """把每个词向量投影到另一个空间：x @ W。"""
    return vectors @ weight


def split_into_qkv(vectors, w_q, w_k, w_v):
    """同样一批词向量，投三次影，得到三个角色。"""
    query = project(vectors, w_q)
    key = project(vectors, w_k)
    value = project(vectors, w_v)
    return query, key, value


def attention_scores(query, key):
    """分数 = Q Kᵀ / √d_key。

    除以 √d 是为了让分数的尺度不随维度涨——`experiment.py` 里有实验。
    """
    return (query @ key.T) / np.sqrt(key.shape[-1])


def attention_weights(scores):
    return scores.softmax(axis=-1)


def attention(query, key, value):
    """self-attention 的完整流程。"""
    weights = attention_weights(attention_scores(query, key))
    return weights @ value, weights


# ---------------------------------------------------------------- 打印


def display_width(text):
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def pad(text, width, align="left"):
    space = " " * max(0, width - display_width(text))
    return text + space if align == "left" else space + text


def bar(value, full_scale, width=28):
    return "#" * int(round(value / full_scale * width))


def show_row(words, weights, center, title):
    print(f"\n{title}")
    print("-" * 60)
    row = weights[center]
    biggest = float(max(row))
    for j, word in enumerate(words):
        mark = "  <- 自己" if j == center else ""
        print(f"  {pad(word, 6)}{row[j]:.4f}  {bar(row[j], biggest)}{mark}")
    print(f"  权重和 = {float(sum(row)):.4f}")


def show_matrix(words, matrix, title):
    print(f"\n{title}")
    print("-" * 60)
    print("  " + pad("", 6) + "".join(pad(w, 7, align="right") for w in words))
    for i, word in enumerate(words):
        cells = "".join(f"{matrix[i][j]:>7.3f}" for j in range(len(words)))
        print("  " + pad(word, 6) + cells)
    print("  （行 = 谁在问，列 = 问到了谁。每一行的和都是 1。）")


def main():
    words = SENTENCE
    vectors = vectors_of(words)

    # 投影矩阵：随机初始化、固定种子。
    # 真实模型里它们是训练出来的；这一章先把机制跑通，下一章我们真的训练它。
    w_q = randn(DIM, D_KEY, seed=14)
    w_k = randn(DIM, D_KEY, seed=24)
    w_v = randn(DIM, D_KEY, seed=34)

    query, key, value = split_into_qkv(vectors, w_q, w_k, w_v)
    scores = attention_scores(query, key)
    weights = attention_weights(scores)
    output, _ = attention(query, key, value)

    print("=" * 60)
    print("句子：", " ".join(words))
    print("=" * 60)
    print(f"\n每个词从 8 维投影成 8 维的 Query / Key / Value：")
    print(f"  投影前 x       : {vectors.shape}")
    print(f"  投影后 Q, K, V : {query.shape}")

    show_matrix(words, weights.data, "注意力权重矩阵（投影还没训练，先看机制）")

    show_row(words, weights.data, CENTER, "只看'苹果'在问谁")

    # ---------------- 和第 13 章比：分数不再对称 ----------------
    print("\n" + "=" * 60)
    print("第 13 章 vs 这一章：分数还是一对一的一个数吗？")
    print("=" * 60)
    plain_scores = (vectors @ vectors.T).data
    i, j = CENTER, words.index("手机")
    print(f"\n  {pad('', 14)}{'苹果 -> 手机':>14}{'手机 -> 苹果':>14}   对称吗")
    print("  " + "-" * 56)
    print(f"  {pad('第 13 章', 14)}{plain_scores[i][j]:>14.3f}{plain_scores[j][i]:>14.3f}"
          f"   {'是' if abs(plain_scores[i][j] - plain_scores[j][i]) < 1e-9 else '否'}")
    print(f"  {pad('这一章', 14)}{scores.data[i][j]:>14.3f}{scores.data[j][i]:>14.3f}"
          f"   {'是' if abs(scores.data[i][j] - scores.data[j][i]) < 1e-9 else '否'}")
    print("\n  第 13 章两个方向共用同一个数，所以'我问你'和'你问我'是一回事。")
    print("  这一章 Q 和 K 是两套投影，'我问你'和'你问我'彻底分开了。")

    # ---------------- 同一个词，问不同的问题 ----------------
    print("\n" + "=" * 60)
    print("同一个'苹果'，换一个问题，就换一套权重")
    print("=" * 60)
    print("（为了看清楚，这里把 W_q 手工设成'只问某一维'，W_k 设成原样不动。）")

    for dim in [5, 4]:
        w_q_hand = np.zeros((DIM, DIM))
        w_q_hand[dim][dim] = QUESTION_STRENGTH
        q_hand = project(vectors, Tensor(w_q_hand))
        rows = attention_weights(attention_scores(q_hand, key)).data[CENTER]
        top = sorted(range(len(words)), key=lambda j: -rows[j])[:3]
        best = "、".join(f"{words[j]}({rows[j]:.3f})" for j in top)
        print(f"\n  问'{pad(FEATURE_NAMES[dim], 10)}'：苹果最关注 {best}")
        for j, word in enumerate(words):
            print(f"      {pad(word, 6)}{rows[j]:.4f}  {bar(rows[j], 0.45, 24)}")

    print("\n  第 13 章做不到这件事：那里只有一套权重，换不了问题。")


if __name__ == "__main__":
    main()
