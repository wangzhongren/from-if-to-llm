"""第 16 章 after.py —— 让模型知道谁在前谁在后。

上一章的注意力有个毛病：换个词序，输出只是跟着换位置，一个数都不变。

这一章给每个位置配一个固定的向量，加在词向量上：

    x[i] = 词向量 + 位置向量[i]

这个位置向量，我们叫它**位置编码**（positional encoding）。
这个文件里有两个方案：先看最笨的那个，再看正弦余弦的那个。

跑法：

    ./.venv/bin/python chapter_16/after.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from toygrad import Tensor, randn

SENTENCE_A = "狗 咬 人".split()
SENTENCE_B = "人 咬 狗".split()

EMBEDDINGS = {
    "狗": [1.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
    "咬": [0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2],
    "人": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
}

DIM = 8


def display_width(text):
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def pad(text, width, align="left"):
    space = " " * max(0, width - display_width(text))
    return text + space if align == "left" else space + text


# ---------------------------------------------------------------- 位置信息


def index_positions(length, dim):
    """最笨的方案：第 i 个词，就给它加上一个「i」。

    每个维度都填上这个位置编号，所以位置 10 的向量是 [10, 10, ..., 10]。
    """
    return np.array([[float(i)] * dim for i in range(length)])


def sinusoidal_positions(length, dim):
    """位置编码：每个位置一个向量，每一维是一个不同频率的波。

        第 i 个位置、第 2k 维   = sin(i / 10000^(2k/dim))
        第 i 个位置、第 2k+1 维 = cos(i / 10000^(2k/dim))

    低频的维度变化慢（管"隔得远不远"），高频的维度变化快（管"隔了几个"）。
    这样一组波，能让每个位置都有一个固定的、不会撞车的编号。
    """
    positions = np.arange(length)[:, None]
    dims = np.arange(dim)[None, :]
    angles = positions / np.power(10000.0, (2 * (dims // 2)) / dim)
    table = np.zeros((length, dim))
    table[:, 0::2] = np.sin(angles[:, 0::2])
    table[:, 1::2] = np.cos(angles[:, 1::2])
    return table


# ---------------------------------------------------------------- 注意力


def attention(words, position_table, seed=16):
    """加了位置信息的注意力：输入 = 词向量 + 位置向量。"""
    word_vectors = np.array([EMBEDDINGS[word] for word in words])
    vector = Tensor(word_vectors + position_table)

    w_q = randn(DIM, DIM, scale=0.5, seed=seed)
    w_k = randn(DIM, DIM, scale=0.5, seed=seed + 1)
    w_v = randn(DIM, DIM, scale=0.5, seed=seed + 2)

    query = vector @ w_q
    key = vector @ w_k
    value = vector @ w_v
    weights = ((query @ key.T) / np.sqrt(DIM)).softmax(axis=-1)
    return weights @ value, weights


def show_positions(table, title, limit=11):
    print(f"\n{title}")
    print("-" * 60)
    print("  位置  " + "".join(f"{'第' + str(d) + '维':>10}" for d in range(table.shape[1])))
    for i in range(min(limit, table.shape[0])):
        print(f"  {i:>4}  " + "".join(f"{v:>10.3f}" for v in table[i]))


def main():
    print("=" * 60)
    print("两句话：『狗 咬 人』 和 『人 咬 狗』")
    print("=" * 60)

    # ---------------- 先看最笨的方案 ----------------
    print("\n" + "=" * 60)
    print("方案一：直接用「第几个词」这个编号当位置向量")
    print("=" * 60)
    index_table = index_positions(11, DIM)
    show_positions(index_table, "前 11 个位置的编号向量（8 维都一样，所以只可能这样）", limit=3)
    print("\n  问题在这里——看看它的长度：")
    word_norm = float(np.linalg.norm(EMBEDDINGS["狗"]))
    print(f"    {'位置':>6}{'编号向量的长度':>16}{'词向量的长度':>14}")
    print("  " + "-" * 40)
    for i in [0, 1, 2, 5, 10, 50]:
        print(f"    {i:>6}{np.linalg.norm(index_table[min(i, 10)]):>16.2f}"
              f"{word_norm:>14.2f}")
    print(f"\n    （位置 50 的编号向量长度 = 50 × √8 ≈ {50 * np.sqrt(8):.1f}）")
    print("  位置编号会长到几百，而词向量只有 1 点几——")
    print("  两者一相加，位置那一项会把词义完全盖住。")
    print("  而且训练时最长只见过 11 个词，遇到第 50 个词，"
          "那个数字它从来没见过。")

    # ---------------- 再看位置编码 ----------------
    print("\n" + "=" * 60)
    print("方案二：位置编码（一组不同频率的正弦余弦）")
    print("=" * 60)
    table = sinusoidal_positions(11, DIM)
    show_positions(table, "前 11 个位置的位置编码", limit=11)
    lengths = np.linalg.norm(table, axis=1)
    print(f"\n  每个位置的长度：{lengths.min():.3f} ~ {lengths.max():.3f}"
          f"（词向量的长度是 {word_norm:.3f}）")
    print("  不管位置是第几个，长度都不变，值永远在 -1 和 1 之间。")

    print("\n  两个位置的相似度（点积），只跟「隔多远」有关：")
    print(f"  {'位置对':>12}{'隔几个词':>10}{'相似度':>10}")
    print("  " + "-" * 36)
    for i, j in [(0, 0), (0, 1), (1, 2), (3, 4), (0, 2), (2, 4), (0, 5)]:
        print(f"  {f'{i} 和 {j}':>12}{abs(i - j):>10}{float(table[i] @ table[j]):>10.3f}")

    # ---------------- 现在再看那两句话 ----------------
    print("\n" + "=" * 60)
    print("把位置编码加上去，再跑一次那两句话")
    print("=" * 60)

    output_a, _ = attention(SENTENCE_A, table[: len(SENTENCE_A)])
    output_b, _ = attention(SENTENCE_B, table[: len(SENTENCE_B)])

    index_b0_in_a = SENTENCE_A.index(SENTENCE_B[0])
    index_b2_in_a = SENTENCE_A.index(SENTENCE_B[2])
    print("\n  人咬狗 的第 0 个词（人）的输出 = "
          f"[{', '.join(f'{v:+.3f}' for v in output_b.data[0])}]")
    print(f"  狗咬人 的第 {index_b0_in_a} 个词（人）的输出 = "
          f"[{', '.join(f'{v:+.3f}' for v in output_a.data[index_b0_in_a])}]")
    print("  狗咬人 的第 0 个词（狗）的输出 = "
          f"[{', '.join(f'{v:+.3f}' for v in output_a.data[0])}]")

    same = np.allclose(output_b.data[0], output_a.data[index_b0_in_a])
    print(f"\n  '人'在两个句子里的输出还一样吗：{same}")

    sum_a = output_a.data.sum(axis=0)
    sum_b = output_b.data.sum(axis=0)
    print(f"\n  狗咬人 的句子向量 = [{', '.join(f'{v:+.3f}' for v in sum_a)}]")
    print(f"  人咬狗 的句子向量 = [{', '.join(f'{v:+.3f}' for v in sum_b)}]")
    print(f"  两句话的句子向量一样吗：{np.allclose(sum_a, sum_b)}")
    print(f"  它们之间的距离：{float(np.linalg.norm(sum_a - sum_b)):.4f}")
    print("\n  同样的三个词，换个顺序，这次终于不一样了。")


if __name__ == "__main__":
    main()
