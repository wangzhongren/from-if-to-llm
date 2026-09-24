"""第 16 章 before.py —— 第 15 章的多头注意力，卡在哪里。

把"狗咬人"和"人咬狗"分别送进上一章的网络，看会发生什么。

这两句话用的是同一批词，只是顺序不同。而上一章的机制里，
没有任何一步用到"这个词排第几"——所以它看不出区别。

跑法：

    ./.venv/bin/python chapter_16/before.py
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


def attention(words, seed=16):
    """第 15 章那一套：投影、打分、归一化、加权求和。

    注意这里没有"位置"这个东西——输入只有每个词自己的向量。
    """
    vector = Tensor(np.array([EMBEDDINGS[word] for word in words]))
    w_q = randn(DIM, DIM, scale=0.5, seed=seed)
    w_k = randn(DIM, DIM, scale=0.5, seed=seed + 1)
    w_v = randn(DIM, DIM, scale=0.5, seed=seed + 2)

    query = vector @ w_q
    key = vector @ w_k
    value = vector @ w_v
    weights = ((query @ key.T) / np.sqrt(DIM)).softmax(axis=-1)
    return weights @ value, weights


def show(sentence, output, weights):
    print(f"\n  『{' '.join(sentence)}』")
    print("  " + "-" * 50)
    for i, word in enumerate(sentence):
        row = weights.data[i]
        top = sorted(range(len(sentence)), key=lambda j: -row[j])
        best = "、".join(f"{sentence[j]}({row[j]:.3f})" for j in top)
        print(f"    {pad(word, 4)}看谁：{best}")
        print(f"          输出 = [{', '.join(f'{v:+.3f}' for v in output.data[i])}]")


def main():
    print("=" * 60)
    print("两句话：『狗 咬 人』 和 『人 咬 狗』")
    print("=" * 60)

    output_a, weights_a = attention(SENTENCE_A)
    output_b, weights_b = attention(SENTENCE_B)

    show(SENTENCE_A, output_a, weights_a)
    show(SENTENCE_B, output_b, weights_b)

    # ---------------- 逐行对比 ----------------
    print("\n" + "=" * 60)
    print("把两句话的输出逐行对起来看")
    print("=" * 60)
    print(f"\n  {'狗咬人':>10}{'':>4}{'人咬狗':>10}")
    print("  " + "-" * 40)
    print(f"  {'第 0 个词':>10}{'':>4}{'第 0 个词':>10}")
    print(f"  {SENTENCE_A[0]:>10}{'':>4}{SENTENCE_B[0]:>10}")
    print(f"  {'狗':>8}的输出{'':>2}{'人':>8}的输出")
    print("  " + "-" * 40)

    # 人咬狗 的第一个词是"人"，它在 狗咬人 里排第 2
    index_b0_in_a = SENTENCE_A.index(SENTENCE_B[0])
    index_b2_in_a = SENTENCE_A.index(SENTENCE_B[2])
    print(f"  人咬狗 的第 0 个词（人）的输出 = "
          f"[{', '.join(f'{v:+.3f}' for v in output_b.data[0])}]")
    print(f"  狗咬人 的第 {index_b0_in_a} 个词（人）的输出 = "
          f"[{', '.join(f'{v:+.3f}' for v in output_a.data[index_b0_in_a])}]")
    print(f"\n  人咬狗 的第 2 个词（狗）的输出 = "
          f"[{', '.join(f'{v:+.3f}' for v in output_b.data[2])}]")
    print(f"  狗咬人 的第 {index_b2_in_a} 个词（狗）的输出 = "
          f"[{', '.join(f'{v:+.3f}' for v in output_a.data[index_b2_in_a])}]")

    same_ren = np.allclose(output_b.data[0], output_a.data[index_b0_in_a])
    same_gou = np.allclose(output_b.data[2], output_a.data[index_b2_in_a])
    print(f"\n  '人'的输出一模一样吗：{same_ren}")
    print(f"  '狗'的输出一模一样吗：{same_gou}")

    # ---------------- 整句话的表示 ----------------
    print("\n" + "=" * 60)
    print("整句话的表示：把所有位置的输出加起来（顺序就不重要了）")
    print("=" * 60)
    sum_a = output_a.data.sum(axis=0)
    sum_b = output_b.data.sum(axis=0)
    print(f"\n  狗咬人 的句子向量 = [{', '.join(f'{v:+.3f}' for v in sum_a)}]")
    print(f"  人咬狗 的句子向量 = [{', '.join(f'{v:+.3f}' for v in sum_b)}]")
    print(f"\n  两句话的句子向量一样吗：{np.allclose(sum_a, sum_b)}")

    print("\n  结论：换个词序，输出只是跟着换了位置，值一个都没变。")
    print("  把整句话合成一个向量之后，「狗咬人」和「人咬狗」完全一样——")
    print("  这套机制根本不知道谁在前谁在后。")


if __name__ == "__main__":
    main()
