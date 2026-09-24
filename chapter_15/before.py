"""第 15 章 before.py —— 第 14 章的单头注意力，卡在哪里。

这个文件做两件事：

  一、把第 14 章的 self-attention 原封不动地跑在下面这句话上，
      看"他"这个字拿到的权重——只有一行。
  二、把"他"身上挂着的两个需求写成两组理想权重，算一算
      一行权重能不能同时满足它们。

跑法：

    ./.venv/bin/python chapter_15/before.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from toygrad import Tensor, randn

SENTENCE = "小王 把 书 给了 小李 因为 他 明天 考试".split()

EMBEDDINGS = {
    "小王": [0.9, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
    "把":   [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 1.0, 0.0],
    "书":   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "给了": [0.0, 0.0, 0.0, 1.0, 0.3, 0.0, 0.2, 0.0],
    "小李": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
    "因为": [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.9, 0.3],
    "他":   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
    "明天": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
    "考试": [0.0, 0.5, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0],
}

HE_AT = SENTENCE.index("他")


def display_width(text):
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def pad(text, width, align="left"):
    space = " " * max(0, width - display_width(text))
    return text + space if align == "left" else space + text


def single_head_weights(words):
    """第 14 章那一套：一次投影，一组 Q/K/V，一组权重。"""
    matrix = np.array([EMBEDDINGS[word] for word in words])
    vectors = Tensor(matrix)
    w_q = randn(8, 8, scale=0.5, seed=51)
    w_k = randn(8, 8, scale=0.5, seed=52)
    query = vectors @ w_q
    key = vectors @ w_k
    scores = (query @ key.T) / np.sqrt(8)
    return scores.softmax(axis=-1).data


def total_variation(p, q):
    """两个分布差多远：把所有位置上的差加起来再除以 2。

    等于 0 就是完全一样；等于 1 就是完全不一样。
    它说的是"有百分之多少的权重放错了地方"。
    """
    return 0.5 * float(np.abs(np.array(p) - np.array(q)).sum())


def main():
    words = SENTENCE

    print("=" * 60)
    print("句子：", " ".join(words))
    print("=" * 60)

    # ---------------- 一、单头只能给一行 ----------------
    weights = single_head_weights(words)
    print("\n第 14 章的做法：'他'这个字拿到的权重（只有这一行）")
    print("-" * 60)
    row = weights[HE_AT]
    for j, word in enumerate(words):
        mark = "  <- 自己" if j == HE_AT else ""
        print(f"  {pad(word, 6)}{row[j]:.4f}  {'#' * int(round(row[j] * 40))}{mark}")
    print(f"  权重和 = {row.sum():.4f}")
    print("  （投影是随机初始化的，所以具体数字没意义；有意义的是："
          "它只有一行。）")

    # ---------------- 二、一行装不下两个需求 ----------------
    print("\n" + "=" * 60)
    print("二、'他'身上同时挂着两个需求")
    print("=" * 60)

    need_reference = {"小王": 0.5, "小李": 0.5}
    need_time = {"明天": 0.5, "考试": 0.5}

    print("\n  需求 A（指代）：'他'是小王还是小李？")
    print("    理想的权重应该是：" + "，".join(f"{w} {v}" for w, v in need_reference.items()))
    print("\n  需求 B（时间）：'他'明天要考试")
    print("    理想的权重应该是：" + "，".join(f"{w} {v}" for w, v in need_time.items()))

    budget_a = sum(need_reference.values())
    budget_b = sum(need_time.values())
    print(f"\n  需求 A 要花掉的权重：{budget_a}")
    print(f"  需求 B 要花掉的权重：{budget_b}")
    print(f"  两个加起来：{budget_a + budget_b}")
    print("  一行权重一共只有：1.0")
    print(f"  → 超出预算 {budget_a + budget_b - 1.0:.1f}")

    def to_vector(need):
        return [need.get(word, 0.0) for word in words]

    compromise = [(need_reference.get(w, 0.0) + need_time.get(w, 0.0)) / 2 for w in words]
    print("\n  一行权重能做到的最好情况，是两边各让一步：")
    print("  " + "  ".join(f"{pad(w, 6)}{compromise[j]:.2f}"
                           for j, w in enumerate(words) if compromise[j] > 0))
    gap_a = total_variation(compromise, to_vector(need_reference))
    gap_b = total_variation(compromise, to_vector(need_time))
    print(f"\n  离需求 A 还差：{gap_a:.2f}（有 {gap_a * 100:.0f}% 的权重放错了地方）")
    print(f"  离需求 B 还差：{gap_b:.2f}")
    print("  两个需求都只做到一半。")

    print("\n  如果有两个头，就是两行权重，两笔预算：")
    print(f"    头 1 只管需求 A：离需求 A 差 {0.0:.2f}")
    print(f"    头 2 只管需求 B：离需求 B 差 {0.0:.2f}")
    print("  （头 1 根本不需要管需求 B，那是头 2 的事。）")

    print("\n  结论：一个头只有一行权重，加起来等于 1。")
    print("  一句话里的关系不止一种，一行权重不够分。")


if __name__ == "__main__":
    main()
