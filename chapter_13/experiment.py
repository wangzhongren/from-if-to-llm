"""第 13 章 experiment.py —— 证明"平均"确实分不出轻重。

做三个对比：

  对比一：权重表。平均法给所有词写死同一个权重。
  对比二：扰动实验。给第 i 个词的向量加同样大小的扰动，
          看"苹果"的上下文向量变化多少。变化量 = 这个位置的影响力。
  对比三：区分度。上下文向量和各个词有多像，最大最小差多少。

跑法：

    ./.venv/bin/python chapter_13/experiment.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from after import (  # noqa: E402
    CENTER,
    SENTENCE,
    attention_output,
    attention_weights,
    bar,
    display_width,
    pad,
    similarity_scores,
    vectors_of,
)
from toygrad import Tensor  # noqa: E402

# 扰动实验用的固定扰动：每个位置加的都是同一个向量
PERTURB = np.array([0.1] * 8)


def average_context(vectors, center):
    """第 12 章的做法：其它所有词的平均（自己不算）。"""
    n = vectors.shape[0]
    weights = np.full(n, 1.0 / (n - 1))
    weights[center] = 0.0
    return weights @ vectors.data


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    words = SENTENCE
    vectors = vectors_of(words)
    n = len(words)

    scores = similarity_scores(vectors)
    weights = attention_weights(scores).data
    plain_context = average_context(vectors, CENTER)
    attention_context = attention_output(vectors, attention_weights(scores)).data[CENTER]

    # ---------------- 对比一：权重 ----------------
    print("=" * 60)
    print("对比一：'苹果'给每个词的权重")
    print("=" * 60)
    print(f"\n  {pad('词', 6)}{'平均法':>9}{'相似度加权':>12}   谁更被重视")
    print("  " + "-" * 56)
    for j, word in enumerate(words):
        w_old = 0.0 if j == CENTER else 1.0 / (n - 1)
        w_new = weights[CENTER][j]
        print(f"  {pad(word, 6)}{w_old:>9.4f}{w_new:>11.4f}   "
              f"{bar(w_new, 0.25, 20)}")
    print(f"\n  平均法：其它 {n - 1} 个词全是 {1.0 / (n - 1):.4f}，"
          f"最大值 / 最小值 = {1.0:.2f}（自己不算）")
    print(f"  相似度加权：最大 {weights[CENTER].max():.4f}，"
          f"最小 {weights[CENTER].min():.4f}，最大值 / 最小值 = "
          f"{weights[CENTER].max() / weights[CENTER].min():.2f}")

    # ---------------- 对比二：扰动实验 ----------------
    print("\n" + "=" * 60)
    print("对比二：扰动实验（给第 i 个词加同样大的扰动，看'苹果'变多少）")
    print("=" * 60)
    print(f"\n  {pad('改的是哪个词', 12)}{'平均法':>12}{'相似度加权':>12}   倍数")
    print("  " + "-" * 56)
    old_changes, new_changes = [], []
    for i, word in enumerate(words):
        if i == CENTER:
            continue                      # 自己不在自己的上下文里，跳过
        perturbed = vectors.data.copy()
        perturbed[i] = perturbed[i] + PERTURB
        perturbed = Tensor(perturbed)

        old_after = average_context(perturbed, CENTER)
        # 注意：这里必须把权重重新算一遍——词变了，权重也会跟着变。
        new_after = attention_output(
            perturbed, attention_weights(similarity_scores(perturbed))
        ).data[CENTER]

        old_change = float(np.linalg.norm(old_after - plain_context))
        new_change = float(np.linalg.norm(new_after - attention_context))
        old_changes.append(old_change)
        new_changes.append(new_change)
        print(f"  {pad(word, 12)}{old_change:>12.6f}{new_change:>12.6f}   "
              f"{new_change / old_change:>5.2f}x")

    print(f"\n  平均法：最大 / 最小 = {max(old_changes) / min(old_changes):.3f}"
          "（完全一样）")
    print(f"  相似度加权：最大 / 最小 = "
          f"{max(new_changes) / min(new_changes):.2f}")
    print("  相似度加权后的变化量基本跟着权重走：'手机''发布''苹果'一改就动，"
          "'的''在''昨天'改了几乎没反应。")

    # ---------------- 对比三：区分度 ----------------
    print("\n" + "=" * 60)
    print("对比三：'苹果'的上下文向量，和句子里的词有多像")
    print("=" * 60)
    old_sims = sorted(((cosine(plain_context, v), w) for w, v in zip(words, vectors.data)),
                      reverse=True)
    new_sims = sorted(((cosine(attention_context, v), w) for w, v in zip(words, vectors.data)),
                      reverse=True)
    print(f"\n  {pad('平均法', 12)}{'像不像':>9}      {pad('相似度加权', 12)}{'像不像':>9}")
    print("  " + "-" * 50)
    for k in range(4):
        print(f"  {pad(old_sims[k][1], 12)}{old_sims[k][0]:>9.4f}      "
              f"{pad(new_sims[k][1], 12)}{new_sims[k][0]:>9.4f}")
    print("  ...")
    for k in range(-2, 0):
        print(f"  {pad(old_sims[k][1], 12)}{old_sims[k][0]:>9.4f}      "
              f"{pad(new_sims[k][1], 12)}{new_sims[k][0]:>9.4f}")
    print(f"\n  平均法：最高的 {old_sims[0][0]:.4f}，最低的 {old_sims[-1][0]:.4f}，"
          f"差 {old_sims[0][0] - old_sims[-1][0]:.4f}")
    print(f"  相似度加权：最高的 {new_sims[0][0]:.4f}，最低的 {new_sims[-1][0]:.4f}，"
          f"差 {new_sims[0][0] - new_sims[-1][0]:.4f}")
    print("  平均出来的向量和谁都差不多像，也就等于和谁都不特别像。")


if __name__ == "__main__":
    main()
