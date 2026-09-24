"""第 13 章 before.py —— 第 12 章的老办法，用在这章的句子上。

第 12 章说：一个词的上下文，就是**其它所有词**的平均。

    context = (v_我 + v_昨天 + ... + v_手机) / 11

这个文件把这套办法原封不动地跑一遍，然后做一个"扰动实验"，
看看它对每个词的重视程度到底是多少。

跑法：

    ./.venv/bin/python chapter_13/before.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from toygrad import Tensor

SENTENCE = "我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机".split()

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

CENTER = SENTENCE.index("苹果")

# 扰动实验用的小扰动：每个位置都加同一个向量，
# 这样"变化了多少"就只反映一件事——这个位置的权重大不大。
PERTURB = np.array([0.1] * 8)


def vectors_of(words):
    return Tensor(np.array([EMBEDDINGS[word] for word in words]))


def average_context(vectors, skip=None):
    """第 12 章的做法：一个词的上下文 = **其它所有词**的平均。

    每个词出一样的力：1/(词数-1)。skip 是"自己的位置"，自己不算在里面。
    """
    total = np.zeros(vectors.shape[-1])
    count = 0
    for i in range(vectors.shape[0]):
        if i == skip:
            continue
        total = total + vectors.data[i]
        count += 1
    return total / count


def display_width(text):
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def pad(text, width, align="left"):
    space = " " * max(0, width - display_width(text))
    return text + space if align == "left" else space + text


def main():
    words = SENTENCE
    vectors = vectors_of(words)

    print("=" * 60)
    print("句子：", " ".join(words))
    print("=" * 60)

    ctx = average_context(vectors, skip=CENTER)
    print("\n'苹果'的上下文向量（其它 10 个词的平均）：")
    for dim, name in enumerate(["人/动物", "时间", "地点", "动作", "物品", "科技商业", "虚词", "程度"]):
        print(f"  {pad(name, 10)}{ctx[dim]:+.4f}")

    print("\n其实根本不用算，谁出了多少力是写死的：")
    print("-" * 60)
    weight = 1.0 / (len(words) - 1)
    for i, word in enumerate(words):
        if i == CENTER:
            print(f"  {pad(word, 6)}{'(自己)' :>6}  （自己不在自己的上下文里）")
        else:
            print(f"  {pad(word, 6)}{weight:>6.2f}  {'#' * 14}")
    print(f"  除自己以外的 {len(words) - 1} 个词，每个都是 "
          f"1/{len(words) - 1} = {weight:.2f}。")
    print("  '发布'是这样，'昨天'也是这样，'的'还是这样。")

    # ---------------- 扰动实验 ----------------
    print("\n" + "=" * 60)
    print("扰动实验：给第 i 个词的向量加一点点（每个位置加的一样多），")
    print("看'苹果'的上下文向量会变多少。")
    print("=" * 60)
    print(f"\n  {pad('改的是哪个词', 14)}{'上下文向量变化了多少':>22}")
    print("  " + "-" * 40)
    changes = []
    for i, word in enumerate(words):
        if i == CENTER:
            continue
        perturbed = vectors.data.copy()
        perturbed[i] = perturbed[i] + PERTURB
        original = average_context(vectors, skip=CENTER)
        changed = average_context(Tensor(perturbed), skip=CENTER)
        change = float(np.linalg.norm(changed - original))
        changes.append(change)
        print(f"  {pad(word, 14)}{change:>22.6f}")
    print(f"\n  最大变化 / 最小变化 = {max(changes) / min(changes):.3f}")
    print(f"  {len(changes)} 个位置，变化量一模一样 —— 平均法根本不知道谁重要。")


if __name__ == "__main__":
    main()
