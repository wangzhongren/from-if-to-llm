"""第 14 章 before.py —— 第 13 章的做法，卡在哪里。

第 13 章的分数是：

    score[i][j] = v_i · v_j

这个文件做两个小实验，把它的两个毛病摆出来：

  实验一：这个分数是对称的。"我问你"和"你问我"用的是同一个数。
  实验二：想换一个"问法"，只能改自己的向量——可那个向量同时还是"我是谁"。

跑法：

    ./.venv/bin/python chapter_14/before.py
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
FEATURE_NAMES = ["人/动物", "时间", "地点", "动作", "物品", "科技与商业", "虚词", "程度"]


def display_width(text):
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def pad(text, width, align="left"):
    space = " " * max(0, width - display_width(text))
    return text + space if align == "left" else space + text


def main():
    words = SENTENCE
    matrix = np.array([EMBEDDINGS[word] for word in words])

    print("=" * 60)
    print("句子：", " ".join(words))
    print("=" * 60)

    # ---------------- 实验一：分数是对称的 ----------------
    scores = matrix @ matrix.T
    print("\n实验一：分数矩阵是对称的吗？")
    print("-" * 60)
    print(f"  整个矩阵 S 和它的转置相等：{np.allclose(scores, scores.T)}")
    i, j = CENTER, words.index("手机")
    print(f"  {pad('苹果 -> 手机', 14)}{scores[i][j]:>8.3f}")
    print(f"  {pad('手机 -> 苹果', 14)}{scores[j][i]:>8.3f}")
    print("\n  两个方向是同一个数。'苹果有多想问手机'和'手机有多想问苹果'，")
    print("  在第 13 章里根本无法区分——因为只有一把尺子。")

    # ---------------- 实验二：想换个问法，只能改自己 ----------------
    print("\n" + "=" * 60)
    print("实验二：假设'苹果'想换一个关注点")
    print("=" * 60)
    print("\n  现在'苹果'问谁，是被它自己的向量锁死的：")
    print(f"  苹果 = {matrix[CENTER]}")
    print(f"  于是它最想问的是'手机'（{scores[CENTER][j]:.3f}）和'发布'"
          f"（{scores[CENTER][words.index('发布')]:.3f}）。")

    # 把"科技与商业"这一维抹掉，看看会发生什么
    changed = matrix.copy()
    changed[CENTER][5] = 0.0
    after = changed @ changed.T
    print("\n  我们试着改一改它：把'科技与商业'这一维从 1.0 改成 0.0。")
    print(f"  苹果' = {changed[CENTER]}")
    print(f"\n  {pad('分数', 16)}{'改之前':>10}{'改之后':>10}")
    print("  " + "-" * 40)
    pairs = [
        ("苹果 -> 手机", scores[CENTER][j], after[CENTER][j]),
        ("苹果 -> 发布", scores[CENTER][words.index("发布")], after[CENTER][words.index("发布")]),
        ("苹果 -> 苹果", scores[CENTER][CENTER], after[CENTER][CENTER]),
        ("手机 -> 苹果", scores[j][CENTER], after[j][CENTER]),
    ]
    for name, before_value, after_value in pairs:
        print(f"  {pad(name, 16)}{before_value:>10.3f}{after_value:>10.3f}")

    print("\n  最后一行是重点：手机一个字都没改，可是'手机怎么看苹果'变了。")
    print("  因为'苹果'只有一个向量——它同时是'我看别人时用的那个'")
    print("  和'别人看我时用的那个'。想改其中一个，另一个必然跟着动。")

    print("\n  结论：第 13 章的每一对词之间只有一个数，这个数还要同时扮演两个角色。")
    print("  '我在找什么'和'我有什么'被绑在一起了。")


if __name__ == "__main__":
    main()
