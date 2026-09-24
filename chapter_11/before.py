"""第 11 章 before.py —— 拿上一章那张随机向量表来找"相似的词"。

第 10 章结尾我们做了一件事：既然每个词现在是一组数字，那就比一比谁跟谁像。
当时的结果是这样的：

    和 手机 最像的 5 个词： 派 +0.28  这个 +0.15  电脑 +0.13  做成 +0.09  小米 +0.09

和 手机 最像的词是 派。这一章要正面处理这件事。

这个文件把上一章那张表原封不动拿过来，用它来做"找相似的词"这件事 ——
也就是这一章真正想解决的问题。跑完你会看到三件事：

1. 找出来的"相似词"是胡话。
2. 换一个随机种子，同样的做法会给出完全不同的答案。
3. 按"科技 / 食品"给词分组，组内和组间的相似度根本没有差别。

第 3 条尤其要紧：它说明这张表里**连一点点分组结构都没有**。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import randn

# 和第 10 章完全一样的词表和语料
VOCAB = "苹果 发布 新 手机 芯片 电脑 华为 小米 好吃 很 甜 香蕉 这个 真 做成 派".split()

TECH_WORDS = "发布 新 手机 芯片 电脑 华为 小米".split()
FOOD_WORDS = "好吃 很 甜 香蕉 这个 真 做成 派".split()

# 第 10 章那张表每个词用 16 个数字。这里改成 8 个，是为了和本章
# 训练出来的那张表对齐（为什么是 8，见 README 的「新机制」一节）。
# 这不影响结论：随机数字有多少个都一样没结构，我们两种都试过。
DIM = 8


def display_width(text):
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def pad(text, width):
    return text + " " * (width - display_width(text))


def build_random_table(seed):
    """第 10 章的产物：每个词一组随机数字，只读，不训练。"""
    return randn(len(VOCAB), DIM, requires_grad=False, seed=seed).data


def similarity_matrix(table):
    """余弦相似度：两个向量方向越一致就越接近 1。"""
    unit = table / np.linalg.norm(table, axis=1, keepdims=True)
    return unit @ unit.T


def neighbours(similarities, word, top=5):
    index = VOCAB.index(word)
    order = np.argsort(-similarities[index])
    return [(VOCAB[j], similarities[index, j]) for j in order[1:top + 1]]


def group_similarity(similarities, group_a, group_b):
    """两组词之间的平均相似度。a 和 b 传同一组，就是组内平均。"""
    index_a = [VOCAB.index(w) for w in group_a]
    index_b = [VOCAB.index(w) for w in group_b]
    values = []
    for i in index_a:
        for j in index_b:
            if i != j:
                values.append(similarities[i, j])
    return float(np.mean(values))


def main():
    print("=" * 62)
    print("第 10 章那张随机表，直接拿来用")
    print("=" * 62)
    table = build_random_table(seed=0)
    similarities = similarity_matrix(table)
    for word in ["手机", "苹果", "好吃"]:
        pairs = "  ".join(f"{w} {s:+.2f}" for w, s in neighbours(similarities, word))
        print(f"  和 {pad(word, 6)} 最像的 5 个词： {pairs}")
    print()
    phone_word, phone_sim = neighbours(similarities, "手机", top=1)[0]
    apple_word, apple_sim = neighbours(similarities, "苹果", top=1)[0]
    print(f"  和 手机 最像的词是 {phone_word}（{phone_sim:+.2f}）。手机 和 {phone_word} 有什么关系？没有。")
    print(f"  和 苹果 最像的词是 {apple_word}（{apple_sim:+.2f}），这个看着倒像那么回事 ——")
    print("  但那只是运气。随机数偶尔会撞出一个「看起来对」的答案，")
    print("  这正是它最会骗人的地方。下面换个种子就知道。")
    print()

    print("=" * 62)
    print("换几个随机种子，再看同一个问题")
    print("=" * 62)
    print("  种子  和「手机」最像的 3 个词")
    print("  " + "-" * 44)
    for seed in [0, 1, 2, 3, 4]:
        table = build_random_table(seed=seed)
        pairs = "  ".join(f"{w} {s:+.2f}" for w, s in neighbours(similarity_matrix(table), "手机", top=3))
        print(f"  {seed:>4}  {pairs}")
    print()
    print("  每换一个种子，答案就换一批。")
    print("  如果这张表里真的装着「意义」，它不该对种子这么敏感。")
    print()

    print("=" * 62)
    print("这张表里有没有「科技词抱团、食品词抱团」的结构？")
    print("=" * 62)
    print("  组内平均相似度 = 同一类词之间的平均相似度")
    print("  组间平均相似度 = 科技词和食品词之间的平均相似度")
    print("  如果表里有结构，组内应该明显高于组间。")
    print()
    print("  种子    组内(科技)  组内(食品)  组间    组内均值 - 组间")
    print("  " + "-" * 56)
    gaps = []
    for seed in range(30):
        similarities = similarity_matrix(build_random_table(seed=seed))
        within_tech = group_similarity(similarities, TECH_WORDS, TECH_WORDS)
        within_food = group_similarity(similarities, FOOD_WORDS, FOOD_WORDS)
        between = group_similarity(similarities, TECH_WORDS, FOOD_WORDS)
        gap = (within_tech + within_food) / 2 - between
        gaps.append(gap)
        if seed < 3:
            print(f"  {seed:>4}    {within_tech:+.3f}      {within_food:+.3f}      {between:+.3f}   {gap:+.3f}")
    print()
    print("  上面是前三个种子。把 30 个种子都试一遍：")
    print(f"    平均 {np.mean(gaps):+.3f}    最小 {min(gaps):+.3f}    最大 {max(gaps):+.3f}")
    print("  不管种子怎么换，这个差距都在 0 附近打转（而且还常常是负的）。")
    print("  也就是说：科技词之间并不比科技词和食品词之间更像。")
    print()
    print("  结论：这张表里的数字是随机的，所以它里面什么结构都没有。")
    print("  我们没法靠换一个更好的种子把它修好 —— 随机数之间本来就没有关系。")
    print()
    print("  问题变成了：这组数字要**怎么才能被训练出来**？")


if __name__ == "__main__":
    main()
