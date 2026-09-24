"""第 11 章 experiment.py —— 训练前 vs 训练后。

这个文件做三个对比实验，全部围绕同一张表：

实验一  训练过程中 loss 怎么走。（随机乱猜是 ln(16) ≈ 2.77）
实验二  「组内相似度 - 组间相似度」这个数，训练前和训练后差多少。
实验三  两张图放在一起看：训练前是一团乱，训练后分成两堆。

为了不把画图和训练那两段代码抄第二遍，这个文件直接复用 after.py 里的函数。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import randn

import after

TECH_WORDS = "发布 新 手机 芯片 电脑 华为 小米".split()
FOOD_WORDS = "好吃 很 甜 香蕉 这个 真 做成 派".split()


def group_similarity(similarities, group_a, group_b):
    """两组词之间的平均相似度。两组传同一组，就是组内平均。"""
    index_a = [after.VOCAB.index(w) for w in group_a]
    index_b = [after.VOCAB.index(w) for w in group_b]
    values = [similarities[i, j]
              for i in index_a for j in index_b if i != j]
    return float(np.mean(values))


def separation(similarities):
    """组内平均 减去 组间平均。越大说明两类词分得越开。"""
    within = (group_similarity(similarities, TECH_WORDS, TECH_WORDS)
              + group_similarity(similarities, FOOD_WORDS, FOOD_WORDS)) / 2
    between = group_similarity(similarities, TECH_WORDS, FOOD_WORDS)
    return within, between, within - between


def experiment_loss():
    print("=" * 62)
    print("实验一：训练过程中的 loss")
    print("=" * 62)

    corpus = after.encode(after.SENTENCES)
    seen, guessed = after.build_training_pairs(corpus)

    _, final_loss, history = after.train_embeddings(seen, guessed, progress_every=40)
    print("  轮数     loss")
    print("  " + "-" * 20)
    for epoch, loss in history:
        bar = "█" * int(round(loss / 2.77 * 30))
        print(f"  {epoch:>4}   {loss:6.4f}  {bar}")
    print()
    print(f"  最后一轮的 loss = {final_loss:.4f}")
    print("  横条的长度是相对于「随机乱猜」的 loss = ln(16) ≈ 2.77 画的。")
    print("  loss 从 2.77 掉到 1 附近，说明模型确实在这 68 个样本上学到了东西。")
    print()


def experiment_separation():
    print("=" * 62)
    print("实验二：组内相似度 - 组间相似度")
    print("=" * 62)
    print("  科技词 7 个、食品词 8 个。如果表里有结构，")
    print("  同类的词应该比不同类的词更像，这个差值应该是正的，而且不小。")
    print()

    random_table = randn(len(after.VOCAB), after.DIM, requires_grad=False, seed=0).data
    random_similarities = after.similarity_matrix(random_table)
    within_r, between_r, gap_r = separation(random_similarities)

    corpus = after.encode(after.SENTENCES)
    seen, guessed = after.build_training_pairs(corpus)
    table, _, _ = after.train_embeddings(seen, guessed)
    trained_similarities = after.similarity_matrix(table.data)
    within_t, between_t, gap_t = separation(trained_similarities)

    print("                   组内平均   组间平均   组内 - 组间")
    print("  " + "-" * 52)
    print(f"  训练前（随机）    {within_r:+.3f}     {between_r:+.3f}     {gap_r:+.3f}")
    print(f"  训练后            {within_t:+.3f}     {between_t:+.3f}     {gap_t:+.3f}")
    print()
    print(f"  训练前这个差值在 0 附近（{gap_r:+.3f}）；训练后是 {gap_t:+.3f}。")
    print("  同一个任务、同一批词、同一张表的形状 —— 只把表里的数字训练了一遍。")
    print()


def experiment_two_pictures():
    print("=" * 62)
    print("实验三：同一张表的两种样子")
    print("=" * 62)
    print()

    random_table = randn(len(after.VOCAB), after.DIM, requires_grad=False, seed=0).data
    after.scatter(random_table, title="训练前（随机数字）：")

    corpus = after.encode(after.SENTENCES)
    seen, guessed = after.build_training_pairs(corpus)
    table, _, _ = after.train_embeddings(seen, guessed)
    after.scatter(table.data, title="训练后（用「猜邻居」训练过）：")

    print("  两张图的画法完全一样（横纵轴各自按自己的范围铺满，比的是相对位置）。")
    print()
    print("  训练前：香蕉 和 发布 挨在一起，华为 和 小米 隔了半张图，")
    print("          完全看不出哪几个词是一类 —— 因为随机数字之间没有关系。")
    print("  训练后：食品词一片在上、科技词一片在下，中间夹着 苹果。")
    print()

    similarities = after.similarity_matrix(table.data)
    print("  再看一眼 苹果 的邻居 —— 这是一个我们还没解决的问题：")
    pairs = "  ".join(f"{w} {s:.2f}" for w, s in after.nearest_neighbours(similarities, "苹果"))
    print(f"    {pairs}")
    print()
    print("  香蕉、好吃 是食品那边的，华为、小米 是科技那边的，")
    print("  它们全都算 苹果 的邻居 —— 因为 苹果 在两种句子里都出现过。")
    print("  可 苹果 只有一个向量。第 12 章就从这个地方接着往下走。")


def main():
    experiment_loss()
    experiment_separation()
    experiment_two_pictures()


if __name__ == "__main__":
    main()
