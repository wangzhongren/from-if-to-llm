"""
第 11 章的测试：测的是本章真正的结论 ——

用「猜邻居」这个任务训练之后，词向量里长出了结构：
同类的词互相靠近，不同类的词互相远离，而 苹果 被两边的用法同时拉扯。

别的章节也可能有同名的 after.py，先清掉缓存，确保导入的是本章的。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

sys.modules.pop("after", None)

import numpy as np

import after

TECH_WORDS = "发布 新 手机 芯片 电脑 华为 小米".split()
FOOD_WORDS = "好吃 很 甜 香蕉 这个 真 做成 派".split()

_cache = {}


def trained():
    """训练一次就够了，几个测试共用同一张表。"""
    if "table" not in _cache:
        corpus = after.encode(after.SENTENCES)
        seen, guessed = after.build_training_pairs(corpus)
        table, loss, _ = after.train_embeddings(seen, guessed)
        _cache["table"] = table
        _cache["loss"] = loss
        _cache["similarities"] = after.similarity_matrix(table.data)
    return _cache["table"], _cache["loss"], _cache["similarities"]


def group_similarity(similarities, group_a, group_b):
    index_a = [after.VOCAB.index(w) for w in group_a]
    index_b = [after.VOCAB.index(w) for w in group_b]
    values = [similarities[i, j]
              for i in index_a for j in index_b if i != j]
    return float(np.mean(values))


def separation(similarities):
    within = (group_similarity(similarities, TECH_WORDS, TECH_WORDS)
              + group_similarity(similarities, FOOD_WORDS, FOOD_WORDS)) / 2
    return within - group_similarity(similarities, TECH_WORDS, FOOD_WORDS)


# ---------------------------------------------------------------- 训练样本


def test_每对相邻的词都会被正反各收一次():
    """语料里「苹果 发布 新 手机」要变成「看到苹果猜发布」和「看到发布猜苹果」。"""
    corpus = after.encode(after.SENTENCES)
    seen, guessed = after.build_training_pairs(corpus, window=1)

    apple = after.VOCAB.index("苹果")
    publish = after.VOCAB.index("发布")

    pairs = set(zip(seen.tolist(), guessed.tolist()))
    assert (apple, publish) in pairs
    assert (publish, apple) in pairs

    # 第一句「苹果 发布 新 手机」贡献 6 条样本
    assert len(seen) == len(guessed)
    assert len(seen) > 50


def test_窗口开大之后训练样本变多():
    corpus = after.encode(after.SENTENCES)
    one = after.build_training_pairs(corpus, window=1)
    two = after.build_training_pairs(corpus, window=2)
    assert len(two[0]) > len(one[0])


# ---------------------------------------------------------------- 训练结果


def test_训练之后同类的词明显比不同类的词更像():
    _, _, similarities = trained()

    # 训练前这个差值在 0 附近晃，训练后应该是个大正数
    assert separation(similarities) > 0.3

    def sim(a, b):
        return similarities[after.VOCAB.index(a), after.VOCAB.index(b)]

    assert sim("手机", "芯片") > sim("手机", "好吃")
    assert sim("好吃", "甜") > sim("好吃", "发布")
    assert sim("手机", "芯片") > 0.8
    assert sim("好吃", "甜") > 0.8


def test_苹果被两边的用法同时拉扯():
    """苹果 的邻居里，食品词和科技词都得有。"""
    _, _, similarities = trained()
    top = [word for word, _ in after.nearest_neighbours(similarities, "苹果", top=5)]
    assert any(word in FOOD_WORDS for word in top)
    assert any(word in TECH_WORDS for word in top)


def test_训练确实改动了向量表而且形状没变():
    table, _, _ = trained()
    initial = after.randn(len(after.VOCAB), after.DIM, scale=0.1,
                          requires_grad=True, seed=after.TABLE_SEED)

    assert table.data.shape == (len(after.VOCAB), after.DIM)
    assert not np.allclose(table.data, initial.data)


def test_训练之后的loss明显低于乱猜():
    _, loss, _ = trained()
    random_guess = float(np.log(len(after.VOCAB)))
    assert loss < random_guess * 0.5


def test_同样的种子跑两次结果完全一样():
    corpus = after.encode(after.SENTENCES)
    seen, guessed = after.build_training_pairs(corpus)
    first, _, _ = after.train_embeddings(seen, guessed)
    second, _, _ = after.train_embeddings(seen, guessed)
    assert np.array_equal(first.data, second.data)


# ---------------------------------------------------------------- 随机表没结构


def test_随机表里没有分组结构():
    """对照组：同样形状的随机表，组内和组间几乎一样。"""
    random_table = after.randn(len(after.VOCAB), after.DIM, requires_grad=False, seed=0).data
    random_separation = separation(after.similarity_matrix(random_table))

    _, _, trained_similarities = trained()
    assert random_separation < 0.15
    assert separation(trained_similarities) > random_separation + 0.3


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
