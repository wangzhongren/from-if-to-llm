"""
第 12 章的测试：测的是本章真正的两个结论 ——

1. 「一个词一个向量」装不下两种意思：两句话里的 苹果 向量完全相同，
   任何只看这个词向量的判断都只能给出同一个答案。
2. 换成「周围词的平均」之后，同一个词在不同句子里有了不同的表示，
   那个判断也就跟着句子翻面了。同时，"平均"本身是粗糙的。

别的章节也可能有同名的 after.py / before.py，先清掉缓存，确保导入的是本章的。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

sys.modules.pop("after", None)
sys.modules.pop("before", None)

import numpy as np

import after
import before

FOOD_WORDS = "好吃 甜 香蕉 派 很 真 这个 做成".split()
TECH_WORDS = "发布 新 手机 芯片 电脑 华为 小米".split()

_cache = {}


def trained_table():
    """训练一次就够了，几个测试共用同一张表。"""
    if "table" not in _cache:
        corpus = after.encode(after.SENTENCES)
        seen, guessed = after.build_training_pairs(corpus)
        table, loss = after.train_embeddings(seen, guessed)
        _cache["table"] = table.data
        _cache["loss"] = loss
    return _cache["table"]


def food_or_tech(vector):
    """本章那个小任务：和食品组更像，还是和科技组更像。"""
    food = np.mean([after.cosine(vector, trained_table()[after.VOCAB.index(w)]) for w in FOOD_WORDS])
    tech = np.mean([after.cosine(vector, trained_table()[after.VOCAB.index(w)]) for w in TECH_WORDS])
    return food - tech


def apple_vector_in(sentence):
    position = sentence.split().index("苹果")
    return after.context_vector(trained_table(), sentence, position)


# ---------------------------------------------------------------- 第 11 章的做法


def test_静态表示里同一个词在任何句子里都是同一个向量():
    """第 11 章的表里，「苹果」只有一行，查多少次都是它。"""
    table = trained_table()
    index = after.VOCAB.index("苹果")

    in_food = table[index]
    in_tech = table[index]
    assert np.array_equal(in_food, in_tech)
    # 自己和自己比，余弦是 1（浮点数上差一点点）
    assert abs(after.cosine(in_food, in_tech) - 1.0) < 1e-9


def test_静态表示对两句里的苹果给出同一个判断():
    """静态表示下，两个句子的「食品分 - 科技分」是同一个数。"""
    table = trained_table()
    apple = table[after.VOCAB.index("苹果")]

    assert food_or_tech(apple) == food_or_tech(apple)

    # 而且是偏食品的 —— 所以三句说公司的句子都会被判错
    assert food_or_tech(apple) > 0


# ---------------------------------------------------------------- 本章的做法


def test_上下文向量就是周围词向量的平均():
    table = trained_table()
    sentence = "苹果 很 好吃"
    ids = [after.VOCAB.index(word) for word in sentence.split()]

    vector = after.context_vector(table, sentence, 0)
    expected = (table[ids[1]] + table[ids[2]]) / 2

    assert np.allclose(vector, expected)


def test_上下文向量把自己排除在外():
    """算 苹果 的上下文时不能把 苹果 自己算进去。"""
    table = trained_table()
    sentence = after.LONG_SENTENCE
    position = sentence.split().index("苹果")
    vector = after.context_vector(table, sentence, position)

    others = [after.VOCAB.index(word) for i, word in enumerate(sentence.split())
              if i != position]
    assert len(others) == 10
    assert np.allclose(vector, table[others].mean(axis=0))


def test_两个句子里的苹果终于是两个不同的表示了():
    table = trained_table()
    in_food = apple_vector_in(after.FOOD_SENTENCE)
    in_tech = apple_vector_in(after.TECH_SENTENCE)

    assert not np.allclose(in_food, in_tech)
    assert after.cosine(in_food, in_tech) < 0.5

    # 而且方向是对着的：食品句里偏食品，科技句里偏科技
    assert food_or_tech(in_food) > 0
    assert food_or_tech(in_tech) < 0


def test_上下文表示让这个小任务判对了科技句():
    """静态表示把「苹果 发布 新 手机」判成水果；上下文表示判成公司。"""
    table = trained_table()
    static_score = food_or_tech(table[after.VOCAB.index("苹果")])
    context_score = food_or_tech(apple_vector_in(after.TECH_SENTENCE))

    assert static_score > 0
    assert context_score < 0


def test_长句里的上下文向量换一句就变():
    """同一个 苹果，在长句里的表示和在其他句子里的也不一样。"""
    table = trained_table()
    long_vector = apple_vector_in(after.LONG_SENTENCE)

    assert not np.allclose(long_vector, apple_vector_in(after.FOOD_SENTENCE))
    assert not np.allclose(long_vector, apple_vector_in(after.TECH_SENTENCE))


# ---------------------------------------------------------------- 平均的粗糙


def test_平均法对句子里每个词一视同仁():
    """拿掉任何一个词，向量的变化量都差不多 —— 它分不出轻重。"""
    table = trained_table()
    sentence = after.LONG_SENTENCE
    words = sentence.split()
    position = words.index("苹果")

    full = after.context_vector(table, sentence, position)
    other_ids = [after.VOCAB.index(word) for i, word in enumerate(words) if i != position]

    changes = []
    for k in range(len(other_ids)):
        kept = [word_id for j, word_id in enumerate(other_ids) if j != k]
        changes.append(float(np.linalg.norm(full - table[kept].mean(axis=0))))

    # 全部落在 [0.15, 0.40] 这一段里，最大和最小相差不到两倍
    assert min(changes) > 0.15
    assert max(changes) < 0.40
    assert max(changes) / min(changes) < 2.0


def test_全句平均会把不相干的高频词也平均进来():
    """长句平均出来最像的词是「的」，而不是「发布」。"""
    table = trained_table()
    vector = apple_vector_in(after.LONG_SENTENCE)

    sim_de = after.cosine(vector, table[after.VOCAB.index("的")])
    sim_publish = after.cosine(vector, table[after.VOCAB.index("发布")])

    assert sim_de > sim_publish


def test_全句平均和只看窗口的结果不一样():
    """同一条句子、同一个位置，换个取法就得到很不相同的表示。"""
    table = trained_table()
    words = after.LONG_SENTENCE.split()
    position = words.index("苹果")

    full = after.context_vector(table, after.LONG_SENTENCE, position)
    window_words = words[max(0, position - 2):position] + words[position + 1:position + 3]
    window = table[[after.VOCAB.index(word) for word in window_words]].mean(axis=0)

    assert after.cosine(full, window) < 0.8


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
