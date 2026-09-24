"""第 13 章的测试：Attention 的三步到底做对了没有。

跑法：

    ./.venv/bin/python -m pytest chapter_13/tests/ -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from after import (  # noqa: E402
    CENTER,
    SENTENCE,
    attention_output,
    attention_weights,
    average_context,
    similarity_scores,
    vectors_of,
)

WORDS = SENTENCE
VECTORS = vectors_of(WORDS)
SCORES = similarity_scores(VECTORS)
WEIGHTS = attention_weights(SCORES).data
INDEX = {word: i for i, word in enumerate(WORDS)}


def test_every_row_of_weights_sums_to_one():
    """权重是一组概率：每一行加起来必须等于 1。"""
    assert np.allclose(WEIGHTS.sum(axis=1), 1.0)


def test_weights_are_never_negative():
    """权重不能是负数——不然"该看谁"就说不通了。"""
    assert (WEIGHTS >= 0).all()


def test_similar_word_gets_more_weight_than_unrelated_word():
    """对"苹果"来说，"手机""发布"应该比"昨天""商场"更受重视。"""
    row = WEIGHTS[CENTER]
    assert row[INDEX["手机"]] > row[INDEX["昨天"]]
    assert row[INDEX["发布"]] > row[INDEX["商场"]]


def test_attention_output_is_the_weighted_sum_of_vectors():
    """新的向量就是所有词向量按权重加权求和，手算一遍应该一样。"""
    output = attention_output(VECTORS, attention_weights(SCORES)).data
    i = INDEX["发布"]
    manual = sum(WEIGHTS[i][j] * VECTORS.data[j] for j in range(len(WORDS)))
    assert np.allclose(output[i], manual)


def test_average_gives_every_word_the_same_weight():
    """第 12 章的平均法：其它每个词都出一样的力——这就是本章要修的问题。"""
    n = len(WORDS)
    plain = average_context(VECTORS, CENTER).data
    manual = (VECTORS.data.sum(axis=0) - VECTORS.data[CENTER]) / (n - 1)
    assert np.allclose(plain, manual)

    # 平均法的"权重表"是常数（自己是 0），而 Attention 的每一行都不一样
    assert not np.allclose(WEIGHTS[CENTER], 1.0 / (n - 1))


def test_attention_weights_depend_on_who_is_looking():
    """"苹果"看谁，和"手机"看谁，不是同一套权重。"""
    assert not np.allclose(WEIGHTS[INDEX["苹果"]], WEIGHTS[INDEX["手机"]])


def test_weights_are_sharper_than_the_flat_average():
    """平均法的权重方差是 0；Attention 的权重是分得开轻重的。"""
    row = WEIGHTS[CENTER]
    assert row.std() > 0.0
    assert row.max() / row.min() > 2.0


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
