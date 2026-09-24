"""第 16 章的测试：位置信息是不是真的进去了。

跑法：

    ./.venv/bin/python -m pytest chapter_16/tests/ -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from after import (  # noqa: E402
    DIM,
    SENTENCE_A,
    SENTENCE_B,
    attention,
    index_positions,
    sinusoidal_positions,
)

LENGTH = 11
TABLE = sinusoidal_positions(LENGTH, DIM)
NO_POSITION = np.zeros((LENGTH, DIM))


def sentence_vector(words, position_table):
    output, _ = attention(words, position_table[: len(words)])
    return output.data.sum(axis=0)


def test_without_positions_the_word_order_does_not_matter():
    """没有位置信息：输出只是跟着换位置，整句话的表示一模一样。"""
    output_a, _ = attention(SENTENCE_A, NO_POSITION[: len(SENTENCE_A)])
    output_b, _ = attention(SENTENCE_B, NO_POSITION[: len(SENTENCE_B)])

    ren_in_a = SENTENCE_A.index("人")
    assert np.allclose(output_b.data[0], output_a.data[ren_in_a])
    assert np.allclose(sentence_vector(SENTENCE_A, NO_POSITION),
                       sentence_vector(SENTENCE_B, NO_POSITION))


def test_with_positions_the_word_order_matters():
    """加了位置编码之后，同样的三个词换个顺序，结果不一样了。"""
    output_a, _ = attention(SENTENCE_A, TABLE[: len(SENTENCE_A)])
    output_b, _ = attention(SENTENCE_B, TABLE[: len(SENTENCE_B)])

    ren_in_a = SENTENCE_A.index("人")
    assert not np.allclose(output_b.data[0], output_a.data[ren_in_a])

    distance = np.linalg.norm(sentence_vector(SENTENCE_A, TABLE)
                              - sentence_vector(SENTENCE_B, TABLE))
    assert distance > 0.1


def test_position_vectors_are_all_different():
    """每个位置一个向量，两两不能撞车。"""
    for i in range(LENGTH):
        for j in range(i + 1, LENGTH):
            assert not np.allclose(TABLE[i], TABLE[j])


def test_position_vectors_have_a_constant_length():
    """位置编码的长度和位置无关，永远在 -1 和 1 之间。"""
    lengths = np.linalg.norm(TABLE, axis=1)
    assert np.allclose(lengths, lengths[0])
    assert TABLE.max() <= 1.0
    assert TABLE.min() >= -1.0


def test_similarity_depends_only_on_distance():
    """两个位置的相似度只看「隔多远」，不看「在第几位」。"""
    for distance in [1, 2, 3]:
        values = [float(TABLE[i] @ TABLE[i + distance]) for i in range(0, 5)]
        assert np.allclose(values, values[0])


def test_index_positions_grow_with_the_position():
    """最笨的那个方案：编号向量的长度随着位置一路涨。"""
    index_table = index_positions(51, DIM)
    assert np.linalg.norm(index_table[50]) > 9 * np.linalg.norm(index_table[5])
    # 位置编码不会：长度一直是 2
    assert np.isclose(np.linalg.norm(TABLE[0]), np.linalg.norm(TABLE[10]))


def test_position_is_added_not_replacing_the_word():
    """位置是「加」在词向量上的，不是把词向量换掉。"""
    output_with, _ = attention(SENTENCE_A, TABLE[: len(SENTENCE_A)])
    output_without, _ = attention(SENTENCE_A, NO_POSITION[: len(SENTENCE_A)])
    assert not np.allclose(output_with.data, output_without.data)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
