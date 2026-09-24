"""第 14 章的测试：Q / K / V 三个角色是不是真的分开了，√d 是不是真的有用。

跑法：

    ./.venv/bin/python -m pytest chapter_14/tests/ -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from after import (  # noqa: E402
    CENTER,
    D_KEY,
    DIM,
    SENTENCE,
    attention,
    attention_scores,
    attention_weights,
    project,
    split_into_qkv,
    vectors_of,
)
from toygrad import Tensor, randn  # noqa: E402

WORDS = SENTENCE
VECTORS = vectors_of(WORDS)
INDEX = {word: i for i, word in enumerate(WORDS)}

W_Q = randn(DIM, D_KEY, seed=14)
W_K = randn(DIM, D_KEY, seed=24)
W_V = randn(DIM, D_KEY, seed=34)

QUERY, KEY, VALUE = split_into_qkv(VECTORS, W_Q, W_K, W_V)
SCORES = attention_scores(QUERY, KEY)
WEIGHTS = attention_weights(SCORES).data


def test_weights_sum_to_one():
    """每一行的权重加起来还是 1。"""
    assert np.allclose(WEIGHTS.sum(axis=1), 1.0)


def test_query_key_value_are_three_different_things():
    """Q、K、V 是三次不同的投影，两两都不一样。"""
    assert not np.allclose(QUERY.data, KEY.data)
    assert not np.allclose(KEY.data, VALUE.data)
    assert not np.allclose(QUERY.data, VALUE.data)


def test_scores_are_no_longer_symmetric():
    """第 13 章的分数矩阵是对称的，这一章不是——"我问你"和"你问我"分开了。"""
    plain = (VECTORS @ VECTORS.T).data
    assert np.allclose(plain, plain.T)

    scores = SCORES.data
    assert not np.allclose(scores, scores.T)


def test_dividing_by_sqrt_d_changes_the_score_scale():
    """除不除 √d，分数的尺度不一样。"""
    raw = (QUERY @ KEY.T).data
    assert np.allclose(raw / np.sqrt(D_KEY), SCORES.data)


def test_output_is_mixed_from_values_not_from_raw_vectors():
    """加权求和的对象是 Value，不是原来的词向量。"""
    output, _ = attention(QUERY, KEY, VALUE)
    manual = WEIGHTS @ VALUE.data
    assert np.allclose(output.data, manual)

    # 如果拿原始词向量去混，结果不一样
    wrong = WEIGHTS @ VECTORS.data
    assert not np.allclose(output.data, wrong)


def test_different_question_gives_different_weights():
    """同一个词，换一个 W_q，问到的词就变了。"""
    w_q_tech = np.zeros((DIM, DIM))
    w_q_tech[5][5] = 4.0            # 只问"科技与商业"
    w_q_item = np.zeros((DIM, DIM))
    w_q_item[4][4] = 4.0            # 只问"物品"

    tech = attention_weights(attention_scores(project(VECTORS, Tensor(w_q_tech)), KEY)).data
    item = attention_weights(attention_scores(project(VECTORS, Tensor(w_q_item)), KEY)).data

    assert not np.allclose(tech[CENTER], item[CENTER])
    assert tech[CENTER][INDEX["发布"]] > tech[CENTER][INDEX["我"]]
    assert item[CENTER][INDEX["手机"]] > item[CENTER][INDEX["商场"]]


def test_sqrt_d_keeps_the_score_scale_stable():
    """维度涨了 16 倍，除以 √d 之后分数的尺度基本不动；不除就飞了。

    单次随机抽样的波动很大，所以每一档都抽 10 次取平均。
    """
    matrix = VECTORS.data

    def mean_std(dim):
        raw_list, scaled_list = [], []
        for t in range(10):
            rng = np.random.default_rng(1000 * t + dim)
            w_q = rng.standard_normal((DIM, dim))
            w_k = rng.standard_normal((DIM, dim))
            raw = (matrix @ w_q) @ (matrix @ w_k).T
            raw_list.append(raw[CENTER].std())
            scaled_list.append((raw / np.sqrt(dim))[CENTER].std())
        return np.mean(raw_list), np.mean(scaled_list)

    small_raw, small_scaled = mean_std(8)
    large_raw, large_scaled = mean_std(128)

    assert large_raw / small_raw > 3.0        # 不缩放：尺度飞了
    assert large_scaled / small_scaled < 2.0  # 缩放：尺度稳住了


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
