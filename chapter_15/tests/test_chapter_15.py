"""第 15 章的测试：多个头到底是不是"各看各的"。

跑法：

    ./.venv/bin/python -m pytest chapter_15/tests/ -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from after import (  # noqa: E402
    DIM,
    NUM_HEADS,
    SENTENCE_2,
    WORD_ID,
    make_parameters,
    multi_head_attention,
    train,
)
from toygrad import Tensor  # noqa: E402

WORDS = SENTENCE_2
IDS = [WORD_ID[word] for word in WORDS]

PARAMS = make_parameters(seed=15)
VECTORS = PARAMS["table"][IDS]
OUTPUT, WEIGHTS = multi_head_attention(VECTORS, PARAMS)
WEIGHTS = WEIGHTS.data


def test_weights_shape_is_one_matrix_per_head():
    """权重的形状是 (头数, 词数, 词数)：每个头有自己完整的一张表。"""
    assert WEIGHTS.shape == (NUM_HEADS, len(WORDS), len(WORDS))


def test_every_head_normalizes_its_own_row():
    """每个头、每一行的权重都加起来等于 1。"""
    assert np.allclose(WEIGHTS.sum(axis=-1), 1.0)


def test_heads_do_not_look_at_the_same_things():
    """不同头的权重矩阵不一样——这是"多头"的全部意义。"""
    for a in range(NUM_HEADS):
        for b in range(a + 1, NUM_HEADS):
            assert not np.allclose(WEIGHTS[a], WEIGHTS[b])


def test_each_head_output_is_its_own_mixture():
    """每个头的输出是"自己的权重 × 自己的 value"，不是几个头共用一份 value。"""
    value = (VECTORS @ PARAMS["w_v"]).data       # (头数, 词数, 每个头的维度)
    for head in range(NUM_HEADS):
        own = WEIGHTS[head] @ value[head]
        neighbour = WEIGHTS[head] @ value[(head + 1) % NUM_HEADS]
        assert not np.allclose(own, neighbour)


def test_one_head_gives_exactly_one_row_per_position():
    """一个头就是一个位置一行权重；四个头就是四行。"""
    params_one = make_parameters(seed=15, num_heads=1)
    _, weights_one = multi_head_attention(params_one["table"][IDS], params_one)
    assert weights_one.shape == (1, len(WORDS), len(WORDS))
    assert WEIGHTS.shape[0] == 4


def test_all_heads_contribute_to_the_output():
    """把任意一个头的结果清零，输出都会变——没有哪个头是白放的。"""
    base = OUTPUT.data.copy()
    for head in range(NUM_HEADS):
        value = VECTORS @ PARAMS["w_v"]
        head_output = WEIGHTS[head] @ value.data[head]
        without = np.concatenate(
            [head_output if h != head else np.zeros_like(head_output)
             for h in range(NUM_HEADS)], axis=-1
        )
        assert not np.allclose(without @ PARAMS["w_o"].data, base)


def test_training_makes_the_loss_go_down():
    """这套东西是能训的：损失会掉下来。"""
    _, history = train(steps=60, lr=0.1, seed=15)
    assert history[-1] < history[0]
    assert history[-1] < 1.0


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
