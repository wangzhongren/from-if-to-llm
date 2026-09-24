"""第 21 章的测试：因果掩码到底挡住了什么。

跑法：

    ./.venv/bin/python -m pytest chapter_21/tests/ -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from toygrad import embedding, no_grad  # noqa: E402

from after import (  # noqa: E402
    CHAR_TO_ID,
    CONTEXT,
    CORPUS_IDS,
    VOCAB_SIZE,
    AttentionLM,
    build_batches,
    causal_mask,
    continue_text,
    evaluate,
    train,
)
from experiment import lookahead_mask, train_model, loss_with  # noqa: E402


def test_causal_mask_blocks_everything_to_the_right():
    """上三角全是 True，对角线和对角线左边全是 False。"""
    mask = causal_mask(6)
    for row in range(6):
        for column in range(6):
            assert mask[row][column] == (column > row)


def test_the_diagonal_is_still_allowed():
    """位置 i 必须能看到自己。连自己都挡掉，它就没法预测了。"""
    mask = causal_mask(8)
    assert not mask.diagonal().any()


def test_a_position_only_sees_itself_and_the_past():
    """把分数填成 -1e9 之后，softmax 出来的权重在后面那些位置上是 0。

    这是掩码生效的直接证据：不是"权重很小"，而是精确的 0。
    """
    model = AttentionLM(VOCAB_SIZE)
    length = 6
    token_ids = CORPUS_IDS[:length][None, :]
    with no_grad():
        x = embedding(model.token_table, token_ids) + model.position_table[:length].reshape(1, length, model.dim)
        attention = (x @ model.query_weight) @ (x @ model.key_weight).transpose(0, 2, 1)
        masked = attention.masked_fill(causal_mask(length), -1e9)
        weights = masked.softmax(axis=-1).data[0]

    for row in range(length):
        for column in range(length):
            if column > row:
                assert weights[row][column] == 0.0
            else:
                assert weights[row][column] > 0.0
        assert np.isclose(weights[row].sum(), 1.0)


def test_the_mask_changes_the_output():
    """挡和不挡，同一个模型出来的东西必须不一样。"""
    model = AttentionLM(VOCAB_SIZE)
    token_ids = CORPUS_IDS[:CONTEXT][None, :]
    with no_grad():
        with_mask = model(token_ids, use_causal_mask=True).data
        without_mask = model(token_ids, use_causal_mask=False).data
    assert not np.allclose(with_mask, without_mask)


def test_the_masked_model_learns_to_continue_the_poem():
    """把未来挡住之后，模型是真的学会了预测，接出来的字是对的。"""
    inputs, targets = build_batches(CORPUS_IDS)
    model = AttentionLM(VOCAB_SIZE)
    train(model, inputs, targets, use_causal_mask=True, steps=1200)
    text = continue_text(model, "床前明月光", length=15, use_causal_mask=True)
    assert text.startswith("床前明月光疑是地上霜举头望明月低头思故乡")


def test_a_peeking_model_collapses_when_the_future_is_masked():
    """偷看的模型，一旦把未来挡上就完全不会了。

    这就是第 21 章要说的那件事，用数字的形式写下来：
    训练 loss 很低（<0.01），但遮住未来之后 loss 会涨到 3 以上。
    """
    inputs, targets = build_batches(CORPUS_IDS)
    peeking = train_model(inputs, targets, lookahead=999, steps=1200)
    assert loss_with(peeking, inputs, targets, 999) < 0.05

    honest = train_model(inputs, targets, lookahead=0, steps=1200)
    assert loss_with(honest, inputs, targets, 0) < 0.05


def test_lookahead_mask_is_the_causal_mask_when_set_to_zero():
    """lookahead = 0 的掩码，就是 after.py 里的 causal_mask。"""
    assert (lookahead_mask(7, 0) == causal_mask(7)).all()


def test_lookahead_mask_lets_the_next_position_through():
    """lookahead = 1 时，位置 i 多看得见一格：i+1。"""
    mask = lookahead_mask(5, 1)
    assert not mask[0][1]
    assert mask[0][2]
    assert not mask[3][4]


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
