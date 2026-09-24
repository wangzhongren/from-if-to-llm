"""第 22 章的测试：零件装对了没有。

跑法：

    ./.venv/bin/python -m pytest chapter_22/tests/ -q

这一章测的东西比前面几章多，因为它第一次把全部零件装到了一起。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from after import (  # noqa: E402
    CONTEXT,
    CORPUS,
    DIM,
    HEADS,
    LAYERS,
    VOCAB_SIZE,
    CharTokenizer,
    TinyLanguageModel,
    TransformerBlock,
    build_batches,
    causal_mask,
    continue_text,
    next_token_probabilities,
)
from toygrad import cross_entropy, no_grad  # noqa: E402

TOKENIZER = CharTokenizer(CORPUS)


def test_tokenizer_round_trips():
    """encode 再 decode，应该原样回来。第 1 章那个拆句子的动作。"""
    for text in ("床前明月光", "鹅鹅鹅曲项向天歌", "低头思故乡"):
        assert TOKENIZER.decode(TOKENIZER.encode(text)) == text


def test_tokenizer_only_knows_the_corpus():
    """词表就是语料里出现过的字符，一个不多一个不少。"""
    assert TOKENIZER.vocab_size == VOCAB_SIZE
    assert sorted(TOKENIZER.vocab) == sorted(set(CORPUS))


def test_every_training_target_is_the_next_character():
    """输入右移一位就是目标。这就是"预测下一个 token"的意思。"""
    ids = TOKENIZER.encode(CORPUS)
    inputs, targets = build_batches(ids)
    for index in range(3):
        assert (targets[index][:-1] == inputs[index][1:]).all()


def test_the_model_outputs_one_distribution_per_position():
    """每个位置都必须给出一个完整的概率分布。"""
    model = TinyLanguageModel(VOCAB_SIZE)
    ids = TOKENIZER.encode(CORPUS)[:CONTEXT][None, :]
    with no_grad():
        probabilities = model(ids).softmax(axis=-1).data[0]
    assert probabilities.shape == (CONTEXT, VOCAB_SIZE)
    assert np.allclose(probabilities.sum(axis=-1), 1.0)


def test_multi_head_splits_the_dimension_evenly():
    """多头就是把最后一维切成几份，切完必须能拼回来。"""
    assert DIM % HEADS == 0
    block = TransformerBlock(DIM, HEADS, seed=1)
    assert block.head_dim == DIM // HEADS


def test_every_block_gets_the_causal_mask():
    """因果掩码必须传进每一个 Block。

    这里检查的是模型结构：Block 的 __call__ 一定要收下 mask 再往下传。
    只要有一层漏传，那一层就能看到未来。
    """
    model = TinyLanguageModel(VOCAB_SIZE, layers=2)
    ids = TOKENIZER.encode(CORPUS)[:CONTEXT][None, :]
    with no_grad():
        before = model(ids).data.copy()

    # 把掩码换掉：手动做一遍 forward，但每一层都不加掩码
    from toygrad import embedding  # noqa: E402

    x = embedding(model.token_embedding, ids)
    x = x + model.position_table[:CONTEXT].reshape(1, CONTEXT, DIM)
    for block in model.blocks:
        x = block(x, None)
    after = (x @ model.head_weight + model.head_bias).data

    assert not np.allclose(before, after)


def test_the_trained_model_continues_the_poem():
    """训练一小会儿，它能接出「...低头思故乡」。

    训练步数刻意压到很小，测试只要跑几秒。
    """
    inputs, targets = build_batches(TOKENIZER.encode(CORPUS))
    model = TinyLanguageModel(VOCAB_SIZE, dim=32, layers=2)
    from toygrad import Adam  # noqa: E402

    optimizer = Adam(model.params(), lr=3e-3)
    rng = np.random.default_rng(20260924)
    for _ in range(400):
        batch = rng.integers(0, len(inputs), size=16)
        optimizer.zero_grad()
        cross_entropy(model(inputs[batch]), targets[batch]).backward()
        optimizer.step()

    text = continue_text(model, TOKENIZER, "床前明月光", length=10)
    assert text.startswith("床前明月光疑是地上霜举头")


def test_the_causal_mask_is_still_there():
    """第 21 章那件事不能在装配的时候丢掉。"""
    mask = causal_mask(CONTEXT)
    assert mask.shape == (CONTEXT, CONTEXT)
    assert not mask.diagonal().any()
    assert mask[0, 1] and mask[0, CONTEXT - 1]


def test_prompt_longer_than_the_context_is_cut_not_crashed():
    """给一段比上下文还长的开头，应该只取最后 CONTEXT 个字符。"""
    model = TinyLanguageModel(VOCAB_SIZE)
    long_prompt = CORPUS[:CONTEXT * 2]
    probabilities = next_token_probabilities(model, TOKENIZER, long_prompt)
    assert probabilities.shape == (VOCAB_SIZE,)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
