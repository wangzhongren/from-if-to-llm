"""第 24 章的测试：探测器本身准不准。

跑法：

    ./.venv/bin/python -m pytest chapter_24/tests/ -q

这一章没有新机制，所以测试测的是"量尺子"本身：
如果尺子不准，我们量出来的所有结论都不算数。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from after import (  # noqa: E402
    CONTEXT,
    CORPUS,
    VOCAB_SIZE,
    CharTokenizer,
    TinyLanguageModel,
    evaluate_text,
    perplexity,
    shuffle_blocks,
    shuffle_whole,
)

TOKENIZER = CharTokenizer(CORPUS)

# 一个没训练过的小模型：探测器测的是"量得准不准"，
# 不是"模型好不好"，所以不需要训练。
SMALL_MODEL = TinyLanguageModel(VOCAB_SIZE, dim=16, layers=1)


def test_perplexity_of_uniform_guess_is_the_vocab_size():
    """瞎猜的时候，困惑度应该正好等于词表大小。

    这是这把尺子的零点。零点不对，后面所有的数都不能信。
    """
    uniform_loss = np.log(VOCAB_SIZE)
    assert np.isclose(perplexity(uniform_loss), VOCAB_SIZE, rtol=1e-9)


def test_perplexity_of_a_perfect_prediction_is_one():
    """百分百确定的时候，困惑度是 1。这是尺子的另一头。"""
    assert np.isclose(perplexity(0.0), 1.0)


def test_shuffling_keeps_every_character():
    """打乱只是换顺序，一个字都不能多、一个字都不能少。

    不然"困惑度涨了"就不是因为顺序，而是因为内容变了。
    """
    shuffled = shuffle_whole(CORPUS, seed=1)
    assert len(shuffled) == len(CORPUS)
    assert sorted(shuffled) == sorted(CORPUS)
    assert shuffled != CORPUS


def test_block_shuffling_keeps_blocks_intact():
    """块内打乱不会把字搬到别的块里去。"""
    for block_size in (2, 4, 8):
        result = shuffle_blocks(CORPUS, block_size, seed=1)
        assert len(result) == len(CORPUS)
        for start in range(0, len(CORPUS), block_size):
            block = CORPUS[start:start + block_size]
            reordered = result[start:start + block_size]
            assert sorted(block) == sorted(reordered)


def test_evaluate_text_reports_what_it_measured():
    """量一段文字应该给出 loss、困惑度、准确率、位置数、错误数五样东西。"""
    result = evaluate_text(SMALL_MODEL, TOKENIZER, CORPUS)
    assert result["positions"] > 0
    assert result["wrong"] <= result["positions"]
    assert 0.0 <= result["accuracy"] <= 1.0
    assert np.isclose(result["perplexity"], np.exp(result["loss"]))
    # loss 和"猜对比例"必须能对上：准确率越高，loss 一般越低
    assert result["accuracy"] > 0.0


def test_evaluate_text_refuses_text_that_is_too_short():
    """太短的文字量不出来——我们不悄悄补零，直接报错。"""
    try:
        evaluate_text(SMALL_MODEL, TOKENIZER, "床前明月光")
    except ValueError:
        return
    raise AssertionError("文字短于上下文长度时应该直接报错")


def test_windows_cover_every_position_of_the_corpus():
    """一段长度为 L 的文字，应该切出 L - CONTEXT 个"猜下一个字"的位置。"""
    ids = TOKENIZER.encode(CORPUS)
    result = evaluate_text(SMALL_MODEL, TOKENIZER, CORPUS)
    assert result["positions"] == (len(ids) - CONTEXT) * CONTEXT


def test_a_trained_model_is_certain_on_the_corpus_and_lost_off_it():
    """这一章的核心对照，用数字的形式写下来。

    训练一小会儿，然后量两段文字：原文，和它的打乱版。
    原文上的困惑度必须远低于打乱版上的。
    """
    from toygrad import Adam, cross_entropy  # noqa: E402

    from after import build_batches  # noqa: E402

    inputs, targets = build_batches(TOKENIZER.encode(CORPUS))
    model = TinyLanguageModel(VOCAB_SIZE, dim=32, layers=2)
    optimizer = Adam(model.params(), lr=3e-3)
    rng = np.random.default_rng(20260924)
    for _ in range(600):
        batch = rng.integers(0, len(inputs), size=16)
        optimizer.zero_grad()
        cross_entropy(model(inputs[batch]), targets[batch]).backward()
        optimizer.step()

    on_corpus = evaluate_text(model, TOKENIZER, CORPUS)["perplexity"]
    on_shuffled = evaluate_text(model, TOKENIZER, shuffle_whole(CORPUS, seed=20260924))["perplexity"]
    assert on_corpus < 2.0
    assert on_shuffled > 100.0


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
