"""第 20 章的测试：下一个字符预测 = 分类，以及答案到底在不在输入里。

跑法：

    ./.venv/bin/python -m pytest chapter_20/tests/ -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from after import (  # noqa: E402
    CHAR_TO_ID,
    CORPUS_IDS,
    ID_TO_CHAR,
    NextCharModel,
    VOCAB,
    VOCAB_SIZE,
    WINDOW,
    build_pairs,
    evaluate,
    predict_next,
    train,
)
from experiment import build_pairs as build_probe_pairs  # noqa: E402
from experiment import copy_probe, run_group  # noqa: E402


def test_the_answer_sits_just_outside_the_window():
    """这一章最要紧的一条：窗口是 [t-WINDOW, t)，答案是 t 位置上的字符。

    答案所在的那个位置**不**在窗口里。第 21 章讲的就是这件事一旦被破坏会怎样。
    """
    inputs, targets = build_pairs(CORPUS_IDS)
    for index in range(len(inputs)):
        position = WINDOW + index
        assert (inputs[index] == CORPUS_IDS[position - WINDOW:position]).all()
        assert targets[index] == CORPUS_IDS[position]


def test_every_position_gets_exactly_one_training_pair():
    """语料里除了开头 WINDOW 个字符，每个位置都应该有一条训练样本。"""
    inputs, _ = build_pairs(CORPUS_IDS)
    assert len(inputs) == len(CORPUS_IDS) - WINDOW
    assert inputs.shape[1] == WINDOW


def test_vocab_is_small_enough_to_train_on_a_laptop():
    """语料是两首小诗，字符种类应该只有三十来个。"""
    assert 25 <= VOCAB_SIZE <= 38
    assert len(VOCAB) == len(set(VOCAB))
    # 编号和字符必须一一对应，语料里每个字符都要在词表里
    for index, char in enumerate(VOCAB):
        assert CHAR_TO_ID[char] == index
        assert ID_TO_CHAR[index] == char
    assert set(CORPUS_IDS) == set(range(VOCAB_SIZE))


def test_model_outputs_a_probability_distribution():
    """模型吐出来的是分数，softmax 之后必须是一组加起来等于 1 的概率。"""
    model = NextCharModel(VOCAB_SIZE)
    probabilities = predict_next(model, "床前明月光疑是地")
    assert probabilities.shape == (VOCAB_SIZE,)
    assert np.isclose(probabilities.sum(), 1.0)
    assert (probabilities >= 0).all()


def test_window_model_beats_looking_at_one_character():
    """看 8 个字符，应该比只看 1 个字符强。

    这条是本章的正题：下一个字符是哪一个，光看它前面那一个字是不够的。
    """
    inputs, targets = build_pairs(CORPUS_IDS)
    model = NextCharModel(VOCAB_SIZE)
    train(model, inputs, targets, steps=400)
    _, accuracy = evaluate(model, inputs, targets)

    single_char_inputs = inputs[:, -1:]
    single_char_model = NextCharModel(VOCAB_SIZE, window=1)
    train(single_char_model, single_char_inputs, targets, steps=400)
    _, single_accuracy = evaluate(single_char_model, single_char_inputs, targets)

    assert accuracy > single_accuracy
    assert accuracy > 0.95


def test_same_last_character_can_have_two_different_answers():
    """'头' 前面是 '举' 还是 '低'，后面的字不一样。

    只看一个字符的模型永远分不清这两个；看过 8 个字符的模型可以。
    """
    model = NextCharModel(VOCAB_SIZE)
    inputs, targets = build_pairs(CORPUS_IDS)
    train(model, inputs, targets, steps=600)

    after_jv = predict_next(model, "光疑是地上霜举头").argmax()
    after_di = predict_next(model, "霜举头望明月低头").argmax()
    assert ID_TO_CHAR[after_jv] == "望"
    assert ID_TO_CHAR[after_di] == "思"


def test_the_answer_in_the_input_teaches_the_model_to_copy():
    """只要答案在输入里，模型就学会抄。

    这是下一章那个坑的缩小模型：窗口 2，最后一格放答案。
    """
    model, _, accuracy = run_group("answer", random_context=True)
    _, _, copy_rate = copy_probe(model)
    assert accuracy > 0.99
    assert copy_rate > 0.95


def test_the_honest_model_does_not_copy():
    """作为对照：最后一格放的是当前字符时，模型不会把它抄出来。"""
    model, _, _ = run_group("current")
    _, _, copy_rate = copy_probe(model)
    assert copy_rate < 0.2


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
