"""第 6 章的测试：梯度到底对不对。

这一章唯一的新东西是"梯度"，所以测试都在问同一件事：
    这个方向，真的能让损失变小吗？
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 每一章都有一个 after.py，名字会撞车，先把上一章留下的清掉
sys.modules.pop("after", None)

from after import (  # noqa: E402
    CORPUS,
    VOCABULARY,
    WEIGHT_TABLE,
    accuracy,
    average_loss,
    gradient,
    numeric_gradient,
    reset_table,
    score,
    split_sentence,
    train,
)


def test_numeric_gradient_matches_a_hand_derived_one():
    """在一个能心算的小函数上，验证数值梯度算得对。

    f(w) = (w - 3)^2，它的梯度是 2 * (w - 3)。
    w = 0 的时候，梯度应该是 -6。
    """
    def f(w):
        return (w - 3.0) ** 2

    h = 1e-6
    w = 0.0
    numeric = (f(w + h) - f(w - h)) / (2 * h)
    assert abs(numeric - (-6.0)) < 1e-4


def test_formula_gradient_matches_numeric_gradient():
    """手推的公式和数值法必须给出同一张表。"""
    reset_table()
    formula = gradient(WEIGHT_TABLE, CORPUS)
    numeric = numeric_gradient(WEIGHT_TABLE, CORPUS)

    for word in VOCABULARY:
        for category_index in (0, 1):
            assert abs(formula[word][category_index]
                       - numeric[word][category_index]) < 1e-5


def test_gradient_points_uphill():
    """梯度是"往上坡走"的方向，所以反着走损失必须变小。"""
    reset_table()
    before = average_loss(CORPUS)
    grads = gradient(WEIGHT_TABLE, CORPUS)

    lr = 0.5
    for word in VOCABULARY:
        for category_index in (0, 1):
            WEIGHT_TABLE[word][category_index] -= lr * grads[word][category_index]

    assert average_loss(CORPUS) < before


def test_training_loss_goes_down_every_step():
    """200 步里，损失必须一步比一步小。"""
    reset_table()
    history = train(steps=100, lr=0.5)
    for earlier, later in zip(history, history[1:]):
        assert later < earlier, "损失出现了上升，说明更新方向或符号错了"


def test_training_reaches_full_accuracy():
    reset_table()
    history = train(steps=200, lr=0.5)
    assert accuracy(CORPUS) == 10
    assert history[-1] < 0.05, f"损失应该降到 0.05 以下，实际是 {history[-1]}"


def test_a_word_that_only_appears_in_one_class_gets_a_big_weight():
    """「发布」只在科技句里出现，它应该拿到一个很大的正权重。"""
    reset_table()
    train(steps=200, lr=0.5)
    assert WEIGHT_TABLE["发布"][0] > 0.8
    assert WEIGHT_TABLE["好吃"][0] < -0.8


def test_the_ambiguous_word_gets_a_small_weight():
    """「苹果」两类句子里都有，它的权重应该被拉平。

    它出现过 3 次在科技句、4 次在食品句，两边几乎抵消。
    """
    reset_table()
    train(steps=200, lr=0.5)
    apple = WEIGHT_TABLE["苹果"][0]
    publish = WEIGHT_TABLE["发布"][0]
    assert abs(apple) < abs(publish) / 2


def test_loss_is_smaller_when_we_push_a_good_weight_up():
    """单独把「发布」的科技权重调大，损失应该变小。"""
    reset_table()
    before = average_loss(CORPUS)
    WEIGHT_TABLE["发布"][0] = 0.1
    assert average_loss(CORPUS) < before


def test_every_sentence_is_classified_correctly():
    reset_table()
    train(steps=200, lr=0.5)
    for sentence, label in CORPUS:
        scores = score(split_sentence(sentence))
        answer = "科技" if scores[0] >= scores[1] else "食品"
        assert answer == label


def test_unknown_words_do_not_crash():
    """「苹果芯片很强」里的「强」不在词表里，它应该被换成一个占位符。"""
    words = split_sentence("苹果芯片很强")
    assert "<UNK>" in words
    assert "苹果" in words


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
