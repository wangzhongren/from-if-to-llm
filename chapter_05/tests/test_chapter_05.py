"""
第 5 章的测试：测的是本章真正的东西——
分数怎么变成概率（softmax），概率怎么变成"错误有多大"（交叉熵损失）。
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 别的章节也可能有同名的 after.py，先清掉缓存，确保导入的是本章的
sys.modules.pop("after", None)

from after import (
    ADJUST_AMOUNT,
    CATEGORIES,
    CORPUS,
    VOCABULARY,
    WEIGHT_TABLE,
    accuracy,
    adjust_weights_once,
    average_loss,
    loss_of,
    probability_of,
    softmax,
    split_sentence,
)


@pytest.fixture(autouse=True)
def restore_weight_table():
    """权重表是全局的，每个测试跑完都恢复原样，免得互相影响。"""
    saved = {word: list(weights) for word, weights in WEIGHT_TABLE.items()}
    yield
    for word, weights in saved.items():
        WEIGHT_TABLE[word][:] = weights


# ---------- 概率这一半 ----------

def test_softmax_outputs_sum_to_one():
    for scores in [[1.7, 1.2], [0.0, 0.0], [-3.0, 5.0], [8.0, 0.0]]:
        assert sum(softmax(scores)) == pytest.approx(1.0)


def test_softmax_keeps_the_order():
    """分数大的，概率也必须大——softmax 不能把名次弄乱。"""
    scores = [1.7, 1.2]
    probabilities = softmax(scores)
    assert probabilities[0] > probabilities[1]

    scores = [-0.1, 0.1]
    probabilities = softmax(scores)
    assert probabilities[1] > probabilities[0]


def test_equal_scores_give_equal_probabilities():
    """分数一样时，两个概率各占一半——这就是"完全不知道"的样子。"""
    assert softmax([0.0, 0.0]) == pytest.approx([0.5, 0.5])
    assert softmax([7.0, 7.0]) == pytest.approx([0.5, 0.5])


def test_softmax_does_not_overflow():
    """分数很大时也不能炸掉（先减掉最大值就是为了这个）。"""
    probabilities = softmax([1000.0, 0.0])
    assert probabilities[0] == pytest.approx(1.0)
    assert probabilities[1] == pytest.approx(0.0)


def test_two_class_softmax_matches_the_one_number_formula():
    """两类的时候，softmax 和 1 / (1 + e 的 -d 次方) 是一回事。"""
    for difference in [-2.0, -0.5, 0.0, 1.0, 3.0]:
        assert softmax([difference, 0.0])[0] == pytest.approx(
            1.0 / (1.0 + math.exp(-difference))
        )


def test_probability_of_the_correct_category_is_one_minus_the_other():
    words = split_sentence("苹果很甜")
    first = probability_of(words, 0)
    second = probability_of(words, 1)
    assert first + second == pytest.approx(1.0)


# ---------- 损失这一半 ----------

def test_the_starting_table_is_completely_unsure():
    """全 0 的表：每句话的概率都是 0.5，损失都是 log(2)。"""
    assert average_loss(CORPUS) == pytest.approx(math.log(2), abs=1e-9)


def test_a_wrong_answer_costs_more_than_a_right_one():
    """答案错了，损失必须比对了大——这是"损失"这两个字的全部意义。"""
    # 先让"甜"给食品加分，这样「苹果很甜」的答案明显偏食品
    WEIGHT_TABLE["甜"] = [0.0, 2.0]
    words = split_sentence("苹果很甜")
    assert loss_of(words, "食品") < loss_of(words, "科技")


def test_the_bigger_the_error_the_bigger_the_loss():
    """错得越离谱，损失越大——这是第 4 章的尺子做不到的事。"""
    scores_small_miss = softmax([-0.1, 0.1])          # 差一点点
    scores_big_miss = softmax([-19.0, 19.0])          # 差得离谱
    assert -math.log(scores_small_miss[0]) < -math.log(scores_big_miss[0])
    assert -math.log(scores_big_miss[0]) > 30


def test_a_confident_correct_answer_costs_almost_nothing():
    probabilities = softmax([8.0, 0.0])
    assert -math.log(probabilities[0]) < 0.001


def test_loss_is_zero_when_the_probability_is_one():
    assert -math.log(1.0) == 0.0


# ---------- 拿损失当尺子，真的能训练 ----------

def test_the_loss_ruler_can_train_the_weights():
    """从全 0 出发，只靠"改动能不能让损失变小"，一路训到全对、损失接近 0。"""
    before = average_loss(CORPUS)
    for _ in range(40):
        _, changed = adjust_weights_once(CORPUS, amount=ADJUST_AMOUNT)
        if changed == 0:
            break
    after = average_loss(CORPUS)
    assert before == pytest.approx(math.log(2), abs=1e-9)
    assert after < 1e-6
    assert accuracy(CORPUS) == 10


def test_the_loss_ruler_fixes_the_shakiest_sentence():
    """
    「苹果芯片很强」是这份语料里最勉强的一句。
    训练完之后，它也被推到很有把握——损失这把尺子不会放过它。
    """
    word = "苹果芯片很强"
    category_index = CATEGORIES.index(dict(CORPUS)[word])
    assert probability_of(split_sentence(word), category_index) == pytest.approx(0.5)

    for _ in range(40):
        _, changed = adjust_weights_once(CORPUS, amount=ADJUST_AMOUNT)
        if changed == 0:
            break

    assert probability_of(split_sentence(word), category_index) > 0.99


def test_weights_stay_the_same_size_for_untouched_words():
    """
    尺子是"一次只动一个参数"。某个参数改不动的时候（比如'电脑'在这份语料里
    怎么改都不影响损失），它就该原样待着。
    """
    adjust_weights_once(CORPUS, amount=ADJUST_AMOUNT)
    assert len(WEIGHT_TABLE) == len(VOCABULARY)
    for weights in WEIGHT_TABLE.values():
        assert len(weights) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
