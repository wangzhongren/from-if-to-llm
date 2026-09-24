"""第 7 章的测试。

这一章要说的话只有一句：**中间层加了，但它没用。**
所以测试也只有一个主题：加了中间层之后，结果必须和没加一模一样。
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 每一章都有一个 after.py，名字会撞车，先把上一章留下的清掉
for name in ("after", "before"):
    sys.modules.pop(name, None)

from after import (  # noqa: E402
    XOR_DATA,
    HIDDEN_SIZE,
    accuracy,
    backward,
    forward,
    init_params,
    loss_of,
    make_data,
    max_score_gap,
    predict,
    softmax,
    train,
)

LN2 = math.log(2)


def test_data_has_four_combinations():
    """两个 0/1 的组合一共 4 种，数据里必须一个不少。"""
    seen = {(x[0], x[1]) for x, _, _ in XOR_DATA}
    assert seen == {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}


def test_data_is_xor_shaped():
    """两个数一样就判 1，不一样就判 0。"""
    for x, label, _ in XOR_DATA:
        assert label == (1 if x[0] == x[1] else 0)


def test_forward_computes_a_middle_layer():
    """forward 必须先算出一层中间结果，再由它算分数。"""
    params = init_params(seed=0, hidden_size=HIDDEN_SIZE)
    h, scores = forward(params, [0.0, 1.0])
    assert len(h) == HIDDEN_SIZE
    assert len(scores) == 2


def test_hand_written_gradient_matches_numeric_gradient():
    """手推的链式法则，必须和数值法算出来的一样。

    数值法只用了 loss 这一个函数，它不知道我们是怎么推的。
    两边对上，才说明那一段下标代码没写错。
    """
    params = init_params(seed=1, hidden_size=3)
    data = make_data()
    grads = backward(params, data)

    h = 1e-6
    for key, indexes in (("w1", [(0, 0), (1, 1)]), ("w2", [(0, 0), (1, 2)])):
        for index in indexes:
            a, b = index
            original = params[key][a][b]
            params[key][a][b] = original + h
            plus = loss_of(params, data)
            params[key][a][b] = original - h
            minus = loss_of(params, data)
            params[key][a][b] = original
            numeric = (plus - minus) / (2 * h)
            assert abs(numeric - grads[key][a][b]) < 1e-6, f"{key}{index} 梯度不一致"


def test_training_gets_stuck_at_half():
    """训练到不能再训练，loss 也只能停在 log(2)。

    log(2) = 0.693147 就是「两个类别各猜一半」的分数。
    """
    params, history = train(steps=3000, lr=0.5, seed=0)
    assert abs(history[-1] - LN2) < 1e-5, f"loss 应该是 {LN2}，实际是 {history[-1]}"
    assert history[-1] < history[0], "训练还是应该有一点点动静"


def test_the_two_classes_get_the_same_score():
    """模型对两个类别给出的分数几乎完全一样 —— 它分不出来。"""
    params, _ = train(steps=3000, lr=0.5, seed=0)
    assert max_score_gap(params, make_data()) < 1e-4


def test_probability_is_always_one_half():
    """不管输入是什么，判成同类的概率都是 0.5。"""
    params, _ = train(steps=3000, lr=0.5, seed=0)
    for x, _ in make_data():
        _, scores = forward(params, x)
        p = softmax(scores)
        assert abs(p[1] - 0.5) < 1e-4


def test_widening_the_middle_layer_does_not_help():
    """中间层从 1 个加到 16 个，loss 一动不动。"""
    data = make_data()
    losses = []
    for size in (1, 2, 4, 8, 16):
        _, history = train(steps=3000, lr=0.5, seed=0, hidden_size=size)
        losses.append(history[-1])

    for value in losses:
        assert abs(value - LN2) < 1e-5, f"中间层加宽之后 loss 变了：{losses}"


def test_more_middle_units_do_not_buy_more_accuracy():
    """中间层再宽，也没法把这 4 个点分开。

    准确率最多只能是 3/4，而且模型实际上连 3/4 都拿不到 ——
    它给出的是两个一模一样的分数，等于弃权。
    """
    data = make_data()
    for size in (1, 4, 16):
        params, _ = train(steps=3000, lr=0.5, seed=0, hidden_size=size)
        assert accuracy(params, data) <= 0.75


def test_the_model_has_no_preference_at_all():
    """训练完之后两个分数变得一模一样。

    这时候 predict 返回 0 还是 1，取决于浮点数最后一位是谁大 ——
    那不是"判断"，那只是符号位。模型其实弃权了。
    """
    params, _ = train(steps=3000, lr=0.5, seed=2)
    for x, _ in make_data():
        _, scores = forward(params, x)
        assert abs(scores[0] - scores[1]) < 1e-9
        # predict 的返回值不稳定，这里只是确认它不会崩
        assert predict(params, x) in (0, 1)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
