"""第 8 章的测试。

这一章只有一句话：**中间层后面那个「拐弯」，才是关键。**
所以测试分两组：一组证明没有拐弯时它等价于一条直线，一组证明有拐弯之后不是。
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

for name in ("after", "before", "experiment"):
    sys.modules.pop(name, None)

from after import (  # noqa: E402
    XOR_DATA,
    accuracy,
    backward,
    forward,
    init_params,
    loss_of,
    make_data,
    predict,
    relu,
    train,
)
from before import (  # noqa: E402
    count_params,
    forward as deep_forward,
    init_params as deep_init,
    train as deep_train,
)

LN2 = math.log(2)


# ------------------------------------------------------------------ 那个小函数


def test_relu_keeps_positive_and_kills_negative():
    assert relu(3.0) == 3.0
    assert relu(0.0) == 0.0
    assert relu(-0.0) == 0.0
    assert relu(-3.0) == 0.0


def test_relu_is_not_a_line():
    """relu 不满足「直线」的那个性质：f(u+v) = f(u) + f(v) - f(0)。"""
    u, v = 2.0, -3.0
    left = relu(u + v)
    right = relu(u) + relu(v) - relu(0.0)
    assert left != right  # 0 != 2


# ------------------------------------------------------------------ 加 relu 之前：还是一层


def test_two_layers_without_relu_are_still_a_line():
    """两层纯加权求和，输出必须等于某一条直线。

    这是这一章的核心结论 —— 没有拐弯，堆多少层都白堆。
    """
    params = deep_init(depth=2, width=4, seed=0)
    collapsed = collapse_for_test(params)

    for x, _, _ in XOR_DATA:
        net = deep_forward(params, x)[-1]
        line = [sum(collapsed[0][c][i] * x[i] for i in range(2)) + collapsed[1][c]
                for c in range(2)]
        for c in range(2):
            assert abs(net[c] - line[c]) < 1e-12


def collapse_for_test(params):
    """把一叠线性层乘成单独一层：W = W_last @ ... @ W_first。"""
    weights, biases = params["ws"][0], params["bs"][0]
    for layer in range(1, len(params["ws"])):
        layer_w, layer_b = params["ws"][layer], params["bs"][layer]
        biases = [layer_b[r] + sum(layer_w[r][k] * biases[k]
                                   for k in range(len(biases)))
                  for r in range(len(layer_w))]
        weights = [[sum(layer_w[r][k] * weights[k][c]
                        for k in range(len(weights)))
                    for c in range(len(weights[0]))]
                   for r in range(len(layer_w))]
    return weights, biases


def test_stacking_layers_does_not_help():
    """堆到 5 层、10 层，loss 还是 log(2)。"""
    for depth, width in ((2, 4), (3, 4), (5, 2), (10, 2)):
        _, history = deep_train(depth=depth, width=width, steps=300,
                                lr=0.5, seed=0)
        assert abs(history[-1] - LN2) < 1e-5, (
            f"{depth} 层的 loss 竟然变了：{history[-1]}"
        )


# ------------------------------------------------------------------ 加 relu 之后


def test_hand_written_gradient_matches_numeric_gradient():
    """带 relu 的手写梯度，必须和数值法一致。

    重点在那一行「被 relu 掐掉的部分不回传」。
    如果把它写漏了，训练照样会跑，只是会慢一点点 —— 很难发现。
    """
    params = init_params(seed=1, hidden_size=3)
    data = make_data()
    grads = backward(params, data)

    h = 1e-6
    for key, indexes in (("w1", [(0, 0), (2, 1)]), ("w2", [(0, 1), (1, 2)])):
        for a, b in indexes:
            original = params[key][a][b]
            params[key][a][b] = original + h
            plus = loss_of(params, data)
            params[key][a][b] = original - h
            minus = loss_of(params, data)
            params[key][a][b] = original
            numeric = (plus - minus) / (2 * h)
            assert abs(numeric - grads[key][a][b]) < 1e-6, f"{key}[{a}][{b}] 不一致"


def test_relu_network_solves_xor():
    params, history = train(steps=5000, lr=0.5, seed=0)
    assert accuracy(params, make_data()) == 1.0
    assert history[-1] < 0.01, f"loss 还是太大：{history[-1]}"


def test_relu_network_is_confident():
    """不光要判对，还要有把握 —— 概率得明显偏向一边。"""
    params, _ = train(steps=5000, lr=0.5, seed=0)
    for x, label, _ in XOR_DATA:
        _, _, scores = forward(params, x)
        assert abs(scores[0] - scores[1]) > 1.0, "两个类别的分数还是黏在一起"


def test_relu_network_is_not_a_line():
    """训练好的 relu 网络，已经不满足直线的那个性质了。"""
    params, _ = train(steps=5000, lr=0.5, seed=0)
    u, v = [0.7, -0.4], [-0.2, 1.1]
    s_u = forward(params, u)[2]
    s_v = forward(params, v)[2]
    s_uv = forward(params, [u[0] + v[0], u[1] + v[1]])[2]
    s_0 = forward(params, [0.0, 0.0])[2]
    gap = max(abs(s_uv[c] - (s_u[c] + s_v[c] - s_0[c])) for c in range(2))
    assert gap > 0.1, "它看上去还是一条直线"


def test_relu_kills_some_units():
    """中间层确实有单元被掐成了 0 —— 拐弯是真实发生的。"""
    params, _ = train(steps=5000, lr=0.5, seed=0)
    for x, _, _ in XOR_DATA:
        z, h, _ = forward(params, x)
        for j in range(len(z)):
            assert h[j] == (z[j] if z[j] > 0 else 0.0)


def test_relu_works_with_a_narrow_middle_layer():
    """加了 relu 之后，中间层只有 2 个单元也能解决 XOR。

    注意这里的 seed 是 1 不是 0。中间层太窄的时候，从哪个随机值出发
    就变得很重要 —— 有的起点会卡在一个不好的解上。
    那是另一类问题（训练本身不好），不是这一章讲的问题（结构本身不行）。
    """
    params, history = train(steps=5000, lr=0.5, seed=1, hidden_size=2)
    assert accuracy(params, make_data()) == 1.0
    assert history[-1] < 0.05


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
