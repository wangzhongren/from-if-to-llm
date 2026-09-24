"""第 9 章的测试。

这一章唯一的新东西是"机器自己算梯度"。
所以测试都在问同一件事：

    它自己算出来的梯度，和搬着手指头数出来的，是不是同一个数？
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

for name in ("after", "before"):
    sys.modules.pop(name, None)

from after import (  # noqa: E402
    Value,
    accuracy,
    all_params,
    batch_loss,
    make_data,
    make_params,
    softmax_loss,
    train,
)
from toygrad import SGD, Tensor, cross_entropy, no_grad, randn, zeros  # noqa: E402


# ------------------------------------------------------------------ 标量版引擎


def test_a_tiny_graph_gets_the_right_gradient():
    """a * b + c，手推一遍，和 backward() 对一对。

        a = 3, b = 4, c = 5   ->   a*b + c = 17
        d/da = b = 4，d/db = a = 3，d/dc = 1
    """
    a, b, c = Value(3.0), Value(4.0), Value(5.0)
    out = a * b + c
    out.backward()

    assert out.data == 17.0
    assert a.grad == 4.0
    assert b.grad == 3.0
    assert c.grad == 1.0


def test_a_value_used_twice_gets_both_paths():
    """x * x 的梯度是 2x —— 两条路径上的梯度加起来了。"""
    x = Value(3.0)
    (x * x).backward()
    assert x.grad == 6.0


def test_backward_accumulates_instead_of_overwriting():
    """连续 backward 两次，梯度翻倍。这正是必须 zero_grad 的原因。"""
    x = Value(3.0)
    y = x * x
    y.backward()
    assert x.grad == 6.0
    y = x * x
    y.backward()
    assert x.grad == 12.0


def test_relu_kills_the_gradient_of_negative_values():
    positive = Value(2.0)
    positive.relu().backward()
    assert positive.grad == 1.0

    negative = Value(-2.0)
    negative.relu().backward()
    assert negative.grad == 0.0


def test_relu_output_of_negative_input_is_zero():
    assert Value(-2.0).relu().data == 0.0
    assert Value(2.0).relu().data == 2.0


def test_backward_goes_through_a_long_chain():
    """链子长一点也要对：((x * 2 + 1) * 3) 对 x 的梯度是 6。"""
    x = Value(1.0)
    out = (x * 2.0 + 1.0) * 3.0
    out.backward()
    assert out.data == 9.0
    assert x.grad == 6.0


def test_scalar_engine_matches_numeric_gradient():
    """标量引擎算出来的梯度，必须和数值法一致。

    数值法只看输入输出，完全不知道中间发生了什么。
    """

    def loss_from(a_data, b_data):
        a = Value(a_data)
        b = Value(b_data)
        return (a * b + a).relu() + b * b + (a - b).exp() * 0.1

    a_data, b_data = 0.7, -0.4
    a, b = Value(a_data), Value(b_data)
    out = ((a * b + a).relu() + b * b + (a - b).exp() * 0.1)
    out.backward()

    h = 1e-6
    numeric_a = (loss_from(a_data + h, b_data).data
                 - loss_from(a_data - h, b_data).data) / (2 * h)
    numeric_b = (loss_from(a_data, b_data + h).data
                 - loss_from(a_data, b_data - h).data) / (2 * h)

    assert abs(a.grad - numeric_a) < 1e-6
    assert abs(b.grad - numeric_b) < 1e-6


def test_the_scalar_engine_solves_xor():
    """没有一行梯度代码，三层网络照样把 XOR 训出来。"""
    params, history = train(steps=300, lr=0.5, seed=0)
    assert accuracy(params, make_data()) == 1.0
    assert history[-1] < 0.01
    assert history[0] > history[-1]


def test_loss_is_a_value_that_knows_where_it_came_from():
    """损失自己也是一个 Value，它记得整条来路。"""
    params = make_params(seed=0)
    loss = batch_loss(params, make_data())
    assert isinstance(loss, Value)
    before = len(all_params(params))
    assert before == 42
    for p in all_params(params):
        assert p.grad == 0.0


# ------------------------------------------------------------------ 数组版引擎（toygrad）


def test_toygrad_matches_numeric_gradient():
    """数组版引擎：22 个参数逐个和数值法对一遍。"""
    x_matrix = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = np.array([1, 0, 0, 1])

    params = [
        randn(2, 4, scale=0.5, requires_grad=True, seed=0),
        randn(4, scale=0.1, requires_grad=True, seed=1),
        randn(4, 2, scale=0.5, requires_grad=True, seed=2),
        randn(2, scale=0.1, requires_grad=True, seed=3),
    ]

    def loss():
        hidden = (Tensor(x_matrix) @ params[0] + params[1]).relu()
        return cross_entropy(hidden @ params[2] + params[3], targets)

    for p in params:
        p.zero_grad()
    loss().backward()

    h = 1e-6
    for p in params:
        flat = p.data.reshape(-1).copy()
        for i in range(flat.size):
            original = flat[i]

            flat[i] = original + h
            p.data = flat.reshape(p.data.shape)
            plus = loss().data

            flat[i] = original - h
            p.data = flat.reshape(p.data.shape)
            minus = loss().data

            flat[i] = original
            p.data = flat.reshape(p.data.shape)

            numeric = (plus - minus) / (2 * h)
            analytic = p.grad.reshape(-1)[i]
            assert abs(analytic - numeric) < 1e-5, (
                f"第 {i} 个参数对不上：{analytic} vs {numeric}"
            )
        p.data = flat.reshape(p.data.shape)


def test_no_grad_builds_no_graph():
    a = randn(3, requires_grad=True, seed=0)
    with no_grad():
        inside = a * 2.0
    assert inside.requires_grad is False
    assert inside._prev == ()

    outside = a * 2.0
    assert outside.requires_grad is True
    assert len(outside._prev) == 2


def test_toygrad_backward_also_accumulates():
    a = randn(4, requires_grad=True, seed=0)
    (a * a).sum().backward()
    first = a.grad.copy()
    assert np.allclose(first, 2 * a.data)

    (a * a).sum().backward()
    assert np.allclose(a.grad, 2 * first), "第二次 backward 应该累加，而不是覆盖"

    a.zero_grad()
    assert np.allclose(a.grad, 0.0)


def test_a_real_training_step_with_toygrad():
    """用玩具引擎把 XOR 训出来，确认它不只是"梯度对"，而是真的能学。"""
    x_matrix = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = np.array([1, 0, 0, 1])

    w1 = randn(2, 4, scale=0.5, requires_grad=True, seed=0)
    b1 = zeros(4, requires_grad=True)
    w2 = randn(4, 2, scale=0.5, requires_grad=True, seed=1)
    b2 = zeros(2, requires_grad=True)
    optimizer = SGD([w1, b1, w2, b2], lr=0.5)

    history = []
    for _ in range(300):
        optimizer.zero_grad()
        hidden = (Tensor(x_matrix) @ w1 + b1).relu()
        loss = cross_entropy(hidden @ w2 + b2, targets)
        history.append(loss.item())
        loss.backward()
        optimizer.step()

    assert history[-1] < history[0] / 10
    logits = ((Tensor(x_matrix) @ w1 + b1).relu() @ w2 + b2).data
    guess = logits.argmax(axis=1)
    assert (guess == targets).all()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
