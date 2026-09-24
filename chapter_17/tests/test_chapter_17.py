"""第 17 章的测试：测"残差连接"这个新机制。

测三件事：
    1. 那一行 `x = x + layer(x)` 真的把输入原样带到了下一层（哪怕这一层什么也没算出来）
    2. 没有那一行，输入就被整个替换掉了
    3. 深一点的模型，有残差的那个真的学得动
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pytest

from toygrad import Tensor, embedding, cross_entropy

import before
from after import AttentionStack, make_blank_pos, make_data


def test_residual_keeps_the_input_when_the_layer_outputs_nothing():
    """把 attention 的输出投影置零，这一层就什么也没算出来。

    有残差时，这一层的输出必须还是原来的 x —— 这就是那条"捷径"。
    """
    model = AttentionStack(n_layers=1)
    model.layers[0]["Wo"].data = np.zeros_like(model.layers[0]["Wo"].data)

    inputs, _ = make_data()
    logits = model.forward(inputs)

    x = embedding(model.p["tok"], inputs) + model.p["pos"]
    expected = x @ model.p["Wout"]
    assert np.allclose(logits.data, expected.data, atol=1e-10)


def test_without_residual_the_input_is_replaced_by_the_layer_output():
    """同一个实验，用第 16 章那种写法：输入被整个替换掉，什么都没剩下。"""
    model = before.AttentionStack(n_layers=1)
    model.layers[0]["Wo"].data = np.zeros_like(model.layers[0]["Wo"].data)

    inputs, _ = make_data()
    logits = model.forward(inputs)

    assert np.allclose(logits.data, 0.0, atol=1e-10), "没有残差时，输入应该被彻底丢掉"


def test_deep_stack_learns_faster_with_residual():
    """4 层，跑 80 步，看谁先学会把两句话背下来。"""
    inputs, targets = make_data()
    losses = {}
    for name, factory in (("before", lambda: before.AttentionStack(n_layers=4)),
                          ("after", lambda: AttentionStack(n_layers=4))):
        model = factory()
        for p in model.params():
            p.zero_grad()
        for _ in range(80):
            loss = cross_entropy(model.forward(inputs), targets)
            model.zero_grad()
            loss.backward()
            model.step(0.05)
        losses[name] = float(loss.data)

    assert losses["after"] < losses["before"] / 4, (
        f"残差没有帮上忙：before={losses['before']:.4f} after={losses['after']:.4f}"
    )
    assert losses["before"] > 1.0, "旧写法应该还卡在高处，这个实验才有意义"


def test_a_shallow_stack_is_already_fine_without_residual():
    """1 层的时候旧写法也是好的 —— 这个毛病是"堆起来"才出现的。"""
    inputs, targets = make_data()
    model = before.AttentionStack(n_layers=1)
    for p in model.params():
        p.zero_grad()
    for _ in range(150):
        loss = cross_entropy(model.forward(inputs), targets)
        model.zero_grad()
        loss.backward()
        model.step(0.05)
    assert float(loss.data) < 0.05, f"1 层就该学会，现在 loss = {loss.data}"


def test_the_residual_model_really_restores_the_two_sentences():
    """跑到底：4 层 + 残差，20 个空位全填对。"""
    inputs, targets = make_data()
    blank_pos = make_blank_pos()
    model = AttentionStack(n_layers=4)
    for p in model.params():
        p.zero_grad()
    for _ in range(400):
        loss = cross_entropy(model.forward(inputs), targets)
        model.zero_grad()
        loss.backward()
        model.step(0.05)

    guess = model.forward(inputs).data.argmax(axis=-1)
    hit = (guess[np.arange(len(blank_pos)), blank_pos]
           == targets[np.arange(len(blank_pos)), blank_pos])
    assert hit.all(), f"只填对了 {hit.sum()}/{len(hit)} 个空位"


def test_the_gradient_does_not_vanish_in_a_deep_residual_stack():
    """8 层：有残差时梯度能一路回到词向量，没有残差时回不去。"""
    inputs, targets = make_data()
    sizes = {}
    for name, factory in (("before", lambda: before.AttentionStack(n_layers=8)),
                          ("after", lambda: AttentionStack(n_layers=8))):
        model = factory()
        for p in model.params():
            p.zero_grad()
        loss = cross_entropy(model.forward(inputs), targets)
        loss.backward()
        sizes[name] = float(np.abs(model.p["tok"].grad).max())

    assert sizes["after"] > 1e-3, f"有残差时梯度不该消失：{sizes['after']:.2e}"
    assert sizes["after"] > 10 * sizes["before"], (
        f"残差应该让梯度更通畅：before={sizes['before']:.2e} after={sizes['after']:.2e}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
