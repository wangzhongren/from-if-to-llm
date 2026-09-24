"""第 18 章的测试：测"归一化"这个新机制。

测三件事：
    1. 归一化把每一行变成均值 0、标准差 1，而且这一步只看这一行自己
    2. 同一个样本，换一批邻居，归一化的结果一模一样
    3. 归一化之后，8 层才训得动
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pytest

from toygrad import Tensor, cross_entropy, layer_norm, randn

import before
from after import AttentionStack, make_data


def test_layer_norm_makes_each_row_mean_zero_and_std_one():
    x = randn(4, 7, requires_grad=True, seed=1) * 3.0 + 5.0
    weight = Tensor(np.ones(7))
    bias = Tensor(np.zeros(7))
    out = layer_norm(x, weight, bias).data

    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=-1), 1.0, atol=1e-3)


def test_the_learnable_scale_is_applied_after_normalising():
    """归一化不是"把数变成 0 到 1 之间"，它后面还跟着一个可以学的缩放。"""
    x = randn(3, 5, requires_grad=True, seed=2)
    weight = Tensor(np.full(5, 2.0))
    bias = Tensor(np.full(5, 1.5))
    out = layer_norm(x, weight, bias).data

    assert np.allclose(out.std(axis=-1), 2.0, atol=1e-3)
    assert np.allclose(out.mean(axis=-1), 1.5, atol=1e-6)


def test_layer_norm_result_does_not_depend_on_the_batch():
    """把同一个样本放进不同的批次，它自己那一行必须一模一样。

    这就是它和"把整批数据一起归一化"最大的区别。
    """
    h = randn(6, 4, 8, seed=3).data
    weight = Tensor(np.ones(8))
    bias = Tensor(np.zeros(8))

    alone = layer_norm(Tensor(h[[0]]), weight, bias).data[0]
    with_two = layer_norm(Tensor(h[[0, 3]]), weight, bias).data[0]
    with_all = layer_norm(Tensor(h), weight, bias).data[0]

    assert np.allclose(alone, with_two, atol=1e-12)
    assert np.allclose(alone, with_all, atol=1e-12)


def test_the_dumb_batch_wide_normalisation_does_depend_on_the_batch():
    """反面教材：把整批数据放在一起归一化，结果就跟着邻居跑。"""

    def dumb(x, eps=1e-5):
        return (x - x.mean(axis=(0, 1), keepdims=True)) / (x.std(axis=(0, 1), keepdims=True) + eps)

    h = randn(6, 4, 8, seed=4).data
    alone = dumb(h[[0]])[0, 0]
    with_all = dumb(h)[0, 0]
    assert np.abs(alone - with_all).max() > 1e-3, "换一批邻居，结果应该变"


def test_every_layer_hands_itself_a_standardised_tensor():
    """本章的核心现象：归一化之后，每一层拿到手的东西都是均值 0、标准差 1。"""
    inputs, _ = make_data()
    model = AttentionStack(n_layers=8)
    _, records = model.blocks(inputs)

    handed_stds = [handed.data.std() for _, handed, _ in records[1:]]
    assert np.allclose(handed_stds, 1.0, atol=1e-3), f"每一层都该是 1：{handed_stds}"

    old = before.AttentionStack(n_layers=8)
    _, old_records = old.blocks(inputs)
    old_stds = [handed.data.std() for _, handed, _ in old_records[1:]]
    assert old_stds == sorted(old_stds), "旧写法是一层比一层大"
    assert old_stds[-1] > 5 * old_stds[0], f"而且要大很多：{old_stds}"


def test_normalising_is_what_makes_eight_layers_trainable():
    """同样 8 层、同样 60 步，加了归一化的那个学得动。

    旧写法会炸成 nan —— 这正是这个测试想看到的事，所以这里把 numpy
    "溢出/无效值"的警告关掉，免得刷屏。
    """
    inputs, targets = make_data()
    losses = {}
    for name, model in (("before", before.AttentionStack(n_layers=8)),
                        ("after", AttentionStack(n_layers=8))):
        for p in model.params():
            p.zero_grad()
        with np.errstate(all="ignore"):
            for _ in range(60):
                loss = cross_entropy(model.forward(inputs), targets)
                model.zero_grad()
                loss.backward()
                model.step(0.05)
        losses[name] = float(loss.data)

    assert losses["after"] < 1.0, f"有归一化应该已经在往下掉了：{losses['after']}"
    assert not np.isfinite(losses["before"]) or losses["before"] > 10, (
        f"无归一化应该在炸：{losses['before']}"
    )


def test_the_normalised_stack_really_restores_the_sentences():
    """跑到底：8 层 + 归一化，两句话一字不差。"""
    inputs, targets = make_data()
    model = AttentionStack(n_layers=8)
    for p in model.params():
        p.zero_grad()
    for _ in range(400):
        loss = cross_entropy(model.forward(inputs), targets)
        model.zero_grad()
        loss.backward()
        model.step(0.05)

    guess = model.forward(inputs).data.argmax(axis=-1)
    assert (guess == targets).all(), f"还原准确率 {(guess == targets).mean():.3f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
