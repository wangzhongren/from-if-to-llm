"""对 toygrad 引擎做数值梯度校验。

思路和第 6 章完全一样：把某个参数挪动一丁点，看 loss 怎么变。

        (loss(w+h) - loss(w-h)) / (2h)   ≈   backward() 算出来的梯度

两者对不上，说明引擎里某个算子的 _backward 写错了。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from toygrad import Tensor, cross_entropy, embedding, layer_norm, randn  # noqa: E402
from toygrad.tensor import no_grad  # noqa: E402

H = 1e-6
TOL = 1e-4


def numeric_grad(f, x):
    """中心差分求数值梯度。

    f 是不带参数的闭包，它读的是 x 当前的值。
    我们一个一个地微调 x 里的数字，看 f() 的输出怎么变。
    """
    shape = x.data.shape
    orig = x.data.copy()
    grads = np.zeros(orig.size)
    for i in range(orig.size):
        for sign, key in ((+H, "p"), (-H, "m")):
            perturbed = orig.copy()
            perturbed.reshape(-1)[i] += sign
            x.data = perturbed
            if key == "p":
                plus = f().data
            else:
                minus = f().data
        grads[i] = (plus - minus) / (2 * H)
    x.data = orig
    return grads.reshape(shape)


def check(f, params, tol=TOL):
    """f 返回一个标量 Tensor；对 params 里每个张量逐个比对解析/数值梯度。"""
    for p in params:
        p.zero_grad()  # 同一个测试里可能 check 好几轮，不清理会累加
    loss = f()
    loss.backward()
    for i, p in enumerate(params):
        ng = numeric_grad(f, p)
        assert np.allclose(p.grad, ng, atol=tol, rtol=1e-3), (
            f"参数 {i} 梯度不一致\n解析梯度:\n{p.grad}\n数值梯度:\n{ng}"
        )


# ------------------------------------------------------------------ 逐个算子


def test_add_broadcast():
    a = randn(3, 4, requires_grad=True, seed=0)
    b = randn(4, requires_grad=True, seed=1)
    check(lambda: (a + b).sum(), [a, b])


def test_mul_and_sub():
    a = randn(2, 3, requires_grad=True, seed=2)
    b = randn(3, requires_grad=True, seed=3)
    check(lambda: (a * b - a).sum() + (b * b).sum(), [a, b])


def test_div_and_pow():
    a = randn(4, requires_grad=True, seed=4)
    check(lambda: (a ** 2).sum() + (1.0 / (a * a + 2.0)).sum(), [a])


def test_matmul_2d():
    a = randn(3, 4, requires_grad=True, seed=5)
    b = randn(4, 5, requires_grad=True, seed=6)
    check(lambda: (a @ b).sum(), [a, b])


def test_matmul_batched_broadcast():
    a = randn(2, 3, 4, requires_grad=True, seed=7)
    b = randn(4, 5, requires_grad=True, seed=8)
    check(lambda: (a @ b).sum(), [a, b])


def test_exp_log_relu_tanh_gelu():
    a = randn(5, requires_grad=True, seed=9)
    check(lambda: a.exp().sum(), [a])
    b = Tensor(np.abs(np.asarray(randn(5, seed=10).data)) + 0.5, requires_grad=True)
    check(lambda: b.log().sum(), [b])
    c = randn(5, requires_grad=True, seed=11)
    check(lambda: c.relu().sum(), [c])
    d = randn(5, requires_grad=True, seed=12)
    check(lambda: d.tanh().sum(), [d])
    e = randn(5, requires_grad=True, seed=13)
    check(lambda: e.gelu().sum(), [e])


def test_reshape_transpose_getitem():
    a = randn(3, 4, requires_grad=True, seed=14)
    check(lambda: a.transpose(1, 0).reshape(12).sum(), [a])
    check(lambda: a[1:3, ::2].sum(), [a])
    check(lambda: (a * a)[0, 1], [a])


def test_reductions():
    a = randn(2, 3, 4, requires_grad=True, seed=15)
    check(lambda: a.sum(), [a])
    check(lambda: a.sum(axis=1).sum(), [a])
    check(lambda: a.sum(axis=1, keepdims=True).sum(), [a])
    check(lambda: a.mean(), [a])
    check(lambda: a.mean(axis=-1).sum(), [a])
    check(lambda: a.mean(axis=(0, 2)).sum(), [a])


def test_softmax():
    a = randn(3, 5, requires_grad=True, seed=16)
    check(lambda: (a.softmax() * a.softmax()).sum(), [a])
    check(lambda: a.softmax(axis=0).sum(), [a])


def test_masked_fill():
    a = randn(3, 3, requires_grad=True, seed=17)
    mask = np.triu(np.ones((3, 3), dtype=bool), k=1)
    check(lambda: a.masked_fill(mask, -1e9).softmax().sum(), [a])


def test_cross_entropy():
    logits = randn(4, 6, requires_grad=True, seed=18)
    targets = np.array([1, 3, 0, 5])
    check(lambda: cross_entropy(logits, targets), [logits])


def test_cross_entropy_3d():
    logits = randn(2, 3, 6, requires_grad=True, seed=19)
    targets = np.arange(6).reshape(2, 3) % 6
    check(lambda: cross_entropy(logits, targets), [logits])


def test_layer_norm():
    x = randn(2, 4, 6, requires_grad=True, seed=20)
    w = Tensor(np.ones(6), requires_grad=True)
    b = Tensor(np.zeros(6), requires_grad=True)
    check(lambda: (layer_norm(x, w, b) ** 2).sum(), [x, w, b])


def test_embedding():
    table = randn(7, 3, requires_grad=True, seed=21)
    idx = np.array([[0, 2], [2, 5]])
    check(lambda: (embedding(table, idx) ** 2).sum(), [table])


def test_embedding_repeated_index():
    """同一个词被查了两次，梯度必须累加而不是覆盖。"""
    table = randn(5, 2, requires_grad=True, seed=22)
    idx = np.array([1, 1, 1, 3])
    check(lambda: (embedding(table, idx) ** 2).sum(), [table])


# ------------------------------------------------------------------ 综合


def test_a_tiny_training_step_actually_learns():
    """把引擎串起来跑一次真正的训练，确认 loss 真的会下降。"""
    rng = np.random.default_rng(23)
    X = rng.standard_normal((64, 3))
    true_w = np.array([2.0, -3.0, 0.5])
    y = X @ true_w + 1.5

    w = Tensor(np.zeros(3), requires_grad=True)
    b = Tensor(0.0, requires_grad=True)

    first = last = None
    for step in range(200):
        pred = Tensor(X) @ w + b
        loss = ((pred - Tensor(y)) ** 2).mean()
        if step == 0:
            first = loss.data
        last = loss.data
        w.zero_grad()
        b.zero_grad()
        loss.backward()
        w.data -= 0.1 * w.grad
        b.data -= 0.1 * b.grad

    assert last < first * 1e-3, f"loss 没降下来：{first} -> {last}"
    assert np.allclose(w.data, true_w, atol=1e-2)
    assert abs(b.data - 1.5) < 1e-2


def test_no_grad_does_not_build_graph():
    a = randn(3, requires_grad=True, seed=24)
    with no_grad():
        out = a * 2.0
    assert out.requires_grad is False
    assert out._prev == ()
    out2 = a * 2.0
    assert out2.requires_grad is True


def test_backward_accumulates_over_branches():
    """一个张量被用了两次，它的梯度必须是两条路径之和。"""
    a = randn(4, requires_grad=True, seed=25)
    (a * a).sum().backward()
    assert np.allclose(a.grad, 2 * a.data)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
