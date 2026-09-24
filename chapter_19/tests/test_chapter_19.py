"""第 19 章的测试：测"Attention 只交换、MLP 才加工"这件事。

测三件事：
    1. attention 的输出永远精确等于"值的加权平均"，堆多少层都一样
    2. MLP 是逐个位置各算各的，而且它能算出"加权平均算不出来"的东西
    3. 探针任务上，只有 attention 学不会，加上 MLP 就学会了
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pytest

from toygrad import Tensor, randn

from after import Block, evaluate, train
from experiment import attention_layer, mlp


def test_attention_output_is_exactly_a_weighted_average():
    """输出 = 权重 @ 值，权重非负、每一行加起来是 1。没有别的事情发生。"""
    x = randn(1, 6, 8, seed=1)
    out, weights = attention_layer(x, seed=5)

    assert np.all(weights >= 0), "权重不能是负数"
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-12), "每一行的权重必须加起来等于 1"
    assert np.allclose(out.data[0], weights @ x.data[0], atol=1e-12), "输出必须精确等于这个加权平均"


def test_eight_attention_layers_are_still_one_weighted_average():
    """把 8 层的权重矩阵连乘起来，得到的仍然是"每一行和为 1、非负"的矩阵。

    也就是说：8 层下来，每个位置拿到的还是输入的一次加权平均。
    """
    x = randn(1, 6, 8, seed=2)
    origin = x.data[0]
    lo, hi = origin.min(axis=0), origin.max(axis=0)

    total = np.eye(6)
    cur = x
    for layer in range(8):
        out, weights = attention_layer(cur, seed=30 + layer)
        # 加权平均不可能跑出"被平均的那些数"的范围
        assert np.all(out.data[0] <= hi + 1e-9) and np.all(out.data[0] >= lo - 1e-9)
        total = weights @ total
        cur = out

    assert np.allclose(total.sum(axis=-1), 1.0, atol=1e-12)
    assert np.all(total >= 0)
    assert np.allclose(cur.data[0], total @ origin, atol=1e-12), "8 层 = 一次加权平均"


def test_the_mlp_can_produce_values_that_averaging_can_never_reach():
    """同一堆输入：加权平均跑不出范围，MLP 可以。"""
    x = randn(6, 8, seed=3).data
    lo, hi = x.min(axis=0), x.max(axis=0)

    averaged = x.mean(axis=0)                 # 随便一种平均
    assert np.all(averaged <= hi) and np.all(averaged >= lo)

    out = mlp(Tensor(x), seed=41).data
    over = max(float(np.max(np.maximum(0, out - hi))),
               float(np.max(np.maximum(0, lo - out))))
    assert over > 0.1, f"MLP 应该能跑出范围，现在越界量只有 {over}"


def test_mlp_only_touches_its_own_position_while_attention_reaches_across():
    """改动第 0 个位置的输入，看看别的 position 有没有跟着变。

    MLP 是逐个位置各算各的：别的 position 一点都不动。
    attention 专门干这个：它会把改动传到其它 position 上去。
    """
    x = randn(2, 5, 8, seed=4).data
    changed = x.copy()
    changed[:, 0, 0] += 2.0        # 只改一个位置上的一个数

    mlp_only = Block(n_blocks=2, use_attention=False, use_mlp=True, d_model=8, n_heads=2)
    assert np.allclose(mlp_only.hidden(changed).data[:, 3], mlp_only.hidden(x).data[:, 3], atol=1e-10)

    attn = Block(n_blocks=2, use_attention=True, use_mlp=False, d_model=8, n_heads=2)
    assert not np.allclose(attn.hidden(changed).data[:, 3], attn.hidden(x).data[:, 3], atol=1e-6)


def test_attention_alone_cannot_learn_the_xor_probe():
    """只给 attention：900 步也学不会"自己两个数异或"，准确率趴在 50%。"""
    model = Block(use_attention=True, use_mlp=False)
    train(model, steps=400, lr=0.05, verbose=False)
    loss, acc = evaluate(model)
    assert 0.45 < acc < 0.60, f"应该停在瞎猜附近，现在是 {acc:.3f}"
    assert loss > 0.6, f"loss 应该停在 ln2 附近，现在是 {loss:.4f}"


def test_adding_the_mlp_makes_the_same_probe_learnable():
    """同一个探针、同样的步数，加上 MLP 就学得动了。"""
    model = Block(use_attention=True, use_mlp=True)
    train(model, steps=600, lr=0.05, verbose=False)
    loss, acc = evaluate(model)
    assert acc > 0.80, f"应该明显超过瞎猜，现在是 {acc:.3f}（loss={loss:.4f}）"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
