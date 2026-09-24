"""第 27 章测试 —— 测我们和 GPT 之间到底差哪几件事。

跑法：

    ./.venv/bin/python -m pytest chapter_27/tests/ -q
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from after import (  # noqa: E402
    VOCAB_SIZE,
    GPTModel,
    sinusoidal_positions,
)

BLOCK = 16


def ours(n_layer=2, **overrides):
    """第 22 章那个模型：三个开关全关。"""
    options = dict(learned_pos=False, activation=F.relu, final_ln=False, pre_ln=True)
    options.update(overrides)
    torch.manual_seed(0)
    return GPTModel(VOCAB_SIZE, dim=32, n_layer=n_layer, n_head=4, block_size=BLOCK,
                    **options)


def gpt(n_layer=2):
    """换上 GPT 的三件事。"""
    torch.manual_seed(0)
    return GPTModel(VOCAB_SIZE, dim=32, n_layer=n_layer, n_head=4, block_size=BLOCK,
                    learned_pos=True, activation=F.gelu, final_ln=True, pre_ln=True)


# ---------------------------------------------------------------- 我们的模型

def test_我们的模型和第_22_章一样是_pre_ln():
    """归一化放在子层前面：把子层输入放大 10 倍，子层算出来的东西不变。"""
    model = ours()
    x = torch.randint(0, VOCAB_SIZE, (1, 8))
    mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    with torch.no_grad():
        hidden = model.tok_emb(x) + model.pos_table[:8]
        block = model.blocks[0]
        small = block.attention(block.ln1(hidden), mask)
        big = block.attention(block.ln1(hidden * 10), mask)
    assert torch.allclose(small, big, atol=1e-5)


def test_我们的模型不带最后那次归一化也不带投影偏置():
    model = ours()
    names = {name for name, _ in model.named_parameters()}
    assert "blocks.0.ln1.weight" in names
    assert "blocks.0.Wq.weight" in names
    assert "blocks.0.Wq.bias" not in names     # 第 14 章写的是 x @ W
    assert "pos_table" in dict(model.named_buffers())   # 第 16 章那张固定的表
    assert "pos_emb.weight" not in names       # 位置不是学出来的


def test_换成gpt的三件事以后结构确实变了():
    model = gpt()
    names = {name for name, _ in model.named_parameters()}
    assert "pos_emb.weight" in names           # 位置变成一张可学习的表
    assert model.final_ln                      # 输出层前多了一次归一化
    assert model.blocks[0].activation is F.gelu
    # 可学习的位置表比固定表多出 位置数 × 维度 个参数，
    # 最后那次归一化又多出 2 × 维度 个（weight 和 bias）
    assert gpt().n_params() - ours().n_params() == BLOCK * 32 + 2 * 32


def test_gelu和relu是两个不同的函数():
    x = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    gelu = F.gelu(x)
    relu = F.relu(x)
    assert abs(gelu[6].item() - 3.0) < 0.01        # 大正数上几乎一样
    assert abs(gelu[5].item() - 0.84) < 0.01       # 1.0 附近 GELU 低一点
    assert relu[5].item() == 1.0
    assert -0.02 < gelu[0] < 0                     # 负数上 GELU 不直接压成 0
    assert relu[0] == 0
    assert abs(gelu[3].item()) < 1e-6


def test_因果掩码在两种位置都挡住了未来():
    for model in (ours(), gpt()):
        x = torch.randint(0, VOCAB_SIZE, (2, BLOCK))
        with torch.no_grad():
            before = model(x)
            changed = x.clone()
            changed[:, 5] = (changed[:, 5] + 1) % VOCAB_SIZE
            after = model(changed)
        assert torch.allclose(before[:, :5], after[:, :5], atol=1e-6)
        assert not torch.allclose(before[:, 5], after[:, 5], atol=1e-6)


def test_固定的位置表不是参数():
    model = ours()
    table = dict(model.named_buffers())["pos_table"]
    assert np.allclose(table.numpy(), sinusoidal_positions(BLOCK, 32), atol=1e-6)
    assert not any(p is table for p in model.parameters())


# ---------------------------------------------------------------- 关键区别

def test_层数一多只有先归一化的写法还能把梯度传到底下():
    """这一章的核心结论，用两个 12 层模型训练 20 步量出来。

    注意：4 层的时候两种写法都没问题（这正是实验一"看不出差别"的原因）。
    """
    from after import TRAIN_DATA, VOCAB_SIZE as V, get_batch

    def bottom_gradient(pre_ln):
        torch.manual_seed(0)
        model = GPTModel(V, dim=64, n_layer=12, n_head=4, block_size=48, pre_ln=pre_ln,
                         learned_pos=False, activation=F.relu, final_ln=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        rng = np.random.default_rng(0)
        for _ in range(20):
            x, y = get_batch(TRAIN_DATA, 48, 32, rng)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model.tok_emb.weight.grad.norm().item()

    ours_value = bottom_gradient(pre_ln=True)
    post_ln = bottom_gradient(pre_ln=False)
    assert ours_value > 1e-3      # 梯度还活着
    assert post_ln < 1e-6         # 梯度已经塌了
    assert ours_value / post_ln > 1e4


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
