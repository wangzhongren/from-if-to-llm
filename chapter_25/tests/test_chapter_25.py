"""第 25 章测试 —— 测这一章真正的新东西：PyTorch 的封装和我们自己写的东西等价。

跑法：

    ./.venv/bin/python -m pytest chapter_25/tests/ -q
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))          # 本章的 before.py / after.py
sys.path.insert(0, os.path.join(HERE, "..", ".."))    # toygrad

from after import TorchModel, load_our_weights, sinusoidal_positions  # noqa: E402
from before import OurModel, VOCAB_SIZE  # noqa: E402
from toygrad import Tensor, layer_norm  # noqa: E402

SIZE = dict(dim=32, n_layer=2, n_head=4, block_size=16)


def make_pair(seed=0):
    """建一对模型：我们的引擎一个、PyTorch 一个，权重完全相同。"""
    ours = OurModel(VOCAB_SIZE, seed=seed, **SIZE)
    torch.manual_seed(seed)
    theirs = TorchModel(VOCAB_SIZE, **SIZE)
    load_our_weights(theirs, ours)
    return ours, theirs


def test_参数量的算法两边一致():
    ours, theirs = make_pair()
    assert ours.n_params() == theirs.n_params()


def test_参数表和第_22_章逐项对齐():
    """第 22 章的模型有两件事容易写错，这里钉死：

      * 四个投影没有偏置（第 14 章写的是 x @ W）
      * 输出层前面没有最后那次 LayerNorm

    将来谁改坏了，这个测试会立刻红。
    """
    ours, theirs = make_pair()
    names = {name for name, _ in theirs.named_parameters()}
    assert "blocks.0.ln1.weight" in names          # 归一化在子层前面用
    assert "blocks.0.ln2.weight" in names
    assert "blocks.0.Wq.weight" in names
    assert "blocks.0.Wq.bias" not in names         # 投影不带偏置
    assert "ln_f.weight" not in names              # 没有最后那次归一化
    assert "ln_f_w" not in ours.params
    assert "block0.qb" not in ours.params
    assert "head_w" in ours.params                 # 输出层本身是带偏置的


def test_归一化放在子层前面():
    """第 22 章的写法：先 LayerNorm，再进子层。

    把子层的输入整体放大 10 倍，子层算出来的东西应该一点不变 ——
    因为 LayerNorm 会先把尺度按回去。如果哪天把归一化挪到残差外面，
    这个测试就会红。
    """
    _, model = make_pair()
    x = torch.randint(0, VOCAB_SIZE, (1, 8))
    mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    with torch.no_grad():
        hidden = model.tok_emb(x) + model.pos_table[:8]
        block = model.blocks[0]
        small = block.attention(block.ln1(hidden), mask)
        big = block.attention(block.ln1(hidden * 10), mask)
    assert torch.allclose(small, big, atol=1e-5)


def test_同一份权重下两个引擎的前向输出一致():
    ours, theirs = make_pair()
    rng = np.random.default_rng(0)
    x = rng.integers(0, VOCAB_SIZE, size=(2, 16))
    with torch.no_grad():
        logits_torch = theirs(torch.from_numpy(x)).numpy()
    logits_ours = ours.forward(x).data
    assert np.abs(logits_torch - logits_ours).max() < 1e-5


def test_linear_就是矩阵乘法加偏置():
    torch.manual_seed(0)
    layer = nn.Linear(6, 4)
    x = torch.randn(3, 5, 6)
    with torch.no_grad():
        expected = x @ layer.weight.T + layer.bias
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_embedding_就是查表():
    torch.manual_seed(0)
    table = nn.Embedding(10, 4)
    idx = torch.tensor([[1, 3], [5, 7]])
    with torch.no_grad():
        out = table(idx)
    assert out.shape == (2, 2, 4)
    assert np.allclose(out.numpy(), table.weight.detach().numpy()[idx.numpy()])


def test_layernorm_和_toygrad_算的一样():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 8))
    weight = rng.standard_normal(8)
    bias = rng.standard_normal(8)

    ours = layer_norm(Tensor(x), Tensor(weight), Tensor(bias)).data
    theirs = F.layer_norm(
        torch.tensor(x, dtype=torch.float32), (8,),
        torch.tensor(weight, dtype=torch.float32),
        torch.tensor(bias, dtype=torch.float32),
    )
    assert np.abs(ours - theirs.numpy()).max() < 1e-5


def test_因果掩码挡住未来():
    _, model = make_pair()
    x = torch.randint(0, VOCAB_SIZE, (2, 8))
    with torch.no_grad():
        before = model(x)
        changed = x.clone()
        changed[:, 5] = (changed[:, 5] + 1) % VOCAB_SIZE
        after = model(changed)
    # 改了第 5 个字，第 0-4 个位置的分数不许变
    assert torch.allclose(before[:, :5], after[:, :5], atol=1e-6)
    # 第 5 个位置自己当然会变
    assert not torch.allclose(before[:, 5], after[:, 5], atol=1e-6)


def test_位置编码表是固定的():
    """第 16 章的固定位置编码：PyTorch 没有替我们实现，得自己写、自己注册成 buffer。"""
    _, model = make_pair()
    names = dict(model.named_buffers())
    assert "pos_table" in names
    assert np.allclose(names["pos_table"].numpy(),
                       sinusoidal_positions(SIZE["block_size"], SIZE["dim"]), atol=1e-6)
    # buffer 不是参数，不会被优化器更新
    assert not any(p is names["pos_table"] for p in model.parameters())


def test_封装好的多头注意力和手写版一致():
    ours, _ = make_pair()
    dim, n_head = SIZE["dim"], SIZE["n_head"]
    # 我们的投影没有偏置（第 14 章写的是 x @ W），所以这里 bias=False
    packed = nn.MultiheadAttention(dim, n_head, batch_first=True, bias=False)

    def to_tensor(w):
        return torch.tensor(np.asarray(w), dtype=torch.float32)

    with torch.no_grad():
        packed.in_proj_weight.copy_(torch.cat([
            to_tensor(ours.params["block0.Wq"].data.T),
            to_tensor(ours.params["block0.Wk"].data.T),
            to_tensor(ours.params["block0.Wv"].data.T),
        ], dim=0))
        packed.out_proj.weight.copy_(to_tensor(ours.params["block0.Wo"].data.T))

    length = 6
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, length, dim))
    mask = np.triu(np.ones((length, length), dtype=bool), k=1)
    x_ours = Tensor(x)
    ours_out = ours.attention(x_ours, 0, mask).data
    with torch.no_grad():
        theirs_out = packed(torch.tensor(x, dtype=torch.float32).clone(),
                            torch.tensor(x, dtype=torch.float32).clone(),
                            torch.tensor(x, dtype=torch.float32).clone(),
                            attn_mask=torch.tensor(mask), need_weights=False)[0].numpy()
    assert np.abs(ours_out - theirs_out).max() < 1e-5


def test_训练几步之后损失确实下降():
    from after import DATA, get_batch, train
    torch.manual_seed(0)
    model = TorchModel(VOCAB_SIZE, **SIZE)
    rng = np.random.default_rng(0)
    x, y = get_batch(DATA, block_size=16, batch_size=8, rng=rng)
    with torch.no_grad():
        first = F.cross_entropy(model(x).reshape(-1, VOCAB_SIZE), y.reshape(-1)).item()
    losses, _ = train(model, steps=40, block_size=16, batch_size=8, lr=3e-3)
    assert losses[-1] < first


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
