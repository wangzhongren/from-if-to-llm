"""第 18 章 experiment.py —— 数值是怎么乱起来的，怎么才能不乱。

三件事：

    一、把每一层"拿到手的那个张量"的均值和标准差打出来。
        无归一化：标准差 1.4 → 15.0。有归一化：每一层都是 0 / 1。

    二、训练 8 层。无归一化第 10 步就 nan；有归一化能一路学到 0.00x。

    三、为什么不能用"把整批数据一起归一化"这个最笨的做法：
        同一个句子，换一批邻居，输出就变了；一批只有一个样本时，
        那个"整批的均值"就是它自己。LayerNorm 没有这个问题。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import Tensor, cross_entropy, embedding, layer_norm, randn

from before import BLANK, L, N_LAYERS, PAD, SENTENCES, V, VOCAB, make_data

STEPS = 400
LR = 0.05


class Stack:
    """一份代码，两种模式：norm=False 是第 17 章的写法，norm=True 是本章的写法。"""

    def __init__(self, n_layers, norm, d_model=32, n_heads=2, seed=0):
        self.n_layers = n_layers
        self.norm = norm
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.p = {}
        s = 1.0 / np.sqrt(d_model)
        self.p["tok"] = randn(V, d_model, scale=1.0, requires_grad=True, seed=seed + 1)
        self.p["pos"] = randn(L, d_model, scale=1.0, requires_grad=True, seed=seed + 2)
        self.layers = []
        for i in range(n_layers):
            base = seed + 100 * (i + 1)
            self.layers.append({
                "Wq": randn(d_model, d_model, scale=s, requires_grad=True, seed=base + 1),
                "Wk": randn(d_model, d_model, scale=s, requires_grad=True, seed=base + 2),
                "Wv": randn(d_model, d_model, scale=s, requires_grad=True, seed=base + 3),
                "Wo": randn(d_model, d_model, scale=s, requires_grad=True, seed=base + 4),
            })
        self.p["Wout"] = randn(d_model, V, scale=s, requires_grad=True, seed=seed + 999)
        for i in range(n_layers):
            self.p["ln%d_w" % i] = Tensor(np.ones(d_model), requires_grad=True)
            self.p["ln%d_b" % i] = Tensor(np.zeros(d_model), requires_grad=True)

    def params(self):
        out = [self.p["tok"], self.p["pos"], self.p["Wout"]]
        for layer in self.layers:
            out.extend(layer.values())
        for i in range(self.n_layers):
            out += [self.p["ln%d_w" % i], self.p["ln%d_b" % i]]
        return out

    def attention(self, x, layer):
        batch, length, dim = x.shape
        heads, head_dim = self.n_heads, self.head_dim

        def split_heads(t):
            return t.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)

        query = split_heads(x @ layer["Wq"])
        key = split_heads(x @ layer["Wk"])
        value = split_heads(x @ layer["Wv"])
        scores = query @ key.transpose(0, 1, 3, 2) * (1.0 / np.sqrt(head_dim))
        out = (scores.softmax(axis=-1) @ value)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, dim)
        return out @ layer["Wo"]

    def handed(self, x, i):
        """第 i 层动手之前，拿到手的是哪个张量。"""
        if not self.norm:
            return x
        return layer_norm(x, self.p["ln%d_w" % i], self.p["ln%d_b" % i])

    def forward(self, tokens, stats=None):
        x = embedding(self.p["tok"], tokens) + self.p["pos"]
        if stats is not None:
            stats.append(("输入（词向量+位置编码）", x, x))
        for i, layer in enumerate(self.layers):
            h = self.handed(x, i)
            x = x + self.attention(h, layer)
            if stats is not None:
                stats.append(("第 %d 层" % (i + 1), h, x))
        return x @ self.p["Wout"]

    def zero_grad(self):
        for p in self.params():
            p.zero_grad()

    def step(self, lr):
        for p in self.params():
            p.data -= lr * p.grad


def train(n_layers, norm, steps=STEPS):
    inputs, targets = make_data()
    model = Stack(n_layers, norm)
    for p in model.params():
        p.zero_grad()
    history = []
    for step in range(steps):
        loss = cross_entropy(model.forward(inputs), targets)
        model.zero_grad()
        loss.backward()
        model.step(LR)
        history.append(float(loss.data))
    return history


def layer_table(n_layers=8):
    """一张表：两种写法下，每一层拿到手的那个张量有多大。"""
    inputs, _ = make_data()
    columns = {}
    for norm in (False, True):
        model = Stack(n_layers, norm)
        stats = []
        model.forward(inputs, stats)
        columns[norm] = stats
    print(f"  {'层':<22}{'无归一化：均值':>16}{'标准差':>10}"
          f"{'有归一化：均值':>18}{'标准差':>10}")
    print("  " + "-" * 74)
    for (name, _, _), (_, h2, _) in zip(columns[False], columns[True]):
        h1 = dict((n, h) for n, h, _ in columns[False])[name]
        print(f"  {name:<22}{h1.data.mean():>+16.3f}{h1.data.std():>10.3f}"
              f"{h2.data.mean():>+18.3f}{h2.data.std():>10.3f}")


def real_hidden_states(n_layers=1):
    """拿真实跑出来的一组隐藏向量来做第三个实验。

    形状 (20, 11, 32)：20 个训练样本，每个 11 个位置，每个位置 32 个数。
    """
    inputs, _ = make_data()
    model = Stack(n_layers, norm=False)
    stats = []
    model.forward(inputs, stats)
    return stats[1][1].data          # 第 1 层的输出


def dumb_normalise(x, eps=1e-5):
    """最笨的归一化：把这一批里所有样本、所有位置的数倒进一个盆，
    逐维算均值和标准差，再拿它来归一化。"""
    mu = x.mean(axis=(0, 1), keepdims=True)
    sd = x.std(axis=(0, 1), keepdims=True)
    return (x - mu) / (sd + eps)


def dumb_normalise_demo():
    h = real_hidden_states()
    target = 0                        # 第一个样本：第一句话，第 0 个位置被挖空
    variants = {
        "只它自己一批（1 个样本）": [target],
        "跟第 6 个样本一批（2 个）": [target, 5],
        "跟另外两个一批（3 个）": [target, 5, 9],
        "跟全部 20 个一批": list(range(len(h))),
    }
    print("  同一个样本，换一批邻居，归一化之后拿到的东西就变了：")
    print(f"    {'这一批里有谁':<26}{'第 0 个位置的头 4 个数':>30}")
    print("    " + "-" * 54)
    outs = {}
    for name, idx in variants.items():
        batch = dumb_normalise(h[idx])
        outs[name] = batch[0, 0, :4]
        print(f"    {name:<26}" + "  ".join(f"{v:+.3f}" for v in batch[0, 0, :4]))
    first = outs["只它自己一批（1 个样本）"]
    last = outs["跟全部 20 个一批"]
    print(f"\n    最大差异：{np.abs(first - last).max():.3f}"
          f"    （同一句话、同一个位置，只是邻居换了）")

    print("\n  换成 LayerNorm，同样几个批次：")
    outs2 = {}
    for name, idx in variants.items():
        x = Tensor(h[idx])
        w = Tensor(np.ones(h.shape[-1]))
        b = Tensor(np.zeros(h.shape[-1]))
        outs2[name] = layer_norm(x, w, b).data[0, 0, :4]
        print(f"    {name:<26}" + "  ".join(f"{v:+.3f}" for v in outs2[name]))
    all_same = all(np.allclose(outs2[k], outs2["只它自己一批（1 个样本）"], atol=1e-12)
                   for k in outs2)
    print(f"\n    所有批次的结果完全一样：{all_same}")
    print("    （LayerNorm 只看这一个位置自己的 32 个数，不看邻居）")


if __name__ == "__main__":
    print("=" * 78)
    print("一、每一层拿到手的数有多大（不训练，只看一次前向，8 层）")
    print("=" * 78)
    layer_table(8)

    print()
    print("=" * 78)
    print("二、训练 8 层，400 步")
    print("=" * 78)
    inputs, targets = make_data()
    for norm in (False, True):
        history = train(8, norm)
        tag = "有归一化" if norm else "无归一化"
        marks = "  ".join(f"第 {s} 步 {history[s]:.4g}" for s in (0, 1, 10, 50, 200, 399))
        print(f"  {tag}：{marks}")
    print()
    print("  无归一化那个：第 0 步的 loss 是 29.8，走一步就涨到 25 亿，第 10 步变成 nan。")

    print()
    print("=" * 78)
    print("三、最笨的做法：把整批数据放在一起归一化")
    print("=" * 78)
    dumb_normalise_demo()
