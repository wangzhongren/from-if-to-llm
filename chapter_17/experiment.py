"""第 17 章 experiment.py —— 层数越多，反而越学不会。

这个文件做什么：

    同一个任务、同一组超参数，只改两个东西：
        层数：      1 / 2 / 4 / 8 / 16
        有没有残差：x = attention(x)      对比      x = x + attention(x)

    每一步都记录 loss，最后看谁能把两句话背下来。

初始化尺度统一取 1/sqrt(d_model)：每个权重都是一个标准差 1/sqrt(d) 的随机数，
一层一层乘下来，数值尺度不多不少。这是第 8 章以来一直在用的取法。
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import cross_entropy, embedding, randn

from before import L, PAD, V, VOCAB, SENTENCES, make_blank_pos, make_data

STEPS = 400
LR = 0.05
INIT_SCALE = 1.0 / np.sqrt(32)


class AttentionStack:
    """词向量 + 位置编码 + n_layers 层多头 attention。residual 决定加不加那条捷径。"""

    def __init__(self, n_layers, residual, d_model=32, n_heads=2, seed=0, scale=INIT_SCALE):
        self.n_layers = n_layers
        self.residual = residual
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.p = {}
        self.p["tok"] = randn(V, d_model, scale=1.0, requires_grad=True, seed=seed + 1)
        self.p["pos"] = randn(L, d_model, scale=1.0, requires_grad=True, seed=seed + 2)
        self.layers = []
        for i in range(n_layers):
            base = seed + 100 * (i + 1)
            self.layers.append({
                "Wq": randn(d_model, d_model, scale=scale, requires_grad=True, seed=base + 1),
                "Wk": randn(d_model, d_model, scale=scale, requires_grad=True, seed=base + 2),
                "Wv": randn(d_model, d_model, scale=scale, requires_grad=True, seed=base + 3),
                "Wo": randn(d_model, d_model, scale=scale, requires_grad=True, seed=base + 4),
            })
        self.p["Wout"] = randn(d_model, V, scale=scale, requires_grad=True, seed=seed + 999)

    def params(self):
        out = [self.p["tok"], self.p["pos"], self.p["Wout"]]
        for layer in self.layers:
            out.extend(layer.values())
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

    def forward(self, tokens):
        x = embedding(self.p["tok"], tokens) + self.p["pos"]
        for layer in self.layers:
            out = self.attention(x, layer)
            x = x + out if self.residual else out
        return x @ self.p["Wout"]

    def zero_grad(self):
        for p in self.params():
            p.zero_grad()

    def step(self, lr):
        for p in self.params():
            p.data -= lr * p.grad


def run(n_layers, residual, steps=STEPS):
    inputs, targets = make_data()
    blank_pos = make_blank_pos()
    model = AttentionStack(n_layers, residual)
    for p in model.params():
        p.zero_grad()
    history = []
    t0 = time.time()
    for step in range(steps):
        logits = model.forward(inputs)
        loss = cross_entropy(logits, targets)
        model.zero_grad()
        loss.backward()
        model.step(LR)
        history.append(float(loss.data))
    used = time.time() - t0
    logits = model.forward(inputs)
    guess = logits.data.argmax(axis=-1)
    hit = (guess[np.arange(len(blank_pos)), blank_pos]
           == targets[np.arange(len(blank_pos)), blank_pos]).sum()
    return history, int(hit), len(blank_pos), used


if __name__ == "__main__":
    print("=" * 78)
    print(f"任务：把被挖空的那个词填回来（一共 {len(make_blank_pos())} 个位置）")
    print(f"训练：{STEPS} 步，学习率 {LR}，隐藏维度 32，2 个头，初始化尺度 {INIT_SCALE:.3f}")
    print("=" * 78)
    header = f"{'层数':>4} {'残差':>6} {'第 1 步':>10} {'第 50 步':>10} {'第 200 步':>10} {'最后':>10} {'填对':>7} {'用时':>7}"
    print(header)
    print("-" * 78)
    results = {}
    for residual in (False, True):
        for n_layers in (1, 2, 4, 8, 16):
            history, hit, total, used = run(n_layers, residual)
            results[(n_layers, residual)] = history
            tag = "有" if residual else "无"
            print(f"{n_layers:>4} {tag:>6} {history[0]:>10.3f} {history[50]:>10.3f} "
                  f"{history[200]:>10.3f} {history[-1]:>10.4f} {str(hit) + '/' + str(total):>7} {used:>6.1f}s")
        print("-" * 78)

    print()
    print("=" * 78)
    print("两句话还原出来长什么样（4 层）：")
    print("=" * 78)
    inputs, targets = make_data()
    for residual in (False, True):
        model = AttentionStack(4, residual)
        for p in model.params():
            p.zero_grad()
        for step in range(STEPS):
            logits = model.forward(inputs)
            loss = cross_entropy(logits, targets)
            model.zero_grad()
            loss.backward()
            model.step(LR)
        guess = model.forward(inputs).data.argmax(axis=-1)
        tag = "有残差" if residual else "无残差"
        print(f"\n[{tag}]")
        for row, sentence in enumerate(SENTENCES):
            words = sentence.split()
            sample = row * L
            restored = [VOCAB[guess[sample][i]] for i in range(len(words))]
            print("  原句：  " + " ".join(words))
            print("  还原：  " + " ".join(restored))
