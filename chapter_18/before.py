"""第 18 章 before.py —— 第 17 章的模型堆到 8 层，数值一路涨上去，训练当场炸掉。

这个文件演示什么：

    先不训练，只做一次前向，把每一层"拿到手的那个张量"的均值和标准差打印出来。
    你会看到标准差从 1.4 一路涨到 15 —— 越往后的层，拿到手的数越大。

    然后再训练。loss 从第 1 步开始就是几十上百，很快就变成 nan。

第 17 章的模型只有 4 层，这一点还没要命；8 层就要命了。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import cross_entropy, embedding, randn

# ---------------------------------------------------------------- 语料
# 第 13 章起一直在用的两句话。第二句短两个词，用 <补> 补齐到一样长。
SENTENCES = [
    "我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机",
    "小王 把 书 给了 小李 因为 他 明天 考试",
]
L = 11
N_LAYERS = 8


def build_vocab():
    words = []
    for sentence in SENTENCES:
        for word in sentence.split():
            if word not in words:
                words.append(word)
    words.append("<空>")
    words.append("<补>")
    return words


VOCAB = build_vocab()
V = len(VOCAB)
BLANK = VOCAB.index("<空>")
PAD = VOCAB.index("<补>")


def make_data():
    """每个样本 = 一句话 + 一个被挖空的位置。目标 = 整句话。"""
    inputs, targets = [], []
    for sentence in SENTENCES:
        words = sentence.split()
        ids = [VOCAB.index(w) for w in words] + [PAD] * (L - len(words))
        for i in range(len(words)):
            masked = list(ids)
            masked[i] = BLANK
            inputs.append(masked)
            targets.append(ids)
    return np.array(inputs), np.array(targets)


class AttentionStack:
    """第 17 章的模型原样搬过来，只是层数变成了 8 层。"""

    def __init__(self, n_layers=N_LAYERS, d_model=32, n_heads=2, seed=0):
        self.n_layers = n_layers
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

    def blocks(self, tokens):
        """一层一层往前跑，把每一层"拿到手"和"交出去"的张量都留下来。"""
        x = embedding(self.p["tok"], tokens) + self.p["pos"]
        records = [("输入（词向量+位置编码）", x, x)]
        for layer in self.layers:
            handed = x                       # 这一层拿到手的
            x = x + self.attention(handed, layer)   # 加完残差，交出去的
            records.append(("第 %d 层" % len(records), handed, x))
        return x, records

    def forward(self, tokens):
        x, _ = self.blocks(tokens)
        return x @ self.p["Wout"]

    def zero_grad(self):
        for p in self.params():
            p.zero_grad()

    def step(self, lr):
        for p in self.params():
            p.data -= lr * p.grad


def show_stats(model, tokens):
    """打印每一层拿到手的那个张量的均值和标准差。"""
    _, records = model.blocks(tokens)
    print(f"  {'层':<22}{'拿到手：均值':>12}{'标准差':>10}{'交出去：标准差':>14}")
    print("  " + "-" * 58)
    for name, handed, out in records:
        print(f"  {name:<22}{handed.data.mean():>+12.3f}{handed.data.std():>10.3f}"
              f"{out.data.std():>14.3f}")


if __name__ == "__main__":
    inputs, targets = make_data()
    model = AttentionStack()

    print("=" * 62)
    print("一、只看一次前向：每一层拿到手的数有多大？")
    print("=" * 62)
    show_stats(model, inputs)
    print()
    print("  标准差 1.435 → 1.790 → 2.376 → ... → 15.036：")
    print("  每过一层，下一层拿到手的数就大一圈。")

    print()
    print("=" * 62)
    print("二、再训练看看")
    print("=" * 62)
    for p in model.params():
        p.zero_grad()
    for step in range(400):
        loss = cross_entropy(model.forward(inputs), targets)
        model.zero_grad()
        loss.backward()
        model.step(0.05)
        if step < 40 and step % 10 == 0 or step % 100 == 0 or step == 399:
            print(f"  第 {step:3d} 步    loss = {float(loss.data):.4f}")
