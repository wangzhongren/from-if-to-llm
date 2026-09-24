"""第 17 章 before.py —— 把前面几章的零件堆起来，结果学不会了。

这个文件演示什么：

    把第 13–16 章一路搭出来的东西（词向量、位置编码、多头 attention）
    串成一个能训练的模型，然后堆 4 层，
    在本章的任务上训练 400 步，loss 卡在 2 上下不动 —— 连训练数据都背不下来。

    1 层：  x = attention(x)                                     ← 够用
    4 层：  x = attention(attention(attention(attention(x))))    ← 学不动

任务：把一句话喂给模型，其中一个词被换成了 <空>，要求模型把整句话还原出来。
      这是个"去噪"任务：最省事的解法是"原样抄一遍，只在空位上补一个词"。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import Tensor, cross_entropy, embedding, randn

# ---------------------------------------------------------------- 语料
# 第 13 章起一直在用的两句话。第二句短两个词，用 <补> 补齐到一样长。
SENTENCES = [
    "我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机",
    "小王 把 书 给了 小李 因为 他 明天 考试",
]
L = 11  # 序列长度


def build_vocab():
    words = []
    for sentence in SENTENCES:
        for word in sentence.split():
            if word not in words:
                words.append(word)
    words.append("<空>")  # 被挖掉的那个位置
    words.append("<补>")  # 补齐用的占位符
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
    """词向量 + 位置编码 + 若干层多头 attention + 输出层。

    第 13–16 章的零件：查表得到词向量，加上位置编码，过 n_layers 层多头 attention。
    """

    def __init__(self, n_layers, d_model=32, n_heads=2, seed=0):
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.p = {}
        # 初始化尺度统一取 1/sqrt(d_model)：一层一层乘下来，数值不会自己涨上去。
        s = 1.0 / np.sqrt(d_model)
        self.p["tok"] = randn(V, d_model, scale=1.0, requires_grad=True, seed=seed + 1)
        self.p["pos"] = randn(L, d_model, scale=1.0, requires_grad=True, seed=seed + 2)
        self.layers = []
        for i in range(n_layers):
            base = seed + 100 * (i + 1)
            layer = {
                "Wq": randn(d_model, d_model, scale=s, requires_grad=True, seed=base + 1),
                "Wk": randn(d_model, d_model, scale=s, requires_grad=True, seed=base + 2),
                "Wv": randn(d_model, d_model, scale=s, requires_grad=True, seed=base + 3),
                "Wo": randn(d_model, d_model, scale=s, requires_grad=True, seed=base + 4),
            }
            self.layers.append(layer)
        self.p["Wout"] = randn(d_model, V, scale=s, requires_grad=True, seed=seed + 999)

    def params(self):
        out = [self.p["tok"], self.p["pos"], self.p["Wout"]]
        for layer in self.layers:
            out.extend(layer.values())
        return out

    def attention(self, x, layer):
        """第 14–15 章那个多头 attention，一个字没改。"""
        batch, length, dim = x.shape
        heads, head_dim = self.n_heads, self.head_dim

        def split_heads(t):
            return t.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)

        query = split_heads(x @ layer["Wq"])
        key = split_heads(x @ layer["Wk"])
        value = split_heads(x @ layer["Wv"])
        scores = query @ key.transpose(0, 1, 3, 2) * (1.0 / np.sqrt(head_dim))
        weights = scores.softmax(axis=-1)
        out = weights @ value
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, dim)
        return out @ layer["Wo"]

    def forward(self, tokens):
        x = embedding(self.p["tok"], tokens) + self.p["pos"]
        for layer in self.layers:
            # ↓↓↓ 一层一层往下串：每一层的输出把上一层整个替换掉 ↓↓↓
            x = self.attention(x, layer)
        return x @ self.p["Wout"]

    def zero_grad(self):
        for p in self.params():
            p.zero_grad()

    def step(self, lr):
        for p in self.params():
            p.data -= lr * p.grad


def train(model, steps=400, lr=0.05, print_every=50):
    inputs, targets = make_data()
    for p in model.params():
        p.zero_grad()
    losses = []
    for step in range(steps):
        logits = model.forward(inputs)
        loss = cross_entropy(logits, targets)
        model.zero_grad()
        loss.backward()
        model.step(lr)
        losses.append(float(loss.data))
        if step % print_every == 0 or step == steps - 1:
            print(f"  第 {step:3d} 步    loss = {loss.data:.4f}")
    return losses


def make_blank_pos():
    """每个样本被挖空的位置。"""
    positions = []
    for sentence in SENTENCES:
        for i in range(len(sentence.split())):
            positions.append(i)
    return np.array(positions)


def show_sentences(model):
    """把模型还原出来的两句话打印出来，和原句对照。"""
    inputs, _ = make_data()
    guess = model.forward(inputs).data.argmax(axis=-1)
    print()
    for row, sentence in enumerate(SENTENCES):
        words = sentence.split()
        sample = row * L  # 这句话的第一个样本（挖空位置 0）
        restored = [VOCAB[guess[sample][i]] for i in range(len(words))]
        print("  原句：  " + " ".join(words))
        print("  还原：  " + " ".join(restored))
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("把 attention 堆 4 层：")
    print("=" * 60)
    model = AttentionStack(n_layers=4)
    train(model, steps=400)

    inputs, targets = make_data()
    blank_pos = make_blank_pos()
    logits = model.forward(inputs)
    guess = logits.data.argmax(axis=-1)
    whole = (guess == targets).mean()
    blank = (guess[np.arange(len(blank_pos)), blank_pos] == targets[np.arange(len(blank_pos)), blank_pos]).mean()
    print(f"\n  整句还原准确率：    {whole * 100:.1f}%")
    print(f"  被挖空位置准确率：  {blank * 100:.1f}%  ({int(blank * len(blank_pos))}/{len(blank_pos)})")
    show_sentences(model)
