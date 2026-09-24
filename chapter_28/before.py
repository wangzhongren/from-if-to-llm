"""第 28 章 before.py —— 我们手上唯一的旋钮：多训练一会儿。

第 27 章把模型修好了，也把"GPT 只是大很多"这句话验证了一半。
现在的问题是：我们怎么让自己的模型再强一点？

手上最直观的旋钮是"训练更久"。这个文件把它拧到底：

    同一个 10 万参数的模型，训练 300 步 / 600 步 / 1200 步，看 loss 怎么变。

跑法：

    ./.venv/bin/python chapter_28/before.py
"""

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------- 语料

# 和上一章同一门"玩具语言"：话题会切换，同一个话题里的词才搭得上。
# 这一章把语料做大到 1 万多句，因为要训练大一点的模型。
TECH_HEADS = ["苹果", "华为", "小米"]
TECH_TAILS = [["发布", "新", "手机"], ["发布", "新", "芯片"], ["发布", "新", "电脑"],
              ["芯片", "很强"], ["手机", "很强"], ["电脑", "很强"]]
FOOD_HEADS = ["苹果", "香蕉", "这个苹果"]
FOOD_TAILS = [["很", "甜"], ["很好吃"], ["做成", "派"], ["真", "甜"], ["真", "好吃"], ["很", "好吃"]]


def build_corpus(n_sentences=12000, seed=0, switch=0.15):
    """每句话是两个小句，和 after.py 用的是同一份语料。"""
    rng = random.Random(seed)
    parts = []
    topic = 0

    def clause():
        if topic == 0:
            return rng.choice(TECH_HEADS) + "".join(rng.choice(TECH_TAILS))
        return rng.choice(FOOD_HEADS) + "".join(rng.choice(FOOD_TAILS))

    for _ in range(n_sentences):
        if rng.random() < switch:
            topic = 1 - topic
        parts.append(clause() + "，" + clause() + "。")
    return "".join(parts)


TEXT = build_corpus()
CHARS = sorted(set(TEXT))
CHAR_TO_ID = {ch: i for i, ch in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)
DATA = np.array([CHAR_TO_ID[ch] for ch in TEXT])
SPLIT = int(len(DATA) * 0.8)
TRAIN_DATA = DATA[:SPLIT]
VAL_DATA = DATA[SPLIT:]


# ---------------------------------------------------------------- 模型（第 27 章定下来的）

class Block(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.ln1 = nn.LayerNorm(dim)
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.Wo = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.W1 = nn.Linear(dim, 4 * dim)
        self.W2 = nn.Linear(4 * dim, dim)

    def attention(self, x, mask):
        batch, length, dim = x.shape
        n_head, head_dim = self.n_head, self.head_dim
        split = lambda t: t.reshape(batch, length, n_head, head_dim).transpose(1, 2)  # noqa: E731
        q, k, v = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))
        scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
        scores = scores.masked_fill(mask, float("-inf"))
        out = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.Wo(out)

    def forward(self, x, mask):
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.W2(nn.functional.gelu(self.W1(self.ln2(x))))
        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layer=2, n_head=4, block_size=48):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, idx):
        length = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(length))
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def get_batch(data, block_size, batch_size, rng):
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in starts])
    y = np.stack([data[i + 1:i + block_size + 1] for i in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def evaluate(model, data, block_size=48, batch_size=32, seed=1, rounds=10):
    model.eval()
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(rounds):
        x, y = get_batch(data, block_size, batch_size, rng)
        logits = model(x)
        total += torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)).item()
    model.train()
    return total / rounds


def train(model, steps, lr=3e-3, block_size=48, batch_size=32, seed=0, checkpoints=()):
    """训练，并在给定的步数上记一次验证 loss。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history = {}
    start = time.time()
    for step in range(steps):
        x, y = get_batch(TRAIN_DATA, block_size, batch_size, rng)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step + 1 in checkpoints:
            history[step + 1] = evaluate(model, VAL_DATA, block_size, batch_size)
    return history, time.time() - start


def pad(text, width):
    display = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)
    return text + " " * max(0, width - display)


def main():
    print("=" * 68)
    print("同一个模型，训练更久会怎样")
    print("=" * 68)
    print(f"语料 {len(TEXT):,} 个字，训练 {len(TRAIN_DATA):,}，验证 {len(VAL_DATA):,}")
    print(f"瞎猜的 loss = ln({VOCAB_SIZE}) = {math.log(VOCAB_SIZE):.2f}")
    print()

    torch.manual_seed(0)
    model = GPTModel(VOCAB_SIZE, dim=64, n_layer=2, n_head=4, block_size=48)
    print(f"模型：dim=64，2 层，参数量 {model.n_params():,}")
    print()
    checkpoints = (150, 300, 600, 1200)
    history, seconds = train(model, steps=1200, checkpoints=checkpoints)
    print(pad("训练步数", 12) + pad("验证 loss", 12) + pad("比上一步降了多少", 18) + "用时")
    print("-" * 68)
    previous = None
    for step in checkpoints:
        loss = history[step]
        gain = "—" if previous is None else f"{previous - loss:+.4f}"
        print(pad(f"{step}", 12) + pad(f"{loss:.4f}", 12) + pad(gain, 18) + f"{seconds * step / 1200:.0f} 秒")
        previous = loss
    print()
    print("步数翻 4 倍（300 -> 1200），loss 只降了 "
          f"{history[300] - history[1200]:.4f}。")
    print("收益在迅速变小：这条曲线的形状是'一开始掉得快，后来几乎不动'。")
    print()
    print("多训练一会儿是有效的，但它不是这一章要的答案。")
    print("真正的问题是：为什么 GPT 要那么大？")


if __name__ == "__main__":
    main()
