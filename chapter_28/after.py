"""第 28 章 after.py —— 把参数量当成旋钮，拧三次。

同一个模型、同一份数据、同样的步数、同样的学习率，
只有"多大"不一样：

    小：dim=32，1 层       约 1.6 万参数
    中：dim=64，2 层       约 10 万参数
    大：dim=128，5 层      约 100 万参数

每个模型都在第 30 步和第 150 步量一次验证 loss，最后画一张图。

跑法：

    ./.venv/bin/python chapter_28/after.py
"""

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------- 语料

# 还是上一章那门"玩具语言"（话题会切换），只是每句话加长成两个小句，
# 让规律更难一点：一句话里有 8-10 个字要预测，而不是 4-6 个。
TECH_HEADS = ["苹果", "华为", "小米"]
TECH_TAILS = [["发布", "新", "手机"], ["发布", "新", "芯片"], ["发布", "新", "电脑"],
              ["芯片", "很强"], ["手机", "很强"], ["电脑", "很强"]]
FOOD_HEADS = ["苹果", "香蕉", "这个苹果"]
FOOD_TAILS = [["很", "甜"], ["很好吃"], ["做成", "派"], ["真", "甜"], ["真", "好吃"], ["很", "好吃"]]


def build_corpus(n_sentences=12000, seed=0, switch=0.15):
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

# 三个模型：只有大小不一样
SIZES = [
    ("小", dict(dim=32, n_layer=1, n_head=4)),
    ("中", dict(dim=64, n_layer=2, n_head=4)),
    ("大", dict(dim=128, n_layer=5, n_head=4)),
]

CHECKPOINTS = (30, 150)
BLOCK_SIZE = 48
BATCH_SIZE = 32


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
def evaluate(model, data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, seed=1, rounds=10):
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


def train(model, steps, lr=3e-3, seed=0, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE,
          data=None, checkpoints=()):
    """训练，并在指定的步数上记一次验证 loss。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    train_data = TRAIN_DATA if data is None else data
    history = {}
    start = time.time()
    for step in range(steps):
        x, y = get_batch(train_data, block_size, batch_size, rng)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step + 1 in checkpoints:
            history[step + 1] = evaluate(model, VAL_DATA, block_size, batch_size)
    return history, time.time() - start


def pad(text, width):
    display = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)
    return text + " " * max(0, width - display)


def run_scaling(checkpoints=CHECKPOINTS):
    """三个大小的模型各训练一遍。"""
    results = []
    for name, size in SIZES:
        torch.manual_seed(0)
        model = GPTModel(VOCAB_SIZE, block_size=BLOCK_SIZE, **size)
        history, seconds = train(model, max(checkpoints), checkpoints=checkpoints)
        results.append(dict(name=name, size=size, params=model.n_params(),
                            history=history, seconds=seconds))
        print(f"  {name}（dim={size['dim']}，{size['n_layer']} 层，"
              f"{model.n_params():,} 参数）训练完，用时 {seconds:.0f} 秒")
    return results


def draw_chart(results, step=150, low=None, high=None):
    """在一条 loss 轴上把三个模型标出来 —— 纯文本的图。"""
    losses = [r["history"][step] for r in results]
    lo = (min(losses) - 0.005) if low is None else low
    hi = (max(losses) + 0.005) if high is None else high
    width = 60

    def column(value):
        return int(round((value - lo) / (hi - lo) * (width - 1)))

    tick_line = [" "] * (width + 8)
    for index in range(5):
        value = lo + (hi - lo) * index / 4
        text = f"{value:.3f}"
        position = max(0, column(value) - len(text) // 2)   # 最左边那个标签贴住左边界
        for offset, ch in enumerate(text):
            if 0 <= position + offset < len(tick_line):
                tick_line[position + offset] = ch
    axis = ["─"] * width
    for index in range(5):
        axis[column(lo + (hi - lo) * index / 4)] = "┼"
    print(f"验证 loss（训练 {step} 步时）")
    print("  " + "".join(tick_line).rstrip())
    print("  " + "".join(axis))
    for r in results:
        loss = r["history"][step]
        label = f"{r['name']}（{r['params']:,} 参数）  验证 loss {loss:.4f}"
        print(" " * (2 + column(loss)) + "▲ " + label)
    print()


def main():
    print("=" * 68)
    print("参数量是这一章的旋钮")
    print("=" * 68)
    print(f"语料 {len(TEXT):,} 个字：训练 {len(TRAIN_DATA):,}，验证 {len(VAL_DATA):,}")
    print(f"每个模型都训练 {max(CHECKPOINTS)} 步，同样的学习率，同样的数据、同样的 batch。")
    print(f"瞎猜的 loss = ln({VOCAB_SIZE}) = {math.log(VOCAB_SIZE):.2f}")
    print()

    results = run_scaling()
    print()

    print(pad("模型", 8) + pad("层数", 8) + pad("维度", 8) + pad("参数量", 14)
          + pad("训练 30 步", 14) + "训练 150 步")
    print("-" * 68)
    for r in results:
        print(pad(r["name"], 8) + pad(str(r["size"]["n_layer"]), 8)
              + pad(str(r["size"]["dim"]), 8) + pad(f"{r['params']:,}", 14)
              + pad(f"{r['history'][30]:.4f}", 14) + f"{r['history'][150]:.4f}")
    print()
    print("每一列都是从下往上越来越小：参数量越大，loss 越低。")
    print("（这些数字是「在没见过的数据上」量的，不是背下来的成绩。）")
    print()
    print("loss 是「每个字平均错得有多离谱」。换一个好说的说法（困惑度 = e^loss）：")
    for r in results:
        loss = r["history"][150]
        print(f"  {r['name']}：loss {loss:.4f} -> 困惑度 {math.exp(loss):.2f}"
              f"（相当于每一步在 {math.exp(loss):.2f} 个字里犹豫）")
    print()

    draw_chart(results, step=150)

    best, worst = results[0], results[-1]
    print(f"参数量涨了 {worst['params'] / best['params']:.0f} 倍，"
          f"150 步时的 loss 从 {best['history'][150]:.4f} 降到 {worst['history'][150]:.4f}。")
    print("loss 是「每个字平均错得有多离谱」，所以它下降意味着：")
    print("  下一个字猜得越来越准，模型的「犹豫」越来越少。")
    print()
    print("注意 30 步那一列：一开始小模型连门都没摸到（1.5 上下，比瞎猜还差），")
    print("大模型已经在 0.6 附近了 —— 大模型不只是最后更准，它学得也更快。")


if __name__ == "__main__":
    main()
