"""第 29 章 before.py —— 会预测下一个字，不等于会聊天。

这个文件只做一件事：把第 20 章那个训练目标（预测下一个字）跑到极致，
然后拿着它去"问问题"。

三组探测：

    一、接着一句陈述句往下写  -> 它做得很好（它确实学会了这门语言）
    二、以问答的格式问它      -> 它不回答，只是接着往下写
    三、问一个它明明"知道"的事实 -> 还是不回答

跑法：

    ./.venv/bin/python chapter_29/before.py
"""

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- 语料

# 一堆没有整理过的文本：陈述句、问句、闲聊混在一起。
# 注意：**只有问句，没有答案** —— 就像从网上抓下来的东西。
FACTS = [
    ("苹果", "甜", "苹果很甜。", "问：苹果甜吗？答："),
    ("苹果", "好吃", "苹果很好吃。", "问：苹果好吃吗？答："),
    ("香蕉", "甜", "香蕉很甜。", "问：香蕉甜吗？答："),
    ("香蕉", "好吃", "香蕉很好吃。", "问：香蕉好吃吗？答："),
    ("这个苹果", "甜", "这个苹果很甜。", "问：这个苹果甜吗？答："),
    ("小米", "发布新手机", "小米发布新手机。", "问：小米发布新手机吗？答："),
    ("华为", "发布新电脑", "华为发布新电脑。", "问：华为发布新电脑吗？答："),
]

CHITCHAT = ["小王在看书。", "今天天气很好。", "小李在跑步。"]


def build_corpus(segments=400, seed=0):
    rng = random.Random(seed)
    pool = [fact[2] for fact in FACTS] + [fact[3] for fact in FACTS] + CHITCHAT
    return "".join(rng.choice(pool) for _ in range(segments))


TEXT = build_corpus()
CHARS = sorted(set(TEXT))
CHAR_TO_ID = {ch: i for i, ch in enumerate(CHARS)}
ID_TO_CHAR = {i: ch for ch, i in CHAR_TO_ID.items()}
VOCAB_SIZE = len(CHARS)
DATA = np.array([CHAR_TO_ID[ch] for ch in TEXT])


# ---------------------------------------------------------------- 模型

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
        x = x + self.W2(F.gelu(self.W1(self.ln2(x))))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layer=2, n_head=4, block_size=32):
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


def get_batch(rng, block_size=32, batch_size=16):
    starts = rng.integers(0, len(DATA) - block_size - 1, size=batch_size)
    x = np.stack([DATA[i:i + block_size] for i in starts])
    y = np.stack([DATA[i + 1:i + block_size + 1] for i in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def generate(model, prompt, n_new=24, temperature=0.8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    ids = [CHAR_TO_ID.get(ch, 0) for ch in prompt]
    for _ in range(n_new):
        window = torch.tensor([ids[-model.block_size:]])
        logits = model(window)[0, -1] / temperature
        ids.append(torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).item())
    return "".join(ID_TO_CHAR[i] for i in ids)


def main():
    print("=" * 70)
    print("一、先把它训练好（目标只有一个：预测下一个字）")
    print("=" * 70)
    print(f"语料 {len(TEXT)} 个字，{VOCAB_SIZE} 种不同的字")
    print()

    torch.manual_seed(0)
    model = TinyLM(VOCAB_SIZE, dim=64, n_layer=2, n_head=4, block_size=32)
    print(f"参数量 {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)
    start = time.time()
    for step in range(600):
        x, y = get_batch(rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 200 == 0:
            print(f"  第 {step + 1:>3} 步  loss = {loss.item():.3f}")
    print(f"用时 {time.time() - start:.0f} 秒")
    print()

    print("=" * 70)
    print("二、它学会了吗？给它半句话，让它接着写")
    print("=" * 70)
    for prompt in ["苹果很", "香蕉", "今天天气很", "小王在"]:
        print(f"  {prompt} -> {generate(model, prompt, n_new=12, seed=0)[len(prompt):]}")
    print()
    print("这部分它做得很好：这门语言的规律它学到了。")
    print()

    print("=" * 70)
    print("三、那我们像用 ChatGPT 那样问它呢？")
    print("=" * 70)
    for _, _, _, question in FACTS[:4]:
        print(f"  {question}{generate(model, question, n_new=16, seed=0)[len(question):]}")
    print()
    print("它一个字都没答。它在**接着往下写**这段文本 ——")
    print("因为它的训练目标从头到尾只有一件事：下一个字最可能是什么。")
    print()

    print("=" * 70)
    print("四、它到底知不知道'苹果很甜'这件事？")
    print("=" * 70)
    print("同一个事实，我们用陈述句问它，它接得又快又准：")
    print(f"  苹果很 -> {generate(model, '苹果很', n_new=10, seed=3)[3:]}")
    print()
    print("用问句问它，它就不会了：")
    print(f"  {FACTS[0][3]}{generate(model, FACTS[0][3], n_new=16, seed=3)[len(FACTS[0][3]):]}")
    print()
    print("所以它**不是不知道**，是**没有人教过它'被问的时候要怎么答'**。")
    print("在它的语料里，问句后面跟着的往往是另一个问句，或者一句闲聊 ——")
    print("它老老实实地学到了这一点。")
    print()
    print("要让它会答，缺的既不是参数也不是数据量，而是：")
    print("**一种新的数据格式。**")


if __name__ == "__main__":
    main()
