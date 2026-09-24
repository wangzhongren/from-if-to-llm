"""第 27 章 before.py —— 我们那个模型，堆到 12 层会怎样。

第 26 章留下的话是"我们和 GPT 还有什么差别"。
要回答这个，先把我们自己的模型（第 22 章定的结构、第 25 章搬进 PyTorch 的那份）
放到 GPT-2 small 的层数上跑一跑：

    一、4 层，学习率 0.003      —— 正常
    二、12 层，学习率 0.01       —— 也训得起来（第 18 章那个"先归一化再算"的功劳）
    三、把归一化挪到残差外面     —— 同一个模型立刻变成砖头

第三条是这一章最值得看的地方：它告诉我们第 18 章那个顺序不是随便定的。

跑法：

    ./.venv/bin/python chapter_27/before.py
"""

import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------- 语料

# 我们造了一门"玩具语言"：每句话属于一个话题（科技 / 食品），话题会时不时切换。
# 同一个话题里的词才搭得上，所以想预测下一个字，得先猜出这一句在聊什么。
# 诗只有 40 个字，模型背下来就完了，看不出架构的差别；这份语料有 4 万字。
TECH_HEADS = ["苹果", "华为", "小米"]
TECH_TAILS = [["发布", "新", "手机"], ["发布", "新", "芯片"], ["发布", "新", "电脑"],
              ["芯片", "很强"], ["手机", "很强"], ["电脑", "很强"]]
FOOD_HEADS = ["苹果", "香蕉", "这个苹果"]
FOOD_TAILS = [["很", "甜"], ["很好吃"], ["做成", "派"], ["真", "甜"], ["真", "好吃"], ["很", "好吃"]]


def build_corpus(n_sentences=6000, seed=0, switch=0.15):
    """生成语料：话题随机切换，每句话从当前话题的词库里取词。"""
    rng = random.Random(seed)
    parts = []
    topic = 0
    for _ in range(n_sentences):
        if rng.random() < switch:
            topic = 1 - topic
        if topic == 0:
            words = [rng.choice(TECH_HEADS)] + rng.choice(TECH_TAILS)
        else:
            words = [rng.choice(FOOD_HEADS)] + rng.choice(FOOD_TAILS)
        parts.append("".join(words) + "。")
    return "".join(parts)


TEXT = build_corpus()
CHARS = sorted(set(TEXT))
CHAR_TO_ID = {ch: i for i, ch in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)
DATA = np.array([CHAR_TO_ID[ch] for ch in TEXT])
TRAIN_DATA = DATA[:int(len(DATA) * 0.8)]
VAL_DATA = DATA[int(len(DATA) * 0.8):]


def get_batch(data, block_size, batch_size, rng):
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in starts])
    y = np.stack([data[i + 1:i + block_size + 1] for i in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------------------------------------------- 我们的模型

def sinusoidal_positions(block_size, dim):
    """第 16 章的固定位置编码。"""
    position = np.arange(block_size)[:, None]
    index = np.arange(dim)[None, :]
    angle = position / np.power(10000.0, (2 * (index // 2)) / dim)
    return np.where(index % 2 == 0, np.sin(angle), np.cos(angle))


class OurBlock(nn.Module):
    """第 19 章的 Block，用我们第 22 章的选择：ReLU + 固定的正弦位置。

    pre_ln=True  是我们第 18 章/第 22 章的写法：先归一化，再进子层。
    pre_ln=False 是"最早那篇 Transformer 论文"的写法：先算，加残差，再归一化。
    这个开关只在 before.py 里用一下，用来看看顺序换了会怎样。
    """

    def __init__(self, dim, n_head, pre_ln=True):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.pre_ln = pre_ln
        # 第 14 章的四个投影：x @ W，不带偏置（和第 22 章一致）
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        # 第 19 章 MLP：升到 4 倍宽 -> ReLU -> 降回来
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
        if self.pre_ln:
            # 我们的写法：先归一化，再进子层，最后加回自己
            x = x + self.attention(self.ln1(x), mask)
            x = x + self.W2(torch.relu(self.W1(self.ln2(x))))
        else:
            # 另一种写法：先算，加上残差，再归一化
            x = self.ln1(x + self.attention(x, mask))
            x = self.ln2(x + self.W2(torch.relu(self.W1(x))))
        return x


class OurModel(nn.Module):
    """第 22 章的模型：pre-LN + ReLU + 固定位置 + 输出层前不加 LN。"""

    def __init__(self, vocab_size, dim=64, n_layer=4, n_head=4, block_size=48,
                 pre_ln=True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.register_buffer(
            "pos_table",
            torch.tensor(sinusoidal_positions(block_size, dim), dtype=torch.float32),
        )
        self.blocks = nn.ModuleList([OurBlock(dim, n_head, pre_ln) for _ in range(n_layer)])
        # 第 22 章的模型在输出层前面没有再归一化一次
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx):
        length = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_table[:length]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def evaluate(model, data, block_size=48, batch_size=32, seed=1, rounds=10):
    """在没学过的数据上量一量 loss。"""
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


def train(model, steps, lr, block_size=48, batch_size=32, log_every=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    for step in range(steps):
        x, y = get_batch(TRAIN_DATA, block_size, batch_size, rng)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % log_every == 0:
            print(f"  第 {step + 1:>4} 步  训练 loss = {loss.item():.3f}")


def bottom_gradient(model):
    """最底下那一层（词嵌入）拿到的梯度有多大 —— 信号还剩多少的温度计。"""
    return model.tok_emb.weight.grad.norm().item()


def loud(title):
    print("=" * 64)
    print(title)
    print("=" * 64)


def main():
    print(f"语料：{len(TEXT)} 个字，{VOCAB_SIZE} 种不同的字")
    print(f"训练用 {len(TRAIN_DATA)} 个字，验证用 {len(VAL_DATA)} 个字")
    print(f"如果瞎猜，loss 大约是 ln({VOCAB_SIZE}) = {math.log(VOCAB_SIZE):.2f}")
    print()

    loud("一、我们的模型，4 层，学习率 0.003")
    torch.manual_seed(0)
    model = OurModel(VOCAB_SIZE, dim=64, n_layer=4, n_head=4, block_size=48)
    print(f"参数量：{model.n_params():,}")
    train(model, steps=200, lr=3e-3)
    print(f"  验证 loss = {evaluate(model, VAL_DATA):.3f}")
    print()

    loud("二、同一个模型，12 层，学习率 0.01")
    torch.manual_seed(0)
    deep = OurModel(VOCAB_SIZE, dim=64, n_layer=12, n_head=4, block_size=48)
    print(f"参数量：{deep.n_params():,}")
    train(deep, steps=200, lr=1e-2)
    print(f"  验证 loss = {evaluate(deep, VAL_DATA):.3f}")
    print(f"  最底下一层的梯度 = {bottom_gradient(deep):.1e}")
    print()
    print("它训起来了 —— 这就是我们第 18 章那个「先归一化再算」的功劳，")
    print("只不过第 22 章写它的时候，我们并不知道自己在为 12 层做准备。")
    print()

    loud("三、把归一化挪到残差外面（另一种写法）")
    print("顺序换成：先算子层、加上残差、最后归一化。")
    print("这是最早那篇 Transformer 论文的写法，也是自己实现时很容易写出来的顺序。")
    print()
    torch.manual_seed(0)
    post = OurModel(VOCAB_SIZE, dim=64, n_layer=12, n_head=4, block_size=48, pre_ln=False)
    start = time.time()
    train(post, steps=200, lr=1e-2)
    print(f"  验证 loss = {evaluate(post, VAL_DATA):.3f}"
          f"   （瞎猜是 {math.log(VOCAB_SIZE):.2f}）")
    print(f"  最底下一层的梯度 = {bottom_gradient(post):.1e}")
    print(f"  用时 {time.time() - start:.0f} 秒")
    print()
    print("同一个模型、同一份数据、同一个学习率，只把归一化的位置挪了一下：")
    print("一边正常收敛，一边从头到尾停在「只会背字频」的水平。")
    print()
    print("所以这一章要问的是：我们和 GPT 之间剩下的那几件事，")
    print("各自值多少？下一节一件一件换上去看。")


if __name__ == "__main__":
    main()
