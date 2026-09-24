"""第 30 章 after.py —— 全书的最后一段代码。

29 章之后，我们手上是这么个东西：

    一个字符级语言模型（第 22 章的结构 + 第 27 章的选择 + 第 25 章的引擎），
    它唯一会做的事是"预测下一个 token"。

这一章要拿它去做第 2 章那件事：给句子分类（科技 / 食品）。

怎么用"预测下一个字"来分类？第 20 章说过：分类就是下一 token 预测。
我们把语料写成这种样子：

    苹果发布新手机分类：科技。
    苹果很甜分类：食品。

然后在测试的时候，让它算一算"科技"和"食品"哪个的概率高。

最后一段代码是全书的最后一行：
    logits = model(tokens)
    next_token = sample(logits[-1])

跑法：

    ./.venv/bin/python chapter_30/after.py
"""

import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ---------------------------------------------------------------- 语料

# 用的还是第 1 章那批词，把它们组合成很多句子，每句都带上类别。
TECH_SUBJECTS = ["苹果", "华为", "小米"]
TECH_ACTIONS = ["发布新手机", "发布新电脑", "发布新芯片", "芯片很强", "手机很强", "电脑很强"]
FOOD_SUBJECTS = ["苹果", "香蕉", "这个苹果"]
FOOD_ACTIONS = ["很好吃", "很甜", "做成派", "真甜", "真好吃"]

LABELS = ["科技", "食品"]
FORMAT = "分类："


def build_training_text(n_sentences=400, seed=0):
    rng = random.Random(seed)
    parts = []
    for _ in range(n_sentences):
        if rng.random() < 0.5:
            sentence = rng.choice(TECH_SUBJECTS) + rng.choice(TECH_ACTIONS)
            label = "科技"
        else:
            sentence = rng.choice(FOOD_SUBJECTS) + rng.choice(FOOD_ACTIONS)
            label = "食品"
        parts.append(sentence + FORMAT + label + "。")
    return "".join(parts)


TEXT = build_training_text()
CHARS = sorted(set(TEXT) | set("".join(LABELS)) | set("".join(s for s, _ in
      [("苹果发布新手机", ""), ("小米直播带货", ""), ("华为发布新平板", ""),
       ("苹果不好吃", ""), ("香蕉做成派很好吃", ""), ("苹果手机很甜", ""),
       ("这个手机真好吃", ""), ("床前明月光", "")])))
CHAR_TO_ID = {ch: i for i, ch in enumerate(CHARS)}
ID_TO_CHAR = {i: ch for ch, i in CHAR_TO_ID.items()}
VOCAB_SIZE = len(CHARS)
DATA = np.array([CHAR_TO_ID[ch] for ch in TEXT])

# 和 before.py 完全相同的测试集
TEST_SET = [
    ("苹果发布新手机", "科技"),
    ("苹果发布新芯片", "科技"),
    ("华为发布新电脑", "科技"),
    ("小米发布新手机", "科技"),
    ("苹果芯片很强", "科技"),
    ("苹果很好吃", "食品"),
    ("苹果很甜", "食品"),
    ("香蕉很好吃", "食品"),
    ("这个苹果真甜", "食品"),
    ("苹果做成派", "食品"),
    ("小米直播带货", "科技"),
    ("华为发布新平板", "科技"),
    ("苹果不好吃", "食品"),
    ("香蕉做成派很好吃", "食品"),
    ("苹果手机很甜", "说不清"),
    ("这个手机真好吃", "说不清"),
]


def encode(text):
    return [CHAR_TO_ID[ch] for ch in text]


def decode(ids):
    return "".join(ID_TO_CHAR[i] for i in ids)


# ---------------------------------------------------------------- 模型（还是那个）

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


def get_batch(rng, block_size=48, batch_size=16):
    starts = rng.integers(0, len(DATA) - block_size - 1, size=batch_size)
    x = np.stack([DATA[i:i + block_size] for i in starts])
    y = np.stack([DATA[i + 1:i + block_size + 1] for i in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


def train(model, steps, lr=3e-3, seed=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses = []
    start = time.time()
    for _ in range(steps):
        x, y = get_batch(rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses, time.time() - start


# ---------------------------------------------------------------- 用它来分类

@torch.no_grad()
def label_probabilities(model, sentence):
    """给一句话打分：它是"科技"还是"食品"？

    做法和语言模型完全一样：把"句子分类："当成提示，
    看后面接"科技"和接"食品"哪个概率高。
    """
    prompt = sentence + FORMAT
    ids = encode(prompt)
    scores = {}
    for label in LABELS:
        label_ids = encode(label)
        # 把"句子分类：科技"整段喂进去，只把最后两个字（标签）的对数概率加起来
        logprobs = model(torch.tensor([ids + label_ids]))[0].log_softmax(dim=-1)
        total = 0.0
        for offset, token in enumerate(label_ids):
            total += logprobs[len(ids) - 1 + offset, token].item()
        scores[label] = total
    # 归一化成概率（两个分数里挑一个，用 softmax 摊开）
    values = torch.tensor([scores[label] for label in LABELS])
    probabilities = values.softmax(dim=-1)
    return {label: probabilities[i].item() for i, label in enumerate(LABELS)}


@torch.no_grad()
def sample(logits, temperature=0.8, generator=None):
    """按概率抽一个 token —— 第 23 章的玩法。"""
    return int(torch.multinomial((logits / temperature).softmax(dim=-1), 1,
                                 generator=generator).item())


@torch.no_grad()
def continue_text(model, prompt, n_new=24, seed=0):
    """全书的最后几行代码：一段 token 进去，下一个 token 出来，再喂回去。"""
    generator = torch.Generator().manual_seed(seed)
    tokens = encode(prompt)
    for _ in range(n_new):
        window = tokens[-model.block_size:]
        logits = model(torch.tensor([window]))
        next_token = sample(logits[0, -1], generator=generator)
        tokens.append(next_token)
    return decode(tokens)


def pad(text, width):
    display = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)
    return text + " " * max(0, width - display)


def main():
    print("=" * 70)
    print("把第 2 章那件事，用 29 章之后的模型再做一遍")
    print("=" * 70)
    print(f"训练语料：{len(TEXT)} 个字，形如「苹果发布新手机分类：科技。」")
    print(f"字表：{len(CHARS)} 个不同的字")
    print()

    torch.manual_seed(0)
    model = TinyLM(VOCAB_SIZE, dim=64, n_layer=2, n_head=4, block_size=48)
    print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")
    print("注意：它没有一行 if，也没有一个「关键词表」。")
    print()
    losses, seconds = train(model, steps=600)
    print(f"训练 600 步，用时 {seconds:.0f} 秒，loss "
          f"{losses[0]:.2f} -> {losses[-1]:.2f}")
    print()

    print("=" * 70)
    print("同一批句子，看它怎么说")
    print("=" * 70)
    print(pad("句子", 20) + pad("它说", 8) + pad("科技", 9) + pad("食品", 9) + "对错")
    print("-" * 70)
    correct, total = 0, 0
    for sentence, label in TEST_SET:
        probabilities = label_probabilities(model, sentence)
        guess = max(probabilities, key=probabilities.get)
        mark = ""
        if label != "说不清":
            total += 1
            correct += guess == label
            mark = "" if guess == label else "<- 错了"
        print(pad(sentence, 20) + pad(guess, 8)
              + pad(f"{probabilities['科技']:.3f}", 9)
              + pad(f"{probabilities['食品']:.3f}", 9) + mark)
    print("-" * 70)
    print(f"在 {total} 句能分出对错的句子上，它对了 {correct} 句。")
    print()
    print("和 before.py 那个 if/else 分类器的成绩比一比 ——")
    print("这件事留到 experiment.py 去做，两张成绩单放在一起看。")
    print()
    print("先注意三件事：")
    print("  1. 它给出的不是一个判断，而是一个**概率**。")
    print("  2. 「小米直播带货」和「华为发布新平板」里都有训练时从没见过的字")
    print("     （「直播带货」「平板」），它照样答对了 ——")
    print("     因为它不是靠匹配关键词，而是靠句子里那些它认得的部分。")
    print("  3. 「苹果手机很甜」它答食品（0.971）。这句话本来就说不清，")
    print("     训练语料里没有这种组合，它只能抓住「甜」这个信号。")
    print()

    # ------------------------------------------------------------ 全书的最后一行代码
    print("=" * 70)
    print("全书的最后一行代码")
    print("=" * 70)
    print()
    print("    tokens = encode(\"床前明月光\")")
    print("    for _ in range(24):")
    print("        window = tokens[-model.block_size:]        # 只看最后这么多个 token")
    print("        logits = model(torch.tensor([window]))     # 一段 token 进去")
    print("        next_token = sample(logits[0, -1])         # 下一个 token 出来")
    print("        tokens.append(next_token)")
    print()
    print("    生成结果：" + continue_text(model, "床前明月光", n_new=24, seed=0))
    print()
    print("（第 1 章的第一行代码是 `if \"手机\" in words: print(\"科技\")`。")
    print("  从那一行到这里，中间隔着 30 章，和一个能自己学出来的函数。）")


if __name__ == "__main__":
    main()
