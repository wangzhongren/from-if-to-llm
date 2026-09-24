"""第 27 章 after.py —— 把我们的模型逐项换成 GPT 的选择。

四件事换掉，模型的骨架一点没动：

                我们的（第 22 章）        GPT 的
    ----------------------------------------------------------
    归一化位置   先算、加残差、再 LN        先 LN、再算、再加残差
    激活函数     ReLU                      GELU
    位置信息     固定的正弦表              一张可学习的表
    词表大小     36（字符）                5 万（BPE）

换完之后，同一个 12 层、学习率 0.01 的模型就训得起来了（before.py 里它纹丝不动）。

跑法：

    ./.venv/bin/python chapter_27/after.py
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------- 语料（和 before.py 同一份）

TECH_HEADS = ["苹果", "华为", "小米"]
TECH_TAILS = [["发布", "新", "手机"], ["发布", "新", "芯片"], ["发布", "新", "电脑"],
              ["芯片", "很强"], ["手机", "很强"], ["电脑", "很强"]]
FOOD_HEADS = ["苹果", "香蕉", "这个苹果"]
FOOD_TAILS = [["很", "甜"], ["很好吃"], ["做成", "派"], ["真", "甜"], ["真", "好吃"], ["很", "好吃"]]


def build_corpus(n_sentences=6000, seed=0, switch=0.15):
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
ID_TO_CHAR = {i: ch for ch, i in CHAR_TO_ID.items()}
VOCAB_SIZE = len(CHARS)
DATA = np.array([CHAR_TO_ID[ch] for ch in TEXT])
TRAIN_DATA = DATA[:int(len(DATA) * 0.8)]
VAL_DATA = DATA[int(len(DATA) * 0.8):]


def get_batch(data, block_size, batch_size, rng):
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in starts])
    y = np.stack([data[i + 1:i + block_size + 1] for i in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


def sinusoidal_positions(block_size, dim):
    """第 16 章那张固定的位置表。

    这一章我们要把它换成可学习的，所以得留着这个函数做对照。
    """
    position = np.arange(block_size)[:, None]
    index = np.arange(dim)[None, :]
    angle = position / np.power(10000.0, (2 * (index // 2)) / dim)
    return np.where(index % 2 == 0, np.sin(angle), np.cos(angle))


# ---------------------------------------------------------------- GPT 的模型

class GPTBlock(nn.Module):
    """第 19 章的 Block，换成 GPT 的写法：先 LayerNorm，再进子层。

    注意两个子层的结构都没变：注意力还是第 14、15、21 章那一套，
    MLP 还是"升 4 倍宽、再降回来"（第 19 章）。变的只是归一化放在哪、激活用哪个。

    pre_ln 和 activation 两个开关是为了做对照实验（experiment.py 用），
    默认值就是 GPT 的选择。
    """

    def __init__(self, dim, n_head, pre_ln=True, activation=torch.nn.functional.gelu):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.pre_ln = pre_ln
        self.activation = activation          # 第 8 章那个"插在中间的一刀"
        self.ln1 = nn.LayerNorm(dim)          # 第 18 章：归一化还是那个归一化
        # 第 14 章那四个投影：我们写的是 x @ W，所以 bias=False（和第 22、25 章一致）
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)
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
        if self.pre_ln:
            # 先归一化，再进子层，最后加回残差 —— 第 18 章我们就是这么写的，
            # 第 22 章原样用，GPT 也是这么写的
            x = x + self.attention(self.ln1(x), mask)
            x = x + self.W2(self.activation(self.W1(self.ln2(x))))
        else:
            # 另一种写法：先算，加上残差，再归一化（最早那篇 Transformer 论文的顺序）
            x = self.ln1(x + self.attention(x, mask))
            x = self.ln2(x + self.W2(self.activation(self.W1(x))))
        return x


class GPTModel(nn.Module):
    """我们的模型 / GPT 的模型，差别就在下面这几个开关上。

        learned_pos=False + activation=relu + final_ln=False
            = 第 22 章那个模型（第 25 章搬进 PyTorch 的那份）
        learned_pos=True  + activation=gelu + final_ln=True
            = 换上了 GPT 的三件事
    """

    def __init__(self, vocab_size, dim=64, n_layer=4, n_head=4, block_size=48,
                 learned_pos=True, pre_ln=True, final_ln=True,
                 activation=torch.nn.functional.gelu):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)          # 第 10 章：没变
        self.learned_pos = learned_pos
        if learned_pos:
            self.pos_emb = nn.Embedding(block_size, dim)      # 第 16 章：改成可学习的
        else:
            self.register_buffer("pos_table", torch.tensor(
                sinusoidal_positions(block_size, dim), dtype=torch.float32))
        self.blocks = nn.ModuleList([GPTBlock(dim, n_head, pre_ln, activation)
                                     for _ in range(n_layer)])
        # 第 22 章的模型在输出层前面没有再归一化；GPT 在这里有一次
        self.final_ln = final_ln
        if final_ln:
            self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)                # 第 20 章：还是"分类"
        self.block_size = block_size

    def forward(self, idx):
        length = idx.shape[1]
        if self.learned_pos:
            positions = self.pos_emb(torch.arange(length))     # 位置也是一张查表
        else:
            positions = self.pos_table[:length]
        x = self.tok_emb(idx) + positions
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        if self.final_ln:
            x = self.ln_f(x)
        return self.head(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


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


@torch.no_grad()
def generate(model, prompt, n_new, temperature=0.9, seed=0):
    """第 23 章的自回归生成。"""
    generator = torch.Generator().manual_seed(seed)
    ids = [CHAR_TO_ID[ch] for ch in prompt]
    for _ in range(n_new):
        window = torch.tensor([ids[-model.block_size:]])
        logits = model(window)[0, -1] / temperature
        next_id = torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).item()
        ids.append(next_id)
    return "".join(ID_TO_CHAR[i] for i in ids)


def pad(text, width):
    display = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)
    return text + " " * max(0, width - display)


def show_mapping():
    print("=" * 78)
    print("一、我们的模型 vs GPT，逐项对照")
    print("=" * 78)
    rows = [
        ("查表（词向量）", "第 10 章", "nn.Embedding 查表", "一样"),
        ("位置信息", "第 16 章", "固定的正弦表", "换成一张可学习的表"),
        ("Q / K / V", "第 14 章", "三个投影 + 打分 + softmax", "一样"),
        ("多头", "第 15 章", "拆成几个头分别算", "一样"),
        ("因果掩码", "第 21 章", "masked_fill(-inf)", "一样"),
        ("残差连接", "第 17 章", "x + 子层(x)", "一样"),
        ("归一化的位置", "第 18 章", "先归一化，再进子层", "一样"),
        ("MLP", "第 19 章", "升 4 倍宽 -> 激活 -> 降回来", "一样"),
        ("激活函数", "第 8 章", "ReLU", "换成 GELU"),
        ("输出层前的 LN", "第 22 章", "没有（直接进输出层）", "有一次"),
        ("输出层", "第 20 章", "D 维 -> 词表大小的分数", "一样"),
        ("训练目标", "第 20 章", "预测下一个 token", "一样"),
    ]
    print(pad("零件", 18) + pad("我们哪一章写的", 18) + pad("我们的做法", 26) + "GPT 的做法")
    print("-" * 78)
    for part, chapter, ours, gpt in rows:
        print(pad(part, 18) + pad(chapter, 18) + pad(ours, 26) + gpt)
    print()
    print("12 项里有 9 项写着'一样'—— 包括第 18 章那个'归一化放在子层前面'，")
    print("我们第 18 章就是这么写的，第 22 章原样用，GPT 也是这么写的。")
    print("真正换掉的只有三件事：位置表、激活函数、输出层前那一次归一化。")
    print("（GPT-2 在注意力的投影上还带偏置，我们没带；那是个无关紧要的细节，")
    print("  一个偏置加不加，不影响任何结论。）")
    print()


def show_scale():
    print("=" * 78)
    print("三、GPT 的大小")
    print("=" * 78)
    print(pad("模型", 16) + pad("层数", 10) + pad("维度", 10) + pad("参数量", 16) + "训练数据")
    print("-" * 78)
    rows = [
        ("我们的模型", "12", "64", "603,677", "4 万字"),
        ("GPT-2 small", "12", "768", "124,000,000", "约 100 亿字"),
        ("GPT-2 medium", "24", "1024", "355,000,000", "约 100 亿字"),
        ("GPT-2 large", "36", "1280", "774,000,000", "约 100 亿字"),
        ("GPT-2 XL", "48", "1600", "1,500,000,000", "约 100 亿字"),
        ("GPT-3", "96", "12288", "175,000,000,000", "约 3000 亿字"),
    ]
    for name, layers, dim, params, data in rows:
        print(pad(name, 16) + pad(layers, 10) + pad(dim, 10) + pad(params, 16) + data)
    print()
    print("（GPT-2 是 124M 到 1.5B，GPT-3 是 175B；层数、维度来自公开的论文，"
          "训练数据量是大约数。）")
    print()
    print("我们的 12 层、64 维，和 GPT-2 small 的 12 层是同一个层数，")
    print("但维度差 12 倍、参数量差 205 倍。第 28 章就来量一量：")
    print("这个倍数到底换来了什么。")
    print()


def main():
    show_mapping()
    print(f"语料：{len(TEXT)} 个字，{VOCAB_SIZE} 种不同的字")
    print()

    print("=" * 78)
    print("二、换成 GPT 的三件事，再跑一次'12 层 + 学习率 0.01'")
    print("=" * 78)
    torch.manual_seed(0)
    model = GPTModel(VOCAB_SIZE, dim=64, n_layer=12, n_head=4, block_size=48)
    print(f"参数量：{model.n_params():,}（和我们自己那个 12 层模型差不多）")
    train(model, steps=200, lr=1e-2)
    print(f"  验证 loss = {evaluate(model, VAL_DATA):.3f}")
    print()
    print("它训起来了。但要看清楚：我们自己那个 12 层模型也训起来了（before.py 第二节），")
    print("两个的验证 loss 挨得很近 —— 这三件事不是'能不能训'的关键，")
    print("（归一化的位置才是，而那一项我们第 18 章就做对了）。")
    print()

    print("它学会了什么？给它两个字，让它往下写：")
    for prompt in ["苹果", "华为", "这个苹果"]:
        print(f"  {prompt} -> {generate(model, prompt, n_new=20, temperature=0.8, seed=0)}")
    print()
    print("话题切换它也会：")
    for prompt in ["苹果很甜。华为", "这个苹果做成派。小米"]:
        print(f"  {prompt} -> {generate(model, prompt, n_new=18, temperature=0.8, seed=1)}")
    print()

    show_scale()


if __name__ == "__main__":
    main()
