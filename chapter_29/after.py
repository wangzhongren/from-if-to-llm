"""第 29 章 after.py —— 同一个模型，喂三种数据，变成三种东西。

    阶段一 预训练      数据：一堆文本（陈述句、问句、闲聊混在一起）
                      训练目标：预测下一个字（第 20 章那个目标，没变）

    阶段二 指令微调    数据：（问，答）配对，只在"答"上面算 loss
                      训练目标：还是交叉熵，只是换了在哪些位置上算

    阶段三 偏好对齐    数据：（问，好回答，差回答）三件套
                      训练目标：让"好回答"的概率比"差回答"更高

模型一行没改，改的只有数据 —— 这就是这一章要讲的全部。

跑法：

    ./.venv/bin/python chapter_29/after.py
"""

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- 数据

# 一张"事实表"。整章的数据都是从它编出来的。
# (主语, 特征, 陈述句, 问句, 直接的回答, 啰嗦的回答)
FACTS = [
    ("苹果", "甜", "苹果很甜。", "问：苹果甜吗？答：", "甜。", "苹果甜吗？甜。"),
    ("苹果", "好吃", "苹果很好吃。", "问：苹果好吃吗？答：", "好吃。", "苹果好吃吗？好吃。"),
    ("香蕉", "甜", "香蕉很甜。", "问：香蕉甜吗？答：", "甜。", "这个问题问得好。香蕉甜。"),
    ("香蕉", "好吃", "香蕉很好吃。", "问：香蕉好吃吗？答：", "好吃。", "香蕉好吃吗？好吃。"),
    ("这个苹果", "甜", "这个苹果很甜。", "问：这个苹果甜吗？答：", "甜。", "这个苹果是甜的。"),
    ("小米", "发布新手机", "小米发布新手机。", "问：小米发布新手机吗？答：", "是。", "小米发布了新手机。"),
    ("华为", "发布新电脑", "华为发布新电脑。", "问：华为发布新电脑吗？答：", "是。", "华为发布了新电脑。"),
]

# 混在预训练文本里的"闲聊"，让这份语料更像一堆没有整理过的文本
CHITCHAT = ["小王在看书。", "今天天气很好。", "小李在跑步。"]

# 说了"我答完了"的那个特殊 token。它只是一个普通的字串，
# 只是预训练语料里从来没有出现过 —— 它是在微调阶段才被引入的。
END = "<结束>"


def build_pretrain_text(segments=400, seed=0):
    """预训练语料：陈述句、问句、闲聊混在一起。

    关键的一点：**这里只有问句，没有答案**。
    像极了从网上抓下来的一堆文本 —— 问题到处都有，可没人回答。
    """
    rng = random.Random(seed)
    pool = ([fact[2] for fact in FACTS] + [fact[3] for fact in FACTS] + CHITCHAT)
    return "".join(rng.choice(pool) for _ in range(segments))


def build_sft_pairs():
    """指令微调数据：一句一个问题，后面紧跟答案。

    注意这里有**两种风格**的答案 —— 短的、和啰嗦的。
    真实的微调数据就是这样：不同的标注员写法不一样。
    这一阶段之后，模型对两种风格都学会了，还没有"更喜欢哪个"。
    """
    pairs = []
    for _, _, _, question, short, verbose in FACTS:
        pairs.append((question, short + END))
        pairs.append((question, verbose + END))
    return pairs


def build_preference_pairs():
    """偏好数据：同一个问题，一个好回答、一个差回答。

    这里「好」和「差」是我们自己定的：
        好：直接回答（「甜。」）
        差：绕一圈再答（「苹果甜吗？甜。」）—— 内容不算错，就是不干脆

    注意数据里已经没有"正确答案"了，只有"人更喜欢哪个"。
    """
    return [(question, short + END, verbose + END)
            for _, _, _, question, short, verbose in FACTS]


PRETRAIN_TEXT = build_pretrain_text()
CHARS = sorted(set(PRETRAIN_TEXT)
               | {ch for fact in FACTS for ch in fact[3] + fact[4] + fact[5]}
               | {ch for sentence in CHITCHAT for ch in sentence}
               | set(END))
CHAR_TO_ID = {ch: i for i, ch in enumerate(CHARS)}
ID_TO_CHAR = {i: ch for ch, i in CHAR_TO_ID.items()}
VOCAB_SIZE = len(CHARS)


def encode(text):
    return [CHAR_TO_ID.get(ch, 0) for ch in text]


def decode(ids):
    return "".join(ID_TO_CHAR[i] for i in ids)


PRETRAIN_DATA = np.array(encode(PRETRAIN_TEXT))


# ---------------------------------------------------------------- 模型（还是那个模型）

class TinyLM(nn.Module):
    """和第 22 章一样的字符级语言模型。这一章不动它一个零件。"""

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


def make_sequences(pairs, block_size):
    """把（提示，回答）拼成一条序列，并标出「哪些位置要算 loss」。

    返回的是三条等长的列表：输入、答案、以及"这个位置算不算 loss"。
    提示部分标 0，回答部分标 1 —— 这就是指令微调和预训练唯一的区别。
    """
    sequences = []
    for prompt, answer in pairs:
        ids = encode(prompt + answer)[:block_size + 1]
        inputs = ids[:-1]
        targets = ids[1:]
        # 第 p 个位置预测的是 ids[p+1]；它属于"回答"就从第 len(prompt)-1 位开始
        mask = [1 if position >= len(prompt) - 1 else 0
                for position in range(len(inputs))]
        sequences.append((inputs, targets, mask))
    return sequences


def pad_sequences(sequences, block_size):
    """把一批不等长的序列补齐（写成张量才能一起送进模型）。"""
    length = min(block_size, max(len(inputs) for inputs, _, _ in sequences))
    batch = len(sequences)
    inputs = torch.zeros(batch, length, dtype=torch.long)
    targets = torch.zeros(batch, length, dtype=torch.long)
    masks = torch.zeros(batch, length)
    for row, (input_ids, target_ids, mask) in enumerate(sequences):
        size = min(length, len(input_ids))
        inputs[row, :size] = torch.tensor(input_ids[:size])
        targets[row, :size] = torch.tensor(target_ids[:size])
        masks[row, :size] = torch.tensor(mask[:size], dtype=torch.float32)
    return inputs, targets, masks


# ---------------------------------------------------------------- 三种训练目标

def pretrain_step(model, optimizer, batch_size, rng):
    """阶段一：第 20 章的目标 —— 每个位置都预测下一个字。"""
    starts = rng.integers(0, len(PRETRAIN_DATA) - model.block_size - 1, size=batch_size)
    x = np.stack([PRETRAIN_DATA[i:i + model.block_size] for i in starts])
    y = np.stack([PRETRAIN_DATA[i + 1:i + model.block_size + 1] for i in starts])
    logits = model(torch.from_numpy(x))
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), torch.from_numpy(y).reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def sft_step(model, optimizer, batch_size, rng, block_size=32):
    """阶段二：还是交叉熵，但只在「答」那几个位置上算。

    提示部分（「问：……答：」）不参与 loss —— 我们不想让模型学「怎么提问」，
    只让它学「被问了以后该怎么答」。
    """
    pairs = build_sft_pairs()
    chosen = [pairs[i] for i in rng.integers(0, len(pairs), size=batch_size)]
    inputs, targets, masks = pad_sequences(make_sequences(chosen, block_size), block_size)
    logits = model(inputs)
    per_token = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                               targets.reshape(-1), reduction="none")
    per_token = per_token.reshape(targets.shape)
    loss = (per_token * masks).sum() / masks.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def sequence_logprob(model, prompt, answer, differentiable=False):
    """模型给「这个回答」打了多少分：每个字概率的对数，加起来。

    differentiable=False 时用 no_grad，只是量一量；
    训练偏好时要 True，因为我们要对它求导。
    """
    ids = encode(prompt + answer)
    context = torch.no_grad() if not differentiable else torch.enable_grad()
    with context:
        logprobs = model(torch.tensor([ids]))[0].log_softmax(dim=-1)
    total = 0.0
    for position in range(len(prompt) - 1, len(ids) - 1):
        total = total + logprobs[position, ids[position + 1]]
    return total


def dpo_step(policy, reference, optimizer, rng, beta=0.1):
    """阶段三：偏好对齐（DPO 的简化版）。

    它要的东西只有一句话：**让「被选中的回答」比「被拒绝的回答」概率更高**，
    而且要相对于「没有偏好之前的自己」（reference）来比。

    为什么要有 reference？因为只推高好回答的话，模型会把所有概率都堆上去，
    说得越短越好、越像复读越好。用"和原来的自己比"当尺子，
    推的是"哪一种更好"，而不是"哪一种概率更大"。

    注意这里已经没有「正确答案」这个概念了 —— 只有「人更喜欢哪个」。
    """
    pairs = build_preference_pairs()
    prompt, chosen, rejected = pairs[rng.integers(0, len(pairs))]
    chosen_ratio = sequence_logprob(policy, prompt, chosen, differentiable=True) \
        - sequence_logprob(reference, prompt, chosen)
    rejected_ratio = sequence_logprob(policy, prompt, rejected, differentiable=True) \
        - sequence_logprob(reference, prompt, rejected)
    # 这个损失函数要的就是：让两个比值之差越大越好
    loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------- 生成

@torch.no_grad()
def generate(model, prompt, n_new=24, temperature=0.8, seed=0):
    """生成。看到"结束"那个 token 就停下 —— 这是微调阶段教给它的。"""
    generator = torch.Generator().manual_seed(seed)
    ids = encode(prompt)
    for _ in range(n_new):
        window = torch.tensor([ids[-model.block_size:]])
        logits = model(window)[0, -1] / temperature
        ids.append(torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).item())
        if decode(ids).endswith(END):
            break
    text = decode(ids)[len(prompt):]
    return text.replace(END, "[结束]")


def pad(text, width):
    display = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)
    return text + " " * max(0, width - display)


def loud(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------- 主程序

def main():
    print(f"字表：{len(CHARS)} 个不同的字")
    print()

    loud("阶段零：三种数据长什么样")
    print("预训练数据（前 60 个字，注意里面只有问句、没有答案）：")
    print("  " + PRETRAIN_TEXT[:60])
    print()
    print("指令微调数据（提示 + 回答，回答是要学的那部分）：")
    for prompt, answer in build_sft_pairs()[:3]:
        print(f"  {prompt}{answer}")
    print()
    print("偏好数据（同一个问题，一个被选中、一个被拒绝）：")
    for prompt, chosen, rejected in build_preference_pairs()[:2]:
        print(f"  问题：{prompt}")
        print(f"    选中：{chosen}")
        print(f"    拒绝：{rejected}")
    print()

    torch.manual_seed(0)
    model = TinyLM(VOCAB_SIZE, dim=64, n_layer=2, n_head=4, block_size=32)
    print(f"模型：和前面几章一样的字符级模型，参数量 {sum(p.numel() for p in model.parameters()):,}")
    print("这一章从头到尾不会改它一个零件。")
    print()

    question = "问：苹果甜吗？答："

    # ---------------------------------------------------------- 阶段一
    loud("阶段一：预训练（预测下一个字）")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)
    start = time.time()
    losses = []
    for step in range(600):
        losses.append(pretrain_step(model, optimizer, 16, rng))
        if (step + 1) % 200 == 0:
            print(f"  第 {step + 1:>3} 步  loss = {losses[-1]:.3f}")
    print(f"用时 {time.time() - start:.0f} 秒")
    print()
    print(f"现在拿一个问题去问它：{question}")
    print("它续写出来的东西是：")
    for seed in (0, 1):
        print(f"  {generate(model, question, n_new=20, seed=seed)}")
    print()
    print("它不是「答错了」，是根本没在答 —— 它在接着往下写这段文本，")
    print("因为它的训练目标从头到尾只有一件事：下一个字最可能是什么。")
    print("在它的语料里，问句后面跟着的往往是另一个问句、或者一句闲聊。")
    print()

    # ---------------------------------------------------------- 阶段二
    loud("阶段二：指令微调（只在回答上算 loss）")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    start = time.time()
    for step in range(300):
        loss = sft_step(model, optimizer, 16, rng)
        if (step + 1) % 100 == 0:
            print(f"  第 {step + 1:>3} 步  loss = {loss:.3f}（只在答案那几个字上算）")
    print(f"用时 {time.time() - start:.0f} 秒")
    print()
    print("同样的问题，再问一次：")
    for fact in FACTS[:4]:
        print(f"  {fact[3]}{generate(model, fact[3], n_new=24, seed=0)}")
    print("它现在会答了，也会在答完之后说「结束」。")
    print("但它还学会了另一种回答方式 —— 因为我们喂给它的数据里两种都有：")
    for fact in FACTS[:3]:
        print(f"  {fact[3]}{generate(model, fact[3], n_new=24, seed=1)}")
    print()

    # ---------------------------------------------------------- 阶段三
    loud("阶段三：偏好对齐（让人喜欢的回答概率更高）")
    print("先看看微调完的模型，给两种回答各打了多少分：")
    preference_prompt = "问：苹果甜吗？答："
    chosen, rejected = "甜。" + END, "苹果甜吗？甜。" + END

    def probability(logprob):
        return math.exp(logprob)

    print()
    print(pad("回答", 28) + pad("类型", 10) + pad("对数概率", 12) + "概率")
    print("-" * 70)
    before_chosen = sequence_logprob(model, preference_prompt, chosen).item()
    before_rejected = sequence_logprob(model, preference_prompt, rejected).item()
    print(pad(chosen.replace(END, "[结束]"), 28) + pad("被选中", 10)
          + pad(f"{before_chosen:.2f}", 12) + f"{probability(before_chosen):.3f}")
    print(pad(rejected.replace(END, "[结束]"), 28) + pad("被拒绝", 10)
          + pad(f"{before_rejected:.2f}", 12) + f"{probability(before_rejected):.3f}")
    print()
    print("两个回答现在的概率差不多（0.49 对 0.47）—— 微调数据里两种风格都有，")
    print("模型还没学会「更喜欢哪个」。这一步要做的就是教它这件事。")
    print()

    import copy
    reference = copy.deepcopy(model)          # 调之前的自己，留个参照
    policy = model
    # 注意这里的学习率比前两个阶段小得多：偏好对齐推的是"哪个更好"，
    # 推得太猛会把模型推坏（连好回答也不会说了）。
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    start = time.time()
    for step in range(100):
        loss = dpo_step(policy, reference, optimizer, rng)
        if (step + 1) % 50 == 0:
            print(f"  第 {step + 1:>3} 步  偏好 loss = {loss:.4f}")
    print(f"用时 {time.time() - start:.0f} 秒")
    print()

    after_chosen = sequence_logprob(policy, preference_prompt, chosen).item()
    after_rejected = sequence_logprob(policy, preference_prompt, rejected).item()
    print(pad("回答", 28) + pad("类型", 10) + pad("调之前", 16) + "调之后")
    print("-" * 70)
    print(pad(chosen.replace(END, "[结束]"), 28) + pad("被选中", 10)
          + pad(f"{probability(before_chosen):.3f}", 16) + f"{probability(after_chosen):.3f}")
    print(pad(rejected.replace(END, "[结束]"), 28) + pad("被拒绝", 10)
          + pad(f"{probability(before_rejected):.3f}", 16) + f"{probability(after_rejected):.3f}")
    print()
    print("被选中的那个上去了，被拒绝的那个掉下去了 ——")
    print("数据里没有一个字提到「要简洁」，但模型的习惯被这张偏好表改变了。")
    print()

    print("同一批问题，用同一个随机种子再生成一遍：")
    for fact in FACTS[:4]:
        print(f"  {fact[3]}{generate(policy, fact[3], n_new=24, seed=1)}")
    print()
    print("（这两个分数是整段回答的对数概率，越长越吃亏。真实系统里")
    print("  还会做长度归一化，或者用一个单独训练出来的「打分模型」。）")
    print()

    loud("三个阶段，一张表")
    rows = [
        ("阶段一 预训练", "一堆没整理过的文本", "每个位置都预测下一个字", "交叉熵"),
        ("阶段二 指令微调", "（问，答）配对", "只在「答」上预测下一个字", "交叉熵（换了位置）"),
        ("阶段三 偏好对齐", "（问，好答，差答）", "让好答的概率高于差答", "不是交叉熵，但没有新东西"),
    ]
    print(pad("阶段", 20) + pad("数据", 26) + pad("训练目标", 28) + "loss")
    print("-" * 70)
    for row in rows:
        print(pad(row[0], 20) + pad(row[1], 26) + pad(row[2], 28) + row[3])
    print()
    print("模型没变、零件没变、连「预测下一个字」这个动作都没变。")
    print("变的是：**我们把什么样的文本摆到了它面前。**")


if __name__ == "__main__":
    main()
