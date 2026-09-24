"""第 23 章 after：从「预测一个 token」到"生成一段文字"。

上一章我们让模型写出了诗，但那个循环里一直有一件事没说：
**我们每次都把概率最大的那个字符拿走，把剩下 32 个概率全扔了。**

这一章把那 32 个捡回来。

三件事：

  1. 给那个循环起名字。它叫**自回归**（autoregressive）。
  2. 学会第一个旋钮：**temperature**。它控制"敢不敢冒险"。
  3. 学会第二个旋钮：**top-k**。它控制"最多在几个候选里挑"。

模型本身一个字都没改。改的只是"拿到那 33 个概率之后怎么办"。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import (
    Tensor,
    Adam,
    cross_entropy,
    embedding,
    layer_norm,
    no_grad,
    randn,
    zeros,
)

# ---------------------------------------------------------------- 语料（和第 20、21、22 章一样）

POEM_A = "床前明月光疑是地上霜举头望明月低头思故乡"
POEM_B = "鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波"

CORPUS = (POEM_A + POEM_B) * 8

VOCAB = sorted(set(CORPUS))
VOCAB_SIZE = len(VOCAB)

SEED = 20260924

DIM = 64
LAYERS = 3
HEADS = 4
CONTEXT = 32
STEPS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 3e-3


# ---------------------------------------------------------------- tokenizer（第 1 章）

class CharTokenizer:
    def __init__(self, text):
        self.vocab = sorted(set(text))
        self.char_to_id = {char: i for i, char in enumerate(self.vocab)}
        self.id_to_char = {i: char for char, i in self.char_to_id.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        return np.array([self.char_to_id[char] for char in text])

    def decode(self, ids):
        return "".join(self.id_to_char[int(i)] for i in ids)


# ---------------------------------------------------------------- 模型（第 22 章的，一个字没改）

def sinusoidal_positions(length, dim):
    """位置编码：第 16 章那张固定的正弦表。

        第 i 个位置、第 2k 维   = sin(i / 10000^(2k/dim))
        第 i 个位置、第 2k+1 维 = cos(i / 10000^(2k/dim))

    注意它是**算出来的**，不是学出来的。
    这张表不参与训练，所以它不在 params() 里。
    """
    positions = np.arange(length)[:, None]
    dims = np.arange(dim)[None, :]
    angles = positions / np.power(10000.0, (2 * (dims // 2)) / dim)
    table = np.zeros((length, dim))
    table[:, 0::2] = np.sin(angles[:, 0::2])
    table[:, 1::2] = np.cos(angles[:, 1::2])
    return table


def causal_mask(length):
    return np.triu(np.ones((length, length), dtype=bool), k=1)


class TransformerBlock:
    def __init__(self, dim, heads, seed):
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.query_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 1)
        self.key_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 2)
        self.value_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 3)
        self.out_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 4)
        self.norm1_weight = Tensor(np.ones(dim), requires_grad=True)
        self.norm1_bias = zeros(dim, requires_grad=True)
        self.fc_weight = randn(dim, 4 * dim, scale=0.3, requires_grad=True, seed=seed + 5)
        self.fc_bias = zeros(4 * dim, requires_grad=True)
        self.proj_weight = randn(4 * dim, dim, scale=0.3, requires_grad=True, seed=seed + 6)
        self.proj_bias = zeros(dim, requires_grad=True)
        self.norm2_weight = Tensor(np.ones(dim), requires_grad=True)
        self.norm2_bias = zeros(dim, requires_grad=True)

    def params(self):
        return [
            self.query_weight, self.key_weight, self.value_weight, self.out_weight,
            self.norm1_weight, self.norm1_bias,
            self.fc_weight, self.fc_bias, self.proj_weight, self.proj_bias,
            self.norm2_weight, self.norm2_bias,
        ]

    def multi_head_attention(self, x, mask):
        batch, length, _ = x.shape

        def split(tensor):
            return tensor.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        query = split(x @ self.query_weight)
        key = split(x @ self.key_weight)
        value = split(x @ self.value_weight)
        scores = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_dim))
        scores = scores.masked_fill(mask, -1e9)
        attended = scores.softmax(axis=-1) @ value
        merged = attended.transpose(0, 2, 1, 3).reshape(batch, length, self.dim)
        return merged @ self.out_weight

    def __call__(self, x, mask):
        normed = layer_norm(x, self.norm1_weight, self.norm1_bias)
        x = x + self.multi_head_attention(normed, mask)
        normed = layer_norm(x, self.norm2_weight, self.norm2_bias)
        hidden = (normed @ self.fc_weight + self.fc_bias).relu()
        x = x + hidden @ self.proj_weight + self.proj_bias
        return x


class TinyLanguageModel:
    def __init__(self, vocab_size, dim=DIM, layers=LAYERS, heads=HEADS, context=CONTEXT, seed=SEED):
        self.dim = dim
        self.context = context
        self.token_embedding = randn(vocab_size, dim, scale=0.3, requires_grad=True, seed=seed)
        # 位置表是固定的正弦表（第 16 章），不需要学，也不放进 params()
        self.position_table = Tensor(sinusoidal_positions(context, dim))
        self.blocks = [TransformerBlock(dim, heads, seed + 100 * (i + 1)) for i in range(layers)]
        self.head_weight = randn(dim, vocab_size, scale=0.3, requires_grad=True, seed=seed + 90)
        self.head_bias = zeros(vocab_size, requires_grad=True)

    def params(self):
        collected = [self.token_embedding]
        for block in self.blocks:
            collected += block.params()
        return collected + [self.head_weight, self.head_bias]

    def parameter_count(self):
        return sum(p.data.size for p in self.params())

    def __call__(self, token_ids):
        batch, length = token_ids.shape
        x = embedding(self.token_embedding, token_ids)
        x = x + self.position_table[:length].reshape(1, length, self.dim)
        mask = causal_mask(length)
        for block in self.blocks:
            x = block(x, mask)
        return x @ self.head_weight + self.head_bias


# ---------------------------------------------------------------- 训练

def build_batches(ids, context=CONTEXT):
    inputs, targets = [], []
    for start in range(len(ids) - context):
        inputs.append(ids[start:start + context])
        targets.append(ids[start + 1:start + context + 1])
    return np.array(inputs), np.array(targets)


def train(model, inputs, targets, steps=STEPS, batch_size=BATCH_SIZE,
          learning_rate=LEARNING_RATE, seed=SEED):
    optimizer = Adam(model.params(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        batch = rng.integers(0, len(inputs), size=batch_size)
        optimizer.zero_grad()
        cross_entropy(model(inputs[batch]), targets[batch]).backward()
        optimizer.step()


def train_default_model(tokenizer):
    """把第 22 章那个模型原样训练一遍。"""
    inputs, targets = build_batches(tokenizer.encode(CORPUS))
    model = TinyLanguageModel(tokenizer.vocab_size)
    train(model, inputs, targets)
    return model, inputs, targets


# ---------------------------------------------------------------- 本章的新东西


def pick_next_token(probabilities, temperature=1.0, top_k=None, rng=None):
    """拿到 33 个概率之后，怎么挑出下一个字符。

    greedy（贪心，第 22 章的做法）            直接取最大的那个
    temperature（温度）                      先按温度把分布压平或削尖，再按概率抽
    top-k                                    只保留概率最大的 k 个，其余扔掉，再抽

    参数
    ----
    probabilities : 模型给的 33 个概率（已经 softmax 过）
    temperature   : 正数。小于 1 更保守，大于 1 更冒险
    top_k         : None 表示不限制；k 表示只在前 k 个候选里抽
    rng           : numpy 随机数发生器
    """
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0；想要贪心就传 top_k=1")

    # 1. 温度：把概率重新分配一遍
    #    p_i 正比于 p_i ** (1 / temperature)
    #    temperature < 1 -> 大的更大，小的更小（变保守）
    #    temperature > 1 -> 大家被拉平（变冒险）
    adjusted = probabilities ** (1.0 / temperature)
    adjusted = adjusted / adjusted.sum()

    # 2. top-k：只留下最大的 k 个
    if top_k is not None and top_k < len(adjusted):
        keep = np.argsort(-adjusted)[:top_k]
        cut = np.zeros_like(adjusted)
        cut[keep] = adjusted[keep]
        adjusted = cut / cut.sum()

    # 3. 抽一次
    return int(rng.choice(len(adjusted), p=adjusted))


def generate(model, tokenizer, prompt, length=40, temperature=1.0, top_k=None, seed=0):
    """自回归生成。

    把模型的输出接回输入，再预测下一个，重复 length 次。
    """
    rng = np.random.default_rng(seed)
    ids = list(tokenizer.encode(prompt))
    with no_grad():
        for _ in range(length):
            window = np.array(ids[-CONTEXT:])[None, :]
            logits = model(window).data[0, -1]
            shifted = logits - logits.max()
            probabilities = np.exp(shifted)
            probabilities = probabilities / probabilities.sum()
            ids.append(pick_next_token(probabilities, temperature, top_k, rng))
    return tokenizer.decode(ids)


def next_token_probabilities(model, tokenizer, prompt):
    ids = tokenizer.encode(prompt)[-CONTEXT:]
    with no_grad():
        logits = model(ids[None, :])
        return logits.softmax(axis=-1).data[0, -1]


# ---------------------------------------------------------------- 主流程


def main():
    tokenizer = CharTokenizer(CORPUS)
    print("=" * 60)
    print("准备")
    print("=" * 60)
    print(f"  词表 {tokenizer.vocab_size} 个字符，语料 {len(CORPUS)} 个字符")
    print("  模型和第 22 章完全一样，同一个随机种子，同一个训练脚本。")
    model, inputs, targets = train_default_model(tokenizer)
    with no_grad():
        loss = cross_entropy(model(inputs), targets).item()
    print(f"  训练完 loss {loss:.4f}，参数量 {model.parameter_count():,}")

    print()
    print("=" * 60)
    print("先给它起个名字")
    print("=" * 60)
    print("  我们一直在做这件事：")
    print("    1. 把已经写好的字喂给模型")
    print("    2. 拿到下一个字的概率分布")
    print("    3. 挑一个，接到后面去")
    print("    4. 回到第 1 步")
    print()
    print("  「把输出接回输入、再预测下一个」这个动作，")
    print("  叫**自回归**（autoregressive）。")
    print("  第 22 章我们就一直在用它了，只是当时还没给它起名字。")
    print()
    print("  第 3 步——「挑一个」——是这一章的正题。")
    print("  模型在每个位置给的是 33 个概率，不是 1 个字符。")
    print("  怎么从 33 个概率里挑，是一种**选择**。")

    print()
    print("=" * 60)
    print("旋钮一：temperature（温度）")
    print("=" * 60)
    print("  温度做的事：把概率重新分配一遍。")
    print("    温度调低（0.5）-> 大的更大、小的更小，模型变保守")
    print("    温度调高（4.0）-> 大家被拉平，模型变冒险")
    print()
    print("  同一个开头，只改温度：")
    print()
    prompt = "举头望明月低头思"
    print(f"    开头：{prompt}")
    print()
    for temperature in (0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
        text = generate(model, tokenizer, prompt, 26, temperature=temperature, seed=7)
        print(f"    温度 {temperature:>4}  -> {text[len(prompt):]}")

    print()
    print("  前三个温度写出来的东西**一模一样**。")
    print("  这不奇怪：我们的模型太自信了。")
    print("  训练集上的 loss 是 "
          f"{loss:.4f}，换算成困惑度大约是 {np.exp(loss):.2f}——")
    print("  意思是它在每个位置几乎只在 1 个候选之间犹豫。")
    print("  温度 0.5 和 1.0 都撼不动它，要到 2 以后才开始松动。")
    print()
    print("  这是「语料太小、模型把它背下来了」的直接后果。")
    print("  第 24 章会专门回来看这件事。")

    print()
    print("=" * 60)
    print("旋钮二：top-k")
    print("=" * 60)
    print("  top-k 做的事：只保留概率最大的 k 个候选，其余扔掉，再按概率抽。")
    print("  它和温度管的不是同一件事：")
    print("    温度管「敢不敢冒险」，但它可能让 33 个候选全都有机会")
    print("    top-k 管「最多在几个候选里挑」，它给了一个硬底")
    print()
    print("  固定温度 4.0，只改 k：")
    print()
    for top_k in (1, 2, 3, 5, 10):
        text = generate(model, tokenizer, prompt, 26, temperature=4.0, top_k=top_k, seed=7)
        print(f"    top-k {top_k:>2}  -> {text[len(prompt):]}")

    print()
    print("  k=1 就是贪心——第 22 章那个版本。")
    print("  k 越大，模型越容易走出语料；k 太小，它又会绕回那两首诗。")

    print()
    print("=" * 60)
    print("三个策略摆在一起")
    print("=" * 60)
    header = "  同一个开头：「举头望明月低头思」"
    print(header)
    print()
    strategies = [
        ("贪心（每次都取最大）", dict(temperature=1.0, top_k=1)),
        ("温度 4.0", dict(temperature=4.0)),
        ("top-k 3 + 温度 4.0", dict(temperature=4.0, top_k=3)),
    ]
    for label, options in strategies:
        print(f"  {label}")
        for run_index in range(2):
            text = generate(model, tokenizer, prompt, 24, seed=run_index, **options)
            print(f"    第 {run_index + 1} 次：{text[len(prompt):]}")
        print()

    print("  贪心那一组，两次一模一样。")
    print("  另外两组，两次不一样。")
    print()
    print("  这就是这一章真正拿到的东西：")
    print("  **同一个模型，同一个开头，可以写出不一样的东西——而且你可以控制有多不一样。**")

    print()
    print("=" * 60)
    print("还有什么没解决")
    print("=" * 60)
    print("  我们现在能控制它写得多疯、在几个候选里挑。")
    print("  但它写出来的东西，为什么总是那两首诗里的字？")
    print("  它知道「明月」和「月光」有关系吗？")
    print("  它见过「月亮」这个词吗——我们的语料里可没有「亮」字。")
    print()
    print("  最关键的是：我们从头到尾都没检验过它到底会什么。")
    print("  我们只看了它写出来的字，觉得「像中文」。")
    print("  「像」是一个很弱的证据。")
    print()
    print("  下一章我们不给它加任何新零件，")
    print("  只做一组实验，看看它里面到底装了什么。")


if __name__ == "__main__":
    main()
