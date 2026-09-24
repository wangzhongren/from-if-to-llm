"""第 22 章 after：第一个真正的小语言模型。

前面 19 章的零件，全部装到一起：

    Tokenizer       第 1 章    把一句话拆成 token
    Embedding       第 10 章   把 token 编号变成一组可以学的数字
    位置编码         第 16 章   告诉模型谁在前谁在后
    Self-Attention  第 14 章   每个位置自己决定看哪里
    因果掩码         第 21 章   不许看未来
    多头             第 15 章   并行几种「看的方式」
    残差连接         第 17 章   x = x + layer(x)
    LayerNorm       第 18 章   把每一层的输入拉回正常尺度
    MLP             第 19 章   attention 之后自己想一遍
    Transformer Block × N       第 19 章   上面这些装成一个积木
    线性层 + softmax 第 3、5 章  输出「下一个字符是哪一个」
    交叉熵           第 5 章    衡量错得有多离谱

跑完这个文件，你会第一次看到一台自己造出来的机器写出中文字。

它用的是 toygrad —— 第 9 章我们亲手写的那个自动求导引擎。
这个文件里的每一行运算，梯度都是它算的。
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

# ---------------------------------------------------------------- 语料

POEM_A = "床前明月光疑是地上霜举头望明月低头思故乡"
POEM_B = "鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波"

# 两首诗一共 38 个字。我们把它重复 8 遍，凑出 304 个字符的训练语料。
CORPUS = (POEM_A + POEM_B) * 8

VOCAB = sorted(set(CORPUS))
VOCAB_SIZE = len(VOCAB)

SEED = 20260924


# ---------------------------------------------------------------- 1. Tokenizer

class CharTokenizer:
    """第 1 章的「把一句话拆开」，用在字符上。

    字符级的 tokenizer 是最好写的一种：
    一个字符就是一个 token，词表就是所有出现过的字符。

    真正的 tokenizer 要复杂得多（第 26 章会做），
    但今天它要做的三件事是一样的：

        encode("床前") -> [7, 12]
        decode([7, 12]) -> "床前"
        vocab_size -> 33
    """

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


# ---------------------------------------------------------------- 2. 模型

DIM = 64          # 每个位置上的向量有多宽
LAYERS = 3        # 堆几个 Transformer Block
HEADS = 4         # 多头：把 DIM 切成几份
CONTEXT = 32      # 一次看多少个字符
STEPS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 3e-3


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
    """第 21 章：位置 i 不许看 j > i。"""
    return np.triu(np.ones((length, length), dtype=bool), k=1)


class TransformerBlock:
    """第 19 章装好的那块积木，这次配上因果掩码。

    两句话概括它的结构：

        x = x + 多头注意力(LayerNorm(x))      第 17、18、14、15 章
        x = x + MLP(LayerNorm(x))             第 17、18、19 章

    第一句叫「注意力子层」，第二句叫「MLP 子层」。
    两个子层都是「先归一化 → 再加工 → 再加回自己」。
    """

    def __init__(self, dim, heads, seed):
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        # 注意力子层
        self.query_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 1)
        self.key_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 2)
        self.value_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 3)
        self.out_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 4)
        self.norm1_weight = Tensor(np.ones(dim), requires_grad=True)
        self.norm1_bias = zeros(dim, requires_grad=True)
        # MLP 子层
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
        """第 14 章的 Q / K / V + 第 15 章的多头 + 第 21 章的掩码。"""
        batch, length, _ = x.shape
        # 把最后一维切成 heads 份：(B, T, D) -> (B*heads, T, D/heads)
        def split(tensor):
            return tensor.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        query = split(x @ self.query_weight)      # 我在找什么
        key = split(x @ self.key_weight)          # 我有什么
        value = split(x @ self.value_weight)      # 我给出什么

        scores = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_dim))
        scores = scores.masked_fill(mask, -1e9)
        weights = scores.softmax(axis=-1)
        attended = weights @ value                # (B*heads, T, D/heads)

        # 拼回去：(B*heads, T, D/heads) -> (B, T, D)
        merged = attended.transpose(0, 2, 1, 3).reshape(batch, length, self.dim)
        return merged @ self.out_weight

    def __call__(self, x, mask):
        # 子层一：让位置之间互相看见
        normed = layer_norm(x, self.norm1_weight, self.norm1_bias)
        x = x + self.multi_head_attention(normed, mask)
        # 子层二：每个位置自己想一遍
        normed = layer_norm(x, self.norm2_weight, self.norm2_bias)
        hidden = (normed @ self.fc_weight + self.fc_bias).relu()
        x = x + hidden @ self.proj_weight + self.proj_bias
        return x


class TinyLanguageModel:
    """从 token 编号到「下一个 token 的概率分布」。"""

    def __init__(self, vocab_size, dim=DIM, layers=LAYERS, heads=HEADS, context=CONTEXT, seed=SEED):
        self.dim = dim
        self.layers = layers
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


# ---------------------------------------------------------------- 3. 训练


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
    history = []
    for step in range(steps):
        batch = rng.integers(0, len(inputs), size=batch_size)
        optimizer.zero_grad()
        loss = cross_entropy(model(inputs[batch]), targets[batch])
        loss.backward()
        optimizer.step()
        if (step + 1) % 250 == 0:
            history.append((step + 1, loss.item()))
    return history


def evaluate(model, inputs, targets):
    """整个训练集上的 loss 和准确率。"""
    with no_grad():
        logits = model(inputs)
        loss = cross_entropy(logits, targets).item()
        accuracy = float((logits.data.argmax(axis=-1) == targets).mean())
    return loss, accuracy


def next_token_probabilities(model, tokenizer, prompt):
    """给一段文字，返回下一个字符的概率分布。

    模型对每一个位置都输出一个分布，我们只要最后一个位置的那个。
    """
    ids = tokenizer.encode(prompt)[-CONTEXT:]
    with no_grad():
        logits = model(ids[None, :])
        return logits.softmax(axis=-1).data[0, -1]


def continue_text(model, tokenizer, prompt, length=40, seed=0):
    """把预测出来的字符接回输入，再预测下一个，重复 length 次。

    （这个动作第 23 章会正式起个名字。这里先用最朴素的版本：
    每次都选概率最大的那一个。）
    """
    ids = list(tokenizer.encode(prompt))
    with no_grad():
        for _ in range(length):
            window = np.array(ids[-CONTEXT:])[None, :]
            logits = model(window)
            ids.append(int(logits.data[0, -1].argmax()))
    return tokenizer.decode(ids)


# ---------------------------------------------------------------- 主流程


def main():
    tokenizer = CharTokenizer(CORPUS)
    ids = tokenizer.encode(CORPUS)

    print("=" * 60)
    print("1. Tokenizer（第 1 章）")
    print("=" * 60)
    print(f"  词表大小 {tokenizer.vocab_size}")
    print(f"  词表     {''.join(tokenizer.vocab)}")
    print(f"  语料     {CORPUS[:len(POEM_A) + len(POEM_B)]}  (重复 8 遍，共 {len(CORPUS)} 个字符)")
    print(f"  encode('床前明月') -> {tokenizer.encode('床前明月').tolist()}")
    print(f"  decode([...])      -> '{tokenizer.decode(tokenizer.encode('床前明月'))}'")

    inputs, targets = build_batches(ids)
    print()
    print(f"  训练数据：{len(inputs)} 个长度为 {CONTEXT} 的窗口")

    print()
    print("=" * 60)
    print("2. 组装模型")
    print("=" * 60)
    print("  tokenizer -> embedding -> 位置 -> Transformer Block x N -> 线性层 -> softmax")
    print()
    model = TinyLanguageModel(tokenizer.vocab_size)
    total = model.parameter_count()
    print(f"  维度 {DIM}，{LAYERS} 层，{HEADS} 个头，上下文 {CONTEXT}")
    print(f"  参数量 {total:,}")
    print(f"    token embedding      {model.token_embedding.data.size:>7,}")
    print(f"    位置编码（固定的，不用学） {model.position_table.data.size:>7,}")
    per_block = sum(p.data.size for p in model.blocks[0].params())
    print(f"    每个 Transformer Block {per_block:>6,}  （一共 {per_block * LAYERS:,}）")
    print(f"    最后的线性层          {model.head_weight.data.size:>7,}")

    print()
    print("=" * 60)
    print("3. 训练")
    print("=" * 60)
    history = train(model, inputs, targets)
    for step, loss in history:
        print(f"  第 {step:>4} 步   loss {loss:.4f}")

    loss, accuracy = evaluate(model, inputs, targets)
    print()
    print(f"  整个训练集上的 loss    {loss:.4f}")
    print(f"  下一个字符猜对的比例   {accuracy:.2%}")
    print(f"  （瞎猜的话是 {1 / tokenizer.vocab_size:.2%}）")

    print()
    print("=" * 60)
    print("4. 它学到了什么")
    print("=" * 60)
    for prompt in ["床前明月光疑是地", "举头望明月低头", "白毛浮绿水红掌拨", "鹅鹅鹅曲项向天"]:
        probabilities = next_token_probabilities(model, tokenizer, prompt)
        order = np.argsort(-probabilities)[:3]
        candidates = "   ".join(
            f"'{tokenizer.id_to_char[int(i)]}' {probabilities[i]:.4f}" for i in order
        )
        print(f"  '{prompt}' -> {candidates}")
    print()
    print("  注意第二个：窗口最后两个字是 '低头'，它接的是 '思'。")
    print("  第 20 章那个只看一个字符的分类器在这一格上只能瞎猜。")

    print()
    print("=" * 60)
    print("5. 让它写下去")
    print("=" * 60)
    print("  给它开头「床前」，我们每次取概率最大的那个字符，")
    print("  接到输入后面，再预测下一个。接 60 次：")
    print()
    generated = continue_text(model, tokenizer, "床前", length=60)
    for start in range(0, len(generated), 20):
        print(f"    {generated[start:start + 20]}")
    print()
    print("  再换几个开头：")
    print()
    for prompt in ["鹅鹅鹅", "举头望明月低", "白毛浮绿水"]:
        text = continue_text(model, tokenizer, prompt, length=40)
        print(f"    {prompt} -> {text}")

    print()
    print("=" * 60)
    print("6. 这个模型是什么")
    print("=" * 60)
    print("  它有 %s 个参数，全都是用 toygrad 学的。" % f"{total:,}")
    print("  toygrad 是第 9 章我们写的一个 300 行的小引擎，")
    print("  它做的事情是：记住每一次运算，然后沿着原路把梯度送回去。")
    print()
    print("  PyTorch 里的 nn.Embedding、nn.Linear、nn.LayerNorm、")
    print("  以及一个 Transformer，做的就是这件事。")
    print("  区别只在它更快、算子更多、能跑在显卡上。")
    print()
    print("  我们刚刚从 if/else 走到了这里，中间没有跳过任何一步。")

    print()
    print("=" * 60)
    print("7. 但它只会预测，不会生成")
    print("=" * 60)
    print("  上面那段诗，是我们自己写了个循环，把预测一个接一个接上去的。")
    print("  我们每次只做了两件事：")
    print("    1. 把已经写好的字喂给模型")
    print("    2. 拿它给的概率最大的那个字，接到后面")
    print()
    print("  模型本身不知道什么叫「写一段诗」。")
    print("  它只会回答一个问题：给你这些字，下一个最可能是什么？")
    print()
    print("  那么——「生成」到底是什么？")
    print("  每次只挑概率最大的那一个，是不是最好的挑法？")
    print("  如果换成按概率随机挑，会发生什么？")


if __name__ == "__main__":
    main()
