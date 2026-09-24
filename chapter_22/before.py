"""第 22 章 before：想堆深，但直接堆 attention 堆不动。

第 21 章的模型长这样：

    embedding + 位置 → attention → 线性层

一层 attention，完了直接就是线性层。它能从 `床前明月光` 一路接到 `白毛浮绿水`，
看起来挺好。

这一章我们要把它做大。做大最直接的想法是：**加层啊。**

一层不够就两层，两层不够就三层。一个「叠起来」的模型无非就是：

    x = attention(x)
    x = attention(x)
    x = attention(x)

这个文件就跑一下这件事，看看会发生什么。

结论先摆在这里：**学不动。**
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import Adam, cross_entropy, embedding, no_grad, randn, zeros, SGD

# ---------------------------------------------------------------- 语料（和第 20、21 章一样）

POEM_A = "床前明月光疑是地上霜举头望明月低头思故乡"
POEM_B = "鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波"

CORPUS = (POEM_A + POEM_B) * 8

VOCAB = sorted(set(CORPUS))
CHAR_TO_ID = {char: i for i, char in enumerate(VOCAB)}
ID_TO_CHAR = {i: char for char, i in CHAR_TO_ID.items()}
VOCAB_SIZE = len(VOCAB)

CORPUS_IDS = np.array([CHAR_TO_ID[char] for char in CORPUS])

SEED = 20260924
DIM = 32
CONTEXT = 32
STEPS = 1200
LEARNING_RATE = 0.01


def causal_mask(length):
    return np.triu(np.ones((length, length), dtype=bool), k=1)


class StackedAttention:
    """把第 21 章的注意力层叠 N 层，中间什么都不加。

    （中间什么都不加是错的。这个文件就是要看它错在哪里。）
    """

    def __init__(self, vocab_size, dim=DIM, context=CONTEXT, layers=3, seed=SEED):
        self.dim = dim
        self.context = context
        self.layers = layers
        self.token_table = randn(vocab_size, dim, scale=0.3, requires_grad=True, seed=seed)
        self.position_table = randn(context, dim, scale=0.3, requires_grad=True, seed=seed + 1)
        # 每一层自己一套 Q / K / V
        self.queries = [randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 10 + i)
                        for i in range(layers)]
        self.keys = [randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 20 + i)
                     for i in range(layers)]
        self.values = [randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 30 + i)
                       for i in range(layers)]
        self.outputs = [randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 40 + i)
                        for i in range(layers)]
        self.head_weight = randn(dim, vocab_size, scale=0.3, requires_grad=True, seed=seed + 99)
        self.head_bias = zeros(vocab_size, requires_grad=True)

    def params(self):
        return ([self.token_table, self.position_table]
                + self.queries + self.keys + self.values + self.outputs
                + [self.head_weight, self.head_bias])

    def __call__(self, token_ids):
        batch, length = token_ids.shape
        x = embedding(self.token_table, token_ids) + self.position_table[:length].reshape(1, length, self.dim)
        mask = causal_mask(length)
        for layer in range(self.layers):
            query = x @ self.queries[layer]
            key = x @ self.keys[layer]
            value = x @ self.values[layer]
            attention = (query @ key.transpose(0, 2, 1)) * (1.0 / np.sqrt(self.dim))
            attention = attention.masked_fill(mask, -1e9)
            attended = attention.softmax(axis=-1) @ value
            # 直接换掉 x，没有残差，没有 LayerNorm
            x = attended @ self.outputs[layer]
        return x @ self.head_weight + self.head_bias


def build_batches(ids, context=CONTEXT):
    inputs, targets = [], []
    for start in range(0, len(ids) - context - 1, context // 2):
        inputs.append(ids[start:start + context])
        targets.append(ids[start + 1:start + context + 1])
    return np.array(inputs), np.array(targets)


def train(model, inputs, targets, steps=STEPS, batch_size=32,
          learning_rate=LEARNING_RATE, seed=SEED):
    # 这一章开始用 Adam。SGD 要调很好的学习率才动得起来，
    # 层数一多就更是寸步难行——那是优化器的问题，会掩盖我们要看的现象。
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
    with no_grad():
        return cross_entropy(model(inputs), targets).item()


def main():
    inputs, targets = build_batches(CORPUS_IDS)
    print("=" * 60)
    print("数据")
    print("=" * 60)
    print(f"  语料 {len(CORPUS)} 个字符，{VOCAB_SIZE} 个不同的字符")
    print(f"  {len(inputs)} 个长度为 {CONTEXT} 的窗口")

    print()
    print("=" * 60)
    print("把 attention 一层一层叠上去（中间不加任何东西）")
    print("=" * 60)
    print(f"  {'层数':>4}   {'参数量':>9}   {'训练集 loss':>12}")
    print("  " + "-" * 34)
    for layers in (1, 2, 3, 4):
        model = StackedAttention(VOCAB_SIZE, layers=layers)
        train(model, inputs, targets)
        parameter_count = sum(p.data.size for p in model.params())
        print(f"  {layers:>4}   {parameter_count:>9}   {evaluate(model, inputs, targets):>12.4f}")

    print()
    print("  一两层还行。从第三层开始，loss 直接掉回到头。")
    print()
    print(f"  参考一下：{VOCAB_SIZE} 个字符，如果模型什么都不学、")
    print(f"  每次都在 {VOCAB_SIZE} 个字符里瞎猜，loss 应该是 {np.log(VOCAB_SIZE):.4f}。")
    print("  三层、四层那两个数，基本就是这个水平——它什么都没学会。")
    print()

    print("=" * 60)
    print("为什么")
    print("=" * 60)
    print("  每一层做的事情都是「把各个位置的信息加权平均一下，再乘个矩阵」。")
    print()
    print("  x = attention(x) @ W  这一行有两个问题：")
    print()
    print("  一、它是**替换**，不是叠加。")
    print("      每过一层，上一个 x 就被整个扔掉了。"
          )
    print("      三层下来，最开始那些「我是谁、我在第几位」的信息被冲掉三遍，")
    print("      到最后每个位置的向量都长得差不多了。")
    print()
    print("  二、尺度会跑偏。")
    print("      x 乘一次矩阵，长度就变一次；再过一个 softmax，又变一次。")
    print("      没有东西把它拉回正常范围，几层之后数值要么爆掉要么塌掉。")
    print()

    print("=" * 60)
    print("第 17、18、19 章各给过一个补丁")
    print("=" * 60)
    print("  第 17 章：残差连接     x = x + layer(x)")
    print("     让 x 有一条「原样传下去」的通路，不会被抹平。")
    print()
    print("  第 18 章：LayerNorm    把每一层的输入拉回正常尺度")
    print("     让每一层的输入都在同一个量级上，不会越走越偏。")
    print()
    print("  第 19 章：MLP          attention 后面加一个小网络")
    print("     attention 只会「交换」信息，不会「加工」信息。")
    print("     每个位置自己想一遍，才能算出新东西。")
    print()
    print("  第 19 章把这些装在一起，起名叫 Transformer Block。")
    print("  但那时候它只是个空壳——它后面还没有语言模型的任务。")
    print()
    print("  现在有了。第 20 章给了目标（预测下一个 token），")
    print("  第 21 章给了掩码（不许偷看），")
    print("  第 22 章要把零件全装进去。")
    print()
    print("  打开 after.py。")


if __name__ == "__main__":
    main()
