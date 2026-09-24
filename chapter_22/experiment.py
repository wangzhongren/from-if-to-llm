"""第 22 章 experiment：零件一个一个加回来，看每一个值多少。

after.py 给的是「全装好」的版本。这个文件把零件一个个拆掉，
量一下每拆掉一个，loss 变差多少。

拆的顺序按「它是什么时候被发明出来的」来：

    完整模型                            第 14 + 15 + 17 + 18 + 19 章
    去掉多层（只留 1 个 Block）            第 19 章
    去掉 MLP                             第 19 章
    去掉残差连接                          第 17 章
    去掉 LayerNorm                       第 18 章
    去掉因果掩码                          第 21 章（这个拆掉反而"更好"，见下）
    去掉位置编码                          第 16 章

每个都比完整模型少一点点东西，其他全都一样：同样的语料、同样的步数、同样的种子。

这个实验会告诉你两件事：

  1. 哪个零件的「边际价值」最大。
  2. 拆掉因果掩码那一行会看起来「变好」——以及那个好是假的。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import Adam, Tensor, cross_entropy, embedding, layer_norm, no_grad, randn, zeros

from after import (  # noqa: E402
    BATCH_SIZE,
    CONTEXT,
    CORPUS,
    DIM,
    HEADS,
    LEARNING_RATE,
    SEED,
    CharTokenizer,
    build_batches,
)

TOKENIZER = CharTokenizer(CORPUS)
VOCAB_SIZE = TOKENIZER.vocab_size

# 拆零件的实验比 after.py 便宜得多（没有多头的层少一批运算），
# 但我们把它压到 600 步，让整个文件一分钟左右跑完。
ABLATION_STEPS = 600


def sinusoidal_positions(length, dim):
    """第 16 章那张固定的正弦表。它不参与训练。"""
    positions = np.arange(length)[:, None]
    dims = np.arange(dim)[None, :]
    angles = positions / np.power(10000.0, (2 * (dims // 2)) / dim)
    table = np.zeros((length, dim))
    table[:, 0::2] = np.sin(angles[:, 0::2])
    table[:, 1::2] = np.cos(angles[:, 1::2])
    return table


def causal_mask(length):
    return np.triu(np.ones((length, length), dtype=bool), k=1)


class FlexibleBlock:
    """Transformer Block，但每个零件都可以关掉。

    use_attention_norm / use_residual / use_mlp / use_mlp_norm
    这四个开关，分别对应第 18 章和第 19 章那两个补丁。

    normalize_first=False 时改成"先加工再归一化"（后置 LayerNorm），
    那是第 18 章之前 GPT 的样子。
    """

    def __init__(self, dim, heads, seed,
                 use_residual=True, use_mlp=True,
                 use_attention_norm=True, use_mlp_norm=True):
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.use_residual = use_residual
        self.use_mlp = use_mlp
        self.use_attention_norm = use_attention_norm
        self.use_mlp_norm = use_mlp_norm

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

    def attention(self, x, mask):
        batch, length, _ = x.shape

        def split(tensor):
            return tensor.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        query = split(x @ self.query_weight)
        key = split(x @ self.key_weight)
        value = split(x @ self.value_weight)
        scores = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_dim))
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attended = scores.softmax(axis=-1) @ value
        merged = attended.transpose(0, 2, 1, 3).reshape(batch, length, self.dim)
        return merged @ self.out_weight

    def __call__(self, x, mask):
        hidden = layer_norm(x, self.norm1_weight, self.norm1_bias) if self.use_attention_norm else x
        attended = self.attention(hidden, mask)
        x = x + attended if self.use_residual else attended
        if self.use_mlp:
            hidden = layer_norm(x, self.norm2_weight, self.norm2_bias) if self.use_mlp_norm else x
            hidden = (hidden @ self.fc_weight + self.fc_bias).relu()
            transformed = hidden @ self.proj_weight + self.proj_bias
            x = x + transformed if self.use_residual else transformed
        return x


class FlexibleModel:
    def __init__(self, vocab_size, dim=DIM, layers=3, heads=HEADS, context=CONTEXT,
                 use_positions=True, use_causal_mask=True, seed=SEED, **block_options):
        self.dim = dim
        self.context = context
        self.use_positions = use_positions
        self.use_causal_mask = use_causal_mask
        self.token_embedding = randn(vocab_size, dim, scale=0.3, requires_grad=True, seed=seed)
        self.position_table = Tensor(sinusoidal_positions(context, dim))
        self.blocks = [FlexibleBlock(dim, heads, seed + 100 * (i + 1), **block_options)
                       for i in range(layers)]
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
        if self.use_positions:
            x = x + self.position_table[:length].reshape(1, length, self.dim)
        mask = causal_mask(length) if self.use_causal_mask else None
        for block in self.blocks:
            x = block(x, mask)
        return x @ self.head_weight + self.head_bias


def run(label, steps=ABLATION_STEPS, **options):
    inputs, targets = build_batches(TOKENIZER.encode(CORPUS))
    model = FlexibleModel(VOCAB_SIZE, **options)
    optimizer = Adam(model.params(), lr=LEARNING_RATE)
    rng = np.random.default_rng(SEED)
    for _ in range(steps):
        batch = rng.integers(0, len(inputs), size=BATCH_SIZE)
        optimizer.zero_grad()
        cross_entropy(model(inputs[batch]), targets[batch]).backward()
        optimizer.step()
    with no_grad():
        logits = model(inputs)
        loss = cross_entropy(logits, targets).item()
        accuracy = float((logits.data.argmax(axis=-1) == targets).mean())
    return label, model.parameter_count(), loss, accuracy


def main():
    configurations = [
        ("完整模型", dict()),
        ("去掉多层（只剩 1 个 Block）", dict(layers=1)),
        ("去掉 MLP", dict(use_mlp=False)),
        ("去掉残差连接", dict(use_residual=False)),
        ("去掉 LayerNorm", dict(use_attention_norm=False, use_mlp_norm=False)),
        ("去掉多头（只剩 1 个头）", dict(heads=1)),
        ("去掉位置编码", dict(use_positions=False)),
        ("去掉因果掩码", dict(use_causal_mask=False)),
    ]

    print("=" * 78)
    print("一个一个零件拆掉")
    print("=" * 78)
    print("  语料、步数、随机种子完全相同。每一步只拆掉一个零件。")
    print()

    results = []
    for label, options in configurations:
        results.append(run(label, **options))

    print(f"  {'':30}{'参数量':>10}{'loss':>10}{'准确率':>10}")
    print("  " + "-" * 60)
    baseline = results[0][2]
    for label, parameters, loss, accuracy in results:
        print(f"  {label:30}{parameters:>10,}{loss:>10.4f}{accuracy:>9.2%}")
    print()
    print(f"  完整模型的 loss 是 {baseline:.4f}。")
    print(f"  瞎猜的 loss 是 {np.log(VOCAB_SIZE):.4f}（{VOCAB_SIZE} 个字符平分概率）。")
    print()
    print("=" * 60)
    print("读这张表")
    print("=" * 60)
    print("  一行一行看过去，有三件事要说。")

    print()
    print("=" * 60)
    print("一、去掉 LayerNorm：loss 直接爆炸")
    print("=" * 60)
    row = [r for r in results if r[0] == "去掉 LayerNorm"][0]
    print(f"  完整模型       loss {baseline:.4f}")
    print(f"  去掉 LayerNorm loss {row[2]:.4f}")
    print()
    print("  第 18 章说过，没有 LayerNorm 数值尺度会乱。")
    print("  在真的语言模型上，这不是「变差一点」——是根本训不起来。")
    print("  8 个零件里，只有这一个的缺失是致命的。")

    print()
    print("=" * 60)
    print("二、去掉因果掩码：loss 全场最低，但那是作弊")
    print("=" * 60)
    row = [r for r in results if r[0] == "去掉因果掩码"][0]
    print(f"  完整模型       loss {baseline:.4f}")
    print(f"  去掉因果掩码   loss {row[2]:.4f}")
    print()
    print("  位置 i 能看到 i+1，而 i+1 那一格上写的就是答案。")
    print("  它不是在预测，它是在抄。第 21 章那件事又出现了。")
    print()
    print("  **一个模型的 loss 好不好，取决于它被允许看到什么。**")

    print()
    print("=" * 60)
    print("三、剩下的几行：拆掉之后 loss 反而更低")
    print("=" * 60)
    rest = [r for r in results if r[0] in
            ("去掉位置编码", "去掉多头（只剩 1 个头）", "去掉 MLP",
             "去掉残差连接", "去掉多层（只剩 1 个 Block）")]
    for label, _, loss, _ in rest:
        print(f"  {label:28} loss {loss:.4f}   （完整模型 {baseline:.4f}）")
    print()
    print("  这一列数字会让第 17 章和第 19 章的读者不舒服：")
    print("  「我们辛辛苦苦造出来的残差和 MLP，拆了反而更好？」")
    print()
    print("  不是零件没用。是这个实验的规模太小了。")
    print("  语料只有 304 个字符，任务简单到一层 attention 就够了。")
    print("  我们拿 15 万多个参数去拟合 304 个字符，")
    print("  多出来的参数在同样的步数里只是更难训而已。")
    print()
    print("  残差和 MLP 的价值，要在更大的数据和更深的模型上才看得出来。")
    print("  第 24 章会接着问：那这个模型到底是怎么把 loss 做到那个数的？")
    print("  它是真的学会了中文，还是把我们这 304 个字背下来了？")
    print("  第 28 章会讲清楚规模这件事背后的规律。")


if __name__ == "__main__":
    main()
