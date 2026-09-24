"""第 25 章 before.py —— 用我们自己写的引擎，把第 22 章的模型做大一点。

第 24 章留下的话是"玩具模型为什么这么弱"。我们想把它做大，
于是先拿自己写的引擎试一试。这个文件做两件事：

  一、用 toygrad 训练第 22 章那个小模型。它能跑，loss 会下降。
  二、把 dim 从 32 提到 128、层数从 2 提到 4 —— 也就是第 24 章说的"变大"，
      量一量每一步要多少时间、内存涨多快。

跑法：

    ./.venv/bin/python chapter_25/before.py
"""

import math
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from toygrad import Adam, Tensor, cross_entropy, embedding, layer_norm  # noqa: E402

# ---------------------------------------------------------------- 语料

# 第 20 章起的固定语料：一首古诗 + 一首古诗，一共 36 个不同的字。
CORPUS = "床前明月光疑是地上霜举头望明月低头思故乡" \
         "春眠不觉晓处处闻啼鸟夜来风雨声花落知多少"

CHARS = sorted(set(CORPUS))
CHAR_TO_ID = {ch: i for i, ch in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)
DATA = np.array([CHAR_TO_ID[ch] for ch in CORPUS])


def get_batch(data, block_size, batch_size, rng):
    """从语料里随机剪 batch_size 段长度为 block_size 的片段。

    x 是这一段的字，y 是"每个字的下一个字"——语言模型的答案永远是下一个字。
    """
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in starts])
    y = np.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x, y


# ---------------------------------------------------------------- 模型

def sinusoidal_positions(block_size, dim):
    """第 16 章的固定位置编码：每个位置一个向量，不参与训练。

    偶数维用 sin，奇数维用 cos，频率从高到低排开。
    """
    position = np.arange(block_size)[:, None]
    index = np.arange(dim)[None, :]
    angle = position / np.power(10000.0, (2 * (index // 2)) / dim)
    return np.where(index % 2 == 0, np.sin(angle), np.cos(angle))


def split_heads(x, batch, length, n_head, head_dim):
    """(B, T, D) -> (B, H, T, head_dim)，把维度切成几个头。"""
    return x.reshape(batch, length, n_head, head_dim).transpose((0, 2, 1, 3))


def merge_heads(x, batch, length, n_head, head_dim):
    """(B, H, T, head_dim) -> (B, T, D)，把几个头拼回去。"""
    return x.transpose((0, 2, 1, 3)).reshape(batch, length, n_head * head_dim)


class OurModel:
    """第 22 章那个字符级小语言模型，跑在我们自己的引擎上。

    结构和 chapter_22/after.py 逐行对齐：

        查表 -> 加位置（第 16 章的固定正弦表）-> [Block × n_layer] -> 输出层

    其中每个 Block（第 19 章装起来的那一套）：

        x = x + 多头注意力(LayerNorm(x))      第 14、15、17、18、21 章
        x = x + MLP(LayerNorm(x))             第 17、18、19 章

    两个子层都是"先归一化 → 再加工 → 再加回自己"（第 18 章我们就是
    先归一化再算的）。注意两件事，它们和第 22 章一致：

      * 注意力那四个投影**没有偏置** —— 第 14 章我们写的就是 x @ W
      * 输出层之前**没有最后那次归一化** —— 最后一层的输出直接进输出层
    """

    def __init__(self, vocab_size, dim=32, n_layer=2, n_head=4, block_size=16, seed=0):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.block_size = block_size
        self.pos_table = sinusoidal_positions(block_size, dim)

        def param(shape, scale):
            return Tensor(rng.standard_normal(shape) * scale, requires_grad=True)

        self.params = {"tok_emb": param((vocab_size, dim), 0.02)}
        for layer in range(n_layer):
            prefix = f"block{layer}."
            # 第 14 章的四个投影，和 chapter_22 一样不带偏置
            for role in ("q", "k", "v", "o"):
                self.params[prefix + "W" + role] = param((dim, dim), 0.02)
            self.params[prefix + "W1"] = param((dim, 4 * dim), 0.02)
            self.params[prefix + "b1"] = Tensor(np.zeros(4 * dim), requires_grad=True)
            self.params[prefix + "W2"] = param((4 * dim, dim), 0.02)
            self.params[prefix + "b2"] = Tensor(np.zeros(dim), requires_grad=True)
            self.params[prefix + "ln1_w"] = Tensor(np.ones(dim), requires_grad=True)
            self.params[prefix + "ln1_b"] = Tensor(np.zeros(dim), requires_grad=True)
            self.params[prefix + "ln2_w"] = Tensor(np.ones(dim), requires_grad=True)
            self.params[prefix + "ln2_b"] = Tensor(np.zeros(dim), requires_grad=True)
        self.params["head_w"] = param((dim, vocab_size), 0.02)
        self.params["head_b"] = Tensor(np.zeros(vocab_size), requires_grad=True)

    def parameters(self):
        return list(self.params.values())

    def n_params(self):
        return sum(p.data.size for p in self.params.values())

    def project(self, x, weight):
        """第 14 章的投影：x @ W，没有偏置。"""
        return x @ self.params[weight]

    def attention(self, x, layer, mask):
        """第 14、15、21 章合起来的东西：多头 + 因果掩码。"""
        batch, length, dim = x.shape
        n_head, head_dim = self.n_head, self.head_dim
        prefix = f"block{layer}."
        q = split_heads(self.project(x, prefix + "Wq"), batch, length, n_head, head_dim)
        k = split_heads(self.project(x, prefix + "Wk"), batch, length, n_head, head_dim)
        v = split_heads(self.project(x, prefix + "Wv"), batch, length, n_head, head_dim)
        scores = q @ k.transpose((0, 1, 3, 2)) * (1.0 / math.sqrt(head_dim))
        scores = scores.masked_fill(mask, -1e9)      # 第 21 章：不许看后面
        weights = scores.softmax(axis=-1)
        out = merge_heads(weights @ v, batch, length, n_head, head_dim)
        return self.project(out, prefix + "Wo")

    def mlp(self, x, layer):
        """第 19 章的 MLP：升 4 倍宽 -> ReLU -> 降回来。"""
        prefix = f"block{layer}."
        hidden = (x @ self.params[prefix + "W1"] + self.params[prefix + "b1"]).relu()
        return hidden @ self.params[prefix + "W2"] + self.params[prefix + "b2"]

    def forward(self, idx):
        batch, length = idx.shape
        x = embedding(self.params["tok_emb"], idx) + Tensor(self.pos_table[:length])
        mask = np.triu(np.ones((length, length), dtype=bool), k=1)
        for layer in range(self.n_layer):
            prefix = f"block{layer}."
            # 子层一：先归一化，再进注意力，再加回自己
            normed = layer_norm(x, self.params[prefix + "ln1_w"], self.params[prefix + "ln1_b"])
            x = x + self.attention(normed, layer, mask)
            # 子层二：先归一化，再进 MLP，再加回自己
            normed = layer_norm(x, self.params[prefix + "ln2_w"], self.params[prefix + "ln2_b"])
            x = x + self.mlp(normed, layer)
        return x @ self.params["head_w"] + self.params["head_b"]


def train(model, steps, block_size, batch_size, lr, seed=0):
    """最普通的训练循环：算 loss、backward、更新参数。"""
    optimizer = Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses = []
    start = time.time()
    for _ in range(steps):
        x, y = get_batch(DATA, block_size, batch_size, rng)
        loss = cross_entropy(model.forward(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses, time.time() - start


def rss_mb():
    """当前进程用掉的内存（峰值，单位 MB）。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main():
    print("=" * 64)
    print(f"语料：{CORPUS}")
    print(f"一共 {len(CORPUS)} 个字，{VOCAB_SIZE} 个不同的字")
    print("=" * 64)

    # ---------------------------------------------------------- 一、小模型
    print()
    print("一、第 22 章的大小：dim=32，2 层")
    print("-" * 64)
    small = OurModel(VOCAB_SIZE, dim=32, n_layer=2, n_head=4, block_size=16, seed=0)
    print(f"参数量：{small.n_params():,}")
    losses, seconds = train(small, steps=200, block_size=16, batch_size=8, lr=3e-3)
    print(f"训练 200 步，用时 {seconds:.2f} 秒（{seconds / 200 * 1000:.1f} 毫秒/步）")
    print("loss 变化：" + "  ".join(f"{losses[i]:.3f}" for i in range(0, 200, 40)))

    # ---------------------------------------------------------- 二、做大一点
    print()
    print("二、第 24 章说的'做大一点'：dim=128，4 层")
    print("-" * 64)
    big = OurModel(VOCAB_SIZE, dim=128, n_layer=4, n_head=4, block_size=32, seed=0)
    print(f"参数量：{big.n_params():,}（比上面大 {big.n_params() / small.n_params():.0f} 倍）")
    before = rss_mb()
    losses, seconds = train(big, steps=5, block_size=32, batch_size=16, lr=3e-3)
    after = rss_mb()
    per_step = seconds / 5
    print(f"只跑了 5 步，用时 {seconds:.2f} 秒（{per_step * 1000:.0f} 毫秒/步）")
    print(f"内存：{before:.0f} MB -> {after:.0f} MB，5 步涨了 {after - before:.0f} MB")
    print()
    print(f"时间还能忍：训练 200 步只要 {per_step * 200:.0f} 秒。")
    print(f"内存不能忍：按每步 {(after - before) / 5:.0f} MB 的速度，"
          f"200 步要 {after + (after - before) / 5 * 195:,.0f} MB。")
    print()
    print("原因：为了 backward，我们的引擎要记住图上每一个中间张量。")
    print("每一个中间张量都是一整个 float64 的 numpy 数组，一步算完它们不会退回去。")
    ratio = 175e9 / big.n_params()
    print(f"GPT-3 有 1750 亿参数，是我们这个模型的 {ratio:,.0f} 倍。")
    print(f"就算时间和内存只按参数量线性增长：一步要 {per_step * ratio / 60:,.0f} 分钟，")
    print(f"一步要吃掉 {(after - before) / 5 * ratio / 1e6:,.0f} TB 内存。这条路走不通。")


if __name__ == "__main__":
    main()
