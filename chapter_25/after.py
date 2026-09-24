"""第 25 章 after.py —— 把第 22 章的模型用 PyTorch 重写一遍。

这个文件里的每一个类，都能在第 10-22 章找到对应的自己：

    我们写的（toygrad）        PyTorch
    ----------------------     ------------------------------------
    Tensor                     torch.Tensor
    embedding(表, idx)         nn.Embedding
    x @ W + b                  nn.Linear
    layer_norm(x, w, b)        nn.LayerNorm
    手写的多头注意力            nn.MultiheadAttention（见 experiment.py）
    Adam(...)                  torch.optim.Adam

模型结构一个字都没变：还是第 19 章装起来的那一套，还是第 21 章的因果掩码。

跑法：

    ./.venv/bin/python chapter_25/after.py
"""

import math
import os
import resource
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ---------------------------------------------------------------- 语料

CORPUS = "床前明月光疑是地上霜举头望明月低头思故乡" \
         "春眠不觉晓处处闻啼鸟夜来风雨声花落知多少"

CHARS = sorted(set(CORPUS))
CHAR_TO_ID = {ch: i for i, ch in enumerate(CHARS)}
ID_TO_CHAR = {i: ch for ch, i in CHAR_TO_ID.items()}
VOCAB_SIZE = len(CHARS)
DATA = np.array([CHAR_TO_ID[ch] for ch in CORPUS])


def get_batch(data, block_size, batch_size, rng):
    """和 before.py 完全一样：随机剪一段，答案是每个字的下一个字。"""
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in starts])
    y = np.stack([data[i + 1:i + block_size + 1] for i in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


def sinusoidal_positions(block_size, dim):
    """第 16 章的固定位置编码。PyTorch 不会替我们想出来，还是要自己写。"""
    position = np.arange(block_size)[:, None]
    index = np.arange(dim)[None, :]
    angle = position / np.power(10000.0, (2 * (index // 2)) / dim)
    return np.where(index % 2 == 0, np.sin(angle), np.cos(angle))


# ---------------------------------------------------------------- 模型


class Block(nn.Module):
    """第 19 章的 Transformer Block：先归一化，再进子层，再加回残差。

    和第 22 章（before.py）逐行对应：

        x = x + 多头注意力(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        # 第 14 章的四个投影。我们写的是 x @ W，所以这里 bias=False。
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)
        # 第 18 章的 LayerNorm
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        # 第 19 章 MLP：先升到 4 倍宽，再降回来
        self.W1 = nn.Linear(dim, 4 * dim)
        self.W2 = nn.Linear(4 * dim, dim)

    def attention(self, x, mask):
        """手写版的多头注意力 —— 和我们第 15 章写的那个一模一样。"""
        batch, length, dim = x.shape
        n_head, head_dim = self.n_head, self.head_dim

        def split(t):
            # (B, T, D) -> (B, H, T, head_dim)
            return t.reshape(batch, length, n_head, head_dim).transpose(1, 2)

        q = split(self.Wq(x))
        k = split(self.Wk(x))
        v = split(self.Wv(x))
        scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
        scores = scores.masked_fill(mask, float("-inf"))   # 第 21 章：不许看后面
        weights = scores.softmax(dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.Wo(out)

    def forward(self, x, mask):
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.W2(torch.relu(self.W1(self.ln2(x))))    # 第 19 章：ReLU
        return x


class TorchModel(nn.Module):
    """第 22 章的模型，用 PyTorch 重写。"""

    def __init__(self, vocab_size, dim=32, n_layer=2, n_head=4, block_size=16):
        super().__init__()
        self.dim = dim
        self.n_layer = n_layer
        self.block_size = block_size
        # 第 10 章的 embedding 查表：(词表大小, 维度) 的一张表，按编号取行
        self.tok_emb = nn.Embedding(vocab_size, dim)
        # 位置编码固定不动，所以放进 buffer 而不是参数
        self.register_buffer(
            "pos_table",
            torch.tensor(sinusoidal_positions(block_size, dim), dtype=torch.float32),
        )
        self.blocks = nn.ModuleList([Block(dim, n_head) for _ in range(n_layer)])
        # 第 20 章的输出层：把向量变成"下一个字是谁"的分数
        # （注意：第 22 章里输出层前面没有最后那次 LayerNorm，这里也不加。）
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx):
        batch, length = idx.shape
        x = self.tok_emb(idx) + self.pos_table[:length]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def load_our_weights(model, our_model):
    """把 before.py 里（我们自己引擎训练出来）的权重灌进 PyTorch 模型。

    只有一个坑：nn.Linear 存的是 (out, in)，因为它算的是 x @ W.T + b；
    我们手写的是 (in, out)，算的是 x @ W + b。所以要转置 ——
    转置只是"怎么存"，算出来的东西一样。这个函数在 experiment.py 里
    用来验证两个引擎的输出。

    our_model 是 before.py 里的 OurModel，参数放在 .params 字典里。
    """
    state = {}
    to_tensor = lambda w: torch.tensor(np.asarray(w, dtype=np.float32))  # noqa: E731
    state["tok_emb.weight"] = to_tensor(our_model.params["tok_emb"].data)
    state["pos_table"] = to_tensor(our_model.pos_table)   # 位置编码是固定的，两边本来就一样
    for layer in range(our_model.n_layer):
        prefix = f"block{layer}."
        state[f"blocks.{layer}.ln1.weight"] = to_tensor(our_model.params[prefix + "ln1_w"].data)
        state[f"blocks.{layer}.ln1.bias"] = to_tensor(our_model.params[prefix + "ln1_b"].data)
        state[f"blocks.{layer}.ln2.weight"] = to_tensor(our_model.params[prefix + "ln2_w"].data)
        state[f"blocks.{layer}.ln2.bias"] = to_tensor(our_model.params[prefix + "ln2_b"].data)
        # 第 14 章的四个投影没有偏置，所以只搬 weight
        for role in ("q", "k", "v", "o"):
            weight = our_model.params[prefix + "W" + role].data
            state[f"blocks.{layer}.W{role}.weight"] = to_tensor(weight.T)
        state[f"blocks.{layer}.W1.weight"] = to_tensor(our_model.params[prefix + "W1"].data.T)
        state[f"blocks.{layer}.W1.bias"] = to_tensor(our_model.params[prefix + "b1"].data)
        state[f"blocks.{layer}.W2.weight"] = to_tensor(our_model.params[prefix + "W2"].data.T)
        state[f"blocks.{layer}.W2.bias"] = to_tensor(our_model.params[prefix + "b2"].data)
    state["head.weight"] = to_tensor(our_model.params["head_w"].data.T)
    state["head.bias"] = to_tensor(our_model.params["head_b"].data)
    model.load_state_dict(state)
    return model


def train(model, steps, block_size, batch_size, lr, seed=0, log_every=None):
    """训练循环。和 before.py 逐行对应，只有求导那一行换了人。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses = []
    start = time.time()
    for step in range(steps):
        x, y = get_batch(DATA, block_size, batch_size, rng)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()          # 这行在 before.py 里是我们自己写的 backward()
        optimizer.step()
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"  第 {step + 1:>4} 步  loss = {loss.item():.3f}")
    return losses, time.time() - start


@torch.no_grad()
def generate(model, prompt, n_new, temperature=0.8, seed=0):
    """第 23 章的自回归生成：把自己吐出来的字喂回给自己。"""
    generator = torch.Generator().manual_seed(seed)
    ids = [CHAR_TO_ID[ch] for ch in prompt]
    for _ in range(n_new):
        window = torch.tensor([ids[-model.block_size:]])
        logits = model(window)[0, -1] / temperature
        next_id = torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).item()
        ids.append(next_id)
    return "".join(ID_TO_CHAR[i] for i in ids)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main():
    print("=" * 64)
    print(f"语料：{CORPUS}")
    print(f"一共 {len(CORPUS)} 个字，{VOCAB_SIZE} 个不同的字")
    print("=" * 64)

    torch.manual_seed(0)
    model = TorchModel(VOCAB_SIZE, dim=32, n_layer=2, n_head=4, block_size=16)
    print(f"参数量：{model.n_params():,}（before.py 里是 27,492，一样）")
    print()
    print("一、训练 200 步（和 before.py 完全相同的设置）")
    print("-" * 64)
    losses, seconds = train(model, steps=200, block_size=16, batch_size=8, lr=3e-3)
    print(f"用时 {seconds:.2f} 秒（{seconds / 200 * 1000:.1f} 毫秒/步）")
    print("loss 变化：" + "  ".join(f"{losses[i]:.3f}" for i in range(0, 200, 40)))

    print()
    print("二、生成（第 23 章的玩法）")
    print("-" * 64)
    print("提示：床前明月光")
    print("续写：" + generate(model, "床前明月光", n_new=24, temperature=0.5, seed=0))

    print()
    print("三、同一个'做大一点'的模型：dim=128，4 层")
    print("-" * 64)
    torch.manual_seed(0)
    big = TorchModel(VOCAB_SIZE, dim=128, n_layer=4, n_head=4, block_size=32)
    print(f"参数量：{big.n_params():,}（和 before.py 的 800,292 一模一样）")
    before = rss_mb()
    _, seconds = train(big, steps=5, block_size=32, batch_size=16, lr=3e-3)
    after = rss_mb()
    print(f"只跑了 5 步，用时 {seconds:.2f} 秒（{seconds / 5 * 1000:.0f} 毫秒/步）")
    print(f"内存：{before:.0f} MB -> {after:.0f} MB，5 步涨了 {after - before:.0f} MB")


if __name__ == "__main__":
    main()
