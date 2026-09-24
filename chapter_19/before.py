"""第 19 章 before.py —— 第 18 章的模型，在"要加工"的任务上：学不会。

这个文件演示什么：

    第 17 章加了残差，第 18 章加了归一化，模型终于能稳稳地堆起来了。
    但它从头到尾只做一件事：把别的位置的信息取过来、混一混。

        x = x + attention(层归一化(x))

    本章的问题是：取过来之后，谁来"加工"？

    我们拿一个探针任务来问这件事（不是为了跑真实语料，是为了把"交换"和"加工"分开看）：

        每个位置给一个 32 维向量（随机、已标准化），
        要模型输出一个 0/1：这个位置自己的前两个数，异或一下
        （一个正一个负就是 1，同号就是 0）。

    答案只取决于这个位置自己，不需要任何"交换"，纯粹考"加工"。

    结果：这个只剩 attention 的模型，900 步跑下来准确率还是 50% 上下 ——
    它连"自己身上的两个数做一次异或"都做不到。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import Tensor, cross_entropy, layer_norm, randn, zeros

L = 11          # 序列长度：还是那两句话的长度
D_MODEL = 32
N_HEADS = 2
N_BLOCKS = 2
STEPS = 900
LR = 0.05
BATCH = 32
TRAIN_SIZE = 512
TEST_SIZE = 256


def make_probe(n, seed):
    """造一批探针数据：每个位置一个 32 维向量，标签是它自己的前两个数的异或。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, L, D_MODEL))
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-5)
    y = ((x[:, :, 0] > 0) ^ (x[:, :, 1] > 0)).astype(int)
    return x, y


TRAIN_X, TRAIN_Y = make_probe(TRAIN_SIZE, seed=1)
TEST_X, TEST_Y = make_probe(TEST_SIZE, seed=2)


class Block:
    """第 18 章的块：只有 Attention。

        x = x + attention(LN(x))
    """

    def __init__(self, n_blocks=N_BLOCKS, d_model=D_MODEL, n_heads=N_HEADS, seed=0):
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.p = {}
        s = 1.0 / np.sqrt(d_model)
        self.p["Wout"] = randn(d_model, 2, scale=0.3, requires_grad=True, seed=seed + 999)
        self.p["bout"] = zeros(2, requires_grad=True)
        self.blocks = []
        for i in range(n_blocks):
            base = seed + 100 * (i + 1)
            block = {
                "Wq": randn(d_model, d_model, scale=s, requires_grad=True, seed=base + 1),
                "Wk": randn(d_model, d_model, scale=s, requires_grad=True, seed=base + 2),
                "Wv": randn(d_model, d_model, scale=s, requires_grad=True, seed=base + 3),
                "Wo": randn(d_model, d_model, scale=s, requires_grad=True, seed=base + 4),
                "ln1w": Tensor(np.ones(d_model), requires_grad=True),
                "ln1b": Tensor(np.zeros(d_model), requires_grad=True),
                "ln2w": Tensor(np.ones(d_model), requires_grad=True),
                "ln2b": Tensor(np.zeros(d_model), requires_grad=True),
            }
            self.blocks.append(block)

    def params(self):
        out = [self.p["Wout"], self.p["bout"]]
        for block in self.blocks:
            out.extend(block.values())
        return out

    def attention(self, x, block):
        batch, length, dim = x.shape
        heads, head_dim = self.n_heads, self.head_dim

        def split_heads(t):
            return t.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)

        query = split_heads(x @ block["Wq"])
        key = split_heads(x @ block["Wk"])
        value = split_heads(x @ block["Wv"])
        scores = query @ key.transpose(0, 1, 3, 2) * (1.0 / np.sqrt(head_dim))
        out = (scores.softmax(axis=-1) @ value)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, dim)
        return out @ block["Wo"]

    def hidden(self, x_in):
        """把输入过一遍所有的块，拿回来的还是 (批大小, 位置数, 32) 这样一堆向量。"""
        x = x_in if isinstance(x_in, Tensor) else Tensor(x_in)
        for block in self.blocks:
            # 交换：这个位置去别的位置那里取信息
            x = x + self.attention(layer_norm(x, block["ln1w"], block["ln1b"]), block)
        return x

    def forward(self, x_in):
        """探针任务用：把每个位置的向量变成 2 个分数（0 / 1 两类）。"""
        return self.hidden(x_in) @ self.p["Wout"] + self.p["bout"]

    def block_params(self):
        out = []
        for block in self.blocks:
            out.extend(block.values())
        return out

    def zero_grad(self):
        for p in self.params():
            p.zero_grad()

    def step(self, lr):
        for p in self.params():
            p.data -= lr * p.grad


def train(model, steps=STEPS, lr=LR, rng_seed=7, verbose=True):
    rng = np.random.default_rng(rng_seed)
    model.zero_grad()
    history = []
    for step in range(steps):
        idx = rng.integers(0, TRAIN_SIZE, BATCH)
        logits = model.forward(TRAIN_X[idx])
        loss = cross_entropy(logits, TRAIN_Y[idx])
        model.zero_grad()
        loss.backward()
        model.step(lr)
        history.append(float(loss.data))
        if verbose and (step % 150 == 0 or step == steps - 1):
            print(f"  第 {step:3d} 步    loss = {loss.data:.4f}")
    return history


def evaluate(model):
    logits = model.forward(TEST_X)
    loss = float(cross_entropy(logits, TEST_Y).data)
    acc = float((logits.data.argmax(axis=-1) == TEST_Y).mean())
    return loss, acc


if __name__ == "__main__":
    print("=" * 62)
    print("第 18 章的块：只有 Attention，没有「加工」的地方")
    print("=" * 62)
    print("""
        输入（每个位置一个 32 维向量）
          │
          ├──────────────┐
          ↓              │
       层归一化           │
          ↓              │
       Attention（交换）   │    ← 从别的 position 取信息
          ↓              │
          + ←────────────┘
          │
          ↓
        输出
""")
    model = Block()
    train(model)
    loss, acc = evaluate(model)
    print(f"\n  测试集：loss = {loss:.4f}   准确率 = {acc * 100:.1f}%")
    print(f"  （瞎猜是 50%，loss 是 {np.log(2):.4f}）")
