"""第 19 章 experiment.py —— 交换和加工，分别是谁在做。

三件事：

    一、在真实语料上把零件拆开：
        只有 attention / 只有 MLP / 两个都有，三种结构跑同一个"挖空还原"任务。
        结果：没有 attention，模型压根看不到别的词。

    二、在探针任务上（每个位置自己两个数做异或）：
        同样的三种结构。只有 attention 的那个停在 50%（瞎猜）。

    三、为什么不平均就没法加工：
        attention 的输出永远精确等于"它这一层那些值的加权平均"，
        堆 8 层也一样。加权平均有一个绕不过去的性质 ——
        结果跑不出被平均的那些数的范围。MLP 没有这个限制。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import Tensor, cross_entropy, embedding, randn, zeros

from after import LR, STEPS, Block, evaluate, train

D_MODEL = 32
CLOZE_STEPS = 400

SENTENCES = [
    "我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机",
    "小王 把 书 给了 小李 因为 他 明天 考试",
]
L = 11


def build_vocab():
    words = []
    for sentence in SENTENCES:
        for word in sentence.split():
            if word not in words:
                words.append(word)
    words.append("<空>")
    words.append("<补>")
    return words


VOCAB = build_vocab()
V = len(VOCAB)
BLANK = VOCAB.index("<空>")
PAD = VOCAB.index("<补>")


def make_cloze_data():
    inputs, targets, positions = [], [], []
    for sentence in SENTENCES:
        words = sentence.split()
        ids = [VOCAB.index(w) for w in words] + [PAD] * (L - len(words))
        for i in range(len(words)):
            masked = list(ids)
            masked[i] = BLANK
            inputs.append(masked)
            targets.append(ids)
            positions.append(i)
    return np.array(inputs), np.array(targets), np.array(positions)


class ClozeModel:
    """词向量 + 位置编码 + 若干块 + 输出层。块里有什么，由开关决定。"""

    def __init__(self, use_attention, use_mlp, seed=0):
        self.block = Block(n_blocks=2, use_attention=use_attention,
                           use_mlp=use_mlp, seed=seed)
        self.p = {
            "tok": randn(V, D_MODEL, scale=1.0, requires_grad=True, seed=seed + 1),
            "pos": randn(L, D_MODEL, scale=1.0, requires_grad=True, seed=seed + 2),
            "Wout": randn(D_MODEL, V, scale=1.0 / np.sqrt(D_MODEL),
                          requires_grad=True, seed=seed + 3),
        }

    def forward(self, tokens):
        x = embedding(self.p["tok"], tokens) + self.p["pos"]
        return self.block.hidden(x) @ self.p["Wout"]

    def params(self):
        return self.block.block_params() + [self.p["tok"], self.p["pos"], self.p["Wout"]]

    def zero_grad(self):
        for p in self.params():
            p.zero_grad()

    def step(self, lr):
        for p in self.params():
            p.data -= lr * p.grad


def run_cloze(use_attention, use_mlp, steps=CLOZE_STEPS, lr=0.05):
    inputs, targets, positions = make_cloze_data()
    model = ClozeModel(use_attention, use_mlp)
    model.zero_grad()
    for _ in range(steps):
        loss = cross_entropy(model.forward(inputs), targets)
        model.zero_grad()
        loss.backward()
        model.step(lr)
    guess = model.forward(inputs).data.argmax(axis=-1)
    hits = int((guess[np.arange(len(positions)), positions]
                == targets[np.arange(len(positions)), positions]).sum())
    return float(loss.data), hits, len(positions)


# ---------------------------------------------------------------- 结构检查


def attention_layer(x, seed, heads=1):
    """一层"标准"的 attention：Q、K 有投影，V 不投影（就直接平均输入的向量）。

    返回：输出的张量、以及这一层用到的权重矩阵。
    """
    batch, length, dim = x.shape
    w_q = randn(dim, dim, scale=1.0 / np.sqrt(dim), seed=seed)
    w_k = randn(dim, dim, scale=1.0 / np.sqrt(dim), seed=seed + 1)
    query = x @ w_q                              # (批, 位置, 维)
    key = x @ w_k
    scores = query @ key.transpose(0, 2, 1) * (1.0 / np.sqrt(dim))
    weights = scores.softmax(axis=-1)            # 每一行加起来等于 1
    values = x                                   # 不投影：被平均的就是输入自己
    out = weights @ values
    return out, weights.data[0]


def mlp(x, seed):
    dim = x.shape[-1]
    w1 = randn(dim, 4 * dim, scale=1.0 / np.sqrt(dim), seed=seed)
    b1 = zeros(4 * dim)
    w2 = randn(4 * dim, dim, scale=1.0 / np.sqrt(dim), seed=seed + 1)
    return (x @ w1 + b1).relu() @ w2


def corpus_inputs():
    """拿真实的那两句话，查出它们的向量（词向量 + 位置编码）。"""
    tokens = [[VOCAB.index(w) for w in s.split()] + [PAD] * (L - len(s.split()))
              for s in SENTENCES]
    tok_table = randn(V, D_MODEL, scale=1.0, seed=17)
    pos_table = randn(L, D_MODEL, scale=1.0, seed=18)
    return embedding(tok_table, np.array(tokens)) + pos_table


def structure_check():
    x = corpus_inputs()
    origin = x.data[0]
    lo, hi = origin.min(axis=0), origin.max(axis=0)

    print("  1) 一层 attention：输出的每一行，是不是「值的加权平均」？")
    out, weights = attention_layer(x, seed=21)
    print(f"     权重矩阵：每行求和 = {weights.sum(axis=-1).min():.6f} ~ "
          f"{weights.sum(axis=-1).max():.6f}，最小权重 = {weights.min():.6f}")
    manual = weights @ origin
    print(f"     手算的平均 vs 库算的输出，最大误差 = {np.abs(manual - out.data[0]).max():.2e}")

    print("\n  2) 堆 8 层：每一层都还是「平均」，只是平均的对象换了")
    total = np.eye(L)
    cur = x
    worst = 0.0
    for layer in range(8):
        lo_cur, hi_cur = cur.data[0].min(axis=0), cur.data[0].max(axis=0)
        out, weights = attention_layer(cur, seed=30 + layer)
        worst = max(worst,
                    float(np.max(np.maximum(0, out.data[0] - hi_cur))),
                    float(np.max(np.maximum(0, lo_cur - out.data[0]))))
        total = weights @ total
        cur = out
    print(f"     8 层的权重连乘之后：每行求和 = {total.sum(axis=-1).min():.6f} ~ "
          f"{total.sum(axis=-1).max():.6f}，最小权重 = {total.min():.6f}")
    print(f"     8 层输出 vs 输入的一次加权平均：最大误差 = "
          f"{np.abs(cur.data[0] - total @ origin).max():.2e}")
    print(f"     8 层里，输出跑出「这一层的值的范围」的最大越界量 = {worst:.10f}")

    print("\n  3) 换成 MLP：输出可以跑到输入范围外面去")
    out_mlp = mlp(Tensor(origin[None]), seed=41)
    over = max(float(np.max(np.maximum(0, out_mlp.data[0] - hi))),
               float(np.max(np.maximum(0, lo - out_mlp.data[0]))))
    print(f"     MLP 输出的最大越界量 = {over:.6f}")
    print(f"     （输入的第 0 维范围是 [{lo[0]:+.3f}, {hi[0]:+.3f}]，"
          f"MLP 输出是 [{out_mlp.data[0][:, 0].min():+.3f}, "
          f"{out_mlp.data[0][:, 0].max():+.3f}]）")


if __name__ == "__main__":
    print("=" * 74)
    print("一、真实语料上的任务：把被挖空的词填回来（2 个块，400 步）")
    print("=" * 74)
    print(f"  {'结构':<20}{'loss':>10}{'被挖空的位置填对':>18}")
    print("  " + "-" * 46)
    for use_attention, use_mlp, tag in ((False, True, "只有 MLP"),
                                        (True, False, "只有 Attention"),
                                        (True, True, "Attention + MLP")):
        loss, hits, total = run_cloze(use_attention, use_mlp)
        print(f"  {tag:<20}{loss:>10.4f}{str(hits) + '/' + str(total):>18}")

    print()
    print("=" * 74)
    print("二、探针任务：每个位置自己两个数做异或（纯粹考「加工」）")
    print("=" * 74)
    print(f"  {'结构':<20}{'训练 loss':>12}{'测试 loss':>12}{'测试准确率':>12}")
    print("  " + "-" * 56)
    for use_attention, use_mlp, tag in ((False, True, "只有 MLP"),
                                        (True, False, "只有 Attention"),
                                        (True, True, "Attention + MLP")):
        model = Block(use_attention=use_attention, use_mlp=use_mlp)
        history = train(model, steps=STEPS, lr=LR, verbose=False)
        loss, acc = evaluate(model)
        print(f"  {tag:<20}{history[-1]:>12.4f}{loss:>12.4f}{acc * 100:>11.1f}%")
    print(f"\n  （瞎猜的准确率是 50.0%，瞎猜的 loss 是 {np.log(2):.4f}）")

    print()
    print("=" * 74)
    print("三、attention 到底做了什么：把真实语料喂进去，一层层看")
    print("=" * 74)
    structure_check()
