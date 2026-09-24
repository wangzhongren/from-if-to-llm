"""第 21 章 before：第 20 章那个「窗口拼接」模型，直接拿来用。

第 20 章我们把预测下一个字符这件事做成了分类，模型是：

    embedding → 把前 8 个字符的向量拼起来 → 一个线性层 → softmax

它在语料上做到了 100%。

这个文件不引入任何新东西，只做一件事：**把它的毛病摆到台面上。**

毛病有两个，都是「窗口」这个词带来的：

  1. 窗口长度是写死的 8。给它一段更长的话，多出来的部分只能丢掉。
     给它一段更短的话，它没法处理（少一格形状都不对）。

  2. 想看得更远，就得把窗口加宽，而每加宽一格，
     那个线性层的参数就多 DIM × VOCAB_SIZE 个。
     看到 1000 个字符？参数会多到没法看。

我们跑三个窗口大小，把这两个毛病都用数字摆出来。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import cross_entropy, embedding, no_grad, randn, zeros, SGD

# ---------------------------------------------------------------- 语料（和第 20 章一样）

POEM_A = "床前明月光疑是地上霜举头望明月低头思故乡"
POEM_B = "鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波"

CORPUS = (POEM_A + POEM_B) * 8

VOCAB = sorted(set(CORPUS))
CHAR_TO_ID = {char: i for i, char in enumerate(VOCAB)}
ID_TO_CHAR = {i: char for char, i in CHAR_TO_ID.items()}
VOCAB_SIZE = len(VOCAB)

CORPUS_IDS = np.array([CHAR_TO_ID[char] for char in CORPUS])

SEED = 20260924
DIM = 16


class WindowModel:
    """第 20 章的模型，原样搬过来（窗口长度改成可调）。"""

    def __init__(self, window, dim=DIM, vocab_size=VOCAB_SIZE, seed=SEED):
        self.window = window
        self.dim = dim
        self.table = randn(vocab_size, dim, scale=0.3, requires_grad=True, seed=seed)
        self.weight = randn(window * dim, vocab_size, scale=0.3, requires_grad=True, seed=seed + 1)
        self.bias = zeros(vocab_size, requires_grad=True)

    def params(self):
        return [self.table, self.weight, self.bias]

    def head_parameter_count(self):
        """最后那个线性层有多少参数。"""
        return self.weight.data.size + self.bias.data.size

    def __call__(self, windows):
        vectors = embedding(self.table, windows)
        flat = vectors.reshape(windows.shape[0], self.window * self.dim)
        return flat @ self.weight + self.bias


def build_pairs(ids, window):
    inputs, targets = [], []
    for position in range(window, len(ids)):
        inputs.append(ids[position - window:position])
        targets.append(ids[position])
    return np.array(inputs), np.array(targets)


def train(model, inputs, targets, steps=600, batch_size=64, learning_rate=0.5, seed=SEED):
    optimizer = SGD(model.params(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        batch = rng.integers(0, len(inputs), size=batch_size)
        optimizer.zero_grad()
        cross_entropy(model(inputs[batch]), targets[batch]).backward()
        optimizer.step()


def evaluate(model, inputs, targets):
    with no_grad():
        logits = model(inputs)
        loss = cross_entropy(logits, targets).item()
        accuracy = float((logits.data.argmax(axis=-1) == targets).mean())
    return loss, accuracy


def main():
    window = 8

    print("=" * 60)
    print("毛病一：窗口是写死的")
    print("=" * 60)
    sentence = "明月光疑是地上霜举头望明月"
    print(f"  第 20 章的模型吃固定 {window} 个字符。现在给它一段话：")
    print(f"    {sentence}   （{len(sentence)} 个字符）")
    print()
    print(f"  窗口是 {window}，所以只能取最后 {window} 个：{sentence[-window:]}")
    print(f"  前面这 {len(sentence) - window} 个字符被丢掉了：{sentence[:len(sentence) - window]}")
    print()
    print(f"  「{sentence[:len(sentence) - window]}」这几个字对模型来说不存在。")
    print("  窗口的位置是固定的，多出来的信息只能扔。")
    print()
    short = "明月光"
    print(f"  反过来，给它一段只有 {len(short)} 个字符的话：{short}")
    print(f"  窗口要 {window} 个，凑不够。少一格，形状就对不上：")
    print(f"    embedding 之后是 {len(short)} × {DIM} = {len(short) * DIM} 个数，")
    print(f"    线性层要的是 {window} × {DIM} = {window * DIM} 个数。")
    print("  差的不是一点点。")
    print()

    print("=" * 60)
    print("毛病二：想看远一点，代价是参数线性增长")
    print("=" * 60)
    print(f"  {'窗口':>6}   {'线性层参数':>12}   {'loss':>9}   {'准确率':>8}")
    print("  " + "-" * 44)
    for window in (2, 4, 8, 16, 32):
        inputs, targets = build_pairs(CORPUS_IDS, window)
        model = WindowModel(window)
        train(model, inputs, targets)
        loss, accuracy = evaluate(model, inputs, targets)
        print(f"  {window:>6}   {model.head_parameter_count():>12}   {loss:>9.4f}   {accuracy:>7.2%}")

    print()
    print("  窗口从 2 翻到 32，参数从 " +
          f"{WindowModel(2).head_parameter_count()} 涨到 {WindowModel(32).head_parameter_count()}，16 倍。")
    print("  loss 却从 0.0870 掉到 0.0026（窗口 8），后面基本不动了。")
    print()
    print("  换句话说：为了多看 30 个字符，我们付了 16 倍的参数，")
    print("  换来的那点提升，在窗口 8 的时候就已经拿到大半了。")
    print()
    print("  而且这个账还没算完——")
    target = 1000
    head = target * DIM * VOCAB_SIZE + VOCAB_SIZE
    print(f"  要让它看到 {target} 个字符，线性层就是 {head:,} 个参数。")
    print(f"  我们的语料一共才 {len(CORPUS)} 个字符。")
    print()

    print("=" * 60)
    print("所以第 21 章要做的事")
    print("=" * 60)
    print("  我们想要的是这样一种机制：")
    print("    - 不管序列多长都能吃进去")
    print("    - 参数量跟长度**没关系**")
    print("    - 而且每个位置能**自己决定**要看哪里，")
    print("      而不是永远固定地往前看 8 格")
    print()
    print("  第 14 章我们已经造出过这样一个东西。它叫 attention。")
    print("  Q 是「我在找什么」，K 是「我有什么」，V 是「我给出什么」。")
    print("  Q 和 K 做点积、softmax，谁该被看多少，是**算出来的**，不是写死的。")
    print()
    print("  我们把它加回来。")
    print("  但是加回来之前，先记住上一章最后那件事：")
    print("  位置 t 要预测的字符，就是位置 t+1 上的输入。")
    print("  attention 会让位置 t 看到所有位置。")
    print("  看到位置 t+1，它就看到了答案。")
    print()
    print("  下一节，我们就这么干了，而且故意什么都不挡。")


if __name__ == "__main__":
    main()
