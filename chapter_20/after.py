"""第 20 章 after：把分类器重新拿回来，让它预测下一个字符。

上一章（before.py）我们把第 3 章的加权分数分类器原样搬了过来：
只能看当前这一个字符，准确率 92%。

这一章我们改两件事：

  1. 给每个字符一个**向量**，而不是一行分数（第 10 章的 embedding）。
  2. 让它不只看到当前字符，还看到**前面 W 个字符**（第 12 章的上下文）。

结构是全书最简单的一个：

    embedding → 把窗口里每个字符的向量拼起来 → 一个线性层 → softmax → 交叉熵

没有 attention，没有 MLP，没有残差。就这么多。

跑完你会看到一个比 92% 好得多的数字。
但最后我们会把一个疑点记下来，留给下一章。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import Tensor, cross_entropy, embedding, no_grad, randn, zeros, SGD

# ---------------------------------------------------------------- 语料

POEM_A = "床前明月光疑是地上霜举头望明月低头思故乡"
POEM_B = "鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波"

CORPUS = (POEM_A + POEM_B) * 8

VOCAB = sorted(set(CORPUS))
CHAR_TO_ID = {char: i for i, char in enumerate(VOCAB)}
ID_TO_CHAR = {i: char for char, i in CHAR_TO_ID.items()}
VOCAB_SIZE = len(VOCAB)

CORPUS_IDS = np.array([CHAR_TO_ID[char] for char in CORPUS])

SEED = 20260924
DIM = 16        # 每个字符用几个数字表示
WINDOW = 8      # 往前看几个字符


# ---------------------------------------------------------------- 模型


class NextCharModel:
    """embedding → 拼接窗口 → 一个线性层 → 下一个字符的分数。

    embedding 表是第 10 章的东西：把「第几号字符」变成「一组数字」。
    拼接是这一章的动作：窗口里有 8 个字符，就把 8 组数字首尾接起来，
    变成一个长向量（第 7 章加中间层的时候我们也干过类似的事）。

    线性层是第 3 章的东西：给一个向量，算出每个候选的分数。
    """

    def __init__(self, vocab_size, dim=DIM, window=WINDOW, seed=SEED):
        self.dim = dim
        self.window = window
        self.table = randn(vocab_size, dim, scale=0.3, requires_grad=True, seed=seed)
        self.weight = randn(window * dim, vocab_size, scale=0.3, requires_grad=True, seed=seed + 1)
        self.bias = zeros(vocab_size, requires_grad=True)

    def params(self):
        return [self.table, self.weight, self.bias]

    def __call__(self, windows):
        """windows 形状 (B, 窗口长度)，返回 (B, 词表大小) 的分数。"""
        vectors = embedding(self.table, windows)          # (B, 窗口, 维度)
        flat = vectors.reshape(windows.shape[0], self.window * self.dim)
        return flat @ self.weight + self.bias

    def parameter_count(self):
        return sum(p.data.size for p in self.params())


# ---------------------------------------------------------------- 数据与训练


def build_pairs(ids, window=WINDOW):
    """切出 (前面 window 个字符, 下一个字符) 对。

    第 position 个样本：输入 ids[position-window : position]，答案 ids[position]。
    输入是 [position-window, position) 这一段，答案在 position 上。

    答案**不在**窗口里。这一点下一章会变得非常重要。
    """
    inputs, targets = [], []
    for position in range(window, len(ids)):
        inputs.append(ids[position - window:position])
        targets.append(ids[position])
    return np.array(inputs), np.array(targets)


def train(model, inputs, targets, steps=600, batch_size=64, learning_rate=0.5, seed=SEED):
    optimizer = SGD(model.params(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        batch = rng.integers(0, len(inputs), size=batch_size)
        optimizer.zero_grad()
        loss = cross_entropy(model(inputs[batch]), targets[batch])
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            history.append((step + 1, loss.item()))
    return history


def evaluate(model, inputs, targets):
    with no_grad():
        logits = model(inputs)
        loss = cross_entropy(logits, targets).item()
        accuracy = float((logits.data.argmax(axis=-1) == targets).mean())
    return loss, accuracy


def predict_next(model, prefix):
    """给一段文字，返回下一个字符的概率分布（一个长度为词表大小的数组）。

    这个模型只吃固定长度的窗口，所以 prefix 至少要有一个窗口那么长。
    """
    if len(prefix) < model.window:
        raise ValueError(f"prefix 至少要有 {model.window} 个字符，现在只有 {len(prefix)} 个")
    ids = np.array([[CHAR_TO_ID[char] for char in prefix[-model.window:]]])
    with no_grad():
        logits = model(ids)
        probabilities = logits.softmax(axis=-1).data[0]
    return probabilities


def show_candidates(model, prefix, top=5):
    probabilities = predict_next(model, prefix)
    order = np.argsort(-probabilities)[:top]
    print(f"  '{prefix}' 后面，下一个字符的候选：")
    for rank, index in enumerate(order, start=1):
        bar = "#" * int(probabilities[index] * 40)
        print(f"    {rank}. '{ID_TO_CHAR[index]}'  {probabilities[index]:.4f}  {bar}")


# ---------------------------------------------------------------- 主流程


def main():
    print("=" * 60)
    print("语料")
    print("=" * 60)
    print(f"  {POEM_A}")
    print(f"  {POEM_B}")
    print(f"  重复 8 遍，一共 {len(CORPUS)} 个字符，{VOCAB_SIZE} 个不同的字符")

    inputs, targets = build_pairs(CORPUS_IDS)
    print(f"\n  切成 {len(inputs)} 个 (前 {WINDOW} 个字符, 下一个字符) 对")

    model = NextCharModel(VOCAB_SIZE)
    print(f"  模型参数一共 {model.parameter_count()} 个"
          f"（其中 {model.weight.data.size} 个在最后那个线性层里）")

    history = train(model, inputs, targets)

    print()
    print("=" * 60)
    print("训练过程")
    print("=" * 60)
    for step, loss in history:
        print(f"  第 {step:>3} 步   loss {loss:.4f}")

    loss, accuracy = evaluate(model, inputs, targets)
    print(f"\n  训练集上的 loss     {loss:.4f}")
    print(f"  训练集上的准确率    {accuracy:.2%}")
    print(f"  对比：before.py 里那个只看一个字符的分类器是 92.23%")

    print()
    print("=" * 60)
    print("它现在能回答了：下一个字符是哪一个，各自多大概率")
    print("=" * 60)
    print("  如果词表是几万个词，这一步的输出长这样：")
    print("    '今天天气很' -> 好 0.52   冷 0.21   热 0.12   苹果 0.0001")
    print("  我们的词表只有 33 个字符，所以长这样：")
    print()
    for prefix in ["床前明月光疑是地", "白毛浮绿水红掌拨"]:
        show_candidates(model, prefix)
        print()

    print("  再看一组。下面这两个窗口的**最后一个字符都是 '头'**：")
    print()
    for prefix in ["光疑是地上霜举头", "霜举头望明月低头"]:
        show_candidates(model, prefix)
        print()
    print("  同样一个 '头'，上一个字是 '举' 就接 '望'，上一个字是 '低' 就接 '思'。")
    print("  只看一个字符的那个分类器，在这一点上永远是错的。")

    print("=" * 60)
    print("还留着的一个疑点")
    print("=" * 60)
    print("  我们这个模型的输入是语料的 [t-8, t)，答案是 t 位置上的字符。")
    print("  答案不在输入里。这 600 步训练，它一直是诚实的。")
    print()
    print("  但是回头看看我们的训练数据是怎么切的：")
    print("    输入：ids[t-8 : t]     答案：ids[t]")
    print("  如果换成「一次喂进整段语料、让每个位置都输出下一个字符」，")
    print("  那就变成：输入 ids[t : t+T]，答案 ids[t+1 : t+T+1]。")
    print("  对窗口里的第 i 个位置来说，它要预测的是 ids[t+i+1]，")
    print("  而这个字符，正好是窗口里第 i+1 个位置上的**输入**。")
    print()
    print("  现在这个模型只看固定 8 个字符的窗口，看不到那一位，")
    print("  所以它还利用不了这个漏洞。")
    print("  可下一章我们要把 attention 加回来 ——")
    print("  attention 干的事情，正是「让每个位置看到所有位置」。")
    print("  位置 t 会看到位置 t+1。它看到的就是答案。")
    print()
    print("  那时候会发生什么？")


if __name__ == "__main__":
    main()
