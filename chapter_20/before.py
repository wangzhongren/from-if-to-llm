"""第 20 章 before：把第 3 章那个加权分数分类器，原样搬到「预测下一个字符」上。

第 3 章我们做的是这样一件事：

    if "手机" in words:
        score_科技 += 1.2

改写成加权求和：

    score = w · x

那一章只有两个类别：科技、食品。

这一章我们要预测的东西是「下一个字符是哪一个」——
我们的语料里有 33 个不同的字符，所以是 33 分类。

分类器还是那个分类器，只是类别从 2 个变成了 33 个：

    score = W[当前字符]

给每个字符配一行分数，softmax 之后就是概率。

这个文件就是要看看：**把分类器原样搬过来，够不够用。**
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import Tensor, cross_entropy, no_grad, randn, SGD

# ---------------------------------------------------------------- 语料

POEM_A = "床前明月光疑是地上霜举头望明月低头思故乡"
POEM_B = "鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波"

# 两首诗加起来只有 38 个字，太短了，喂不饱一个模型。
# 我们把它重复 8 遍拼成一段 304 个字的语料。
# （重复的后果，第 24 章会专门回来看。）
CORPUS = (POEM_A + POEM_B) * 8

VOCAB = sorted(set(CORPUS))          # 33 个字符
CHAR_TO_ID = {char: i for i, char in enumerate(VOCAB)}
ID_TO_CHAR = {i: char for char, i in CHAR_TO_ID.items()}
VOCAB_SIZE = len(VOCAB)

CORPUS_IDS = np.array([CHAR_TO_ID[char] for char in CORPUS])

SEED = 20260924


# ---------------------------------------------------------------- 模型


class ScoreTable:
    """第 3 章的加权分数分类器，33 个类别。

    输入是「当前这个字符是谁」，输出是「下一个字符的分数」。

    做法和第 3 章一模一样：给每一种输入配一组分数。
    33 种输入 × 33 个类别 = 一张 33×33 的表。
    """

    def __init__(self, vocab_size, seed=SEED):
        # 表里每一行，就是「看到这个字符时，下一个字符各自得多少分」
        self.scores = randn(vocab_size, vocab_size, scale=0.3, requires_grad=True, seed=seed)

    def params(self):
        return [self.scores]

    def __call__(self, current_ids):
        # 查表：current_ids 形状 (B,)，输出 (B, 词表大小)
        return self.scores[current_ids]


# ---------------------------------------------------------------- 训练


START = 8   # 从第 8 个字符开始切，好和 after.py 评估在同一批位置上


def build_pairs(ids, start=START):
    """把语料切成一堆 (当前字符, 下一个字符) 对。"""
    inputs, targets = [], []
    for position in range(start, len(ids)):
        inputs.append(ids[position - 1])
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


def main():
    print("=" * 60)
    print("语料")
    print("=" * 60)
    print(f"  {POEM_A}")
    print(f"  {POEM_B}")
    print(f"  重复 8 遍，一共 {len(CORPUS)} 个字符，{VOCAB_SIZE} 个不同的字符")
    print(f"  词表：{''.join(VOCAB)}")

    inputs, targets = build_pairs(CORPUS_IDS)
    print(f"\n  切成 {len(inputs)} 个 (输入字符, 下一个字符) 对")

    model = ScoreTable(VOCAB_SIZE)
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

    print()
    print("=" * 60)
    print("它答错了哪些地方")
    print("=" * 60)
    with no_grad():
        predictions = model(inputs).data.argmax(axis=-1)
    wrong = np.nonzero(predictions != targets)[0]
    # 同一个错误会重复出现很多次（语料重复了 8 遍），这里只列不同的
    seen = set()
    for index in wrong:
        key = (ID_TO_CHAR[inputs[index]], ID_TO_CHAR[targets[index]], ID_TO_CHAR[predictions[index]])
        if key in seen:
            continue
        seen.add(key)
        print(f"  语料里的 '{key[0]}' 后面跟着 '{key[1]}'，它却猜了 '{key[2]}'")

    print()
    print("=" * 60)
    print("为什么会答错")
    print("=" * 60)
    print("  '头' 在语料里出现了两次，后面跟着的字不一样：")
    print("    ...举头望明月...      '头' 后面是 '望'")
    print("    ...低头思故乡...      '头' 后面是 '思'")
    print("  这个分类器只能看到 '头' 这一个字符，")
    print("  它没有任何办法知道这一次的 '头' 是哪一个 '头'。")
    print()
    print("  '月' 和 '鹅' 也是一样：")
    print("    ...床前明月光...      '月' 后面是 '光'")
    print("    ...举头望明月...      '月' 后面是 '低'")
    print("    ...故乡鹅鹅鹅曲项...   '鹅' 后面还是 '鹅'")
    print("    ...鹅鹅鹅曲项向天...   '鹅' 后面是 '曲'")
    print()
    print("  换句话说：它只知道「这个字符后面最常跟什么」，")
    print("  不知道「这句话现在说到哪了」。")


if __name__ == "__main__":
    main()
