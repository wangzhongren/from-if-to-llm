"""第 12 章 after.py —— 一个词不能只有一种意思。

第 11 章我们训练出了一张词向量表。苹果 拿到了一个向量 —— 只有一个。

可语料里 苹果 明明有两种用法：

    苹果 很 好吃          这里说的是水果
    苹果 发布 新 手机      这里说的是公司

两种情况拿到的是同一组数字，模型没有任何办法把它们分开。

这一章的做法很笨：**不用 苹果 自己的向量，改用 苹果 周围那些词的向量。**

    苹果 很 好吃          → 用 (很 + 好吃) / 2
    苹果 发布 新 手机      → 用 (发布 + 新 + 手机) / 3

一个词所在位置周围的那些词，我们叫它**上下文**。
把它们平均起来得到的那个向量，叫**上下文向量**。

这个做法笨归笨，但它第一次让"同一个词在不同句子里有不同的表示"成为可能。
跑完你也会看到它笨在哪里 —— 那正是下一章要接着走的路。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import SGD, cross_entropy, embedding, randn, zeros

# 第 11 章那 13 句短句，加上第 13-16 章要用的那两句长句，再加上几句支撑句。
# 支撑句是为了让长句里的新词（我、昨天、商场、小王、书……）
# 也有足够多的地方出现，不然它们的向量训练不出来。
SENTENCES = [
    # ---- 第 11 章那批短句（16 个词）
    "苹果 发布 新 手机",
    "苹果 发布 新 芯片",
    "华为 发布 新 电脑",
    "小米 发布 新 手机",
    "苹果 发布 新 电脑",
    "华为 发布 新 手机",
    "苹果 很 好吃",
    "苹果 很 甜",
    "香蕉 很 好吃",
    "这个 苹果 真 甜",
    "苹果 做成 派",
    "香蕉 做成 派",
    "这个 香蕉 真 好吃",
    # ---- 两条长句
    "我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机",
    "小王 把 书 给了 小李 因为 他 明天 考试",
    # ---- 支撑句
    "我 昨天 在 商场 看到 手机",
    "我 昨天 在 商场 看到 香蕉",
    "小王 昨天 在 商场 看到 苹果",
    "小李 明天 在 商场 看到 新 手机",
    "我 把 手机 给了 小王",
    "小王 把 手机 给了 小李",
    "他 昨天 看到 苹果 发布 的 新 手机",
    "我 明天 看到 小李",
    "苹果 刚刚 发布 的 新 手机",
    "小李 昨天 看到 了 苹果",
]

# 词表直接从语料里取，不手写，免得漏词
VOCAB = sorted({word for sentence in SENTENCES for word in sentence.split()})

DIM = 16
TABLE_SEED = 17
OUT_SEED = 105
EPOCHS = 600
LR = 0.3

# 本章反复要用的四句话
FOOD_SENTENCE = "苹果 很 好吃"
TECH_SENTENCE = "苹果 发布 新 手机"
LONG_SENTENCE = "我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机"


def display_width(text):
    """中文在终端里占两格，英文数字占一格。"""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def pad(text, width):
    return text + " " * (width - display_width(text))


def encode(sentences):
    word_to_id = {word: i for i, word in enumerate(VOCAB)}
    return [[word_to_id[word] for word in sentence.split()] for sentence in sentences]


def build_training_pairs(corpus, window=1):
    """和第 11 章一模一样：每个词都去猜它左右的邻居。"""
    seen, guessed = [], []
    for sentence in corpus:
        for position in range(len(sentence)):
            left = max(0, position - window)
            right = min(len(sentence), position + window + 1)
            for neighbour in range(left, right):
                if neighbour == position:
                    continue
                seen.append(sentence[neighbour])
                guessed.append(sentence[position])
    return np.array(seen), np.array(guessed)


def train_embeddings(seen, guessed, epochs=EPOCHS, lr=LR, seed=TABLE_SEED):
    """和第 11 章一模一样。训练出来的还是一张「一个词一个向量」的表。"""
    table = randn(len(VOCAB), DIM, scale=0.1, requires_grad=True, seed=seed)
    out_weight = randn(DIM, len(VOCAB), scale=0.1, requires_grad=True, seed=OUT_SEED)
    out_bias = zeros(len(VOCAB), requires_grad=True)
    optimizer = SGD([table, out_weight, out_bias], lr=lr)

    loss = None
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = embedding(table, seen) @ out_weight + out_bias
        loss = cross_entropy(logits, guessed)
        loss.backward()
        optimizer.step()
    return table, loss.item()


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def context_vector(table_data, sentence, position):
    """本章的新东西：一个位置的表示 = 句中**其它所有词**的向量的平均。

    注意 "其它所有词"：不区分远近，也不区分谁重要。
    这就是本章说的"最笨方案"。
    """
    ids = [VOCAB.index(word) for word in sentence.split()]
    others = [ids[i] for i in range(len(ids)) if i != position]
    return table_data[others].mean(axis=0)


def most_similar(table_data, vector, top=6):
    """这个向量最像哪几个词。"""
    scored = [(VOCAB[i], cosine(vector, table_data[i])) for i in range(len(VOCAB))]
    scored.sort(key=lambda pair: -pair[1])
    return scored[:top]


def show_neighbours(title, table_data, vector, top=6):
    pairs = "  ".join(f"{word} {score:+.2f}" for word, score in most_similar(table_data, vector, top))
    print(f"  {pad(title, 22)} → {pairs}")


def main():
    corpus = encode(SENTENCES)
    seen, guessed = build_training_pairs(corpus)
    table, loss = train_embeddings(seen, guessed)

    print("=" * 66)
    print("先按第 11 章的办法，训练一张「一个词一个向量」的表")
    print("=" * 66)
    print(f"  句子 {len(SENTENCES)} 条，词 {len(VOCAB)} 个，维度 {DIM}")
    print(f"  训练样本 {len(seen)} 条，训练 {EPOCHS} 轮之后的 loss = {loss:.4f}")
    print()

    apple = VOCAB.index("苹果")
    print(f"  苹果 的向量（只有这一个）：")
    print("    [" + "  ".join(f"{v:+.2f}" for v in table.data[apple]) + "]")
    print()

    # ---------------------------------------------------------- 上下文化

    print("=" * 66)
    print("同一个 苹果，在两个句子里给它算两个不同的表示")
    print("=" * 66)
    print(f"  A 句：{FOOD_SENTENCE}")
    print(f"  B 句：{TECH_SENTENCE}")
    print()

    vector_a = context_vector(table.data, FOOD_SENTENCE, 0)
    vector_b = context_vector(table.data, TECH_SENTENCE, 0)

    print("  A 句里 苹果 的上下文向量 = (很 + 好吃) / 2")
    print("    [" + "  ".join(f"{v:+.2f}" for v in vector_a) + "]")
    print("  B 句里 苹果 的上下文向量 = (发布 + 新 + 手机) / 3")
    print("    [" + "  ".join(f"{v:+.2f}" for v in vector_b) + "]")
    print()
    print(f"  两个向量的余弦相似度 = {cosine(vector_a, vector_b):+.4f}")
    print(f"  而两句话里的 苹果 自己的那个向量，余弦相似度 = {cosine(table.data[apple], table.data[apple]):.4f}")
    print()

    show_neighbours("A 句里 苹果 最像", table.data, vector_a)
    show_neighbours("B 句里 苹果 最像", table.data, vector_b)
    print()

    # ---------------------------------------------------------- 平均的粗糙

    print("=" * 66)
    print("但是「平均」这件事很粗糙。看长句：")
    print("=" * 66)
    print(f"  {LONG_SENTENCE}")
    print()

    words = LONG_SENTENCE.split()
    position = words.index("苹果")
    others = [word for i, word in enumerate(words) if i != position]

    print(f"  句中除 苹果 之外还有 {len(others)} 个词：{' '.join(others)}")
    print("  它们的权重是这样的：")
    weight = 1.0 / len(others)
    print("    " + "  ".join(f"{word} {weight:.2f}" for word in others))
    print()
    print("  每个词的权重都一样。「昨天」和「发布」各占一份，一模一样。")
    print()

    long_vector = context_vector(table.data, LONG_SENTENCE, position)
    show_neighbours("全句平均 最像", table.data, long_vector, top=8)
    print()
    print("  排第一的是「的」。")
    print()

    print("  再做一个小实验：把句中的某个词拿掉，看 苹果 的上下文向量变了多少。")
    print()
    other_ids = [VOCAB.index(word) for i, word in enumerate(words) if i != position]
    print("    拿掉的词    变化量")
    print("    " + "-" * 22)
    changes = {}
    for k, word in enumerate(others):
        kept = [word_id for j, word_id in enumerate(other_ids) if j != k]
        shorter_vector = table.data[kept].mean(axis=0)
        changes[word] = float(np.linalg.norm(long_vector - shorter_vector))
        print(f"    {pad(word, 10)}  {changes[word]:.3f}")
    print()
    print(f"  如果这个表示真的读懂了这句话，拿掉「我」应该几乎没影响，")
    print(f"  拿掉「发布」应该有影响。实际是 {changes['我']:.3f} 和 {changes['发布']:.3f} —— 差不多。")
    print("  变化量的大小只跟「这个词的向量离平均值有多远」有关，")
    print("  跟「这个词重不重要」没有关系。平均法对每个词一视同仁。")
    print()

    # ---------------------------------------------------------- 窗口版

    print("=" * 66)
    print("换一种取法：不看全句，只看 苹果 左右各两个词")
    print("=" * 66)

    left = max(0, position - 2)
    right = min(len(words), position + 3)
    window = [words[i] for i in range(left, right) if i != position]
    window_vector = table.data[[VOCAB.index(word) for word in window]].mean(axis=0)

    print(f"  窗口里是：{' '.join(window)}")
    show_neighbours("窗口平均 最像", table.data, window_vector, top=8)
    print()
    print(f"  全句平均 和 窗口平均 的余弦相似度 = {cosine(long_vector, window_vector):+.4f}")
    print("  同一条句子、同一个位置，只是换了「看多远」，算出来的表示就差了一半。")
    print()
    print("  那到底应该看全句、还是看窗口？窗口该开 2 个词还是 5 个词？")
    print("  平均法回答不了这个问题 —— 它只能把选中的词一视同仁地平均掉。")
    print()
    print("  真正的问题是：在「我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机」里，")
    print(f"  「发布」和「昨天」对 苹果 这个词的意思，贡献显然不一样。")
    print(f"  可平均法给它们各 {weight:.2f} 份，一模一样。")
    print("  这个权重，该由谁来定？")


if __name__ == "__main__":
    main()
