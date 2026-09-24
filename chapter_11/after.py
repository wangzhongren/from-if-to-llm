"""第 11 章 after.py —— 让相似的词靠近。

上一章我们给每个词发了一组随机数字，然后发现：这组数字里什么都没有。
和 手机 最像的词是 派，和 苹果 最像的词是 做成。

这一章把这张表**变成可训练的参数**，用一个极其简单的任务去训练它：

    看一个词，猜它旁边站的是谁。

比如语料里有「苹果 发布 新 手机」，我们就造出这些训练样本：

    看到 苹果  →  猜 发布
    看到 发布  →  猜 苹果
    看到 发布  →  猜 新
    看到 新    →  猜 发布
    ...

就这么一条规则，把 13 句话里所有相邻的词对都用上。然后训练。

训练完你会发现：我们从来没有告诉过模型"手机 和 芯片 是一类东西"，
但 手机 的向量自己就挪到了 芯片 旁边。
因为能站在 手机 旁边的词和能站在 芯片 旁边的词，是同一批。

这说明一件事：**意义可以从"它跟谁一起出现"里长出来。**
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import SGD, cross_entropy, embedding, randn, zeros

# 第 1 章那张词表，一个词都不多、一个都不少
VOCAB = "苹果 发布 新 手机 芯片 电脑 华为 小米 好吃 很 甜 香蕉 这个 真 做成 派".split()

# 语料：全部由上面那 16 个词组成。苹果 依然两边都出现，这是故意的。
SENTENCES = [
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
]

DIM = 8           # 每个词用几个数字（第 10 章用了 16，这一章为什么变小，看 README）
TABLE_SEED = 17   # 向量表的初始值（这是要被训练的那张表）
OUT_SEED = 105    # 输出层的初始值
EPOCHS = 400
LR = 0.3
WINDOW = 1        # 看多远的邻居


def display_width(text):
    """中文在终端里占两格，英文数字占一格。"""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def pad(text, width):
    return text + " " * (width - display_width(text))


def encode(sentences):
    """把句子里的词换成编号。"""
    word_to_id = {word: i for i, word in enumerate(VOCAB)}
    return [[word_to_id[word] for word in sentence.split()] for sentence in sentences]


def build_training_pairs(corpus, window=WINDOW):
    """造训练样本：每看到一个词，就让它去猜它左右 window 步以内的邻居。

    返回两个数组：(猜的时候看到的词, 要猜的词)，长度一样，一一对应。
    """
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


def train_embeddings(seen, guessed, epochs=EPOCHS, lr=LR, seed=TABLE_SEED,
                     progress_every=None):
    """训练词向量表。

    模型是：
        用「看到的那个词」的向量，去预测「应该出现的那个词」
    预测的办法：向量和输出层相乘，得到 16 个词的分数，softmax 之后算交叉熵。

    注意看：这里**没有** w，也没有 b。要学的就是那张表本身。
    """
    table = randn(len(VOCAB), DIM, scale=0.1, requires_grad=True, seed=seed)
    out_weight = randn(DIM, len(VOCAB), scale=0.1, requires_grad=True, seed=OUT_SEED)
    out_bias = zeros(len(VOCAB), requires_grad=True)
    optimizer = SGD([table, out_weight, out_bias], lr=lr)

    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        seen_vectors = embedding(table, seen)          # (样本数, DIM)
        logits = seen_vectors @ out_weight + out_bias  # (样本数, 词表大小)
        loss = cross_entropy(logits, guessed)
        loss.backward()
        optimizer.step()

        if progress_every and (epoch + 1) % progress_every == 0:
            history.append((epoch + 1, loss.item()))

    return table, loss.item(), history


def normalise(matrix):
    """把每一行变成单位长度，这样点积就等于余弦相似度。"""
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def similarity_matrix(table_data):
    unit = normalise(table_data)
    return unit @ unit.T


def nearest_neighbours(similarities, word, top=5):
    """和 word 最像的几个词（不含它自己）。"""
    index = VOCAB.index(word)
    order = np.argsort(-similarities[index])
    return [(VOCAB[j], similarities[index, j]) for j in order[1:top + 1]]


def show_similarity_blocks(similarities):
    """把 16x16 的相似度表分左右两半打印出来。

    列宽固定，这样终端里能对齐。
    """
    half = len(VOCAB) // 2
    for start, stop in [(0, half), (half, len(VOCAB))]:
        header = " " * 6 + "".join(pad(VOCAB[j], 6) for j in range(start, stop))
        print(header)
        for i in range(len(VOCAB)):
            row = pad(VOCAB[i], 6) + "".join(f"{similarities[i, j]:6.2f}" for j in range(start, stop))
            print(row)
        print()


def to_2d(matrix):
    """把一堆高维向量压成 2 个数字，好画到平面上。

    做法：先减掉中心，再找两个"能把这些点分得最开"的方向。
    这两行线性代数不是本章的重点，看不懂可以跳过去 —— 重要的是图。
    返回 (每个点的横纵坐标, 两个方向各自解释了多少差异)。
    """
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, directions = np.linalg.svd(centred, full_matrices=False)
    points = centred @ directions[:2].T
    explained = (singular ** 2) / (singular ** 2).sum()
    return points, explained[:2]


def scatter(matrix, width=54, height=18, title=""):
    """在终端里画一张散点图。每个词就是图上的一个标签。"""
    points, explained = to_2d(matrix)
    low, high = points.min(axis=0), points.max(axis=0)
    span = high - low
    span[span == 0] = 1.0
    normalised = (points - low) / span
    columns = (normalised[:, 0] * (width - 1)).round().astype(int)
    rows = ((1 - normalised[:, 1]) * (height - 1)).round().astype(int)

    grid = [[None] * width for _ in range(height)]
    taken = []

    def is_free(row, column, size):
        if row < 0 or column < 0 or column + size > width:
            return False
        return all(not (row == r and column < c + w and c < column + size)
                   for r, c, w in taken)

    for i, word in enumerate(VOCAB):
        size = display_width(word)
        spot = None
        for radius in range(0, 10):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    if is_free(rows[i] + dr, columns[i] + dc, size):
                        spot = (rows[i] + dr, columns[i] + dc)
                        break
                if spot:
                    break
            if spot:
                break
        if spot is None:
            continue
        row, column = spot
        position = column
        for ch in word:
            grid[row][position] = ch
            if display_width(ch) == 2:
                grid[row][position + 1] = ""
            position += display_width(ch)
        taken.append((row, column, size))

    if title:
        print(title)
    print("┌" + "─" * width + "┐")
    for row in grid:
        print("│" + "".join(" " if ch is None else ch for ch in row) + "│")
    print("└" + "─" * width + "┘")
    print(f"横轴：第 1 个方向（解释 {explained[0] * 100:.0f}% 的差异）")
    print(f"纵轴：第 2 个方向（解释 {explained[1] * 100:.0f}% 的差异）")
    print()


def main():
    corpus = encode(SENTENCES)
    seen, guessed = build_training_pairs(corpus)

    print("=" * 62)
    print("语料和训练样本")
    print("=" * 62)
    print(f"  句子 {len(SENTENCES)} 条，词 {len(VOCAB)} 个，维度 {DIM}")
    print(f"  训练样本 {len(seen)} 条，都是「看到某个词 → 猜它旁边的词」")
    print("  前 6 条：")
    for k in range(6):
        print(f"    看到 {pad(VOCAB[seen[k]], 6)} → 猜 {VOCAB[guessed[k]]}")
    print()

    table, loss, _ = train_embeddings(seen, guessed)
    print("=" * 62)
    print(f"训练 {EPOCHS} 轮之后的 loss = {loss:.4f}")
    print("=" * 62)
    print("  （随机乱猜的话，16 个词的 loss 是 ln(16) ≈ 2.77。）")
    print()

    similarities = similarity_matrix(table.data)

    print("=" * 62)
    print("每个词的「邻居」——和它最像的 5 个词")
    print("=" * 62)
    for word in ["手机", "芯片", "发布", "新", "好吃", "很", "香蕉", "派", "苹果"]:
        pairs = "  ".join(f"{w} {s:.2f}" for w, s in nearest_neighbours(similarities, word))
        print(f"  {pad(word, 6)} → {pairs}")
    print()

    print("=" * 62)
    print("相似度表（余弦相似度，1.00 表示最像）")
    print("=" * 62)
    show_similarity_blocks(similarities)

    scatter(table.data, title=f"把 {DIM} 个数字压成 2 个数字之后：")


if __name__ == "__main__":
    main()
