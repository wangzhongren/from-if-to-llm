"""第 12 章 before.py —— 第 11 章的做法，撞上「一个词两种意思」。

第 11 章我们训练出了一张词向量表，规则是：**一个词，一个向量。**

这个规则在这两句话上会撞墙：

    苹果 很 好吃          这里说的是水果
    苹果 发布 新 手机      这里说的是公司

两句话里的 苹果 是同一个词，所以按第 11 章的规则，它们拿到的是
同一组数字 —— 不是"差不多"，是一个数字都不差。

这个文件把后果一条一条摆出来：

1. 两句话里的 苹果，向量完全相同，余弦相似度是 1.0000。
2. "和 苹果 最像的词"这个问题的答案，在两句话里一模一样。
3. 于是任何只看 苹果 这个词向量的判断，在两句话里都只能给出同一个答案。

我们是在第 11 章那张表上跑这个实验的，训练方式和语料都没有改。
问题不在训练得不够好，在于**那个表示法本身装不下两种意思**。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import SGD, cross_entropy, embedding, randn, zeros

# 和第 11 章一样的语料（加上第 13-16 章那两句长句和几句支撑句）
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
    "我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机",
    "小王 把 书 给了 小李 因为 他 明天 考试",
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

VOCAB = sorted({word for sentence in SENTENCES for word in sentence.split()})

DIM = 16
TABLE_SEED = 17
OUT_SEED = 105
EPOCHS = 600
LR = 0.3

FOOD_SENTENCE = "苹果 很 好吃"
TECH_SENTENCE = "苹果 发布 新 手机"


def display_width(text):
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def pad(text, width):
    return text + " " * (width - display_width(text))


def encode(sentences):
    word_to_id = {word: i for i, word in enumerate(VOCAB)}
    return [[word_to_id[word] for word in sentence.split()] for sentence in sentences]


def build_training_pairs(corpus, window=1):
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


def most_similar(table_data, word, top=6):
    index = VOCAB.index(word)
    scored = [(VOCAB[i], cosine(table_data[index], table_data[i]))
              for i in range(len(VOCAB)) if i != index]
    scored.sort(key=lambda pair: -pair[1])
    return scored[:top]


def main():
    corpus = encode(SENTENCES)
    seen, guessed = build_training_pairs(corpus)
    table, loss = train_embeddings(seen, guessed)

    print("=" * 66)
    print("第 11 章的做法：一个词，一个向量")
    print("=" * 66)
    print(f"  语料 {len(SENTENCES)} 句，词 {len(VOCAB)} 个，训练 {EPOCHS} 轮，loss = {loss:.4f}")
    print()

    apple = VOCAB.index("苹果")
    print("  现在做一件事：把两句话里的 苹果 拿出来，比一比。")
    print()
    print(f"    句子 A：{FOOD_SENTENCE}          （水果）")
    print(f"    句子 B：{TECH_SENTENCE}     （公司）")
    print()

    vector_a = table.data[apple]
    vector_b = table.data[apple]

    print("  A 句里 苹果 的向量：")
    print("    [" + "  ".join(f"{v:+.2f}" for v in vector_a) + "]")
    print("  B 句里 苹果 的向量：")
    print("    [" + "  ".join(f"{v:+.2f}" for v in vector_b) + "]")
    print()
    print(f"  两个向量的最大差值 = {np.abs(vector_a - vector_b).max():.7f}")
    print(f"  余弦相似度 = {cosine(vector_a, vector_b):.4f}")
    print()

    print("  这两个向量是同一个东西。不是「很像」，是同一个。")
    print("  因为在这个表示法里，「苹果」就是一个编号，")
    print("  编号查出来的行永远是那一行 —— 它在哪句话里都一样。")
    print()

    print("=" * 66)
    print("后果一：问「和 苹果 最像的词」，两句得到同一个答案")
    print("=" * 66)
    pairs = "  ".join(f"{word} {score:+.2f}" for word, score in most_similar(table.data, "苹果"))
    print(f"  在 A 句里问：{pairs}")
    print(f"  在 B 句里问：{pairs}")
    print()
    print("  两行一模一样。这个问题根本没把句子考虑进去。")
    print()

    print("=" * 66)
    print("后果二：任何只看这个词向量的判断，都必须给出同一个答案")
    print("=" * 66)
    print("  假设下游有个任务要判断「这里的 苹果 是水果还是公司」。")
    print("  它能用的输入是 苹果 的向量，以及它和别的词的相似度：")
    print()
    targets = ["好吃", "甜", "香蕉", "发布", "手机", "芯片"]
    print("                     " + "".join(f"{pad(t, 8)}" for t in targets))
    for name, label in [(FOOD_SENTENCE, "A"), (TECH_SENTENCE, "B")]:
        row = f"  {label} 句里的 苹果    "
        print(row + "".join(f"{cosine(vector_a, table.data[VOCAB.index(t)]):+7.2f} " for t in targets))
    print()
    print("  两行一模一样。也就是说，模型看见的「苹果」在两句话里毫无区别。")
    print("  它唯一能分辨这两句话的办法，是绕开 苹果 去看别的词 ——")
    print("  可我们想知道的，恰恰是 苹果 在这个位置上的意思。")
    print()
    print("=" * 66)
    print("结论")
    print("=" * 66)
    print("  问题不是训练得不够久，也不是维度不够高。")
    print("  问题是我们给每个词只留了一个位置 —— 一个位置只能装一种意思。")
    print()
    print("  要让 苹果 在两句话里不一样，必须让它的表示**跟着句子变**。")
    print("  那这个表示该怎么算？下一节我们从最笨的一种算法开始试。")


if __name__ == "__main__":
    main()
