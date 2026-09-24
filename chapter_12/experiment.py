"""第 12 章 experiment.py —— 同一个「苹果」，两种表示法比一比。

这个文件做三个实验：

实验一  造一个下游小任务：判断句子里「苹果」这个位置说的是水果还是公司。
        用第 11 章的静态表示，5 个句子给出的是同一个答案 —— 2 个碰巧蒙对，
        3 个判错（它一律判成水果）。

实验二  同一个任务，只把「苹果」的表示换成上下文向量，5 个句子全部判对。

实验三  把「平均」的粗糙之处量出来。

为了不把训练那段代码抄第二遍，这个文件直接复用 after.py 里的函数。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import after

FOOD_WORDS = "好吃 甜 香蕉 派 很 真 这个 做成".split()
TECH_WORDS = "发布 新 手机 芯片 电脑 华为 小米".split()

# 五个含「苹果」的句子。前两个说的是水果，后三个说的是公司。
TEST_SENTENCES = [
    "苹果 很 好吃",
    "苹果 很 甜",
    "苹果 发布 新 手机",
    "苹果 发布 新 芯片",
    "我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机",
]


def food_or_tech(table_data, vector):
    """这个小任务的全部内容：算它和两组词的平均相似度，谁高算谁。

    返回 (和食品组的平均相似度, 和科技组的平均相似度, 差值)。
    差值为正 → 判成食品；为负 → 判成科技。
    """
    food = np.mean([after.cosine(vector, table_data[after.VOCAB.index(w)]) for w in FOOD_WORDS])
    tech = np.mean([after.cosine(vector, table_data[after.VOCAB.index(w)]) for w in TECH_WORDS])
    return food, tech, food - tech


def truth_of(sentence):
    """这五个句子里，哪些说的是水果，哪些说的是公司。"""
    return "水果" if "好吃" in sentence or "很 甜" in sentence else "公司"


def experiment_old_way(table_data):
    print("=" * 70)
    print("实验一：造一个小任务，用第 11 章的静态表示来做")
    print("=" * 70)
    print("  任务：判断句子里「苹果」这个位置说的是水果还是公司。")
    print("  做法：算它和一组食品词、一组科技词的平均相似度，谁高算谁。")
    print()
    print("  句子                                       食品分  科技分  判定     正确答案")
    print("  " + "-" * 68)

    apple = after.VOCAB.index("苹果")
    correct = 0
    for sentence in TEST_SENTENCES:
        food, tech, gap = food_or_tech(table_data, table_data[apple])
        guess = "水果" if gap > 0 else "公司"
        truth = truth_of(sentence)
        correct += guess == truth
        mark = "  ✓" if guess == truth else "  ✗"
        print(f"  {after.pad(sentence, 40)} {food:+.2f}  {tech:+.2f}  {guess}{mark}      {truth}")
    print()
    print(f"  答对 {correct} / {len(TEST_SENTENCES)}。")
    print("  注意「食品分」那一列：五个句子全都是 +0.25，一个字都没变。")
    print("  因为静态表示里，这五个「苹果」本来就是同一个向量。")
    print("  模型手上只有这一个答案，它只能五句话都答同一个。")
    print()


def experiment_new_way(table_data):
    print("=" * 70)
    print("实验二：同一个任务，换成上下文表示")
    print("=" * 70)
    print("  除了把「苹果」的表示换成上下文向量，别的什么都没变。")
    print()
    print("  句子                                       食品分  科技分  判定     正确答案")
    print("  " + "-" * 68)

    correct = 0
    for sentence in TEST_SENTENCES:
        position = sentence.split().index("苹果")
        vector = after.context_vector(table_data, sentence, position)
        food, tech, gap = food_or_tech(table_data, vector)
        guess = "水果" if gap > 0 else "公司"
        truth = truth_of(sentence)
        correct += guess == truth
        mark = "  ✓" if guess == truth else "  ✗"
        print(f"  {after.pad(sentence, 40)} {food:+.2f}  {tech:+.2f}  {guess}{mark}      {truth}")
    print()
    print(f"  答对 {correct} / {len(TEST_SENTENCES)}。")
    print("  前两句「食品分」高，后三句「科技分」高 —— 分开了。")
    print()

    print("  把两种表示法并排放在一起，看这个差值：")
    print()
    print("  句子                                     静态表示   上下文表示")
    print("  " + "-" * 62)
    for sentence in TEST_SENTENCES:
        position = sentence.split().index("苹果")
        _, _, gap_static = food_or_tech(table_data, table_data[after.VOCAB.index("苹果")])
        _, _, gap_context = food_or_tech(table_data, after.context_vector(table_data, sentence, position))
        print(f"  {after.pad(sentence, 40)} {gap_static:+.3f}      {gap_context:+.3f}")
    print()
    print("  静态表示那一列是个常数；上下文表示那一列跟着句子正负翻面。")
    print()


def experiment_average_is_crude(table_data):
    print("=" * 70)
    print("实验三：但这个「平均」很粗糙")
    print("=" * 70)

    sentence = after.LONG_SENTENCE
    words = sentence.split()
    position = words.index("苹果")

    print(f"  {sentence}")
    print()

    others = [word for i, word in enumerate(words) if i != position]
    weight = 1.0 / len(others)
    print("  每个词的权重：")
    print("    " + "  ".join(f"{word} {weight:.2f}" for word in others))
    print()
    print("  十个词，一人一份。「昨天」和「发布」一模一样。")
    print()

    long_vector = after.context_vector(table_data, sentence, position)
    print("  平均出来的向量，最像哪几个词？")
    pairs = "  ".join(f"{w} {s:+.2f}" for w, s in after.most_similar(table_data, long_vector, top=8))
    print(f"    {pairs}")
    print("  排第一的是「的」。")
    print()

    other_ids = [after.VOCAB.index(word) for word in others]
    print("  把某个词从平均里拿掉，看这个向量变了多少：")
    print()
    print("    拿掉的词    变化量")
    print("    " + "-" * 22)
    changes = {}
    for k, word in enumerate(others):
        kept = [word_id for j, word_id in enumerate(other_ids) if j != k]
        changes[word] = float(np.linalg.norm(long_vector - table_data[kept].mean(axis=0)))
        print(f"    {after.pad(word, 10)}  {changes[word]:.3f}")
    print()
    print(f"  拿掉「我」变 {changes['我']:.3f}，拿掉「发布」变 {changes['发布']:.3f} —— 差不了多少。")
    print("  如果这个表示真的读懂了这句话，前者应该接近 0，后者应该很大。")
    print()


def main():
    corpus = after.encode(after.SENTENCES)
    seen, guessed = after.build_training_pairs(corpus)
    table, loss = after.train_embeddings(seen, guessed)

    print(f"（先按第 11 章的办法训练一张表：{len(after.SENTENCES)} 句、"
          f"{len(after.VOCAB)} 个词，loss = {loss:.4f}）")
    print()

    experiment_old_way(table.data)
    experiment_new_way(table.data)
    experiment_average_is_crude(table.data)


if __name__ == "__main__":
    main()
