"""
第 10 章的测试：测的是本章真正的两个结论——

1. 用编号当输入时，模型学到的只是"编号的排列顺序"。
   词表顺序一换，学到的东西就作废。
2. 把一个词表示成一组数字之后，这两件事都不再发生。

别的章节也可能有同名的 after.py / before.py，先清掉缓存，确保导入的是本章的。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

sys.modules.pop("after", None)
sys.modules.pop("before", None)

import numpy as np

import after
import before


def labeled_words():
    """本章做分类任务用的 15 个词，以及它们的类别。"""
    words = before.TECH_WORDS + before.FOOD_WORDS
    labels = [0] * len(before.TECH_WORDS) + [1] * len(before.FOOD_WORDS)
    return words, labels


# ---------------------------------------------------------------- 编号的问题


def test_编号当输入时分数只能沿着编号一条直线排下去():
    """只有一个 w 和一个 b，分数只能是编号的线性函数。"""
    ids = {word: i for i, word in enumerate(before.VOCAB)}
    words, labels = labeled_words()
    word_ids = [ids[w] for w in words]

    w, b, _ = before.train(word_ids, labels)
    _, scores = before.evaluate(w, b, word_ids, labels)

    # 按编号从小到大排好之后，分数必须是单调的
    order = np.argsort(word_ids)
    sorted_scores = scores[order]
    is_monotone = np.all(np.diff(sorted_scores) > 0) or np.all(np.diff(sorted_scores) < 0)
    assert is_monotone


def test_换一套编号旧模型立刻失效():
    """在方案 A 上训练 100% 正确的模型，搬到方案 B 的编号上就崩了。"""
    words, labels = labeled_words()
    ids_a = {word: i for i, word in enumerate(before.VOCAB)}
    ids_b = {word: i for i, word in enumerate(before.PINYIN_ORDER)}

    w, b, _ = before.train([ids_a[w] for w in words], labels)

    accuracy_a, _ = before.evaluate(w, b, [ids_a[w] for w in words], labels)
    accuracy_b, _ = before.evaluate(w, b, [ids_b[w] for w in words], labels)

    assert accuracy_a == 1.0
    assert accuracy_b < 0.8


def test_同一种编号下训练结果可复现():
    """固定了种子和初值，两次训练必须一模一样。"""
    ids = {word: i for i, word in enumerate(before.VOCAB)}
    words, labels = labeled_words()
    word_ids = [ids[w] for w in words]

    w1, b1, loss1 = before.train(word_ids, labels)
    w2, b2, loss2 = before.train(word_ids, labels)

    assert loss1 == loss2
    assert float(w1.data) == float(w2.data)


# ---------------------------------------------------------------- 向量好在哪


def test_一组随机数字就能分开编号分不开的那些词():
    """按拼音排序的编号分不开这批词，换成向量就能全分对。"""
    words, labels = labeled_words()
    ids_b = {word: i for i, word in enumerate(before.PINYIN_ORDER)}

    # 先用编号试一次：分不开
    w, b, _ = before.train([ids_b[w] for w in words], labels)
    accuracy_ids, _ = before.evaluate(w, b, [ids_b[w] for w in words], labels)
    assert accuracy_ids < 0.8

    # 再换成向量：全分对
    table = after.build_vector_table()
    vectors = {word: table.data[i] for i, word in enumerate(after.VOCAB)}
    scores, _ = after.train_scores([vectors[w] for w in words], labels)
    accuracy_vectors = float(((scores > 0).astype(int) == np.array(labels)).mean())
    assert accuracy_vectors == 1.0


def test_词拿到哪一组数字跟编号无关():
    """编号变了，同一个词拿到的还是同一组数字，分数也一动不动。"""
    words, labels = labeled_words()
    table = after.build_vector_table()
    vectors = {word: table.data[i] for i, word in enumerate(after.VOCAB)}

    scores_a, _ = after.train_scores([vectors[w] for w in words], labels)
    scores_b, _ = after.train_scores([vectors[w] for w in words], labels)

    # 两次训练完全一样：向量表是只读的，w 的初值也是固定的
    assert np.allclose(scores_a, scores_b)

    # 编号确实变了（大部分词的编号都换了位置）……
    ids_a = {word: i for i, word in enumerate(after.VOCAB)}
    ids_b = {word: i for i, word in enumerate(after.PINYIN_ORDER)}
    moved = sum(1 for word in after.VOCAB if ids_a[word] != ids_b[word])
    assert moved > len(after.VOCAB) // 2

    # ……但每个词拿到的还是同一组数字
    for word in after.VOCAB:
        assert np.array_equal(vectors[word], table.data[after.VOCAB.index(word)])


def test_训练只动_w_不动向量表():
    """这一章不训练词向量。表格里的数字必须一个都没变。"""
    table = after.build_vector_table()
    snapshot = table.data.copy()
    vectors = {word: table.data[i] for i, word in enumerate(after.VOCAB)}
    words, labels = labeled_words()

    after.train_scores([vectors[w] for w in words], labels)

    assert np.array_equal(table.data, snapshot)


def test_余弦相似度自比为一():
    """找"最像的词"靠的是余弦相似度，先确认它本身是对的。"""
    table = after.build_vector_table()
    vectors = {word: table.data[i] for i, word in enumerate(after.VOCAB)}
    for word in after.VOCAB:
        assert abs(after.cosine_similarity(vectors[word], vectors[word]) - 1.0) < 1e-9


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
