"""第 7 章的 before：第 6 章的模型，换到这一章的数据上。

第 6 章最后，我们的模型在 10 句话上拿到了 100% 的准确率。
这一章我们把任务换成"判断两个数字是不是一样"，
模型一个字都没改，梯度公式也一个字都没改。

跑一下，看看它还灵不灵。
"""

import math
import random

NUM_FEATURES = 2
NUM_CLASSES = 2

# 和 after.py 用同一份数据
XOR_DATA = [
    ([0.0, 0.0], 1, "苹果很好吃 / 香蕉很好吃"),          # 都是食品
    ([0.0, 1.0], 0, "苹果很好吃 / 苹果发布新手机"),      # 一食一科技
    ([1.0, 0.0], 0, "苹果发布新手机 / 苹果很甜"),        # 一科技一食
    ([1.0, 1.0], 1, "苹果发布新手机 / 苹果发布新芯片"),  # 都是科技
]


def make_data():
    return [(x, label) for x, label, _ in XOR_DATA]


# ------------------------------------------------------------------ 第 6 章的模型，原样搬过来


def forward(w, b, x):
    """每一类算一个分数：输入和这一类的权重逐项相乘再相加。"""
    scores = []
    for c in range(NUM_CLASSES):
        total = b[c]
        for i in range(NUM_FEATURES):
            total += w[c][i] * x[i]
        scores.append(total)
    return scores


def softmax(scores):
    biggest = max(scores)
    exps = [math.exp(s - biggest) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def loss_of(w, b, data):
    total = 0.0
    for x, label in data:
        p = softmax(forward(w, b, x))
        total += -math.log(p[label])
    return total / len(data)


def gradient(w, b, data):
    """第 6 章手推的那个公式：(预测概率 - 正确答案) × 输入。"""
    gw = [[0.0] * NUM_FEATURES for _ in range(NUM_CLASSES)]
    gb = [0.0] * NUM_CLASSES
    n = len(data)
    for x, label in data:
        p = softmax(forward(w, b, x))
        for c in range(NUM_CLASSES):
            diff = (p[c] - (1.0 if c == label else 0.0)) / n
            gb[c] += diff
            for i in range(NUM_FEATURES):
                gw[c][i] += diff * x[i]
    return gw, gb


def train(steps, lr, seed):
    rng = random.Random(seed)
    w = [[rng.uniform(-1.0, 1.0) for _ in range(NUM_FEATURES)]
         for _ in range(NUM_CLASSES)]
    b = [0.0] * NUM_CLASSES

    data = make_data()
    history = []
    for _ in range(steps):
        gw, gb = gradient(w, b, data)
        for c in range(NUM_CLASSES):
            b[c] -= lr * gb[c]
            for i in range(NUM_FEATURES):
                w[c][i] -= lr * gw[c][i]
        history.append(loss_of(w, b, data))
    return w, b, history


def predict(w, b, x):
    scores = forward(w, b, x)
    return 0 if scores[0] >= scores[1] else 1


# ------------------------------------------------------------------ main


def main():
    data = make_data()

    print("=" * 60)
    print("第 6 章的模型，跑这一章的数据")
    print("=" * 60)
    print(f"{'第一个数':>6}  {'第二个数':>6}  {'答案':>4}   来自哪两句")
    for x, label, note in XOR_DATA:
        print(f"{x[0]:>6.0f}  {x[1]:>6.0f}  {label:>4}   {note}")
    print()

    print("=" * 60)
    print("从 5 个不同的初始值出发，各训练 5000 步")
    print("=" * 60)
    print(f"{'初始值':>6}{'训练后的 loss':>16}{'科技类的 w[0]':>14}"
          f"{'食品类的 w[0]':>14}")
    for seed in range(5):
        w, b, history = train(steps=5000, lr=0.5, seed=seed)
        print(f"{seed:>6}{history[-1]:>16.6f}{w[0][0]:>14.4f}{w[1][0]:>14.4f}")

    print()
    print("5 次训练，不管从哪里出发，结果全是同一个数：0.693147。")
    print("它等于 -log(1/2)，也就是「每个类别各猜一半」的分数。")
    print("更直白的是最后两列：两个类别的权重长得一模一样，")
    print("模型认为所有输入属于哪一类都行 —— 它把所有信息都扔了。")
    print()

    print("=" * 60)
    print("它给出的答案")
    print("=" * 60)
    w, b, _ = train(steps=5000, lr=0.5, seed=0)
    print(f"{'输入':>8}  {'答案':>5}  {'两个分数':>20}  {'判成同类的概率':>14}")
    for x, label, _ in XOR_DATA:
        scores = forward(w, b, x)
        p = softmax(scores)
        print(f"{str([int(v) for v in x]):>8}  {label:>5}  "
              f"{scores[0]:+.4f} / {scores[1]:+.4f}  {p[1]:>14.4f}")
    print()
    print("不管输入是哪两个数、答案应该是几，模型给出的概率都是 0.5000。")
    print("它不是「学得不好」，而是「根本没法学」——")
    print("4 个点摆成一个正方形，一条直线怎么摆，都分不对它们的标签。")
    print()
    print("我们需要的不是更好的训练方法，而是一个能画出「不是直线」的东西。")


if __name__ == "__main__":
    main()
