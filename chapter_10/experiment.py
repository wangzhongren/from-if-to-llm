"""第 10 章 experiment.py —— 亲眼看看编号有多不靠谱。

这个文件做三个实验：

实验一  编号里的"远近"是我们随手定的。
        同一个"苹果"，在两种词表顺序下，离它最近的词是两个不同的词。

实验二  15 个词一共有 15! 种编号方式。随机撒一批，数一数其中有多少种
        能让一条直线把两类词分开。

实验三  三种编号方案下，同一个模型的 loss 曲线。
        看得出来：能不能学会，跟模型没关系，跟编号有关系。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import SGD, Tensor, cross_entropy

VOCAB = "苹果 发布 新 手机 芯片 电脑 华为 小米 好吃 很 甜 香蕉 这个 真 做成 派".split()
TECH_WORDS = "发布 新 手机 芯片 电脑 华为 小米".split()
FOOD_WORDS = "好吃 很 甜 香蕉 这个 真 做成 派".split()
PINYIN_ORDER = "电脑 发布 好吃 很 华为 派 苹果 手机 甜 香蕉 小米 新 芯片 这个 真 做成".split()

STEPS = 3000
LR = 0.1
REPORT_EVERY = 300


def display_width(text):
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def pad(text, width):
    return text + " " * (width - display_width(text))


def experiment_distances():
    """编号给出的"远近"是编号自己的，不是词的。"""
    print("=" * 62)
    print("实验一：编号里的「远近」")
    print("=" * 62)

    rng = np.random.default_rng(0)
    schemes = [
        ("A  第 1 章的词表顺序", VOCAB),
        ("B  按拼音排序", PINYIN_ORDER),
        ("C  随机打乱", [VOCAB[i] for i in rng.permutation(len(VOCAB))]),
    ]

    word = "苹果"
    nearest = []
    for name, order in schemes:
        ids = {w: i for i, w in enumerate(order)}
        others = [w for w in VOCAB if w != word]
        distances = sorted((abs(ids[w] - ids[word]), w) for w in others)
        near = distances[0]
        far = distances[-1]
        nearest.append(near[1])
        print(f"  {pad(name, 20)} 苹果 的编号 = {ids[word]:>2}")
        print(f"  {'':20} 编号上离它最近：{pad(near[1], 6)} 距离 {near[0]}")
        print(f"  {'':20} 编号上离它最远：{pad(far[1], 6)} 距离 {far[0]}")
    print()
    print(f"  三种编号下，「离 苹果 最近的词」分别是：{'、'.join(nearest)}。")
    print("  三个答案互相矛盾，而它们说的是同一个 苹果。")
    print("  词还是那批词，编号一动，「谁离谁近」就全部重排。")
    print("  编号里的距离是词表顺序留下的影子，不是词的意思。")
    print()


def experiment_how_many_schemes():
    """15 个词的编号方案有 15! 种，能学会的极少。"""
    print("=" * 62)
    print("实验二：能「学会」的编号方案有多少种")
    print("=" * 62)

    n_tech, n_food = len(TECH_WORDS), len(FOOD_WORDS)
    total = n_tech + n_food
    n_all = 1
    for k in range(1, total + 1):
        n_all *= k
    n_good_exact = 2
    for k in range(1, n_tech + 1):
        n_good_exact *= k
    for k in range(1, n_food + 1):
        n_good_exact *= k

    print(f"  {total} 个词，编号方案一共有 {total}! = {n_all:,} 种。")
    print("  一条直线只能把编号切一刀 —— 想让 7 个科技词全在一边、")
    print("  8 个食品词全在另一边，只有两种排法：科技词全都排在前面，或者全排在后面。")
    print(f"  这样的排法有 2 x {n_tech}! x {n_food}! = {n_good_exact:,} 种。")
    print(f"  精确比例 = {n_good_exact / n_all:.8f}  ≈ 1 / {n_all / n_good_exact:.0f}")
    print()

    samples = 200_000
    rng = np.random.default_rng(0)
    n_ok = 0
    for _ in range(samples):
        order = rng.permutation(total)
        tech_ids = order[:n_tech]
        food_ids = order[n_tech:]
        if tech_ids.max() < food_ids.min() or food_ids.max() < tech_ids.min():
            n_ok += 1
    print(f"  随机撒 {samples:,} 种编号方案，实际数出来能分开的有 {n_ok} 种")
    print(f"  实际比例 = {n_ok / samples:.8f}")
    print(f"  平均要试 {samples / max(n_ok, 1):.0f} 次才碰上一次。")
    print()
    print("  换句话说：如果模型能学会，功劳多半在编号，不在模型。")
    print()


def train_curve(word_ids, labels, steps=STEPS, lr=LR):
    """训练 分数 = w * 编号 + b，返回每 REPORT_EVERY 步记一次的 loss。"""
    w = Tensor(0.0, requires_grad=True)
    b = Tensor(0.0, requires_grad=True)
    x = Tensor(np.array(word_ids, dtype=float))
    one_two = Tensor(np.array([0.0, 1.0]))
    targets = np.array(labels)
    optimizer = SGD([w, b], lr=lr)

    history = []
    for step in range(steps):
        optimizer.zero_grad()
        score = w * x + b
        logits = score.reshape(len(word_ids), 1) * one_two
        loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        if (step + 1) % REPORT_EVERY == 0:
            history.append(loss.item())
    return history


def experiment_loss_curves():
    """同一个模型，三种编号，三条 loss 曲线。"""
    print("=" * 62)
    print("实验三：三种编号方案下的 loss 曲线")
    print("=" * 62)

    rng = np.random.default_rng(0)
    schemes = [
        ("A  第 1 章的词表顺序", VOCAB),
        ("B  按拼音排序", PINYIN_ORDER),
        ("C  随机打乱", [VOCAB[i] for i in rng.permutation(len(VOCAB))]),
    ]

    words = TECH_WORDS + FOOD_WORDS
    labels = [0] * len(TECH_WORDS) + [1] * len(FOOD_WORDS)

    header = "  步数  " + "".join(f"{name.split()[0]:>10}" for name, _ in schemes)
    print(header)
    print("  " + "-" * (len(header) - 2))

    curves = []
    for _, order in schemes:
        ids = {w: i for i, w in enumerate(order)}
        curves.append(train_curve([ids[w] for w in words], labels))

    for i, step in enumerate(range(REPORT_EVERY, STEPS + 1, REPORT_EVERY)):
        line = f"  {step:>4}  "
        line += "".join(f"{curve[i]:>10.4f}" for curve in curves)
        print(line)
    print()
    print("  A 的 loss 一路往下掉，而且还在往下掉；")
    print("  B 和 C 从第 300 步起就一动不动 —— 它们已经撞到那条直线的天花板了。")
    print("  三个模型一模一样，词一模一样，任务一模一样。")
    print("  唯一的区别是：编号是怎么发的。")
    print()


def main():
    experiment_distances()
    experiment_how_many_schemes()
    experiment_loss_curves()


if __name__ == "__main__":
    main()
