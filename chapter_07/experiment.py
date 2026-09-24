"""这一章要证明一件事：

    这 4 个点，用一条直线是分不开的。

不是"我们没训练好"，也不是"学习率没调对"。
我们要把所有可能的直线全都试一遍，看到底有没有一条能全分对。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from before import (  # noqa: E402
    forward,
    make_data,
    train,
)
from after import (  # noqa: E402
    max_score_gap as two_layer_gap,
    train as two_layer_train,
)


def display_width(text):
    """中文占两个字符宽，打印表格时要把这件事算进去。"""
    return sum(2 if ord(ch) > 127 else 1 for ch in text)


def pad(text, width):
    return text + " " * max(0, width - display_width(text))


def predict_line(w1, w2, b, x):
    """一条直线的判断方式：分数大于 0 就判 1，否则判 0。"""
    return 1 if w1 * x[0] + w2 * x[1] + b > 0 else 0


def line_accuracy(w1, w2, b, data):
    correct = 0
    for x, label in data:
        if predict_line(w1, w2, b, x) == label:
            correct += 1
    return correct / len(data)


# ------------------------------------------------------------------ 第 1 步


def step1_picture():
    print("=" * 60)
    print("第 1 步：把 4 个点摆出来")
    print("=" * 60)
    print()
    print("  x2=1 |   (0,1) 答案 0        (1,1) 答案 1")
    print("       |")
    print("  x2=0 |   (0,0) 答案 1        (1,0) 答案 0")
    print("       +--------------------------------------")
    print("            x1=0                 x1=1")
    print()
    print("从左下角 (0,0) 走到右上角 (1,1)：两个角都是答案 1。")
    print("从左上角 (0,1) 走到右下角 (1,0)：两个角都是答案 0。")
    print("两条对角线上的答案正好相反 —— 这就是问题的全部形状。")
    print()


# ------------------------------------------------------------------ 第 2 步


def step2_brute_force():
    print("=" * 60)
    print("第 2 步：把所有直线都试一遍")
    print("=" * 60)
    print("一条直线的全部可能，就是三个数：w1、w2、b。")
    print("我们让它们各自从 -3 走到 3，每隔 0.1 取一个值，")
    print("三个循环套起来，一共是 61 × 61 × 61 条直线。")
    print()

    data = make_data()
    best = 0.0
    best_lines = []
    total = 0

    steps = 61
    for a in range(steps):
        w1 = -3.0 + 0.1 * a
        for b_i in range(steps):
            w2 = -3.0 + 0.1 * b_i
            for c in range(steps):
                b = -3.0 + 0.1 * c
                total += 1
                acc = line_accuracy(w1, w2, b, data)
                if acc > best:
                    best = acc
                    best_lines = [(w1, w2, b)]
                elif acc == best and len(best_lines) < 5:
                    best_lines.append((w1, w2, b))

    print(f"一共试了 {total} 条直线。")
    print(f"其中最好的一条，准确率是 {best:.2f}（{best * 4:.0f}/4）。")
    print("没有一条能到 1.00。")
    print()
    print("挑三条有代表性的直线看看：")
    print(f"{'w1':>7}{'w2':>7}{'b':>7}{'准确率':>9}   判对了第几个   判错了第几个")
    for w1, w2, b in ((1.0, 1.0, -1.5), (1.0, 1.0, -0.5), (-1.0, -1.0, 0.5)):
        hits = [i for i, (x, label) in enumerate(data)
                if predict_line(w1, w2, b, x) == label]
        misses = [i for i in range(len(data)) if i not in hits]
        acc = line_accuracy(w1, w2, b, data)
        print(f"{w1:>7.1f}{w2:>7.1f}{b:>7.1f}{acc:>9.2f}   {pad(str(hits), 16)}{misses}")
    print()
    print("每一条直线都会错掉至少一个点。这不是巧合 ——")
    print("正方形的一条对角线上两个角标签相同，另一条对角线上两个角标签也相同，")
    print("而直线只能把平面切成两半，切不出这种形状。")
    print()


# ------------------------------------------------------------------ 第 3 步


def step3_training():
    print("=" * 60)
    print("第 3 步：那用第 6 章的办法训练呢")
    print("=" * 60)
    print("也许好直线是有的，只是我们没找到？我们从 10 个不同的初始值出发，")
    print("每个训练 5000 步，把最后的 loss 和权重列出来。")
    print()

    print(f"{'初始值':>6}{'训练后的 loss':>16}{'科技类的 w[0]':>14}"
          f"{'食品类的 w[0]':>14}")
    for seed in range(10):
        w, b, history = train(steps=5000, lr=0.5, seed=seed)
        print(f"{seed:>6}{history[-1]:>16.6f}{w[0][0]:>14.4f}{w[1][0]:>14.4f}")

    print()
    print("10 次，一次例外都没有，全都停在 0.693147。")
    print("这不是「卡住了」，这是这个模型的极限：")
    print("它能表达的全部东西，就是一条直线。而直线不够用。")
    print()


# ------------------------------------------------------------------ 第 4 步


def step4_middle_layer():
    print("=" * 60)
    print("第 4 步：那加一层呢")
    print("=" * 60)
    print("把 after.py 里那个带中间层的网络拿过来，和一层摆在一起比。")
    print()

    data = make_data()
    print(f"{pad('模型', 20)}{'训练后的 loss':>16}{'两个类别的分数最大差':>22}")

    w, b, history = train(steps=5000, lr=0.5, seed=0)
    gap = max(abs(forward(w, b, x)[0] - forward(w, b, x)[1]) for x, _ in data)
    print(f"{pad('一层（一条直线）', 20)}{history[-1]:>16.6f}{gap:>22.2e}")

    for size in (1, 4, 16, 64):
        name = f"两层（中间 {size} 个）"
        params, hist = two_layer_train(steps=5000, lr=0.5, seed=0,
                                       hidden_size=size)
        print(f"{pad(name, 20)}{hist[-1]:>16.6f}"
              f"{two_layer_gap(params, data):>22.2e}")

    print()
    print("一层、两层、中间层加宽到 64 个 —— loss 全是 0.693147，一模一样。")
    print("我们辛辛苦苦加的那一层，什么也没改变。")
    print()
    print("这是这一章留下的问题。下一章我们去看看到底是哪里不对。")


def main():
    step1_picture()
    step2_brute_force()
    step3_training()
    step4_middle_layer()


if __name__ == "__main__":
    main()
