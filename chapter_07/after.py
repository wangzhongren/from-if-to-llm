"""第 7 章的 after：在输入和输出之间，加一层。

上一章最后留下的问题是：每一层的梯度都要人推一遍。
这一章先不管梯度，先看一个更要命的问题：
**一层不够用。**

我们在输入和分数之间塞进一层"中间结果"，
让它先算一遍，再由中间结果算最终的分数。
这个文件把它写出来、训练起来、然后看看结果。
"""

import math
import random

NUM_FEATURES = 2
NUM_CLASSES = 2
HIDDEN_SIZE = 4
STEPS = 5000      # 训练多少轮

# ------------------------------------------------------------------ 数据
# 语料还是那 10 句。我们把每句话压成一个数字：
# 科技类的句子记 1，食品类的句子记 0。
#
# 新的任务是：给两个这样的数字，判断两个数字是不是一样。
# 一样 = 同类 = 1，不一样 = 不同类 = 0。
#
# 4 种组合，每一种配一对真实的句子：
XOR_DATA = [
    ([0.0, 0.0], 1, "苹果很好吃 / 香蕉很好吃"),          # 都是食品
    ([0.0, 1.0], 0, "苹果很好吃 / 苹果发布新手机"),      # 一食一科技
    ([1.0, 0.0], 0, "苹果发布新手机 / 苹果很甜"),        # 一科技一食
    ([1.0, 1.0], 1, "苹果发布新手机 / 苹果发布新芯片"),  # 都是科技
]


def make_data():
    """训练的时候不需要那句例句，去掉它。"""
    return [(x, label) for x, label, _ in XOR_DATA]


# ------------------------------------------------------------------ 两层网络


def init_params(seed=0, hidden_size=HIDDEN_SIZE):
    """随机初始化。中间层如果全是同一个值，它们就会永远学得一样 —— 所以要随机。"""
    rng = random.Random(seed)
    params = {
        "w1": [[rng.uniform(-0.5, 0.5) for _ in range(NUM_FEATURES)]
               for _ in range(hidden_size)],
        "b1": [0.0] * hidden_size,
        "w2": [[rng.uniform(-0.5, 0.5) for _ in range(hidden_size)]
               for _ in range(NUM_CLASSES)],
        "b2": [0.0] * NUM_CLASSES,
    }
    return params


def forward(params, x):
    """两层。先把 x 变成中间结果 h，再由 h 算出两个分数。

    中间层的大小直接从参数里读出来，这样换大小不用改这个函数。
    注意 h 只是又一个加权求和 —— 它和输入之间没有别的东西。
    """
    hidden_size = len(params["b1"])

    h = []
    for j in range(hidden_size):
        total = params["b1"][j]
        for i in range(NUM_FEATURES):
            total += params["w1"][j][i] * x[i]
        h.append(total)

    scores = []
    for c in range(NUM_CLASSES):
        total = params["b2"][c]
        for j in range(hidden_size):
            total += params["w2"][c][j] * h[j]
        scores.append(total)
    return h, scores


def softmax(scores):
    biggest = max(scores)
    exps = [math.exp(s - biggest) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def loss_of(params, data):
    total = 0.0
    for x, label in data:
        _, scores = forward(params, x)
        p = softmax(scores)
        total += -math.log(p[label])
    return total / len(data)


# ------------------------------------------------------------------ 手写的梯度
# 下面每一行，都是人拿着链式法则推出来的。
# 中间层让推导长出了一截：输出层的误差要先乘上 w2，倒着传回去。


def backward(params, data):
    hidden_size = len(params["b1"])
    grads = {
        "w1": [[0.0] * NUM_FEATURES for _ in range(hidden_size)],
        "b1": [0.0] * hidden_size,
        "w2": [[0.0] * hidden_size for _ in range(NUM_CLASSES)],
        "b2": [0.0] * NUM_CLASSES,
    }
    n = len(data)

    for x, label in data:
        h, scores = forward(params, x)
        p = softmax(scores)

        # 输出层：和上一章一模一样，(预测概率 - 正确答案)
        dscore = [(p[c] - (1.0 if c == label else 0.0)) / n
                  for c in range(NUM_CLASSES)]
        for c in range(NUM_CLASSES):
            grads["b2"][c] += dscore[c]
            for j in range(hidden_size):
                grads["w2"][c][j] += dscore[c] * h[j]

        # 中间层（新增的一截）：把输出层的误差，按 w2 的大小分回每一个中间结果
        dh = []
        for j in range(hidden_size):
            total = 0.0
            for c in range(NUM_CLASSES):
                total += dscore[c] * params["w2"][c][j]
            dh.append(total)
        for j in range(hidden_size):
            grads["b1"][j] += dh[j]
            for i in range(NUM_FEATURES):
                grads["w1"][j][i] += dh[j] * x[i]

    return grads


def train(steps=2000, lr=0.5, seed=0, hidden_size=HIDDEN_SIZE):
    params = init_params(seed, hidden_size)
    data = make_data()
    history = []

    for _ in range(steps):
        grads = backward(params, data)
        for key in ("w1", "w2"):
            for a in range(len(params[key])):
                for b in range(len(params[key][a])):
                    params[key][a][b] -= lr * grads[key][a][b]
        for key in ("b1", "b2"):
            for a in range(len(params[key])):
                params[key][a] -= lr * grads[key][a]
        history.append(loss_of(params, data))

    return params, history


def predict(params, x):
    _, scores = forward(params, x)
    return 0 if scores[0] >= scores[1] else 1


def accuracy(params, data):
    correct = 0
    for x, label in data:
        if predict(params, x) == label:
            correct += 1
    return correct / len(data)


def max_score_gap(params, data):
    """两个类别的分数最大差多少。

    这个数字最能说明问题：如果模型真的学会了区分，它会很大；
    如果模型什么也没学会，它会接近 0。
    """
    gap = 0.0
    for x, _ in data:
        _, scores = forward(params, x)
        gap = max(gap, abs(scores[0] - scores[1]))
    return gap


# ------------------------------------------------------------------ main


def main():
    data = make_data()

    print("=" * 60)
    print("数据：两个数字，判断它们是不是一样")
    print("=" * 60)
    print(f"{'第一个数':>6}  {'第二个数':>6}  {'答案':>4}   来自哪两句")
    for x, label, note in XOR_DATA:
        print(f"{x[0]:>6.0f}  {x[1]:>6.0f}  {label:>4}   {note}")
    print()
    print("两个数都是 0 或者都是 1 —— 答案 1。两个数不一样 —— 答案 0。")
    print("把 4 个点画在平面上，它们是正方形的四个角。")
    print()

    print("=" * 60)
    print(f"两层网络：中间层 {HIDDEN_SIZE} 个单元")
    print("=" * 60)

    params, history = train(steps=STEPS, lr=0.5, seed=0)
    print("前 5 步的 loss："
          + "  ".join(f"{v:.6f}" for v in history[:5]))
    print(f"第 {STEPS} 步的 loss：{history[-1]:.6f}")
    print(f"准确率：{accuracy(params, data):.2f}")
    print()
    print("注意这个 loss 停在哪：0.693147。")
    print("它正好是 -log(1/2) —— 「两个类别各猜一半」的分数。")
    print()

    print("=" * 60)
    print("它到底学到了什么")
    print("=" * 60)
    print(f"{'输入':>8}  {'中间层算出来的 4 个数':>40}  {'两个分数':>18}  判成同类的概率")
    for x, label, _ in XOR_DATA:
        h, scores = forward(params, x)
        hidden_text = "  ".join(f"{v:+.3f}" for v in h)
        p = softmax(scores)
        print(f"{str([int(v) for v in x]):>8}  {hidden_text:>40}  "
              f"{scores[0]:+.3f} / {scores[1]:+.3f}          {p[1]:.3f}")
    print()
    print("不管输入是什么，两个类别的分数几乎完全一样，概率都是 0.500。")
    print("模型学到的唯一一件事，就是「各一半」。它放弃了。")
    print()

    print("=" * 60)
    print("把中间层加宽，有用吗")
    print("=" * 60)
    print(f"{'中间层大小':>10}{'训练后的 loss':>16}{'两个类别的分数差':>18}")
    for size in (1, 2, 4, 8, 16):
        params_size, hist = train(steps=STEPS, lr=0.5, seed=0, hidden_size=size)
        print(f"{size:>10}{hist[-1]:>16.6f}{max_score_gap(params_size, data):>18.2e}")
    print()
    print("1 个、2 个、4 个、8 个、16 个 —— loss 一模一样，全都是 0.693147；")
    print("两个类别的分数差全都在十万分之一以下。中间层加宽，一点用都没有。")
    print()
    print("这一章我们把中间层搭起来了：代码能跑，梯度也是对的。")
    print("但是：中间层加了，它一点用都没有。")


if __name__ == "__main__":
    main()
