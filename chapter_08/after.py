"""第 8 章的 after：中间层后面，加一个小函数。

上一章的中间层没有用。这一章我们要给它加一样东西，
让它在平面上"拐一个弯"。

这个小函数只有一行：负数变成 0，正数原样留下。
加完之后，XOR 那个任务，4 个点全部分对。
"""

import math
import random

NUM_FEATURES = 2
NUM_CLASSES = 2
HIDDEN_SIZE = 4
STEPS = 5000

XOR_DATA = [
    ([0.0, 0.0], 1, "苹果很好吃 / 香蕉很好吃"),          # 都是食品
    ([0.0, 1.0], 0, "苹果很好吃 / 苹果发布新手机"),      # 一食一科技
    ([1.0, 0.0], 0, "苹果发布新手机 / 苹果很甜"),        # 一科技一食
    ([1.0, 1.0], 1, "苹果发布新手机 / 苹果发布新芯片"),  # 都是科技
]


def make_data():
    return [(x, label) for x, label, _ in XOR_DATA]


# ------------------------------------------------------------------ 那个小函数


def relu(value):
    """负数变成 0，正数原样留下。

    它自己不做什么了不起的事，但它让整条链条"拐了个弯"。
    """
    return value if value > 0.0 else 0.0


# ------------------------------------------------------------------ 两层网络（这次带拐弯）


def init_params(seed=0, hidden_size=HIDDEN_SIZE):
    rng = random.Random(seed)
    return {
        "w1": [[rng.uniform(-0.5, 0.5) for _ in range(NUM_FEATURES)]
               for _ in range(hidden_size)],
        "b1": [0.0] * hidden_size,
        "w2": [[rng.uniform(-0.5, 0.5) for _ in range(hidden_size)]
               for _ in range(NUM_CLASSES)],
        "b2": [0.0] * NUM_CLASSES,
    }


def forward(params, x):
    """中间层算完之后，先过一遍 relu，再交给输出层。

    返回三样东西：还没过 relu 的 z、过了 relu 的 h、最终的分数。
    反向传播的时候要用到 z —— 它决定了 relu 有没有把值掐掉。
    """
    hidden_size = len(params["b1"])

    z = []
    for j in range(hidden_size):
        total = params["b1"][j]
        for i in range(NUM_FEATURES):
            total += params["w1"][j][i] * x[i]
        z.append(total)
    h = [relu(value) for value in z]

    scores = []
    for c in range(NUM_CLASSES):
        total = params["b2"][c]
        for j in range(hidden_size):
            total += params["w2"][c][j] * h[j]
        scores.append(total)
    return z, h, scores


def softmax(scores):
    biggest = max(scores)
    exps = [math.exp(s - biggest) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def loss_of(params, data):
    total = 0.0
    for x, label in data:
        _, _, scores = forward(params, x)
        p = softmax(scores)
        total += -math.log(p[label])
    return total / len(data)


# ------------------------------------------------------------------ 手写的梯度
# 和上一章比，只多了一行 —— 但那一行决定了整件事成不成立。


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
        z, h, scores = forward(params, x)
        p = softmax(scores)

        # 输出层：还是 (预测概率 - 正确答案)
        dscore = [(p[c] - (1.0 if c == label else 0.0)) / n
                  for c in range(NUM_CLASSES)]
        for c in range(NUM_CLASSES):
            grads["b2"][c] += dscore[c]
            for j in range(hidden_size):
                grads["w2"][c][j] += dscore[c] * h[j]

        # 中间层：输出层的误差按 w2 分回来（和上一章一样）
        dh = []
        for j in range(hidden_size):
            total = 0.0
            for c in range(NUM_CLASSES):
                total += dscore[c] * params["w2"][c][j]
            dh.append(total)

        # 新加的一行：relu 把负的掐成了 0，那部分梯度也传不回去。
        # 一个中间单元这次被掐掉了，这次就别给它记功。
        dz = [dh[j] if z[j] > 0.0 else 0.0 for j in range(hidden_size)]

        for j in range(hidden_size):
            grads["b1"][j] += dz[j]
            for i in range(NUM_FEATURES):
                grads["w1"][j][i] += dz[j] * x[i]

    return grads


def train(steps=STEPS, lr=0.5, seed=0, hidden_size=HIDDEN_SIZE):
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
    _, _, scores = forward(params, x)
    return 0 if scores[0] >= scores[1] else 1


def accuracy(params, data):
    correct = 0
    for x, label in data:
        if predict(params, x) == label:
            correct += 1
    return correct / len(data)


# ------------------------------------------------------------------ main


def main():
    data = make_data()

    print("=" * 60)
    print("还是那个任务：两个数字，判断它们是不是一样")
    print("=" * 60)
    print(f"{'输入':>8}  {'答案':>5}   来自哪两句")
    for x, label, note in XOR_DATA:
        print(f"{str([int(v) for v in x]):>8}  {label:>5}   {note}")
    print()

    print("=" * 60)
    print(f"中间层 {HIDDEN_SIZE} 个单元，每个算完先过一遍 relu")
    print("=" * 60)

    params, history = train(steps=STEPS, lr=0.5, seed=0)
    print("前 5 步的 loss："
          + "  ".join(f"{v:.6f}" for v in history[:5]))
    print(f"第 {STEPS} 步的 loss：{history[-1]:.8f}")
    print(f"准确率：{accuracy(params, data):.2f}")
    print()

    print("=" * 60)
    print("它到底学会了什么")
    print("=" * 60)
    print(f"{'输入':>8}  {'中间层（过 relu 之前）':>26}  "
          f"{'过完 relu':>26}  {'两个分数':>18}")
    for x, label, _ in XOR_DATA:
        z, h, scores = forward(params, x)
        z_text = "  ".join(f"{v:+.2f}" for v in z)
        h_text = "  ".join(f"{v:+.2f}" for v in h)
        print(f"{str([int(v) for v in x]):>8}  {z_text:>26}  {h_text:>26}  "
              f"{scores[0]:+.3f} / {scores[1]:+.3f}")
    print()
    print("看第二列：有的值被 relu 掐成了 0，有的原样留下。")
    print("输入是 [0, 1] 的时候，四个中间单元全被掐成了 0 ——")
    print("模型干脆用「一个都不亮」来表示这一种情况。")
    print()
    print("再看最后一列：上一章那两个「完全一样」的分数，现在被彻底拉开了。")
    print("[0, 0] 是 -4.54 比 +3.93，[0, 1] 是 +3.40 比 -3.40。")
    print()

    print("=" * 60)
    print("逐句检查")
    print("=" * 60)
    for x, label, note in XOR_DATA:
        _, _, scores = forward(params, x)
        p = softmax(scores)
        guess = predict(params, x)
        mark = "对" if guess == label else "错"
        print(f"{mark}  {note:<26} 判成 {guess}，概率 {max(p):.4f}")


if __name__ == "__main__":
    main()
