"""这一章的核心实验：让中间层自己招供。

上一章我们加了中间层，它一点用都没有。原因只有两种可能：

    A. 中间层还不够宽 / 还不够深。
    B. 中间层从一开始就不可能有用。

这个实验要把 A 和 B 分开。

办法是：把训练好的多层网络，**乘开**，看看它到底长什么样。
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from before import (  # noqa: E402
    XOR_DATA,
    forward as deep_forward,
    init_params as deep_init,
    make_data,
    train as deep_train,
)
from after import (  # noqa: E402
    accuracy as relu_accuracy,
    forward as relu_forward,
    init_params as relu_init,
    make_data as relu_data,
    softmax,
    train as relu_train,
)


def collapse(params):
    """把一叠"纯加权求和"的层，压成单独一层。

    从第一层开始，一层一层乘进去，最后得到一组 w 和一个 b。
    """
    weights = params["ws"][0]
    biases = params["bs"][0]

    for layer in range(1, len(params["ws"])):
        layer_w = params["ws"][layer]
        layer_b = params["bs"][layer]

        new_b = []
        for r in range(len(layer_w)):
            total = layer_b[r]
            for k in range(len(biases)):
                total += layer_w[r][k] * biases[k]
            new_b.append(total)

        new_w = []
        for r in range(len(layer_w)):
            row = []
            for c in range(len(weights[0])):
                total = 0.0
                for k in range(len(weights)):
                    total += layer_w[r][k] * weights[k][c]
                row.append(total)
            new_w.append(row)

        weights, biases = new_w, new_b

    return weights, biases


def collapsed_output(weights, biases, x):
    return [sum(weights[c][i] * x[i] for i in range(len(x))) + biases[c]
            for c in range(len(weights))]


# ------------------------------------------------------------------ 第 1 步


def step1_two_layers_are_one_line():
    print("=" * 60)
    print("第 1 步：把训练好的两层网络，乘开")
    print("=" * 60)
    print("先把上一章那个两层网络训出来（400 步），然后把它乘开：")
    print("把中间层那一层的式子代进输出层的式子里，一层层乘进去，")
    print("最后只剩下一组 w 和一个 b。")
    print()

    params, history = deep_train(depth=2, width=4, steps=400, lr=0.5, seed=0)
    weights, biases = collapse(params)

    print(f"训练之后 loss = {history[-1]:.6f}")
    print(f"乘开之后得到的「等效直线」：", end="")
    print(f"分数 = {weights[0][0]:+.4f} * x1 {weights[0][1]:+.4f} * x2 "
          f"{biases[0]:+.4f}")
    print()

    print(f"{'输入':>8}  {'两层网络的输出':>26}  {'等效直线的输出':>26}  {'差':>10}")
    worst = 0.0
    for x, label, _ in XOR_DATA:
        net = deep_forward(params, x)[-1]
        line = collapsed_output(weights, biases, x)
        diff = max(abs(net[c] - line[c]) for c in range(len(net)))
        worst = max(worst, diff)
        print(f"{str([int(v) for v in x]):>8}  "
              f"{net[0]:>+12.8f} /{net[1]:>+12.8f}  "
              f"{line[0]:>+12.8f} /{line[1]:>+12.8f}  {diff:>10.2e}")

    print()
    print(f"最大的差是 {worst:.2e} —— 这不是接近，这是完全相同。")
    print("两层网络和一条直线，输出一模一样。")
    print()


# ------------------------------------------------------------------ 第 2 步


def step2_any_depth():
    print("=" * 60)
    print("第 2 步：换成 3 层、5 层、10 层呢")
    print("=" * 60)
    print("我们不需要训练，也不需要知道权重是多少 ——")
    print("这个乘法对任何一叠「纯加权求和」的层都成立。")
    print("随便取几组随机权重试一下：")
    print()

    print(f"{'层数':>6}{'参数个数':>10}{'和等效直线相差':>18}")
    for depth, width in ((2, 4), (3, 4), (5, 3), (10, 2)):
        params = deep_init(depth, width, seed=7)
        weights, biases = collapse(params)
        gap = 0.0
        for x, _, _ in XOR_DATA:
            net = deep_forward(params, x)[-1]
            line = collapsed_output(weights, biases, x)
            gap = max(gap, max(abs(net[c] - line[c]) for c in range(len(net))))
        n_params = sum(len(w) * len(w[0]) + len(b)
                       for w, b in zip(params["ws"], params["bs"]))
        print(f"{depth:>6}{n_params:>10}{gap:>18.2e}")

    print()
    print("不管堆多少层，它永远等于一条直线。")
    print("中间层不是在「加工」数据，它只是把数据重新摆了一遍。")
    print()


# ------------------------------------------------------------------ 第 3 步


def step3_relu_changes_everything():
    print("=" * 60)
    print("第 3 步：给中间层加上 relu 之后")
    print("=" * 60)
    print("直线（严格说是这种加权求和）有一个性质：")
    print()
    print("    分数(u + v)  =  分数(u) + 分数(v) - 分数(0)")
    print()
    print("u、v 是两个随便取的输入。如果模型是直线，这个式子必须成立。")
    print("我们拿几组随机的 u、v 去试，看两边的差有多大。")
    print()

    rng = random.Random(99)
    pairs = [([rng.uniform(-2, 2), rng.uniform(-2, 2)],
              [rng.uniform(-2, 2), rng.uniform(-2, 2)]) for _ in range(4)]

    print("没有 relu 的两层网络：")
    params = deep_init(2, 4, seed=0)
    print(f"{'u':>16}{'v':>16}{'两边相差':>14}")
    for u, v in pairs:
        s_u = deep_forward(params, u)[-1]
        s_v = deep_forward(params, v)[-1]
        s_uv = deep_forward(params, [u[0] + v[0], u[1] + v[1]])[-1]
        s_0 = deep_forward(params, [0.0, 0.0])[-1]
        gap = max(abs(s_uv[c] - (s_u[c] + s_v[c] - s_0[c])) for c in range(2))
        print(f"{str([round(t, 3) for t in u]):>16}"
              f"{str([round(t, 3) for t in v]):>16}{gap:>14.2e}")
    print("  -> 两边完全相等。它确实是一条直线。")
    print()

    print("加上 relu 的两层网络（也是随机权重，还没训练）：")
    relu_params = relu_init(seed=0)
    print(f"{'u':>16}{'v':>16}{'两边相差':>14}")
    worst = 0.0
    for u, v in pairs:
        s_u = relu_forward(relu_params, u)[2]
        s_v = relu_forward(relu_params, v)[2]
        s_uv = relu_forward(relu_params, [u[0] + v[0], u[1] + v[1]])[2]
        s_0 = relu_forward(relu_params, [0.0, 0.0])[2]
        gap = max(abs(s_uv[c] - (s_u[c] + s_v[c] - s_0[c])) for c in range(2))
        worst = max(worst, gap)
        print(f"{str([round(t, 3) for t in u]):>16}"
              f"{str([round(t, 3) for t in v]):>16}{gap:>14.2e}")
    print(f"  -> 差得最远的一次差了 {worst:.4f}。它已经不是直线了。")
    print()
    print("只多了一个「负数变 0」，这个式子就再也对不上了。")
    print()
    print("不过 relu 也不是完全乱来：它把平面切成好几片，")
    print("在每一片里面，它确实还是一条直线。所以有的 u、v 碰巧落在同一片里，")
    print("两边依然对得上。但整体上，它已经不是一条直线了 ——")
    print("这正是中间层一直缺的那样东西。")
    print()


# ------------------------------------------------------------------ 第 4 步


def step4_relu_solves_it():
    print("=" * 60)
    print("第 4 步：加上 relu 之后，XOR 被解决了")
    print("=" * 60)

    params, history = relu_train(steps=5000, lr=0.5, seed=0)
    data = relu_data()

    print(f"{'模型':<24}{'训练后的 loss':>16}{'准确率':>10}")
    _, no_relu_history = deep_train(depth=2, width=4, steps=400, lr=0.5, seed=0)
    print(f"{'两层，中间不加东西':<24}{no_relu_history[-1]:>16.6f}{'0.50':>10}")
    print(f"{'两层，中间加 relu':<24}{history[-1]:>16.8f}"
          f"{relu_accuracy(params, data):>10.2f}")

    print()
    print("4 个点全部分对。概率都在 0.99 以上：")
    for x, label, note in XOR_DATA:
        _, _, scores = relu_forward(params, x)
        p = softmax(scores)
        guess = 0 if scores[0] >= scores[1] else 1
        print(f"  {note:<26} 判成 {guess}（答案 {label}），概率 {max(p):.4f}")
    print()
    print("回头看这三步：")
    print("  一层：loss 0.693147")
    print("  两层：loss 0.693147（我们以为加了层就会变）")
    print("  两层 + relu：loss 0.0000")
    print("中间层一直都在，缺的只是那一个「拐弯」。")


def main():
    step1_two_layers_are_one_line()
    step2_any_depth()
    step3_relu_changes_everything()
    step4_relu_solves_it()


if __name__ == "__main__":
    main()
