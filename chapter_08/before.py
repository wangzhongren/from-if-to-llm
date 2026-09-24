"""第 8 章的 before：加一层不行？那就多加几层。

上一章我们在输入和分数之间加了一层，结果 loss 一动不动。

最直觉的反应是：**一层不够，那就堆。**

这个文件把网络堆到 3 层、5 层、10 层，每一层都是加权求和，
中间不加别的东西，看看 loss 会不会动。

梯度这里我们直接用第 6 章的数值法 —— 堆了几层都要重推一遍链式法则，
太麻烦了，而数值法换什么结构都能用。
"""

import math
import random

NUM_FEATURES = 2
NUM_CLASSES = 2

XOR_DATA = [
    ([0.0, 0.0], 1, "苹果很好吃 / 香蕉很好吃"),
    ([0.0, 1.0], 0, "苹果很好吃 / 苹果发布新手机"),
    ([1.0, 0.0], 0, "苹果发布新手机 / 苹果很甜"),
    ([1.0, 1.0], 1, "苹果发布新手机 / 苹果发布新芯片"),
]


def make_data():
    return [(x, label) for x, label, _ in XOR_DATA]


# ------------------------------------------------------------------ 可以堆任意多层的网络


def init_params(depth, width, seed=0):
    """depth 层，每层 width 个单元，最后一层输出 2 个分数。

    每一层都是纯加权求和：拿上一层的输出，乘权重、加偏置、交给下一层。
    """
    rng = random.Random(seed)
    sizes = [NUM_FEATURES] + [width] * (depth - 1) + [NUM_CLASSES]

    params = {"ws": [], "bs": []}
    for layer in range(depth):
        fan_in, fan_out = sizes[layer], sizes[layer + 1]
        params["ws"].append(
            [[rng.uniform(-0.5, 0.5) for _ in range(fan_in)] for _ in range(fan_out)]
        )
        params["bs"].append([0.0] * fan_out)
    return params


def forward(params, x):
    """一层接一层，中间不加任何别的东西。返回每一层的输出。"""
    activations = [x]
    current = x
    for layer in range(len(params["ws"])):
        weights = params["ws"][layer]
        biases = params["bs"][layer]
        nxt = []
        for row in range(len(weights)):
            total = biases[row]
            for k in range(len(current)):
                total += weights[row][k] * current[k]
            nxt.append(total)
        activations.append(nxt)
        current = nxt
    return activations


def softmax(scores):
    biggest = max(scores)
    exps = [math.exp(s - biggest) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def loss_of(params, data):
    total = 0.0
    for x, label in data:
        scores = forward(params, x)[-1]
        total += -math.log(softmax(scores)[label])
    return total / len(data)


def numeric_gradient(params, data, h=1e-3):
    """第 6 章的数值梯度，一字不改地用在这里。

    它只认 loss 这个函数，不关心网络长什么样 —— 这就是它的好处。
    """
    grads = {"ws": [], "bs": []}
    for layer in range(len(params["ws"])):
        w_grad = []
        for row in range(len(params["ws"][layer])):
            row_grad = []
            for k in range(len(params["ws"][layer][row])):
                original = params["ws"][layer][row][k]
                params["ws"][layer][row][k] = original + h
                plus = loss_of(params, data)
                params["ws"][layer][row][k] = original - h
                minus = loss_of(params, data)
                params["ws"][layer][row][k] = original
                row_grad.append((plus - minus) / (2 * h))
            w_grad.append(row_grad)
        grads["ws"].append(w_grad)

        b_grad = []
        for k in range(len(params["bs"][layer])):
            original = params["bs"][layer][k]
            params["bs"][layer][k] = original + h
            plus = loss_of(params, data)
            params["bs"][layer][k] = original - h
            minus = loss_of(params, data)
            params["bs"][layer][k] = original
            b_grad.append((plus - minus) / (2 * h))
        grads["bs"].append(b_grad)
    return grads


def count_params(params):
    total = 0
    for layer in range(len(params["ws"])):
        total += len(params["ws"][layer]) * len(params["ws"][layer][0])
        total += len(params["bs"][layer])
    return total


def train(depth, width, steps=400, lr=0.5, seed=0):
    params = init_params(depth, width, seed)
    data = make_data()
    history = []

    for _ in range(steps):
        grads = numeric_gradient(params, data)
        for layer in range(len(params["ws"])):
            for row in range(len(params["ws"][layer])):
                for k in range(len(params["ws"][layer][row])):
                    params["ws"][layer][row][k] -= lr * grads["ws"][layer][row][k]
            for k in range(len(params["bs"][layer])):
                params["bs"][layer][k] -= lr * grads["bs"][layer][k]
        history.append(loss_of(params, data))

    return params, history


# ------------------------------------------------------------------ main


def main():
    print("=" * 60)
    print("一层不行，那就堆")
    print("=" * 60)
    print("每一层都是纯加权求和，中间不加任何别的东西。")
    print("层数越多，能调的参数越多 —— 直觉上应该越厉害。")
    print("每多一层，我们就要多推一截链式法则，所以这里用数值梯度。")
    print()

    print(f"{'层数':>4}{'每层宽度':>10}{'参数个数':>10}{'每步要跑几次前向':>18}"
          f"{'训练后的 loss':>16}")
    for depth, width in ((1, 4), (2, 4), (3, 4), (5, 2), (10, 2)):
        params = init_params(depth, width, seed=0)
        n_params = count_params(params)
        forwards = n_params * 2
        _, history = train(depth, width, steps=400, lr=0.5, seed=0)
        print(f"{depth:>4}{width:>10}{n_params:>10}{forwards:>18}"
              f"{history[-1]:>16.6f}")

    print()
    print("（层数一多，每层再放 4 个单元就太慢了 —— 数值梯度每走一步要跑的")
    print("  前向次数，是参数个数的两倍。所以深的那几层我们放窄了一点，")
    print("  这不影响结论：反正 loss 一动不动。）")
    print()
    print("1 层、2 层、3 层、5 层、10 层 —— loss 全部停在 0.693147。")
    print("注意最右边那一列：不管堆多少层，它连小数点后六位都没变过。")
    print()
    print("堆深这条路，走不通。")
    print()
    print("问题不在「层数够不够」。我们得换一个方向去想：")
    print("每一层算完之后，我们是不是漏做了什么？")


if __name__ == "__main__":
    main()
