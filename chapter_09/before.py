"""第 9 章的 before：三层网络，每一层梯度都是人推的。

第 8 章结束的时候，我们的网络是这个样子：

    输入 -> 中间层（relu）-> 输出 -> softmax -> loss

梯度是我们拿链式法则一段一段推出来、一行一行写下来的。
两层的时候还好。这一章我们先把网络堆到三层，
看看"手写"这件事会变成什么样。

这个文件能跑通，loss 也能降下去。问题不在它能不能跑。
"""

import inspect
import math
import random

NUM_FEATURES = 2
HIDDEN_SIZE = 4
NUM_CLASSES = 2

XOR_DATA = [
    ([0.0, 0.0], 1, "苹果很好吃 / 香蕉很好吃"),
    ([0.0, 1.0], 0, "苹果很好吃 / 苹果发布新手机"),
    ([1.0, 0.0], 0, "苹果发布新手机 / 苹果很甜"),
    ([1.0, 1.0], 1, "苹果发布新手机 / 苹果发布新芯片"),
]


def make_data():
    return [(x, label) for x, label, _ in XOR_DATA]


def make_params(seed=0):
    rng = random.Random(seed)
    return {
        "w1": [[rng.uniform(-0.5, 0.5) for _ in range(NUM_FEATURES)]
               for _ in range(HIDDEN_SIZE)],
        "b1": [0.0] * HIDDEN_SIZE,
        "w2": [[rng.uniform(-0.5, 0.5) for _ in range(HIDDEN_SIZE)]
               for _ in range(HIDDEN_SIZE)],
        "b2": [0.0] * HIDDEN_SIZE,
        "w3": [[rng.uniform(-0.5, 0.5) for _ in range(HIDDEN_SIZE)]
               for _ in range(NUM_CLASSES)],
        "b3": [0.0] * NUM_CLASSES,
    }


def relu(value):
    return value if value > 0.0 else 0.0


# ------------------------------------------------------------------ 前向


def forward(params, x):
    """三层。每一层算完都过一遍 relu。

    这里只是把上一章的代码又多抄了一遍 —— 多一层，多一段。
    """
    z1 = [params["b1"][j] + params["w1"][j][0] * x[0] + params["w1"][j][1] * x[1]
          for j in range(HIDDEN_SIZE)]
    h1 = [relu(v) for v in z1]

    z2 = []
    for j in range(HIDDEN_SIZE):
        total = params["b2"][j]
        for k in range(HIDDEN_SIZE):
            total += params["w2"][j][k] * h1[k]
        z2.append(total)
    h2 = [relu(v) for v in z2]

    scores = []
    for c in range(NUM_CLASSES):
        total = params["b3"][c]
        for k in range(HIDDEN_SIZE):
            total += params["w3"][c][k] * h2[k]
        scores.append(total)

    return z1, h1, z2, h2, scores


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


# ------------------------------------------------------------------ 反向
# 下面这一段，每一行都是人拿着链式法则推出来的。


def backward(params, data):
    """这就是"每层梯度都要手推"的样子。

    读的时候可以注意三件事：
      - 输出层那一段，和一层模型时几乎一样；
      - 每往前退一层，就要重写一遍"误差乘上权重再传回去"；
      - 每退一层，还要多写一遍 relu 的开关（z > 0）。
    """
    grads = {
        "w1": [[0.0] * NUM_FEATURES for _ in range(HIDDEN_SIZE)],
        "b1": [0.0] * HIDDEN_SIZE,
        "w2": [[0.0] * HIDDEN_SIZE for _ in range(HIDDEN_SIZE)],
        "b2": [0.0] * HIDDEN_SIZE,
        "w3": [[0.0] * HIDDEN_SIZE for _ in range(NUM_CLASSES)],
        "b3": [0.0] * NUM_CLASSES,
    }
    n = len(data)

    for x, label in data:
        z1, h1, z2, h2, scores = forward(params, x)
        p = softmax(scores)

        # 输出层
        dscore = [(p[c] - (1.0 if c == label else 0.0)) / n
                  for c in range(NUM_CLASSES)]
        for c in range(NUM_CLASSES):
            grads["b3"][c] += dscore[c]
            for k in range(HIDDEN_SIZE):
                grads["w3"][c][k] += dscore[c] * h2[k]

        # 第二层中间层：误差乘上 w3 传回来，再乘上 relu 的开关
        dz2 = []
        for j in range(HIDDEN_SIZE):
            total = 0.0
            for c in range(NUM_CLASSES):
                total += dscore[c] * params["w3"][c][j]
            dz2.append(total if z2[j] > 0.0 else 0.0)
        for j in range(HIDDEN_SIZE):
            grads["b2"][j] += dz2[j]
            for k in range(HIDDEN_SIZE):
                grads["w2"][j][k] += dz2[j] * h1[k]

        # 第一层中间层：再来一遍，这次乘的是 w2
        dz1 = []
        for k in range(HIDDEN_SIZE):
            total = 0.0
            for j in range(HIDDEN_SIZE):
                total += dz2[j] * params["w2"][j][k]
            dz1.append(total if z1[k] > 0.0 else 0.0)
        for k in range(HIDDEN_SIZE):
            grads["b1"][k] += dz1[k]
            for i in range(NUM_FEATURES):
                grads["w1"][k][i] += dz1[k] * x[i]

    return grads


# ------------------------------------------------------------------ 数值法验证
# 写完之后，我们不敢直接相信它。用第 6 章的数值法回头对一遍。


def all_parameter_slots(params):
    """把参数字典里每一个位置摊成一个 (名字, 第一层下标, 第二层下标) 列表。"""
    slots = []
    for name in ("w1", "w2", "w3"):
        for a in range(len(params[name])):
            for b in range(len(params[name][a])):
                slots.append((name, a, b))
    return slots


def compare_gradients(params, data, how_many=6):
    """把手写梯度最大的那几个参数，和数值法对一对。

    挑最大的几个是因为：有些中间单元被 relu 掐死了，
    它们的梯度本来就是 0，拿 0 去比 0 说明不了什么。
    """
    grads = backward(params, data)

    slots = sorted(all_parameter_slots(params),
                   key=lambda slot: -abs(grads[slot[0]][slot[1]][slot[2]]))
    rows = []

    for name, a, b in slots[:how_many]:
        row = params[name][a]
        original = row[b]

        row[b] = original + 1e-6
        plus = loss_of(params, data)
        row[b] = original - 1e-6
        minus = loss_of(params, data)
        row[b] = original
        numeric = (plus - minus) / (2 * 1e-6)

        rows.append((name, a, b, grads[name][a][b], numeric))

    return rows


# ------------------------------------------------------------------ 训练


def train(steps=300, lr=0.5, seed=0):
    params = make_params(seed)
    data = make_data()
    history = []

    for _ in range(steps):
        grads = backward(params, data)
        for key in params:
            for a in range(len(params[key])):
                if isinstance(params[key][a], list):
                    for b in range(len(params[key][a])):
                        params[key][a][b] -= lr * grads[key][a][b]
                else:
                    params[key][a] -= lr * grads[key][a]
        history.append(loss_of(params, data))

    return params, history


def predict(params, x):
    scores = forward(params, x)[-1]
    return 0 if scores[0] >= scores[1] else 1


def accuracy(params, data):
    correct = 0
    for x, label in data:
        if predict(params, x) == label:
            correct += 1
    return correct / len(data)


def count_lines(function):
    """数一个函数有多少行（包括定义行和注释行）。"""
    return len(inspect.getsourcelines(function)[0])


# ------------------------------------------------------------------ main


def main():
    data = make_data()
    params, history = train(steps=300, lr=0.5, seed=0)

    print("=" * 60)
    print("手写的三层网络")
    print("=" * 60)
    print(f"网络：2 -> {HIDDEN_SIZE}（relu）-> {HIDDEN_SIZE}（relu）-> 2")
    print(f"训练 300 步之后：loss = {history[-1]:.8f}，"
          f"准确率 = {accuracy(params, data):.2f}")
    print()
    print("它是能跑的。问题不在它能不能跑。")
    print()

    print("=" * 60)
    print("我们为了让它跑起来，手写了多少东西")
    print("=" * 60)
    forward_lines = count_lines(forward)
    backward_lines = count_lines(backward)
    print(f"前向 forward：{forward_lines} 行")
    print(f"反向 backward：{backward_lines} 行")
    print(f"一共 {forward_lines + backward_lines} 行，全是人手写的。")
    print()
    print("其中真正是「推」出来的部分：")
    print("  「输出层的误差」       1 段   （第 6 章推的，一直没变）")
    print("  「误差乘权重往回传」   2 段   （每多一层，就多一段）")
    print("  「relu 的开关」        2 处   （每多一层，就多一处）")
    print()

    print("=" * 60)
    print("写完之后，我们还得确认没写错")
    print("=" * 60)
    print("梯度写错了不会报错，只会安静地越学越差。")
    print("所以每一段新写的梯度，都要用第 6 章的数值法回头对一遍：")
    print()
    print(f"{'参数':>10}{'手写的梯度':>16}{'数值法的梯度':>16}{'差':>12}")
    worst = 0.0
    for name, a, b, hand, numeric in compare_gradients(params, data):
        diff = abs(hand - numeric)
        worst = max(worst, diff)
        label = f"{name}[{a}][{b}]"
        print(f"{label:>10}{hand:>16.8f}{numeric:>16.8f}{diff:>12.2e}")
    print()
    print(f"最大的差是 {worst:.2e} —— 这次侥幸是对的。")
    print("（注意「侥幸」两个字。写错的地方，程序是不会报错的。）")
    print()

    print("=" * 60)
    print("那现在想加第四层呢")
    print("=" * 60)
    print("要做的每一件事，都得再做一遍：")
    print("  1. forward 里再抄一段（多一层加权求和 + 多一个 relu）")
    print("  2. backward 里再推一段（多一次「误差乘权重」，多一个 relu 开关）")
    print("  3. 参数里再多一组 w4、b4，训练循环里再多一处下标")
    print("  4. 再用数值法验一遍新写的那一段")
    print()
    print("第五层、第六层……每一层都是这样。")
    print("而 GPT-3 有 96 层。")


if __name__ == "__main__":
    main()
