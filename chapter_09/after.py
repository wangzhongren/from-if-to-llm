"""第 9 章的 after：让机器自己算梯度。

上一章最后留下的话是：每加一层，梯度代码就要重推、重写、重新验一遍。

这个文件里没有一行梯度代码。

我们把每一个运算都改造成"会记住自己来历"的东西：
算的时候顺手记下"我是用谁、用什么算子算出来的"，
要梯度的时候，从最后那个数出发，沿着记下的路线倒着走一遍。

下面这个文件是自包含的，不需要任何第三方库 ——
一口气读完它，你就读完了一个自动求导引擎的全部。
（完整的数组版在仓库根目录 toygrad/tensor.py，道理完全一样。）
"""

import math
import random

# ------------------------------------------------------------------ 核心：一个会记事的数字


class Value:
    """一个数字，外加两样东西。

    1. 它是怎么被算出来的：`_prev` 记着它的输入，`_op` 记着用了什么运算。
       这两样合起来，就是一张"计算图"。
    2. 这个运算该怎么把梯度往回传：`_backward` 是一个函数，
       调用它就等于说"把我的梯度分给我的输入"。
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        # 默认什么都不做：叶子节点（参数）没有上游
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f}, op={self._op or '叶子'})"

    # -------------------------------------------------- 加减

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # 加法：梯度原样分给两边，一点都不打折
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    # -------------------------------------------------- 乘除幂

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # 乘法：我的梯度乘上对方的数值，送给对方
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f"**{n}")

        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    # -------------------------------------------------- 逐元素函数

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # e^x 的导数就是它自己
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # 被掐掉的（负数），梯度也一起掐掉
            self.grad += (out.grad if self.data > 0 else 0.0)

        out._backward = _backward
        return out

    # -------------------------------------------------- 反向传播

    def backward(self):
        """从自己出发，把梯度送回所有祖先节点。

        分两步：

        1. 拓扑排序。一个节点的梯度要等它所有的"下游"都算完才齐，
           所以必须先排出一个"谁在谁后面"的顺序。
        2. 倒着遍历这个顺序，逐个调用各自记下的 _backward。
        """
        order = []
        visited = set()

        def build(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for child in node._prev:
                build(child)
            order.append(node)

        build(self)

        self.grad = 1.0
        for node in reversed(order):
            node._backward()


# ------------------------------------------------------------------ 损失函数


def softmax_loss(scores, label):
    """两个分数 -> 一个损失。和前面几章一样：softmax 之后取正确类别的 -log。"""
    biggest = max(s.data for s in scores)
    exps = [(s - biggest).exp() for s in scores]
    total = exps[0] + exps[1]
    return -(exps[label] / total).log()


# ------------------------------------------------------------------ 模型
# 注意这个文件里没有 backward 函数。
# 一行都没有。

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


def make_params(seed=0, hidden_size=HIDDEN_SIZE):
    """参数就是一堆 Value。它们没有 _prev，因为它们是"起点"。"""
    rng = random.Random(seed)
    return {
        "w1": [[Value(rng.uniform(-0.5, 0.5)) for _ in range(NUM_FEATURES)]
               for _ in range(hidden_size)],
        "b1": [Value(0.0) for _ in range(hidden_size)],
        "w2": [[Value(rng.uniform(-0.5, 0.5)) for _ in range(hidden_size)]
               for _ in range(hidden_size)],
        "b2": [Value(0.0) for _ in range(hidden_size)],
        "w3": [[Value(rng.uniform(-0.5, 0.5)) for _ in range(hidden_size)]
               for _ in range(NUM_CLASSES)],
        "b3": [Value(0.0) for _ in range(NUM_CLASSES)],
    }


def all_params(params):
    """把参数字典摊平成一个列表。"""
    flat = []
    for value in params.values():
        for item in value:
            flat.extend(item if isinstance(item, list) else [item])
    return flat


def forward(params, x):
    """三层。中间两层算完都过一遍 relu。

    整个函数里没有一行字是在算梯度 —— 它只是在做乘法和加法。
    """
    hidden_size = len(params["b1"])

    layer1 = []
    for j in range(hidden_size):
        total = params["w1"][j][0] * x[0] + params["w1"][j][1] * x[1] + params["b1"][j]
        layer1.append(total.relu())

    layer2 = []
    for j in range(hidden_size):
        total = params["b2"][j]
        for k in range(hidden_size):
            total = total + params["w2"][j][k] * layer1[k]
        layer2.append(total.relu())

    scores = []
    for c in range(NUM_CLASSES):
        total = params["b3"][c]
        for k in range(hidden_size):
            total = total + params["w3"][c][k] * layer2[k]
        scores.append(total)
    return scores


def batch_loss(params, data):
    """把 4 个样本的损失加起来，除以 4。"""
    total = None
    for x, label in data:
        loss = softmax_loss(forward(params, x), label)
        total = loss if total is None else total + loss
    return total * (1.0 / len(data))


def predict(params, x):
    scores = [s.data for s in forward(params, x)]
    return 0 if scores[0] >= scores[1] else 1


def accuracy(params, data):
    correct = 0
    for x, label in data:
        if predict(params, x) == label:
            correct += 1
    return correct / len(data)


# ------------------------------------------------------------------ 训练


def train(steps=300, lr=0.5, seed=0):
    params = make_params(seed)
    flat = all_params(params)
    data = make_data()
    history = []

    for _ in range(steps):
        # 梯度是累加的，每一步开始前必须清零
        for p in flat:
            p.grad = 0.0

        loss = batch_loss(params, data)
        history.append(loss.data)

        loss.backward()          # <- 全部梯度，就这一行

        for p in flat:
            p.data -= lr * p.grad

    return params, history


# ------------------------------------------------------------------ main


def main():
    data = make_data()

    print("=" * 60)
    print("同一个任务，同一个网络结构，但这次没有一行梯度代码")
    print("=" * 60)
    print(f"网络：2 -> {HIDDEN_SIZE}（relu）-> {HIDDEN_SIZE}（relu）-> 2")
    print(f"参数：{len(all_params(make_params()))} 个 Value")
    print()

    params, history = train(steps=300, lr=0.5, seed=0)
    print(f"{'轮数':>4}{'loss':>14}")
    for i in range(0, 300, 50):
        print(f"{i + 1:>4}{history[i]:>14.8f}")
    print(f"{'300':>4}{history[-1]:>14.8f}")
    print()
    print(f"准确率：{accuracy(params, data):.2f}")
    print()

    print("=" * 60)
    print("逐句检查")
    print("=" * 60)
    for x, label, note in XOR_DATA:
        scores = [s.data for s in forward(params, x)]
        guess = predict(params, x)
        mark = "对" if guess == label else "错"
        print(f"{mark}  {note:<26} 判成 {guess}（答案 {label}）  "
              f"分数 {scores[0]:+.3f} / {scores[1]:+.3f}")
    print()

    print("=" * 60)
    print("我们到底写了什么")
    print("=" * 60)
    print("前向：一堆乘法和加法，加上两个 relu。")
    print("反向：什么都没写。loss.backward() 一行。")
    print()
    print("那梯度是从哪来的？")
    print("每一个 Value 在算出来的时候，顺手记下了『我是怎么来的』，")
    print("以及『该怎么把梯度还给我的输入』。")
    print("backward() 只是沿着这些记录，倒着走了一遍。")


if __name__ == "__main__":
    main()
