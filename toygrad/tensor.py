"""toygrad.tensor —— 一个 300 行的自动求导引擎（第 9 章的产物）。

第 6 章我们用数值法求梯度：

    w += 0.001
    loss 变小了没有？

一次只能问一个参数，而且每个参数都要重新跑一遍整个模型。
第 9 章我们把这个过程自动化：

    每个 Tensor 记住「我是怎么被算出来的」，
    backward() 的时候沿着原路把梯度倒着送回去。

这个文件就是那件事的实现。它刻意写得很慢、很啰嗦，
因为它的目的是让人看懂，不是跑得快。
"""

import numpy as np

# ---------------------------------------------------------------- 全局开关

_GRAD_ENABLED = True


class no_grad:
    """推理时用的开关：这个块里的运算不再记录计算图。

    PyTorch 里叫 ``torch.no_grad()``，做的事情一模一样。
    """

    def __enter__(self):
        global _GRAD_ENABLED
        self._prev = _GRAD_ENABLED
        _GRAD_ENABLED = False

    def __exit__(self, *exc):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev
        return False


def _need_grad(*tensors):
    """只要有一个输入需要梯度，输出就参与计算图。"""
    return _GRAD_ENABLED and any(t.requires_grad for t in tensors)


def _as_tensor(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def _unbroadcast(grad, shape):
    """把被广播撑大的梯度「加」回原来的形状。

    例：a 的形状是 (3,)，b 的形状是 (4, 3)，a + b 之后 a 被复制了 4 份。
    反向传播时这 4 份梯度必须加起来还给 a。
    """
    grad = np.asarray(grad, dtype=np.float64)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


# ---------------------------------------------------------------- 核心类


class Tensor:
    """一个带着「我该怎么被求导」记忆的多维数组。

    属性
    ----
    data : np.ndarray
        真正的数值。所有新机制最后都只是在改这个数组。
    grad : np.ndarray
        和 data 形状相同，累积损失对它的梯度。每次 backward 之前记得清零。
    requires_grad : bool
        这个张量是不是「参数」——是需要被学习、被更新的东西。
    """

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)
        self.grad = np.zeros_like(self.data)
        # 不需要梯度的张量不记录来路，这样推理时不会一直攒着计算图。
        self._prev = tuple(_children) if self.requires_grad else ()
        self._op = _op
        self._backward = lambda: None  # 默认什么也不做

    # -------------------------------------------------- 一点便利属性

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        return self.transpose()

    def item(self):
        return float(self.data.reshape(-1)[0]) if self.data.size == 1 else float(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    # -------------------------------------------------- 加、减

    def __add__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data + other.data, _need_grad(self, other), (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * (-1.0)

    def __sub__(self, other):
        return self + (-_as_tensor(other))

    def __rsub__(self, other):
        return _as_tensor(other) + (-self)

    # -------------------------------------------------- 乘、除、幂

    def __mul__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data * other.data, _need_grad(self, other), (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __pow__(self, n):
        """只支持常数次幂，足够我们用了。"""
        out = Tensor(self.data ** n, _need_grad(self), (self,), f"**{n}")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * n * (self.data ** (n - 1))

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (_as_tensor(other) ** -1)

    def __rtruediv__(self, other):
        return _as_tensor(other) * (self ** -1)

    # -------------------------------------------------- 矩阵乘法

    def __matmul__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data @ other.data, _need_grad(self, other), (self, other), "@")

        def _backward():
            if self.requires_grad:
                g = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad += _unbroadcast(g, self.data.shape)
            if other.requires_grad:
                g = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += _unbroadcast(g, other.data.shape)

        out._backward = _backward
        return out

    # -------------------------------------------------- 逐元素函数

    def exp(self):
        out = Tensor(np.exp(self.data), _need_grad(self), (self,), "exp")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * out.data  # d(e^x)/dx = e^x，就是它自己

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), _need_grad(self), (self,), "log")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), _need_grad(self), (self,), "relu")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _need_grad(self), (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (1 - t * t)

        out._backward = _backward
        return out

    def gelu(self):
        """GPT 系列用的激活函数。第 25 章之后的模型里会见到它。"""
        c = np.sqrt(2.0 / np.pi)
        inner = c * (self.data + 0.044715 * self.data ** 3)
        t = np.tanh(inner)
        out = Tensor(0.5 * self.data * (1 + t), _need_grad(self), (self,), "gelu")

        def _backward():
            if self.requires_grad:
                dinner = c * (1 + 3 * 0.044715 * self.data ** 2)
                self.grad += out.grad * (0.5 * (1 + t) + 0.5 * self.data * (1 - t * t) * dinner)

        out._backward = _backward
        return out

    def maximum(self, value):
        """逐元素取较大值，例如 maximum(x, 0) 就是 relu。"""
        out = Tensor(np.maximum(self.data, value), _need_grad(self), (self,), "maximum")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > value)

        out._backward = _backward
        return out

    # -------------------------------------------------- 形状变换

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), _need_grad(self), (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = tuple(reversed(range(self.data.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = Tensor(np.transpose(self.data, axes), _need_grad(self), (self,), "transpose")
        inverse = tuple(np.argsort(axes))

        def _backward():
            if self.requires_grad:
                self.grad += np.transpose(out.grad, inverse)

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        """支持切片和「按整数数组取」，后者就是 embedding 查表要用的。"""
        out = Tensor(self.data[idx], _need_grad(self), (self,), "getitem")

        def _backward():
            if self.requires_grad:
                # add.at 而不是 += ：同一个位置被取了多次时要乖乖累加
                np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    # -------------------------------------------------- 归约

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _need_grad(self), (self,), "sum")

        def _backward():
            if not self.requires_grad:
                return
            g = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                shape = list(self.data.shape)
                for a in axes:
                    shape[a] = 1
                g = g.reshape(shape)
            self.grad += np.broadcast_to(g, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), _need_grad(self), (self,), "mean")

        def _backward():
            if not self.requires_grad:
                return
            if axis is None:
                n = self.data.size
                g = out.grad
            else:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                n = int(np.prod([self.data.shape[a] for a in axes]))
                g = out.grad
                if not keepdims:
                    shape = list(self.data.shape)
                    for a in axes:
                        shape[a] = 1
                    g = g.reshape(shape)
            self.grad += np.broadcast_to(g / n, self.data.shape)

        out._backward = _backward
        return out

    # -------------------------------------------------- 常用组合

    def softmax(self, axis=-1):
        """把一组分数变成一组加起来等于 1 的概率。

        先减去最大值只是为了不溢出，不影响结果。
        """
        x = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(x)
        y = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(y, _need_grad(self), (self,), "softmax")

        def _backward():
            if self.requires_grad:
                dy = out.grad
                self.grad += y * (dy - (dy * y).sum(axis=axis, keepdims=True))

        out._backward = _backward
        return out

    def masked_fill(self, mask, value):
        """mask 为 True 的位置填上 value。

        第 21 章的因果掩码靠它实现：把「未来」的位置填成 -1e9，
        这样 softmax 之后那边的权重就约等于 0。
        """
        mask = np.asarray(mask, dtype=bool)
        out = Tensor(
            np.where(mask, value, self.data), _need_grad(self), (self,), "masked_fill"
        )

        def _backward():
            if self.requires_grad:
                self.grad += np.where(mask, 0.0, out.grad)

        out._backward = _backward
        return out

    # -------------------------------------------------- 反向传播

    def backward(self):
        """从自己出发，把梯度送回所有祖先节点。

        分两步：
        1. 拓扑排序，保证「算一个节点的梯度时，它上游的梯度已经齐了」。
        2. 倒着遍历，逐个调用各自记下的 _backward。
        """
        topo, visited = [], set()

        def build(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            for child in t._prev:
                build(child)
            topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)


# ---------------------------------------------------------------- 函数式接口
# 有些运算写成函数比写成方法更清楚。它们的 backward 和方法版是同一套道理。


def cross_entropy(logits, targets):
    """交叉熵：衡量「预测的概率分布」和「正确答案」差多远。

    logits  : 形状 (..., 词表大小) 的原始分数（softmax 之前）
    targets : 形状 (...)  的整数，正确答案是第几个词
    返回    : 一个标量 Tensor —— 所有位置的平均损失
    """
    z = logits.data
    z = z - z.max(axis=-1, keepdims=True)          # 防溢出
    logsumexp = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    logp = z - logsumexp                            # log_softmax，形状 (..., V)

    V = logp.shape[-1]
    flat_logp = logp.reshape(-1, V)
    flat_t = np.asarray(targets).reshape(-1)
    n = flat_t.size

    loss = -flat_logp[np.arange(n), flat_t].mean()
    out = Tensor(loss, logits.requires_grad, (logits,), "cross_entropy")

    def _backward():
        if not logits.requires_grad:
            return
        p = np.exp(flat_logp)                       # 已经就是 softmax 的结果
        p[np.arange(n), flat_t] -= 1.0              # 预测概率 - 1（只对正确那一项）
        p /= n
        logits.grad += p.reshape(logits.data.shape) * out.grad

    out._backward = _backward
    return out


def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm：把最后一维拉成均值 0、方差 1，再用 weight/bias 缩放平移。

    第 18 章会解释它为什么必须存在。
    """
    mu = x.data.mean(axis=-1, keepdims=True)
    var = x.data.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x.data - mu) * inv
    y = xhat * weight.data + bias.data

    out = Tensor(y, _need_grad(x, weight, bias), (x, weight, bias), "layer_norm")

    def _backward():
        dy = out.grad
        D = x.data.shape[-1]
        if x.requires_grad:
            dxhat = dy * weight.data
            dvar = (dxhat * (x.data - mu) * (-0.5) * inv ** 3).sum(axis=-1, keepdims=True)
            dmu = (-dxhat * inv).sum(axis=-1, keepdims=True) \
                + dvar * (-2.0 * (x.data - mu)).mean(axis=-1, keepdims=True)
            dx = dxhat * inv + dvar * 2.0 * (x.data - mu) / D + dmu / D
            x.grad += dx
        if weight.requires_grad:
            weight.grad += _unbroadcast(dy * xhat, weight.data.shape)
        if bias.requires_grad:
            bias.grad += _unbroadcast(dy, bias.data.shape)

    out._backward = _backward
    return out


def embedding(table, idx):
    """查表：table 形状 (词表大小, 维度)，idx 是任意形状的整数数组。

    返回形状 idx.shape + (维度,)。反向传播就是把梯度散回对应的行上。
    """
    idx = np.asarray(idx)
    D = table.data.shape[-1]
    out = Tensor(table.data[idx], _need_grad(table), (table,), "embedding")

    def _backward():
        if table.requires_grad:
            np.add.at(table.grad, idx.reshape(-1), out.grad.reshape(-1, D))

    out._backward = _backward
    return out


def masked_fill(x, mask, value):
    """函数版的 masked_fill，见 Tensor.masked_fill。"""
    return x.masked_fill(mask, value)


# ---------------------------------------------------------------- 参数初始化


def randn(*shape, scale=1.0, requires_grad=False, seed=None):
    """按正态分布随机初始化参数。"""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    rng = np.random.default_rng(seed)
    return Tensor(rng.standard_normal(shape) * scale, requires_grad=requires_grad)


def zeros(*shape, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.zeros(shape), requires_grad=requires_grad)
