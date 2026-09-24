"""参数怎么更新。

第 6 章我们写下的那一行是：

    w = w - learning_rate * gradient

这个文件只是把那一行包装成了一个可以重复调用的对象。
Adam 是它的加强版，第 22 章才会用到。
"""

import numpy as np


class SGD:
    """最朴素的更新：沿着梯度的反方向走一小步。

        w ← w - lr * grad
    """

    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


class Adam:
    """SGD 的两个毛病，各打一个补丁。

    毛病一：梯度方向来回抖 → 用「动量」把历史梯度做平滑。
    毛病二：每个参数的尺度差别很大 → 用「自适应步长」除以其梯度方差的平方根。

    这里只是工具，不是本章的主角。看不懂细节可以先跳过去，
    它做的事情和 SGD 完全一样：拿到 grad，改 data。
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            # 前几步 m、v 都还很小，除以下面这一项把它们放大回正常尺度
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)
