[English](exercises.en.md) | **中文**

# 第 9 章练习

## 1. 给引擎加一个算子

`after.py` 里的 `Value` 只有 `+ - * / ** exp log relu`。

给它加一个 `tanh`：

```python
def tanh(self):
    t = math.tanh(self.data)
    out = Value(t, (self,), "tanh")
    def _backward():
        self.grad += (1 - t * t) * out.grad
    out._backward = _backward
    return out
```

加完之后，用数值法验一下（照抄 `test_scalar_engine_matches_numeric_gradient` 的写法）。

再试着自己加一个 `maximum(other)`（取较大值，`maximum(x, 0)` 就是 relu）。
提示：`x` 比 `other` 大的时候，梯度给 x；否则给 other。

## 2. 把 `+=` 改成 `=`

`_backward` 里所有的 `self.grad += ...`，全部改成 `self.grad = ...`。

然后跑这个：

```python
x = Value(3.0)
(x * x).backward()
print(x.grad)
```

结果是多少？应该是多少？差在哪里？

改回去之后，再跑一遍完整训练，看 loss 还能不能正常下降。

## 3. 把拓扑排序删掉

现在的 `backward()` 先排一个序，再倒着走。

改成不排序，直接递归往下走：

```python
def backward_broken(node):
    node.grad = 1.0

    def go(n):
        n._backward()
        for child in n._prev:
            go(child)

    go(node)
```

然后跑这一段：

```python
a = Value(2.0)
b = a * 3.0     # b = 3a
c = b * a       # c = 3a²
d = b + c       # d = 3a + 3a²
backward_broken(d)
print(a.grad)
```

正确答案是 `3 + 6a = 15`。实际跑出来是多少？

多出来的那一部分是从哪来的？

（提示：`b` 有两个下游 —— `d` 和 `c`。
不排序的话，`b._backward()` 可能在 `c` 还没把梯度给它的时候就跑了。
那时候 `b.grad` 还没收齐，它按一个偏小的值往下传。
数一下：偏小了多少？最后又多算了多少？）

## 4. 为什么 backward 一开始要设成 1.0

`backward()` 的第一句是 `self.grad = 1.0`，不是 `self.grad += 1.0`。

如果你算两次 `loss.backward()`，loss 自己的梯度是 1 还是 2？

这件事重要吗？把它和"参数为什么要清零"放在一起想一想。

## 5. 训练的时候把 `forward` 套上 `no_grad()`

用 `toygrad` 搭个小网络，然后在训练循环里这样写：

```python
opt.zero_grad()
with no_grad():
    logits = (Tensor(X) @ w1 + b1).relu() @ w2 + b2
    loss = cross_entropy(logits, Y)
loss.backward()
opt.step()
```

跑 50 步，把每一步的 loss 打出来。

会发生什么？程序报错了吗？参数动了吗？

（这一题值钱的地方在这里：**梯度全变成 0 了，可是没有任何报错。
训练只是安静地停在那里。** 以后你在 PyTorch 里写错一行，
看到 loss 一动不动，先想想是不是这个。）

## 6. 一次性算 4 个样本 vs 一个一个算

`after.py` 里，我们把 4 个样本的 loss 加起来当总 loss。

现在改成**一个一个算**：每算完一个样本就 `backward()` 一次，
把 4 个样本的梯度攒起来，再更新一次参数。

跑一下，结果和原来的相比：loss 的下降快慢一样吗？最终准确率一样吗？

想清楚：两种做法算出来的梯度，是不是同一个东西？
（提示：和的导数 = 导数的和。但**累加**这个性质在这里帮了我们大忙 ——
如果 `backward()` 是覆盖而不是累加，第二种写法就直接废了。）

## 7. 用手把一张图算一遍

取 `a = 2`、`b = 3`、`c = 4`，画出 `loss = a * b + c` 的计算图。

然后在纸上走一遍：

- 每个节点的 `data` 是多少；
- 每个节点的 `grad` 是多少；
- 每一条边上传的是哪个数。

走完之后，把 `a` 改成 `2.001` 重新算一遍 `loss`，看变化了多少。
和你在纸上算出来的 `a.grad` 对得上吗？

## 8. 这个引擎慢在哪

`after.py` 跑 300 步（每步 4 个样本）用不了 1 秒。
但 `toygrad` 那个数组版，同样的任务只用了几十毫秒。

数一数：标量版每算一个样本，要造多少个 `Value` 对象？
如果有 100 万条训练数据，这个数字会变成多少？

（这就是为什么真实的框架用数组，而不是一个数一个数地算。
但两者的道理，和你手写的这个 `Value` 一模一样。）
