[English](exercises.en.md) | **中文**

# 第 21 章 练习

改 `after.py`（或 `experiment.py`）里的代码，重跑，看输出有没有变。每题都给一个能自己验证的结论。

---

## 1. 把对角线也挡掉会怎样

`causal_mask` 里这一行：

```python
return np.triu(np.ones((length, length), dtype=bool), k=1)
```

把 `k=1` 改成 `k=0`，重跑 `after.py`。注意：位置 0 那一整行现在全是 `X`。

先猜一下位置 0 会看到什么，再动手验证。验证办法：把 attention 的权重打印出来——

```python
weights = attention.masked_fill(causal_mask(length), -1e9).softmax(axis=-1)
print(np.round(weights.data[0][0], 4))
```

- 位置 0 那一行的权重加起来是多少？每一项是多少？它看到了哪些位置？
- 想明白之后，再解释：`-1e9` 为什么这一次**没有**起作用？（提示：去看 `toygrad` 里 `softmax` 的第一行代码。）
- 因果掩码那一组的 loss 变成多少？接出来的字还对吗？

---

## 2. -1e9 到底要多负才够

把 `after.py` 里的

```python
attention = attention.masked_fill(causal_mask(length), -1e9)
```

改成 `-1e1`、`-10`、`-1`、`0`，各跑一遍 `after.py`，看因果掩码那一档的 loss 和接出来的字。

- 哪几个值还能正常训练？
- 写几行代码打印一下 softmax 之后那些位置上的权重，看看它们到底是"很小的正数"，还是"精确的 0"。
- 为什么 -1e9 是安全的，-10 不是？（提示：attention 的分数本身有多大？量级是多少？）

---

## 3. 一格一格地放开

`experiment.py` 里有：

```python
LOOKAHEADS = (0, 1, 3, 7, 999)
```

改成 `(0, 1, 2, 3, 4, 5, 6, 7, 8)`，重跑。

- 从第几格开始，"遮住未来之后的 loss" 超过了 1？
- 那个临界值说明了什么？——提示：位置 i 能看到 i+k 的时候，它要预测的 i+1 在不在可见范围里？
- 有没有某一档是"能看见未来，但没学会抄"的？如果没找到，把训练步数改成 400 再试一次，看"抄"这件事是不是需要先学一阵子。

---

## 4. 掩码矩阵有多大

现在打印出来的掩码是 8×8。把 `after.py` 里那一行改成 `causal_mask(32)`，看看它多大。

- 32×32 里有几个 `X`？（提示：`32*31/2`。）
- 如果上下文是 1000，这个矩阵有多少个数？如果上下文是 128000（这是今天真实模型的量级）呢？
- 这个矩阵**不是**参数（它不需要学），但它每次前向传播都要存在。这跟第 20 章 exercises 第 4 题算的那个"参数量"是同一件事吗？为什么？

---

## 5. 把掩码用在错误的地方

`attention = attention.masked_fill(...)` 这一行，作用的是 **softmax 之前**的分数。

试着把它挪到 softmax **之后**：

```python
weights = attention.softmax(axis=-1)
weights = weights.masked_fill(causal_mask(length), 0.0)   # 注意这里填的是 0
attended = weights @ value
```

跑 `after.py`。

- 训练还能正常进行吗？接出来的字还对吗？
- 为什么这里填 `0`，而 softmax 之前要填 `-1e9`？（提示：softmax 之前那些数叫什么？第 5 章我们叫它……）
- 两种做法哪种更省事？更省事的那种为什么不能用在 softmax 之前？

---

## 6. 一句话回答

不用写代码。

这一章学到的是"训练 loss 低，不代表模型学会了你想让它学的东西"。

现在设想一个场景：你要训练一个模型，它会给出很低的 loss。你手上只有 loss 这一个数字。**在不重新设计实验的前提下，你能想到什么办法来判断它是不是在抄？**

（想完之后再去看 `experiment.py` 里"遮住未来之后再测一遍"这个做法。你想的办法和它像吗？）
