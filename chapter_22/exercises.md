[English](exercises.en.md) | **中文**

# 第 22 章 练习

改 `after.py` 里的常数，重跑，看输出有没有变。每题都给一个能自己验证的结论。

---

## 1. 层数加到底

`after.py` 里有：

```python
LAYERS = 3        # 堆几个 Transformer Block
```

依次改成 6、12，每次都重跑。

- 参数量分别是多少？
- loss 变好了吗？（我这边 3 层是 0.0349，6 层和 12 层反而更差。）
- 如果变差了，是"层数太多不好"吗？还是别的什么原因？
  提示：想想这个模型一共有多少参数，语料一共多少个字符。

---

## 2. 头的数量

`HEADS = 4`，`DIM = 64`，所以每个头分到 `head_dim = 16`。

把 `HEADS` 改成 1，再改成 8、16，各跑一遍。

- 参数量变了吗？（提示：`query_weight` 的形状跟 `HEADS` 有关系吗？）
- loss 有变化吗？
- `HEADS = 16` 的时候 `head_dim` 是多少？它还能正常工作吗？
- 如果有人跟你说"多头注意力让参数量变大了"，你会怎么回答他？

---

## 3. 拆掉残差

`TransformerBlock.__call__` 里有这么两行：

```python
x = x + self.multi_head_attention(normed, mask)
...
x = x + hidden @ self.proj_weight + self.proj_bias
```

把两个 `x = x + ...` 都改成 `x = ...`，重跑 `after.py`。

- loss 变成多少？
- 它写出来的诗还对吗？
- 再把层数改成 6 试一次。为什么这个改动在小层数下也许还能忍，层数一多就完全不行了？

---

## 4. 不要每次都挑最大的

`continue_text` 里这一行：

```python
ids.append(int(logits.data[0, -1].argmax()))
```

改成按概率抽：

```python
probabilities = logits.softmax(axis=-1).data[0, -1]
rng = np.random.default_rng(0)
ids.append(int(rng.choice(len(probabilities), p=probabilities)))
```

（记得在函数开头建一次 `rng`，别每步都重建。）

- 同一个开头 `床前`，跑三次，结果一样吗？
- 开头换成 `举头望明月低头`，结果有意思吗？还是很快就变成乱码了？
- 猜一下原因。下一章会给你答案。

---

## 5. 上下文切一半

`CONTEXT = 32`。改成 8、16，各跑一遍 `after.py`。

- 训练集上的准确率变成多少？
- 用 `continue_text` 接 60 个字，还接得对吗？
- 想想第 22 章 `before.py` 里的那条曲线（窗口 2 到 32）。这和第 20 章那个"窗口拼接"模型遇到的问题是不是同一件事？
- 那么 attention 相比拼接窗口，赢在哪里？

---

## 6. 语料重复几遍

顶上这一行：

```python
CORPUS = (POEM_A + POEM_B) * 8
```

把 `8` 改成 `1`（语料只有 38 个字符），重跑。

- `build_batches` 能切出多少个窗口？
- loss 是多少？写出来的东西还对吗？
- 再把 `1` 改成 `80`，重跑。loss 呢？
- 这两次实验说明了什么？（第 28 章会把这件事实讲成一条规律。）
