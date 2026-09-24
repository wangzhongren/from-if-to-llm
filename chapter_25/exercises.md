[English](exercises.en.md) | **中文**

# 第 25 章练习

## 1. 把手写注意力换成封装好的

`after.py` 里 `Block.attention` 是我们自己写的（拆头、打分、掩码、softmax、拼回来）。

把它换成 `nn.MultiheadAttention`：

```python
self.attn = nn.MultiheadAttention(dim, n_head, batch_first=True)
...
out = self.attn(x, x, x, attn_mask=mask, need_weights=False)[0]
```

然后跑 `after.py`，和原来的 loss 曲线比一比。

提示：`experiment.py` 里的实验三已经把权重搬过去的方法写好了，
可以直接抄那段 `in_proj_weight` 的拼法，把我们的权重灌进 `nn.MultiheadAttention`，
这样就能验证"换过去以后，训练曲线在数值上是不是也一样"。

## 2. 用 PyTorch 自带的 Transformer 层

PyTorch 有一个现成的 `nn.TransformerEncoderLayer`。用 1 层它替换掉我们的 `Block`：

```python
layer = nn.TransformerEncoderLayer(dim, n_head, dim_feedforward=4*dim,
                                   batch_first=True, activation="relu")
x = layer(x, src_mask=mask)
```

它有两个参数值得摆弄：

- `norm_first`：`False` 是"先算再 LN"（我们的写法），`True` 是"先 LN 再算"
  （GPT 的写法，第 27 章会讲为什么）。
- `activation`：`"relu"` 还是 `"gelu"`。

跑一跑这四种组合，看看在我们这个小模型上 loss 有没有区别。
（剧透：几乎没有。第 27 章会解释为什么"几乎没有"这件事本身就是重点。）

## 3. 把模型做大，看倍数的变化

打开 `experiment.py`，把 `BENCH_SIZES` 改一改：

```python
BENCH_SIZES = {
    "80 万参数": (dict(dim=128, n_layer=4, n_head=4, block_size=32), 20),
    "800 万参数": (dict(dim=256, n_layer=8, n_head=4, block_size=32), 5),
}
```

重跑，看"倍数"那一列。注意两件事：

- 我们的引擎那两行的内存涨了多少？你的机器还撑得住吗？
- 速度的倍数是在变大还是变小？为什么？

## 4. 亲手验证转置那件事

`load_our_weights` 里，每个权重都做了一次 `.T`。把某个 `.T` 去掉，
让程序报错或者输出变得完全不靠谱，然后说清楚：

- `nn.Linear` 的 `weight` 形状是什么？
- 它的 `forward` 到底算的是 `x @ W` 还是 `x @ W.T`？
- 如果不转置，模型还能训练吗？loss 会变成什么样？

## 5. 关掉 `no_grad` 试试

`after.py` 的 `generate` 上面有一行 `@torch.no_grad()`。把它删掉再跑。

- 结果一样吗？
- 内存和时间有什么变化？
- 为什么生成的时候不需要计算图？（回想第 23 章：生成的时候我们在做什么，有没有"答案"可以对照？）

## 6. 让 before.py 崩溃一次

`before.py` 里那个大模型只跑了 5 步。把步数改成 50，
在自己的机器上看看内存涨到哪里、什么时候开始变卡。

这就是第 24 章那个问题的一半答案：**不完全是模型太小，也是我们的工具太小。**
