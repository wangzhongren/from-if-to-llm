[English](exercises.en.md) | **中文**

# 第 27 章练习

## 1. 层数到底要多少才会出事

在 `experiment.py` 里把层数从 12 改成 4、8、16、24，各跑一遍，
把 post-LN 和我们的写法都量一遍：

| 层数 | 我们的写法（第 30 步底层梯度） | post-LN |
|---|---|---|
| 4 | | |
| 8 | | |
| 12 | | |
| 24 | | |

先猜再跑。然后回答：

- 层数少的时候，为什么两种写法都学得动？
- "post-LN 在 12 层会塌"这件事，是层数的问题还是学习率的问题？
  把学习率从 0.01 降到 0.003 再试一次。

## 2. 用学习率预热救一救

给 post-LN 那个模型加一段学习率预热：

```python
def lr_at(step, warmup=50, base=1e-2):
    if step < warmup:
        return base * (step + 1) / warmup
    return base
```

（在训练循环里改 `optimizer.param_groups[0]["lr"] = lr_at(step)`。）

- 能救回来吗？
- 如果能，说明"学习率太大"和"归一化放错位置"这两个原因是什么关系？
- 预热是 GPT 训练里的标准做法。现在你知道它为什么需要了。

## 3. 输出层前那次 LayerNorm 到底管什么

`after.py` 里 `final_ln=True` 是 GPT 的选择。现在做一组对照：

| 层数 | `final_ln=True` 的验证 loss | `final_ln=False` 的验证 loss |
|---|---|---|
| 4 | | |
| 12 | | |
| 24 | | |

- 层数越多，差别越明显吗？
- 想一想为什么：pre-LN 的残差流是"一路加下去"的，层数多了以后，
  最后那个向量的尺度会变成什么样？拿它直接去分类会有什么问题？
- 用代码验证你的猜想：把每一层输出后的标准差打印出来（`hidden.std()`），
  看看它随层数是变大还是变小。

## 4. 把位置表换回正弦表

`after.py` 里 `learned_pos=True` 用的是可学习的位置表。改成 `False` 重新训练，
比较：

| | 验证 loss | 生成的句子 |
|---|---|---|
| 可学习位置表 | | |
| 固定正弦表 | | |

然后想一个问题：如果测试时给它一个比 `block_size` 更长的句子，
两种做法各会发生什么？（提示：查表会越界，正弦表可以算更长。）

## 5. 亲手验证 pre-LN 的两条性质

测试文件里的 `test_我们的模型和第_22_章一样是_pre_ln` 验证了
"放大输入 10 倍，子层输出不变"。现在验证第二条：

```python
with torch.no_grad():
    hidden = model.tok_emb(x) + positions
    for i, block in enumerate(model.blocks):
        hidden = block(hidden, mask)
        print(i, hidden.std().item())
```

对 pre-LN 和 post-LN 各跑一遍：

- 两条曲线有什么不同？
- 为什么 post-LN 不需要最后那次 LayerNorm，而 pre-LN 需要？

## 6. 让 12 层的模型也"思考"一下

拿训练好的 12 层模型，试一个它不该会的句子：

```python
print(generate(model, "苹果", n_new=40, temperature=0.5, seed=3))
print(generate(model, "香蕉发布新手机", n_new=40, temperature=0.5, seed=3))
```

第二句是语料里不可能出现的组合（香蕉不会发布手机）。看它怎么接：

- 它把这句话当成科技话题还是食品话题？
- 它是从哪几个字开始"决定"话题的？
- 这算不算"理解"？（这个问题第 23、24 章问过一次，现在你有更大的模型可以问第二次。）
