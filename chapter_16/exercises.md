[English](exercises.en.md) | **中文**

# 第 16 章 练习

改 `after.py` 和 `experiment.py` 里的东西，重跑，看输出有没有变。

---

## 1. 位置编码加多少合适

`after.py` 里位置向量是**直接加**上去的：

```python
vector = Tensor(word_vectors + position_table)
```

试着乘一个系数再试：

```python
vector = Tensor(word_vectors + 0.1 * position_table)
vector = Tensor(word_vectors + 10.0 * position_table)
```

跑一遍 `experiment.py`，看：

- 两句话的句子向量距离变成了多少？
- 系数特别大的时候，输入向量还像原来的词吗？
  （用 `experiment.py` 实验三那张表的方法算一算。）
- 位置信息要"多响"才听得见？词义又会被盖掉多少？

---

## 2. 不用 sin/cos，改成随机向量

位置编码不一定要用正弦余弦，随便给每个位置一个固定的随机向量行不行？
试一下：

```python
def random_positions(length, dim, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((length, dim))
```

用这个表跑 `experiment.py` 实验一。

- 两句话能分开了吗？（能。）
- 那 `experiment.py` 实验二那张"相似度只看距离"的表还成立吗？（不成立了。）
- 想一想：为什么"相似度只看距离"这件事对语言有用？
  （提示：一个词和它前面那个词的关系，和它和前面第二个词的关系，是不一样的；
   而这两句话里"隔一个词"这件事情本身，出现在任何位置都应该差不多。）

---

## 3. 位置向量加在哪儿

现在的做法是"加"。试着换成"拼"：

```python
vector = Tensor(np.concatenate([word_vectors, position_table], axis=-1))
```

这样一来 `DIM` 要改成多少？后面 `w_q`、`w_k`、`w_v` 的形状要跟着变吗？
跑得通吗？

（加和拼是两种真实存在过的做法。"加"的好处是维度不变，代价是位置和词义
混在一起；"拼"保住了两边，代价是维度翻倍。）

---

## 4. 更长的句子

把 `LENGTH` 从 11 改成 100，跑 `experiment.py` 实验三。

- 位置 99 的位置编码长度是多少？值域呢？
- 位置向量里有重复的吗？（提示：看距离 0 那一行的相似度。）
- 训练时只见过 11 个位置，突然来 100 个——位置编码"外推"得好不好？

---

## 5. 打乱的是哪一句

"狗咬人"和"人咬狗"是**整个句子掉了个个儿**。试着只换两个词：

```python
SENTENCE_C = "人 咬 狗".split()      # 已知
SENTENCE_D = "狗 人 咬".split()      # 只把后两个词换了一下
```

两句话的输出向量距离是多少？和"整个掉个儿"比起来呢？

这一题想说的是：位置编码记的是**每个位置上的东西**，
不是"这句话是不是倒过来的"。倒过来只是它顺带能表达的一种情况。
