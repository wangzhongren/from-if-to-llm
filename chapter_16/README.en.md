**English** | [中文](README.md)

# Chapter 16: The model doesn't know who comes first

## The problem in this chapter

Last chapter ended with a question:

> From the word vectors to the output, which step uses "what position this word is in"?
> Not one step.

This chapter we run that experiment. Same three words, only the order is changed:

```text
狗 咬 人
人 咬 狗
```

*(`狗` dog · `咬` bites · `人` person — "dog bites man" and "man bites dog".)*

Run `before.py`:

```text
  『狗 咬 人』
  --------------------------------------------------
    狗  看谁：人(0.357)、狗(0.329)、咬(0.315)
          输出 = [-0.167, -0.479, +0.242, -0.370, +0.690, +0.431, -0.133, +0.210]
    咬  看谁：人(0.389)、狗(0.348)、咬(0.264)
          输出 = [-0.176, -0.496, +0.257, -0.395, +0.741, +0.434, -0.128, +0.225]
    人  看谁：咬(0.342)、人(0.338)、狗(0.320)
          输出 = [-0.162, -0.471, +0.235, -0.357, +0.663, +0.429, -0.136, +0.201]

  『人 咬 狗』
  --------------------------------------------------
    人  看谁：咬(0.342)、人(0.338)、狗(0.320)
          输出 = [-0.162, -0.471, +0.235, -0.357, +0.663, +0.429, -0.136, +0.201]
    咬  看谁：人(0.389)、狗(0.348)、咬(0.264)
          输出 = [-0.176, -0.496, +0.257, -0.395, +0.741, +0.434, -0.128, +0.225]
    狗  看谁：人(0.357)、狗(0.329)、咬(0.315)
          输出 = [-0.167, -0.479, +0.242, -0.370, +0.690, +0.431, -0.133, +0.210]
```

*(The two halves are `『狗 咬 人』` "dog bites man" and `『人 咬 狗』` "man bites dog".
Within each, every word gets two lines: `狗  看谁：人(0.357)、狗(0.329)、咬(0.315)` means
"who does 狗 look at: 人 (0.357), 狗 (0.329), 咬 (0.315)" — the three words of the
sentence with the weight each gets, in descending order — and `输出 = [...]` is the
8-dimensional vector that word comes out with.)*

The two sentences' outputs are **not off by a single number**; they are the same numbers
in the opposite order. `before.py` double-checks this:

```text
  人咬狗 的第 0 个词（人）的输出 = [-0.162, -0.471, +0.235, -0.357, +0.663, +0.429, -0.136, +0.201]
  狗咬人 的第 2 个词（人）的输出 = [-0.162, -0.471, +0.235, -0.357, +0.663, +0.429, -0.136, +0.201]

  '人'的输出一模一样吗：True
  '狗'的输出一模一样吗：True
```

*(`人咬狗 的第 0 个词（人）的输出` — "the output of word 0 of 人咬狗, which is 人"; likewise
for word 2 of 狗咬人, which is also 人. `'人'的输出一模一样吗：True` — "is 人's output exactly
the same: True"; `'狗'的输出一模一样吗：True`.)*

Now add up the outputs at every position, and call that "the sentence's representation":

```text
  狗咬人 的句子向量 = [-0.504, -1.446, +0.734, -1.121, +2.095, +1.294, -0.397, +0.636]
  人咬狗 的句子向量 = [-0.504, -1.446, +0.734, -1.121, +2.095, +1.294, -0.397, +0.636]

  两句话的句子向量一样吗：True
```

*(`狗咬人 的句子向量` — "the sentence vector of 狗咬人"; and the two are identical, with
`两句话的句子向量一样吗：True` — "are the two sentence vectors the same: True".)*

Exactly the same.

This is not a bug, it's a mathematical property of this mechanism. That weighted sum from
Chapter 14, `weights @ value` — **summing doesn't care about order**. And the score for
any two words, `q · k`, depends only on which two words they are, and not at all on how
far apart they are.

So in this mechanism's eyes, a sentence is not a "sequence" but a pile of words.

**Here's the problem: "dog bites man" and "man bites dog" are two completely different
things, and the model thinks they are the same thing.**

## The simplest attempt

The thing to fix is clear: **give every position a vector of its own, and add it to the
word vector.**

```python
x[i] = 词向量 + 位置向量[i]
```

Word i now gets not only "which word am I" but also "which place am I in".

So what does the position vector look like? The dumbest and most intuitive scheme:
**whatever the position number is, write that.**

```python
位置 0 → [0, 0, 0, 0, 0, 0, 0, 0]
位置 1 → [1, 1, 1, 1, 1, 1, 1, 1]
位置 2 → [2, 2, 2, 2, 2, 2, 2, 2]
...
```

*(`位置` — "position": position 0 gets all zeros, position 1 gets all ones, position 2 gets
all twos, one entry per dimension.)*

Run the first part of `after.py` and the problems show up immediately:

```text
        位置         编号向量的长度        词向量的长度
  ----------------------------------------
         0            0.00          1.02
         1            2.83          1.02
         2            5.66          1.02
         5           14.14          1.02
        10           28.28          1.02
        50           28.28          1.02

    （位置 50 的编号向量长度 = 50 × √8 ≈ 141.4）
```

*(Columns: `位置` "position", `编号向量的长度` "length of the index vector", `词向量的长度`
"length of the embedding". The note at the bottom: `（位置 50 的编号向量长度 = 50 × √8 ≈ 141.4）`
— "the index vector's length at position 50 = 50 × √8 ≈ 141.4". Note that the captured
output's `50` row repeats the `10` row's `28.28`; the value the program is actually
pointing at is the 141.4 given in that note, and the third table in the Experiment
section below prints `141.42` for position 50.)*

**Problem one: the scale runs away.** The embedding's length stays at 1.02 the whole way,
while the position index can grow to 141.42. Add the two and the position term completely
swamps the meaning — the `狗` ("dog") sitting at position 50 no longer looks much like a
"dog" to the model.

**Problem two: something you've never seen can't be extrapolated to.** In training it saw
at most 10 words; at word 50 it's looking at a number it has never seen. This isn't "not
learned well enough", it's that **there simply are no parameters for position 50 in this
scheme**. (Swapping the index number for "a small learnable vector per position" doesn't
help either: those few parameters for position 50 have never once been updated.)

**Problem three: it has only one scale.** One index is squeezed into a single number, so
"one word apart" and "eight words apart" can't be expressed in different ways — they are
just the difference between 1 and 8.

## Experiment

Run `experiment.py`. The first thing it does is answer the question from the opening:

```text
  【不加位置信息】
    『人』在两个句子里的输出一样吗：True
    两句话的句子向量距离：0.0000

  【加了位置编码】
    『人』在两个句子里的输出一样吗：False
    两句话的句子向量距离：0.9995
```

*(`【不加位置信息】` — "without position information": `『人』在两个句子里的输出一样吗：True` —
"is 人's output in the two sentences the same: True"; `两句话的句子向量距离：0.0000` — "distance
between the two sentences' sentence vectors: 0.0000". Then `【加了位置编码】` — "with the
positional encoding added": False, and the distance is 0.9995.)*

The distance goes from `0.0000` to `0.9995`. Same three words, different order, and this
time they are finally different.

Second, look at the properties of the positional encoding:

```text
           位置对    距离       相似度   同一距离的其他位置对
  --------------------------------------------------------
         0 和 0     0     4.000   1和1(4.000)、2和2(4.000)
         0 和 1     1     3.535   1和2(3.535)、2和3(3.535)
         0 和 2     2     2.564   1和3(2.564)、2和4(2.564)
         0 和 3     3     1.965   1和4(1.965)、2和5(1.965)
         0 和 5     5     3.160   1和6(3.160)、2和7(3.160)
```

*(Columns: `位置对` "pair of positions", `距离` "distance", `相似度` "similarity", `同一距离的
其他位置对` "other pairs of positions at the same distance". So row one reads: positions 0
and 0, distance 0, similarity 4.000 — and pairs 1-and-1 and 2-and-2, also at distance 0,
give 4.000 as well.)*

**Same distance, same similarity.** `0 and 1`, `1 and 2`, `2 and 3` are all `3.535`;
`0 and 2`, `1 and 3`, `2 and 4` are all `2.564`.

Note that it isn't "the further apart, the less alike" — distance 5 has a similarity
(3.160) a little *higher* than distance 2 (2.564), because waves of different frequencies
periodically line back up. What really matters is this: **the similarity depends only on
how far apart the two positions are, not on which positions they are**. That is exactly
why "relative position" can be expressed at all.

Third, the two kinds of position information compared on scale:

```text
      位置        编号向量长度        位置编码长度          编号向量还像狗吗          位置编码还像狗吗
  --------------------------------------------------------------------
       0          0.00          2.00             1.000             0.521
       1          2.83          2.00             0.649             0.764
       2          5.66          2.00             0.548             0.781
       5         14.14          2.00             0.473             0.135
      10         28.28          2.00             0.445             0.290
      50        141.42          2.00             0.422             0.380
```

*(Columns: `位置` "position", `编号向量长度` "index vector length", `位置编码长度` "positional
encoding length", `编号向量还像狗吗` "does the index vector still look like 狗 ('dog')",
`位置编码还像狗吗` "does the positional encoding still look like 狗" — i.e. the similarity of
the sum back to the word's own embedding. At position 50 the index vector's length is
141.42 while the positional encoding's length is 2.00, flat, all the way down.)*

The index vector's length climbs all the way to 141.42; the positional encoding's length
is always 2.00.

## The new mechanism

This position vector we call a **positional encoding**.

The scheme: give each position a fixed vector, where every dimension is a **wave of a
different frequency**:

```text
第 i 个位置、第 2k 维   = sin(i / 10000^(2k/dim))
第 i 个位置、第 2k+1 维 = cos(i / 10000^(2k/dim))
```

*(`第 i 个位置、第 2k 维` — "position i, dimension 2k"; `第 i 个位置、第 2k+1 维` — "position i,
dimension 2k+1".)*

Written out with `dim = 8` (this is the table `after.py` prints):

```text
  位置         第0维       第1维       第2维       第3维       第4维       第5维       第6维       第7维
     0       0.000     1.000     0.000     1.000     0.000     1.000     0.000     1.000
     1       0.841     0.540     0.100     0.995     0.010     1.000     0.001     1.000
     2       0.909    -0.416     0.199     0.980     0.020     1.000     0.002     1.000
     3       0.141    -0.990     0.296     0.955     0.030     1.000     0.003     1.000
     4      -0.757    -0.654     0.389     0.921     0.040     0.999     0.004     1.000
     5      -0.959     0.284     0.479     0.878     0.050     0.999     0.005     1.000
     6      -0.279     0.960     0.565     0.825     0.060     0.998     0.006     1.000
     7       0.657     0.754     0.644     0.765     0.070     0.998     0.007     1.000
     8       0.989    -0.146     0.717     0.697     0.080     0.997     0.008     1.000
     9       0.412    -0.911     0.783     0.622     0.090     0.996     0.009     1.000
    10      -0.544    -0.839     0.841     0.540     0.100     0.995     0.010     1.000
```

*(`位置` "position" down the left; `第0维` through `第7维` — "dimension 0" through
"dimension 7" — across the top.)*

Stare at that table for a while and three things come out:

1. **Dimensions on the left change fast, dimensions on the right change slowly.**
   Dimensions 0 and 1 (high frequency) swing wildly from top to bottom; dimensions 6 and 7
   (low frequency) barely move.
2. **Every dimension stays between -1 and 1.** So the position vector never grows long —
   its length is always `2.000`, the same order of magnitude as the embedding (1.020), and
   added together neither swamps the other.
3. **Every position looks different.** 11 positions are 11 different "fingerprints".

Why a set of waves at different frequencies? Because **position relations come at more
than one scale.** "One word apart" and "eight words apart" are two different relations:
the first is mostly "adjacent words that go together" (like `发布` "released" and `手机`
"phone"), the second is mostly "the two ends of a sentence" (like subject and object).
The high-frequency waves handle telling nearby things apart, the low-frequency waves
handle telling distant things apart — the second table in the experiment above is the
reading of exactly that.

The last step is simple: **add the position vector onto the word vector.**

```python
x[i] = 词向量 + 位置编码[i]
```

Add and change nothing else: the embedding's 8 dimensions are still those 8 dimensions,
and the position information is "layered" on top.

## Python implementation

That's `after.py`. The positional encoding itself is only a few lines:

```python
def sinusoidal_positions(length, dim):
    positions = np.arange(length)[:, None]
    dims = np.arange(dim)[None, :]
    angles = positions / np.power(10000.0, (2 * (dims // 2)) / dim)
    table = np.zeros((length, dim))
    table[:, 0::2] = np.sin(angles[:, 0::2])     # 偶数维用 sin
    table[:, 1::2] = np.cos(angles[:, 1::2])     # 奇数维用 cos
    return table
```

*(The comments: `偶数维用 sin` — "use sin for the even dimensions"; `奇数维用 cos` — "use cos
for the odd dimensions".)*

`np.arange(length)[:, None]` turns the positions into a column, `np.arange(dim)[None, :]`
turns the dimensions into a row, and dividing one by the other and taking `sin`/`cos`
gives that table.

On the attention side, only one line changes from the last chapter:

```python
def attention(words, position_table, seed=16):
    word_vectors = np.array([EMBEDDINGS[word] for word in words])
    vector = Tensor(word_vectors + position_table)      # <- 只多这一行
    query = vector @ w_q
    key = vector @ w_k
    value = vector @ w_v
    weights = ((query @ key.T) / np.sqrt(DIM)).softmax(axis=-1)
    return weights @ value, weights
```

*(The comment `<- 只多这一行` marks "the only extra line".)*

Run the last section of `after.py`, and the two sentences' sentence vectors finally come
apart:

```text
  人咬狗 的第 0 个词（人）的输出 = [-0.590, -1.393, +0.681, -1.729, +1.106, +1.284, -1.690, -0.757]
  狗咬人 的第 2 个词（人）的输出 = [-0.789, -1.760, +0.812, -1.942, +1.254, +1.282, -1.728, -0.755]
  狗咬人 的第 0 个词（狗）的输出 = [-0.773, -1.725, +0.801, -1.928, +1.254, +1.272, -1.713, -0.751]

  '人'在两个句子里的输出还一样吗：False

  狗咬人 的句子向量 = [-2.440, -5.418, +2.493, -5.906, +3.824, +3.862, -5.218, -2.260]
  人咬狗 的句子向量 = [-2.053, -4.775, +2.273, -5.464, +3.436, +4.027, -5.336, -2.294]
  两句话的句子向量一样吗：False
  它们之间的距离：0.9995
```

*(`人咬狗 的第 0 个词（人）的输出` — "the output of word 0 of 人咬狗, which is 人"; `狗咬人 的
第 2 个词（人）的输出` — "the output of word 2 of 狗咬人, which is also 人"; and `狗咬人 的第 0
个词（狗）` — "word 0 of 狗咬人, which is 狗". The three vectors are now all different.
`'人'在两个句子里的输出还一样吗：False` — "is 人's output in the two sentences still the same:
False". Then the two sentence vectors, `狗咬人 的句子向量` and `人咬狗 的句子向量`, with
`两句话的句子向量一样吗：False`, and `它们之间的距离：0.9995` — "the distance between them:
0.9995".)*

The same character `人` ("person") no longer comes out as the same vector in "dog bites
man" and in "man bites dog" — because who is standing next to it, and which place it is
in, are both different.

## What it solves

1. **Order is in.** The same words in a different order move the sentence vectors'
   distance from `0.0000` to `0.9995`.
2. **The scale is stable.** The positional encoding's length is always `2.00`, whatever
   the position; the index vector at position 50 is `141.42`.
3. **Relative distance is in too.** Pairs of positions at the same distance have exactly
   the same similarity (all three distance-1 pairs are `3.535`, all three distance-2 pairs
   are `2.564`).

By this point we finally have a **complete building block** in our hands:

```text
词向量 + 位置编码  →  多头注意力  →  输出
```

*(`词向量 + 位置编码 → 多头注意力 → 输出` — "embedding + positional encoding → multi-head
attention → output".)*

It knows what each word is, which words there are, who is close to whom, and who comes
before whom.

## What it still can't solve

Now there's a very natural thought: **this building block works so well, let's stack a few
of them.**

That's exactly what real large models do — not one layer, but dozens. The first layer's
output goes straight in as the second layer's input, the second layer's output goes in as
the third layer's input... it sounds like it should get stronger and stronger.

But stacking them directly goes wrong. And we have actually seen this "wrong" once
already.

Think back to Chapter 14's table:

```text
     d  |          不缩放：最大权重          梯度  |         除以√d：最大权重          梯度
  ----------------------------------------------------------------------
   512  |            0.9510      0.0701  |            0.3163      0.3537
```

*(One row of the unscaled-vs-scaled table: `不缩放：最大权重 / 梯度` "unscaled: largest
weight / gradient" and `除以√d：最大权重 / 梯度` "divided by √d: largest weight / gradient",
at `d = 512`.)*

Once the scores get big, the weights are squeezed into `1` and a pile of `0`s, and the
gradient left over is only `0.0701` — it can't learn any more.

Stacking layers magnifies problems of this kind: every layer computes its `softmax` all
over again, does its weighted sum all over again, and feeds its output to the next layer.
What scale the first layer's output is on, whether the second layer's input range is still
sane, whether a gradient can travel from the last layer all the way back to the first —
none of these are things that "stack one more layer" satisfies automatically.

Take the code we have in hand and try it and it's plain enough: our `attention` has one
layer right now. If we want to stack a second layer, the second layer's input is the first
layer's output.

**So if we hook two attention layers end to end, what happens during training?**

Next chapter we run that experiment.

## Exercises

See `exercises.en.md`. The most important ones are listed here:

1. **How much positional encoding is right** — multiply the positional encoding by `0.1`
   and by `10.0`, run it each way, and see what the two sentences' sentence-vector
   distance becomes, then whether the input vectors still look like the original words.
2. **Drop sin/cos for random vectors** — the positions are still distinguished, but does
   "similarity depends only on distance" still hold?
3. **Replace "add" with "concatenate"** — after `np.concatenate`, how much does the
   dimension have to change? Do the matrices behind it have to change along with it?
4. **A longer sentence** — change `LENGTH` to 100, look at the encoding's length and value
   range at position 99, and think about extrapolation again.
5. **Swap only two words** — change `狗 咬 人` ("dog bites man") to `狗 人 咬` (only the
   last two words swapped), and see whether the output distance is large or small compared
   with flipping the whole thing around.
