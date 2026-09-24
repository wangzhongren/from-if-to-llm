**English** | [中文](README.md)

# Chapter 14: What I'm looking for, what you've got

## The problem in this chapter

Last chapter left us with this one line:

> It only knows "are they alike". It doesn't know "what I'm looking for".

Let's put that problem on the table where you can see it. Run `before.py`:

```text
实验一：分数矩阵是对称的吗？
------------------------------------------------------------
  整个矩阵 S 和它的转置相等：True
  苹果 -> 手机     1.600
  手机 -> 苹果     1.600

  两个方向是同一个数。'苹果有多想问手机'和'手机有多想问苹果'，
  在第 13 章里根本无法区分——因为只有一把尺子。
```

*(`实验一：分数矩阵是对称的吗？` — "Experiment one: is the score matrix symmetric?" `整个矩阵 S 和它的转置相等：True` — "the whole matrix S equals its transpose: True". Then `苹果 -> 手机` (`apple -> phone`) `1.600` and `手机 -> 苹果 1.600`. The closing lines: "the two directions are the same number. `How much does 苹果 want to ask 手机` and `how much does 手机 want to ask 苹果` cannot be told apart in Chapter 13 — because there is only one ruler.")*

"What I want to ask you" and "what you want to ask me" are **the same number** in the
last chapter. That is not a coincidence — the score *is* `v_i · v_j`, and a dot product
is symmetric.

The second defect is worse. Suppose `苹果` ("apple") wants to change what it looks at:

```text
  现在'苹果'问谁，是被它自己的向量锁死的：
  苹果 = [0.  0.  0.  0.  0.7 1.  0.  0. ]
  于是它最想问的是'手机'（1.600）和'发布'（1.210）。

  我们试着改一改它：把'科技与商业'这一维从 1.0 改成 0.0。
  苹果' = [0.  0.  0.  0.  0.7 0.  0.  0. ]

  分数                   改之前       改之后
  ----------------------------------------
  苹果 -> 手机         1.600     0.700
  苹果 -> 发布         1.210     0.210
  苹果 -> 苹果         1.490     0.490
  手机 -> 苹果         1.600     0.700

  最后一行是重点：手机一个字都没改，可是'手机怎么看苹果'变了。
```

*(`现在'苹果'问谁，是被它自己的向量锁死的` — "who `苹果` asks right now is locked in by its own vector". The eight numbers are its embedding, and this chapter's embeddings are 8-dimensional with these dimensions in this order: 人/动物 person-or-animal, 时间 time, 地点 place, 动作 action, 物品 object, 科技与商业 tech-&-business, 虚词 function word, 程度 degree. So `苹果`'s vector is 0.7 of "object" and 1.0 of "tech-&-business", which is why it most wants to ask 手机 (`phone`, score `1.600`) and 发布 (`released`, `1.210`). We then change one number: the "tech-&-business" dimension goes from 1.0 to 0.0, giving the second vector. The table's columns are 分数 "score", 改之前 "before the change", 改之后 "after the change", and the four rows are 苹果→手机, 苹果→发布, 苹果→苹果, 手机→苹果. The last line is the point: `最后一行是重点：手机一个字都没改，可是'手机怎么看苹果'变了` — "the last row is the important one: not one character of 手机 was changed, and yet how 手机 sees 苹果 changed.")*

All we wanted was for `苹果` to look at `手机` a little less — and "how `手机` sees
`苹果`" changed along with it. Because `苹果` has only one vector, and it has to do two
jobs at once:

- **When I go and ask somebody**, this is what I use
- **When somebody comes and asks me**, this is what they use too

One vector, two roles. That is why "what I'm looking for" can't be said on its own.

**Here's the problem: a word ought to have three identities — "what I'm looking for",
"what I have", and "what I can give you".** Right now it has one.

## The simplest attempt

First let's get the three identities straight. Take `苹果` as the example:

| role | what it says | who uses it |
|---|---|---|
| what I'm looking for | what I want to ask right now is "who else is a tech company like me" | only `苹果` itself has this |
| what I have | `苹果` has the labels "tech", "object" stuck on it | others look at this when they come to ask |
| what I can give you | the vector `苹果` actually hands over when somebody looks at it | used when it is being looked at |

Chapter 13 stuffed the first two roles into a single vector, which is exactly why we got
that "move one thing and everything moves".

The most intuitive repair: **give each role a matrix of its own, and project three times,
once per role.**

We've done projection before — back in Chapter 3, `x @ W`, turning one set of numbers
into another set of numbers. Back then `W` was weights we wrote by hand; now we make it
a parameter that has to be learned:

```python
query = x @ W_q     # 我在找什么
key   = x @ W_k     # 我有什么
value = x @ W_v     # 我能给你什么
```

*(The three comments are the three roles: `我在找什么` "what I'm looking for", `我有什么`
"what I have", `我能给你什么` "what I can give you".)*

Why does that pull them apart? Because `W_q` and `W_k` are two **independent matrices**.
What `苹果`'s query looks like and what its key looks like are no longer tied together.
The defect from the previous section — "not one character of `手机` changed, yet the
score changed" — is gone.

How do we compute the score? Chapter 13 used `v_i · v_j`; now that the two roles are
different, we compute it as **how well "my question" matches "your label"**:

```python
score[i][j] = query[i] · key[j]
```

Finally, the **thing being mixed** in the weighted sum has to change too. Chapter 13
mixed the word vectors themselves; now we mix the value:

```python
output[i] = sum_j weights[i][j] * value[j]
```

The weights decide "who to look at"; the value decides "what you see when you look".
These two always should have been separate: a word can be very much worth looking at,
and what it can give you needn't be its own vector.

## Experiment

Don't rush to write code. That `score[i][j] = query[i] · key[j]` above hides a trap, and
running `experiment.py` will show you.

Q and K are vectors of 8, 16, dozens of dimensions. Two vectors multiplied dimension by
dimension and summed — the higher the dimension, the more there is to add up, and the
bigger the score naturally gets. We measured it on the real data (the `苹果` row,
averaging over 20 random initializations at each setting):

```text
        |        分数标准差        |        最大权重         |        有效词数
     d  |      不缩放      除以√d  |      不缩放      除以√d  |      不缩放      除以√d
  ----------------------------------------------------------------------
     8  |     2.57      0.91  |    0.540     0.271  |     3.88      8.05
    16  |     4.70      1.18  |    0.673     0.292  |     2.65      7.29
    32  |     5.54      0.98  |    0.780     0.317  |     1.92      7.35
    64  |     7.69      0.96  |    0.751     0.272  |     1.93      7.63
   128  |    12.20      1.08  |    0.883     0.300  |     1.47      7.49
   256  |    16.86      1.05  |    0.940     0.305  |     1.25      7.29
   512  |    24.62      1.09  |    0.951     0.316  |     1.20      6.98

  不缩放：分数标准差从 2.57 一路涨到 24.62，有效词数从 3.88 掉到 1.20。
  除以 √d：无论维度多高，分数标准差都在 0.91~1.18 之间，有效词数都在 6.98~8.05 之间。
```

*(The column groups are 分数标准差 "standard deviation of the scores", 最大权重 "the
largest weight", 有效词数 "effective word count"; under each group, 不缩放 "unscaled" and
除以√d "divided by √d". The rows are the dimension `d`. The closing lines: unscaled, the
score standard deviation climbs from 2.57 all the way to 24.62 and the effective word
count falls from 3.88 to 1.20; divided by √d, however high the dimension, the score
standard deviation stays in 0.91–1.18 and the effective word count stays in 6.98–8.05.)*

(The "effective word count" is "how many words the attention is spread over on
average": if all 11 words matter equally it's 11; if it stares at exactly one word it's 1.)

Without scaling, the scores become absurdly large as the dimension grows. And when the
scores get large, what happens to `softmax`?

```text
d = 512 时'苹果'那一行，长什么样

  不缩放：
    发布  1.00000000  <- 只有它还站着
    看到  0.00000000
    新    0.00000000
    ...
    最大权重 = 1.00000000   有效词数 = 1.00   梯度 = 0.0000000001

  除以 √d：
    发布  0.63569118
    看到  0.22377703
    新    0.05033896
    刚刚  0.02469449
    手机  0.02412959
    苹果  0.02062780
    昨天  0.01675324
    商场  0.00171199
    的    0.00106865
    在    0.00073927
    我    0.00046779
    最大权重 = 0.63569118   有效词数 = 3.10   梯度 = 0.3639007455
```

*("What the `苹果` row looks like at d = 512". Top half, 不缩放 "unscaled": `发布` gets
`1.00000000` with the note `<- 只有它还站着` "only it is still standing", and every other
word is `0.00000000`. The summary line reads 最大权重 "largest weight" = 1.00000000,
有效词数 "effective word count" = 1.00, 梯度 "gradient" = 0.0000000001. Bottom half,
除以 √d "divided by √d": the weights are spread across all eleven words, from 发布 0.63569118
down to 我 0.00046779, with largest weight 0.63569118, effective word count 3.10, gradient
0.3639007455. The words, top to bottom, are 发布 released, 看到 saw, 新 new, 刚刚 just now,
手机 phone, 苹果 apple, 昨天 yesterday, 商场 mall, 的 the particle, 在 at, 我 I.)*

In the unscaled row, the weights have been squeezed into "a 1 and ten 0s". This is not
"the attention is very focused" — this is **degenerate**: softmax has turned into an
operation that just picks the biggest one and zeroes out all the rest.

What's worse is the last thing in that block, the "gradient":

```text
     d  |          不缩放：最大权重          梯度  |         除以√d：最大权重          梯度
  ----------------------------------------------------------------------
     8  |            0.5404      0.3489  |            0.2706      0.3338
    32  |            0.7801      0.2102  |            0.3168      0.3399
   128  |            0.8833      0.1341  |            0.2996      0.3391
   512  |            0.9510      0.0701  |            0.3163      0.3537
```

*(`不缩放：最大权重 / 梯度` — "unscaled: largest weight / gradient"; `除以√d：最大权重 / 梯度` — "divided by √d: largest weight / gradient".)*

The "gradient" column is "if the score changes a little, how much does the weight change"
— the number backpropagation has to use. Once the weights are squeezed into 1s and 0s,
this number becomes 0.0000000001, which is as good as nothing. **The weights stop
responding to the scores, and the words that got squeezed to 0 can never get back up.**

The fix for this trap is simple enough to be a little funny: **divide the scores by √d.**

If d is 512, divide by 22.6; if d is 8, divide by 2.83. Look at the two tables above:
after dividing by √d, no matter how high the dimension, the score standard deviation
holds around 1, the effective word count holds around 7, and the gradient is always there.

Why √d specifically, and not d or 1? Because with random initialization, `query · key` is
the sum of d random numbers: **the more things you add up, the easier the sum drifts, and
the standard deviation is proportional to √d**. Divide it back out and the scale returns
to 1. This is a divisor that *pins* the scale down, not a guess.

## The new mechanism

Put everything above together and you have this chapter's new mechanism:

```python
query = x @ W_q        # 我在找什么
key   = x @ W_k        # 我有什么
value = x @ W_v        # 我能给你什么

scores  = query @ key.T / sqrt(d)     # 我的问题和你身上的标签有多合
weights = softmax(scores, axis=-1)    # 归一化成权重
output  = weights @ value             # 按权重把 value 混起来
```

*(The comments again: `我在找什么` what I'm looking for, `我有什么` what I have, `我能给你什么`
what I can give you. Then `scores = query @ key.T / sqrt(d)` — 我的问题和你身上的标签有多合
"how well my question matches the label on you"; `weights = softmax(scores, axis=-1)` —
归一化成权重 "normalize into weights"; `output = weights @ value` — 按权重把 value 混起来
"mix the value by the weights".)*

Written in one line:

```text
Attention(Q, K, V) = softmax(Q Kᵀ / √d) V
```

The names of the three roles, from today on, are these:

- **Query**: what I'm looking for
- **Key**: what I have
- **Value**: the information I can actually hand over

Note that Q, K and V are **the same batch of word vectors projected three times**, which
is why this scheme is called **self-attention** — a sentence looking at itself.

One more detail: the rows of the weight matrix are still "who is looking", but the
columns have become "whose labels are being looked at", and what gets mixed in the end is
the value. So the relationship between `weights` and `output` is no longer "knead the
original vectors around" but "**pick up goods as ordered**".

## Python implementation

That's `after.py`. The implementation maps one-to-one onto the formulas:

```python
def project(vectors, weight):
    """把每个词向量投影到另一个空间：x @ W。"""
    return vectors @ weight


def split_into_qkv(vectors, w_q, w_k, w_v):
    """同样一批词向量，投三次影，得到三个角色。"""
    query = project(vectors, w_q)
    key = project(vectors, w_k)
    value = project(vectors, w_v)
    return query, key, value


def attention_scores(query, key):
    """分数 = Q Kᵀ / √d_key。"""
    return (query @ key.T) / np.sqrt(key.shape[-1])


def attention(query, key, value):
    """单头 self-attention 的完整流程。"""
    weights = attention_weights(attention_scores(query, key))
    return weights @ value, weights
```

*(The docstrings: 把每个词向量投影到另一个空间 "project each word vector into another space";
同样一批词向量，投三次影，得到三个角色 "the same batch of word vectors, projected three
times, giving the three roles"; 分数 = Q Kᵀ / √d_key; 单头 self-attention 的完整流程
"the complete flow of single-head self-attention".)*

Run `after.py`:

**First thing, look at the weight matrix.**

```text
注意力权重矩阵（投影还没训练，先看机制）
------------------------------------------------------------
             我   昨天     在   商场   看到   苹果   刚刚   发布     的     新   手机
  我      0.026  0.009  0.008  0.045  0.212  0.093  0.013  0.373  0.005  0.091  0.126
  昨天    0.036  0.187  0.012  0.088  0.077  0.146  0.083  0.078  0.009  0.121  0.161
  在      0.059  0.013  0.087  0.100  0.043  0.200  0.009  0.164  0.073  0.021  0.231
  商场    0.025  0.014  0.077  0.111  0.034  0.214  0.011  0.151  0.059  0.028  0.276
  看到    0.056  0.060  0.014  0.013  0.259  0.061  0.084  0.209  0.025  0.134  0.083
  苹果    0.018  0.025  0.262  0.248  0.168  0.022  0.021  0.033  0.156  0.014  0.033
  刚刚    0.041  0.251  0.011  0.069  0.079  0.066  0.179  0.044  0.008  0.166  0.086
  发布    0.026  0.046  0.056  0.026  0.586  0.012  0.053  0.060  0.085  0.030  0.020
  的      0.091  0.021  0.135  0.138  0.069  0.133  0.015  0.122  0.107  0.023  0.145
  新      0.034  0.176  0.062  0.090  0.127  0.011  0.314  0.015  0.039  0.113  0.018
  手机    0.010  0.015  0.290  0.390  0.079  0.021  0.011  0.020  0.123  0.009  0.032
  （行 = 谁在问，列 = 问到了谁。每一行的和都是 1。）
```

*(Title: "Attention weight matrix (the projections haven't been trained yet; for now look
at the mechanism)". **Read it the same way as Chapter 13's matrix:** the column headers,
left to right, are `我` I · `昨天` yesterday · `在` at · `商场` mall · `看到` saw · `苹果` apple ·
`刚刚` just now · `发布` released · `的` particle · `新` new · `手机` phone. The leftmost label
on each row is the same list, and row `苹果` is `苹果` doing the looking. The note at the
bottom says 行 = 谁在问，列 = 问到了谁 "row = who is asking, column = who got asked", and
每一行的和都是 1 "every row sums to 1".)*

**These numbers don't mean anything yet**, because we haven't trained `W_q`, `W_k`, `W_v`
— they're randomly initialized. In a real model these three matrices are learned (next
chapter we'll actually train it). What to look at in that table is the mechanism itself:
each row sums to 1, each row is different, and the rows are no longer symmetric with each
other.

**Second thing, the scores are no longer symmetric.**

```text
                      苹果 -> 手机      手机 -> 苹果   对称吗
  --------------------------------------------------------
  第 13 章               1.600         1.600   是
  这一章                -0.798        -0.897   否
```

*(Columns: 第 13 章 "Chapter 13", 这一章 "this chapter", 对称吗 "symmetric?". Chapter 13's
`苹果 -> 手机` and `手机 -> 苹果` are both `1.600`; this chapter's are `-0.798` and `-0.897`.
So the answer moved from 是 "yes" to 否 "no".)*

**Third thing, the same word can now ask different questions.** In this section we set
`W_q` by hand to "ask about one dimension only", purely to see clearly how "what I'm
looking for" gets expressed:

```text
  问'科技与商业'：苹果最关注 发布(0.261)、苹果(0.250)、商场(0.176)
      我    0.0235  #
      昨天  0.0211  #
      在    0.0099  #
      商场  0.1758  #########
      看到  0.0616  ###
      苹果  0.2496  #############
      刚刚  0.0056
      发布  0.2611  ##############
      的    0.0038
      新    0.0220  #
      手机  0.1660  #########

  问'物品      '：苹果最关注 手机(0.348)、苹果(0.267)、昨天(0.083)
      我    0.0376  ##
      昨天  0.0832  ####
      在    0.0275  #
      商场  0.0231  #
      看到  0.0096  #
      苹果  0.2669  ##############
      刚刚  0.0455  ##
      发布  0.0568  ###
      的    0.0536  ###
      新    0.0482  ###
      手机  0.3480  ###################
```

*(`问'科技与商业'` — "asking about *tech & business*": 苹果 most attends to 发布 (0.261),
苹果 (0.250), 商场 (0.176), and the lines under it are the eleven words with their weights
and a bar. `问'物品'` — "asking about *object*": 苹果 most attends to 手机 (0.348),
苹果 (0.267), 昨天 (0.083). In both halves the words run down the list `我` I, `昨天`
yesterday, `在` at, `商场` mall, `看到` saw, `苹果` apple, `刚刚` just now, `发布` released,
`的` particle, `新` new, `手机` phone.)*

Same `苹果`, two completely different sets of weights. The last chapter couldn't do this —
there the weights were locked in by `苹果`'s own vector, one set per word.

## What it solves

Three things, each backed by the numbers above:

1. **"I ask you" and "you ask me" are separated.** In Chapter 13 both directions shared
   the one number `1.600`; here they are `-0.798` and `-0.897`, two independent numbers.
   Changing your own query no longer changes the key others see you through.
2. **The same word can ask different questions.** Ask about "tech & business" and `苹果`
   attends to `发布` (0.261); ask about "object" and it attends to `手机` (0.348). This is
   the first time "what I'm looking for" can be expressed on its own.
3. **The scale is stable.** After dividing by √d, as the dimension grows from 8 to 512,
   the score standard deviation stays near 1 (0.91–1.18) and the effective word count
   stays near 7 (6.98–8.05). Without scaling, at d=512 the weights are squeezed into
   `1.00000000` and ten `0.00000000`s, and the gradient left over is `0.0000000001`.

## What it still can't solve

Look at the `苹果` row that `after.py` prints:

```text
  在    0.2621  ############################
  商场  0.2483  ###########################
  看到  0.1683  ##################
  的    0.1557  #################
```

*(`在` at 0.2621, `商场` mall 0.2483, `看到` saw 0.1683, `的` particle 0.1557 — these are
the four heaviest entries of the row, and the bars are drawn from them.)*

(These numbers are random right now, because the projections haven't been trained. But
the structure they expose is real.)

**Every word gets exactly one row of weights.** One row is one set of numbers that sums
to 1. That means attention is a **fixed budget**: give this word a bit more and you have
to take it from somebody else.

Now let's look at this through a different sentence:

```text
小王 把 书 给了 小李 因为 他 明天 考试
```

*(Nine words: `小王` Xiao Wang (a person's name) · `把` the *bǎ* particle, which marks the
thing being acted on · `书` book · `给了` gave (to) · `小李` Xiao Li (another person) ·
`因为` because · `他` he / him · `明天` tomorrow · `考试` exam / to sit an exam.
"Xiao Wang gave the book to Xiao Li because he has an exam tomorrow." Keep this list
too — the whole of the next chapter is built on this sentence, and on which word the
pronoun `他` is looking at.)*

On the single character `他` ("he") there hang several things at once:

- **Reference**: is 他 小王 or 小李? — that needs `小王` and `小李`
- **Time**: what is he doing tomorrow? — that needs `明天` and `考试`
- **Cause**: why was this sentence said at all? — that needs `因为`

`他` has only one row of weights, summing to 1. If it gives 0.5 to `小王` (reference),
then only 0.5 is left for `明天` and `考试` (time) — **the two things are competing for
the same budget**.

This is not something "not trained well enough" can fix. Train it to perfection and one
row of weights still expresses only one preference: either it leans toward reference, or
it leans toward time, or both sides give ground and neither is focused.

**So if a sentence has several kinds of relation that all need looking at, and a word has
only one row of weights — what then?**

## Exercises

See `exercises.en.md`. The most important ones are listed here:

1. **What if the three matrices become one** — set `W_q`, `W_k` and `W_v` all to the
   identity matrix, run it, and see whether it degenerates back into Chapter 13.
2. **Take the √d away** — make `attention_scores` return `query @ key.T` directly, move
   `d` from 8 up to 64, and see what the weight matrix turns into.
3. **Scale but don't switch V** — change `weights @ value` back to `weights @ vectors`,
   and see how the output differs.
4. **Invent a question of your own** — set `W_q` by hand to "ask about time only"
   (dimension 1), and see who `苹果` goes and asks. Can it get anything out of them? Why?
5. **Give the two roles different lengths** — move `QUESTION_STRENGTH` from 4.0 down to
   0.1, then up to 20, and see whether the weights flatten out or sharpen.
