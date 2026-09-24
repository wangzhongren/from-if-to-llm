**English** | [中文](README.md)

# Chapter 13: Not every word matters equally

## The problem in this chapter

Last chapter we represented a word by its **context**:

```python
context = 其它所有词的平均      # 自己不算
```

*(The comment `自己不算` means "not counting itself".)*

Inside that sentence, `苹果` ("apple" — in this sentence it's the company, not the
fruit) can only be pinned down as a company rather than a fruit through its context.
The direction is right. But "average" is the move we are going to tear apart to its
face today.

We're still using the same sentence:

```text
我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机
```

*("I saw the new phone Apple just released yesterday at the mall." Word by word:
`我` I · `昨天` yesterday · `在` at · `商场` the mall · `看到` saw · `苹果` Apple ·
`刚刚` just now · `发布` released · `的` the attributive particle ("'s" / "that") ·
`新` new · `手机` phone. **This word list is the key to everything below.** In every
table in this chapter, the rows and columns are exactly these eleven Chinese words,
in exactly this order; the row says who is doing the looking and the column says who
is being looked at. Keep the list next to you and read the Chinese directly.)*

Run `before.py` first, and look at the "weights" Chapter 12's method produces on this
sentence:

```text
句子： 我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机

其实根本不用算，谁出了多少力是写死的：
------------------------------------------------------------
  我      0.10  ##############
  昨天    0.10  ##############
  在      0.10  ##############
  商场    0.10  ##############
  看到    0.10  ##############
  苹果    (自己)  （自己不在自己的上下文里）
  刚刚    0.10  ##############
  发布    0.10  ##############
  的      0.10  ##############
  新      0.10  ##############
  手机    0.10  ##############
  除自己以外的 10 个词，每个都是 1/10 = 0.10。
  '发布'是这样，'昨天'也是这样，'的'还是这样。
```

*(Line by line: `句子：` "the sentence is"; `其实根本不用算，谁出了多少力是写死的` "there's
nothing to compute — how much each word contributes is hard-coded"; the header is
"word / weight / bar", and the closing lines say "the 10 words other than itself get
1/10 = 0.10 each — `发布` is like this, `昨天` is like this, `的` is still like this".
`苹果`'s own row is marked `(自己)` — "itself" — with the note `（自己不在自己的上下文里）`,
"it isn't in its own context".)*

Ten words, one share each, nobody more and nobody less.

This isn't "roughly the same" — it's **exactly** the same. To see it more clearly,
`before.py` also runs a perturbation experiment: add a tiny perturbation to the i-th
word's vector (the same vector, added at every dimension), and look at how far `苹果`'s
context vector moves. That movement is how much influence this word has on `苹果`:

```text
  改的是哪个词              上下文向量变化了多少
  ----------------------------------------
  我                          0.028284
  昨天                        0.028284
  在                          0.028284
  商场                        0.028284
  看到                        0.028284
  刚刚                        0.028284
  发布                        0.028284
  的                          0.028284
  新                          0.028284
  手机                        0.028284

  最大变化 / 最小变化 = 1.000
  10 个位置，变化量一模一样 —— 平均法根本不知道谁重要。
```

*(The columns are 改的是哪个词 "which word was changed" and 上下文向量变化了多少 "how much
the context vector moved". The closing lines: 最大变化 / 最小变化 "largest change /
smallest change" `= 1.000`, and "10 positions, the movement is exactly the same —
averaging has no idea who matters".)*

Change `发布` ("released") and change `昨天` ("yesterday"), and the effect on `苹果`
("apple") is identical. And yet anyone can see at a glance that `发布` and `手机`
("phone") have everything to do with `苹果`, while `昨天` and `商场` ("the mall") have
nothing much to do with it.

**There's the problem, laid out: averaging has no concept of "important".** It doesn't
even have a *place* to express importance — the weights are hard-coded `1/10`, not
computed.

## The simplest attempt

The thing to fix is clear: **the weights can't be hard-coded; the model has to compute
them.**

So what should the weights be computed *from*? The most intuitive answer is
**similarity**:

> `苹果` and `手机` are alike, so what `手机` says should count for more;
> `苹果` and `昨天` are not alike, so what `昨天` says shouldn't matter much.

Anyone would think of this, and it has one enormous advantage: **it needs nothing new.**
We already have word vectors from Chapter 10, and whether two vectors are alike is
something we can just compute.

So the most plain-vanilla approach is three steps:

1. **Compute scores**: the center word and every word get a similarity score
2. **Normalize into weights**: turn the scores into a set of weights that sum to 1
3. **Weighted sum**: mix all the words' vectors by weight into a new vector for the
   center word

Step 3 is the same as Chapter 12, except that "average" becomes "average by weight".
The genuinely new parts are the first two steps.

## Experiment

Run `experiment.py`.

**Comparison one: the weight table.** The difference between averaging and the new
method is visible at first glance:

```text
============================================================
对比一：'苹果'给每个词的权重
============================================================

  词          平均法       相似度加权   谁更被重视
  --------------------------------------------------------
  我       0.1000     0.0446   ####
  昨天     0.1000     0.0446   ####
  在       0.1000     0.0446   ####
  商场     0.1000     0.0672   #####
  看到     0.1000     0.0446   ####
  苹果     0.0000     0.1978   ################
  刚刚     0.1000     0.0446   ####
  发布     0.1000     0.1495   ############
  的       0.1000     0.0446   ####
  新       0.1000     0.0972   ########
  手机     0.1000     0.2208   ##################

  平均法：其它 10 个词全是 0.1000，最大值 / 最小值 = 1.00（自己不算）
  相似度加权：最大 0.2208，最小 0.0446，最大值 / 最小值 = 4.95
```

*(Title: "Comparison one: the weights `苹果` gives each word". The four columns are
词 "word", 平均法 "averaging", 相似度加权 "similarity-weighted", 谁更被重视 "who gets taken
more seriously" (drawn as bars). The last two lines say: averaging gives all ten other
words `0.1000`, max / min `= 1.00` (itself not counted); similarity weighting gives
a max of `0.2208` and a min of `0.0446`, max / min `= 4.95`.)*

Similarity weighting gives `手机` ("phone") 0.2208 and `昨天` ("yesterday") 0.0446 — a
factor of 4.95 apart. It can tell heavy from light.
(That column doesn't have a name yet. Read on and you'll find out what it's called.)

**Comparison two: the perturbation experiment again.** Still adding a perturbation of
the same size to each word's vector, and looking at how much `苹果` moves:

```text
  改的是哪个词         平均法       相似度加权   倍数
  --------------------------------------------------------
  我              0.028284    0.017055    0.60x
  昨天            0.028284    0.016606    0.59x
  在              0.028284    0.016466    0.58x
  商场            0.028284    0.024832    0.88x
  看到            0.028284    0.016852    0.60x
  刚刚            0.028284    0.017822    0.63x
  发布            0.028284    0.056596    2.00x
  的              0.028284    0.016019    0.57x
  新              0.028284    0.036859    1.30x
  手机            0.028284    0.078303    2.77x

  平均法：最大 / 最小 = 1.000（完全一样）
  相似度加权：最大 / 最小 = 4.89
```

*(Columns: 改的是哪个词 "which word was changed", 平均法, 相似度加权, 倍数 "ratio". Closing
lines: averaging has max / min `= 1.000` (completely identical); similarity weighting
has max / min `= 4.89`.)*

In the averaging column, all ten numbers are identical. In the new method's column,
changing `手机` moves `苹果` 4.8 times as much as changing `的` does.

**Comparison three: how different is the vector that comes out?** Take `苹果`'s context
vector and compare it against the words in the sentence:

```text
  平均法            像不像      相似度加权        像不像
  --------------------------------------------------
  发布           0.6391      苹果           0.9404
  新             0.6103      手机           0.9248
  苹果           0.5937      发布           0.8223
  手机           0.5844      新             0.6498
  ...
  昨天           0.3873      的             0.1262
  我             0.3280      我             0.0994

  平均法：最高的 0.6391，最低的 0.3280，差 0.3111
  相似度加权：最高的 0.9404，最低的 0.0994，差 0.8410
  平均出来的向量和谁都差不多像，也就等于和谁都不特别像。
```

*(The two halves are "averaging / how alike" and "similarity weighting / how alike".
Closing lines: averaging — the highest is 0.6391, the lowest 0.3280, a gap of 0.3111;
similarity weighting — the highest is 0.9404, the lowest 0.0994, a gap of 0.8410; "an
averaged vector is more or less alike to everything, which is the same as being
especially alike to nothing".)*

The averaged vector sits between 0.33 and 0.64 in similarity to *every* word in the
sentence — it's a "blurry" vector. The new method's vector is 0.92 to `手机` ("phone")
and 0.10 to `我` ("I") — it has a preference.

## The new mechanism

Three steps, one function each.

**Step one: compute scores.** How alike are two vectors? The most direct way is the
**dot product**: multiply dimension by dimension and add it all up.

```python
score[i][j] = vectors[i] · vectors[j]
```

The more their directions agree, the bigger the score. Swap the two words and the
score doesn't change, because the dot product is symmetric.

**Step two: normalize into weights.** Scores can be any positive number of any size,
or even negative — they can't be used as weights directly. What we want is a set of
"probabilities": every one greater than 0, all of them summing to 1. That's the
`softmax` from Chapter 5:

```python
weights[i][j] = exp(score[i][j]) / sum_k exp(score[i][k])
```

All it does is one sentence: **big scores get pushed up, small scores get pushed down,
and not one of them is zeroed out.**

**Step three: weighted sum.** Use the weights to mix the whole sentence's vectors:

```python
output[i] = sum_j weights[i][j] * vectors[j]
```

After these three steps, every word has a new vector. This new vector is no longer
"what this word means in the dictionary" — it is "**what this word should be taken to
mean in this sentence**". It's a weighted mixture of the other words.

`weights` is a square matrix, and row i, column j means:

> How much word i is "looking at" word j.

This matrix is the thing we're going to be staring at for the rest of the way. It is
the entire secret of this mechanism.

This approach has a name. From today we'll call it **attention**.
The name sounds intimidating; it's really three sentences: **score similarity →
normalize → weighted sum**.

## Python implementation

That's `after.py`. The core is three functions:

```python
def similarity_scores(vectors):
    """第一步：两两算相似度。"""
    return vectors @ vectors.T


def attention_weights(scores):
    """第二步：把分数变成权重，每一行加起来等于 1。"""
    return scores.softmax(axis=-1)


def attention_output(vectors, weights):
    """第三步：加权求和。"""
    return weights @ vectors
```

All three are a single matrix multiplication, because these three steps *are* matrix
operations:

- `vectors` has shape `(number of words, 8)`
- `vectors @ vectors.T` gives the `(number of words, number of words)` score matrix
- `.softmax(axis=-1)` normalizes along each row, giving the weight matrix
- `weights @ vectors` gives the new `(number of words, 8)` vectors

Run `after.py` and it prints the full weight matrix:

```text
完整的权重矩阵
------------------------------------------------------------
             我   昨天     在   商场   看到   苹果   刚刚   发布     的     新   手机
  我      0.214  0.072  0.092  0.072  0.088  0.072  0.077  0.072  0.097  0.072  0.072
  昨天    0.068  0.194  0.068  0.068  0.068  0.068  0.175  0.068  0.068  0.084  0.068
  在      0.086  0.067  0.164  0.111  0.074  0.067  0.079  0.067  0.150  0.067  0.067
  商场    0.064  0.064  0.105  0.198  0.078  0.096  0.064  0.085  0.064  0.080  0.103
  看到    0.083  0.068  0.075  0.083  0.200  0.068  0.068  0.151  0.068  0.068  0.068
  苹果    0.045  0.045  0.045  0.067  0.045  0.198  0.045  0.149  0.045  0.097  0.221
  刚刚    0.065  0.158  0.072  0.062  0.062  0.062  0.198  0.062  0.075  0.124  0.062
  发布    0.044  0.044  0.044  0.059  0.098  0.148  0.044  0.248  0.044  0.082  0.146
  的      0.093  0.069  0.153  0.069  0.069  0.069  0.084  0.069  0.187  0.069  0.069
  新      0.053  0.064  0.053  0.066  0.053  0.115  0.106  0.098  0.053  0.216  0.123
  手机    0.041  0.041  0.041  0.066  0.041  0.203  0.041  0.136  0.041  0.096  0.251
  （行 = 谁在看，列 = 看谁。每一行的和都是 1。）
```

*(The line above the table is "the complete weight matrix". **Read the table like
this:** the column headers, left to right, are `我` I · `昨天` yesterday · `在` at ·
`商场` mall · `看到` saw · `苹果` apple · `刚刚` just now · `发布` released · `的` particle ·
`新` new · `手机` phone — the same eleven words as the sentence. The leftmost label on
each row is the same list. So the row labelled `苹果` is `苹果` doing the looking, and
the numbers across it are how much weight it puts on each of the eleven words. The
note inside the block says 行 = 谁在看，列 = 看谁 "row = who is looking, column = who is
looked at", and 每一行的和都是 1 "every row sums to 1".)*

A few cells worth reading side by side:

- The `苹果` (apple) row: `手机` (phone) 0.221, `发布` (released) 0.149, while `昨天`
  (yesterday) is only 0.045.
- The `昨天` row: the heaviest are itself and `刚刚` ("just now", 0.175) — the word in
  this sentence most like "yesterday" is "just now".
- The `的` row: the heaviest are `在` ("at", 0.153) and itself (0.187). Both are
  function words.

**Every row sums to 1, and every row is different.** The Chapter 12 problem of "everyone
looks at everyone the same way" is gone.

## What it solves

Three things, each backed by numbers above:

1. **Weights are no longer hard-coded.** Averaging's weights are always `1/10`;
   attention's weights are computed: `手机` 0.2208, `发布` 0.1495, `昨天` 0.0446 — a
   factor of 4.95 between the largest and the smallest.
2. **Influence follows the weights.** In the perturbation experiment, the effect of
   changing one word on `苹果` is 0.028284 for every word under averaging, and under
   attention it is 0.078 (`手机`) against 0.016 (`的`) — 4.89 times.
3. **The vector that comes out has a preference.** Averaging's context vector sits
   between 0.33 and 0.64 in similarity to everything; attention's vector is 0.92 to
   `手机` and 0.10 to `我`.

And the method is **general**: at the end, `after.py` tries three different center words:

```text
  苹果  最关注的三个词：手机(0.221)、苹果(0.198)、发布(0.149)
  手机  最关注的三个词：手机(0.251)、苹果(0.203)、发布(0.136)
  商场  最关注的三个词：商场(0.198)、在(0.105)、手机(0.103)
```

*(`最关注的三个词` = "the three words it pays most attention to". `、` is just the list
separator. Line 1: 苹果's top three are 手机 (0.221), 苹果 itself (0.198), 发布 (0.149);
line 2: 手机's are 手机 (0.251), 苹果 (0.203), 发布 (0.136); line 3: 商场 (`mall`)'s are
商场 (0.198), 在 (`at`, 0.105), 手机 (0.103).)*

Change the word you look from and you get a different set of weights. Averaging would
print three identical lines of 0.1000 here.

## What it still can't solve

Look at the `苹果` (apple) row: it gives most of its attention to `手机` (phone, 0.221)
and `发布` (released, 0.149). That looks pretty right — but it's right **by accident**.

Because "who to look at" is decided entirely by one thing: whether `苹果`'s vector and
that word's vector **are alike**. And "are alike" is a single number, a **symmetric**
number:

Run `similarity_scores` and you'll see it: the score matrix is symmetric.
`苹果 → 手机` and `手机 → 苹果` are the same number, `1.600`.

That's a problem. Suppose I want the `苹果` row to attend to **time** — because I want
to ask "when was it released", so I'd like the attention to land on `刚刚` ("just now").
Can't be done. The reason is direct:

- The similarity between `刚刚` and `苹果` is **0.000**, so its weight is pinned at
  0.0446 — exactly the same as `的`.
- The only thing I can adjust is `苹果`'s own vector. But that vector is simultaneously
  playing a second role: it's the vector **other words use when they look at 苹果**.
  If I reshape it into "looking for time", the way others look at `苹果` changes too.

In other words, right now there is only **one ruler**: are they alike. But "what I'm
looking for" and "what I have" are plainly two different things:

- What I'm looking for: I want to know "when was it released" → this is a **direction
  of questioning**
- What I have: this word carries the information "time" on it → this is the **content
  being asked about**

One ruler measuring two roles can only produce one number: "how alike are the two".

**So how do we get "what I'm looking for" and "what I have" to come apart?**

## Exercises

See `exercises.en.md`. The most important ones are listed here:

1. **What happens without normalization** — change `attention_weights` to return the
   scores directly, and see whether the weights still sum to 1, and which words turn
   into 0 or negative numbers.
2. **Make softmax "hard"** — give weight 1 only to the highest-scoring word and 0 to
   everything else. See what `苹果` turns into, then think about whether Chapter 9's
   `backward()` can still walk through it.
3. **A different kind of "alike"** — swap the dot product for cosine similarity, and
   see whether the gaps between the weights grow or shrink.
4. **One irrelevant word in the sentence** — add a `香蕉` ("banana"), and see how much
   averaging and attention are each affected.
5. **Make the "looker" into somebody else** — change `苹果`'s vector, and see whether
   "who 苹果 looks at" and "who looks at 苹果" change together.
