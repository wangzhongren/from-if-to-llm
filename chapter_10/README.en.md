**English** | [中文](README.md)

# Chapter 10: An ID number has no meaning

## The problem in this chapter

The last chapter left us one question: **a word is still only an ID number, and an ID number has no meaning.**

Let's take that sentence apart first and see what it is actually saying.

In Chapter 1 we split sentences into words, and then handed every word an integer:

```text
苹果 = 0    发布 = 1    新 = 2    手机 = 3    芯片 = 4    …    派 = 15
```

*(苹果 "apple" = 0, 发布 "release" = 1, 新 "new" = 2, 手机 "phone" = 3, 芯片 "chip" = 4, … 派 "pie" = 15.)*

This integer is called a token id. It lets us tell "which word this is" — that is its only use.

The engine from Chapter 9 is built now, so we can feed these into a model and let the model learn for itself.
But what we feed in is this integer. And that gives us two kinds of trouble.

**Trouble one: an ID number carries its own sense of "bigger" and "nearer", and we handed it out on a whim.**

Inside the IDs there is `3 > 2 > 1`. But what is "手机 > 新 > 发布" supposed to mean?
Now look at distance: 苹果's ID is 0, 香蕉's is 11, 手机's is 3.
By ID, 苹果 is close to 手机 (difference 3) and far from 香蕉 (difference 11).
But in meaning, 苹果 and 香蕉 are one kind of thing, and 手机 is not.
The distance inside the IDs is a shadow left behind by the order we wrote the vocabulary in. It has nothing whatsoever to do with what the words mean.

**Trouble two: we hand out the IDs. Change the vocabulary order and every ID changes.**

15 words have 15! ways of being numbered — over 1.3 trillion.
We only wrote the vocabulary as "tech words first, then food words" on a whim, and that is the only reason the IDs look the way they do.
Arrange it any other way and everything Chapter 9 trained so painfully has to start over.

Neither of these is a guess. We'll run it and show you.

## The simplest attempt

The most intuitive move is: **ignore it and keep using it.**

That's exactly what we did in Chapter 3 — turn the input into numbers, multiply by weights, add them up, get a score.
Here we're only swapping "whether a word is in the sentence" for "which word of the vocabulary this is".

Let's give the model a task that could not be simpler:

```text
只在科技句里出现过的词：发布 新 手机 芯片 电脑 华为 小米
只在食品句里出现过的词：好吃 很 甜 香蕉 这个 真 做成 派
```

*(Words that appear only in tech sentences: 发布 release, 新 new, 手机 phone, 芯片 chip, 电脑 computer, 华为 Huawei, 小米 Xiaomi. Words that appear only in food sentences: 好吃 delicious, 很 very, 甜 sweet, 香蕉 banana, 这个 this, 真 really, 做成 make-into, 派 pie.)*

Those 15 words were grouped by counting the ten sentences from Chapter 1 — we didn't invent the grouping.
(苹果 appears on both sides. This chapter we set it aside; Chapter 12 is where it gets its own treatment.)

The model is still the same model, with only two parameters:

```python
score = w * token_id + b      # 分数低的算科技词，分数高的算食品词
```

*(The comment says: a low score counts as a tech word, a high score counts as a food word.)*

`w` and `b` get trained with toygrad. This looks perfectly reasonable — it's what we did in Chapter 3 and Chapter 6.
Let's run it.

## Experiment

Run `before.py` first. It does the same thing three times: the same batch of words, the same model, the same task. The only difference is **what order the vocabulary is in** (which is to say, how the IDs get handed out).

Scheme A is the vocabulary from Chapter 1, where the tech words happen to come first:

```text
编号方案 A  第 1 章的词表顺序（科技词恰好排在前面）
==============================================================
词表：苹果=0  发布=1  新=2  手机=3  芯片=4  电脑=5  华为=6  小米=7  好吃=8  很=9  甜=10  香蕉=11  这个=12  真=13  做成=14  派=15

训练 3000 步，最终 loss = 0.0912
w = +1.173   b = -8.608

  编号   词     分数     应该属于   分数算出来的类别
  ----------------------------------------------------
     1   发布   -7.44    科技         科技
     2   新     -6.26    科技         科技
     3   手机   -5.09    科技         科技
     4   芯片   -3.91    科技         科技
     5   电脑   -2.74    科技         科技
     6   华为   -1.57    科技         科技
     7   小米   -0.39    科技         科技
     8   好吃   +0.78    食品         食品
     9   很     +1.95    食品         食品
    10   甜     +3.13    食品         食品
    11   香蕉   +4.30    食品         食品
    12   这个   +5.47    食品         食品
    13   真     +6.65    食品         食品
    14   做成   +7.82    食品         食品
    15   派     +8.99    食品         食品

准确率 = 100.0%
分数只能沿着编号 往上走 —— 因为 w = +1.173 这一个数就定死了方向。
也就是说：发布   和 派     被放在了分数轴的两个极端。
这两个词凭什么站在两端？凭它们的编号。
```

*(Scheme A. The header reads "Numbering scheme A — Chapter 1's vocabulary order (the tech words happen to come first)"; the next line prints the whole vocabulary with its IDs. Each row of the table gives a word's ID, the word, its score, the class it should belong to, and the class its score puts it in. All 15 rows agree. 发布 "release" ends up at the low end of the score line and 派 "pie" at the high end, purely because they are ID 1 and ID 15.)*

100% correct. Looks like it worked.

But look carefully at that table: **the score climbs all the way along the IDs without bending once.**
That is not a rule the model "learned" — it is the only thing the model can do. With a single `w`, the score can only be a straight line over the IDs.
发布 sits at the lowest-score end and 派 at the highest-score end. What entitles those two? One is number 1, the other is number 15.

Now switch to a different numbering. The same batch of words, the same table, just reordered by pinyin (Chinese romanisation, i.e. alphabetical order):

```text
编号方案 B  按拼音排序（同一批词，同一张表）
==============================================================
词表：电脑=0  发布=1  好吃=2  很=3  华为=4  派=5  苹果=6  手机=7  甜=8  香蕉=9  小米=10  新=11  芯片=12  这个=13  真=14  做成=15

训练 3000 步，最终 loss = 0.6638
w = +0.101   b = -0.624

  编号   词     分数     应该属于   分数算出来的类别
  ----------------------------------------------------
     1   发布   -0.52    科技         科技
    11   新     +0.48    科技         食品   <-- 错
     7   手机   +0.08    科技         食品   <-- 错
    12   芯片   +0.58    科技         食品   <-- 错
     0   电脑   -0.62    科技         科技
     4   华为   -0.22    科技         科技
    10   小米   +0.38    科技         食品   <-- 错
     2   好吃   -0.42    食品         科技   <-- 错
     3   很     -0.32    食品         科技   <-- 错
     8   甜     +0.18    食品         食品
     9   香蕉   +0.28    食品         食品
    13   这个   +0.68    食品         食品
    14   真     +0.78    食品         食品
    15   做成   +0.89    食品         食品
     5   派     -0.12    食品         科技   <-- 错

准确率 = 53.3%
```

*(Scheme B. Same words, same table, IDs handed out in pinyin order. Now read the ID column: 1, 11, 7, 12, 0, 4, 10, 2, 3, 8, 9, 13, 14, 15, 5 — tech words and food words strictly alternating. The rows marked `错` are the ones the line gets wrong; accuracy drops to 53.3%.)*

Look at that ID column: tech words and food words **alternate**.
ID 0 is a tech word, ID 2 is a food word, ID 4 is a tech word again…
Cut with a straight line and wherever you cut, you cut wrong.

That is why the loss sits at 0.66 and won't move — not because we didn't train long enough, but because there is nothing to learn.

```text
==============================================================
同一个模型、同一批词、同一个任务，只换了编号
==============================================================
  A  第 1 章的词表顺序（科技词恰好排在前面）   准确率 100.0%
  B  按拼音排序（同一批词，同一张表）          准确率  53.3%
  C  随机打乱（同一批词，同一张表）            准确率  66.7%
```

*(The same model, the same batch of words, the same task — only the numbering changed. A: Chapter 1's vocabulary order, 100.0%. B: pinyin order, 53.3%. C: random shuffle, 66.7%.)*

Now take the `w` and `b` that scheme A trained and drop them, untouched, onto scheme B's numbering:

```text
准确率 = 60.0%
```

*(Accuracy = 60.0%.)*

Better than blind guessing (53.3%), but a world away from 100%.
**The model didn't change a single character — only the numbering did.**

`experiment.py` puts a number on this:

```text
==============================================================
实验一：编号里的「远近」
==============================================================
  A  第 1 章的词表顺序 苹果 的编号 =  0
                       编号上离它最近：发布   距离 1
                       编号上离它最远：派     距离 15
  B  按拼音排序        苹果 的编号 =  6
                       编号上离它最近：手机   距离 1
                       编号上离它最远：做成   距离 9
  C  随机打乱          苹果 的编号 =  4
                       编号上离它最近：甜     距离 1
                       编号上离它最远：派     距离 11

  三种编号下，「离 苹果 最近的词」分别是：发布、手机、甜。
  三个答案互相矛盾，而它们说的是同一个 苹果。
```

*(Experiment one: "near" and "far" inside the IDs. For each scheme it prints 苹果's ID and which word sits closest and furthest from it on the ID line. So "the word closest to 苹果" comes out as 发布 under scheme A, 手机 under scheme B, 甜 under scheme C — three contradictory answers, about one and the same 苹果.)*

Three answers that contradict each other, and none of them is the "right" one. Nearness inside the IDs is not a thing you can use at all.

The second experiment counted how many numbering schemes like this are learnable.

```text
==============================================================
实验二：能「学会」的编号方案有多少种
==============================================================
  15 个词，编号方案一共有 15! = 1,307,674,368,000 种。
  一条直线只能把编号切一刀 —— 想让 7 个科技词全在一边、
  8 个食品词全在另一边，只有两种排法：科技词全都排在前面，或者全排在后面。
  这样的排法有 2 x 7! x 8! = 406,425,600 种。
  精确比例 = 0.00031080  ≈ 1 / 3218

  随机撒 200,000 种编号方案，实际数出来能分开的有 67 种
  实际比例 = 0.00033500
  平均要试 2985 次才碰上一次。

  换句话说：如果模型能学会，功劳多半在编号，不在模型。
```

*(Experiment two. There are 15! = 1,307,674,368,000 numbering schemes. A single straight line can only make one cut, so to get all 7 tech words on one side and all 8 food words on the other, only two arrangements work: all the tech words first, or all of them last. That is 2 × 7! × 8! = 406,425,600 schemes — a proportion of 0.00031080, roughly 1 in 3218. Sampling 200,000 random schemes found 67 separable ones, i.e. 0.00033500, so on average you'd try 2985 times to hit one. The count is over 15 words, not 16: 苹果 is excluded because it appears in both classes.)*

1.3 trillion numbering schemes, and only three in ten thousand are learnable. That 100% accuracy of ours was purely because, when we wrote the vocabulary back then, we "happened" to put the tech words first.

The third experiment puts the three loss curves side by side:

```text
==============================================================
实验三：三种编号方案下的 loss 曲线
==============================================================
  步数           A         B         C
  ----------------------------------
   300      0.2418    0.6641    0.6484
   600      0.1777    0.6639    0.6478
   900      0.1491    0.6638    0.6478
  1200      0.1321    0.6638    0.6478
  1500      0.1205    0.6638    0.6478
  1800      0.1119    0.6638    0.6478
  2100      0.1052    0.6638    0.6478
  2400      0.0997    0.6638    0.6478
  2700      0.0951    0.6638    0.6478
  3000      0.0912    0.6638    0.6478

  A 的 loss 一路往下掉，而且还在往下掉；
  B 和 C 从第 300 步起就一动不动 —— 它们已经撞到那条直线的天花板了。
  三个模型一模一样，词一模一样，任务一模一样。
  唯一的区别是：编号是怎么发的。
```

*(Experiment three: loss over 3000 steps under the three schemes. A falls the whole way (0.2418 → 0.0912) and is still falling; B and C are frozen at 0.6638 and 0.6478 from step 300 onward — they have already hit the ceiling of that straight line.)*

B and C aren't "trained for too short a time". There is **nothing to learn**. `w` can only make one cut in the IDs, and in both of these numberings the tech words and food words alternate, so wherever you cut, you cut wrong. Training longer cannot change that.

The conclusion is plain: **with IDs as input, what the model learns is the arrangement order of the IDs, not the meaning of the words.**

## The new mechanism

The trouble with an ID is that it is too poor: a word gets one number, and that number has to carry both "which position in the list this is" and "what this word is". Those two things were never related, and cramming them together only makes them interfere with each other.

So let's stop cramming. **A word is not one number — it is a list of numbers.**

```text
苹果 = [+0.13, -0.13, +0.64, +0.10, -0.54, +0.36, +1.30, +0.95, …]
香蕉 = [-1.47, +1.03, -1.93, -0.24, -0.20, -1.04, +0.61, -0.20, …]
手机 = [+1.80, +1.32, +0.36, -1.21, -0.00, +0.66, -1.29, +0.40, …]
```

*(Each word is now a list of 16 numbers, written out the same way for 苹果, 香蕉 and 手机.)*

(Here each word is given 16 numbers. Why 16? We'll come to that in a moment.)

Why does this fix it? Watch two things.

**First, every word gets a direction of its own.** The model can no longer only say "whose ID is bigger". It says:

```python
score = 向量 · w + b
```

*(score = vector · w + b)*

`w` is 16 numbers too. How high a word's score goes depends on whether its 16 numbers and `w`'s 16 numbers "get along". Want 手机 to score high and 香蕉 low? Point `w` in 手机's direction. That has nothing to do with which position they occupy — the ID never enters the model at all.

**Second, the ID has been taken out of the model.** A vector travels with its word, not with its position. Reorder the vocabulary and we only have to carry the table's rows along with their words; the model sees exactly the same thing.

From here on we call this thing an **embedding** — 词向量 in Chinese, literally "word vector".

One thing must be said clearly: **in this chapter we do not train this group of numbers.** It is scattered at random to begin with, and after that it doesn't move. The only things trained are `w` and `b`.
We do it this way because this chapter wants to answer exactly one question:

> Is a group of **random** numbers really better than a carefully arranged ID?

If even this beats IDs, then the trouble really is with "IDs" — not with "the random numbers not being good enough".

## Python implementation

The full code is in `after.py`. The core is two pieces.

The first piece builds a read-only table. The table's shape is `(number of words, numbers per word)`:

```python
DIM = 16   # 每个词用几个数字

def build_vector_table():
    return randn(len(VOCAB), DIM, requires_grad=False, seed=SEED)
```

`randn` is toygrad's normal-distribution initialiser. Note `requires_grad=False` — this table is read-only; backpropagation won't touch it.

Why 16 numbers? Two reasons.
One, the number itself is free to tune (in real models it runs from hundreds to thousands).
Two, we have only 15 words here, and 16 numbers is already more numbers than there are words — 15 randomly scattered points in a 16-dimensional space can almost always be split into two piles by a plane in any way you like. Exercise 2 has you turn the dimension down to 2 and 4, and see with your own eyes what happens when there aren't enough dimensions.

To look up the table, use the `embedding` from Chapter 9, or simply index into it as we do here:

```python
vectors = {word: table.data[i] for i, word in enumerate(VOCAB)}
```

The second piece is training. **The task is exactly the same as `before.py`** — we've only swapped "the ID" for "the vector":

```python
def train_scores(vectors, labels, steps=STEPS, lr=LR):
    x = Tensor(np.stack(vectors))            # (15, 16)，每个词一行
    w = randn(DIM, requires_grad=True, seed=123)
    b = Tensor(0.0, requires_grad=True)
    ...
    score = x @ w + b
```

`x @ w` is a matrix times a vector: `(15, 16) @ (16,) -> (15,)`.
Each word's score is the dot product of its 16 numbers with `w`.

Running it looks like this:

```text
==============================================================
每个词现在是一组 16 个数字（随机撒的，没有训练过）
==============================================================
  苹果   = [+0.13  -0.13  +0.64  +0.10  -0.54  +0.36  +1.30  +0.95  -0.70  -1.27  -0.62  +0.04  -2.33  -0.22  -1.25  -0.73]
  香蕉   = [-1.47  +1.03  -1.93  -0.24  -0.20  -1.04  +0.61  -0.20  -0.44  +0.52  -0.48  +1.39  +0.35  -0.47  -1.94  -1.31]
  手机   = [+1.80  +1.32  +0.36  -1.21  -0.00  +0.66  -1.29  +0.40  +0.43  +0.70  -1.18  -0.66  -0.44  -1.17  +1.74  -0.50]
  很     = [+0.32  -0.36  -1.90  -0.11  -0.80  +1.08  -0.29  +0.08  -0.85  -0.51  -0.01  -1.49  +0.30  -0.11  -1.19  -2.40]

==============================================================
同一个任务（把科技词和食品词分开），三种编号方案
==============================================================
  A  第 1 章的词表顺序 loss = 0.0203   准确率 = 100.0%
  B  按拼音排序        loss = 0.0203   准确率 = 100.0%
  C  随机打乱          loss = 0.0203   准确率 = 100.0%

三种编号方案的准确率一模一样。
因为模型从头到尾就没有见过编号 —— 它见到的是向量。

==============================================================
编号在变，每个词的分数一动不动
==============================================================
  词     编号A   编号B    分数     应该属于   分数算出来的类别
  ------------------------------------------------------------
  发布      1      1   -3.04    科技         科技
  新        2     11   -7.39    科技         科技
  手机      3      7   -3.55    科技         科技
  芯片      4     12   -6.60    科技         科技
  电脑      5      0   -4.71    科技         科技
  华为      6      4   -4.50    科技         科技
  小米      7     10   -3.14    科技         科技
  好吃      8      2   +3.39    食品         食品
  很        9      3   +3.85    食品         食品
  甜       10      8   +4.41    食品         食品
  香蕉     11      9  +15.20    食品         食品
  这个     12     13  +10.88    食品         食品
  真       13     14   +7.73    食品         食品
  做成     14     15   +3.06    食品         食品
  派       15      5   +2.94    食品         食品
```

*(The output has three parts. First it prints the raw 16-number rows for 苹果, 香蕉, 手机, 很. Then the same task under the three numbering schemes: loss = 0.0203 and accuracy = 100.0% for A, B and C alike. Then a table with the ID each word has under scheme A and under scheme B side by side next to its score — the IDs jump around and the scores don't move.)*

Look at the middle table: **the ID goes from 2 to 11, from 5 to 0, and the score doesn't move at all.**

The last piece of code confirms that the table really was left alone during training:

```python
snapshot = table.data.copy()   # 训练前留个底
...
print(f"训练前后最大的变化量 = {np.abs(table.data - snapshot).max():.7f}")
```

```text
==============================================================
最后确认一下：表里的数字从头到尾没有变过
==============================================================
  表里一共 16 x 16 = 256 个数字
  训练前后最大的变化量 = 0.0000000
  被训练的只有 w 和 b，一共 17 个数，表本身是只读的。
```

*(Final check: the table holds 16 × 16 = 256 numbers, and the largest change from before training to after is 0.0000000. Only `w` and `b` were trained — 17 numbers in all — and the table itself is read-only.)*

## What it solves

Put the two chapters side by side:

| | ID number (before Chapter 10) | A list of numbers (Chapter 10) |
|---|---|---|
| Accuracy under vocabulary order A | 100.0% | 100.0% |
| Accuracy under vocabulary order B | 53.3% | 100.0% |
| Accuracy under vocabulary order C | 66.7% | 100.0% |
| Scheme A's model moved onto scheme B | 60.0% | 100.0% (identical to begin with) |
| How many parameters control these 15 words | 2 (`w` and `b`) | 17 (`w` and `b`) + 16 numbers per word |
| Is the score locked to the ID | Yes — only a straight line over the IDs | No — every word has a direction of its own |

Concretely, this chapter gets us three things.

1. **The ID no longer affects the result.** All three numbering schemes give 100% accuracy, because the model never sees an ID from start to finish — it sees vectors.
2. **"Who is near whom" has been taken out of the IDs.** The distance inside the IDs was a by-product of how we ordered the vocabulary; now it doesn't enter the model, so it can't cause trouble any more.
3. **Every word has a direction of its own.** The relations the model can express went from one number to 16 numbers. Giving 手机 a high score and 派 a high score are no longer in conflict.

The old problem from Chapter 1 — "苹果 appears in both classes of sentence" — is still hanging.
What this chapter achieved is: we finally have a **container that can hold more than one side of a word**. Each word has 16 numbers, which can hold far more than a single number can.

## What it still can't solve

Back to the last part of `after.py`'s output. We did something very natural: now that every word is a list of numbers, **let's compare which is like which**.

The comparison is called **cosine similarity**: the more two vectors point the same way, the closer it gets to 1; perpendicular is 0; opposite is -1. It ignores how long the vectors are and looks only at direction.

```text
==============================================================
用余弦相似度找「最像的词」
==============================================================
  和 手机   最像的 5 个词： 派 +0.28  这个 +0.15  电脑 +0.13  做成 +0.09  小米 +0.09
  和 苹果   最像的 5 个词： 华为 +0.36  做成 +0.25  发布 +0.24  很 +0.21  小米 +0.19
```

*(Cosine similarity, top 5 words nearest to 手机 "phone" and to 苹果 "apple". The five words nearest 手机 are 派 pie +0.28, 这个 this +0.15, 电脑 computer +0.13, 做成 make-into +0.09, 小米 Xiaomi +0.09; the five nearest 苹果 are 华为 Huawei +0.36, 做成 +0.25, 发布 +0.24, 很 very +0.21, 小米 +0.19.)*

The word most like 手机 is 派. The word most like 苹果 is 做成.

This is nonsense. And not the "a little off" kind of nonsense — the highest similarity is only +0.28, which says these 16 numbers aren't comparable to each other at all, and who comes out on top is pure luck.

Where is the trouble? Back to this chapter's own sentence: **this group of numbers was scattered at random.**

We built the **container** that says "a word is a list of numbers". As for what numbers to put inside it, we decided by rolling dice.
The model has no choice: it has to force one direction out of that heap of random numbers and use it — like looking for a pattern in noise. Whatever it finds, of course it isn't the meaning of the word.

Which leaves a very natural question:

> **Where should this group of numbers come from?**

There are two possibilities.

- One: random is good enough, we just need to scatter more numbers and train a bit longer.
  Then we would have to say how good "good enough" is — at the very least it should let us answer "who is like whom".
- The other: this group of numbers should not be scattered at all, it should be **learned out of the corpus**.
  A word's meaning, after all, already shows itself in "which words it usually appears alongside".

The similarity table in `after.py` has already told us that the first possibility doesn't hold:
there is no structure among random numbers, and no amount of training can make `w` conjure structure out of them,
because `w` has only 16 numbers and it **reads** this table, it does not **change** it.

So only one road is left: **make this table itself a trainable parameter.**

Which forces a more concrete question: **what task do we train it on?**
A random table — what do we feed it so that it puts 香蕉 and 苹果 together, and 手机 and 芯片 together?

## Exercises

See `exercises.en.md`. Here are the 5 most important ones:

1. Change the random seed for scheme C in `before.py` a few times and see what range the accuracy jumps around in. What does that tell you?
2. In `after.py`, change `DIM` to 2, 4, 8 and run each. Once the dimension is small, why do the vectors fail to separate as well? Is that the same reason as "the IDs can't separate"?
3. In `before.py` the model has only two parameters, `w` and `b`. If you give every ID an offset of its own (that is, `score = w * id + b_id`), can scheme B learn? What are those 15 `b_id`s?
4. In `after.py`'s similarity table, the word most like 手机 is 派. Change the seed to 1 through 5, run each, and see who becomes the word most like 手机. Is there a pattern in those answers?
5. Without changing the model and without changing the words, only the ID order, try to construct a numbering scheme that is **less accurate than scheme A but more accurate than scheme B**. Does such a scheme exist?
