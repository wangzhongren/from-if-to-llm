**English** | [中文](README.md)

# Chapter 11: Bringing similar words closer

## The problem in this chapter

The last chapter ended on a question: **where do these numbers come from? Are they random?**

We weren't asking idly. Chapter 10 finished with this output:

```text
和 手机   最像的 5 个词： 派 +0.28  这个 +0.15  电脑 +0.13  做成 +0.09  小米 +0.09
和 苹果   最像的 5 个词： 华为 +0.36  做成 +0.25  发布 +0.24  很 +0.21  小米 +0.19
```

*(From Chapter 10: the five words most similar to 手机 "phone" and to 苹果 "apple", in the random table. 派 "pie" tops the list for 手机.)*

The word most like 手机 is 派. That isn't "a bit off the mark" — that is nonsense.

`before.py` digs into this a little further. It takes Chapter 10's random table and does three things:

(Chapter 10's table gave every word 16 numbers; this chapter switches to 8 across the board, so that it lines up with the table we train later. This doesn't change the conclusion — random numbers have no structure whether you have eight of them or sixteen; we tried both.)

**First, look at what those "similar words" actually look like.**

```text
  和 手机   最像的 5 个词： 派 +0.72  华为 +0.71  这个 +0.57  真 +0.41  发布 +0.35
  和 苹果   最像的 5 个词： 芯片 +0.78  电脑 +0.62  很 +0.61  甜 +0.59  做成 +0.46
  和 好吃   最像的 5 个词： 电脑 +0.41  香蕉 +0.34  芯片 +0.25  苹果 +0.14  很 +0.05

  和 手机 最像的词是 派（+0.72）。手机 和 派 有什么关系？没有。
  和 苹果 最像的词是 芯片（+0.78），这个看着倒像那么回事 ——
  但那只是运气。随机数偶尔会撞出一个「看起来对」的答案，
  这正是它最会骗人的地方。下面换个种子就知道。
```

*(The top 5 words most similar to 手机 "phone", to 苹果 "apple" and to 好吃 "delicious" in the random table. The commentary underneath the table says: the word most like 手机 is 派 "pie" (+0.72) — what does a phone have to do with a pie? Nothing. The word most like 苹果 is 芯片 "chip" (+0.78), and that one does look plausible — but it is only luck. Random numbers will occasionally collide into an answer that "looks right", and that is exactly where they fool you the most. Change the seed below and you'll see.)*

**Second, change the seed and see whether the answer holds still.**

```text
  种子  和「手机」最像的 3 个词
  --------------------------------------------
     0  派 +0.72  华为 +0.71  这个 +0.57
     1  电脑 +0.78  新 +0.29  好吃 +0.28
     2  发布 +0.67  华为 +0.43  甜 +0.39
     3  新 +0.57  这个 +0.56  芯片 +0.34
     4  华为 +0.44  香蕉 +0.27  派 +0.17

  每换一个种子，答案就换一批。
  如果这张表里真的装着「意义」，它不该对种子这么敏感。
```

*(Five seeds, and the three words most similar to 手机 "phone" under each. Every seed produces a different list. If this table really did hold "meaning" inside it, it would not be this sensitive to the seed.)*

**Third, directly check whether the table contains the structure we want.**

The structure we want is unambiguous: the tech words (发布 release, 新 new, 手机 phone, 芯片 chip, 电脑 computer, 华为 Huawei, 小米 Xiaomi) should sit close to one another, the food words (好吃 delicious, 很 very, 甜 sweet, 香蕉 banana, 这个 this, 真 really, 做成 make-into, 派 pie) should sit close to one another, and the two clumps should be further apart.

So we compute one number: **the average similarity within a group, minus the average similarity between the two groups.**
If there is structure, it should be positive, and not small.

```text
  种子    组内(科技)  组内(食品)  组间    组内均值 - 组间
  --------------------------------------------------------
     0    -0.058      -0.032      +0.062   -0.107
     1    -0.092      -0.067      +0.010   -0.089
     2    +0.056      -0.058      +0.059   -0.060

  上面是前三个种子。把 30 个种子都试一遍：
    平均 -0.030    最小 -0.118    最大 +0.071
  不管种子怎么换，这个差距都在 0 附近打转（而且还常常是负的）。
  也就是说：科技词之间并不比科技词和食品词之间更像。
```

*(For the first three seeds: average similarity within the tech group, within the food group, between the groups, and the difference. Trying all 30 seeds, the difference averages −0.030, with a minimum of −0.118 and a maximum of +0.071. However you change the seed, the gap wanders around zero — and is usually negative. Which is to say: tech words are no more similar to each other than they are to food words.)*

Negative. The within-group number is even a little lower than the between-group one.

So the problem is not "this seed isn't good enough". **The problem is that random numbers have no relation to each other at all**, and no amount of changing seeds can repair that. For this table to grow structure there is exactly one road: **treat it as a parameter and train it.**

## The simplest attempt

Train it on what task, then? What do we actually have? — 13 sentences.

Nobody ever told us "手机 and 芯片 are one kind of thing", and not one sentence in the corpus says so either.
But the corpus does have something ready-made: **who stands next to whom.**

Look back at that batch of sentences from Chapter 1:

```text
苹果 发布 新 手机
苹果 很 好吃
香蕉 很 好吃
```

*(苹果 发布 新 手机 — "Apple shipped a new phone"; 苹果 很 好吃 — "apples are delicious"; 香蕉 很 好吃 — "bananas are delicious".)*

Take the two neighbouring words out of each sentence and you get a pile of small facts:

```text
苹果 旁边是 发布        苹果 旁边是 很
发布 旁边是 新          很 旁边是 好吃
新   旁边是 手机        好吃 旁边是 香蕉
```

*(Each line says "X 旁边是 Y" — "X stands next to Y". So: 苹果 is next to 发布, 苹果 is next to 很, 发布 is next to 新, 很 is next to 好吃, 新 is next to 手机, 好吃 is next to 香蕉.)*

The meaning of the word 手机 is hidden inside these small facts:
the word standing next to it is 新, and the word standing next to 新 is 发布.
Collect every neighbouring word pair in the corpus and we have a batch of training samples that needed no human labelling at all.

The first idea that comes to mind is **counting directly**: build a table and count how many times each pair of words shows up together.
苹果–发布 four times, say, and 苹果–好吃 twice.
And then what? That table is a heap of numbers; it won't turn itself into "who should be near whom". We'd still have to find a way to turn the heap into vectors — that's another layer of trouble.

Better to go the other way round. **Don't count. Build a task.**

```text
看到 苹果  →  猜 它旁边站的是谁（答案是 发布 或 很 …）
```

*(See 苹果 → guess who is standing beside it (the answer is 发布 or 很 …).)*

The nice thing about this task is that we have seen its shape before, in Chapter 5.
Score a bunch of options, compute the loss, backpropagate, change the parameters — the full toolkit has been sitting there since Chapter 9.
And this task **needs nobody to label the answers**: the answers are written in the corpus.

Which leaves exactly one question: **who are we training?**

## Experiment

Run `after.py`. It does one thing:

```text
==============================================================
语料和训练样本
==============================================================
  句子 13 条，词 16 个，维度 8
  训练样本 68 条，都是「看到某个词 → 猜它旁边的词」
  前 6 条：
    看到 发布   → 猜 苹果
    看到 苹果   → 猜 发布
    看到 新     → 猜 发布
    看到 发布   → 猜 新
    看到 手机   → 猜 新
    看到 新     → 猜 手机

==============================================================
训练 400 轮之后的 loss = 1.0748
==============================================================
  （随机乱猜的话，16 个词的 loss 是 ln(16) ≈ 2.77。）
```

*(The corpus: 13 sentences, 16 words, dimension 8. 68 training samples, all of the form "see a word → guess the word beside it" — the first 6 are printed. After 400 epochs the loss is 1.0748; guessing at random over 16 words would give ln(16) ≈ 2.77.)*

Then look at the neighbours:

```text
==============================================================
每个词的「邻居」——和它最像的 5 个词
==============================================================
  手机   → 芯片 0.99  电脑 0.98  发布 0.58  做成 0.08  小米 -0.01
  芯片   → 手机 0.99  电脑 0.98  发布 0.57  小米 0.08  华为 0.06
  发布   → 手机 0.58  芯片 0.57  电脑 0.54  这个 0.36  很 0.24
  新     → 华为 0.69  小米 0.66  苹果 0.21  派 0.08  电脑 -0.07
  好吃   → 甜 0.95  香蕉 0.72  苹果 0.61  小米 0.07  派 0.05
  很     → 真 0.98  这个 0.71  做成 0.48  发布 0.24  甜 -0.04
  香蕉   → 苹果 0.82  好吃 0.72  甜 0.68  派 0.60  做成 0.10
  派     → 香蕉 0.60  苹果 0.36  这个 0.31  做成 0.24  新 0.08
  苹果   → 香蕉 0.82  华为 0.61  好吃 0.61  小米 0.61  甜 0.56
```

*(Each line is one word followed by its five nearest words with the cosine similarity between them. Read the clusters: 手机 "phone" → 芯片 "chip" 0.99 and 电脑 "computer" 0.98, and they point straight back at 手机 0.99 and 0.98 — a tight cluster of phone, chip and computer. 好吃 "delicious" → 甜 "sweet" 0.95; 很 "very" → 真 "really" 0.98; 香蕉 "banana" → 苹果 "apple" 0.82. Note also 新 "new" → 华为 "Huawei" 0.69 and 小米 "Xiaomi" 0.66, and that the whole food side scores around 0.08 or below on the tech side.)*

Compare that with the random table in `before.py` and the change is total:

| Question | Random table (Chapter 10) | After training (Chapter 11) |
|---|---|---|
| Word most like 手机 "phone" | 派 "pie" (+0.72) | 芯片 "chip" (+0.99) |
| Word most like 芯片 "chip" | 电脑 "computer" (+0.81) | 手机 "phone" (+0.99) |
| Word most like 好吃 "delicious" | 电脑 "computer" (+0.41) | 甜 "sweet" (+0.95) |
| Word most like 香蕉 "banana" | 电脑 "computer" (+0.69) | 苹果 "apple" (+0.82) |
| Does changing the seed change the answer | Every time | No (the seed is fixed in the code) |

(Two of the four answers in the random-table column happen to "look reasonable" — 电脑 really is a tech word. But change the seed a couple of times in `before.py` and you'll find that was pure collision.)

The full similarity table looks like this (these are the cosine similarities between all 16 words, printed in two halves so the columns line up in a terminal):

```text
      苹果  发布  新    手机  芯片  电脑  华为  小米
苹果    1.00 -0.68  0.21 -0.52 -0.46 -0.49  0.61  0.61
发布   -0.68  1.00 -0.20  0.58  0.57  0.54 -0.38 -0.40
新      0.21 -0.20  1.00 -0.22 -0.12 -0.07  0.69  0.66
手机   -0.52  0.58 -0.22  1.00  0.99  0.98 -0.03 -0.01
芯片   -0.46  0.57 -0.12  0.99  1.00  0.98  0.06  0.08
电脑   -0.49  0.54 -0.07  0.98  0.98  1.00  0.06  0.08
华为    0.61 -0.38  0.69 -0.03  0.06  0.06  1.00  0.99
小米    0.61 -0.40  0.66 -0.01  0.08  0.08  0.99  1.00
好吃    0.61 -0.44 -0.26 -0.43 -0.41 -0.52  0.04  0.07
很     -0.51  0.24 -0.55 -0.19 -0.29 -0.32 -0.72 -0.73
甜      0.56 -0.51 -0.26 -0.36 -0.35 -0.45  0.07  0.12
香蕉    0.82 -0.71 -0.13 -0.73 -0.73 -0.74  0.08  0.09
这个   -0.62  0.36 -0.49 -0.10 -0.18 -0.14 -0.93 -0.95
真     -0.51  0.12 -0.54 -0.21 -0.32 -0.33 -0.73 -0.72
做成   -0.21 -0.14 -0.64  0.08 -0.02  0.01 -0.53 -0.54
派      0.36 -0.48  0.08 -0.62 -0.64 -0.52 -0.11 -0.11

      好吃  很    甜    香蕉  这个  真    做成  派
苹果    0.61 -0.51  0.56  0.82 -0.62 -0.51 -0.21  0.36
发布   -0.44  0.24 -0.51 -0.71  0.36  0.12 -0.14 -0.48
新     -0.26 -0.55 -0.26 -0.13 -0.49 -0.54 -0.64  0.08
手机   -0.43 -0.19 -0.36 -0.73 -0.10 -0.21  0.08 -0.62
芯片   -0.41 -0.29 -0.35 -0.73 -0.18 -0.32 -0.02 -0.64
电脑   -0.52 -0.32 -0.45 -0.74 -0.14 -0.33  0.01 -0.52
华为    0.04 -0.72  0.07  0.08 -0.93 -0.73 -0.53 -0.11
小米    0.07 -0.73  0.12  0.09 -0.95 -0.72 -0.54 -0.11
好吃    1.00 -0.06  0.95  0.72 -0.27 -0.03 -0.22  0.05
很     -0.06  1.00 -0.04 -0.12  0.71  0.98  0.48 -0.11
甜      0.95 -0.04  1.00  0.68 -0.34  0.01 -0.19 -0.04
香蕉    0.72 -0.12  0.68  1.00 -0.12 -0.09  0.10  0.60
这个   -0.27  0.71 -0.34 -0.12  1.00  0.71  0.57  0.31
真     -0.03  0.98  0.01 -0.09  0.71  1.00  0.52 -0.04
做成   -0.22  0.48 -0.19  0.10  0.57  0.52  1.00  0.24
派      0.05 -0.11 -0.04  0.60  0.31 -0.04  0.24  1.00
```

*(The complete 16 × 16 similarity matrix. The left half uses the first 8 words as columns (苹果 through 小米), the right half the last 8 (好吃 through 派); each row is one word, so reading across both halves gives that word's similarity to all 16. Everything on the diagonal is 1.00. Find the 手机 "phone" row: 芯片 "chip" 0.99, 电脑 "computer" 0.98, 发布 "release" 0.58, and then the food columns go negative. Find the 好吃 "delicious" row: 甜 "sweet" 0.95, 香蕉 "banana" 0.72, 苹果 "apple" 0.61. Find the 很 "very" row: 真 "really" 0.98 and 这个 "this" 0.71.)*

Look at the 好吃 / 甜 / 香蕉 / 很 / 真 / 这个 / 做成 rows in the right half, and then the 手机 / 芯片 / 电脑 rows in the left half —
**those big numbers, above 0.9 and below −0.7, are what "structure" looks like when it grows.**

`experiment.py` puts before and after side by side:

```text
==============================================================
实验一：训练过程中的 loss
==============================================================
  轮数     loss
  --------------------
    40   2.4362  ██████████████████████████
    80   2.1062  ███████████████████████
   120   1.8164  ████████████████████
   160   1.5809  █████████████████
   200   1.4031  ███████████████
   240   1.2742  ██████████████
   280   1.1855  █████████████
   320   1.1298  ████████████
   360   1.0960  ████████████
   400   1.0748  ████████████

  最后一轮的 loss = 1.0748
  横条的长度是相对于「随机乱猜」的 loss = ln(16) ≈ 2.77 画的。
  loss 从 2.77 掉到 1 附近，说明模型确实在这 68 个样本上学到了东西。

==============================================================
实验二：组内相似度 - 组间相似度
==============================================================
                   组内平均   组间平均   组内 - 组间
  ----------------------------------------------------
  训练前（随机）    -0.045     +0.062     -0.107
  训练后            +0.246     -0.330     +0.576

  训练前这个差值在 0 附近（-0.107）；训练后是 +0.576。
  同一个任务、同一批词、同一张表的形状 —— 只把表里的数字训练了一遍。
```

*(The block holds two experiments. Experiment one is the loss during training, one line every 40 epochs, with a bar whose length is drawn relative to the loss of guessing at random, ln(16) ≈ 2.77 — it falls from 2.4362 to 1.0748. Experiment two is average similarity within a group minus average similarity between the two groups: before training (random) it is −0.045 within and +0.062 between, a difference of −0.107; after training it is +0.246 within and −0.330 between, a difference of +0.576.)*

Now the pictures. `experiment.py` draws the same table at two different moments:

```text
训练前（随机数字）：
┌──────────────────────────────────────────────────────┐
│                                                  发布│
│                                                      │
│                                                  真  │
│               香蕉                                   │
│                                                      │
│             好吃                                     │
│                               小米                   │
│                    芯片        新                    │
│电脑                                                  │
│                       苹果                           │
│                                      甜          这个│
│                                                手机  │
│                                                      │
│                                         派           │
│                                                      │
│                                                      │
│                    做成                              │
│                              很         华为         │
└──────────────────────────────────────────────────────┘

训练后（用「猜邻居」训练过）：
┌──────────────────────────────────────────────────────┐
│              香蕉                                    │
│               好吃                                 很│
│                    派                          真    │
│                  甜                          做成    │
│   苹果                                               │
│                                              这个    │
│                                                      │
│                                                      │
│                                                      │
│                                                      │
│                                                      │
│         小米                                         │
│    华为                                              │
│                                                      │
│                                                      │
│新                               芯片                 │
│                                                  发布│
│                                   电脑手机           │
└──────────────────────────────────────────────────────┘
```

*(The two ASCII plots are the 16 word vectors squeezed from 8 dimensions down to 2 and drawn on a character grid; each label is a Chinese word placed at that word's coordinates. Top plot, before training (random numbers): 香蕉 "banana" is sitting next to 发布 "release", while 华为 "Huawei" and 小米 "Xiaomi" are half a plot apart from each other — you cannot tell which words belong together. Bottom plot, after training (trained on "guess the neighbour"): the food words — 香蕉 banana, 好吃 delicious, 甜 sweet, 派 pie, 苹果 apple, 很 very, 真 really, 这个 this, 做成 make-into — form one band across the top, the tech words — 新 new, 小米 Xiaomi, 华为 Huawei, 芯片 chip, 电脑 computer, 手机 phone, 发布 release — form another band along the bottom, and 苹果 "apple" is caught in the middle.)*

In the first plot, 香蕉 and 发布 are neighbours and 华为 and 小米 are half a plot apart — you can't see which words belong together.
In the second plot, the food words form one band at the top and the tech words one band at the bottom, with 苹果 wedged in between.

(Both plots are drawn by squeezing 8 numbers down to 2; the squeezing code is in `to_2d` in `after.py`, four lines, using an off-the-shelf tool from linear algebra. Skip it if it doesn't make sense — the picture itself is what matters; the number table above it is the main evidence.)

## The new mechanism

This chapter's new thing is one sentence long:

> **Treat that vector table itself as a parameter, and train it with the "guess the neighbour" task.**

Concretely, the model looks like this. Given one training sample, "see `c`, guess `t`":

```python
v = 表[c]                        # 查表，拿到 c 的向量，形状 (8,)
logits = v @ 输出层 + 输出层偏置   # 得到 16 个分数，每个词一个
loss = cross_entropy(logits, t)  # 第 5 章那个交叉熵
```

*(Look up `c`'s vector in the table; multiply it by the output layer to get a score for each of the 16 words; compute the cross-entropy loss against the true neighbour `t`.)*

When `backward()` runs, the gradient travels all the way back into **that table** and changes the row `表[c]` a little.

Compare with Chapter 10 and it's clear:

| | Chapter 10 | Chapter 11 |
|---|---|---|
| That table | Scattered at random, read-only | A parameter, trained |
| Who is learning | Only `w` and `b` (17 numbers) | The table itself (16 × 8 = 128 numbers) |
| Task | Split the words into tech / food | See a word, guess the word beside it |

**Why does training this way pull similar words together?**

Take 手机 and 芯片. In our corpus they both appear after "新":

```text
苹果 发布 新 手机
苹果 发布 新 芯片
```

*(苹果 发布 新 手机 "Apple shipped a new phone" and 苹果 发布 新 芯片 "Apple shipped a new chip" — same shape, and 手机 and 芯片 occupy the same slot.)*

So the training samples contain both of these:

```text
看到 手机  →  猜 新
看到 芯片  →  猜 新
```

*(See 手机 "phone" → guess 新 "new"; see 芯片 "chip" → guess 新 "new".)*

Both samples demand that "the vector of this word, fed into the output layer, gives 新 the highest score".
What is the laziest way to satisfy both? **Make 手机's vector and 芯片's vector point in roughly the same direction.**
If they pointed in different directions, the output layer would have to find some way to map two different directions onto 新, and that costs more effort.

The model doesn't "think" any of this. It just pushes the loss down. But the result of pushing the loss down is that words which are able to stand in the same position end up looking more and more alike.

We never told it "手机 and 芯片 are one kind of thing". Not one sentence in the corpus says so.
**That "kind" is something it worked out for itself, from "who can stand in the same position".**

One detail: in Chapter 10 each word used 16 numbers, but this chapter uses only 8. Why?

Because in this chapter the numbers in the table have to **be trained**, and we have only 68 training samples and 16 words.
Give it too many numbers and the model has enough room to memorise all 68 samples as they are (memorising them is the same as learning nothing), and the structure won't grow at all. Give it too few numbers and there's nowhere to put the structure.

We tried all four settings (measuring "within-group similarity − between-group similarity"; bigger means clearer structure):

| Numbers per word | Within − between |
|---|---|
| 2 | −0.127 (can't hold it) |
| 4 | +0.130 |
| 8 | +0.576 (the one this chapter uses) |
| 16 | +0.452 (enough spare capacity to start rote-memorising) |

Both ends are bad; the middle is best. **More dimensions is never simply better** —
we'll run into this once more in Chapter 28 when we talk about scale, except that time the leading role goes to other numbers.

## Python implementation

The full code is in `after.py`, and the core is three functions.

**Step one: build the training samples.** Scan every sentence; at every position, take the words one place to either side as samples:

```python
def build_training_pairs(corpus, window=WINDOW):
    seen, guessed = [], []
    for sentence in corpus:
        for position in range(len(sentence)):
            left = max(0, position - window)
            right = min(len(sentence), position + window + 1)
            for neighbour in range(left, right):
                if neighbour == position:
                    continue
                seen.append(sentence[neighbour])
                guessed.append(sentence[position])
    return np.array(seen), np.array(guessed)
```

13 sentences, window of 1, giving 68 samples in total.

**Step two: train.** Note that there is no separate `w` here — the thing to be learned is the table itself:

```python
table = randn(len(VOCAB), DIM, scale=0.1, requires_grad=True, seed=seed)
out_weight = randn(DIM, len(VOCAB), scale=0.1, requires_grad=True, seed=OUT_SEED)
out_bias = zeros(len(VOCAB), requires_grad=True)
optimizer = SGD([table, out_weight, out_bias], lr=0.3)

for epoch in range(400):
    optimizer.zero_grad()
    seen_vectors = embedding(table, seen)          # (68, 8)
    logits = seen_vectors @ out_weight + out_bias  # (68, 16)
    loss = cross_entropy(logits, guessed)
    loss.backward()
    optimizer.step()
```

`embedding(table, seen)` is the same table lookup as in Chapter 9:
`seen` is 68 IDs, and after the lookup they become 68 rows of vectors.
During backpropagation the gradients scatter back, by ID, onto the rows of the table they came from.

**Step three: look at the results.** Make every row vector unit length; then the dot product of any two of them is their cosine similarity:

```python
def normalise(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

def similarity_matrix(table_data):
    unit = normalise(table_data)
    return unit @ unit.T
```

Then sort and print each word's neighbours. Run these three steps and you get the output in the "Experiment" section above.

(There is also an ASCII scatter plot, with the code in `scatter` and `to_2d`. It squeezes 8 dimensions into 2 and draws them onto a grid of characters, lining things up by the rule that "a Chinese character takes two cells". It's there to give you the gist; it isn't what this chapter is teaching.)

## What it solves

The question Chapter 10 left behind gets its answer here: **that group of numbers is trained, not scattered.**

Concretely, what we got:

1. **Similar words really did move closer.** The similarity between 手机 and 芯片 is 0.99; between 好吃 and 甜 it is 0.95.
   In Chapter 10's random table, the word most like 手机 was 派.
2. **The structure is measured, not eyeballed.** "Average similarity within a group − average similarity between groups" went from −0.107 (before training) to +0.576 (after). Same task, same batch of words, same table shape — only the numbers inside the table were trained.
3. **Nobody had to label anything.** We gave it 13 sentences. The model received not one piece of knowledge of the form "手机 and 芯片 are one kind" — it counted who each word stood next to, and worked it out by itself.
4. **"Meaning can become spatial structure" is in our hands for the first time.**
   Words became points in a space; the relations between words became distances and directions between points.
   This matters more than any single number in this chapter — every road after this one starts here.

Back to the sentence from Chapter 1: 苹果 appears in both tech sentences and food sentences.
What did this chapter do about it? Nothing.
It appears on both sides in the corpus, so it gets pulled from both sides in the space, and it ends up parked between the two clumps. Look at its neighbours:

```text
  苹果   → 香蕉 0.82  华为 0.61  好吃 0.61  小米 0.61  甜 0.56
```

*(苹果 "apple"'s five nearest words after training: three from the food side — 香蕉 "banana" 0.82, 好吃 "delicious" 0.61, 甜 "sweet" 0.56 — and two from the tech side — 华为 "Huawei" 0.61, 小米 "Xiaomi" 0.61.)*

香蕉, 好吃 and 甜 are on the food side; 华为 and 小米 are on the tech side.
All of them count as neighbours of 苹果. That result is perfectly reasonable —
but it leads straight to a nastier question.

## What it still can't solve

Notice one thing about that line of numbers: **苹果 has exactly one vector.**

But our corpus contains both kinds of sentence:

```text
苹果 很 好吃          （这里说的是水果）
苹果 发布 新 手机      （这里说的是公司）
```

*(苹果 很 好吃 "apples are delicious" — here it means the fruit; 苹果 发布 新 手机 "Apple shipped a new phone" — here it means the company.)*

The 苹果 in the first sentence and the 苹果 in the second are, as far as the model is concerned, **exactly the same thing**.
Not "more or less the same" — exactly the same: same ID, same row, same 8 numbers, cosine similarity 1.0000.

You can check this yourself: take the 苹果 out of both sentences and compute how much their vectors differ. The answer is: not one number differs.

So the model has no way at all to express "these two 苹果 are not the same thing".
It cannot even express "the 苹果 I'm talking about now means something different from the 苹果 in the last sentence".

Let's drive it into a corner:

> In the corpus, the 苹果 in 「苹果 很 好吃」 and the 苹果 in 「苹果 发布 新 手机」
> are plainly two different things.
> Yet the model gives them **completely identical** representations — so whatever we want to do downstream,
> if it requires telling these two 苹果 apart, it is doomed to get it wrong.

What we need is not "train longer", and not "turn the dimension up or down".
What we need is **a word's representation that changes with the sentence it is in**:

The same 苹果 looks one way next to "很 好吃" and another way next to "发布 新 手机".

So what raw material do we already have in hand? The vectors of the words around it.
**How should those vectors be used, so that we can compute "what does the 苹果 at this position mean"?**

## Exercises

See `exercises.en.md`. Here are the 5 most important ones:

1. Change `WINDOW` in `after.py` to 2 and 3 and run again. The training samples grow from 68 to 110 and 126, and the loss actually rises from 1.07 to 1.63 and 1.89 — but "within − between" also rises from +0.58 to +1.05 and +1.27. Are those two things in conflict?
2. Change `DIM` to 2, 4, 16 and run each, and see how the "within − between" gap moves. Why are both ends bad?
3. Add three tech sentences of the form "苹果 发布 新 芯片" to the corpus and train again. See how 苹果's neighbours change — which way did it shift, and by how much?
4. In the corpus, 华为 and 小米 appear only in the one pattern "发布 新 X". Pull out their two rows and compare: are they almost identical (0.99)? Does that show "the model can't learn a distinction the corpus doesn't make", or does it show something else?
5. Change the training samples to collect only one direction (use only the word on the left to guess the word on the right). The samples drop from 68 to 34. The loss falls from 1.07 to 0.69 — yet the word most like 好吃 becomes 芯片. Why does halving the samples make that much difference?
