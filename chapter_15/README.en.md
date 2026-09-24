**English** | [中文](README.md)

# Chapter 15: Why we need more than one head

## The problem in this chapter

Last chapter ended with this line:

> A word gets only one row of weights, summing to 1. That row is a fixed budget.

This chapter we look at that through a different sentence:

```text
小王 把 书 给了 小李 因为 他 明天 考试
```

*(Nine words: `小王` Xiao Wang (a person's name) · `把` the *bǎ* particle, marking the
thing being acted on · `书` book · `给了` gave (to) · `小李` Xiao Li (another person) ·
`因为` because · `他` he / him · `明天` tomorrow · `考试` exam / to sit an exam.
"Xiao Wang gave the book to Xiao Li because he has an exam tomorrow." The word we're
going to be staring at is `他`, "he" — and the question is who "he" is.)*

On the single character `他` ("he"), several things hang at once.

Run `before.py` first, and see what Chapter 14's method produces here:

```text
第 14 章的做法：'他'这个字拿到的权重（只有这一行）
------------------------------------------------------------
  小王  0.1157  #####
  把    0.0890  ####
  书    0.1058  ####
  给了  0.1173  #####
  小李  0.1098  ####
  因为  0.0937  ####
  他    0.1065  ####  <- 自己
  明天  0.1326  #####
  考试  0.1295  #####
  权重和 = 1.0000
  （投影是随机初始化的，所以具体数字没意义；有意义的是：它只有一行。）
```

*(`第 14 章的做法：'他'这个字拿到的权重（只有这一行）` — "Chapter 14's method: the weights
the character 他 gets (this one row is all there is)". Top to bottom the rows are 小王
Xiao Wang, 把 (particle), 书 book, 给了 gave, 小李 Xiao Li, 因为 because, 他 he (marked
`<- 自己` "itself"), 明天 tomorrow, 考试 exam. `权重和 = 1.0000` — "the weights sum to
1.0000". The closing note: `投影是随机初始化的，所以具体数字没意义；有意义的是：它只有一行。`
— "the projections are randomly initialized, so the specific numbers are meaningless;
what is meaningful is that there is only one row.")*

**Only one row.** And this one row has to answer both "is `他` 小王 or 小李?" and
"`他` has an exam tomorrow".

Write those two needs down as two sets of ideal weights (this is only to get the conflict
onto paper):

```text
  需求 A（指代）：'他'是小王还是小李？
    理想的权重应该是：小王 0.5，小李 0.5

  需求 B（时间）：'他'明天要考试
    理想的权重应该是：明天 0.5，考试 0.5

  需求 A 要花掉的权重：1.0
  需求 B 要花掉的权重：1.0
  两个加起来：2.0
  一行权重一共只有：1.0
  → 超出预算 1.0
```

*(`需求 A（指代）：'他'是小王还是小李？` — "need A (reference): is 他 小王 or 小李?" `理想的
权重应该是：小王 0.5，小李 0.5` — "the ideal weights would be: 小王 0.5, 小李 0.5".
`需求 B（时间）：'他'明天要考试` — "need B (time): he has an exam tomorrow"; `理想的权重应该
是：明天 0.5，考试 0.5`. Then the arithmetic: `需求 A 要花掉的权重：1.0` — "the weight need A
would spend: 1.0"; same for B; `两个加起来：2.0` — "the two together: 2.0"; `一行权重一共
只有：1.0` — "one row of weights has only: 1.0"; `→ 超出预算 1.0` — "→ over budget by 1.0".)*

One row of weights is one dollar; the two needs come to two dollars. This is not "not
trained well enough" — this is **the arithmetic doesn't work**. The best a single head
can do is to have each side give a little ground:

```text
  一行权重能做到的最好情况，是两边各让一步：
  小王  0.25  小李  0.25  明天  0.25  考试  0.25

  离需求 A 还差：0.50（有 50% 的权重放错了地方）
  离需求 B 还差：0.50
  两个需求都只做到一半。
```

*(`一行权重能做到的最好情况，是两边各让一步` — "the best one row of weights can do is for
each side to give some ground": 小王 0.25, 小李 0.25, 明天 0.25, 考试 0.25. Then `离需求 A
还差：0.50` — "short of need A by 0.50", with the note `（有 50% 的权重放错了地方）` —
"50% of the weight is in the wrong place"; `离需求 B 还差：0.50`; and `两个需求都只做到一半`
— "both needs are only half met".)*

**Here's the problem: a sentence holds more than one kind of relation, and one row of
weights isn't enough to divide up between them.**

## The simplest attempt

The fix is direct: **if one row isn't enough, use several rows.**

In Chapter 14 we learned to build one set of Query / Key / Value for a word. Now build
several sets — each set with a **projection matrix of its own**:

```python
第 1 组：q1 = x @ W_q1    k1 = x @ W_k1    v1 = x @ W_v1
第 2 组：q2 = x @ W_q2    k2 = x @ W_k2    v2 = x @ W_v2
第 3 组：q3 = x @ W_q3    k3 = x @ W_k3    v3 = x @ W_v3
```

*(`第 1 组` — "group 1"; the same for groups 2 and 3, each with its own `W_q`, `W_k`,
`W_v`.)*

Each group computes its own set of weights and mixes the value in its own way:

```python
第 1 组的输出 = softmax(q1 k1ᵀ / √d) v1
第 2 组的输出 = softmax(q2 k2ᵀ / √d) v2
```

Then **concatenate** those outputs:

```python
输出 = 拼起来(第1组输出, 第2组输出, 第3组输出) @ W_o
```

*(`第 1 组的输出` — "group 1's output"; `拼起来` — "concatenate".)*

Each group is called a **head**. This scheme we call
**multi-head attention**.

Note one detail: **each head's projection matrices are different.** If several heads
shared one set of `W_q`/`W_k`/`W_v`, they would compute identical weights and having more
of them would buy nothing. That's what the "multi" in the name means — not computing the
same thing several times, but **looking at the same sentence in several different ways
at once**.

One more detail: each head's dimension should be smaller. Chapter 14 used a full 24
dimensions for one head; now 4 heads take 6 dimensions each. The total parameter count is
basically unchanged — the same money, cut into 4 parts.

## Experiment

Run `experiment.py` (control group: 1 head vs 4 heads, same task, same parameter count).

```text
      步数          1 个头          4 个头
  ----------------------------------
       0       11.5659        8.2376
      50        0.2932        0.0672
     100        0.1924        0.0176
     200        0.0176        0.0058
     400        0.0044        0.0022

  两边最后都把这两句话背下来了（数据只有两句，这是必然的）。
  差别不在损失，在于每个位置能拿出几套权重。
```

*(Columns: `步数` "step", `1 个头` "1 head", `4 个头` "4 heads". The closing lines: `两边最后
都把这两句话背下来了（数据只有两句，这是必然的）` — "both sides end up having memorized the two
sentences (with only two sentences of data, that was bound to happen)"; `差别不在损失，在于
每个位置能拿出几套权重` — "the difference isn't in the loss, it's in how many sets of weights
each position can bring out".)*

The loss can't tell the difference — two sentences is far too little data, and one head
can memorize them too. What you actually need to look at is the `他` row:

```text
【1 个头】只有一行：
    小王  0.0064
    把    0.0007
    书    0.0007
    给了  0.0671  ###
    小李  0.0019
    因为  0.0012
    他    0.0019
    明天  0.0679  ###
    考试  0.8523  ##################################

【4 个头】四行，各看各的：
  头 0：最大值 0.675   前三个：明天(0.675)、考试(0.134)、把(0.054)
  头 1：最大值 0.484   前三个：书(0.484)、因为(0.149)、把(0.079)
  头 2：最大值 0.263   前三个：明天(0.263)、因为(0.163)、把(0.161)
  头 3：最大值 0.482   前三个：考试(0.482)、给了(0.434)、小王(0.043)
```

*(`【1 个头】只有一行` — "[1 head] only one row": 小王 0.0064, 把 0.0007, 书 0.0007,
给了 0.0671, 小李 0.0019, 因为 0.0012, 他 0.0019, 明天 0.0679, 考试 0.8523.

`【4 个头】四行，各看各的` — "[4 heads] four rows, each looking at its own things".
Then `头 0：最大值 0.675   前三个：明天(0.675)、考试(0.134)、把(0.054)` — "head 0:
maximum 0.675, top three: 明天 tomorrow (0.675), 考试 exam (0.134), 把 particle (0.054)".
Head 1: max 0.484, top three 书 book (0.484), 因为 because (0.149), 把 (0.079).
Head 2: max 0.263, top three 明天 (0.263), 因为 (0.163), 把 (0.161).
Head 3: max 0.482, top three 考试 (0.482), 给了 gave (0.434), 小王 (0.043).)*

Put the four rows side by side:

```text
  1 个头：只有一行，0.852 压在'考试'上，剩下 0.148 留给 8 个词。
           它想再表达点别的（比如'他 = 小王'），就只能从这 0.852 里往外抠。

  4 个头：每一行各自挑自己的词——
           头 0 最关注 明天（0.675）
           头 1 最关注 书（0.484）
           头 2 最关注 明天（0.263）
           头 3 最关注 考试（0.482）

  注意 4 个头并没有比 1 个头'更尖'——它们做的事情是：把一份注意力拆成 4 份，
  各挑各的词。1 个头押了'考试'就押不了'书'；4 个头里，头 1 押'书'，头 3 押'考试'和'给了'。
  同样的参数预算，能同时表达的东西多了几倍。
```

*(`1 个头：只有一行，0.852 压在'考试'上，剩下 0.148 留给 8 个词` — "1 head: only one row,
0.852 piled on 考试 ('exam'), 0.148 left over for the other 8 words"; `它想再表达点别的
（比如'他 = 小王'），就只能从这 0.852 里往外抠` — "if it wants to express anything else
(say 'he = Xiao Wang'), it can only dig it out of that 0.852". Then, for 4 heads, each row
picks its own words: head 0 most attends to 明天 (0.675), head 1 to 书 (0.484), head 2 to
明天 (0.263), head 3 to 考试 (0.482). The closing lines: `注意 4 个头并没有比 1 个头'更尖'` —
"note that 4 heads are not 'sharper' than 1 head" — `它们做的事情是：把一份注意力拆成 4 份，
各挑各的词` "what they do is split one portion of attention into 4, each picking its own
words"; `1 个头押了'考试'就押不了'书'` "if 1 head bets on 考试 it can't also bet on 书";
`同样的参数预算，能同时表达的东西多了几倍` "for the same parameter budget, the number of
things that can be expressed at once goes up several fold".)*

And one thing has to be said honestly: **we cannot put labels on these heads.** With only
two sentences of data, what any head learns is half luck. Change `seed` to 7 and run it
again, and the four heads attend to different things. What we can say for certain is:
**at the same position, 4 heads produce 4 different sets of weights.**

## The new mechanism

Multi-head attention is just Chapter 14's scheme copied out h times, then concatenated:

```python
for 每个头 h:
    q_h = x @ W_q[h]                          # 我在找什么（第 h 种问法）
    k_h = x @ W_k[h]                          # 我有什么（第 h 种标签）
    v_h = x @ W_v[h]                          # 我能给你什么（第 h 份信息）
    头_h = softmax(q_h k_hᵀ / √d_h) v_h        # 第 h 种看法的结果

输出 = 拼起来(头_1, 头_2, ..., 头_h) @ W_o
```

*(The comments: `我在找什么（第 h 种问法）` "what I'm looking for (the h-th way of asking)",
`我有什么（第 h 种标签）` "what I have (the h-th set of labels)", `我能给你什么（第 h 份信息）`
"what I can give you (the h-th portion of information)", `第 h 种看法的结果` "the result of
the h-th way of looking".)*

Written in one line:

```text
MultiHead(Q, K, V) = Concat(头_1, ..., 头_h) W_o
   其中  头_i = Attention(Q W_q[i], K W_k[i], V W_v[i])
```

*(`头_i` — "head i".)*

Three new things, each with its own reason:

- **Each head has its own projection**: this is why a "way of looking" can be expressed
  separately.
- **Each head is smaller (d_h = d / h)**: the total parameter count is basically unchanged.
  4 heads don't multiply the cost by 4; they spend the same money in 4 portions.
- **Concatenate and then project once more (W_o)**: several heads each mind their own
  business, and in the end their information has to be mixed together, and the output has
  to get back to the original dimension so the layers behind can connect to it.

The o in `W_o` stands for output; the q, k, v in `W_q`/`W_k`/`W_v` are last chapter's
Query / Key / Value.

## Python implementation

That's `after.py`. The core is `multi_head_attention`:

```python
def multi_head_attention(vectors, params):
    head_dim = params["w_q"].shape[-1]

    # 三个投影一次算完所有头：矩阵乘法会自动在"头"这一维上广播
    query = vectors @ params["w_q"]          # (头数, 词数, HEAD_DIM)
    key = vectors @ params["w_k"]
    value = vectors @ params["w_v"]

    # 每个头自己算一套分数、一套权重
    scores = (query @ key.transpose(0, 2, 1)) / np.sqrt(head_dim)
    weights = scores.softmax(axis=-1)        # (头数, 词数, 词数)

    # 每个头自己混一遍 value
    head_output = weights @ value            # (头数, 词数, HEAD_DIM)

    # 把几个头拼起来
    joined = head_output.transpose(1, 0, 2).reshape(num_words, DIM)
    return joined @ params["w_o"], weights
```

*(The comments: `三个投影一次算完所有头：矩阵乘法会自动在"头"这一维上广播` — "all heads are
projected in one go: matrix multiplication broadcasts along the 'head' dimension
automatically"; `每个头自己算一套分数、一套权重` — "each head computes its own scores and its
own weights"; `每个头自己混一遍 value` — "each head mixes the value in its own way"; `把几个头
拼起来` — "concatenate the heads".)*

Three shapes to keep your eye on:

- `params["w_q"]` has shape `(number of heads, dimension, per-head dimension)` — that is,
  **4 matrices of 24×6 stacked together**.
- `weights` has shape `(number of heads, number of words, number of words)`. `weights[0]`
  is head 0's attention matrix — it looks just like Chapter 14's table.
- That final `transpose(1, 0, 2).reshape(...)` is the "concatenate the 4 heads" step:
  from `(4, words, 6)` to `(words, 4, 6)` and then flattened to `(words, 24)`.

Run `after.py`; it trains for 400 steps and then prints each head's attention matrix. In
the heat characters `.` is the lightest and `@` the heaviest (each head is drawn against
its own maximum — read the shapes).

```text
  【头 0】（本头的最大值 = 0.675）
          小王把  书  给了小李因为他  明天考试
    小王  ............::::............----****
    把    ....----........::::========::::....
    书    ....----............++++++++........
    给了  ::::::::........----::::====........
    小李  ............----............::::%%%%
    因为  ....::::........::::----::::----....
    他    ............................@@@@::::
    明天  ............++++................####
    考试  ----........++++::::............----

  【头 1】（本头的最大值 = 0.790）
          小王把  书  给了小李因为他  明天考试
    小王  ....::::----........::::::::........
    把    ............................%%%%::::
    书    ....::::----====................::::
    给了  ::::........====................::::
    小李  ....::::****........::::............
    因为  ............................@@@@::::
    他    ........++++........::::............
    明天  ........++++........::::....----....
    考试  ....::::----::::....::::............
```

*(`【头 0】（本头的最大值 = 0.675）` — "Head 0 (this head's maximum = 0.675)", and the same
for head 1 (0.790). **How to read this:** the rows, top to bottom, are 小王, 把, 书, 给了,
小李, 因为, 他, 明天, 考试 — the nine words of the sentence, in order. The squeezed line
across the top, `小王把  书  给了小李因为他  明天考试`, is the same nine words as column
labels (they're crammed together because Chinese characters are double-width in a
terminal; the sentence is "Xiao Wang gave the book to Xiao Li because he has an exam
tomorrow"). So the row labelled `他` is "he" doing the looking, and the cells across it
show where "he" looks. In head 0 that whole row is `@` — piled on one column; in head 1
the same row is a different shape. The nine columns, left to right, are 小王 Xiao Wang ·
把 (particle) · 书 book · 给了 gave · 小李 Xiao Li · 因为 because · 他 he · 明天 tomorrow ·
考试 exam.)*

Put the four tables side by side and it's obvious at a glance that they differ: in head 0
the `他` row lies entirely on `明天` ("tomorrow"), in head 1 the `他` row lies on `书`
("book"). This is **one position, four ways of looking**.

The full output also has head 2's and head 3's matrices. Here are the exact numbers for
the `他` row:

```text
同一个位置，4 个头给出 4 套不同的权重
============================================================
  头 0：明天(0.675)、考试(0.134)、把(0.054)
  头 1：书(0.484)、因为(0.149)、把(0.079)
  头 2：明天(0.263)、因为(0.163)、把(0.161)
  头 3：考试(0.482)、给了(0.434)、小王(0.043)

  如果只有一个头，'他'就只能有一套权重——上面这 4 种看的方式，只能留一种。
```

*(`同一个位置，4 个头给出 4 套不同的权重` — "the same position, 4 heads giving 4 different sets
of weights". Head 0: 明天 0.675, 考试 0.134, 把 0.054. Head 1: 书 0.484, 因为 0.149, 把 0.079.
Head 2: 明天 0.263, 因为 0.163, 把 0.161. Head 3: 考试 0.482, 给了 0.434, 小王 0.043. The
closing line: `如果只有一个头，'他'就只能有一套权重——上面这 4 种看的方式，只能留一种。` —
"with only one head, 他 could have only one set of weights — of these 4 ways of looking,
only one could be kept.")*

## What it solves

1. **One position can have several sets of weights at once.** The 4 heads' answers for
   `他` are `明天(0.675)`, `书(0.484)`, `明天(0.263)`, `考试(0.482)+给了(0.434)` — four
   completely different distributions, existing at the same time.
2. **The budget is no longer one person's problem.** `before.py` works it out clearly:
   the two needs come to 2.0, but one row of weights has only 1.0, and the compromise
   leaves both sides 0.50 short. Once it's split into two heads, each head has its own
   budget of 1.0, and both can get to 0.00.
3. **The cost didn't grow.** 4 heads of 6 dimensions each still add up to 24 dimensions.
   The loss curves for 1 head vs 4 heads (step 50: `0.2932` vs `0.0672`; step 400:
   `0.0044` vs `0.0022`) show that this wasn't bought by piling on parameters.

## What it still can't solve

Now read this chapter's code from top to bottom and ask one question:

**From the word vectors to the output, which step uses "what position this word is in"?**

- `vectors @ W_q`: each word projects itself, with no relation to position.
- `scores = q kᵀ / √d`: the score for any two words depends only on those two words, and
  not at all on how far apart they are.
- `weights @ value`: a weighted sum. Summing doesn't care about order.

**Not one step.** What this mechanism sees is not a "sentence" but a bag of words — it
knows which words there are and who is close to whom, and it has no idea at all about who
comes first.

The consequence is concrete. These two sentences:

```text
狗 咬 人
人 咬 狗
```

*(`狗` dog · `咬` bites · `人` person. So "dog bites man" and "man bites dog".)*

use the same three words. In both sentences `狗` ("dog") will look at `咬` ("bites") and
`人` ("person"), and `咬` will look at `狗` and `人` too — because in this mechanism, the
score for any pair of words depends only on which two words they are.

**So what exactly is the difference between "dog bites man" and "man bites dog"? How is
the model supposed to know?**

Next chapter we'll actually run this experiment: shuffle the order of the input words and
see whether the attention output changes.

## Exercises

See `exercises.en.md`. The most important ones are listed here:

1. **Keep only one head** — set `NUM_HEADS` to 1 and see how many rows of weights `他`
   has left, and whether the loss can still come down.
2. **Make several heads share one set of Q/K/V** — set all three projections to
   `params["w_q"][0:1]` and see whether the 4 heads' weight matrices are still different.
3. **Open more heads** — go to 8 or 12; then try 5 (and think about where it leaves you
   after `24 // 5 = 4` — where did the remaining 4 dimensions go).
4. **Change the seed** — `train(steps=400, seed=7)`, and see whether the four heads are
   looking at the same words every time. The point of this exercise is to confirm for
   yourself: with this little data, **don't put labels on the heads**.
5. **Replace concatenation with addition** — `head_output.sum(axis=0)`, see whether the
   shapes still line up, then think about which of concatenation and addition keeps more
   information.
