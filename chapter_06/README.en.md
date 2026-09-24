**English** | [中文](README.md)

# Chapter 6: How the machine knows which way to change a parameter

## The problem in this chapter

In the last chapter we got one thing done: give the model a sentence and it outputs two
probabilities, and we use cross-entropy to squash that into a single number — the loss.

Now stare at that number and ask a very practical question:

**The loss is big. Which parameter do I change?**

Our parameter table has 32 numbers in it (16 words × 2 categories). The loss tells us only
what the "total score" is; it does not tell us where any of the individual points came from.
It's like an exam that hands back only the total and not the paper — you know you did badly,
but you don't know which chapter to review.

Chapter 5's approach was to try things one at a time: nudge some parameter a little,
recompute the loss. `before.py` ran that approach from beginning to end, and here is what it
concluded:

```text
       每次挪多少        达到损失 < 0.001 要几轮         用掉的前向次数
       0.005        200 轮还没到（0.0396）           13001
        0.01        200 轮还没到（0.0052）           13001
        0.05                      57            3706
         0.5                       6             391
         5.0                       1              66
```
*(Columns: how much we nudge each time; how many rounds it takes to get loss < 0.001; how
many forward passes that used up.)*

Same approach, same model, and the only thing that changes is "how much we nudge each time"
— and the result goes from "200 rounds and still not there" to "done in a single round",
a difference of several hundred times the work. And there is no basis at all, beforehand,
for what number to put there.

## The simplest attempt

Take the "try a change" approach as far as it will go, and you have `before.py`.

Its logic is so simple it needs no explanation: add a little to some parameter and compute
the loss; if the loss got smaller, keep the change, if it got bigger, try subtracting a
little instead. Do that for all 32 parameters and you have completed one round. It really
does run, and it really does push the loss down — but it has three flaws you cannot get
around.

**One: how far to nudge is a guess.**

That's the table above. Nudging by 0.005 gets nowhere after two hundred rounds; 5.0 lands in
one. We didn't "find a good method" — we just tried all five values. Change the model, change
the data, and that number has to be tried all over again.

**Two: it can only move one parameter at a time.**

While we nudge `发布` ("release"), the other 31 parameters sit there perfectly still,
waiting. And yet they are obviously related — `手机` ("phone") and `新` ("new") have to pull
together for a sentence like `苹果发布新手机` ("Apple shipped a new phone") to be pushed over
to the technology side.

So what about moving all 32 parameters at once? Each parameter has two options, up or down,
and together that's 2³² ≈ 4.3 billion combinations. Try them one at a time and you'll still
be going at the end of the universe.

**Three: the number of runs it needs is twice the number of parameters.**

We only have 32 parameters, 64 runs per round, and that's survivable. A million parameters?
Two million runs per round. The biggest models today have hundreds of billions of parameters
— this road goes nowhere.

The three are really the same thing:

> Right now we can only ask "this one came out wrong".
> We can't ask "**which way should each parameter move, and how far**".

That is exactly what the end of Chapter 5 was saying.

## Experiment

Run `experiment.py`. We'll take the problem apart in six steps.

### Step 1: Fix your eyes on one parameter

```text
               「发布」的科技权重          损失       比 0 的时候
                   -0.05    0.703272     +0.010125
                   -0.01    0.695152     +0.002005
                    0.00    0.693147     +0.000000
                    0.01    0.691152     -0.001995
                    0.05    0.683272     -0.009875
```
*(For `发布` — "release" — the technology weight, the loss, and the change relative to the
value at 0.)*

It's clear enough: move this parameter in the positive direction and the loss gets smaller.
"Which way to change it" has an answer, and that answer can be measured — not just "better or
worse" but "better by how much".

### Step 2: Compute all 32 directions at once

Trying one at a time is too slow, but the act of "nudge a little and watch what happens" can
itself be written as a formula:

```text
(损失(参数 + h) - 损失(参数 - h)) / (2h)
```
*(loss(parameter + h) - loss(parameter - h)) / (2h)*

Nudge once in each direction and subtract; the first-order errors from the two sides cancel
each other out, which is far more accurate than nudging in one direction only. Run that over
all 32 parameters:

```text
  词              科技类的梯度        食品类的梯度   出现在几句科技句 / 食品句
  苹果           0.05000      -0.05000        3 / 4
  发布          -0.20000       0.20000        4 / 0
  新            -0.20000       0.20000        4 / 0
  手机          -0.10000       0.10000        2 / 0
  芯片          -0.10000       0.10000        2 / 0
  电脑          -0.05000       0.05000        1 / 0
  华为          -0.05000       0.05000        1 / 0
  小米          -0.05000       0.05000        1 / 0
  好吃           0.10000      -0.10000        0 / 2
  很             0.10000      -0.10000        1 / 3
  甜             0.10000      -0.10000        0 / 2
  香蕉           0.05000      -0.05000        0 / 1
  这个           0.05000      -0.05000        0 / 1
  真             0.05000      -0.05000        0 / 1
  做成           0.05000      -0.05000        0 / 1
  派             0.05000      -0.05000        0 / 1
```
*(Columns: word; gradient for the technology class; gradient for the food class; how many
technology sentences / food sentences it appears in. `苹果` is "apple", `好吃` is
"delicious", `甜` is "sweet", `电脑` is "computer", `香蕉` is "banana", `做成派` is "make
into a pie".)*

You can read this table like a report:

- `发布` is -0.20, the largest negative number → it should be pushed up.
- `好吃` and `甜` are +0.10 → they should be pushed down.
- The last column explains the magnitudes: `发布` appears in 4 technology sentences and 0
  food sentences, so its gradient is the largest; `电脑` appears only once, so its gradient
  is only 0.05.

### Step 3: Does that direction agree with the one we found by trying, one at a time?

```text
  把「发布」的科技权重往右挪 0.001：损失 0.693147 -> 0.692947
  把「发布」的科技权重往左挪 0.001：损失 0.693147 -> 0.693347

  往右挪损失变小了 —— 所以要往右改，也就是给这个参数加一点点。
  数值法给出的梯度 = -0.20000
```
*(Move `发布`'s technology weight right by 0.001: loss 0.693147 -> 0.692947. Move it left by
0.001: loss 0.693147 -> 0.693347. Moving right made the loss smaller — so we change it to the
right, that is, we add a little to this parameter. The numeric method gives
gradient = -0.20000.)*

The gradient is negative, so we change it to the right. The gradient is negative, and the
formula has a **minus sign** in it. Negative times negative is positive — that is why the
update formula uses a minus and not a plus.

### Step 4: The numeric method vs the hand-derived formula

The numeric method is very solid, but it has one hard defect: every gradient it computes
requires running the whole model twice. 32 parameters, 64 forward passes per round.

So for this specific model we derived the gradient by hand. The result is a little
surprisingly simple:

```text
  词               数值法          公式           差
  苹果        0.048482    0.048482    1.91e-10
  发布       -0.152213   -0.152213    3.75e-09
  新         -0.152213   -0.152213    3.75e-09
  ...
  32 个参数里，最大的差是 3.75e-09。
```
*(Columns: word; numeric method; formula; difference. The last line: among the 32 parameters
the largest difference is 3.75e-09.)*

The two sides are the same thing. The only difference is that the numeric method needs 64
forward passes and the formula needs 1.

### Step 5: Spend the same compute, see how far each one gets

```text
  方法            步数      前向次数         最后的损失     正确率
  数值法           30      1920      0.105704      10/10
  公式法          960      1920    0.00370599      10/10
```
*(Columns: method; steps; forward passes; final loss; accuracy.)*

Same 1920 forward passes, and the formula method pushed the loss down another 28 times over.
And the gap only widens — because every step of the numeric method recomputes the model 64
times, while the formula method needs 2.

Note that the two rows use exactly the same update rule, `w = w - learning rate × gradient`.
The only difference is how the gradient was computed.

### Step 6: How large should the learning rate be

```text
         学习率         200 步后的损失       正确率         参数最大绝对值
      0.0005          0.663841         9/10            0.02
        0.05          0.147115        10/10            0.63
         0.5          0.017895        10/10            1.96
         5.0          0.001740        10/10            3.43
        20.0          0.000284        10/10            4.50
      1000.0          0.000000        10/10          200.00
```
*(Columns: learning rate; loss after 200 steps; accuracy; largest absolute parameter value.)*

0.0005 is too small: the 200 steps are used up and the loss is still 0.66, with one sentence
still misclassified.

That 0.000000 in the 1000.0 row is not "learned perfectly". The probabilities have been
pushed to 1.0, and `-log(1)` is exactly 0 — the ruler has hit its end and can't measure any
difference. Look at the last column: the parameters have grown to 200.

Why doesn't a large learning rate "blow up" the way it does elsewhere? Because our 10
sentences were separable to begin with: the larger the parameters, the more confident the
model, and the smaller the loss. So there is no "lowest point" for it to settle into; it just
keeps going up. **A loss that keeps falling is not always good news.**

We pick 0.5. That number has no formula behind it; it can only be tried.

## The new mechanism

### Gradient

The thing in that Step 2 table gets a name: **gradient**.

> A gradient is a table with one entry per parameter. Every number in it tells you two
> things: whether nudging this parameter in the positive direction makes the loss larger or
> smaller, and how fast it changes.

With that, the question "which parameter do I change" has a complete answer:
**every parameter knows which way it should move, and how hard it should be pushed.**

The numeric method and the formula compute the same gradient. The numeric method works it out
by "trying", the formula works it out by "deriving". The first is general but slow; the second
is fast but has to be derived once for each model.

### Learning rate

The gradient says only the direction, not how far to go. How far we go is up to us, and that
number is the **learning rate**.

And so the update rule from Chapter 4 grows into its final shape:

```text
w = w - 学习率 × 梯度
```
*(w = w - learning rate × gradient)*

Why the minus? Because the gradient points in the direction where the loss **rises** fastest.
We want to go down, so we go the other way.

That line, `w = w - lr * grad`, is the entire meaning of the word "learning" in code.

## Python implementation

The two most important pieces of `after.py`.

First the numeric method, which is where every idea in this chapter comes from:

```python
def numeric_gradient(table, corpus, h=1e-3):
    """数值法求梯度：把每个参数单独挪一点点，看损失怎么变。

        (损失(参数 + h) - 损失(参数 - h)) / (2h)
    """
    for word in VOCABULARY:
        for category_index in (0, 1):
            original = table[word][category_index]

            table[word][category_index] = original + h
            plus = average_loss(corpus)
            table[word][category_index] = original - h
            minus = average_loss(corpus)
            table[word][category_index] = original

            result[word][category_index] = (plus - minus) / (2 * h)
```
*(The docstring reads: "the numeric gradient — move each parameter a little on its own and
watch how the loss changes — (loss(p + h) - loss(p - h)) / (2h)".)*

Then we derive the whole chain "add up the word weights + softmax + cross-entropy" in one go,
which gives the formula version:

```python
def gradient(table, corpus):
    """每个样本贡献的梯度 = 预测概率 - 正确答案。"""
    for sentence, label in corpus:
        words = split_sentence(sentence)
        probabilities = softmax(score(words))
        correct_index = CATEGORIES.index(label)

        for category_index in (0, 1):
            difference = probabilities[category_index] - (1.0 或者 0.0)
            difference /= n
            for word in words:
                if word in result:
                    result[word][category_index] += difference
```
*(The docstring reads: "the gradient contributed by each sample = predicted probability -
correct answer". `或者` in the code means "or".)*

`p - y` is "the probability the model thinks it is, minus the probability it should be". If
the model is too confident about one class, that difference is positive and the weights have
to be pushed down; if it isn't confident enough, the difference is negative and the weights
have to be pushed up. **The sign itself carries the direction.**

The training loop is:

```python
def train(steps=200, lr=0.5):
    for _ in range(steps):
        grads = gradient(WEIGHT_TABLE, CORPUS)
        for word in VOCABULARY:
            for category_index in (0, 1):
                WEIGHT_TABLE[word][category_index] -= lr * grads[word][category_index]
        history.append(average_loss(CORPUS))
```

When `after.py` finishes it looks like this:

```text
  训练结束：损失 = 0.01789499，正确率 = 10/10
```
*(Training finished: loss = 0.01789499, accuracy = 10/10.)*

In the parameter table it learned, `发布` is 1.035 (4 technology sentences, 0 food sentences)
and `苹果` is -0.342 (3 technology sentences, 4 food sentences — the two sides very nearly
cancel out):

```text
  词            科技      食品   出现在几句科技句 / 几句食品句
  苹果      -0.342   0.342        3 / 4
  发布       1.035  -1.035        4 / 0
  芯片       1.957  -1.957        2 / 0
  电脑       0.267  -0.267        1 / 0
  好吃      -1.275   1.275        0 / 2
  甜        -1.188   1.188        0 / 2
```
*(Columns: word; technology; food; how many technology sentences / food sentences it appears
in. `芯片` is "chip".)*

We never told it which word belongs to which class — it read that out of the loss by itself.

## What it solves

- The loss drops from 0.6931 to 0.0179, and all 10 sentences are classified correctly.
- One forward pass produces all 32 gradients; no more running everything 64 times.
- "Which way should a parameter move, and how far" turned from something you have to try by
  hand into something you can compute.

More important than any of that: **at this point we have implemented machine learning
training, end to end.**
Score the forward pass → measure how wrong it is with the loss → use the gradient to decide
which way to change → change. Every model being trained today, GPT included, runs exactly
these four steps and not one more.

## What it still can't solve

Look back at the most valuable code in `after.py`:

```python
difference = probabilities[category_index] - (1.0 或者 0.0)
for word in words:
    result[word][category_index] += difference
```

That `(p - y)` didn't fall out of the sky. We derived it for **one chain**: sum the word
weights → softmax → cross-entropy. Derive it once and it works for one chain, and one chain
only.

But our model has a single weighted sum. What if the model gets a little more complicated?

Say, like this: compute an intermediate result first, then use that intermediate result to
compute the score:

```text
中间 = 各个词的权重加起来（再来一遍）
分数 = 中间结果再乘一遍权重
```
*(intermediate = add up the weights of each word (one more time); score = multiply the
intermediate result by weights again)*

The chain got longer, so the chain rule has to multiply one more piece. Make it more complex
still and you have to derive it all over again. Every layer has to be derived by a person, and
after deriving it you still have to check it against the numeric method — because if the
formula is wrong, the program won't raise an error. It will just quietly learn worse and
worse.

Our model has one layer; deriving it once is acceptable. But what about ten layers? A
hundred?

**That is the problem for the next chapter: add a layer in the middle, and how do we derive
the gradient?**

## Exercises

See `exercises.en.md`. Three of the most important ones:

1. Set `h` in `numeric_gradient` to `1.0`, `1e-6` and `1e-12` and check each against the
   formula. Both ends get worse — the numeric method isn't just slow, it also makes you pick
   a suitable `h` by hand.
2. Change every `-=` in `train()` to `+=` and run it. Training won't raise an error; it will
   just quietly learn worse and worse.
3. Starting from the three expressions "score, softmax, -log(probability of the correct
   answer)", derive `(p - y)` yourself. Differentiating softmax and cross-entropy separately
   is a mess, yet put them together and the result is shockingly short — that is not a
   coincidence.
