**English** | [中文](README.md)

# Chapter 7: One straight line is no longer enough

## The problem in this chapter

Chapter 6 finished the job of training: score the forward pass → measure how wrong it is with
the loss → use the gradient to decide which way to change → change. All 10 sentences
classified correctly, loss 0.0179.

The last chapter ended by leaving a question behind:

> The gradient was derived by us. This model has only one layer, so deriving it once is
> enough. What about when there are more layers?

To answer it, we first need a reason to *want* to derive two layers. So let's not rush into
the gradient question. Let's look back at something else first:

**Is the model we have now actually good enough?**

What is it doing? It adds up the weights of each word in a sentence, gets two scores, and
compares them. Each word has two numbers, one adding to "technology" and one adding to "food"
— put that in terms of a picture: **it draws a straight line between the two classes**, with
technology on one side and food on the other.

(More precisely, if you take "how many times each word appeared" as coordinates, the score is
a weighted sum of those coordinates, which draws exactly a straight line on a plane. That
"score" from Chapter 3 is what this is about.)

So the obvious question: is that line good enough?

Let's give the corpus a new task. Same 10 sentences, same vocabulary. We compress each
sentence into a single number: a sentence in the technology class scores 1, a sentence in the
food class scores 0. Then we ask a simpler question:

**Given two numbers like that, are they the same?**

Same = the two sentences are of the same class, score 1; different = score 0. Four
combinations, each paired with a real pair of sentences:

```text
第一个数  第二个数  答案   来自哪两句
   0        0      1     苹果很好吃 / 香蕉很好吃
   0        1      0     苹果很好吃 / 苹果发布新手机
   1        0      0     苹果发布新手机 / 苹果很甜
   1        1      1     苹果发布新手机 / 苹果发布新芯片
```
*(Columns: first number; second number; answer; which two sentences it comes from.
`苹果很好吃` is "apples are delicious", `香蕉很好吃` is "bananas are delicious",
`苹果发布新手机` is "Apple shipped a new phone", `苹果很甜` is "apples are sweet",
`苹果发布新芯片` is "Apple shipped a new chip".)*

## The simplest attempt

The laziest approach is: **don't change a single character of the model — take Chapter 6's
code and train it.**

It's two numbers in and two classes out either way, and it still uses the same gradient
formula `(predicted probability - correct answer) × input`. It's just different data, so the
code can be carried over as is. That is what `before.py` does.

The result isn't hard to guess, but it's worth watching with your own eyes:

```text
   初始值       训练后的 loss     科技类的 w[0]     食品类的 w[0]
     0        0.693147        0.2650        0.2650
     1        0.693147       -0.1019       -0.1019
     2        0.693147        0.0126        0.0126
     3        0.693147       -0.3921       -0.3921
     4        0.693147       -0.3679       -0.3679
     5        0.693147        0.4181        0.4181
     6        0.693147        0.2784        0.2784
     7        0.693147       -0.0252       -0.0252
     8        0.693147       -0.6470       -0.6470
     9        0.693147       -0.3985       -0.3985
```
*(Columns: starting value; loss after training; `w[0]` for the technology class; `w[0]` for
the food class.)*

Ten different starting values, trained for 5000 steps, and the loss stops at the same number
every time: **0.693147**.

That equals `-log(1/2)`, the score for "guess half and half for each class". Now look at the
last two columns: the two classes' weights come out exactly the same. The model classifies
every input as "either one is fine" — it gave up.

## Experiment

Run `experiment.py`. We're going to turn "one straight line isn't enough" from a feeling into
evidence.

### Step 1: Lay out the 4 points

```text
  x2=1 |   (0,1) 答案 0        (1,1) 答案 1
       |
  x2=0 |   (0,0) 答案 1        (1,0) 答案 0
       +--------------------------------------
            x1=0                 x1=1
```
*(`答案` means "answer". The four input points, with x1 across and x2 up.)*

Walk from the bottom-left corner (0,0) to the top-right corner (1,1): both corners have
answer 1.
Walk from the top-left corner (0,1) to the bottom-right corner (1,0): both corners have
answer 0.

**The answers on the two diagonals are exactly opposite.** That is the whole shape of the
problem.

### Step 2: Try every straight line there is

Every possible straight line is three numbers: `w1`, `w2`, `b`.

We let each of them run from -3 to 3, taking a value every 0.1, three loops nested inside
each other:

```text
一共试了 226981 条直线。
其中最好的一条，准确率是 0.75（3/4）。
没有一条能到 1.00。

挑三条有代表性的直线看看：
     w1     w2      b      准确率   判对了第几个   判错了第几个
    1.0    1.0   -1.5     0.75   [1, 2, 3]       [0]
    1.0    1.0   -0.5     0.25   [3]             [0, 1, 2]
   -1.0   -1.0    0.5     0.75   [0, 1, 2]       [3]
```
*(226981 lines were tried in total. The best of them has accuracy 0.75 (3/4). Not one of
them reaches 1.00. Three representative lines are shown, with accuracy, which points it
classified correctly, and which it got wrong.)*

226,000 lines, and not one can get them all right. The ceiling on accuracy is 3/4 — every
straight line gets at least one point wrong.

This is not "we didn't find one", and it is not "the training method isn't good enough".
**This is the limit of the thing called a straight line**: it can only cut the plane into two
halves, and it cannot produce a shape where "the answers on the two diagonals are opposite".

### Step 3: So what about training with Chapter 6's method

```text
   初始值       训练后的 loss     科技类的 w[0]     食品类的 w[0]
     0        0.693147        0.2650        0.2650
     ...
10 次，一次例外都没有，全都停在 0.693147。
```
*(Ten runs, not one exception — every one of them stops at 0.693147.)*

Brute-force search or gradient descent, the conclusion is the same: **0.693147, and it will
not come down a single step.**

### Step 4: What if we add a layer

Intuition tells us: one layer isn't enough, so add a layer.

Squeeze a layer of intermediate results in between the input and the score — compute `h`
first, then compute the score from `h`:

```text
h = w1 * x + b1        （一层中间结果）
分数 = w2 * h + b2      （再由它算分）
```
*(h = w1 * x + b1 (one layer of intermediate results); score = w2 * h + b2 (and the score
comes from it))*

This is "two weighted sums strung together". Here is what comes out:

```text
模型                       训练后的 loss            两个类别的分数最大差
一层（一条直线）                  0.693147              4.44e-16
两层（中间 1 个）                0.693147              1.05e-15
两层（中间 4 个）                0.693147              1.68e-16
两层（中间 16 个）               0.693147              1.52e-16
两层（中间 64 个）               0.693147              1.53e-16
```
*(Columns: model; loss after training; largest difference between the two classes' scores.
The rows are: one layer (a single straight line); two layers (1 unit in the middle); two
layers (4 in the middle); two layers (16 in the middle); two layers (64 in the middle).)*

One layer, two layers, and widening the middle layer all the way out to 64 units — the loss
is 0.693147 every single time, identical.

The layer we went to all that trouble to add **changed nothing at all**.

## The new mechanism

The thing this chapter introduces is called the **middle layer**.

> The original model had a single weighted sum: input → score.
> Now there is one more in the middle: input → intermediate result → score.
> This middle layer doesn't face the input directly, and it doesn't give the answer directly
> either. We call it the **hidden layer**.
>
> String several weighted sums like this together and you have a **neural network**.

The name is new, the code is a new chunk, and the gradient was derived all over again too
(with one more piece of the chain rule: the output layer's error first has to be multiplied
by `w2` before it can travel back to the middle layer).

But its effect is: **zero.**

This step is worth stopping to think about. We are not introducing something new because
"adding a layer" failed; we are being forced to look for another cause precisely *because
adding a layer also failed*. If adding a layer had been enough, Chapter 8 would not exist.

## Python implementation

In `after.py`, the middle layer is computed like this:

```python
def forward(params, x):
    h = []
    for j in range(hidden_size):
        total = params["b1"][j]
        for i in range(NUM_FEATURES):
            total += params["w1"][j][i] * x[i]
        h.append(total)

    scores = []
    for c in range(NUM_CLASSES):
        total = params["b2"][c]
        for j in range(hidden_size):
            total += params["w2"][c][j] * h[j]
        scores.append(total)
    return h, scores
```

`h` is the middle layer. There is nothing else at all between it and the input — that step
matters. Hold onto that sentence; the next chapter will come back for it.

The gradient grew a piece too. The output layer's part is still the same as in the last
chapter:

```python
dscore = [(p[c] - (1.0 if c == label else 0.0)) / n for c in range(NUM_CLASSES)]
```

The middle layer is newly added: the output layer's error is divided back to each
intermediate result according to how big `w2` is:

```python
dh = []
for j in range(hidden_size):
    total = 0.0
    for c in range(NUM_CLASSES):
        total += dscore[c] * params["w2"][c][j]
    dh.append(total)
```

This piece is **derived by hand**. Get one index wrong and the program won't raise an error;
it will just quietly learn worse and worse. So we checked it against Chapter 6's numeric
gradient (`tests/test_chapter_07.py::test_hand_written_gradient_matches_numeric_gradient`):
the two sides agree.

## What it solves

Honestly: **this chapter does not solve the problem from before.**

XOR still can't be separated. The loss is still 0.693147. Accuracy hasn't moved at all.

What it solves is a different thing: **we now have the structure of "stacking layers".**

- "Input straight to score" became "input → intermediate result → score".
- The code went from a hard-coded single layer to `hidden_size`, which you can set to
  whatever you like.
- The chain rule got walked through once as well: the middle layer's gradient is passed back
  by multiplying the output layer's error by `w2`.

All three of those are the foundation everything later is built on. But a foundation is not a
house — right now it is not remotely fit to live in.

## What it still can't solve

Look at the Step 4 table in the experiment one more time:

```text
两层（中间 1 个）                0.693147
两层（中间 4 个）                0.693147
两层（中间 16 个）               0.693147
两层（中间 64 个）               0.693147
```
*(Two layers with 1 unit in the middle, 4, 16, 64 — 0.693147 every time.)*

One layer is a straight line. After adding a middle layer, the loss hasn't moved at all,
**which says it is still that same straight line.**

And yet the middle layer clearly has a whole extra row of weights that can be tuned. On what
grounds is it completely useless?

There are two possible answers to that question, and their consequences are completely
different:

- If the answer is "the middle layer is too narrow / too shallow, it just can't learn yet" —
  then we widen it, deepen it, and we're fine.
- If the answer is "the middle layer simply cannot do this, by nature" — then stacking any
  number of them is a waste.

We have to tell those two apart. We can't guess by feel.

**The first thing the next chapter has to do is make the middle layer confess:
has it actually changed the shapes this model can draw, or hasn't it?**

## Exercises

See `exercises.en.md`. Three of the most important ones:

1. Draw those 4 points on paper with a pen, and use a ruler to try to place a straight line
   that gets them all right. If you can't, that's exactly the point.
2. Change the labels to "if one of them is 1, answer 1", then change them again to "only if
   both are 1, answer 1". Can these two tasks be separated? How do they differ from XOR?
3. Run with `hidden_size` set to 1 and then to 0. Does a middle layer of 1 unit express more
   than the one-layer model?
