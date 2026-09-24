**English** | [中文](README.md)

# Chapter 8: Why we need an activation function

## The problem in this chapter

Last chapter we did one thing: we squeezed a layer of "intermediate results" in between the
input and the score.

And it was completely useless. After adding the middle layer, the loss is still 0.693147,
exactly the same as when there was only a straight line — it hasn't changed in the sixth
decimal place either.

The middle layer clearly has a whole extra row of tunable weights. On what grounds is it
completely useless?

There are two possible answers to that question, and their consequences are completely
different:

- **Answer A**: the middle layer isn't wide enough or deep enough to learn. Then widen it and
  deepen it.
- **Answer B**: the middle layer simply cannot do this, by nature. Then stacking any number of
  them is a waste.

Until we know which one it is, everything we do is guessing in the dark.

## The simplest attempt

The most intuitive reaction is A: **one layer isn't enough, so stack more.**

`before.py` takes this as far as it goes. We wrote a network that can stack however many layers
you like:

```python
def forward(params, x):
    """一层接一层，中间不加任何别的东西。"""
    current = x
    for layer in range(len(params["ws"])):
        current = [加上偏置的加权求和 ...]
    return current
```
*(The docstring: "layer after layer, with nothing else added in between". `加上偏置的加权求和`
is "weighted sum plus bias".)*

Every layer is a pure weighted sum: take the previous layer's output, multiply by the weights,
add the bias, hand it to the next layer. Nothing else is added in between.

For the gradient we just use Chapter 6's numeric method — stack n layers and you have to
derive the chain rule n times over, which is too much trouble, and the numeric method works
with any structure at all.

Here's what comes out:

```text
  层数      每层宽度      参数个数          每步要跑几次前向       训练后的 loss
   1         4         6                12        0.693147
   2         4        22                44        0.693147
   3         4        42                84        0.693147
   5         2        30                60        0.693147
  10         2        60               120        0.693147
```
*(Columns: number of layers; width of each layer; number of parameters; how many forward
passes each step needs; loss after training.)*

1 layer, 2 layers, 3 layers, 5 layers, 10 layers — the loss stops at 0.693147 for all of them.

The 10-layer network has 60 parameters, ten times as many as 1 layer. Those ten times the
parameters are worth nothing at all.

**Answer A is ruled out. Now we have to prove Answer B.**

## Experiment

Run `experiment.py`.

### Step 1: Multiply out a trained two-layer network

We're going to make the middle layer confess.

The method is crude: substitute the middle layer's expression into the output layer's
expression. Two layers is "compute `h` first, then compute the score from `h`":

```text
h    = w1 * x + b1
分数 = w2 * h + b2
```
*(h = w1 * x + b1; score = w2 * h + b2 — `分数` is "score".)*

Substitute the first line into the second, multiply the brackets out, and all that is left is
`some number * x + some number`.

We took the network from the previous chapter — the one that was trained for 400 steps —
multiplied it out, and compared it point by point:

```text
      输入                     两层网络的输出                     等效直线的输出           差
  [0, 0]   +0.00069280 / +0.00087466   +0.00069280 / +0.00087466    0.00e+00
  [0, 1]   -0.07086816 / -0.07011921   -0.07086816 / -0.07011921    1.39e-17
  [1, 0]   -0.09196347 / -0.09269373   -0.09196347 / -0.09269373    0.00e+00
  [1, 1]   -0.16352442 / -0.16368759   -0.16352442 / -0.16368759    2.78e-17

最大的差是 2.78e-17 —— 这不是接近，这是完全相同。
```
*(Columns: input; the two-layer network's output; the equivalent straight line's output;
difference. The last line: the largest difference is 2.78e-17 — this is not "close", this is
exactly the same.)*

This is not a coincidence, and it is not "roughly the same". The two-layer network and a
straight line are mathematically **the same thing**.

### Step 2: What about 3 layers, 5 layers, 10 layers

```text
    层数      参数个数           和等效直线相差
     2        22          5.55e-17
     3        42          0.00e+00
     5        53          3.47e-18
    10        60          8.13e-20
```
*(Columns: number of layers; parameter count; difference from the equivalent straight line.)*

Take a few random sets of weights and this multiplication still holds — it has nothing to do
with training, and nothing to do with what the weights happen to be. **Any stack of "pure
weighted sums" is always equal to a single layer on its own.**

So what the middle layer does is not "process" the data. It just lays the data out again. Lay
it out any number of times and it is still one straight thing.

### Step 3: Add the function that bends

Now we add a small function after the middle layer:

```python
def relu(value):
    """负数变成 0，正数原样留下。"""
    return value if value > 0.0 else 0.0
```
*(The docstring: "negatives become 0, positives stay as they are".)*

It has exactly one `if` in it. Then we test from a different angle: straight lines all share a
property —

```text
分数(u + v)  =  分数(u) + 分数(v) - 分数(0)
```
*(score(u + v) = score(u) + score(v) - score(0); `分数` is "score".)*

Plug in a few random `u` and `v`; if the model is a straight line, the two sides must be
equal.

```text
没有 relu 的两层网络：
               u               v          两边相差
  [-0.384, -1.2][-1.285, -1.006]      5.55e-17
  [1.04, -0.995] [-0.468, 0.737]      1.04e-17
  ...

加上 relu 的两层网络：
               u               v          两边相差
  [-0.384, -1.2][-1.285, -1.006]      3.17e-02
  [1.04, -0.995] [-0.468, 0.737]      4.38e-02
  ...
  -> 差得最远的一次差了 0.0438。它已经不是直线了。
```
*(A two-layer network without relu, then with relu added: the columns are u, v, and how far
the two sides differ. The last line: the worst case differs by 0.0438. It is no longer a
straight line.)*

All we added was one "negatives become 0", and this equation can never be made to hold again.

### Step 4: Once it's added, XOR is solved

```text
模型                             训练后的 loss       准确率
两层，中间不加东西                       0.693147      0.50
两层，中间加 relu                   0.00039921      1.00
```
*(Rows: two layers, nothing added in the middle; two layers, relu added in the middle.)*

The task from the previous chapter that "couldn't be separated no matter how you trained
it": all four points classified correctly, with probabilities above 0.99.

## The new mechanism

The thing this chapter introduces has two names.

### Activation function

> After each layer has computed its result, don't rush to hand it to the next layer.
> Let it pass through a simple function first, and then hand it over.
> This little function wedged between layers is called an **activation function**.

Ours is ReLU (negatives become 0). It is simple enough to fit on one line, and yet it is the
key to the whole thing working:

- Without it, a stack of layers is just "multiplying out" against each other, and however many
  you stack, it is the same straight line.
- With it, each layer's result gets **bent** a little, and once you stack bent things up, they
  can no longer be multiplied out into a simple expression.

The experiment shows it plainly: some of the values in the middle layer get clipped off by
ReLU, and from there on the output no longer satisfies the property of a straight line.

### Nonlinearity

Give this "bending" property a name.

> A straight line plus a bend is no longer a straight line.
> This property of "cannot be reduced to a straight line" is called **nonlinearity**.

Look back at these three models:

```text
第 6 章    输入 → 分数                          一条直线
第 7 章    输入 → 中间层 → 分数                  还是一条直线（乘开就没了）
第 8 章    输入 → 中间层 → 拐一下 → 分数          不是直线了
```
*(Chapter 6: input → score, a straight line. Chapter 7: input → middle layer → score, still a
straight line (it disappears when you multiply it out). Chapter 8: input → middle layer → a
bend → score, not a straight line any more. `输入` is "input", `中间层` is "middle layer", `拐一下`
is "give it a bend".)*

**To bend, there has to be a function that is not a straight line inserted in the middle.**
That sentence is the entire content of this chapter.

## Python implementation

In `after.py`, the only line of `forward` that changes is this one:

```python
z = [...]                                   # 中间层的原始输出
h = [relu(value) for value in z]            # 拐一下

scores = [...]                              # 再由 h 算分数
```
*(Comments, top to bottom: "the middle layer's raw output"; "give it a bend"; "then compute
the score from h".)*

`relu` itself is:

```python
def relu(value):
    return value if value > 0.0 else 0.0
```

The gradient has to change along with it. When the middle layer's gradient travels back,
**whatever ReLU clipped away, the gradient gets clipped away too**:

```python
# 新加的一行：relu 把负的掐成了 0，那部分梯度也传不回去
dz = [dh[j] if z[j] > 0.0 else 0.0 for j in range(hidden_size)]
```
*(The comment: "the newly added line — relu clipped the negatives to 0, so that part of the
gradient can't travel back either".)*

`z[j] <= 0` means this unit's output this time is 0, and it had no effect whatsoever on the
final score — so this time, don't give it any credit either.

If this line gets left out, the program **will not raise an error**, and training will still
run, just a little slower and a little worse. We caught it by checking against Chapter 6's
numeric gradient (`tests/test_chapter_08.py::test_hand_written_gradient_matches_numeric_gradient`).

Running it looks like this:

```text
前 5 步的 loss：0.685748  0.683558  0.681219  0.678840  0.678241
第 5000 步的 loss：0.00039921
准确率：1.00
```
*(The loss over the first 5 steps; the loss at step 5000; accuracy.)*

The values inside the middle layer look like this:

```text
      输入              中间层（过 relu 之前）                     过完 relu                两个分数
  [0, 0]  +2.51  +0.00  -0.00  +0.00  +2.51  +0.00  +0.00  +0.00  -4.541 / +3.933
  [0, 1]  -0.00  -0.24  -1.92  -3.62  +0.00  +0.00  +0.00  +0.00  +3.398 / -3.398
  [1, 0]  +5.07  -0.08  +1.92  +3.62  +5.07  +0.00  +1.92  +3.62  +3.998 / -5.100
  [1, 1]  +2.56  -0.32  -0.00  +0.00  +2.56  +0.00  +0.00  +0.00  -4.682 / +4.063
```
*(Columns: input; the middle layer before relu; after relu has been applied; the two scores.)*

Look at the two middle columns: some values got clipped to 0, others were left exactly as they
were.
When the input is `[0, 1]`, all four middle units got clipped to 0 — the model simply uses
"nothing lights up" to represent that case.

## What it solves

- XOR is solved: accuracy 1.00, loss down from 0.693147 to 0.0004.
- The two classes' scores are pulled completely apart (`-4.54` versus `+3.93`) instead of
  "0.5 each".
- For the first time we have a model that **can draw a bent line**, rather than only being
  able to draw one straight line.

And one more important thing: we figured out what exactly went wrong in Chapter 7.

It wasn't that there weren't enough layers, it wasn't that they weren't wide enough, and it
wasn't that the learning rate wasn't tuned well. It was that **between the middle layer and
the input, a bend was missing**.

With that understood, the words "neural network" finally hold up:
the layer-by-layer weighted sums do the "carrying and mixing", and the bends in between do
the "processing". Without the bends, all of the earlier work is wasted.

## What it still can't solve

Look back at this piece of code. It is the most critical line in this chapter:

```python
dz = [dh[j] if z[j] > 0.0 else 0.0 for j in range(hidden_size)]
```

This line is **something I derived with the chain rule in hand**. To derive it, I had to be
clear about:

- the derivative of the score with respect to the middle layer is `w2`;
- the derivative of the middle layer with respect to `z` is "keep the positives, zero out the
  negatives";
- multiply the two pieces together, and only then do you have `dz`.

That's two layers. Writing it out takes 20 lines. Fine.

But we already know one layer isn't enough — Chapter 7 tried it. Real tasks need ten or dozens
of layers. And every added layer means:

1. Deriving the chain rule all over again (one more index is one more place to get it wrong);
2. Rewriting a chunk of indexing code;
3. Checking it against the numeric gradient afterwards — **because getting it wrong doesn't
   raise an error.**

In the Step 4 experiment, when we stacked up to 10 layers we used the numeric gradient, simply
because hand-deriving the chain rule for ten layers was too much to write. And the numeric
gradient has a hard defect of its own: however many parameters there are, every single step
needs twice that many forward passes.

Ten layers and it becomes unwritable. So how many layers does GPT have? GPT-3 has 96.

**What the next chapter has to ask is: can the machine compute these gradients for itself?**

## Exercises

See `exercises.en.md`. Three of the most important ones:

1. Multiply the brackets out by hand in `分数 = w2 * (w1 * x + b1) + b2` (where `分数` is
   "score") and see why it is still a straight line.
2. Replace `relu` with `leaky_relu` (negatives get smaller but don't go to zero) and with
   `step` (negatives become 0, positives become 1), and run each of them. Why can't `step`
   train?
3. Move ReLU to after the output layer (`output layer → relu → softmax`). What happens? Why
   is that change wrong?
