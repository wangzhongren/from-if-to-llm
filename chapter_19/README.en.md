**English** | [中文](README.md)

# Chapter 19: Why there is an MLP after attention

## The problem in this chapter

By the end of last chapter our model was one that can be stacked:

```text
词向量 + 位置编码 → [ 归一化 → Attention → 加回去 ] × 8 → 输出
```

*(Embedding + positional encoding → [ normalization → Attention → add back ] × 8 → output.)*

8 layers, stable, and it learns. If we took it out and used it right now, you would notice one problem — the only thing it can do is one thing:

**Carry other positions' information over.**

Layer 8 does the same thing as layer 1, only the things "carried" differ. Over 8 layers every word has looked at other words for 8 rounds; information has been shuttled back and forth many times, but **no position has ever performed an operation of its own on the information carried over**.

Back to our earliest question: in this sentence, is `苹果` ("Apple"/"apple") the fruit or the company? To answer it, carrying over the words around `苹果` is not enough — after they are carried over, **something has to make the judgement**.

So what this chapter asks is:

> After a position has fetched other positions' information, who is responsible for processing it?

We have to build a task first, one that separates "exchange" from "processing". A task on the real corpus can't do this — the tasks there (blank-filling, say) are satisfied by "carrying" alone, so the difference never shows up. So we build the smallest possible probe task:

```text
每个位置给一个 32 维向量（随机生成，并且先做一次标准化）
要求每个位置输出一个 0/1：
    它自己前两个数，异或一下
    （一个正一个负 → 1；同号 → 0）
```

*(Each position is given a 32-dimensional vector (randomly generated, and standardized first). Each position must output a 0/1: XOR its own first two numbers (one positive and one negative → 1; same sign → 0).)*

The answer to this task **depends only on the position itself** — no "exchange" is needed anywhere, it is a pure test of "processing". And it is the old friend XOR from Chapter 7: a straight line can't separate it, and neither can a weighted average.

## The simplest attempt

The simplest attempt is "stack a few more layers".

Chapters 17 and 18 have just repaired "stacking layers" (residuals, normalization), so it is natural to think: **with more rounds of exchange, won't processing just appear on its own?**

Open `before.py`. It is Chapter 18's model — residuals, normalization, Attention, two blocks — with one difference: there is nowhere in this model that "processes". Run 900 steps:

```text
  第   0 步    loss = 1.3330
  第 150 步    loss = 0.6997
  第 300 步    loss = 0.6961
  第 450 步    loss = 0.6954
  第 600 步    loss = 0.6875
  第 750 步    loss = 0.6816
  第 899 步    loss = 0.6730

  测试集：loss = 0.7035   准确率 = 50.3%
  （瞎猜是 50%，loss 是 0.6931）
```

*(`第 N 步` is "step N". `测试集：loss = 0.7035 准确率 = 50.3%` = "test set: loss = 0.7035, accuracy = 50.3%"; `（瞎猜是 50%，loss 是 0.6931）` = "(guessing blindly gives 50%, with loss 0.6931)".)*

**50.3%.** Over 900 steps the loss moved from 0.699 to 0.673 and then never moved again — this model does exactly as well buying lottery tickets as answering the questions.

## Experiment

Run `experiment.py`. It does three things, each one clearer than the last.

**One, take the parts apart on the real corpus.**

Using those two sentences for "blank-filling restoration" (the task from Chapters 17 and 18), 2 blocks, 400 steps, removing Attention and MLP in turn:

```text
  结构                        loss          被挖空的位置填对
  ----------------------------------------------
  只有 MLP                  0.0621             11/20
  只有 Attention            0.0111             20/20
  Attention + MLP         0.0051             20/20
```

*(`结构` = structure; `被挖空的位置填对` = "blanked positions filled correctly". `只有 MLP` = MLP only, `只有 Attention` = Attention only, `Attention + MLP` = both.)*

The 11/20 on the "MLP only" row is the "blind guess" score (each blank has two possible answers: it could be the word from the first sentence, or the word from the second). The reason is plain: **without Attention, this position simply cannot see the other words**; it can only guess from the information at its own position.

So "exchange" really does have to be done by Attention.

**Two, hand the same probe task to three structures.**

```text
  结构                       训练 loss     测试 loss       测试准确率
  --------------------------------------------------------
  只有 MLP                    0.0253      0.1427       94.1%
  只有 Attention              0.6730      0.7035       50.3%
  Attention + MLP           0.0331      0.2304       91.3%
```

*(`结构` = structure; `训练 loss` = training loss, `测试 loss` = test loss, `测试准确率` = test accuracy.)*

- **MLP only: 94.1%** — it doesn't look at other positions at all, but "processing" it can do.
- **Attention only: 50.3%** — it can't even "XOR the two numbers on itself".
- **Attention + MLP: 91.3%** — with both, both work.

(One aside: this probe task never needed "exchange" in the first place, so "MLP only" comes out a hair better than "both" — Attention is no help here, it just tags along. Real tasks need both; see the table in part one.)

**Three, why can't Attention do it?**

We take the real two sentences, feed them to an attention layer, and pull its weights out to look at:

```text
  1) 一层 attention：输出的每一行，是不是「值的加权平均」？
     权重矩阵：每行求和 = 1.000000 ~ 1.000000，最小权重 = 0.000142
     手算的平均 vs 库算的输出，最大误差 = 0.00e+00

  2) 堆 8 层：每一层都还是「平均」，只是平均的对象换了
     8 层的权重连乘之后：每行求和 = 1.000000 ~ 1.000000，最小权重 = 0.018808
     8 层输出 vs 输入的一次加权平均：最大误差 = 4.44e-16
     8 层里，输出跑出「这一层的值的范围」的最大越界量 = 0.0000000000

  3) 换成 MLP：输出可以跑到输入范围外面去
     MLP 输出的最大越界量 = 5.597893
     （输入的第 0 维范围是 [-3.087, +1.605]，MLP 输出是 [-4.555, +2.775]）
```

*(1) One attention layer: is every row of the output a "weighted average of the values"? The weight matrix: every row sums to 1.000000 ~ 1.000000, the smallest weight is 0.000142. Hand-computed average vs the library's output: max error 0.00e+00.

2) Stack 8 layers: every layer is still an "average", only the thing being averaged changes. After multiplying the 8 weight matrices together: every row still sums to 1.000000 ~ 1.000000, the smallest weight is 0.018808. The 8-layer output vs a single weighted average of the input: max error 4.44e-16. Over the 8 layers, the largest amount by which the output escapes "the range of this layer's values" = 0.0000000000.

3) Switch to MLP: the output can go outside the input's range. The MLP output's largest escape = 5.597893. (Dimension 0 of the input ranges over [-3.087, +1.605]; the MLP output is [-4.555, +2.775].)*

Those three numbers are the point. Take them one at a time:

1. In one attention layer's weights, **every row sums to exactly 1 and no entry is negative** (an inevitable consequence of softmax). And its output is **exactly "these weights times these values"** (error 0.00e+00). In other words, this layer did nothing else — it is a weighted average.
2. Multiply the 8 layers' weight matrices together and the resulting matrix **still has every row summing to 1**, still non-negative. So after 8 layers, what each position receives is **still a single weighted average of the input** (max error 4.44e-16). An average of averages is still an average.
3. Weighted averaging has a property it cannot get around: **the result must lie inside the range of the numbers being averaged.** Over 8 layers the escape is identically `0.0000000000`. An MLP is not an average — its output escapes by 5.6, reaching places the input could never reach.

In one sentence: **Attention can only redistribute among "the numbers that already exist"; it cannot create anything new.** And a judgement like "XOR two numbers" needs a genuinely nonlinear operation — which is exactly Chapter 7's lesson: one straight line is not enough, you have to add a middle layer.

## The new mechanism

We need something that **does its arithmetic position by position, each on its own**. So we hang a small network after Attention:

```text
线性（升维）  →  ReLU  →  线性（降维）
```

*(linear (widen) → ReLU → linear (narrow back).)*

That's it. Three details:

**One, it is "per position".** The same network is applied at every position, each computing for itself — when position 3 computes, it cannot see position 5 at all. This is exactly complementary to Attention: Attention only looks at others, the MLP only looks at itself.

**Two, why widen in the middle.** 32 dimensions come in, become 128, get ReLU'd (negatives wiped to 0), then are squeezed back to 32. The middle is wider to leave room for nonlinearity — squeezed into the original 32 dimensions there would be too few transforms available.

**Three, it is wired the same way Attention is**: normalization → do the work → add back. Nothing from last chapter or the chapter before it changes a character.

```python
x = x + attention(layer_norm(x))   # exchange
x = x + MLP(layer_norm(x))         # process
```

Put these two together and you have a complete block. This thing has a formal name — from Chapter 25 onward you will see it every day:

> **This whole block, we call it a Transformer block.**

Now we have every part in hand:

```text
        输入（每个位置一个 32 维向量）
          │
          ├──────────────┐
          ↓              │
       层归一化           │
          ↓              │
       Attention（交换）   │    ← 从别的 position 取信息
          ↓              │
          + ←────────────┘
          │
          ├──────────────┐
          ↓              │
       层归一化           │
          ↓              │
       MLP（加工）        │    ← 自己 position 内部做一次非线性变换
          ↓              │
          + ←────────────┘
          │
          ↓
        输出
```

*(The diagram, top to bottom: the input (a 32-dimensional vector at every position) splits; the left branch goes 层归一化 ("normalization") → Attention ("exchange" — fetch information from other positions) → `+`, with the residual arrow coming back around from the right; the result splits again, into 层归一化 → MLP ("processing" — one nonlinear transform inside its own position) → `+`, again with the residual coming around; then 输出, the output.)*

Residual connections (Chapter 17), layer normalization (Chapter 18), Attention (Chapters 14–16), the MLP (this chapter). Not one more, not one fewer.

## Python implementation

Open `after.py`. Compared with Chapter 18's model it adds exactly one method:

```python
def mlp(self, x, block):
    """A small per-position network: widen -> ReLU -> narrow. Each position computes for itself, blind to the others."""
    hidden = (x @ block["W1"] + block["b1"]).relu()
    return hidden @ block["W2"] + block["b2"]
```

plus one extra line inside `hidden()`:

```python
def hidden(self, x_in):
    x = x_in if isinstance(x_in, Tensor) else Tensor(x_in)
    for block in self.blocks:
        # exchange: this position fetches information from other positions
        x = x + self.attention(layer_norm(x, block["ln1w"], block["ln1b"]), block)
        # process: this position performs one nonlinear transform internally
        x = x + self.mlp(layer_norm(x, block["ln2w"], block["ln2b"]), block)
    return x
```

`block["W1"]` has shape `(32, 128)` and `block["W2"]` is `(128, 32)` — widen first, then narrow. There are two sets of normalization parameters (`ln1` for Attention, `ln2` for the MLP), because the two of them receive different things.

Run it:

```text
  第   0 步    loss = 2.6079
  第 150 步    loss = 0.6298
  第 300 步    loss = 0.4190
  第 450 步    loss = 0.1480
  第 600 步    loss = 0.1017
  第 750 步    loss = 0.0461
  第 899 步    loss = 0.0331

  测试集：loss = 0.2304   准确率 = 91.3%
  （瞎猜是 50%，loss 是 0.6931）
```

*(`第 N 步` is "step N". `测试集：loss = 0.2304 准确率 = 91.3%` = "test set: loss = 0.2304, accuracy = 91.3%"; `（瞎猜是 50%，loss 是 0.6931）` = "(guessing blindly gives 50%, with loss 0.6931)".)*

## What it solves

Put the three rows side by side:

| Structure | Probe task (needs processing) | Real task (needs exchange) |
|---|---|---|
| MLP only | **94.1%** | 11/20 (blind guessing) |
| Attention only | **50.3%** (blind guessing) | 20/20 |
| Attention + MLP | **91.3%** | 20/20 |

Each of the two jobs goes to its own owner:

- **Exchange**: for a position to get other positions' information, only Attention can do it. Because the MLP "cannot see others", it is completely useless at this.
- **Processing**: for a position to compute on what it has received, only the MLP can do it. Because Attention "only knows how to take weighted averages", it is completely useless at this too (50.3% — that is buying lottery tickets).

And real language needs both: first fetch the context, then make a judgement about what was fetched.

After this chapter our model is "complete" for the first time — starting from an embedding, going through exchange and processing, and ending with a vector you can make judgements from. This structure later came to be called the Transformer block, and after Chapter 25 we will rewrite it with PyTorch; you will find that not one part of it has changed.

## What it still can't solve

What does the model output now? Still **a pile of vectors**: after two blocks, every position gets a 32-dimensional vector.

But what do we want? **A word.**

Look back at this task: we make the model fill in the blanked word — and the reason we can is that we bolted a `(32, vocabulary size)` output layer onto the end, which turns vectors into "a score for each word". That output layer is specific to this task.

But what about "predict the next word"? What if I want this model to write sentences on its own? Right now it only spits out vectors, not words. And there is a more basic problem:

> **The tasks we have been doing are all "here is a sentence, here is a hole, fill it back in". That is more like a fill-in-the-blank exercise than speaking. A model that can write on by itself should be able to take a sentence and continue it — and at that point what it has to output is not "pick a word out of this sentence to put back in", but "what is the next word".**

From a "fill-in-the-blank model" to a "model that can talk", what is the missing step called? In Chapter 20 we bring back an old friend from Chapter 2: the **classifier**.

## Exercises

See `exercises.en.md`. The 4 most important ones:

1. Change the probe task's labels to "or" (1 if either of the two numbers is positive) and run the three structures again. This task is linear — can only Attention learn it?
2. Change the MLP's middle dimension in `after.py` from 4x to 1x (no widening) and run it. Does it still learn?
3. Replace the `relu()` in the MLP with doing nothing (remove the activation) and run it. Which part of the MLP is doing the work?
4. Do Chapter 18's "per-layer statistics table" for this complete block as well (print the standard deviation of what each layer receives). Do both normalizations (`ln1`, `ln2`) do their job?
