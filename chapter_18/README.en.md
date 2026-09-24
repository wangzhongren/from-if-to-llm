**English** | [中文](README.md)

# Chapter 18: The data gets messier the further it travels

## The problem in this chapter

Last chapter we rescued "stacking layers" with `x = x + layer(x)`. But half of that experiment table is still broken:

```text
   8      有     29.770        nan        nan        nan    1/20    1.2s
  16      有    446.262        nan        nan        nan    1/20    2.5s
```

*(`有` = "with residual"; the columns are the loss at step 1, 50, 200 and at the end, then how many of the 20 blanks were filled correctly and the time taken.)*

It is not "learning slowly" — **the first step hasn't even finished and the numbers are already absurd**: 29.77 at 8 layers, 446.26 at 16 layers. At 1 layer it was only 4.52.

This chapter we look at that carefully first. Open `before.py`: it is Chapter 17's model, not one character changed, only now the layer count is 8. It doesn't train at all; it does a single forward pass and prints **the mean and standard deviation of the tensor each layer receives**:

```text
==============================================================
一、只看一次前向：每一层拿到手的数有多大？
==============================================================
  层                           拿到手：均值       标准差       交出去：标准差
  ----------------------------------------------------------
  输入（词向量+位置编码）                -0.130     1.435         1.435
  第 1 层                       -0.130     1.435         1.790
  第 2 层                       -0.138     1.790         2.376
  第 3 层                       -0.144     2.376         3.059
  第 4 层                       -0.196     3.059         4.139
  第 5 层                       -0.077     4.139         5.502
  第 6 层                       +0.280     5.502         7.804
  第 7 层                       +0.268     7.804        10.743
  第 8 层                       -0.412    10.743        15.036
```

*(`层` = layer; `拿到手：均值 / 标准差` = the mean and standard deviation of what this layer receives; `交出去：标准差` = the standard deviation of what it hands on. `输入（词向量+位置编码）` is the input (embedding + positional encoding).)*

Look at the "standard deviation" column: **1.435 → 1.790 → 2.376 → 3.059 → …… → 15.036.**

Every layer, the numbers the next layer receives are one size bigger. Layer 8 receives numbers 10 times larger than layer 1's.

## The simplest attempt

The most intuitive idea: **just pull the numbers back, no?**

How? Look at a whole batch together, compute the mean and standard deviation of each dimension, then subtract the mean and divide by the standard deviation. This approach has one advantage: the data tells you the answer, so we never have to invent a scaling factor out of thin air. We pour every sample and every position in the batch into one big tub and compute it dimension by dimension:

```python
def dumb_normalise(x):
    mu = x.mean(axis=(0, 1), keepdims=True)   # computed over the whole batch at once
    sd = x.std(axis=(0, 1), keepdims=True)
    return (x - mu) / sd
```

Can we use it? `experiment.py`'s third part ran it. The conclusion: there are three problems it cannot get around.

## Experiment

Run `experiment.py`. It does three things:

**One, how big are the numbers each layer receives** (no training, a single forward pass, 8 layers):

```text
==============================================================================
一、每一层拿到手的数有多大（不训练，只看一次前向，8 层）
==============================================================================
  层                              无归一化：均值       标准差           有归一化：均值       标准差
  --------------------------------------------------------------------------
  输入（词向量+位置编码）                    -0.130     1.435            -0.130     1.435
  第 1 层                           -0.130     1.435            -0.000     1.000
  第 2 层                           -0.138     1.790            +0.000     1.000
  第 3 层                           -0.144     2.376            -0.000     1.000
  第 4 层                           -0.196     3.059            -0.000     1.000
  第 5 层                           -0.077     4.139            +0.000     1.000
  第 6 层                           +0.280     5.502            +0.000     1.000
  第 7 层                           +0.268     7.804            -0.000     1.000
  第 8 层                           -0.412    10.743            -0.000     1.000
```

*(`无归一化` = without normalization, `有归一化` = with normalization. `输入（词向量+位置编码）` is the input (embedding + positional encoding).)*

The three left-hand columns are Chapter 17's version: standard deviation 1.435 → 10.743, growing as it gets passed along. The three right-hand columns are this chapter's version — we get to that later; for now just remember that column of 1.000.

**Two, train 8 layers for 400 steps**:

```text
  无归一化：第 0 步 29.77  第 1 步 2.541e+09  第 10 步 nan  第 50 步 nan  第 200 步 nan  第 399 步 nan
  有归一化：第 0 步 4.665  第 1 步 3.841  第 10 步 0.9429  第 50 步 0.08215  第 200 步 0.01251  第 399 步 0.005239
```

*(`无归一化` = without normalization, `有归一化` = with normalization. `第 0 步 / 第 1 步 / 第 10 步 / 第 50 步 / 第 200 步 / 第 399 步` are the loss at step 0, 1, 10, 50, 200 and 399. `2.541e+09` is 2.541 billion.)*

The one without normalization: step 0's loss is 29.8, **one step and it climbs to 2.5 billion**, and by step 10 it is `nan`.

**Three, actually run "the dumbest approach"**:

```text
  同一个样本，换一批邻居，归一化之后拿到的东西就变了：
    这一批里有谁                                    第 0 个位置的头 4 个数
    ------------------------------------------------------
    只它自己一批（1 个样本）             +0.748  +0.307  +0.277  -0.673
    跟第 6 个样本一批（2 个）           +0.765  +0.275  +0.224  -0.584
    跟另外两个一批（3 个）              +0.749  +0.260  +0.195  -0.538
    跟全部 20 个一批                +0.702  +0.300  -0.231  -0.490

    最大差异：0.508    （同一句话、同一个位置，只是邻居换了）
```

*(`同一个样本，换一批邻居，归一化之后拿到的东西就变了` = "the same sample, batched with a different set of neighbors, gets a different result after normalization". `这一批里有谁` = "who is in this batch"; `第 0 个位置的头 4 个数` = "the first 4 numbers at position 0". The four rows are: this sample on its own (1 sample); batched with sample 6 (2 samples); batched with two others (3 samples); batched with all 20. `最大差异` = the largest difference.)*

Same sentence, same position, and only because of "who it got batched with", the output differs by 0.508. This "dumbest approach" has three problems it cannot get around:

**Problem one: the same sample, a different set of neighbors, and the result changes.**

**Problem two: when a batch holds only one sample, the statistic is that sample itself.**

What does it look like when we actually use it? A user sends us one sentence and the model has to answer. At that moment "the whole batch's mean" means nothing — there is one sample in the batch.

**Problem three: training and inference run on two different sets of logic.**

During training we have a whole batch in hand, so we can use "this batch's mean and standard deviation"; at inference time there is no "batch" any more, so we can only save the statistics accumulated during training and reuse them. Two sets of things that don't line up is a hidden trap nobody can reason about.

## The new mechanism

Let's think in a different direction.

The reason "the dumbest approach" goes wrong is that **it looks at others**: it has to ask "what do the other samples in this batch look like". But in our model, the 32 numbers at one position are already a complete description of one word — does it need to consult other words? No.

So:

> **Don't look at others. Look only at these 32 numbers at this one position.**
>
> Pull them to mean 0 and standard deviation 1.

That is **layer normalization** (LayerNorm). The formula is almost the same as "the dumbest approach", except that the mean and standard deviation are computed **along the last dimension, separately for each position**:

```text
mu  = 这 32 个数的平均值
var = 这 32 个数的方差
x_hat = (x - mu) / sqrt(var + eps)
y = x_hat * weight + bias

（eps 是个很小的数，比如 1e-5，只为了防止除以 0）
```

*(`mu` = the mean of these 32 numbers, `var` = the variance of these 32 numbers, `x_hat` = the normalized value, `y` = the output. `weight` and `bias` are learned. `eps` is a very small number, 1e-5 say, there only to prevent dividing by 0.)*

`weight` and `bias` are two learnable vectors. This step matters: **after normalizing, the model can still scale and shift it back.** We don't want to block the "scale" road off for good — we only want **the scale each layer receives to stop running away**, and "how big a scale I actually want" is left to the weights to learn.

Hold it next to "the dumbest approach" and it fixes all three things at once:

```text
  换成 LayerNorm，同样几个批次：
    只它自己一批（1 个样本）             +0.754  +0.372  +0.037  -0.661
    跟第 6 个样本一批（2 个）           +0.754  +0.372  +0.037  -0.661
    跟另外两个一批（3 个）              +0.754  +0.372  +0.037  -0.661
    跟全部 20 个一批                +0.754  +0.372  +0.037  -0.661

    所有批次的结果完全一样：True
```

*(The same four batch arrangements as before. With LayerNorm every row gives the same numbers; `所有批次的结果完全一样：True` = "the results of all batches are exactly the same: True".)*

- It doesn't look at neighbors — it makes no difference who the neighbors are.
- A batch of one is fine — because there is no need for a "batch" at all.
- Training and inference use the same logic.

One aside: people later found that the mean step can actually be dropped, and dividing by the root mean square alone is enough (this is called RMSNorm). It saves one subtraction, computes a little faster, and works about as well. We won't go into it here; you only need to know that "normalization" is a family with more than one member.

## Python implementation

Open `after.py`. Compared with Chapter 17's model it adds exactly one thing:

```python
def blocks(self, tokens):
    x = embedding(self.p["tok"], tokens) + self.p["pos"]
    for i, layer in enumerate(self.layers):
        handed = layer_norm(x, self.p["ln%d_w" % i], self.p["ln%d_b" % i])
        x = x + self.attention(handed, layer)
    return x, records
```

Two things to notice:

1. `layer_norm` already exists in toygrad (the engine from Chapter 9 ships with it; its interface is listed in `WRITING_SPEC`). It normalizes along the last dimension, which is exactly the "one position's own 32 numbers" we asked for.
2. The normalization is applied **before the work**: `x` itself doesn't move, we only normalize the copy handed to attention. The residual path is still the clean `x + ...`. That way we hold the scale down without breaking last chapter's through lane. So the order is: normalize → the sublayer does its work → add back onto `x`. Not the other way round.

Normalization has two parameters: `weight` starts as all 1s and `bias` as all 0s — that is, "do nothing, for now". They train together with the model.

Run it:

```text
==============================================================
一、只看一次前向：每一层拿到手的数有多大？
==============================================================
  层                           拿到手：均值       标准差       交出去：标准差
  ----------------------------------------------------------
  输入（词向量+位置编码）                -0.130     1.435         1.435
  第 1 层                       -0.000     1.000         1.522
  第 2 层                       +0.000     1.000         1.547
  第 3 层                       -0.000     1.000         1.599
  第 4 层                       -0.000     1.000         1.660
  第 5 层                       +0.000     1.000         1.708
  第 6 层                       +0.000     1.000         1.754
  第 7 层                       -0.000     1.000         1.809
  第 8 层                       -0.000     1.000         1.853
```

*(`拿到手：均值 / 标准差` = the mean and standard deviation of what the layer receives, `交出去：标准差` = the standard deviation of what it hands on. Notice that the "receives" column now reads `0.000 / 1.000` at every layer.)*

The "receives" column reads `0.000 / 1.000` at every layer. The "hands on" column still creeps up (the residual keeps adding on top), but over 8 layers it only reaches 1.853 — it never gets to 15 again.

Then training:

```text
  第   0 步    loss = 4.6653
  第  50 步    loss = 0.0821
  第 100 步    loss = 0.0313
  第 150 步    loss = 0.0182
  第 200 步    loss = 0.0125
  第 250 步    loss = 0.0094
  第 300 步    loss = 0.0075
  第 350 步    loss = 0.0062
  第 399 步    loss = 0.0052

  整句还原准确率：    100.0%

  原句：  我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机
  还原：  我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机

  原句：  小王 把 书 给了 小李 因为 他 明天 考试
  还原：  小王 把 书 给了 小李 因为 他 明天 考试
```

*(`第 N 步` is "step N". `整句还原准确率` is whole-sentence reconstruction accuracy: 100%. `原句` / `还原` are "original sentence" and "reconstruction" — the model now gives both sentences back word for word.)*

## What it solves

`experiment.py`'s second part ran both versions: 8 layers, 400 steps, identical hyperparameters:

```text
  无归一化：第 0 步 29.77  第 1 步 2.541e+09  第 10 步 nan  第 50 步 nan  第 200 步 nan  第 399 步 nan
  有归一化：第 0 步 4.665  第 1 步 3.841  第 10 步 0.9429  第 50 步 0.08215  第 200 步 0.01251  第 399 步 0.005239
```

- Without normalization: step 0's loss is 29.77, **one step and it is 2.5 billion**, step 10 it is `nan`.
- With normalization: starts at 4.665 and falls all the way to **0.0052**, with both sentences reproduced word for word.

Now put it beside Chapter 17's table: there, 4 layers with residuals worked and 8 layers blew up; here, 8 layers with normalization learns. **The residual connection makes "deep" possible; normalization makes "deep" actually work.**

In one sentence: **normalization is not about how smart the model is — it is about keeping the scale of the numbers each layer receives under control.** It turns "which layer am I" from a variable into a constant: whether you are layer 1 or layer 8, what you receive has mean 0 and standard deviation 1. As a result:

- one learning rate is enough (no tuning it per layer);
- deep stacks no longer fear "it just keeps climbing";
- if you want the scale bigger or smaller, the `weight`/`bias` pair after the normalization learns it.

## What it still can't solve

By now Chapter 17's residual and Chapter 18's normalization are both in place, and our model stacks 8 layers steadily. But if you look back at what it is actually doing, you notice something odd:

**Every layer does the same thing — carry other tokens' information over.**

Layer 8 does the same thing as layer 1, only the things it "carries" differ. Stack 8 of them and the tokens have looked at each other for 8 rounds; information has been shuttled back and forth many times.

But our original question was: in this sentence, does `苹果` ("Apple"/"apple") mean the fruit or the company? To answer it, carrying over the words around `苹果` is not enough — **something has to process the information that was carried over**.

Our model has "carrying", but no "processing": from the embedding to the output, every step in between is a weighted average or a linear transform.

So the next chapter's question is concrete:

> **After a token has fetched other tokens' information, does it ever get a chance, inside its own layer, to compute on that information itself? If not, where do we add that?**

With that question, we go into Chapter 19.

## Exercises

See `exercises.en.md`. The 4 most important ones:

1. Set `before.py` to 4 layers and to 16 layers and look at the two statistics tables. From which layer does it run away?
2. Manually set the `weight` of `layer_norm` in `after.py` to 3.0 and see what the "hands on" column becomes.
3. Move the normalization to **after** the residual (`x = layer_norm(x + attention(x), ...)`) and run 8 layers. Does it still learn?
4. Replace `layer_norm` with a version that only divides by the root mean square and doesn't subtract the mean (RMSNorm), run it, and compare with LayerNorm's result.
