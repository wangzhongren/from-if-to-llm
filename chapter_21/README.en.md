**English** | [中文](README.md)

# Chapter 21: Why training must not peek at the answer

## The problem in this chapter

Last chapter ended with a loose end. We'll copy it here:

> If we instead feed the whole corpus in at once and make every position output its next
> character, the input becomes `ids[t : t+T]` and the answers `ids[t+1 : t+T+1]`.
> For the i-th position in the window the target is `ids[t+i+1]`,
> and that character is exactly the **input** at position i+1 of the window.

Last chapter also left an unsolved problem. Run `before.py` first and see:

```text
============================================================
毛病一：窗口是写死的
============================================================
  第 20 章的模型吃固定 8 个字符。现在给它一段话：
    明月光疑是地上霜举头望明月   （13 个字符）

  窗口是 8，所以只能取最后 8 个：地上霜举头望明月
  前面这 5 个字符被丢掉了：明月光疑是

  「明月光疑是」这几个字对模型来说不存在。
  窗口的位置是固定的，多出来的信息只能扔。

  反过来，给它一段只有 3 个字符的话：明月光
  窗口要 8 个，凑不够。少一格，形状就对不上：
    embedding 之后是 3 × 16 = 48 个数，
    线性层要的是 8 × 16 = 128 个数。
  差的不是一点点。
```

*(The header says "Problem one: the window is hard-coded". The chapter-20 model eats a fixed 8 characters, so handed a 13-character line it can only take the last 8 — 地上霜举头望明月 — and throws away the first 5, 明月光疑是; the text says "for the model, the characters 明月光疑是 do not exist… the window's position is fixed, extra information can only be discarded". Handed a 3-character line (明月光) it can't fill 8 slots at all: after embedding you have 3 × 16 = 48 numbers, while the linear layer wants 8 × 16 = 128.)*

The second problem is in the parameter count:

```text
      窗口          线性层参数        loss        准确率
  --------------------------------------------
       2           1089      0.0870    94.70%
       4           2145      0.0046   100.00%
       8           4257      0.0026   100.00%
      16           8481      0.0016   100.00%
      32          16929      0.0010   100.00%

  窗口从 2 翻到 32，参数从 1089 涨到 16929，16 倍。
  loss 却从 0.0870 掉到 0.0026（窗口 8），后面基本不动了。

  要让它看到 1000 个字符，线性层就是 528,033 个参数。
  我们的语料一共才 304 个字符。
```

*(Columns: window size, linear-layer parameters, loss, accuracy. Windows 2 → 32, so parameters go 1,089 → 16,929, a factor of 16, while the loss only moves from 0.0870 down to 0.0026 (at window 8) and then barely changes. To let it see 1,000 characters, the linear layer alone would be 528,033 parameters — and our whole corpus is only 304 characters.)*

So what we want is something else: **it takes a sequence of any length, its parameter count has nothing to do with that length, and every position decides for itself where to look.**

We built that thing back in Chapter 14. It's called attention. Q is "what am I looking for", K is "what do I have", V is "what do I hand over", and the weights are computed, not written down by hand.

So let's put it back.

But before we do, keep last chapter's loose end in mind: **the character position i must predict is the input at position i+1. And attention lets position i see every position.**

## The simplest attempt

The simplest thing to do is: **nothing at all.**

Replace the window concatenation with attention and leave everything else as it was:

```python
def __call__(self, token_ids, use_causal_mask=True):
    x = embedding(self.token_table, token_ids) + self.position_table[:length]
    query = x @ self.query_weight
    key = x @ self.key_weight
    value = x @ self.value_weight
    attention = (query @ key.transpose(0, 2, 1)) * (1.0 / np.sqrt(self.dim))
    if use_causal_mask:
        attention = attention.masked_fill(causal_mask(length), -1e9)
    weights = attention.softmax(axis=-1)
    return (weights @ value) @ self.head_weight + self.head_bias
```

`after.py` trains it twice, once with `use_causal_mask=True` and once with `False`.

## Experiment

First look at the training loss:

```text
============================================================
对照组一：不做任何遮挡
============================================================
  第  500 步   loss 0.00445
  第 1000 步   loss 0.00159
  第 1500 步   loss 0.00094
  第 2000 步   loss 0.00061
  整个数据集上的 loss   0.00061

============================================================
对照组二：把未来挡住（因果掩码）
============================================================
  第  500 步   loss 0.00541
  第 1000 步   loss 0.00280
  第 1500 步   loss 0.00351
  第 2000 步   loss 0.00595
  整个数据集上的 loss   0.00309
```

*(Control group one is "no masking at all", ending at a whole-dataset loss of 0.00061. Control group two is "block the future (causal mask)", ending at 0.00309.)*

The unmasked one: **0.00061**. The masked one: **0.00309**. The lower side wins by 5×.

If you look at only that line of numbers, you'll conclude: "Blocking? What blocking? Let it look."

Then let it write a couple of characters and see:

```text
  开头：床前明月光

  不做遮挡：床前明月光低低低红红掌望明绿明月明绿明月明前明绿明绿明绿明月明绿水明月
  因果掩码：床前明月光疑是地上霜举头望明月低头思故乡鹅鹅鹅曲项向天歌白毛浮绿水低头
```

*(Prompt: 床前明月光, the first five characters of the Li Bai poem, "moonlight before my bed". Unmasked, the model echoes those five characters and then collapses into repeating scraps — 低低低, then 红红掌, then mostly 明月明绿 over and over. Masked, it continues the poem correctly: 疑是地上霜 ("I took it for frost on the ground"), then the whole poem to 低头思故乡, then the whole goose poem — 鹅鹅鹅曲项向天歌白毛浮绿水 — and ends by starting 低头 again.)*

**That's the answer.**

The unmasked one has a much lower loss and writes garbage. The masked one has a higher loss and writes poetry.

`experiment.py` takes this finer: give "how many slots position i is allowed to look ahead" a number, and open it up step by step from 0 to no blocking at all, watching what each setting does.

```text
============================================================
结果
============================================================
  往后看       训练 loss   遮住未来之后   接出来的文字
  --------------------------------------------------------------
      因果掩码     0.00309      0.00309   床前明月光疑是地上霜举头望明月低头思故乡鹅鹅鹅曲项向天
       1 格     0.00046      4.87624   床前明月光疑月月明月明月明月明月明月明月明月明月明月明
       3 格     0.00052      5.48393   床前明月光疑上霜故疑故举是举是举歌故故疑是举歌举是疑是
       7 格     0.00059      5.67865   床前明月光疑望明月明月低低低前明月明月明月低举举清望明
        不挡     0.00061      5.71646   床前明月光低低低红红掌望明绿明月明绿明月明前明绿明绿明
```

*(Columns: how far ahead it may look, training loss, loss after the future is blocked, and the text it continues with. The rows are the causal mask; looking 1, 3, 7 slots ahead; and no blocking. The last column is Chinese: with the causal mask it reproduces the real poem; with 1 slot of lookahead it degenerates into 月月明月明月…; with more lookahead it wanders through poem characters out of order; unblocked it writes the same junk as before.)*

The middle column is "**measure it again after blocking the future**". Same model, same dataset — only the evaluation blocks the future.

- The causal-mask row: 0.00309 → 0.00309, not a hair moved. It only ever looked at the past, so blocking makes no difference to it.
- The other three rows: from about 0.0005 straight up to around 5.

**Same training data, same model structure, same loss function. The only change is whether position i can see i+1.**

The moment it can see i+1, the model stops learning "prediction". It learns "glance to the right and copy it over". It copies fast and accurately, and the loss falls beautifully.

And when the time comes to actually write, the cell on the right is empty. There's nothing to copy.

So:

> **A low training loss does not mean the model learned what you wanted it to learn.**

This is the most valuable sentence in the chapter. It applies to every chapter after this one — Chapter 28 on scale, Chapter 29 on how a model that can write turns into an assistant that can chat: you will run into it again.

## The new mechanism

The new mechanism is almost comically simple: **fill the score of any position that is not allowed to be seen with a very large negative number.**

```python
mask = np.triu(np.ones((length, length), dtype=bool), k=1)
attention = attention.masked_fill(mask, -1e9)
```

`mask` is an upper-triangular matrix, and `True` means "not allowed to look". `k=1` means: leave the diagonal alone, start filling one cell to the upper right of the diagonal.

The point of `-1e9` is to make that cell come out exactly 0 after the softmax. Softmax is `exp(x) / sum(exp(x))`, and `exp(-1e9)` is 0. Not "very small" — exactly 0.

It looks like this (the length-8 case):

```text
        0  1  2  3  4  5  6  7     <- 被看的位置
   0   .  X  X  X  X  X  X  X    <- 看的位置 0
   1   .  .  X  X  X  X  X  X    <- 看的位置 1
   2   .  .  .  X  X  X  X  X    <- 看的位置 2
   3   .  .  .  .  X  X  X  X    <- 看的位置 3
   4   .  .  .  .  .  X  X  X    <- 看的位置 4
   5   .  .  .  .  .  .  X  X    <- 看的位置 5
   6   .  .  .  .  .  .  .  X    <- 看的位置 6
   7   .  .  .  .  .  .  .  .    <- 看的位置 7

  X = 挡掉   . = 允许
```

*(Rows are "the position doing the looking", columns are "the position being looked at"; `X` = blocked, `.` = allowed.)*

**The positions position i can see are always only 0..i.** A position can be influenced by what came before it, and never the other way round.

This mask is called a **causal mask**. "Causal" means: what happens later cannot affect what happened earlier — just as a cause cannot come after its effect.

Notice the `.` on the diagonal. Position i must be able to see itself, or it wouldn't even know "what I just wrote".

## Python implementation

`after.py` gains one function and one line that calls it:

```python
def causal_mask(length):
    """上三角（不含对角线）为 True 的方阵。"""
    return np.triu(np.ones((length, length), dtype=bool), k=1)
```

*(The docstring reads "a square matrix that is True on the upper triangle, excluding the diagonal".)*

```python
if use_causal_mask:
    attention = attention.masked_fill(causal_mask(length), -1e9)
```

`masked_fill` is an interface that was already in `toygrad` (the engine from Chapter 9 carries it); it does "wherever mask is True, fill in value". In Chapter 14 we learned to weight attention; in this chapter we learned to force certain weights to 0.

Compared with last chapter, only one other part of the model is swapped out:

| | Chapter 20 | Chapter 21 |
|---|---|---|
| How positions get to see each other | concatenate the 8 vectors in the window | attention (Q / K / V) |
| Does the parameter count depend on length | yes, it grows linearly | no |
| Can it take variable-length input | no | yes |
| Can it peek during training | no (the window only holds the past) | **yes by default; a mask has to block it** |

One more detail: `position_table` is mandatory. Attention on its own cannot tell who comes first and who comes last — it does a weighted sum, and shuffling the input order gives exactly the same result. This is where Chapter 16's positional encoding finally earns its keep.

## What it solves

```text
  解决了：
    - 序列多长都能吃（参数量跟长度无关）
    - 每个位置自己决定看哪里，不是固定往前看 8 格
    - 训练的时候不偷看答案了
```

*(The list reads "solved: it takes a sequence of any length (the parameter count is independent of length); each position decides for itself where to look instead of always looking back a fixed 8 slots; it no longer peeks at the answer during training".)*

The third one is this chapter's real subject. The causal mask makes "what the model sees during training" and "what the model has available during generation" the same thing:

- During training, position i may only look at 0..i
- During generation, position i only has 0..i available too (the characters after i haven't been written yet)

The two sides agree, and that's the only way the skill the model learned is usable for generation. In the last experiment block of `after.py`, the masked model continues from 床前明月光 with:

```text
  因果掩码：床前明月光疑是地上霜举头望明月低头思故乡鹅鹅鹅曲项向天歌白毛浮绿水低头
```

*(It reproduces the poem, then loops into the goose poem, then starts 低头 again — one character after another, all correct.)*

Not a single character wrong.

## What it still can't solve

Right now it has only **one layer of attention**, with a linear layer straight after it.

That runs, but it doesn't stack deep. Chapter 19 said attention only **exchanges** information, it doesn't **process** it — it moves things between positions, but does no per-position transformation. Chapters 17 and 18 said that without residual connections and LayerNorm, it stops learning once you add layers.

So far, the parts we have in hand are:

| Part | Built in | Where is it now |
|---|---|---|
| tokenizer / vocabulary | Chapter 1 | stood in for by `CHAR_TO_ID` |
| embedding | Chapter 10 | used |
| positional encoding | Chapter 16 | used |
| self-attention | Chapter 14 | used (with a mask) |
| multi-head | Chapter 15 | **not used** |
| residual connection | Chapter 17 | **not used** |
| LayerNorm | Chapter 18 | **not used** |
| MLP | Chapter 19 | **not used** |
| stacking layers | Chapter 19 | **not used** |

Half the parts are still sitting in the drawer. And not one of the reasons they were built has gone away in this chapter:

**If we stack up the Transformer block we assembled in Chapter 19, layer after layer, will it get better?**

## Exercises

See `exercises.en.md`. Here are the 4 most important ones:

1. Change `causal_mask`'s `k=1` to `k=0`. The diagonal is now blocked — what does the loss become? Why?
2. Change the `-1e9` in `attention.masked_fill(mask, -1e9)` to `-1e1`, to `-10`, to `0`. Is the mask still working? `before.py` says "not a very small weight, but exactly 0" — try to prove that statement from the output.
3. In `experiment.py`, change `LOOKAHEADS` to `(0, 1, 2, 3, 4, 5,...)`, opening it up one slot at a time. From which slot does the "loss after blocking the future" start exploding?
4. The table `causal_mask(8)` prints in `after.py`: if the sequence length were 32, it would have 32×32 = 1024 cells, 496 of them `X`. What if the sequence length is 1000? How big is this mask matrix? What is the relation between the memory it occupies and the "parameter count"?
