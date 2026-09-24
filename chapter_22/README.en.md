**English** | [中文](README.md)

# Chapter 22: The First Real Small Language Model

## The problem in this chapter

At the end of last chapter we spread the parts list out and took a look:

| Part | Built in | Used in Chapter 21? |
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

Half the parts are lying in the drawer. Last chapter's model had a single attention layer with a linear layer straight after it.

So the most obvious idea is: **add layers.**

Run `before.py` and see what happens:

```text
============================================================
把 attention 一层一层叠上去（中间不加任何东西）
============================================================
    层数         参数量       训练集 loss
  ----------------------------------
     1        7265         0.0030
     2       11361         0.0000
     3       15457         2.8491
     4       19553         3.6271

  一两层还行。从第三层开始，loss 直接掉回到头。

  参考一下：33 个字符，如果模型什么都不学、
  每次都在 33 个字符里瞎猜，loss 应该是 3.4965。
  三层、四层那两个数，基本就是这个水平——它什么都没学会。
```

*(Title: "stacking attention up layer after layer, with nothing in between". Columns: number of layers, parameter count, training-set loss. One or two layers are fine; from layer three the loss falls straight back to the start — 2.8491 and 3.6271. For reference, if the model learned nothing and just guessed blindly among 33 characters, the loss would be 3.4965, which is roughly where the three- and four-layer numbers sit: it learned nothing at all.)*

Stack to layer three and the model is ruined. `before.py` explains why:

```python
x = attention(x) @ W     # 直接把 x 换掉
```

*(The comment says "replace x outright".)*

That line has two problems. First, it **replaces** rather than accumulates: every layer throws away the previous layer's x entirely, so after three layers things like "who am I" and "which position am I at" have been washed out three times. Second, **the scale drifts** — multiply x by a matrix once and its length changes once, run it through a softmax and it changes again, and nothing pulls it back into a sane range.

Chapters 17, 18 and 19 already handed us the patches. Time to install them.

## The simplest attempt

First assemble the parts the way "that Transformer block from Chapter 19" does it:

```python
x = x + 多头注意力(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

*(multi-head attention; MLP.)*

Then add two other things:

```python
x = embedding(token_ids) + sinusoidal_positions(...)   # 第 16 章那张固定的正弦表
mask = causal_mask(length)                            # 第 21 章
```

*(Chapter 16's fixed sine table; Chapter 21.)*

The positional encoding uses the **computed** sine table from Chapter 16, and it doesn't take part in training — so Chapter 22 really does not introduce a single new parameter.

Then stack that brick three high and put a linear layer on top to output 33 scores.

That's everything. **There is not one line of new mechanism in this file** — every line is something some earlier chapter did. What Chapter 22 does is **assembly**.

## Experiment

`after.py` prints this:

```text
============================================================
2. 组装模型
============================================================
  tokenizer -> embedding -> 位置 -> Transformer Block x N -> 线性层 -> softmax

  维度 64，3 层，4 个头，上下文 32
  参数量 153,441
    token embedding        2,112
    位置编码（固定的，不用学）   2,048
    每个 Transformer Block 49,728  （一共 149,184）
    最后的线性层            2,112

============================================================
3. 训练
============================================================
  第  250 步   loss 0.0144
  第  500 步   loss 0.0090
  第  750 步   loss 0.0162
  第 1000 步   loss 0.0119

  整个训练集上的 loss    0.0162
  下一个字符猜对的比例   99.49%
  （瞎猜的话是 3.03%）
```

*(Section 2, "assembling the model", is just the pipeline: tokenizer → embedding → position → Transformer Block × N → linear layer → softmax. Dimension 64, 3 layers, 4 heads, context 32, 153,441 parameters, split into token embedding 2,112; positional encoding 2,048 ("fixed, not learned"); each Transformer block 49,728 (149,184 in total); final linear layer 2,112. Section 3, "training", gives the loss at steps 250/500/750/1000, the loss over the whole training set (0.0162), and the fraction of next characters guessed correctly: 99.49%, against 3.03% for blind guessing.)*

99.49%. Whereas the three-layer stack in `before.py` was 0.00%.

Then we ask it "which character comes next":

```text
  '床前明月光疑是地' -> '上' 0.9997   '地' 0.0003   '明' 0.0001
  '举头望明月低头' -> '思' 1.0000   '故' 0.0000   '望' 0.0000
  '白毛浮绿水红掌拨' -> '清' 1.0000   '床' 0.0000   '疑' 0.0000
  '鹅鹅鹅曲项向天' -> '歌' 1.0000   '白' 0.0000   '绿' 0.0000
```

*(Four prompts and their top three candidates. After "moonlight before my bed, I took it for grou…" it answers 上 "on" at 0.9997; after "I raise my head and gaze at the bright moon, I lower my head" it answers 思 "think of" at 1.0000, completing 低头思故乡; after "white feathers float on green water, red feet pad…" it answers 清 "clear" at 1.0000; after "goose, goose, goose, neck curved, toward the sky…" it answers 歌 "sing" at 1.0000.)*

Take the second line. The window's last two characters are 低头 "lower head", and it continues with 思.

`低头思故乡`. The one-character classifier from Chapter 20 is forever wrong in this cell.

**So let it write.** Give it the opening `床前` and, 60 times over, take the highest-probability character, append it, and predict the next one:

```text
    床前明月光疑是地上霜举头望明月低头思故乡
    鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波床前
    明月光疑是地上霜举头望明月低头思故乡鹅鹅
    鹅曲
```

*(This is the model's own 60-character continuation, wrapped into four lines by us. Read it as one run: it writes the whole of Li Bai's poem — "moonlight before my bed, I took it for frost on the ground; I raise my head and gaze at the bright moon, I lower my head and think of home" — then the whole of the goose poem, then starts Li Bai's poem over again at 床前, and is two characters (鹅鹅) into the goose poem a second time when we stop it.)*

Now try a few other openings:

```text
    鹅鹅鹅 -> 鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波床前明月光疑是地上霜举头望明月低头思故乡鹅鹅鹅曲项
    举头望明月低 -> 举头望明月低头思故乡鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波床前明月光疑是地上霜举头望明月低头思
    白毛浮绿水 -> 白毛浮绿水红掌拨清波床前明月光疑是地上霜举头望明月低头思故乡鹅鹅鹅曲项向天歌白毛浮绿水红掌
```

*(Each line is "prompt → what the model continues with". From 鹅鹅鹅 ("goose, goose, goose") it finishes the goose poem, writes Li Bai's poem in full, and starts the goose poem again. From 举头望明月低 it finishes that line into 低头思故乡, writes the goose poem, then Li Bai again up to 举头望明月低头思. From 白毛浮绿水 it writes the second half of the goose poem, then Li Bai, then the goose poem again through 白毛浮绿水红掌.)*

Every one of these characters is produced by gradients that `toygrad` computed.

`toygrad` is the 300-line little engine we wrote in Chapter 9. It does exactly one thing: remember every operation, then send the gradients back along the same path in reverse. Every one of those 153,441 parameters above was nudged into place by it, one cell at a time.

**From `split()` in Chapter 1 to here, we have not skipped a single step.**

## The new mechanism

This chapter has no new mechanism. Really, none.

For every part inside `after.py` you can point at the chapter it came from:

| Code | Where it came from |
|---|---|
| `CharTokenizer.encode / decode` | Chapter 1, "splitting a sentence apart" |
| `embedding(self.token_embedding, token_ids)` | Chapter 10 |
| `x + self.position_table[:length]` | Chapter 16 (the fixed sine table, not trained) |
| `query @ key.transpose(...)` | Chapter 14 |
| splitting into `heads` and merging back | Chapter 15 |
| `scores.masked_fill(mask, -1e9)` | Chapter 21 |
| `x = x + self.multi_head_attention(...)` | Chapter 17 |
| `layer_norm(x, self.norm1_weight, self.norm1_bias)` | Chapter 18 |
| `(hidden @ self.fc_weight + self.fc_bias).relu()` | Chapter 19 (the ReLU inside it is Chapter 8) |
| `cross_entropy(logits, targets)` | Chapter 5 |
| `Adam(...) / loss.backward() / optimizer.step()` | Chapter 9, plus that one line `w = w - lr * grad` from Chapter 4 |

The only new thing in this chapter is that **assembled together, they run**.

So what this chapter has to say isn't "here's yet another new mechanism", it's: **look back.** Every detour in the previous 21 chapters that seemed to wander off has a place in this file.

## Python implementation

The model has two layers: `TransformerBlock` (the brick) and `TinyLanguageModel` (the brick plus a head and a tail).

The attention sub-layer inside the brick stacks Chapters 14, 15 and 21 on top of each other:

```python
def multi_head_attention(self, x, mask):
    batch, length, _ = x.shape

    def split(tensor):
        # (B, T, D) -> (B*heads, T, D/heads)
        return tensor.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    query = split(x @ self.query_weight)
    key = split(x @ self.key_weight)
    value = split(x @ self.value_weight)

    scores = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_dim))
    scores = scores.masked_fill(mask, -1e9)          # 第 21 章
    weights = scores.softmax(axis=-1)
    attended = weights @ value
    merged = attended.transpose(0, 2, 1, 3).reshape(batch, length, self.dim)
    return merged @ self.out_weight
```

*(The only comment is "Chapter 21" next to the masked fill.)*

Both sub-layers are the same sentence: **normalize first, then process, then add it back to itself.**

```python
def __call__(self, x, mask):
    normed = layer_norm(x, self.norm1_weight, self.norm1_bias)
    x = x + self.multi_head_attention(normed, mask)
    normed = layer_norm(x, self.norm2_weight, self.norm2_bias)
    hidden = (normed @ self.fc_weight + self.fc_bias).relu()
    x = x + hidden @ self.proj_weight + self.proj_bias
    return x
```

Notice that both `layer_norm` calls act on the x from **before** the addition. This is called "pre-normalization" (pre-norm), and it's the order from Chapter 18.

The whole model:

```python
def __call__(self, token_ids):
    batch, length = token_ids.shape
    x = embedding(self.token_embedding, token_ids)
    x = x + self.position_table[:length].reshape(1, length, self.dim)
    mask = causal_mask(length)
    for block in self.blocks:
        x = block(x, mask)
    return x @ self.head_weight + self.head_bias
```

Seven lines. Seven lines is all of it.

## What it solves

**For the first time, we have a model that can write.**

What it continues with is `床前明月光疑是地上霜举头望明月低头思故乡`. This is not a template, not a table lookup — it's the result of 153,441 floating-point numbers being computed inside `toygrad`.

Look back at the parts table again: the three patches from Chapters 17, 18 and 19 were not added for nothing. In `before.py`, stacking three attention layers directly gives a loss of 2.85 (equivalent to learning nothing); assembled into a Transformer block, the loss is 0.0162.

But `experiment.py` throws a bucket of cold water on us. It removes the parts one at a time and measures what each one is worth:

```text
                                       参数量      loss       准确率
  ------------------------------------------------------------
  完整模型                             153,441    0.0189   99.52%
  去掉多层（只剩 1 个 Block）                53,985    0.0124   99.52%
  去掉 MLP                           153,441    0.0177   99.33%
  去掉残差连接                           153,441    0.0270   99.44%
  去掉 LayerNorm                     153,441 4027.3297    4.95%
  去掉多头（只剩 1 个头）                    153,441    0.0116   99.60%
  去掉位置编码                           153,441    0.0523   99.36%
  去掉因果掩码                           153,441    0.0019  100.00%
```

*(Columns: parameters, loss, accuracy. Rows, top to bottom: the complete model (153,441 parameters, loss 0.0189, accuracy 99.52%); removing the multiple layers, leaving 1 block (53,985, 0.0124, 99.52%); removing the MLP (0.0177, 99.33%); removing the residual connections (0.0270, 99.44%); removing LayerNorm (4027.3297, 4.95%); removing the multiple heads, leaving 1 head (0.0116, 99.60%); removing the positional encoding (0.0523, 99.36%); removing the causal mask (0.0019, 100.00%).)*

Three rows deserve their own paragraph.

**One: remove LayerNorm and the loss is 4027.**

Not "a bit worse" — it simply won't train. Of the eight parts, only this one's absence is fatal.

**Two: remove the causal mask and the loss is 0.0019, the lowest in the table.**

That's cheating. Position i can see i+1, and what sits in cell i+1 is the answer. The thing from Chapter 21 shows up again here:

> How good a model's loss looks depends on what it is allowed to see.

**Three: for the remaining rows, taking the part out makes the loss go *lower*.**

This column of numbers is uncomfortable for readers of Chapters 17 and 19: "the residuals and the MLP we worked so hard to build are *better* when removed?"

It's not that the parts are useless. The scale of this experiment is too small. **Our corpus is only 304 characters.** The task is simple enough that one attention layer suffices. We're fitting 153,441 parameters to 304 characters; the surplus parameters just make training harder within the same number of steps.

The value of residuals and the MLP only becomes visible with more data and deeper models. Chapter 28 will state the rule behind this properly.

## What it still can't solve

Look back at the poem in section 5 of `after.py`. How did that poem come out?

We wrote a loop. Each round it does two things:

1. Feed the characters written so far to the model
2. Take the character with **the highest probability** that it gives, and append it

The model itself doesn't know what "write a poem" means. It only answers one question: given these characters, what's most likely to come next?

So strictly speaking, **this model only predicts; it does not generate**. That poem was stitched together by our hand-rolled loop.

And there are several questions in that "hand-rolled loop" we haven't answered:

- Is always picking the highest-probability one the only choice? What if we pick at random, according to the probabilities?
- We just tried three openings. With random picking, would the same opening write something different every time?
- If we always pick the largest, will it always be the same sentence? And if we pick the second largest, is that "wrong"?
- The longer the opening we feed it, does it write better or worse?

The next question is: **what is "generation", exactly? Why does chaining predictions one after another turn into "writing" ?**

## Exercises

See `exercises.en.md`. Here are the 4 most important ones:

1. Change `LAYERS` from 3 to 6, to 12, and re-run. Does the loss get better? What does the parameter count become? If it doesn't get better, why not?
2. Change `HEADS` from 4 to 1, to 8. Does the loss change? (With 8 heads `head_dim` is 8 — is that enough?)
3. In `HEAD`, take the two lines of `block()` and change `x = x + ...` to `x = ...` (removing the residual). Re-run and look at the loss and the poem it writes.
4. In `after.py`, `continue_text` takes the `argmax` every time. Change it to `np.random.choice`, sampling according to the probabilities. Same opening, run it three times — are the results the same?
