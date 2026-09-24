**English** | [中文](README.md)

# Chapter 17: More layers, and it can't learn

## The problem in this chapter

At the end of Chapter 16, our model looked like this:

```text
词向量 + 位置编码 → 一层多头 attention → 输出
```

*(Embedding + positional encoding → one layer of multi-head attention → output.)*

One layer. Positional encoding had settled "who comes before whom", and that one attention layer was doing fine.

But the Chapter 16 model has one property worth staring at: **it has no "learning" step.** In that chapter we computed the attention weights directly and looked at them. And a forward pass can't tell us what stacking up will do — we have to actually train it. So in this chapter we first string these parts together into a model that can be trained, and then we stack it up.

The previous chapter ended by leaving a question behind: **"If we hook two attention layers end to end and train, what happens?"**

This chapter answers it. And the answer is stranger than "it breaks": **two layers are fine, four are not.**

In Chapter 7 our answer was "add a layer in the middle"; in Chapter 15 it was "use more heads". **"If it isn't enough, add more" is the most-used move in this book**, so of course we thought: just stack them.

So we tried it, honestly. Open `before.py`: it strings together the parts built up all the way from Chapters 13–16 (embeddings, positional encoding, multi-head attention) into one trainable model, and then **copies that model 4 times over**, layer after layer. Nothing changed except the layer count.

The task is the two sentences we have been using throughout this part:

```text
我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机
小王 把 书 给了 小李 因为 他 明天 考试
```

*("Yesterday at the mall I saw the new phone Apple had just released", and "Xiao Wang gave the book to Xiao Li because he has an exam tomorrow.")*

We replace **one word** with `<空>` ("empty") and ask the model to give the whole sentence back. The task isn't hard, it's almost dumb: the laziest correct solution is "copy the sentence, and in the blank write the one word that belongs there". That is exactly what we want — if it can't even learn to copy, the problem is out in the open.

## The simplest attempt

The simplest attempt is not to attempt anything: **change the layer count from 1 to 4 and touch nothing else.**

`before.py` prints this:

```text
============================================================
把 attention 堆 4 层：
============================================================
  第   0 步    loss = 3.1516
  第  50 步    loss = 2.2353
  第 100 步    loss = 2.0701
  第 150 步    loss = 2.0156
  第 200 步    loss = 1.9483
  第 250 步    loss = 1.9626
  第 300 步    loss = 2.0683
  第 350 步    loss = 2.0345
  第 399 步    loss = 2.0343

  整句还原准确率：    17.3%
  被挖空位置准确率：  10.0%  (2/20)

  原句：  我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机
  还原：  手机 手机 手机 手机 手机 手机 手机 手机 手机 手机 手机

  原句：  小王 把 书 给了 小李 因为 他 明天 考试
  还原：  明天 <补> 明天 <补> 明天 明天 明天 明天 <补>
```

*(Reading the block: `第 N 步` is "step N" and `loss` is the loss. `整句还原准确率` is whole-sentence reconstruction accuracy (17.3%); `被挖空位置准确率` is accuracy on the blanked position (10.0%, that is 2 out of 20). `原句` is "the original sentence" and `还原` is "the reconstruction" — what the model produced. The first reconstruction is `手机` ("phone") eleven times over; the second is `明天` ("tomorrow") alternating with `<补>` (the padding placeholder, since the second sentence is two words shorter than the first).)*

The loss wobbles from 3.15 to 2.03; **over 400 steps it barely moved.** What comes back out is a string of `手机` ("phone").

One detail here is worth a look: `<补>` is the placeholder we pad with (the second sentence is two words shorter than the first). The model gets wrong even the positions where the answer is pure position — "a `<补>` goes here". It isn't "learning slowly", it **is not learning at all**.

## Experiment

We set the layer count to 1, 2, 4, 8, 16 in turn, ran 400 steps for each, with every other hyperparameter identical. Run `experiment.py`:

```text
==============================================================================
任务：把被挖空的那个词填回来（一共 20 个位置）
训练：400 步，学习率 0.05，隐藏维度 32，2 个头，初始化尺度 0.177
==============================================================================
  层数     残差      第 1 步     第 50 步    第 200 步         最后      填对      用时
------------------------------------------------------------------------------
   1      无      3.834      0.285      0.028     0.0096   20/20    0.2s
   2      无      3.440      0.266      0.009     0.0030   20/20    0.4s
   4      无      3.152      2.235      1.948     2.0343    2/20    0.7s
   8      无      3.084      2.346      2.349     2.3451    1/20    1.2s
  16      无      3.150      2.420      2.343     2.3460    0/20    3.0s
------------------------------------------------------------------------------
   1      有      4.520      0.150      0.023     0.0096   20/20    0.2s
   2      有      5.793      0.040      0.006     0.0027   20/20    0.4s
   4      有      9.388      0.017      0.003     0.0014   20/20    0.7s
   8      有     29.770        nan        nan        nan    1/20    1.4s
  16      有    446.262        nan        nan        nan    1/20    3.1s
------------------------------------------------------------------------------
```

*(`层数` = layer count, `残差` = residual, `无` = without, `有` = with. `第 1 步 / 第 50 步 / 第 200 步 / 最后` are the loss at step 1, 50, 200 and at the end. `填对` = how many of the 20 blanks were filled correctly. `用时` = time taken.)*

First, the top half: the rows where the `残差` (residual) column says `无` ("without").

By rights a model with more layers is more capable: whatever 4 layers can represent, 2 layers may not be able to. The numbers say the opposite:

| Layers | Final loss | Blanks filled correctly (out of 20) |
|---|---|---|
| 1 | 0.0096 | 20 |
| 2 | 0.0030 | 20 |
| 4 | **2.0343** | **2** |
| 8 | **2.3451** | **1** |
| 16 | **2.3460** | **0** |

**With more layers it doesn't just fail to improve — it stops learning altogether.** Two layers can recite the two sentences word for word; four layers can't even copy them.

(The last column is how long the run took; it may differ on your machine. The numbers before it come from a fixed seed and won't change. When you run this, your terminal will also spit out a pile of numpy overflow warnings — that is the noise "the numbers blew up" makes, and we deal with it head-on in the next chapter.)

## The new mechanism

At this point we have to think one thing through: **why does stacking one more layer make the model worse?**

One attention layer does this: every position looks at all positions, pulls their `V` back by similarity, takes a weighted average, and hands it to the next layer. So what does layer 2 receive? **The vectors layer 1 produced, already "re-mixed".** And layer 3 receives what layer 2 produced...

The trouble is in those five words, "receives what it produced".

Think of it this way: if every layer **fully replaces the previous layer's result with something it just computed**, then the input to layer 8 has been replaced 8 times. Is the original word's information still in there? That depends entirely on whether each of those 8 replacements happened to keep it.

And what is the laziest solution to our task? **Copying.** That is, what we actually want this model to learn is:

> This layer: please do nothing. Pass the input through unchanged.

But in a "replacement"-style network, "do nothing" is the hardest thing to learn — every layer has to use its own parameters (one matrix multiply plus one weighted average) to fake an identity map. The more layers there are, the harder that is to fake.

So the new mechanism is:

**Don't replace. Add.**

```python
x = x + layer(x)
```

We no longer let this layer "produce the next layer's input"; we let it **produce "the thing to change"**. Its meaning is completely different:

- In the "replacement" style, this layer has to learn "what should I output".
- In the "add" style, this layer only has to learn "what should I change". If nothing needs changing, it just drives its output toward 0 — **and that is the correct way to say "do nothing".**

We call that line — the one that runs from `x` straight past the layer to the other side — the **residual connection**, and the branch being added is the **residual**. "Residual" means "what is left over, the part not yet explained" — the layer doesn't have to explain the whole output, only "what is still missing relative to the input".

There is also a more concrete benefit. In Chapter 9 we said backpropagation sends gradients back along the same path they came. In the "replacement" style, a gradient has to pass through matrix multiply after matrix multiply and softmax, and each of those multiplies it by a coefficient; pass through 8 of them and it may be multiplied down to 0. In the "add" style, the derivative along `x = x + layer(x)` is

```text
d(x + layer(x))/dx = 1 + layer'(x)
```

*(The derivative of the residual path is 1 plus the layer's own derivative.)*

That **1** is a through lane: the gradient can skip every matrix multiply and come back to the first few layers completely intact.

## Python implementation

Open `after.py`. It differs from `before.py` **by one character**:

```python
def forward(self, tokens):
    x = embedding(self.p["tok"], tokens) + self.p["pos"]
    for layer in self.layers:
        # in before.py this line is:
        #   x = self.attention(x, layer)
        # in after.py this line is:
        x = x + self.attention(x, layer)
    return x @ self.p["Wout"]
```

The whole `self.attention(x, layer)` function (splitting heads, computing scores, softmax, weighted average, merging, output projection) is unchanged, character for character — it is the one from Chapters 14 and 15. The only thing that changed is what we do with its result once it's computed.

Also note this initialization line:

```python
s = 1.0 / np.sqrt(d_model)
```

Every weight is a random number with standard deviation `1/sqrt(d)`. This is nothing new; we have been doing it since Chapter 8. The reason to mention it here is that in the next chapter you will find that after adding residuals, this rule is no longer enough.

Run it:

```text
============================================================
加了残差的 4 层模型：
============================================================
  第   0 步    loss = 9.3879
  第  50 步    loss = 0.0166
  第 100 步    loss = 0.0070
  第 150 步    loss = 0.0043
  第 200 步    loss = 0.0031
  第 250 步    loss = 0.0024
  第 300 步    loss = 0.0019
  第 350 步    loss = 0.0016
  第 399 步    loss = 0.0014

  整句还原准确率：    100.0%
  被挖空位置准确率：  100.0%  (20/20)

  原句：  我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机
  还原：  我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机

  原句：  小王 把 书 给了 小李 因为 他 明天 考试
  还原：  小王 把 书 给了 小李 因为 他 明天 考试
```

*(`第 N 步` is "step N". `整句还原准确率` is whole-sentence reconstruction accuracy (100%), `被挖空位置准确率` is accuracy on the blanked position (100%, 20/20). `原句` / `还原` are "original sentence" and "reconstruction" — and this time the `还原` line repeats the `原句` line exactly, word for word (`我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机`, "yesterday at the mall I saw the new phone Apple had just released"), where the version without residuals answered with a string of `手机`.)*

Same task, same layer count, same hyperparameters. Add that one plus sign and all 20 blanks are correct.

One more thing is worth noticing: **the loss at step 0 actually got bigger** (3.15 before, 9.39 now). That is not a bug, it is the other face of the residual connection — every layer adds something on top, four additions, so the numbers are naturally bigger. At 4 layers this "getting bigger" is still harmless, but it is a time bomb, and we step on it in the next chapter.

## What it solves

Go back to that table and read the two halves together:

| Layers | Without residual | With residual |
|---|---|---|
| 1 | 0.0096, 20/20 | 0.0096, 20/20 |
| 2 | 0.0030, 20/20 | 0.0027, 20/20 |
| 4 | **2.0343, 2/20** | **0.0014, 20/20** |
| 8 | **2.3451, 1/20** | the numbers blew up |
| 16 | **2.3460, 0/20** | the numbers blew up |

At 1 and 2 layers both versions work — which is why the Chapter 16 model was fine all along, and why this defect only shows up once you stack.

From 4 layers the two part ways: **the old version cannot learn at all (the loss is stuck at 2.03), the new version fills all 20 blanks correctly.**

One sentence for what this chapter bought us: **the residual connection makes "deep" possible.** Not smarter — it just stops "stacking 8 layers" from meaning "mangle the input 8 times". A newly added layer goes from "what should I output" to "what should I change".

## What it still can't solve

Now look at the bottom half of the table: the `有` ("with") rows.

8 layers — `nan`. 16 layers — `nan`.

Not "learning slowly": training **blew up on the spot**. And look at the step-1 loss: 4.52 at 1 layer, 9.39 at 4 layers, **29.77** at 8 layers, **446** at 16 layers. The model hasn't started learning yet and the numbers have already climbed.

The reason is plain, and it is sitting in the line we just wrote:

```python
x = x + self.attention(x, layer)
```

Every layer adds a fresh vector on top. If what each layer adds is roughly as large as what is already there, then after 8 layers the length of `x` has grown several times over; after 16 layers, dozens of times. The initialization rule from Chapter 8 (every weight `1/sqrt(d)`) was designed **for one layer**; it has no say over "adding 8 times".

So the next chapter's question is concrete:

> **Now that we have residuals, every layer keeps piling things onto `x`. After 8 layers the numbers inside `x` have grown too large for training to survive — how do we make the numbers each layer receives have roughly the same scale, no matter which layer it is?**

With that question, we go into Chapter 18.

## Exercises

See `exercises.en.md`. The 4 most important ones:

1. Change `after.py`'s `n_layers` to 8, run it, and watch the `nan` with your own eyes.
2. In `before.py`, set the layer count to 1 and 2, and see how many layers the old version survives.
3. Change `x = x + self.attention(x, layer)` to `x = 0.5 * x + self.attention(x, layer)` (discount the residual halfway) and see whether 8 layers can run.
4. Add a piece to `after.py`: print the standard deviation of each layer's output. Count how many times it grows from layer 1 to layer 4.
