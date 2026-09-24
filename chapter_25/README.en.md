**English** | [中文](README.md)

# Chapter 25: Rewriting Our Model in PyTorch

## The problem in this chapter

The last chapter ended on a question: **why is our toy model so weak?**

In Chapter 24 we took the model apart and looked at it from every angle. It has every
part it should have: embeddings, positions, attention, residuals, LayerNorm, an MLP, an
output layer. It is just pitifully small — 2 layers, 30,000 parameters — while the thing
on the other side that can hold a conversation has hundreds of billions.

So let's make it bigger. But before we touched anything, we ran a test.

## The simplest attempt

The most intuitive move: our engine runs, doesn't it? Just make the numbers bigger.

That is what `before.py` does. First it scales the small model from Chapter 22 up a
little:

```python
small = OurModel(VOCAB_SIZE, dim=32, n_layer=2, n_head=4, block_size=16, seed=0)
big   = OurModel(VOCAB_SIZE, dim=128, n_layer=4, n_head=4, block_size=32, seed=0)
```

`dim` goes from 32 to 128, the layer count from 2 to 4, and the parameter count from
27,492 to 800,292 — 29 times bigger, and still 200,000 times short of GPT-3.

Running it looks like this:

```text
一、第 22 章的大小：dim=32，2 层
----------------------------------------------------------------
参数量：27,492
训练 200 步，用时 0.31 秒（1.6 毫秒/步）
loss 变化：3.578  3.041  1.697  0.379  0.246

二、第 24 章说的'做大一点'：dim=128，4 层
----------------------------------------------------------------
参数量：800,292（比上面大 29 倍）
只跑了 5 步，用时 0.19 秒（38 毫秒/步）
内存：169 MB -> 870 MB，5 步涨了 701 MB

时间还能忍：训练 200 步只要 8 秒。
内存不能忍：按每步 140 MB 的速度，200 步要 28,215 MB。

原因：为了 backward，我们的引擎要记住图上每一个中间张量。
每一个中间张量都是一整个 float64 的 numpy 数组，一步算完它们不会退回去。
GPT-3 有 1750 亿参数，是我们这个模型的 218,670 倍。
就算时间和内存只按参数量线性增长：一步要 138 分钟，
一步要吃掉 31 TB 内存。这条路走不通。
```

*(Section one is the Chapter 22 size — dim=32, 2 layers — which trains 200 steps in 0.31
seconds. Section two is "make it bigger" — dim=128, 4 layers — which got through only 5
steps, at 38 milliseconds per step and 140 MB of new memory per step; 200 steps at that
rate would take 28 GB. The engine can't hold it.)*

The problem is not that it **won't run**. The problem is that it **won't run fast, and
won't fit**.

To do backward, our engine has to remember the value of every intermediate tensor on the
computation graph (that is how we designed it back in Chapter 9). An 800,000-parameter
model eats 140 MB per step, and that memory only comes back when Python reclaims the
whole graph. Ten times the model, and it is 1.5 GB per step; a hundred times, and this
machine lies down flat.

So to build a big model, we first need a real engine.

## Experiment

Before we swap engines, we have to settle one thing: **after the swap, is it still the
same thing we're computing?**

(Everything below is real output. **The numbers — loss, accuracy, parameter counts — are
fixed**, because we pinned the random seeds. **Times and memory will drift** (a 10%
difference in time and a few MB in memory on the same machine are normal), so what you
get may not match what is here exactly.)

`experiment.py` runs four experiments.

### Experiment 1: the same weights — do the two engines compute the same scores?

Pour the parameters (numpy arrays) from the engine in `before.py` into the PyTorch model
unchanged, and feed both the same sentence:

```text
输入形状 (8, 16)，输出形状 (8, 16, 36)（每个位置 36 个分数）
两条输出的最大差值：5.46e-08
（差的这一点来自 float32 和 float64 的舍入，不是算错了）
```

The difference is `5.46e-08`. Not "about the same" — the same thing.

### Experiment 2: same starting point, same data, same seed, 100 steps each

```text
步数    我们的引擎    PyTorch       差值
----------------------------------------------------------------
10      3.2902        3.2902        0.00000
20      3.2823        3.2823        0.00000
30      3.0938        3.0938        0.00000
40      2.9906        2.9906        0.00000
50      2.6217        2.6217        0.00000
60      2.3389        2.3389        0.00000
70      2.1182        2.1182        0.00000
80      1.8654        1.8654        0.00000
90      1.5900        1.5900        0.00000
100     1.0396        1.0396        0.00000
```

Over 100 steps the two loss curves barely separate. Even Adam's updates line up — because
PyTorch's `Adam` and the one we wrote in Chapter 22 are the same algorithm to begin with.

### Experiment 3: PyTorch even packages up attention for you

The multi-head attention we wrote over Chapters 14, 15 and 21 — dozens of lines — is one
line in PyTorch: `nn.MultiheadAttention(dim, n_head)`. Copy the weights across
unchanged:

```text
两条输出的最大差值：1.02e-08
我们写的几十行，变成了一个 nn.MultiheadAttention(...)。里面做的事没有变：
三个投影 -> 拆头 -> 打分 -> 掩码 -> softmax -> 加权求和 -> 拼回来 -> 输出投影。
```

*(What we wrote in dozens of lines has become one `nn.MultiheadAttention(...)`. Nothing
inside it changed: three projections → split heads → score → mask → softmax → weighted
sum → stitch back → output projection. The maximum difference between the two outputs is
1.02e-08.)*

(The only difference is that it packs the Q, K and V projections into **one matrix**,
`in_proj_weight`, so we need a `torch.cat` to get our three weights in.)

### Experiment 4: the same two models, timed and measured

```text
模型                参数量        引擎          毫秒/步     训练涨的内存
----------------------------------------------------------------
dim=128，4 层       800,292       我们的引擎    30          1882 MB
dim=128，4 层       800,292       PyTorch       25          118 MB
                                  倍数          1.2x        省下 94%
----------------------------------------------------------------
dim=192，6 层       2,678,436     我们的引擎    79          1975 MB
dim=192，6 层       2,678,436     PyTorch       51          167 MB
                                  倍数          1.5x        省下 92%
----------------------------------------------------------------
```

At least 92% less memory, and only 1.2–1.9x the speed.

That speed figure is not flattering, but it is real, and it deserves a straight
explanation: **on CPU, at this model size, PyTorch simply isn't much faster** — both
sides are calling the same numpy-level matrix multiplies, and the bottleneck is Python's
interpreter overhead, not the math. Each step is only tens of milliseconds, and if you
re-run it the multiplier floats between 1.2 and 1.9; but on every run, the row with more
parameters gets the bigger multiplier — the gap grows with scale.

What really opens up the gap is three other things, and none of them can be measured with
that table:

1. **Memory.** Our engine allocates fresh float64 arrays for the whole graph every step;
   PyTorch hands the intermediates back after backward, and it uses float32.
2. **GPU.** One line — `model.to("cuda")` — and all those matrix multiplies run on
   thousands of cores. Our engine can't even dream of it; it only has numpy.
3. **The backward pass for every new operator.** In Chapter 9 we hand-wrote `exp`, `log`,
   `relu`, `tanh`, `gelu`, `softmax`, `masked_fill`… each one had to have its formula
   derived and verified by hand. PyTorch has hundreds of operators, and every one of them
   is already written.

## The new mechanism

This chapter has no new **model mechanism**. Not one word of the model changed.

What is new here is this: **swap the engine, then confirm item by item that every part in
the new engine is the one we already wrote ourselves.**

That is what PyTorch and its relatives really are — **a tensor library + automatic
differentiation + a set of standard parts**. The table below is the most important thing
in the chapter; read it with the code open beside you:

| What we wrote ourselves (Ch. 9–22) | PyTorch | Which chapter |
|---|---|---|
| `Tensor`, `.backward()`, `zero_grad()` | `torch.Tensor`, `autograd` | Ch. 9 |
| `no_grad()` | `torch.no_grad()` | Ch. 9 |
| `embedding(table, idx)` (fetch a row by index) | `nn.Embedding` | Ch. 10 |
| `x @ W + b` | `nn.Linear` | Ch. 3 |
| In the Chapter 22 model: `x @ W` (projections without bias), `LayerNorm` before the sublayer, no LN before the output layer | `nn.Linear(bias=False)`, `x + self.attention(self.ln1(x))`, no `ln_f` | Ch. 14, 18, 22 |
| `layer_norm(x, w, b)` | `nn.LayerNorm` | Ch. 18 |
| Hand-written multi-head attention (split heads → score → mask → softmax → stitch) | `nn.MultiheadAttention` | Ch. 14, 15, 21 |
| `masked_fill(mask, -1e9)` | `Tensor.masked_fill` | Ch. 21 |
| `cross_entropy(logits, targets)` | `F.cross_entropy` | Ch. 5, 20 |
| `Adam(params, lr)` | `torch.optim.Adam` | Ch. 22 |
| A class holding parameters + a hand-written `forward` | `nn.Module` + `forward` | Ch. 19, 22 |
| The training loop (zero_grad → backward → step) | Identical | Ch. 22 |

One small difference is worth remembering on its own: **`nn.Linear` stores its weight as
`(out, in)`**, because it computes `x @ W.T + b`; in Chapter 3 we wrote `x @ W + b` and
stored `(in, out)`. So moving weights from our model into PyTorch needs a transpose (see
`load_our_weights` in `after.py`). This is only about *how it is stored*, not *how it is
computed*.

There are also three details that we carried over from Chapter 22 exactly as they were —
get them wrong and the behaviour changes, so they get their own table:

| How Chapter 22 does it | How to write it in PyTorch |
|---|---|
| `LayerNorm(x)` first, then the sublayer, then `x + sublayer(...)` (Ch. 18) | `x = x + self.attention(self.ln1(x), mask)` |
| The four projections are `x @ W`, with no bias (Ch. 14) | `nn.Linear(dim, dim, bias=False)` |
| There is **no** final LayerNorm before the output layer | Don't write `self.ln_f` |

(`chapter_25/tests/test_chapter_25.py` has two tests pinning these three things down:
`test_参数表和第_22_章逐项对齐` and `test_归一化放在子层前面` — "the parameter table lines up
with Chapter 22 item by item" and "normalization goes before the sublayer".)

There are also a few new words, all PyTorch vocabulary:

- **`nn.Module`**: a box that holds parameters. Put your parameters and your `forward`
  into it, and it automatically knows how to list all the parameters
  (`parameters()`), how to save them to disk (`state_dict()`), and how to switch to
  inference mode (`eval()`).
- **buffer**: a tensor that lives with the model but is **not trained**. Our sinusoidal
  position table is a buffer (`register_buffer`) — it has to be saved and moved to GPU
  along with the model, but the optimizer is not allowed to touch it.
- **dtype**: the precision of floating point. We have used `float64` throughout; PyTorch
  defaults to `float32`. Half the memory, faster matrix multiplies, and the price is a
  few fewer decimal digits — that is where the `5.46e-08` in experiment 1 comes from.

## Python implementation

`after.py` is the Chapter 22 model, line by line, turned into PyTorch. Reading them side
by side is the clearest way.

**Table lookup + positions**:

```python
self.tok_emb = nn.Embedding(vocab_size, dim)          # the table from Ch. 10
self.register_buffer("pos_table", torch.tensor(...))  # the fixed table from Ch. 16
...
x = self.tok_emb(idx) + self.pos_table[:length]
```

`nn.Embedding` is "fetch a row by index", exactly like our `embedding(table, idx)`.

**Attention** (hand-written, because this version maps line for line onto our code and is
the easiest to compare):

```python
q = split(self.Wq(x))                     # Ch. 14: what I'm looking for
k = split(self.Wk(x))                     # Ch. 14: what you have
scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
scores = scores.masked_fill(mask, float("-inf"))   # Ch. 21: not allowed to look ahead
weights = scores.softmax(dim=-1)
out = (weights @ v).transpose(1, 2).reshape(batch, length, dim)
return self.Wo(out)
```

You can replace it with `nn.MultiheadAttention` too; the output differs by only `1e-08`
(experiment 3).

**Block** (line for line against Chapter 22):

```python
def forward(self, x, mask):
    x = x + self.attention(self.ln1(x), mask)                       # Ch. 17, 18
    x = x + self.W2(torch.relu(self.W1(self.ln2(x))))               # Ch. 19
    return x
```

Both sublayers are "**normalize first → then process → then add back onto itself**" —
the order we settled on in Chapter 18, used unchanged in Chapter 22, and carried over
unchanged here. The four projection `nn.Linear`s all carry `bias=False`, because Chapter
14 wrote `x @ W`, with no bias.

**The training loop**: line for line against `before.py`, with only the differentiation
line changing hands.

```python
def train(model, steps, block_size, batch_size, lr, seed=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ...
    for step in range(steps):
        x, y = get_batch(DATA, block_size, batch_size, rng)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()          # in before.py, we wrote this line ourselves
        optimizer.step()
```

Run `after.py` and it does one more thing that is quietly reassuring: it generates text
the Chapter 23 way.

```text
提示：床前明月光
续写：床前明月光疑是地上霜举头望明月低头思故乡春眠不觉晓处处闻啼
```

*(Prompt: `床前明月光` ("moonlight in front of my bed"). Continuation: the model writes
out the rest of that poem and then runs straight into the opening of the next one.)*

The model recognised the poem. It is overfitting to a 40-character corpus, but at least
it shows the **rewrite didn't break anything**.

## What it solves

- With the same weights, the two engines' forward passes differ by at most `5.46e-08`,
  and the 100-step loss curves are identical line by line (difference `0.00000`).
  **Swapping engines did not change the model.**
- For the same 800,000-parameter model, training memory drops from 1882 MB to 118 MB
  (94% saved); at 2.7 million parameters, from 1975 MB to 167 MB (92% saved).
- Speed on CPU is only 1.2–1.9x faster, but PyTorch can do `.to("cuda")` and our engine
  can't.
- From this chapter on, we never have to hand-write a backward pass for a new operator
  again.

More important is the feeling: `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`,
`nn.MultiheadAttention` — those names look intimidating, but taken apart, **every one of
them is something we wrote ourselves somewhere in Chapters 10 to 22**. PyTorch gave us
the packaging, the speed, and the GPU, but it did not invent a single mechanism for us.

## What it still can't solve

With the engine swapped, we moved the Chapter 22 model into PyTorch without changing a
thing, and it is still that little thing that can only recite a poem. Because **the parts
didn't change, and neither did the scale** — all this chapter bought us is the tool that
lets us scale up.

But there is one part that now looks glaring, and it has nothing to do with scale.

Look back at how we split text: we feed the model one character at a time. The character
table in Chapter 22 has 36 characters, so `床前明月光` ("moonlight before my bed") is 5
tokens.

Now try a sentence from Chapter 1:

```text
苹果很好吃
```

*(`苹果很好吃` — "apples are delicious".)*

`好吃` ("delicious") is one word, and what we feed in is two tokens, `好` and `吃`. And
this is the more troublesome case:

```text
苹果不好吃
```

*(`苹果不好吃` — "apples are not delicious".)*

Our character table has `好` and it has `吃`, but **`不好` ("not good") is not `好` plus
something** — the model would have to learn on its own that the character `不` ("not")
flips the judgement of the character after it. That is not learnable from a 36-character
corpus, because that corpus doesn't even contain `不` (the `不` in the poem is the one in
`不觉晓`, "unaware of dawn", and that is a different thing from the `不` in `甜不甜`,
"sweet or not").

Think one step further and it gets worse: the real world produces new words every day —
`平板` ("tablet"), `折叠屏` ("folding screen"), `直播带货` ("livestream selling"). Our
character table is **fixed before training starts**. When it meets a character outside
the table, all the model can spit out is "I don't know this one".

And what if we make that character table into a **word table** instead? Chinese has
millions of common words, and Chapter 2's nightmare of "tens of thousands of ifs" comes
back wearing a different coat.

So the next question is: **how should the vocabulary actually be decided? Why is our
current scheme — one character, one token — so crude?**

## Exercises

See `exercises.en.md`; here are the 3 most important ones:

1. Replace the hand-written attention in `after.py` with `nn.MultiheadAttention`, run the
   training again, and see how much the loss curve differs from the current one.
2. Replace `Block` with `nn.TransformerEncoderLayer` (the Transformer layer that ships
   with PyTorch), and pay attention to what its `norm_first` argument corresponds to
   among the choices we made in Chapters 17 and 18.
3. In `experiment.py`, change `dim` to 256 and the layer count to 8, re-measure the time
   and memory, and see how the "multiplier" column moves.
