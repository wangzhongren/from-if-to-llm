**English** | [中文](README.md)

# Chapter 28: Why bigger models are better

## The problem in this chapter

The last chapter ended with a sentence: **"our model already learns this corpus down to a
loss of 0.47 — what do we want all that size for?"**

That sentence is worth taking apart. We have 600,000 parameters, the corpus is only 40,000
characters, and training converges in three hundred steps. GPT-3 has 175 billion
parameters, nearly 300,000 times as many as ours.

What did all those extra parameters actually buy?

This chapter gives no formulas. It does one experiment first: treat parameter count as a
knob, turn it three times, and watch what loss does.

## The simplest attempt

Before turning that new knob, turn the one we already have all the way: **train longer**.

`before.py` takes the 100,000-parameter model from Chapter 27, trains it for 1200 steps in
total, and measures validation loss at steps 150, 300, 600, and 1200:

```text
训练步数    验证 loss   比上一步降了多少  用时
--------------------------------------------------------------------
150         0.4758      —                 1 秒
300         0.4625      +0.0133           2 秒
600         0.4571      +0.0054           5 秒
1200        0.4552      +0.0019           10 秒

步数翻 4 倍（300 -> 1200），loss 只降了 0.0073。
```

*(Rows: steps, validation loss, how much it fell from the previous measurement, and the
time taken. The gap from step 300 to step 1200 — four times the steps — is only 0.0073.)*

The shape of the curve is clear: **it drops fast at first, and then barely moves.**

Training longer does work, but it isn't this chapter's answer. It isn't even the reason
GPT is that big — if all it took were more training, GPT-3 should be trained for 100,000
years rather than built that large.

## Experiment

(Everything below is real output. **The numbers — loss, accuracy, parameter counts — are
fixed**, because we pinned the random seeds. **Times vary with the machine**, so what you
get may not match what is here.)

### 1. Parameter count: at the same moment, whose loss is lower

`after.py` prepares three models that differ only in size:

```text
模型    层数    维度    参数量        训练 30 步    训练 150 步
--------------------------------------------------------------------
小      1       32      16,224        1.5916        0.5998
中      2       64      107,008       0.7228        0.4758
大      5       128     1,005,440     0.6026        0.4698
```

*(Three models — small (1 layer, dim 32), medium (2 layers, dim 64), large (5 layers, dim
128) — with their parameter counts and their loss at 30 and 150 steps.)*

Every column gets smaller as you read from the bottom row upward. Parameters grew 62
times, and the loss at 150 steps fell from 0.5998 to 0.4698.

(`after.py` also converts the loss into **perplexity**: perplexity = `e^loss`. It is a
number that is easier to talk about: it means "how many characters the model is hesitating
among at each step". A loss of 0.4698 corresponds to a perplexity of 1.60 — roughly
picking one out of two characters.)

Drawn as a chart (`after.py` prints it):

```text
验证 loss（训练 150 步时）
  0.465        0.500          0.535         0.570          0.605
  ┼──────────────┼──────────────┼─────────────┼──────────────┼
                                                           ▲ 小（16,224 参数）  验证 loss 0.5998
       ▲ 中（107,008 参数）  验证 loss 0.4758
    ▲ 大（1,005,440 参数）  验证 loss 0.4698
```

*(A number line running from 0.465 to 0.605, with the three models marked on it: small
16,224 parameters, validation loss 0.5998; medium 107,008, loss 0.4758; large 1,005,440,
loss 0.4698.)*

Notice the 30-step column: the small model is still around 1.5 (better than the 3.40 of
guessing blindly, but a long way from "learned"), while the big model is already at 0.6.
**A big model isn't just more accurate in the end — it also learns faster.**

### 2. Three models, three timelines

`experiment.py` trains all three models to 600 steps, recording loss the whole way:

```text
模型      参数量        30 步       150 步      600 步
--------------------------------------------------------------------
小        16,224        1.5916      0.5998      0.4618      
中        107,008       0.7228      0.4758      0.4571      
大        1,005,440     0.6026      0.4698      0.4748      
```

*(The same three models, with loss at 30, 150, and 600 steps. At 600 steps the large model
is at 0.4748 — slightly worse than the medium model's 0.4571.)*

The first two columns are perfectly clear: more parameters, lower loss.

By step 600 the three models have squeezed into the narrow band 0.45 – 0.48, and **the
ordering no longer matters** — the one with a million parameters is even slightly higher
than the one with a hundred thousand (0.4748 vs 0.4571).

This is not a broken experiment. It is the thing most worth remembering from this chapter:
**loss has a floor.**

### 3. Can time and parameters be traded for each other

```text
配置                            参数量        训练步数    验证 loss
--------------------------------------------------------------------
小模型，训练 150 步             16,224        150         0.5998
小模型，训练 600 步（4 倍时间） 16,224        600         0.4618
中模型，训练 150 步             107,008       150         0.4758
中模型，训练 600 步             107,008       600         0.4571
大模型，训练 150 步             1,005,440     150         0.4698
大模型，训练 600 步             1,005,440     600         0.4748
```

*(Three models, each trained for 150 steps and for 600 steps, with validation loss for
each of the six runs. The small model with four times the training time reaches 0.4618 —
matching what the medium model got in 150 steps. The large model gets no better with more
time.)*

The small model at four times the training time (0.5998 → 0.4618) **catches up with what
the medium model achieved in 150 steps** (0.4758). But the medium model at four times the
time reaches 0.4571, still better; and the large model can't be pressed any lower no
matter how much more time it gets.

The conclusion: **time and parameters can partly substitute for each other, but not
without limit.**

### 4. Compare against a purely statistical model

If parameters were only "memory capacity", then a statistical model that counts
frequencies properly should do about as well. `experiment.py` contains an 8-gram backoff
model (it only counts "given these 8 characters, which character usually comes next"):

```text
模型                      参数量          验证 loss
--------------------------------------------------------------------
2-gram 回退               140 个组合      0.7147
4-gram 回退               495 个组合      0.7146
8-gram 回退               3,261 个组合    0.7147
我们的模型（小，600 步）  16,224          0.4618
我们的模型（中，600 步）  107,008         0.4571
我们的模型（大，600 步）  1,005,440       0.4748
```

*(Rows: 2-gram, 4-gram, and 8-gram backoff models with their number of combinations and
validation loss — all around 0.715 — then our three models at 600 steps: 0.4618, 0.4571,
0.4748.)*

The 8-gram counted every combination of 8 characters and still sits at 0.71; our smallest
model is at 0.46.

Why? Because the regularity of this language is **not in the adjacent characters** — it is
in "what topic this sentence is about". A topic needs several characters' worth of
evidence before it can be pinned down, and that is exactly what the things built in
Chapters 13 to 19 (attention, residuals, multi-layer composition) are doing.

(A counterexample, while we're here: the small model is at 1.59 for its first 30 steps,
worse than the 4-gram. When there hasn't been enough training, a statistical model that
only counts frequencies can beat a neural network.)

## The new mechanism

This chapter has no new mechanism and no new code module. What it has is an empirical
regularity, and that regularity has a name: **scaling law**.

It is not "big is good". It is a relationship among three quantities:

| Quantity | What it is | In our experiment |
|---|---|---|
| Parameter count N | how big the model is | 16 thousand / 100 thousand / 1 million |
| Data D | how many characters it has seen | 166,000 characters in total (33,000 of them validation) |
| Compute C | how much was computed during training | roughly `6 × N × D` (2 forward and 4 backward matrix multiplies) |

**First observation: at a fixed step count, loss falls monotonically with parameter count,
and the shape of that fall is very regular.**

Public research (Kaplan et al. 2020, Hoffmann et al. 2022) found on real data that loss
and parameter count follow an approximate power law:

```text
L(N) ≈ 常数 / N^α        （α 大约是 0.05 - 0.1）
```

*(L(N) ≈ constant / N^α, where α is roughly 0.05 – 0.1.)*

A power law means this: **every time you make the model 10 times bigger, loss falls only a
little — but that little bit is guaranteed to arrive.** On real data, trading GPT-2's 150
million parameters for GPT-3's 175 billion (more than 1000 times bigger) drops the loss by
a lot — because real language has so much learnable structure that even 175 billion
parameters can't hold it all.

Our toy language can't do that, because it has too little regularity: it bottoms out at
0.46. That is also why the chart above flattens out for us.

**Second observation: parameters, data, and compute have to grow together.**

This is the most important conclusion of Hoffmann et al. (2022, the paper called
Chinchilla): under a fixed compute budget, parameter count and data should grow in
proportion — their empirical ratio is **about 20 tokens per parameter**. In other words:

- Add parameters without adding data, and the extra parameters can only be spent memorizing
  the training set, and validation loss will not improve (the 1-million-parameter run in
  our experiment being slightly worse at 600 steps is the toy version of this phenomenon);
- Add data without adding parameters, and the model can't hold that much regularity, so the
  data is wasted too.

**Third observation: loss never reaches 0.**

Any language has a floor on its loss, equal to the randomness of the text itself (entropy,
in information theory). Public research puts English's "irreducible loss" at **around 1.7
nats/token**, and real models land near 2.1 nats/token by the time they reach a few
billion parameters. (A nat is the unit of loss. Chapter 5's loss is computed with `ln`, so
its unit is the nat; with base 2 it would be "bits", and the two units differ by a
constant.) In other words: **"predict the next word" always has a part that can't be
predicted** — it is not a task you can grind to a perfect score, only one you can keep
approaching from above.

**Put the three observations together, and they answer this chapter's question:**

> GPT has to be that big not because big models are "smarter",
> but because thousands of years of human text contain far more regularity than can be
> learned, and only an equally enormous capacity can hold it.
> And that regularity is **statistical** — make the model 10 times bigger and loss falls by
> a steady little bit. No surprises, no accidents, just a straight line.

## Python implementation

All the code in this chapter is "train and measure"; there is no new module. Three things
are genuinely worth noticing:

**One: the validation set must really never have been trained on.**

```python
SPLIT = int(len(DATA) * 0.8)
TRAIN_DATA = DATA[:SPLIT]      # the first 80%: used for training
VAL_DATA = DATA[SPLIT:]        # the last 20%: look only, never train
```

We keep saying "the loss went down", but if loss is measured on the training data, going
down only means the model memorized it. Every comparison in this chapter uses `VAL_DATA`.

**Two: use `@torch.no_grad()` and `model.eval()` when evaluating.**

```python
@torch.no_grad()
def evaluate(model, data, ...):
    model.eval()
    ...
    model.train()      # switch back after measuring, don't forget
```

Chapter 25's exercises already asked why generating doesn't need a computation graph. Same
reasoning here: evaluation only does a forward pass, so there is no backward and no need
to keep the intermediate results.

**Three: a fair comparison has to hold every other variable fixed.**

The three models use the same data, the same batch size, the same learning rate, and the
same random seeds (`torch.manual_seed(0)` and `np.random.default_rng(seed)`). Otherwise
what you measure may be noise rather than the parameter count.

## What it solves

- It confirms with **numbers from a real run**: at the same step count, growing the
  parameter count from 16 thousand to 1 million takes validation loss from 0.5998 down to
  0.4698.
- A big model isn't only more accurate in the end, **it also learns faster early on**: at
  30 steps the small model is still at 1.59 while the big one is already at 0.60.
- It also confirms that the curve flattens: at 600 steps all three models sit near 0.46 and
  the ordering stops meaning anything. **The flattening is because the data only carries a
  finite amount of information** — which is exactly what the "data D" term in the scaling
  law is about.
- And along the way it confirms that our model is not an "advanced n-gram": the 8-gram
  backoff gets 0.71, our smallest model gets 0.46.

So the Chapter 27 question can now be answered like this:

> GPT has no extra module. What it has extra is **capacity**.
> Capacity buys the ability to hold more regularity, and that regularity comes from data.
> Parameters, data, and compute have to grow together — that is the scaling law.

## What it still can't solve

Now look back at where we stand after Chapter 27:

We have GPT's architecture (Chapter 27), real tokenization (Chapter 26), an engine that can
make the model big (Chapter 25), and we know what being big is good for (this chapter).

So we trained a sufficiently large model following GPT-3's recipe. What it can do now is:
**given some text, predict the next token very accurately.**

Then you open a chat box and ask it:

```text
今天天气怎么样？
```

*(`今天天气怎么样？` — "How's the weather today?")*

It will continue that with very high probability — for instance:

```text
今天天气怎么样？明天天气怎么样？后天天气怎么样？
```

*("How's the weather today? How's the weather tomorrow? How's the weather the day after
tomorrow?")*

Or:

```text
今天天气怎么样？——这是一个很好的问题，很多网站都有天气预报。
```

*("How's the weather today? — That's a very good question, lots of websites have weather
forecasts.")*

It is not **answering** you. It is **continuing** the text you handed it, because its
training objective has been one thing from beginning to end: **make the predicted
probability of the next token higher.**

A whole step is missing in between. We have taken "predict the next word" as far as it
goes, but **being able to predict the next word is not the same as being able to chat.**

The next question is: **how did GPT-3 become ChatGPT?**

## Exercises

See `exercises.en.md`; here are the 3 most important ones:

1. Set `n_head` to 1 for all three models (single head) and re-run `after.py`. With the
   same parameter count, how much worse does the loss get? Which matters more,
   "parameters" or "structure"?
2. In `experiment.py`, cut the corpus from 12,000 sentences to 1,500, change nothing else,
   and re-run. What happens in the 600-step column? Does it line up with the "data D" term?
3. Change `draw_chart`'s `step` argument to 30 and draw the chart for 30 steps. Compared
   with the 150-step chart, is the spacing between the three points bigger or smaller?
   Why?
