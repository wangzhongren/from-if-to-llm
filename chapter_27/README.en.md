**English** | [中文](README.md)

# Chapter 27: GPT

## The problem in this chapter

Chapter 26 ended with a sentence: **what differences are left between us and GPT?**

Back there we listed a table whose right-hand column read "LayerNorm before computing",
"GELU", "a learnable position table" — things we had never written. They look like new
modules.

But one thing is still unverified. We have kept saying "our model is only small". Is that
actually true? Blow our model up — the structure settled in Chapter 22, the copy moved
into PyTorch in Chapter 25 — to GPT-2 small's layer count, and does it hold up?

To ask that question we switched to a bigger corpus. The poem is only 40 characters; the
model memorizes it and that is the end of it, and no architectural difference shows up.
`before.py` invents a "toy language":

- every sentence belongs to a topic (tech / food), and the topic switches from time to
  time;
- only words from the same topic go together (`苹果` "apple" + `发布新手机` "shipped a new
  phone", `香蕉` "banana" + `很好吃` "is delicious");
- so to predict the next character, you first have to guess from the preceding characters
  what this sentence is talking about.

That corpus has 40,000 characters and 29 distinct characters, 1000 times bigger than the
poem, and its regularity lives in the **context**, not in any single character. That is
enough for us to see the difference between "it can learn" and "it can't".

## The simplest attempt

The most intuitive move: GPT-2 small has 12 layers and so do we — turn the layer count
up, turn the learning rate up a little, and go.

`before.py` runs three configurations. The first is 4 layers with a normal learning rate:

```text
一、我们的模型，4 层，学习率 0.003
----------------------------------------------------------------
参数量：202,653
  第   50 步  训练 loss = 0.540
  第  100 步  训练 loss = 0.493
  第  150 步  训练 loss = 0.484
  第  200 步  训练 loss = 0.484
  验证 loss = 0.477
```

*(Model: 202,653 parameters. Training loss after 50/100/150/200 steps, then validation
loss 0.477.)*

The second is 12 layers with the learning rate raised to 0.01:

```text
二、同一个模型，12 层，学习率 0.01
----------------------------------------------------------------
参数量：600,477
  第   50 步  训练 loss = 0.540
  第  100 步  训练 loss = 0.500
  第  150 步  训练 loss = 0.480
  第  200 步  训练 loss = 0.492
  验证 loss = 0.480
  最底下一层的梯度 = 8.7e-03
```

*(600,477 parameters, same reporting, plus the gradient of the bottom layer: 8.7e-03.)*

**It trained.** 12 layers, a big learning rate, no problem at all.

That is not a given. When we wrote LayerNorm back in Chapter 18, the order was "normalize
first → then into the sublayer → then add back onto itself" (this is called pre-LN). At
the time we only wanted to "hold the scale down", and had no idea that this order would
one day decide whether a 12-layer network can be trained at all. The third configuration
is the counterexample:

```text
三、把归一化挪到残差外面（另一种写法）
----------------------------------------------------------------
  第   50 步  训练 loss = 3.173
  第  100 步  训练 loss = 3.165
  第  150 步  训练 loss = 3.170
  第  200 步  训练 loss = 3.172
  验证 loss = 3.187   （瞎猜是 3.37）
  最底下一层的梯度 = 5.1e-14
```

*(Same model, same data, same learning rate, only the normalization moved outside the
residual. The loss sits at 3.17 from start to finish, against 3.37 for guessing blindly,
and the gradient reaching the bottom layer is 5.1e-14 — essentially nothing.)*

Same model, same data, same learning rate, and all we did was move the normalization
outside the residual (compute the sublayer first, add the residual, then normalize —
that is how the original Transformer paper wrote it, and it is an easy order to produce
when you implement it yourself), and it sits at "only knows character frequencies" from
beginning to end.

So this chapter's question splits in two:

1. How much is each of the things still standing between us and GPT worth?
2. And how much did that "normalize first, then compute" from Chapter 18 actually buy us?

## Experiment

(Everything below is real output. **The numbers — loss, parameter counts — are fixed**,
because we pinned the random seeds. **Times vary with the machine**, so what you get may
not match what is here.)

`experiment.py` runs three experiments.

### 1. Shallow + normal learning rate: swapping makes no difference

```text
配置                                        训练 loss   验证 loss   参数量
--------------------------------------------------------------------------
我们的全部选择（ReLU + 固定位置 + 输出层前不加 LN）0.495       0.478       202,653
只换成 GELU                                 0.497       0.476       202,653
只换成可学习的位置表                        0.506       0.482       205,725
只加上输出层前那一次 LN                     0.494       0.477       202,781
GPT 的全部选择                              0.495       0.480       205,853
```

*(Five configurations, each with its training loss, validation loss, and parameter count:
all of our choices, then only GELU swapped in, then only a learnable position table, then
only the extra LayerNorm before the output layer, then all of GPT's choices. Training
loss lands between 0.494 and 0.506; validation loss between 0.476 and 0.482.)*

The five numbers are virtually identical.

That is the first conclusion: **those three things are not "magic modules".** Shallow,
with a normal learning rate, you cannot tell the difference.

### 2. Deep + big learning rate: still the same, except for that one thing

```text
配置                                    第 1/4    第 2/4    第 3/4    第 4/4    验证 loss
--------------------------------------------------------------------------
我们的全部选择（= 第 22 章的模型）      0.562     0.514     0.542     0.499     0.492
我们的 + GELU + 可学习位置 + 最后那次 LN0.546     0.491     0.490     0.502     0.480
归一化挪到残差外面（post-LN）           3.192     3.177     3.174     3.163     3.181
```

*(Each configuration's training loss after each quarter of training, then validation
loss. The first two rows land at 0.48–0.49; the post-LN row never leaves 3.17.)*

The first two rows sit right next to each other (0.48 – 0.49, a difference inside the
noise), and the third row stays at 3.17 from beginning to end.

**Whether those three things are swapped or not, we tie with GPT. What actually decides
whether the thing can be trained is where the normalization goes** — and that is the one
item we got right back in Chapter 18.

### 3. Why that happens: how much gradient survives on the way to the bottom

```text
配置                              第 1 步     第 10 步    第 20 步    第 30 步
--------------------------------------------------------------------------
我们的写法（先归一化再算）        1.1e-01     1.5e-01     4.3e-02     3.3e-02     
post-LN（先算再归一化）           5.0e-02     6.9e-09     7.4e-12     4.5e-13     

再看一眼每个 block 拿到的梯度（12 层，训练 30 步之后）：

配置                              第 1 层     第 4 层     第 8 层     第 12 层
--------------------------------------------------------------------------
我们的写法（先归一化再算）        2.4e-01     1.3e-01     1.2e-01     1.9e-01     
post-LN（先算再归一化）           5.1e-12     3.0e-10     1.1e-06     7.0e-02     
```

*(Top table: the gradient reaching the bottom layer at steps 1, 10, 20, and 30. Our
ordering holds at 1.1e-01 down to 3.3e-02; post-LN collapses from 5.0e-02 to 4.5e-13.
Bottom table: the gradient each block receives after 30 steps, for layers 1, 4, 8, and
12. Ours stays around 1e-01 at every depth; post-LN is 5.1e-12 in the bottom layer and
only 7.0e-02 in the top one.)*

(Layer 1 is the bottom layer; layer 12 is right next to the output.)

The post-LN row: the bottom-layer gradient falls from `5.0e-02` to `4.5e-13` (more than a
dozen orders of magnitude), the bottom layers receive no signal at all, and only the top
few layers are still moving.

**It isn't "not smart enough" — it's "can't be trained".** And on this one item, we got it
right in Chapter 18.

### The item-by-item comparison table

The table `after.py` prints is the most important thing in this chapter:

```text
零件              我们哪一章写的    我们的做法                GPT 的做法
------------------------------------------------------------------------------
查表（词向量）    第 10 章          nn.Embedding 查表         一样
位置信息          第 16 章          固定的正弦表              换成一张可学习的表
Q / K / V         第 14 章          三个投影 + 打分 + softmax 一样
多头              第 15 章          拆成几个头分别算          一样
因果掩码          第 21 章          masked_fill(-inf)         一样
残差连接          第 17 章          x + 子层(x)               一样
归一化的位置      第 18 章          先归一化，再进子层        一样
MLP               第 19 章          升 4 倍宽 -> 激活 -> 降回来一样
激活函数          第 8 章           ReLU                      换成 GELU
输出层前的 LN     第 22 章          没有（直接进输出层）      有一次
输出层            第 20 章          D 维 -> 词表大小的分数    一样
训练目标          第 20 章          预测下一个 token          一样

12 项里有 9 项写着'一样'—— 包括第 18 章那个'归一化放在子层前面'，
我们第 18 章就是这么写的，第 22 章原样用，GPT 也是这么写的。
真正换掉的只有三件事：位置表、激活函数、输出层前那一次归一化。
（GPT-2 在注意力的投影上还带偏置，我们第 22 章那一版没带 —— 那是无关紧要的细节，
  一个偏置加不加，不影响任何结论。后面几章为了和 GPT 一致，又把它加回去了。）
```

*(Twelve rows, each naming a part, the chapter where we wrote it, what we do, and what GPT
does. Nine of the twelve say "same" — including the normalization placement, which both
we and GPT put before the sublayer. Only three things are actually swapped: the position
table, the activation function, and the extra normalization before the output layer. GPT-2
also carries a bias on the attention projections while our Chapter 22 version doesn't;
the block itself calls that an irrelevant detail.)*

Nine of the twelve items say "same" — including that "normalization goes before the
sublayer" from Chapter 18, which we wrote that way in Chapter 18, used unchanged in
Chapter 22, and which GPT also writes that way. Only three things are really swapped: the
position table, the activation function, and that one normalization before the output
layer. (GPT-2 also keeps a bias on the attention projections, and our Chapter 22 version
didn't — that is an irrelevant detail; whether one bias is there or not changes no
conclusion. In the chapters ahead, to line up with GPT, we put it back.)

The second conclusion: **there is no magic module in GPT that appeared from nowhere.**
The difference between it and our model is three engineering choices plus scale.

## The new mechanism

This chapter has no new **model parts**. What is new is three "choices", and what each of
them means.

**Choice one: position information becomes a learnable table.**

In Chapter 16 we hand-wrote a sinusoidal table, where the vector for position *i* is
computed by a fixed formula. GPT's approach is lazier: **treat position as a word too**,
build a `(max_length, dim)` table, and let the model learn it.

```python
self.pos_emb = nn.Embedding(block_size, dim)   # position is a table lookup too
...
x = self.tok_emb(idx) + self.pos_emb(torch.arange(length))
```

With that, position encoding has no "design" left in it — like a word embedding, it is a
trained parameter. The cost: the **longest length** the model can handle is fixed before
training (it can only see as far as the table is big), whereas a sinusoidal table can
extrapolate. From our experiments, the two approaches can't be told apart on loss; GPT
chose learnable mainly because it is less work and scales up more easily.

**Choice two: use GELU rather than ReLU for the activation.**

When we introduced ReLU in Chapter 8, its virtue was "squash negatives to 0, let positives
through unchanged" — simple, easy to differentiate. GELU looks like it, but **negatives
are not squashed straight to 0**; instead they approach 0 along a smooth curve:

```text
x = -3.0  ->  ReLU 0.0     GELU -0.004
x = -1.0  ->  ReLU 0.0     GELU -0.159
x =  1.0  ->  ReLU 1.0     GELU  0.841
x =  3.0  ->  ReLU 3.0     GELU  2.996
```

*(ReLU flattens every negative input to 0; GELU lets a negative through as a small
negative, and leaves positives almost unchanged.)*

The intuitive benefit: a negative neuron still keeps a thin path "backward", so it doesn't
die completely once it falls into the negative region the way ReLU does. At our scale, the
difference from ReLU is invisible (experiment 1); GPT picked it because it is
statistically more stable at large scale.

**Choice three: one more LayerNorm before the output layer.**

In Chapter 22, after the last block we went straight into the output layer. GPT adds one
more LayerNorm at the very end:

```python
if final_ln:
    x = self.ln_f(x)     # before the output layer, press the scale down once more
return self.head(x)
```

Why is it needed? Because the pre-LN residual stream **accumulates all the way down**
(every block adds its result back onto the trunk), and the more layers there are, the
bigger the scale of that final vector. An output layer fed a vector whose scale has run
away trains unstably. One normalization presses it back to a normal scale before
classification.

**One last thing: decoder-only.**

The original Transformer paper had two halves: an encoder (which can see the whole
sentence) and a decoder (which can only see what came before). The causal mask we wrote
in Chapter 21 is the decoder half. GPT has only that half — the structure later came to
be called **decoder-only**. We and GPT are completely in agreement on this item, which is
why the comparison table says "same".

**And one detail we left alone**: GPT-2's output layer has no bias, and its attention
projections do have one; we have it the other way round. These are genuinely irrelevant
details — whether one bias is added or not changes none of the conclusions above. In this
book we always "write what we have written", and only line up with GPT where it actually
means something.

## Python implementation

What `after.py` implements is **the same code plus four switches**:

```python
GPTModel(vocab_size, ..., learned_pos=True, pre_ln=True, final_ln=True,
         activation=torch.nn.functional.gelu)
```

- All four switches off = the Chapter 22 model (the one moved into PyTorch in Chapter 25);
- Three switches on (`learned_pos`, `final_ln`, `activation=gelu`) = GPT's choices;
- `pre_ln=False` is the version neither model uses, kept around as the counterexample for
  the experiments.

Both orderings of the block live in these dozen lines:

```python
def forward(self, x, mask):
    if self.pre_ln:
        # normalize first, then into the sublayer, then add back onto the residual
        # -- this is how we wrote it in Chapter 18
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.W2(self.activation(self.W1(self.ln2(x))))
    else:
        # the other way: compute first, add the residual, then normalize
        x = self.ln1(x + self.attention(x, mask))
        x = self.ln2(x + self.W2(self.activation(self.W1(x))))
    return x
```

After the swap, the same 12-layer model at learning rate 0.01 runs like this:

```text
  第   50 步  训练 loss = 0.526
  第  100 步  训练 loss = 0.485
  第  150 步  训练 loss = 0.492
  第  200 步  训练 loss = 0.484
  验证 loss = 0.477
```

*(Training loss at steps 50/100/150/200: 0.526, 0.485, 0.492, 0.484; validation loss
0.477.)*

You can see what it learned, too:

```text
  苹果 -> 苹果很好吃。香蕉很好吃。香蕉真甜。这个苹果很
  华为 -> 华为电脑很强。苹果真甜。苹果真甜。香蕉很好吃

话题切换它也会：
  苹果很甜。华为 -> 苹果很甜。华为发布新电脑。这个苹果芯片很强。小米发
  这个苹果做成派。小米 -> 这个苹果做成派。小米发布新电脑。这个苹果很好吃。苹果做成
```

*(Two prompts and their continuations. Given `苹果` "apple" it produces food-topic
sentences; given `华为` "Huawei" it produces tech-topic ones. The lower pair shows topic
switching: `苹果很甜。华为` "apples are sweet. Huawei" continues into `发布新电脑`
"shipped a new computer", and `这个苹果做成派。小米` "this apple made into a pie. Xiaomi"
continues into `发布新电脑` as well.)*

Give it `苹果很甜。` ("apples are sweet." — a food topic) followed by `华为`, and it
switches to `发布新电脑` ("shipped a new computer"); give it `这个苹果做成派。` followed
by `小米`, and it switches back. **It really is following the topic of the preceding
text**, rather than reciting sentences.

## What it solves

- Our model stacked to 12 layers at learning rate 0.01 gets validation loss `0.480`;
  swapping in GPT's three things gives `0.477`. **At the same scale the two can't be told
  apart** (we have been verifying this from Chapter 25 through Chapter 27).
- What does tell them apart is where the normalization goes: the same 12 layers under
  post-LN give a validation loss of `3.187` (blind guessing is 3.37) and a bottom-layer
  gradient of `5.1e-14`, where our order gives `8.7e-03`. **That is the difference between
  "can train" and "can't train", and we chose right back in Chapter 18.**
- Nine of the twelve parts in the comparison are exactly the same. The three that were
  swapped are engineering details that make no visible difference on a shallow network
  (experiment 1).

At this point, "our model is only small" can be revised:

> GPT = the parts we already wrote (normalization placement included)
> + three engineering choices that change no conclusion
> + vastly more scale.

The first two halves we now have completely. What is left is that last word.

## What it still can't solve

Now look at the last table `after.py` prints:

```text
模型            层数      维度      参数量          训练数据
------------------------------------------------------------------------------
我们的模型      12        64        603,677         4 万字
GPT-2 small     12        768       124,000,000     约 100 亿字
GPT-2 medium    24        1024      355,000,000     约 100 亿字
GPT-2 large     36        1280      774,000,000     约 100 亿字
GPT-2 XL        48        1600      1,500,000,000   约 100 亿字
GPT-3           96        12288     175,000,000,000 约 3000 亿字
```

*(Rows for our model, GPT-2 small/medium/large/XL, and GPT-3: layers, dimension, parameter
count, and training data. Our model: 12 layers, 64 wide, 603,677 parameters, 40,000
characters. GPT-2 small: 12 layers, 768 wide, 124 million parameters, about 10 billion
characters. GPT-3: 96 layers, 12288 wide, 175 billion parameters, about 300 billion
characters.)*

Our 12 layers and GPT-2 small's 12 layers are the same layer count, with a 12-fold
difference in dimension and a 205-fold difference in parameters.

At this point a very natural question comes up: **our model already learns this corpus to
a loss of 0.48. What do we need all that size for?**

Put another way: if I can press the loss down to 0.48 with 600,000 parameters, then what
do I actually get for turning those 600,000 into 175 billion? What are the extra
parameters doing?

## Exercises

See `exercises.en.md`; here are the 3 most important ones:

1. In experiment 2 of `experiment.py`, change the layer count from 12 to 4 and to 24 and
   run each. At 4 layers, can post-LN still learn? (Guess first, then run it.)
2. Add a learning-rate warmup to the post-LN model (raise the learning rate linearly from
   0 to 0.01 over the first 50 steps) and see whether it can be rescued. What does that
   tell you about the relationship between "the learning rate is too big" and "the
   normalization is in the wrong place"?
3. Open `after.py`, change `final_ln` from `True` to `False` (drop GPT's extra
   normalization) and set the layer count to 24. What does the loss become? That explains
   why GPT adds that normalization.
