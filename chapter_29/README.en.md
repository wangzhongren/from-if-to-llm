**English** | [中文](README.md)

# Chapter 29: From a language model to ChatGPT

## The problem in this chapter

The last chapter ended with a sentence: **being able to predict the next word is not the
same as being able to chat.**

That sentence is not a figure of speech. We train a language model down to loss 0.46, and
it predicts the next character very accurately. Then you open the chat box and ask it:

```text
今天天气怎么样？
```

*(`今天天气怎么样？` — "How's the weather today?")*

It will not answer you. It will keep writing — because in its training data, what follows
the sentence `今天天气怎么样？` is usually another sentence, not an answer.

Let's make this concrete in code first.

## The simplest attempt

The two most intuitive fixes: **train longer**, or **make the model bigger**.

But before rushing to do either, ask a question: does our model actually "know" that
apples are sweet?

`before.py` probes for exactly that. The same fact, asked two ways:

```text
同一个事实，我们用陈述句问它，它接得又快又准：
  苹果很 -> 甜。华为发布新电脑。

用问句问它，它就不会了：
  问：苹果甜吗？答：这个苹果很甜。香蕉很甜。华为发布
```

*(Given the statement `苹果很` "apples are very", it continues immediately and correctly
with `甜` "sweet". Given the question `苹果甜吗？` "are apples sweet?", what it produces
after `答：` "A:" has nothing to do with answering — it wanders off into nearby
sentences.)*

Do you see it? It **knows** that "apples are sweet" — give it `苹果很` and it immediately
continues `甜`. Even when asked a question, what it produces still mentions `这个苹果很甜`
("this apple is very sweet").

But it is not **answering**. It is **continuing**:

```text
  问：苹果甜吗？答：问：这个苹果甜吗？答：苹果很好吃
  问：苹果好吃吗？答：问：华为发布新电脑吗？答：香蕉很
```

*(Asked `苹果甜吗？` "are apples sweet?", it produces `问：这个苹果甜吗？答：苹果很好吃` —
that is, another question and then a different sentence. Asked `苹果好吃吗？` "are apples
delicious?", it answers with a question about Huawei and then `香蕉很` "bananas are
very".)*

It isn't answering wrong. It isn't answering at all. Because in its training corpus, a
question is never followed by an answer — it is followed by another question, or by some
chitchat. It learned that too.

**So what it lacks is not parameters, and not data volume. It lacks a new data format.**

## Experiment

`after.py` runs a three-stage comparison. The same model (108,032 parameters, not one part
changed), fed three kinds of data in turn.

(Everything below is real output. **The numbers — loss, accuracy, parameter counts — are
fixed**, because we pinned the random seeds. **Times vary with the machine**, so what you
get may not match what is here.)

### Stage one: pretraining (predict the next character)

The data is "a pile of unedited text": statements, questions, and chitchat mixed together,
**with questions but no answers**.

```text
  第 200 步  loss = 0.406
  第 400 步  loss = 0.357
  第 600 步  loss = 0.361
用时 2 秒

现在拿一个问题去问它：问：苹果甜吗？答：
它续写出来的东西是：
  问：香蕉甜吗？答：问：这个苹果甜吗？答：
  小米发布新手机。华为发布新电脑。小王在看
```

*(After 600 steps of pretraining the loss is 0.361. Asked a question, the model does not
answer it — it produces more questions and unrelated sentences.)*

### Stage two: instruction tuning (loss computed only on the "answer")

The data becomes (question, answer) pairs: `问：苹果甜吗？答：甜。<结束>` ("Q: are apples
sweet? A: sweet.<end>").

The training objective barely changed — still predicting the next character, still
cross-entropy. The only difference is that **the prompt part (`问：……答：`) doesn't count
toward the loss**; only the characters of the answer do.

```text
  第 100 步  loss = 0.079（只在答案那几个字上算）
  第 200 步  loss = 0.074
  第 300 步  loss = 0.067
用时 1 秒

同样的问题，再问一次：
  问：苹果甜吗？答：甜。[结束]
  问：苹果好吃吗？答：苹果好吃吗？好吃。[结束]
  问：香蕉甜吗？答：这个问题问得好。香蕉甜。[结束]
  问：香蕉好吃吗？答：香蕉好吃吗？好吃。[结束]
```

*(300 steps of instruction tuning, loss computed only on the answer characters. Asked the
same question again, it now answers `甜。[结束]` "sweet.[end]" — and it also answers other
questions, though in two different styles, one of which echoes the question back before
answering.)*

It can answer now, and it says "end" when it is done (`<结束>` is a special token we added
— it never appeared in the pretraining corpus, and the model learned "when I'm finished I
should stop" only at this stage).

Notice the two styles in there: `甜。[结束]` and `苹果甜吗？好吃。[结束]` ("are apples
sweet? delicious.[end]"). That is deliberate: the fine-tuning data we fed it contains both,
just as real annotators each have their own way of writing. After this stage the model
knows both, but **it doesn't yet "prefer" either**.

### Stage three: preference alignment (make the favoured answers more probable)

The data becomes triples: (question, good answer, bad answer).

```text
回答                        类型      对数概率    概率
----------------------------------------------------------------------
甜。[结束]                  被选中    -0.71       0.491
苹果甜吗？甜。[结束]        被拒绝    -0.75       0.472
```

*(Two candidate answers to `苹果甜吗？` — `甜。[结束]` is the chosen one, `苹果甜吗？甜。
[结束]` the rejected one — with their log-probabilities and probabilities: 0.491 against
0.472. Nearly a coin flip.)*

The two answers currently have almost the same probability (0.49 against 0.47). Run 100
steps of preference alignment:

```text
  第  50 步  偏好 loss = 0.6737
  第 100 步  偏好 loss = 0.5214
用时 0 秒

回答                        类型      调之前          调之后
----------------------------------------------------------------------
甜。[结束]                  被选中    0.491           0.968
苹果甜吗？甜。[结束]        被拒绝    0.472           0.011
```

*(After 100 steps of preference alignment the chosen answer has gone from 0.491 to 0.968
and the rejected one from 0.472 to 0.011.)*

The chosen one went up, the rejected one went down.
Not one word in the data mentions "be concise", and yet the model's habits were changed by
this preference table:

```text
同一批问题，用同一个随机种子再生成一遍：
  问：苹果甜吗？答：甜。[结束]
  问：苹果好吃吗？答：好吃。[结束]
  问：香蕉甜吗？答：甜。[结束]
  问：香蕉好吃吗？答：好吃。[结束]
```

*(Regenerating from the same questions with the same random seed, every answer is now the
short form.)*

### Three rulers measuring three branches

`experiment.py` copies the same pretrained model several times, feeds each copy different
data, and measures with three rulers:

```text
模型                      答对率      原始文本 loss   短答比例
--------------------------------------------------------------------------
只做了预训练              0/7         0.396           0%
继续喂原始文本 300 步     0/7         0.412           0%
喂指令数据 300 步         5/7         2.349           46%
指令数据 + 偏好对齐       7/7         2.592           100%
```

*(Rows: pretraining only; continuing to feed it raw text for 300 steps; instruction data
for 300 steps; instruction data plus preference alignment. Columns: answer accuracy,
loss on raw text, and the share of short answers. Accuracy goes 0/7 → 0/7 → 5/7 → 7/7;
raw-text loss goes 0.396 → 0.412 → 2.349 → 2.592; short-answer share goes 0% → 0% → 46% →
100%.)*

Each of the three numbers says one thing:

**One: answer accuracy.** However long you pretrain, it gets none of them right; switch to
instruction data and 300 steps gets 5 right, and alignment gets all 7. Same model, same
step count, roughly the same number of characters — what differs is the **shape** of the
data.

**Two: loss on raw text.** It goes from 0.40 up to 2.35. It has almost completely
forgotten "how to continue text". This is the extreme version of the real-world "alignment
tax": our fine-tuning data is far too little and the steps far too many, so a few hundred
steps wash out the original skill. Real models have far more data and don't regress this
hard — but "learned to obey, forgot everything else" is happening all the time.

**Three: the share of short answers.** Before preference alignment, blunt answers and
roundabout ones split the field evenly; after alignment, "answer directly" becomes the
default choice. And we never wrote a rule saying "be concise".

## The new mechanism

What is new in this chapter **is not in the model. It is in the data.**

The model has not changed since Chapter 25 (that line in `after.py` is true: not one part
was touched). What changed is what the text we feed it looks like, and **at which
positions the loss is computed**.

The three stages, and what each one is trying to do:

| Stage | What the data looks like | Training objective | Is the loss still cross-entropy? |
|---|---|---|---|
| **Pre-training** | a pile of unedited text | predict the next character at every position | yes |
| **Instruction tuning** (also called SFT) | (question, answer) pairs | predict the next character **only on the characters of the answer** | yes (the positions changed) |
| **Preference alignment** | (question, good answer, bad answer) triples | make the "good answer" more probable than the "bad answer" | no, but there is nothing new in it |

That table is the whole chapter. Let's take them one at a time.

**One: pretraining — the objective is the one we have been using all along.**

The Chapter 20 objective: "given the preceding characters, predict the next character."
Pretraining's relationship to it is: **the same objective, with an endless supply of text
fed to it**. What it learns is the regularity of the language and facts about the world,
but the "way of replying" it learns is counted out of the text — and on the internet, a
question is usually not followed by an answer.

**Two: instruction tuning — the objective is unchanged; what changes is which positions
count toward the loss.**

This is the step most worth stopping to look at in the whole book. The instruction-tuning
loss is:

```python
per_token = cross_entropy(logits, targets, reduction="none")
loss = (per_token * mask).sum() / mask.sum()
```

`mask` is 1 only on the characters of the answer. **The prompt part gets multiplied by 0.**

In other words: we are not teaching it new knowledge (it has known "apples are sweet" for
a long time); we are telling it "in this format, you should spend your effort producing an
answer". Put in one sentence:

> Pretraining teaches it **what the world is like**; instruction tuning teaches it **what
> to do on this kind of occasion**.

One detail while we're here: that `<结束>` token. It is an ordinary token, it just never
appeared in the pretraining corpus — so the model learned "when you're done talking, stop"
**only during fine-tuning**. Those special tokens in ChatGPT (`<|im_start|>`,
`<|endoftext|>`) are there for exactly this.

**Three: preference alignment — the objective changes for the first time.**

In the first two stages, the model is predicting the next character. At this step we stop
asking "what is the next character" and start asking:

> **Of these two answers, which one does a person prefer?**

This is the first time in the whole book that the training objective changes **in kind**.
Its implementation (a simplified version of DPO, which is what we use) is:

```python
chosen_ratio  = log P_new(good answer) - log P_old(good answer)
rejected_ratio = log P_new(bad answer) - log P_old(bad answer)
loss = -log sigmoid(β * (chosen_ratio - rejected_ratio))
```

All it asks for is one thing: **make the "good answer" more probable than the "bad
answer"** — and measured against the version of itself from **before** the preference
tuning (that old model is called the **reference model**).

Why do we need that old self? Because if you only push the good answer up, the model piles
all its probability into "the shortest, most repetitive" direction and ends up ruining its
speech (we tried it: with a slightly larger learning rate the good answer's probability
shoots from 0.49 to 0, and both answers are ruined together). Using "the former self" as a
ruler makes it push on "which one is better", not on "which one has higher probability".

Also notice: the data used at this step is **comparison**, not correct answers. If we could
train on correct answers, we wouldn't need preference alignment at all — preference
alignment exists precisely because "is this answer good" very often **has no single correct
answer**, only "which one do people prefer". That is the step from "predict what" to
"predict what people want".

## Python implementation

Three functions in `after.py`, one per stage. Read them side by side and the difference is
obvious.

**Pretraining**:

```python
def pretrain_step(model, optimizer, batch_size, rng):
    x, y = slice a random piece of the corpus      # (the Chapter 20 thing)
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
    ...
```

**Instruction tuning** (only two extra lines):

```python
def sft_step(model, optimizer, batch_size, rng, block_size=32):
    pairs = build_sft_pairs()
    chosen = draw a batch of (question, answer) pairs at random
    inputs, targets, masks = pad_sequences(make_sequences(chosen, block_size), block_size)
    logits = model(inputs)
    per_token = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                               targets.reshape(-1), reduction="none")
    per_token = per_token.reshape(targets.shape)
    loss = (per_token * masks).sum() / masks.sum()      # <- only this line is different
```

`make_sequences` is what produces that mask:

```python
mask = [1 if position >= len(prompt) - 1 else 0 for position in range(len(inputs))]
```

Position `p` predicts character `p+1`, so the position corresponding to "the first
character of the answer" is `len(prompt) - 1`. From there onward is the part to be learned.

**Preference alignment** (the formula is longer, but what it does is simple):

```python
def sequence_logprob(model, prompt, answer, differentiable=False):
    """How much the model scored this answer: the log of each character's probability, summed."""
    ids = encode(prompt + answer)
    logprobs = model(torch.tensor([ids]))[0].log_softmax(dim=-1)
    return sum(logprobs[position, ids[position + 1]]
               for position in range(len(prompt) - 1, len(ids) - 1))
```

```python
def dpo_step(policy, reference, optimizer, rng, beta=0.1):
    prompt, chosen, rejected = draw one preference example at random
    chosen_ratio = sequence_logprob(policy, prompt, chosen, differentiable=True) \
        - sequence_logprob(reference, prompt, chosen)
    rejected_ratio = sequence_logprob(policy, prompt, rejected, differentiable=True) \
        - sequence_logprob(reference, prompt, rejected)
    loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Note the number `beta` (we use 0.1): it controls "how hard the push is". The bigger `beta`
is, the more single-mindedly it obeys the preference data; the smaller it is, the more
reluctant it is to leave its former self.

One engineering detail is worth a look too: **the learning rate for preference alignment
has to be much smaller than in the first two stages** (we use `1e-4`, while pretraining and
fine-tuning both use `3e-3`). The gradients at this stage are very "sharp", and push too
hard and the model breaks.

## What it solves

- With the same model, **answer accuracy goes from 0/7 to 5/7 to 7/7**, with not one part
  of the model changed.
- **The share of short answers goes from 0% to 46% to 100%** — preference alignment really
  did change the model's habits, and not one line of the data contains a rule saying "be
  concise".
- For the first time we see that **the training objective can change from "predict the next
  character" to "predict which answer people prefer"**, and the latter is not
  cross-entropy.
- And we saw the price: raw-text loss regresses from 0.40 to 2.35. Alignment has a cost.

Now let's state clearly what the three stages have in common:

> In all three stages, the model "sees a piece of text and gives the probability of the
> next token".
> All that changes is **what text we put in front of it**, and **at which positions we hold
> it responsible**.

That also explains something many people are confused by: why do prompts work?
Because what the model learned, from beginning to end, is "given text like this, what comes
next". Give it different text and the "what comes next" it produces is different. **A
prompt is changing its input, not changing its knowledge.**

## What it still can't solve

From Chapter 1 to this chapter we have walked 29 chapters:

```text
if/else 规则  →  数字化的规则  →  让机器学习规则  →  学特征  →  学词表示
→  学上下文  →  学信息关系  →  深层组合  →  预测下一个 token  →  回答问题
```

*(The chain of the book so far: if/else rules → digitised rules → let the machine learn
the rules → learn features → learn word representations → learn context → learn information
relationships → deep composition → predict the next token → answer questions.)*

With the last three steps (tokenization, scale, alignment) done, the thing in our hands is
already shaped like ChatGPT.

But there is one thing we have never done: **go back and look at where we started.**

In Chapter 2 we wrote an if/else classifier that used a pile of rules to classify
sentences:

```python
if "手机" in words:
    return "科技"
```

That was the first line of code in the whole book. At the time we said it doesn't work —
are you going to write tens of thousands of ifs for tens of thousands of words? So we
replaced it, step by step, with numbers, weights, gradients, layers, attention, a language
model.

Now take it out and **put it on the same test** as the model we have after 29 chapters:

- the same set of sentences;
- the same judgement (is this sentence tech or food);
- and see what each of them says.

We're not comparing parameter counts (one has 0 parameters, the other 100,000), and we're
not comparing speed. What we're comparing is exactly this: **what 30 chapters of distance
adds up to.**

That is what the last chapter is for.

## Exercises

See `exercises.en.md`; here are the 3 most important ones:

1. Change the instruction-tuning steps in `after.py` from 300 to 30 and to 3000, and re-run.
   How do answer accuracy and raw-text loss move? What is the relationship between the
   "alignment tax" and the number of training steps?
2. In `FACTS`, replace the "verbose answer" with another style you dislike (for instance
   `这个我不知道。`, "I don't know that."), and re-run stage three. Can the model learn
   this kind of preference judgement?
3. Add a `top_k` to `generate` (we wrote one in Chapter 23) and run generation again after
   stage three. With top-k added, do those low-probability "verbose answers" still show up?
   Why?
