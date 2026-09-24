**English** | [中文](README.md)

# Chapter 30: Back to the first line of code

## The problem in this chapter

This is the last chapter of the book. By this book's rules, every chapter starts from the
question the previous chapter left behind — but this chapter leaves no question, so it has
only one thing to do: **go back and look at where we started.**

The first line of code in the whole book was written in Chapter 2:

```python
if "手机" in words:
    print("科技")
```

*(If the word `手机` "phone" is in the sentence, print `科技` "tech".)*

Back then our question was: "tens of thousands of words — are you going to write tens of
thousands of ifs?"

So we walked on: turn the ifs into numbers, let the program adjust the weights itself, add
an intermediate layer, add an activation function, write our own automatic
differentiation, let words become vectors, average the context, attention, Q/K/V,
multi-head, positional encoding, residuals, LayerNorm, MLP, Transformer block, causal
mask, language model, autoregressive generation, swap the engine, real tokenization, GPT's
choices, scale, alignment…

This chapter puts the two ends together: **the if/else classifier from Chapter 2, and the
model we have after Chapter 29, doing the same task.**

## The simplest attempt

The task is still the Chapter 1 one: classify a sentence (tech / food).

**The Chapter 2 side** (`before.py`): split into words, then a chain of ifs.

```python
TECH_WORDS = ["手机", "芯片", "电脑", "发布", "华为", "小米"]
FOOD_WORDS = ["好吃", "很甜", "甜", "香蕉"]

def classify(words):
    for word in words:
        if word in TECH_WORDS:
            return "科技"
        if word in FOOD_WORDS:
            return "食品"
    return "食品"      # hit no keyword at all, so guess food first
```

*(The keyword lists read: phone, chip, computer, launch, Huawei, Xiaomi for tech; delicious,
very sweet, sweet, banana for food. The labels are `科技` "tech" and `食品` "food".)*

**The Chapter 30 side** (`after.py`): write the corpus as "sentence + category", then let
the language model predict the probability of the two characters of the category.

```text
苹果发布新手机分类：科技。
苹果很甜分类：食品。
```

*(Two training lines: `苹果发布新手机分类：科技。` "Apple shipped a new phone; category:
tech." and `苹果很甜分类：食品。` "apples are very sweet; category: food.")*

At test time we treat `句子分类：` ("sentence; category:") as the prompt and see whether
`科技` ("tech") or `食品` ("food") has the higher probability after it — that is the
Chapter 20 sentence: **classification is next-token prediction.**

## Experiment

(Everything below is real output. **The numbers — loss, accuracy, parameter counts — are
fixed**, because we pinned the random seeds. **Times vary with the machine**, so what you
get may not match what is here.)

### The same set of sentences, two report cards

```text
句子                if/else 说  模型说    模型的把握    人给的标签
------------------------------------------------------------------------------
苹果发布新手机      科技        科技      0.999         科技
苹果发布新芯片      科技        科技      1.000         科技
华为发布新电脑      科技        科技      0.998         科技
小米发布新手机      科技        科技      0.999         科技
苹果芯片很强        科技        科技      1.000         科技
苹果很好吃          食品        食品      1.000         食品
苹果很甜            食品        食品      1.000         食品
香蕉很好吃          食品        食品      1.000         食品
这个苹果真甜        食品        食品      1.000         食品
苹果做成派          食品        食品      0.999         食品
小米直播带货        科技        科技      0.987         科技
华为发布新平板      科技        科技      0.975         科技
苹果不好吃          食品        食品      1.000         食品
香蕉做成派很好吃    食品        食品      1.000         食品
苹果手机很甜        科技        食品      0.971         说不清（有歧义）
这个手机真好吃      科技        食品      1.000         说不清（有歧义）
------------------------------------------------------------------------------
if/else 分类器：14/14
我们的模型：    14/14
```

Each row is a sentence, then what the if/else classifier says, what the model says, the
model's confidence, and the label a person gave it. The sentences, in order:

- Tech-topic: `苹果发布新手机` "Apple shipped a new phone", `苹果发布新芯片` "Apple shipped a
  new chip", `华为发布新电脑` "Huawei shipped a new computer", `小米发布新手机` "Xiaomi
  shipped a new phone", `苹果芯片很强` "the Apple chip is very strong".
- Food-topic: `苹果很好吃` "apples are delicious", `苹果很甜` "apples are very sweet",
  `香蕉很好吃` "bananas are delicious", `这个苹果真甜` "this apple is really sweet",
  `苹果做成派` "apples made into a pie".
- Then four more that are new to this test: `小米直播带货` "Xiaomi livestream selling" (tech),
  `华为发布新平板` "Huawei shipped a new tablet" (tech), `苹果不好吃` "apples are not
  delicious" (food), `香蕉做成派很好吃` "bananas made into a pie are delicious" (food).

**The last two rows are the only place where the two sides disagree, and both are sentences
with no clean answer.** `苹果手机很甜` "the Apple phone is very sweet": the if/else
classifier sees `手机` and says tech, while the model says food with confidence 0.971.
`这个手机真好吃` "this phone is really delicious": the if/else classifier says tech, the
model says food with confidence 1.000. The right-hand column marks both of them
`说不清（有歧义）`, "can't be decided (ambiguous)". The two totals at the bottom are what
each side scored on the fourteen unambiguous sentences: 14 out of 14 each.

**A tie.**

The thing that took 30 chapters to build did not beat those 4 lines of if/else on this
test. That should be said plainly, and it is worth thinking about:

**This set of sentences was written to match those rules in the first place.** The rules
were written by a person, and the person who wrote them knew the answers; the model learned
from data, and the data was manufactured by us according to the rules. On its own exam
paper, of course the rules score full marks.

So the difference is not on this report card. The difference is in the two places below.

### One: delete a few words from the rules

Suppose that when writing the rules, we hadn't thought of the two brands `华为` and `小米`:

```text
规则表                    成绩
------------------------------------------------------------------------------
完整规则                  14/14
删掉'华为''小米'          13/14
再删掉'发布'              12/14
再把'甜'删掉              12/14
```

*(Rows: the complete rule table scores 14/14; deleting `华为` and `小米` gives 13/14;
deleting `发布` as well gives 12/14; deleting `甜` too still gives 12/14.)*

Only 1 point, then 2 points lost — which is not what the intuition "rules are fragile"
predicts. The reason is that our rules are **redundant**: in `小米发布新手机`, the three
words `小米`, `发布`, and `手机` are all in the tech word list, so deleting one still
leaves two to catch it. That is how people write rules: write a few extra, so they cover
for each other.

But look carefully at which sentence was lost: after deleting `小米`, `小米直播带货` falls
straight through to the last line, "hit no keyword at all, so guess food first".
**The knowledge in the rules is that word list and nothing more — not one word more.**

### Two: delete part of the model's training data

Do the same thing: delete every sentence containing `华为` or `小米` from the training
corpus, and retrain.

```text
模型                          完整数据    删掉'华为''小米'
------------------------------------------------------------------------------
苹果发布新手机                科技（1.00）科技（1.00）
华为发布新电脑                科技（1.00）科技（1.00）
小米发布新手机                科技（1.00）科技（1.00）
苹果很甜                      食品（1.00）食品（1.00）

完整数据训练：14/14
删掉两个品牌的句子之后：12/14
```

*(Rows: four test sentences, with the model's answer and confidence on the full data
versus after the two brands were deleted. Both give tech (1.00) for `苹果发布新手机`,
`华为发布新电脑`, and `小米发布新手机`, and food (1.00) for `苹果很甜`. Totals: 14/14 on
the full data, 12/14 after deleting the sentences with the two brands.)*

The model lost points too, but it did not collapse the way the rules did, and **it still
recognises `华为发布新电脑`** — because what it learned from the corpus is not just the two
words `华为` and `小米`, but also fragments like `发布`, `新`, `电脑` and the relationships
between them. **The knowledge in the rules is written in the rules; the knowledge in the
model is written in the parameters.** Delete the former and it is simply gone; the latter
is a single whole whose parts hold each other up.

### Three: the two approaches

```text
                    if/else 分类器（第 2 章） 我们的模型（第 30 章）
------------------------------------------------------------------------------
怎么来的            人写规则                  从数据里学
有多少个数字        4 条规则、10 个关键词     109,312 个参数
判断一句话要多久    几乎 0                    3 毫秒
为什么这么判        指着某一行 if 就能解释    只能看概率，说不清内部
换个任务            重写规则                  换一批数据，代码一行不改
没见过的词          词表外就没辙              按字处理，能靠上下文猜
想变得更准          再多想几条规则            再喂更多数据（第 28 章）
按什么分类          是 / 否                   一个概率（第 5 章）
```

*(A table comparing the if/else classifier against our model along seven rows: where it
comes from (rules written by people vs. learned from data); how many numbers it has (4
rules and 10 keywords vs. 109,312 parameters); how long one judgement takes (almost 0 vs.
3 milliseconds); why it decided that (point at an `if` line and explain vs. only
probabilities, with no account of its insides); **switching to another task (rewrite the
rules vs. swap in another dataset, without changing a line of code)**; words it has never
seen (stuck once outside the vocabulary vs. handled character by character, able to guess
from context); how to make it more accurate (think up a few more rules vs. feed it more
data, Chapter 28); and what it classifies by (yes/no vs. a probability, Chapter 5).)*

In that table, the row that matters most is "switching to another task". We already
measured it once in Chapter 29: **the same model code, fed three different datasets, became
three different things** (a language model that can only continue text, an assistant that
can answer questions, and a more obedient assistant). Not one line of code changed.

## The new mechanism

This chapter has no new mechanism.

It is the only chapter in the book that "doesn't move forward": it puts Chapter 1's first
line of code next to Chapter 29's model, and then compresses the whole book into one
diagram.

```text
人工写规则 → 规则数字化 → 让机器学习规则 → 学习特征 → 学习词表示
→ 学习上下文 → 学习信息关系 → 深层组合 → 预测下一个 token
```

*(The evolution: writing rules by hand → rules turned into numbers → letting the machine
learn the rules → learning features → learning word representations → learning context →
learning information relationships → deep composition → predicting the next token.)*

Every arrow in that diagram is the "what it still can't solve" at the end of some earlier
chapter:

| Arrow in the diagram | Chapter | The problem at the time |
|---|---|---|
| Writing rules by hand | Ch. 2 | classifying sentences with a pile of ifs |
| Rules turned into numbers | Ch. 3 | rewriting `if "手机" in words` as a weighted sum |
| Letting the machine learn the rules | Ch. 4–6 | who decides the weights? → loss and gradients |
| Learning features | Ch. 7–9 | one straight line isn't enough → an intermediate layer, an activation function, automatic differentiation |
| Learning word representations | Ch. 10–12 | an index means nothing → word vectors, context |
| Learning context | Ch. 13 | not every word matters equally → a weighted sum |
| Learning information relationships | Ch. 14–16 | pulling out Q/K/V, multi-head, positions |
| Deep composition | Ch. 17–19 | residuals, LayerNorm, MLP → the Transformer block |
| Predicting the next token | Ch. 20–24 | classification becomes a language model, causal masks, generation |
| (Scale and alignment) | Ch. 25–29 | swapping the engine, real tokenization, GPT's choices, getting bigger, alignment |

**And then the last lines of code in the whole book:**

```python
tokens = encode("床前明月光")
for _ in range(24):
    window = tokens[-model.block_size:]        # only look at the last few tokens
    logits = model(torch.tensor([window]))     # a stretch of tokens goes in
    next_token = sample(logits[0, -1])         # the next token comes out
    tokens.append(next_token)
```

`after.py` prints this (this model was trained on the classification corpus, so what it
writes out is classification text too):

```text
    生成结果：床前明月光苹果很甜分类：食品。华为发布新手机分类：科技。小
```

*(The prompt is `床前明月光` — "moonlight in front of my bed", the opening of the poem. The
model continues with `苹果很甜分类：食品。华为发布新手机分类：科技。小` — that is, "apples
are very sweet; category: food. Huawei shipped a new phone; category: tech. Xi…" — the
classification-style text it was actually trained on.)*

Between one line of `if` and these few lines lie 30 chapters.

## Python implementation

The last chapter's code has only two new things, and both are small.

**One: how to do classification with a language model.** Treat the category as "the next
thing to predict":

```python
def label_probabilities(model, sentence):
    prompt = sentence + "分类："
    ids = encode(prompt)
    scores = {}
    for label in LABELS:
        label_ids = encode(label)
        # feed in the whole of "sentence; category: tech", and only add up the log
        # probabilities of the last two characters (the label)
        logprobs = model(torch.tensor([ids + label_ids]))[0].log_softmax(dim=-1)
        total = sum(logprobs[len(ids) - 1 + offset, token].item()
                    for offset, token in enumerate(label_ids))
        scores[label] = total
    # normalise into probabilities
    values = torch.tensor([scores[label] for label in LABELS])
    probabilities = values.softmax(dim=-1)
    return {label: probabilities[i].item() for i, label in enumerate(LABELS)}
```

Notice that every line here can be traced back to an earlier chapter:
`encode` is Chapter 1's splitting, `log_softmax` is Chapter 5's probability,
adding log probabilities is Chapter 20's "probability of a whole sequence",
and the `softmax` normalisation is the same function from Chapter 5.

**Two: generation.** It is the Chapter 23 thing, not one line more:

```python
@torch.no_grad()
def continue_text(model, prompt, n_new=24, seed=0):
    generator = torch.Generator().manual_seed(seed)
    tokens = encode(prompt)
    for _ in range(n_new):
        window = tokens[-model.block_size:]
        logits = model(torch.tensor([window]))
        next_token = sample(logits[0, -1], generator=generator)
        tokens.append(next_token)
    return decode(tokens)
```

The code on `before.py`'s side is simpler still — it is just those few lines from Chapter
2. The amount of code differs by three orders of magnitude, and the score on this test is
the same (14/14 against 14/14).

## What it solves

This chapter solves no new problem. What it does is **settle the accounts**.

Thirty chapters ago, what we held was 4 lines of `if`, whose rules were written by people,
whose knowledge sat in a word list, which had to be changed by hand every time, and which
was helpless the moment it met a phrasing nobody had thought of.

Thirty chapters later, what we hold is 109,312 parameters (at GPT-3's real scale, 175
billion), whose rules were learned, whose knowledge sits in the parameters, which is
changed by retraining, and which can still guess from context when it meets a phrasing
nobody thought of.

And across those 30 chapters, **we never copied a module from anywhere**. Every part was
forced into existence in some earlier chapter because "what we had last chapter wasn't
enough". That is what this book wants you to see:

> **An LLM is not a complicated machine that fell out of the sky.**
> It can be seen as an enormously large, **continuous conditional classification system**
> that learns its own internal representations and context relationships — continuous,
> because what it gives is a probability rather than a yes or no; conditional, because
> every token it outputs is conditioned on all the tokens before it.

In one sentence: **it is doing the same thing we did in Chapter 2 — looking at a stretch of
text and producing a judgement. It has only split "judgement" into several billion
probability outputs, swapped "rules" for parameters, and swapped "written by a person" for
"learned".**

## What it still can't solve

The book ends here. So this section doesn't lead into a next chapter; instead it lists **the
things it didn't cover** — all of them real, all of them important, and all of them things
we deliberately went around.

**One: real pretraining data.**

Our corpus is 160,000 characters at most, and we made it ourselves. Real pretraining data
is trillions of tokens, crawled from web pages, books, code, and papers, and then put
through a whole cleaning pipeline: deduplication, removing ads, removing harmful content,
stratifying by quality, and balancing by domain. **The effect of this step on the final
model is larger than many architectural choices**, and this book did not write one line
about it.

**Two: distributed training.**

We ran everything on one CPU in one laptop. GPT-3-scale training needs thousands of GPUs,
stitched together with data parallelism, tensor parallelism, and pipeline parallelism; it
has to handle communication, fault tolerance, resuming from checkpoints, mixed precision,
gradient accumulation, activation recomputation… any one of these engineering problems can
make the training fail. The "one line of `.to("cuda")` and you're done" mentioned in this
book is only the first step of a very long march at real scale.

**Three: the details of alignment.**

In Chapter 29 we covered preference alignment with a 20-line DPO. In the real world it is:
tens of thousands to hundreds of thousands of human-annotated preference examples, a
trained reward model, reinforcement learning with a KL constraint (PPO), a pile of tricks
to stop the model "gaming the system" (reward hacking), and the entire direction of "safety
alignment". This book only covered its skeleton.

**Four: multimodality.**

We only ever handled text, from beginning to end. Today's models look at images, listen to
audio, and run code. Once the input form changes, the model's structure, its training
objective, and the way its data is constructed all change — every part in this book would
have to be thought through again to see how it connects.

**Five: models that "think".**

In Chapters 23 and 24 we looked at "why does it seem to be thinking", and concluded that it
is only predicting the next token. But in recent years a class of models has appeared: they
write out a line of reasoning first and then give the answer; they are rewarded by
reinforcement learning for "reasoning correctly"; they spend more compute at inference time
to buy better answers. In both training objective and inference method they differ from
what we did in this book.

**Six, and the most important one, is the problem we never managed to solve:**

> What does it actually "understand"?

This book has gone around that question the whole way. We looked only at inputs and
outputs, and only at loss numbers, because those are the only things we can measure. A
model with a loss of 0.46 can memorize a whole poem and can answer 14 classification
questions correctly — but does it "know" what an apple is? Does it know that apples can be
eaten, that they grow on trees, and that they rot? Or does it only know that `很甜`
("very sweet") often follows `苹果`?

That question has not been answered in 30 chapters, and it is not within this book's power
to answer.

But at least now you know what it is made of: not magic — a pile of things you wrote with
your own hands, and a container big enough to hold that much regularity.

## Exercises

See `exercises.en.md`; here are the 3 most important ones:

1. Expand `TEST_SET` to 30 sentences and **deliberately write a few that both the rules and
   the model will get wrong.** Watch how the two report cards change. Why is it easier to
   "set a hard question" for the model than for the rules?
2. Change `after.py`'s corpus to three categories (tech / food / weather). How many lines
   does the model side need? (Hint: one more entry in `LABELS`, a few more sentences in the
   corpus.) How many lines does the rules side need?
3. Go back to the sentence from Chapter 1: "tens of thousands of words — are you going to
   write tens of thousands of ifs?" How many different levels of answer can you give now?
   (Hint: rules → weights → representations → context → scale.)
